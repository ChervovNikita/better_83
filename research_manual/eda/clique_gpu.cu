// GPU clique harvester: many DISTINCT maximal cliques of size omega from one
// graph, not one clique faster.  fleet_solver.py's delete-and-resolve stage run
// wide -- one walker per warp, all sharing a read-only graph, differing in
// (ban set, seed).
//
// Design, measurements, and where this deviates from the design:
// research_manual/eda/gpu_clique_design.md and README.md.
//
// Two compile-time arms:
//   SN83_LANES         32 (default) / 64 / 128 lanes per walker
//   SN83_CANDS_PREFIX  0 = saturating carry (default), 1 = prefix/suffix ANDs
//
// Contains a host mirror (namespace ref) of the same search, used by the stage-1
// and stage-2 gates in gates.py.
//
// Build: gpu_lib.py.  nvcc -O3 -arch=sm_86 -Xptxas -v.

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

// --------------------------- compile-time shape ----------------------------

// n <= 1024 covers the live round set (measured max n = 900 => W = 15).
#define SN83_MAXN 1024
#define SN83_MAXW ((SN83_MAXN + 63) / 64)

// Measured omega tops out at 110; a walker that would exceed this stops adding
// and increments C_KMAXHIT rather than truncating silently.
#define SN83_KMAX 160

#define SN83_MAXBANS 8
#define SN83_NLEV 5   // level == |ban set|, 0..4; pop scans ascending (§5)

// 32 => one warp per walker, 4 walkers per 128-thread block.  64/128 => the
// block IS the walker, so __syncthreads() is exactly the walker barrier and no
// walker can deadlock on a sibling that branched differently.
#ifndef SN83_LANES
#define SN83_LANES 32
#endif

#if SN83_LANES == 32
#define SN83_WPB 4
#define SN83_BLOCK 128
#else
#define SN83_WPB 1
#define SN83_BLOCK SN83_LANES
#endif
#define SN83_NWARP (SN83_LANES / 32)

#ifndef SN83_CANDS_PREFIX
#define SN83_CANDS_PREFIX 0
#endif

#if SN83_CANDS_PREFIX
#define SN83_PS_WORDS ((SN83_KMAX + 1) * SN83_MAXW)
#endif

typedef unsigned long long u64;
typedef unsigned int u32;

enum {
    C_JOBS = 0,   // jobs run to completion (synthesized ones included)
    C_NEW,        // results new to the dedup table
    C_DUP,        // results already held      -> §6 stall detector
    C_SHORT,      // results below target      -> requeued at 2x max_steps
    C_DROP,       // enqueues that hit a full ring
    C_BANBACK,    // full-G extension re-added a banned vertex (§5/§7)
    C_STEPS,      // total search steps over all walkers
    C_SYNTH,      // jobs a starved warp made for itself
    C_ENQ,        // jobs ever made poppable   } quiescence: read DONE first,
    C_DONE,       // jobs ever fully finished  } then ENQ
    C_OVERFLOW,   // result slots exhausted
    C_KMAXHIT,    // a walker hit SN83_KMAX
    // C_DUP/C_NEW cover the whole recorded band (omega and the omega-1 spares),
    // and a round with many distinct spares therefore reads a LOWER stall than
    // the omega pool alone justifies.  §6's ceiling is about omega-cliques, so
    // the pair below counts only results at the current target.
    C_DUPMAX,
    C_NEWMAX,
    C_NCTR = 16
};

// splitmix64 with an explicit state, so the host mirror and the device consume
// randomness in the same order.

__host__ __device__ __forceinline__ u64 mix64(u64 z) {
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

__host__ __device__ __forceinline__ u64 sm64(u64 &x) {
    x += 0x9E3779B97F4A7C15ull;
    return mix64(x);
}

// BMS argmax key: score desc, age asc, vertex asc.  clique.cpp breaks the last
// tie by IndexSet scan order, which has no counterpart here; equivalence is
// asserted over the candidate SETS, not over which vertex wins.
__host__ __device__ __forceinline__ u64 bms_key(int score, int age, int v) {
    return ((u64)(unsigned)score << 44) |
           ((u64)(0xFFFFFFFFu - (unsigned)age) << 12) |
           (u64)(unsigned)(4095 - v);
}
__host__ __device__ __forceinline__ int bms_vertex(u64 key) {
    return 4095 - (int)(key & 0xFFFull);
}

// --------------------------- shared PODs -----------------------------------

struct DGraph {
    int n, W;
    u64 tail;                        // valid-bit mask of the last word
    const u64 *__restrict__ bits;    // n rows of W words, self bit clear
    const int *__restrict__ deg;
};

struct Cfg {
    int bms_k;          // 64,   clique.cpp Params::bms_k
    int restart;        // 4000, clique.cpp Params::restart_steps
    int keep;           // 4,    clique.cpp Params::perturb_keep
    int spare_margin;   // 1,    fleet_solver.SPARE_MARGIN
    int max_steps_cap;  // ceiling for the short-return doubling (§5)
    int allow_synth;    // 1 in harvest, 0 in batch
    int gen_children;   // 1 in harvest, 0 in batch
};

// 32 B, so a warp loads one in a single transaction.
struct Job {
    u64 seed;
    u32 max_steps;
    u32 epoch;
    unsigned short bans[SN83_MAXBANS];
    unsigned char n_bans;
    unsigned char level;
    unsigned char dead;
    unsigned char pad;
};

struct Queues {
    Job *ring;   // SN83_NLEV * cap
    u32 *head;   // [NLEV] popped
    u32 *ready;  // [NLEV] published
    u32 *tail;   // [NLEV] reserved
    u32 cap;
};

struct Pool {
    u64 *fp;            // open-addressed dedup table, 0 == empty
    u32 fpcap;          // power of two
    int *res_size;
    short *res_vert;    // res_cap rows of SN83_KMAX
    u32 *n_res;
    u32 res_cap;
    int *target;        // largest clique seen so far
    u32 *epoch;
    int *done_flag;
    u64 *ctr;           // [C_NCTR]
    volatile int *stop;   // mapped pinned, host writes 1 at the deadline
};

// per-walker scratch in shared memory
struct WS {
    u64 Cb[SN83_MAXW];       // members of C
    u64 conf[SN83_MAXW];     // SCC: bit set == allowed to (re-)enter C
    u64 allow[SN83_MAXW];    // ALL & ~bans; P_0 in the design's terms
    u64 freeb[SN83_MAXW];    // allow & ~Cb
    u64 ge1[SN83_MAXW];      // misses >= 1 member of C
    u64 ge2[SN83_MAXW];      // misses >= 2 members of C
    u64 c0[SN83_MAXW];       // slack == 0
    u64 c1[SN83_MAXW];       // slack == 1
    u64 pm[SN83_MAXW];       // the mask currently being picked from
    u64 pa[SN83_LANES];      // (i,w)-tiling partials for the carry recurrence
    u64 pb[SN83_LANES];
    u64 red[SN83_NWARP];     // cross-warp reduction scratch (LANES > 32)
    u64 bcast_u;
    u64 rng;
    u64 traj;                // rolling move hash, the stage-2 diff signal
    int pref[SN83_MAXW + 1]; // exclusive popcount prefix of the pick mask
    short C[SN83_KMAX];
    int k;
    int best_size;
    int kmax_hit;
    int q_flags;
    int res_slot;
    int keep_flag;
    long long step;
};

struct Grp {
    int lane;
};

// ------------------------- group primitives --------------------------------

__device__ __forceinline__ void gsync(WS &s) {
    (void)s;
#if SN83_LANES == 32
    __syncwarp();
#else
    __syncthreads();
#endif
}

// after a lane-0 write to GLOBAL memory that other lanes will read
__device__ __forceinline__ void gsync_mem(WS &s) {
    __threadfence_block();
    gsync(s);
}

__device__ __forceinline__ u64 gmax_u64(WS &s, const Grp &gr, u64 v) {
    for (int o = 16; o; o >>= 1) {
        u64 t = __shfl_xor_sync(0xFFFFFFFFu, v, o);
        v = t > v ? t : v;
    }
#if SN83_NWARP > 1
    if ((gr.lane & 31) == 0) s.red[gr.lane >> 5] = v;
    __syncthreads();
    v = 0;
    for (int i = 0; i < SN83_NWARP; ++i) v = s.red[i] > v ? s.red[i] : v;
    __syncthreads();
#else
    (void)s;
    (void)gr;
#endif
    return v;
}

__device__ __forceinline__ int gsum_int(WS &s, const Grp &gr, int v) {
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, o);
#if SN83_NWARP > 1
    if ((gr.lane & 31) == 0) s.red[gr.lane >> 5] = (u64)(unsigned)v;
    __syncthreads();
    v = 0;
    for (int i = 0; i < SN83_NWARP; ++i) v += (int)(unsigned)s.red[i];
    __syncthreads();
#else
    (void)s;
    (void)gr;
#endif
    return v;
}

__device__ __forceinline__ u64 gxor_u64(WS &s, const Grp &gr, u64 v) {
    for (int o = 16; o; o >>= 1) v ^= __shfl_xor_sync(0xFFFFFFFFu, v, o);
#if SN83_NWARP > 1
    if ((gr.lane & 31) == 0) s.red[gr.lane >> 5] = v;
    __syncthreads();
    v = 0;
    for (int i = 0; i < SN83_NWARP; ++i) v ^= s.red[i];
    __syncthreads();
#else
    (void)s;
    (void)gr;
#endif
    return v;
}

// one draw from the walker RNG, the same value on every lane
__device__ __forceinline__ u64 g_next(WS &s, const Grp &gr) {
    if (gr.lane == 0) s.bcast_u = sm64(s.rng);
    gsync(s);
    u64 r = s.bcast_u;
    gsync(s);
    return r;
}

// Bitsets are W <= 16 words in shared memory, so any lane can read any word;
// lanes stride over words for the bulk ops.

__device__ __forceinline__ u64 allmask(const DGraph &g, int w) {
    return (w == g.W - 1) ? g.tail : ~0ull;
}

__device__ __forceinline__ int bs_count(const DGraph &g, WS &s, const Grp &gr,
                                        const u64 *a) {
    int c = 0;
    for (int w = gr.lane; w < g.W; w += SN83_LANES) c += __popcll(a[w]);
    return gsum_int(s, gr, c);
}

// exclusive popcount prefix over `a`, written by lane 0 (W <= 16 words)
__device__ __forceinline__ void bs_pref(const DGraph &g, WS &s, const Grp &gr,
                                        const u64 *a) {
    if (gr.lane == 0) {
        int acc = 0;
        for (int w = 0; w < g.W; ++w) {
            s.pref[w] = acc;
            acc += __popcll(a[w]);
        }
        s.pref[g.W] = acc;
    }
    gsync(s);
}

// the rank-th set bit of `a`, given a prefix built by bs_pref
__device__ __forceinline__ int bs_select(const DGraph &g, const WS &s,
                                         const u64 *a, int rank) {
    int w = 0;
    while (w + 1 < g.W && s.pref[w + 1] <= rank) ++w;
    u64 x = a[w];
    int r = rank - s.pref[w];
    for (int i = 0; i < r; ++i) x &= x - 1;
    return (w << 6) + (__ffsll((long long)x) - 1);
}

__device__ __forceinline__ bool bs_test(const u64 *a, int v) {
    return (a[v >> 6] >> (v & 63)) & 1ull;
}

// Candidate sets: two arms, both producing slack==0 in c0, slack==1 in c1.

struct Ctx;

#if SN83_CANDS_PREFIX

// P_i = N(c_0)&..&N(c_{i-1}), S_i = N(c_i)&..&N(c_{k-1}), rebuilt each step.
// The incremental update is deliberately not taken: this arm exists to measure
// the structure as specified.
__device__ void compute_cands(const DGraph &g, WS &s, const Grp &gr, u64 *prefix,
                              u64 *suffix) {
    const int W = g.W, k = s.k;
    for (int w = gr.lane; w < W; w += SN83_LANES) {
        s.freeb[w] = s.allow[w] & ~s.Cb[w];
        prefix[w] = s.allow[w];   // P_0 = allowed; bans propagate everywhere
        suffix[(size_t)k * W + w] = allmask(g, w);
    }
    gsync(s);
    for (int i = 0; i < k; ++i) {
        const u64 *row = g.bits + (size_t)s.C[i] * W;
        for (int w = gr.lane; w < W; w += SN83_LANES)
            prefix[(size_t)(i + 1) * W + w] = prefix[(size_t)i * W + w] & row[w];
    }
    for (int i = k - 1; i >= 0; --i) {
        const u64 *row = g.bits + (size_t)s.C[i] * W;
        for (int w = gr.lane; w < W; w += SN83_LANES)
            suffix[(size_t)i * W + w] = suffix[(size_t)(i + 1) * W + w] & row[w];
    }
    gsync(s);
    for (int w = gr.lane; w < W; w += SN83_LANES) {
        s.c0[w] = prefix[(size_t)k * W + w] & ~s.Cb[w];
        s.c1[w] = 0ull;
    }
    gsync(s);
    // cand1 = union_j (P_j & S_{j+1}) & ~N(c_j) & ~Cbits; the j-sets are
    // disjoint, so OR-ing them loses nothing.
    for (int j = 0; j < k; ++j) {
        const u64 *row = g.bits + (size_t)s.C[j] * W;
        for (int w = gr.lane; w < W; w += SN83_LANES)
            s.c1[w] |= prefix[(size_t)j * W + w] &
                       suffix[(size_t)(j + 1) * W + w] & ~row[w] & ~s.Cb[w];
    }
    gsync(s);
}

#else

// ge1/ge2 are "misses >= 1" and "misses >= 2" over the members of C.  slack(v)
// IS the miss count, so cand0 = free & ~ge1 and cand1 = free & ge1 & ~ge2.
//
// Tiled as lane -> (group gi = lane/W, word w = lane%W): ngrp groups of W lanes
// each walk a stride-ngrp slice of C.  The combine
//     (a1,b1) o (a2,b2) = (a1|a2, b1|b2|(a1&a2))
// is associative and commutative, so the per-group partials merge correctly.
__device__ void compute_cands(const DGraph &g, WS &s, const Grp &gr, u64 *,
                              u64 *) {
    const int W = g.W, lane = gr.lane, k = s.k;
    for (int w = lane; w < W; w += SN83_LANES) s.freeb[w] = s.allow[w] & ~s.Cb[w];
    gsync(s);

    if (W > SN83_LANES) {
        // n > 2048; unreachable for this round set, kept so the code is total
        for (int w = lane; w < W; w += SN83_LANES) {
            u64 fr = s.freeb[w], a = 0ull, b = 0ull;
            for (int i = 0; i < k; ++i) {
                u64 m = ~g.bits[(size_t)s.C[i] * W + w] & fr;
                b |= a & m;
                a |= m;
            }
            s.ge1[w] = a;
            s.ge2[w] = b;
        }
        gsync(s);
    } else {
        const int ngrp = SN83_LANES / W;
        const int w = lane % W, gi = lane / W;
        u64 a = 0ull, b = 0ull;
        if (gi < ngrp) {
            u64 fr = s.freeb[w];
            for (int i = gi; i < k; i += ngrp) {
                u64 m = ~g.bits[(size_t)s.C[i] * W + w] & fr;
                b |= a & m;
                a |= m;
            }
        }
        s.pa[lane] = a;
        s.pb[lane] = b;
        gsync(s);
        if (lane < W) {
            u64 aa = s.pa[lane], bb = s.pb[lane];
            for (int t = 1; t < ngrp; ++t) {
                u64 a2 = s.pa[t * W + lane], b2 = s.pb[t * W + lane];
                bb = bb | b2 | (aa & a2);
                aa = aa | a2;
            }
            s.ge1[lane] = aa;
            s.ge2[lane] = bb;
        }
        gsync(s);
    }

    for (int w = lane; w < W; w += SN83_LANES) {
        u64 fr = s.freeb[w], a1 = s.ge1[w], a2 = s.ge2[w];
        s.c0[w] = fr & ~a1;
        s.c1[w] = fr & a1 & ~a2;
    }
    gsync(s);
}

#endif

// ------------------------------ the search ---------------------------------

struct Ctx {
    DGraph g;
    Cfg cfg;
    WS *s;
    Grp gr;
    int *age;      // [n_pad] for this walker, global
    short *bestv;  // [SN83_KMAX] for this walker, global
#if SN83_CANDS_PREFIX
    u64 *prefix;
    u64 *suffix;
#endif
    volatile int *stop;
};

#if SN83_CANDS_PREFIX
#define CANDS(c) compute_cands((c).g, *(c).s, (c).gr, (c).prefix, (c).suffix)
#else
#define CANDS(c) compute_cands((c).g, *(c).s, (c).gr, (u64 *)0, (u64 *)0)
#endif

// clique.cpp::pick over a bitset: sample at most bms_k of `mask`, keep the
// largest overlap with the mask itself, oldest breaks ties.
__device__ int bms_pick(Ctx &c, const u64 *mask, int m) {
    WS &s = *c.s;
    const Grp &gr = c.gr;
    if (m <= 0) return -1;
    bs_pref(c.g, s, gr, mask);
    if (m == 1) return bs_select(c.g, s, mask, 0);

    // Drawn even when unused, so host and device stay in RNG lockstep.
    const u64 base = g_next(s, gr);
    const bool enumerate = (m <= c.cfg.bms_k);
    const int tries = enumerate ? m : c.cfg.bms_k;

    u64 best = 0ull;
    for (int t = gr.lane; t < tries; t += SN83_LANES) {
        int r = enumerate ? t
                          : (int)(mix64(base + (u64)t * 0x9E3779B97F4A7C15ull) %
                                  (unsigned)m);
        int v = bs_select(c.g, s, mask, r);
        const u64 *row = c.g.bits + (size_t)v * c.g.W;
        int sc = 0;
        for (int w = 0; w < c.g.W; ++w) sc += __popcll(row[w] & mask[w]);
        u64 key = bms_key(sc, c.age[v], v);
        if (key > best) best = key;
    }
    best = gmax_u64(s, gr, best);
    return bms_vertex(best);
}

__device__ __forceinline__ void trace(WS &s, int tag, int v) {
    s.traj = (s.traj ^ (u64)(unsigned)(v + 1) ^ ((u64)tag << 32)) * 0x100000001B3ull;
}

// `free_neighbours` is what separates an add from a swap under strong CC.
__device__ void w_add(Ctx &c, int v, bool free_neighbours) {
    WS &s = *c.s;
    if (c.gr.lane == 0) {
        if (s.k < SN83_KMAX) {
            s.C[s.k] = (short)v;
            s.k++;
        } else {
            s.kmax_hit = 1;
        }
        s.Cb[v >> 6] |= 1ull << (v & 63);
        c.age[v] = (int)s.step;
        trace(s, free_neighbours ? 1 : 2, v);
    }
    gsync_mem(s);
    if (free_neighbours) {
        const u64 *row = c.g.bits + (size_t)v * c.g.W;
        for (int w = c.gr.lane; w < c.g.W; w += SN83_LANES) s.conf[w] |= row[w];
        gsync(s);
    }
}

__device__ void w_drop(Ctx &c, int v) {
    WS &s = *c.s;
    if (c.gr.lane == 0) {
        for (int i = 0; i < s.k; ++i)
            if (s.C[i] == (short)v) {
                s.C[i] = s.C[s.k - 1];
                break;
            }
        s.k--;
        s.Cb[v >> 6] &= ~(1ull << (v & 63));
        s.conf[v >> 6] &= ~(1ull << (v & 63));  // SCC: a drop forbids re-entry
        c.age[v] = (int)s.step;
        trace(s, 3, v);
    }
    gsync_mem(s);
}

__device__ void w_clear(Ctx &c) {
    WS &s = *c.s;
    if (c.gr.lane == 0) s.k = 0;
    for (int w = c.gr.lane; w < c.g.W; w += SN83_LANES) s.Cb[w] = 0ull;
    gsync(s);
}

// clique.cpp draws rnd(n); a banned vertex must never enter C, so this draws
// uniformly over the allowed vertices.  With no bans the two coincide.
__device__ int rand_allowed(Ctx &c) {
    WS &s = *c.s;
    int m = bs_count(c.g, s, c.gr, s.allow);
    if (m <= 0) return -1;
    u64 r = g_next(s, c.gr);
    bs_pref(c.g, s, c.gr, s.allow);
    return bs_select(c.g, s, s.allow, (int)(r % (unsigned)m));
}

__device__ void w_seed_from(Ctx &c, int v) {
    WS &s = *c.s;
    w_clear(c);
    for (int w = c.gr.lane; w < c.g.W; w += SN83_LANES) s.conf[w] = allmask(c.g, w);
    gsync(s);
    if (v >= 0) w_add(c, v, true);
}

// Greedy construction.  clique.cpp's construct() does NOT apply the conf filter
// -- it scans cand0 raw -- and neither does this.
__device__ void w_construct(Ctx &c) {
    WS &s = *c.s;
    for (;;) {
        CANDS(c);
        int m = bs_count(c.g, s, c.gr, s.c0);
        if (m == 0) return;
        if (s.k >= SN83_KMAX) return;
        int v = bms_pick(c, s.c0, m);
        if (v < 0) return;
        w_add(c, v, true);
    }
}

// The one clique member v is not adjacent to.  clique.cpp returns the first in
// C order; the warp scan matches it by taking the minimum index.
__device__ int w_blocker(Ctx &c, int v) {
    WS &s = *c.s;
    const u64 *row = c.g.bits + (size_t)v * c.g.W;
    u64 best = 0ull;
    for (int i = c.gr.lane; i < s.k; i += SN83_LANES) {
        int u = s.C[i];
        if (!((row[u >> 6] >> (u & 63)) & 1ull)) {
            u64 key = ((u64)(unsigned)(0x3FFFFFFF - i) << 16) | (u64)(unsigned)u;
            if (key > best) best = key;   // max over (big - i) == min over i
        }
    }
    best = gmax_u64(s, c.gr, best);
    if (best == 0ull) return -1;
    return (int)(best & 0xFFFFull);
}

__device__ void w_save_best(Ctx &c) {
    WS &s = *c.s;
    for (int i = c.gr.lane; i < s.k; i += SN83_LANES) c.bestv[i] = s.C[i];
    if (c.gr.lane == 0) s.best_size = s.k;
    gsync_mem(s);
}

// Keep a few vertices of the incumbent and rebuild around them; occasionally
// restart cold so the portfolio does not collapse onto one basin.
//
// clique.cpp shuffles a COPY of best; here the shuffle is in place on bestv.
// Same set and same uniform choice of survivors, only the stored order differs.
// The host mirror does the same, so the stage-2 diff still binds.
__device__ void w_perturb(Ctx &c) {
    WS &s = *c.s;
    // Short circuit: clique.cpp does not draw rnd(10) when best is empty.
    bool cold = (s.best_size == 0);
    if (!cold) cold = (g_next(s, c.gr) % 10ull) == 0ull;
    if (cold) {
        w_seed_from(c, rand_allowed(c));
        w_construct(c);
        return;
    }
    const int bs = s.best_size;
    const int m = c.cfg.keep < bs ? c.cfg.keep : bs;
    if (c.gr.lane == 0) {
        for (int i = 0; i < m; ++i) {
            int j = i + (int)(sm64(s.rng) % (unsigned)(bs - i));
            short t = c.bestv[i];
            c.bestv[i] = c.bestv[j];
            c.bestv[j] = t;
        }
    }
    gsync_mem(s);
    w_clear(c);
    for (int w = c.gr.lane; w < c.g.W; w += SN83_LANES) s.conf[w] = allmask(c.g, w);
    gsync(s);
    for (int i = 0; i < m; ++i) {
        int v = c.bestv[i];
        CANDS(c);
        if (bs_test(s.c0, v)) w_add(c, v, true);  // clique.cpp: if slack[v]==0
    }
    if (s.k == 0) w_seed_from(c, rand_allowed(c));
    w_construct(c);
}

// clique.cpp::Search::run with max_steps in place of the deadline, so a
// measurement run is reproducible.  `stop` still lets a live deadline cut in.
__device__ void w_run(Ctx &c, long long max_steps) {
    WS &s = *c.s;
    if (c.gr.lane == 0) {
        s.step = 0;
        s.best_size = 0;
    }
    gsync(s);

    w_seed_from(c, rand_allowed(c));
    w_construct(c);
    w_save_best(c);
    long long last_gain = 0;

    for (;;) {
        if ((s.step & 63) == 0 && *c.stop) break;
        if (max_steps > 0 && s.step >= max_steps) break;
        if (c.gr.lane == 0) s.step++;
        gsync(s);

        // Every step, not only when the search runs out of moves: the gated
        // variant measured 3.1M swaps against 2 adds and never restarted.
        if (s.step - last_gain > c.cfg.restart) {
            w_perturb(c);
            last_gain = s.step;
            continue;
        }

        CANDS(c);

        for (int w = c.gr.lane; w < c.g.W; w += SN83_LANES)
            s.pm[w] = s.c0[w] & s.conf[w];
        gsync(s);
        int ma = bs_count(c.g, s, c.gr, s.pm);
        if (ma > 0) {
            int v = bms_pick(c, s.pm, ma);
            if (s.k >= SN83_KMAX) break;
            w_add(c, v, true);
            if (s.k > s.best_size) {
                w_save_best(c);
                last_gain = s.step;
            }
            continue;
        }

        for (int w = c.gr.lane; w < c.g.W; w += SN83_LANES)
            s.pm[w] = s.c1[w] & s.conf[w];
        gsync(s);
        int ms = bs_count(c.g, s, c.gr, s.pm);
        if (ms > 0) {
            int v = bms_pick(c, s.pm, ms);
            int u = w_blocker(c, v);
            if (u >= 0) {
                w_drop(c, u);
                w_add(c, v, false);   // SCC: a swap does not free neighbours
            }
            continue;
        }

        if (s.k > 0) {
            u64 r = g_next(s, c.gr);
            int v = s.C[(int)(r % (unsigned)s.k)];
            w_drop(c, v);
        } else {
            w_seed_from(c, rand_allowed(c));
        }
    }

    w_construct(c);   // never return an extendable clique
    if (s.k > s.best_size) w_save_best(c);
}

// A clique maximal in G-bans can be extendable in G, and the validator rejects
// anything extendable, zeroing both reward terms.  So the answer is extended in
// the FULL graph with bans ignored, by fleet_solver._extend's rule: highest
// degree wins, lowest index breaks ties.  Returns how many bans came back.
__device__ int w_extend_full(Ctx &c, const u64 *banbits) {
    WS &s = *c.s;
    int banback = 0;
    for (;;) {
        // each lane owns its words, so the k-loop needs no inner barrier
        for (int w = c.gr.lane; w < c.g.W; w += SN83_LANES) {
            u64 x = allmask(c.g, w) & ~s.Cb[w];
            for (int i = 0; i < s.k; ++i) x &= c.g.bits[(size_t)s.C[i] * c.g.W + w];
            s.pm[w] = x;
        }
        gsync(s);
        u64 best = 0ull;
        for (int v = c.gr.lane; v < c.g.n; v += SN83_LANES)
            if (bs_test(s.pm, v)) {
                u64 key = ((u64)(unsigned)c.g.deg[v] << 16) | (u64)(unsigned)(4095 - v);
                if (key > best) best = key;
            }
        best = gmax_u64(s, c.gr, best);
        if (best == 0ull) break;
        int v = 4095 - (int)(best & 0xFFFFull);
        if (s.k >= SN83_KMAX) break;
        if (banbits && bs_test(banbits, v)) ++banback;
        if (c.gr.lane == 0) {
            s.C[s.k] = (short)v;
            s.k++;
            s.Cb[v >> 6] |= 1ull << (v & 63);
        }
        gsync(s);
    }
    return banback;
}

// XOR of per-vertex hashes, order-independent and warp-parallel, mixed with the
// size.  Collides at ~1e-7 over ~1e6 cliques; the host re-verifies exactly, so a
// collision costs a lost clique and never a bad answer.
__device__ u64 w_fingerprint(Ctx &c) {
    WS &s = *c.s;
    u64 h = 0ull;
    for (int i = c.gr.lane; i < s.k; i += SN83_LANES) h ^= mix64((u64)(s.C[i]) + 1ull);
    h = gxor_u64(s, c.gr, h);
    h = mix64(h ^ ((u64)(unsigned)s.k * 0x9E3779B97F4A7C15ull));
    return h ? h : 1ull;   // 0 is the empty-slot marker
}

// -------------------------------- queues -----------------------------------

__device__ bool q_pop(Queues &q, Job *out) {
    for (int L = 0; L < SN83_NLEV; ++L) {
        for (;;) {
            u32 h = atomicAdd(&q.head[L], 0u);
            u32 r = atomicAdd(&q.ready[L], 0u);
            if (h >= r) break;
            if (atomicCAS(&q.head[L], h, h + 1u) == h) {
                *out = q.ring[(size_t)L * q.cap + (h % q.cap)];
                return true;
            }
        }
    }
    return false;
}

// C_ENQ is bumped BEFORE the job becomes poppable, so the quiescence test (read
// C_DONE, then C_ENQ) can never see a job in flight.
__device__ void q_push(Queues &q, Pool &p, int L, Job j) {
    if (L < 0) L = 0;
    if (L >= SN83_NLEV) L = SN83_NLEV - 1;
    atomicAdd(&p.ctr[C_ENQ], 1ull);
    u32 t = atomicAdd(&q.tail[L], 1u);
    u32 h = atomicAdd(&q.head[L], 0u);
    if (t - h >= q.cap) {
        j.dead = 1;   // ring full: publish a tombstone so slots stay contiguous
        atomicAdd(&p.ctr[C_DROP], 1ull);
    }
    q.ring[(size_t)L * q.cap + (t % q.cap)] = j;
    __threadfence();
    while (atomicCAS(&q.ready[L], t, t + 1u) != t) {
#if __CUDA_ARCH__ >= 700
        __nanosleep(32);
#endif
    }
}

// Returns 1 new, 0 duplicate, -1 table saturated.

__device__ int pool_insert(Pool &p, u64 fp) {
    const u32 mask = p.fpcap - 1u;
    u32 slot = (u32)(fp >> 32) & mask;
    for (u32 probe = 0; probe < 128u; ++probe) {
        u64 cur = p.fp[slot];
        if (cur == fp) return 0;
        if (cur == 0ull) {
            u64 old = atomicCAS(&p.fp[slot], 0ull, fp);
            if (old == 0ull) return 1;
            if (old == fp) return 0;
        }
        slot = (slot + 1u) & mask;
    }
    return -1;
}

// -------------------------- one job, start to finish -----------------------

struct JobOut {
    int size;
    u64 fp;
    int banback;
};

__device__ JobOut run_job(Ctx &c, const Job &job, u64 *banbits) {
    WS &s = *c.s;
    // allow = ALL & ~bans, i.e. P_0, so every candidate set inherits the ban
    // mask without a separate AND anywhere else.
    for (int w = c.gr.lane; w < c.g.W; w += SN83_LANES) {
        banbits[w] = 0ull;
        s.allow[w] = allmask(c.g, w);
    }
    gsync(s);
    if (c.gr.lane == 0)
        for (int i = 0; i < (int)job.n_bans && i < SN83_MAXBANS; ++i) {
            int v = job.bans[i];
            if (v >= 0 && v < c.g.n) banbits[v >> 6] |= 1ull << (v & 63);
        }
    gsync(s);
    for (int w = c.gr.lane; w < c.g.W; w += SN83_LANES) s.allow[w] &= ~banbits[w];
    for (int v = c.gr.lane; v < c.g.n; v += SN83_LANES) c.age[v] = 0;
    if (c.gr.lane == 0) {
        s.rng = mix64(job.seed ^ 0xA5A5A5A5DEADBEEFull);
        s.traj = 0xCBF29CE484222325ull;
        s.kmax_hit = 0;
    }
    gsync_mem(s);

    w_run(c, (long long)job.max_steps);

    // clique.cpp returns `best`, not wherever the loop stopped.
    w_clear(c);
    for (int w = c.gr.lane; w < c.g.W; w += SN83_LANES) s.conf[w] = allmask(c.g, w);
    gsync(s);
    if (c.gr.lane == 0) {
        for (int i = 0; i < s.best_size; ++i) {
            int v = c.bestv[i];
            s.C[i] = (short)v;
            s.Cb[v >> 6] |= 1ull << (v & 63);
        }
        s.k = s.best_size;
    }
    gsync_mem(s);

    JobOut o;
    o.banback = w_extend_full(c, job.n_bans ? banbits : (const u64 *)0);
    o.size = s.k;
    o.fp = w_fingerprint(c);
    return o;
}

// ------------------------------- kernels -----------------------------------

struct WalkerMem {
    int *age;
    short *bestv;
    int n_pad;
#if SN83_CANDS_PREFIX
    u64 *ps;
#endif
};

__device__ __forceinline__ void ctx_init(Ctx &c, const DGraph &g, const Cfg &cfg,
                                         WS *s, int lane, const WalkerMem &wm,
                                         int wid, volatile int *stop) {
    c.g = g;
    c.cfg = cfg;
    c.s = s;
    c.gr.lane = lane;
    c.age = wm.age + (size_t)wid * wm.n_pad;
    c.bestv = wm.bestv + (size_t)wid * SN83_KMAX;
    c.stop = stop;
#if SN83_CANDS_PREFIX
    c.prefix = wm.ps + (size_t)wid * 2 * SN83_PS_WORDS;
    c.suffix = c.prefix + SN83_PS_WORDS;
#endif
}

struct BlockShared {
    WS ws[SN83_WPB];
    u64 ban[SN83_WPB][SN83_MAXW];
    Job job[SN83_WPB];
    int idx[SN83_WPB];
};

// Static job list, no dynamic enqueue.

__global__ __launch_bounds__(SN83_BLOCK) void batch_kernel(
    DGraph g, Cfg cfg, WalkerMem wm, const Job *__restrict__ jobs, int n_jobs,
    u32 *cursor, int *out_size, short *out_vert, u64 *out_fp, u64 *out_traj,
    u64 *ctr, volatile int *stop) {
    __shared__ BlockShared sh;
    const int lane = threadIdx.x % SN83_LANES;
    const int wib = threadIdx.x / SN83_LANES;
    const int wid = blockIdx.x * SN83_WPB + wib;
    WS &s = sh.ws[wib];
    Ctx c;
    ctx_init(c, g, cfg, &s, lane, wm, wid, stop);

    for (;;) {
        if (lane == 0) {
            sh.idx[wib] = (int)atomicAdd(cursor, 1u);
            if (sh.idx[wib] < n_jobs) sh.job[wib] = jobs[sh.idx[wib]];
        }
        gsync(s);
        const int idx = sh.idx[wib];
        if (idx >= n_jobs) break;
        if (*stop) break;

        JobOut o = run_job(c, sh.job[wib], sh.ban[wib]);

        if (lane == 0) {
            out_size[idx] = o.size;
            out_fp[idx] = o.fp;
            if (out_traj) out_traj[idx] = s.traj;
            atomicAdd(&ctr[C_STEPS], (u64)s.step);
            atomicAdd(&ctr[C_JOBS], 1ull);
            atomicAdd(&ctr[C_BANBACK], (u64)o.banback);
            if (s.kmax_hit) atomicAdd(&ctr[C_KMAXHIT], 1ull);
        }
        gsync(s);
        for (int i = lane; i < o.size; i += SN83_LANES)
            out_vert[(size_t)idx * SN83_KMAX + i] = s.C[i];
        gsync(s);
    }
}

// Per-walker steps/s.

__global__ __launch_bounds__(SN83_BLOCK) void probe_kernel(
    DGraph g, Cfg cfg, WalkerMem wm, u64 seed, int max_steps, int n_walkers,
    int *out_best, long long *out_steps, volatile int *stop) {
    __shared__ BlockShared sh;
    const int lane = threadIdx.x % SN83_LANES;
    const int wib = threadIdx.x / SN83_LANES;
    const int wid = blockIdx.x * SN83_WPB + wib;
    if (wid >= n_walkers) return;
    WS &s = sh.ws[wib];
    Ctx c;
    ctx_init(c, g, cfg, &s, lane, wm, wid, stop);
    if (lane == 0) {
        Job j;
        j.seed = seed + 0x9E3779B97F4A7C15ull * (u64)(wid + 1);
        j.max_steps = (u32)max_steps;
        j.epoch = 0;
        j.n_bans = 0;
        j.level = 0;
        j.dead = 0;
        j.pad = 0;
        sh.job[wib] = j;
    }
    gsync(s);
    JobOut o = run_job(c, sh.job[wib], sh.ban[wib]);
    if (lane == 0) {
        out_best[wid] = o.size;
        out_steps[wid] = s.step;
    }
}

// Multi-level queues, epoch, device-side child enqueue.

__global__ __launch_bounds__(SN83_BLOCK) void harvest_kernel(DGraph g, Cfg cfg,
                                                             WalkerMem wm,
                                                             Queues q, Pool p) {
    __shared__ BlockShared sh;
    const int lane = threadIdx.x % SN83_LANES;
    const int wib = threadIdx.x / SN83_LANES;
    const int wid = blockIdx.x * SN83_WPB + wib;
    WS &s = sh.ws[wib];
    Ctx c;
    ctx_init(c, g, cfg, &s, lane, wm, wid, p.stop);
    u64 spin = mix64((u64)wid * 0x9E3779B97F4A7C15ull + 7ull);

    for (;;) {
        if (*p.stop || *(volatile int *)p.done_flag) break;

        if (lane == 0) {
            bool got = q_pop(q, &sh.job[wib]);
            if (!got && cfg.allow_synth) {
                // A starved warp escalates to more seeds at LEVEL 1, not to
                // level 2: a level-1 seed still aims at omega, level 2 aims
                // below it.  Same reason fleet_solver settled on BAN_N = 1.
                u32 nres = atomicAdd(p.n_res, 0u);
                if (nres > p.res_cap) nres = p.res_cap;
                if (nres > 0u) {
                    u32 which = (u32)(mix64(spin) % nres);
                    spin += 0x9E3779B97F4A7C15ull;
                    int sz = p.res_size[which];
                    if (sz > 0) {
                        int bi = (int)(mix64(spin ^ 0xF00Dull) % (unsigned)sz);
                        spin += 0x9E3779B97F4A7C15ull;
                        Job j;
                        j.seed = mix64(spin ^ ((u64)wid << 40));
                        spin += 0x9E3779B97F4A7C15ull;
                        j.max_steps = (u32)(cfg.max_steps_cap / 4 > 0
                                                ? cfg.max_steps_cap / 4
                                                : 1);
                        j.epoch = atomicAdd(p.epoch, 0u);
                        j.n_bans = 1;
                        j.bans[0] =
                            (unsigned short)p.res_vert[(size_t)which * SN83_KMAX + bi];
                        j.level = 1;
                        j.dead = 0;
                        j.pad = 0;
                        sh.job[wib] = j;
                        got = true;
                        atomicAdd(&p.ctr[C_ENQ], 1ull);
                        atomicAdd(&p.ctr[C_SYNTH], 1ull);
                    }
                }
            }
            s.q_flags = got ? 1 : 0;
            if (!got) {
                // Exact quiescence: read DONE first, then ENQ.  If they agree,
                // then at the earlier instant every enqueued job was finished,
                // and nothing running can create more.  Never fires while
                // synthesis is on: harvest runs to the deadline by design.
                u64 d = atomicAdd(&p.ctr[C_DONE], 0ull);
                __threadfence();
                u64 e = atomicAdd(&p.ctr[C_ENQ], 0ull);
                if (d == e) *(volatile int *)p.done_flag = 1;
            }
        }
        gsync(s);
        if (!s.q_flags) {
#if __CUDA_ARCH__ >= 700
            __nanosleep(256);
#endif
            continue;
        }

        const Job job = sh.job[wib];
        if (job.dead || job.epoch < atomicAdd(p.epoch, 0u)) {
            if (lane == 0) atomicAdd(&p.ctr[C_DONE], 1ull);
            gsync(s);
            continue;
        }

        JobOut o = run_job(c, job, sh.ban[wib]);

        if (lane == 0) {
            atomicAdd(&p.ctr[C_STEPS], (u64)s.step);
            atomicAdd(&p.ctr[C_JOBS], 1ull);
            atomicAdd(&p.ctr[C_BANBACK], (u64)o.banback);
            if (s.kmax_hit) atomicAdd(&p.ctr[C_KMAXHIT], 1ull);

            int prev = atomicMax(p.target, o.size);
            int cur = prev > o.size ? prev : o.size;
            // A job came back LARGER than omega: every pool clique is stale.
            if (o.size > prev) atomicAdd(p.epoch, 1u);

            s.keep_flag = 0;
            if (o.size >= cur - cfg.spare_margin) {
                int r = pool_insert(p, o.fp);
                if (r == 1) {
                    atomicAdd(&p.ctr[C_NEW], 1ull);
                    if (o.size >= cur) atomicAdd(&p.ctr[C_NEWMAX], 1ull);
                    u32 slot = atomicAdd(p.n_res, 1u);
                    if (slot < p.res_cap) {
                        p.res_size[slot] = o.size;
                        s.res_slot = (int)slot;
                        s.keep_flag = 1;
                    } else {
                        atomicSub(p.n_res, 1u);
                        atomicAdd(&p.ctr[C_OVERFLOW], 1ull);
                    }
                } else if (r == 0) {
                    atomicAdd(&p.ctr[C_DUP], 1ull);
                    if (o.size >= cur) atomicAdd(&p.ctr[C_DUPMAX], 1ull);
                }
            }
            if (o.size < cur) {
                atomicAdd(&p.ctr[C_SHORT], 1ull);
                // Short-return doubling, the port of BAN_FRAC *= 2.0.
                if ((int)job.max_steps < cfg.max_steps_cap) {
                    Job j = job;
                    u32 ms = job.max_steps * 2u;
                    if ((int)ms > cfg.max_steps_cap || ms < job.max_steps)
                        ms = (u32)cfg.max_steps_cap;
                    j.max_steps = ms;
                    j.seed = mix64(job.seed ^ 0x5DEECE66Dull);
                    j.epoch = atomicAdd(p.epoch, 0u);
                    q_push(q, p, j.level, j);
                }
            }
        }
        gsync_mem(s);

        if (s.keep_flag) {
            const int slot = s.res_slot;
            for (int i = lane; i < o.size; i += SN83_LANES)
                p.res_vert[(size_t)slot * SN83_KMAX + i] = s.C[i];
            __threadfence();
            gsync(s);
            if (lane == 0) {
                // Children only from full-size cliques: a spare's children
                // would aim at omega-2.
                if (cfg.gen_children && o.size >= atomicAdd(p.target, 0)) {
                    for (int i = 0; i < o.size; ++i) {
                        Job j;
                        j.seed = mix64(o.fp ^ ((u64)i * 0x9E3779B97F4A7C15ull));
                        j.max_steps = job.max_steps;
                        j.epoch = atomicAdd(p.epoch, 0u);
                        j.n_bans = 1;
                        j.bans[0] = (unsigned short)p.res_vert[(size_t)slot * SN83_KMAX + i];
                        j.level = 1;
                        j.dead = 0;
                        j.pad = 0;
                        q_push(q, p, 1, j);
                    }
                }
            }
            gsync(s);
        }

        if (lane == 0) atomicAdd(&p.ctr[C_DONE], 1ull);
        gsync(s);
    }
}

// Dumps the candidate sets for a host diff.

__global__ __launch_bounds__(SN83_BLOCK) void cand_dump_kernel(
    DGraph g, Cfg cfg, WalkerMem wm, const short *Cs, const int *ks,
    const u64 *allows, int trials, u64 *o0, u64 *o1, volatile int *stop) {
    __shared__ BlockShared sh;
    const int lane = threadIdx.x % SN83_LANES;
    const int wib = threadIdx.x / SN83_LANES;
    const int wid = blockIdx.x * SN83_WPB + wib;
    if (wid >= trials) return;
    WS &s = sh.ws[wib];
    Ctx c;
    ctx_init(c, g, cfg, &s, lane, wm, wid, stop);

    if (lane == 0) {
        s.k = ks[wid];
        for (int i = 0; i < s.k; ++i) s.C[i] = Cs[(size_t)wid * SN83_KMAX + i];
    }
    for (int w = lane; w < g.W; w += SN83_LANES) {
        s.allow[w] = allows[(size_t)wid * SN83_MAXW + w];
        s.Cb[w] = 0ull;
    }
    gsync(s);
    if (lane == 0)
        for (int i = 0; i < s.k; ++i) {
            int v = s.C[i];
            s.Cb[v >> 6] |= 1ull << (v & 63);
        }
    gsync(s);
    CANDS(c);
    for (int w = lane; w < g.W; w += SN83_LANES) {
        o0[(size_t)wid * SN83_MAXW + w] = s.c0[w];
        o1[(size_t)wid * SN83_MAXW + w] = s.c1[w];
    }
}

// Seeds the pool with a clique found elsewhere (the CPU champion).

__global__ void seed_kernel(Cfg cfg, Queues q, Pool p, const short *verts,
                            int size, u64 seed, int boot_steps) {
    if (threadIdx.x || blockIdx.x) return;
    u64 h = 0ull;
    for (int i = 0; i < size; ++i) h ^= mix64((u64)verts[i] + 1ull);
    h = mix64(h ^ ((u64)(unsigned)size * 0x9E3779B97F4A7C15ull));
    if (!h) h = 1ull;
    atomicMax(p.target, size);
    if (pool_insert(p, h) == 1) {
        u32 slot = atomicAdd(p.n_res, 1u);
        if (slot < p.res_cap) {
            p.res_size[slot] = size;
            for (int i = 0; i < size; ++i)
                p.res_vert[(size_t)slot * SN83_KMAX + i] = verts[i];
            atomicAdd(&p.ctr[C_NEW], 1ull);
        } else {
            atomicSub(p.n_res, 1u);
        }
    }
    (void)cfg;
    for (int i = 0; i < size; ++i) {
        Job j;
        j.seed = mix64(seed ^ ((u64)i * 0x9E3779B97F4A7C15ull));
        j.max_steps = (u32)(boot_steps > 0 ? boot_steps : 1);
        j.epoch = 0;
        j.n_bans = 1;
        j.bans[0] = (unsigned short)verts[i];
        j.level = 1;
        j.dead = 0;
        j.pad = 0;
        q_push(q, p, 1, j);
    }
}

// Host reference for the stage-1 and stage-2 gates.  An independent serial
// implementation of the same search, same RNG in the same call order, so a
// one-walker device run must produce the same trajectory hash and clique.  Its
// candidate sets count misses one vertex at a time -- the definition -- so the
// diff tests the device against the definition and not against itself.

namespace ref {

struct Graph {
    int n = 0, W = 0;
    u64 tail = 0;
    std::vector<u64> bits;
    std::vector<int> deg;
    const u64 *row(int v) const { return &bits[(size_t)v * W]; }
    bool adj(int u, int v) const {
        return (bits[(size_t)u * W + (v >> 6)] >> (v & 63)) & 1ull;
    }
};

struct State {
    const Graph *g = nullptr;
    Cfg cfg;
    std::vector<u64> Cb, conf, allow, c0, c1;
    std::vector<int> age, C, best;
    int k = 0;
    long long step = 0;
    u64 rng = 0, traj = 0;
    bool kmax_hit = false;

    void init(const Graph &gg, const Cfg &cc) {
        g = &gg;
        cfg = cc;
        Cb.assign(gg.W, 0ull);
        conf.assign(gg.W, 0ull);
        allow.assign(gg.W, 0ull);
        c0.assign(gg.W, 0ull);
        c1.assign(gg.W, 0ull);
        age.assign(gg.n, 0);
        C.clear();
        best.clear();
        k = 0;
    }
    u64 allm(int w) const { return w == g->W - 1 ? g->tail : ~0ull; }
};

// slack(v) by brute force: the definition the device is asserted equal to.
inline void cands_bruteforce(const Graph &g, const std::vector<u64> &allow,
                             const std::vector<int> &C, std::vector<u64> &c0,
                             std::vector<u64> &c1) {
    c0.assign(g.W, 0ull);
    c1.assign(g.W, 0ull);
    std::vector<char> inC(g.n, 0);
    for (int v : C) inC[v] = 1;
    for (int v = 0; v < g.n; ++v) {
        if (inC[v]) continue;
        if (!((allow[v >> 6] >> (v & 63)) & 1ull)) continue;
        int miss = 0;
        for (int u : C)
            if (!g.adj(v, u)) ++miss;
        if (miss == 0) c0[v >> 6] |= 1ull << (v & 63);
        else if (miss == 1) c1[v >> 6] |= 1ull << (v & 63);
    }
}

inline void cands(State &s) { cands_bruteforce(*s.g, s.allow, s.C, s.c0, s.c1); }

inline int count(const Graph &g, const std::vector<u64> &a) {
    int c = 0;
    for (int w = 0; w < g.W; ++w) c += __builtin_popcountll(a[w]);
    return c;
}

inline int select_bit(const Graph &g, const std::vector<u64> &a, int rank) {
    for (int w = 0; w < g.W; ++w) {
        int pc = __builtin_popcountll(a[w]);
        if (rank < pc) {
            u64 x = a[w];
            for (int i = 0; i < rank; ++i) x &= x - 1;
            return (w << 6) + __builtin_ctzll(x);
        }
        rank -= pc;
    }
    return -1;
}

inline int bms_pick(State &s, const std::vector<u64> &mask, int m) {
    if (m <= 0) return -1;
    if (m == 1) return select_bit(*s.g, mask, 0);
    const u64 base = sm64(s.rng);
    const bool enumerate = (m <= s.cfg.bms_k);
    const int tries = enumerate ? m : s.cfg.bms_k;
    u64 best = 0ull;
    for (int t = 0; t < tries; ++t) {
        int r = enumerate ? t
                          : (int)(mix64(base + (u64)t * 0x9E3779B97F4A7C15ull) %
                                  (unsigned)m);
        int v = select_bit(*s.g, mask, r);
        const u64 *row = s.g->row(v);
        int sc = 0;
        for (int w = 0; w < s.g->W; ++w) sc += __builtin_popcountll(row[w] & mask[w]);
        u64 key = bms_key(sc, s.age[v], v);
        if (key > best) best = key;
    }
    return bms_vertex(best);
}

inline void tr(State &s, int tag, int v) {
    s.traj = (s.traj ^ (u64)(unsigned)(v + 1) ^ ((u64)tag << 32)) * 0x100000001B3ull;
}

inline void add(State &s, int v, bool freeN) {
    if (s.k < SN83_KMAX) {
        s.C.push_back(v);
        s.k++;
    } else {
        s.kmax_hit = true;
    }
    s.Cb[v >> 6] |= 1ull << (v & 63);
    s.age[v] = (int)s.step;
    tr(s, freeN ? 1 : 2, v);
    if (freeN)
        for (int w = 0; w < s.g->W; ++w) s.conf[w] |= s.g->row(v)[w];
}

inline void drop(State &s, int v) {
    for (size_t i = 0; i < s.C.size(); ++i)
        if (s.C[i] == v) {
            s.C[i] = s.C.back();
            s.C.pop_back();
            break;
        }
    s.k--;
    s.Cb[v >> 6] &= ~(1ull << (v & 63));
    s.conf[v >> 6] &= ~(1ull << (v & 63));
    s.age[v] = (int)s.step;
    tr(s, 3, v);
}

inline void clear(State &s) {
    s.C.clear();
    s.k = 0;
    std::fill(s.Cb.begin(), s.Cb.end(), 0ull);
}

inline int rand_allowed(State &s) {
    int m = count(*s.g, s.allow);
    if (m <= 0) return -1;
    u64 r = sm64(s.rng);
    return select_bit(*s.g, s.allow, (int)(r % (unsigned)m));
}

inline void seed_from(State &s, int v) {
    clear(s);
    for (int w = 0; w < s.g->W; ++w) s.conf[w] = s.allm(w);
    if (v >= 0) add(s, v, true);
}

inline void construct(State &s) {
    for (;;) {
        cands(s);
        int m = count(*s.g, s.c0);
        if (m == 0) return;
        if (s.k >= SN83_KMAX) return;
        int v = bms_pick(s, s.c0, m);
        if (v < 0) return;
        add(s, v, true);
    }
}

inline int blocker(State &s, int v) {
    for (int u : s.C)
        if (!s.g->adj(v, u)) return u;
    return -1;
}

inline void perturb(State &s) {
    bool cold = s.best.empty();
    if (!cold) cold = (sm64(s.rng) % 10ull) == 0ull;
    if (cold) {
        seed_from(s, rand_allowed(s));
        construct(s);
        return;
    }
    const int bs = (int)s.best.size();
    const int m = s.cfg.keep < bs ? s.cfg.keep : bs;
    for (int i = 0; i < m; ++i) {
        int j = i + (int)(sm64(s.rng) % (unsigned)(bs - i));
        std::swap(s.best[i], s.best[j]);
    }
    clear(s);
    for (int w = 0; w < s.g->W; ++w) s.conf[w] = s.allm(w);
    for (int i = 0; i < m; ++i) {
        int v = s.best[i];
        cands(s);
        if ((s.c0[v >> 6] >> (v & 63)) & 1ull) add(s, v, true);
    }
    if (s.k == 0) seed_from(s, rand_allowed(s));
    construct(s);
}

inline void run(State &s, long long max_steps) {
    s.step = 0;
    s.best.clear();
    seed_from(s, rand_allowed(s));
    construct(s);
    s.best = s.C;
    long long last_gain = 0;
    std::vector<u64> pm(s.g->W);
    for (;;) {
        if (max_steps > 0 && s.step >= max_steps) break;
        s.step++;
        if (s.step - last_gain > s.cfg.restart) {
            perturb(s);
            last_gain = s.step;
            continue;
        }
        cands(s);
        for (int w = 0; w < s.g->W; ++w) pm[w] = s.c0[w] & s.conf[w];
        int ma = count(*s.g, pm);
        if (ma > 0) {
            int v = bms_pick(s, pm, ma);
            if (s.k >= SN83_KMAX) break;
            add(s, v, true);
            if (s.k > (int)s.best.size()) {
                s.best = s.C;
                last_gain = s.step;
            }
            continue;
        }
        for (int w = 0; w < s.g->W; ++w) pm[w] = s.c1[w] & s.conf[w];
        int ms = count(*s.g, pm);
        if (ms > 0) {
            int v = bms_pick(s, pm, ms);
            int u = blocker(s, v);
            if (u >= 0) {
                drop(s, u);
                add(s, v, false);
            }
            continue;
        }
        if (s.k > 0) {
            u64 r = sm64(s.rng);
            drop(s, s.C[(int)(r % (unsigned)s.k)]);
        } else {
            seed_from(s, rand_allowed(s));
        }
    }
    construct(s);
    if (s.k > (int)s.best.size()) s.best = s.C;
}

inline int extend_full(State &s, const std::vector<u64> &ban, bool have_ban) {
    int banback = 0;
    std::vector<u64> pm(s.g->W);
    for (;;) {
        for (int w = 0; w < s.g->W; ++w) {
            u64 x = s.allm(w) & ~s.Cb[w];
            for (int v : s.C) x &= s.g->row(v)[w];
            pm[w] = x;
        }
        u64 best = 0ull;
        for (int v = 0; v < s.g->n; ++v)
            if ((pm[v >> 6] >> (v & 63)) & 1ull) {
                u64 key = ((u64)(unsigned)s.g->deg[v] << 16) | (u64)(unsigned)(4095 - v);
                if (key > best) best = key;
            }
        if (!best) break;
        int v = 4095 - (int)(best & 0xFFFFull);
        if (s.k >= SN83_KMAX) break;
        if (have_ban && ((ban[v >> 6] >> (v & 63)) & 1ull)) ++banback;
        s.C.push_back(v);
        s.k++;
        s.Cb[v >> 6] |= 1ull << (v & 63);
    }
    return banback;
}

// the host mirror of run_job
inline void run_job(const Graph &g, const Cfg &cfg, const Job &job, State &s,
                    std::vector<int> *out, u64 *traj_out, long long *steps_out) {
    s.init(g, cfg);
    std::vector<u64> ban(g.W, 0ull);
    for (int w = 0; w < g.W; ++w) s.allow[w] = s.allm(w);
    for (int i = 0; i < (int)job.n_bans && i < SN83_MAXBANS; ++i) {
        int v = job.bans[i];
        if (v >= 0 && v < g.n) ban[v >> 6] |= 1ull << (v & 63);
    }
    for (int w = 0; w < g.W; ++w) s.allow[w] &= ~ban[w];
    s.rng = mix64(job.seed ^ 0xA5A5A5A5DEADBEEFull);
    s.traj = 0xCBF29CE484222325ull;

    run(s, (long long)job.max_steps);

    clear(s);
    for (int w = 0; w < g.W; ++w) s.conf[w] = s.allm(w);
    for (int v : s.best) {
        s.C.push_back(v);
        s.Cb[v >> 6] |= 1ull << (v & 63);
    }
    s.k = (int)s.C.size();
    extend_full(s, ban, job.n_bans != 0);
    if (out) *out = s.C;
    if (traj_out) *traj_out = s.traj;
    if (steps_out) *steps_out = s.step;
}

}  // namespace ref

// ------------------------------- host side ---------------------------------

namespace {

char g_err[512] = {0};

#define CK(call)                                                              \
    do {                                                                      \
        cudaError_t e_ = (call);                                              \
        if (e_ != cudaSuccess) {                                              \
            snprintf(g_err, sizeof(g_err), "%s:%d %s: %s", __FILE__, __LINE__, \
                     #call, cudaGetErrorString(e_));                          \
            return -1;                                                        \
        }                                                                     \
    } while (0)

struct Handle {
    int n = 0, W = 0, n_pad = 0;
    u64 tail = 0;
    std::vector<u64> hbits;
    std::vector<int> hdeg;

    u64 *d_bits = nullptr;
    int *d_deg = nullptr;

    int n_walkers = 0, n_blocks = 0;
    int *d_age = nullptr;
    short *d_bestv = nullptr;
#if SN83_CANDS_PREFIX
    u64 *d_ps = nullptr;
#endif

    int *h_stop = nullptr;   // mapped pinned
    int *d_stop = nullptr;

    Job *d_ring = nullptr;
    u32 *d_qcnt = nullptr;   // head[NLEV] ready[NLEV] tail[NLEV]
    u32 q_cap = 0;
    u64 *d_fp = nullptr;
    u32 fpcap = 0;
    int *d_res_size = nullptr;
    short *d_res_vert = nullptr;
    u32 *d_n_res = nullptr;
    u32 res_cap = 0;
    int *d_target = nullptr;
    u32 *d_epoch = nullptr;
    int *d_done = nullptr;
    u64 *d_ctr = nullptr;
    short *d_seedv = nullptr;

    cudaStream_t stream = nullptr;
    ref::Graph rg;
};

DGraph dgraph(const Handle &h) {
    DGraph g;
    g.n = h.n;
    g.W = h.W;
    g.tail = h.tail;
    g.bits = h.d_bits;
    g.deg = h.d_deg;
    return g;
}

WalkerMem wmem(const Handle &h) {
    WalkerMem w;
    w.age = h.d_age;
    w.bestv = h.d_bestv;
    w.n_pad = h.n_pad;
#if SN83_CANDS_PREFIX
    w.ps = h.d_ps;
#endif
    return w;
}

Queues queues(const Handle &h) {
    Queues q;
    q.ring = h.d_ring;
    q.head = h.d_qcnt;
    q.ready = h.d_qcnt + SN83_NLEV;
    q.tail = h.d_qcnt + 2 * SN83_NLEV;
    q.cap = h.q_cap;
    return q;
}

Pool poolof(const Handle &h) {
    Pool p;
    p.fp = h.d_fp;
    p.fpcap = h.fpcap;
    p.res_size = h.d_res_size;
    p.res_vert = h.d_res_vert;
    p.n_res = h.d_n_res;
    p.res_cap = h.res_cap;
    p.target = h.d_target;
    p.epoch = h.d_epoch;
    p.done_flag = h.d_done;
    p.ctr = h.d_ctr;
    p.stop = h.d_stop;
    return p;
}

Cfg make_cfg(int bms_k, int restart, int keep, int spare_margin, int cap,
             int synth, int children) {
    Cfg c;
    c.bms_k = bms_k > 0 ? bms_k : 64;
    c.restart = restart > 0 ? restart : 4000;
    c.keep = keep > 0 ? keep : 4;
    c.spare_margin = spare_margin >= 0 ? spare_margin : 1;
    c.max_steps_cap = cap > 0 ? cap : (1 << 20);
    c.allow_synth = synth;
    c.gen_children = children;
    return c;
}

Cfg default_cfg() { return make_cfg(64, 4000, 4, 1, 1 << 20, 1, 1); }

// Size the grid to what is actually resident rather than assuming, so register
// pressure shows up as a smaller grid instead of silently halving occupancy.
int resident_blocks(const void *kernel, int *per_sm_out) {
    int per_sm = 0;
    cudaError_t e =
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, SN83_BLOCK, 0);
    if (e != cudaSuccess || per_sm <= 0) per_sm = 1;
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    if (per_sm_out) *per_sm_out = per_sm;
    return per_sm * prop.multiProcessorCount;
}

double now_s() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

}  // namespace

extern "C" {

int sn83_gpu_last_error(char *buf, int len) {
    if (buf && len > 0) {
        strncpy(buf, g_err, (size_t)len - 1);
        buf[len - 1] = 0;
    }
    return (int)strlen(g_err);
}

int sn83_gpu_config(int *lanes, int *wpb, int *kmax, int *maxn, int *prefix_arm,
                    int *nctr) {
    if (lanes) *lanes = SN83_LANES;
    if (wpb) *wpb = SN83_WPB;
    if (kmax) *kmax = SN83_KMAX;
    if (maxn) *maxn = SN83_MAXN;
    if (prefix_arm) *prefix_arm = SN83_CANDS_PREFIX;
    if (nctr) *nctr = C_NCTR;
    return 0;
}

int sn83_gpu_device_info(char *buf, int len) {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return -1;
    cudaDeviceProp p;
    if (cudaGetDeviceProperties(&p, dev) != cudaSuccess) return -1;
    int per_sm = 0;
    int blocks = resident_blocks((const void *)harvest_kernel, &per_sm);
    snprintf(buf, (size_t)len,
             "%s sm_%d%d SMs=%d L2=%dKB mem=%zuMB shared/block=%zuB blocks/SM=%d "
             "resident_blocks=%d walkers=%d lanes=%d cands=%s kmax=%d",
             p.name, p.major, p.minor, p.multiProcessorCount, p.l2CacheSize / 1024,
             (size_t)(p.totalGlobalMem >> 20), sizeof(BlockShared), per_sm, blocks,
             blocks * SN83_WPB, SN83_LANES,
             SN83_CANDS_PREFIX ? "prefix/suffix" : "carry", SN83_KMAX);
    return 0;
}

void *sn83_gpu_open(const unsigned char *adj, int n, int walkers_hint) {
    g_err[0] = 0;
    if (n <= 0 || n > SN83_MAXN) {
        snprintf(g_err, sizeof(g_err), "n=%d out of range (1..%d)", n, SN83_MAXN);
        return nullptr;
    }
    Handle *h = new Handle();
    h->n = n;
    h->W = (n + 63) / 64;
    h->n_pad = ((n + 31) / 32) * 32;
    int rem = n - 64 * (h->W - 1);
    h->tail = (rem >= 64) ? ~0ull : ((1ull << rem) - 1ull);
    h->hbits.assign((size_t)n * h->W, 0ull);
    h->hdeg.assign(n, 0);
    for (int i = 0; i < n; ++i) {
        const unsigned char *r = adj + (size_t)i * n;
        int d = 0;
        for (int j = 0; j < n; ++j) {
            if (j == i) continue;
            if (r[j]) {
                h->hbits[(size_t)i * h->W + (j >> 6)] |= 1ull << (j & 63);
                ++d;
            }
        }
        h->hdeg[i] = d;
    }
    h->rg.n = n;
    h->rg.W = h->W;
    h->rg.tail = h->tail;
    h->rg.bits = h->hbits;
    h->rg.deg = h->hdeg;

    cudaError_t e;
    auto fail = [&](const char *what) -> void * {
        snprintf(g_err, sizeof(g_err), "%s: %s", what, cudaGetErrorString(e));
        delete h;
        return nullptr;
    };
    if ((e = cudaSetDeviceFlags(cudaDeviceMapHost)) != cudaSuccess &&
        e != cudaErrorSetOnActiveProcess)
        return fail("cudaSetDeviceFlags");
    e = cudaSuccess;
    if ((e = cudaStreamCreate(&h->stream)) != cudaSuccess) return fail("cudaStreamCreate");
    if ((e = cudaMalloc(&h->d_bits, h->hbits.size() * sizeof(u64))) != cudaSuccess)
        return fail("cudaMalloc bits");
    if ((e = cudaMalloc(&h->d_deg, (size_t)n * sizeof(int))) != cudaSuccess)
        return fail("cudaMalloc deg");
    cudaMemcpy(h->d_bits, h->hbits.data(), h->hbits.size() * sizeof(u64),
               cudaMemcpyHostToDevice);
    cudaMemcpy(h->d_deg, h->hdeg.data(), (size_t)n * sizeof(int), cudaMemcpyHostToDevice);

    int blocks = resident_blocks((const void *)harvest_kernel, nullptr);
    if (walkers_hint > 0) {
        int want = (walkers_hint + SN83_WPB - 1) / SN83_WPB;
        if (want < blocks) blocks = want;
    }
    if (blocks < 1) blocks = 1;
    h->n_blocks = blocks;
    h->n_walkers = blocks * SN83_WPB;

    if ((e = cudaMalloc(&h->d_age, (size_t)h->n_walkers * h->n_pad * sizeof(int))) !=
        cudaSuccess)
        return fail("cudaMalloc age");
    if ((e = cudaMalloc(&h->d_bestv,
                        (size_t)h->n_walkers * SN83_KMAX * sizeof(short))) != cudaSuccess)
        return fail("cudaMalloc bestv");
#if SN83_CANDS_PREFIX
    if ((e = cudaMalloc(&h->d_ps, (size_t)h->n_walkers * 2 * SN83_PS_WORDS *
                                      sizeof(u64))) != cudaSuccess)
        return fail("cudaMalloc prefix/suffix");
#endif
    if ((e = cudaHostAlloc(&h->h_stop, sizeof(int), cudaHostAllocMapped)) != cudaSuccess)
        return fail("cudaHostAlloc stop");
    *h->h_stop = 0;
    if ((e = cudaHostGetDevicePointer(&h->d_stop, h->h_stop, 0)) != cudaSuccess)
        return fail("cudaHostGetDevicePointer stop");

    h->q_cap = 1u << 16;
    h->res_cap = 1u << 13;
    h->fpcap = 1u << 17;
    if ((e = cudaMalloc(&h->d_ring, (size_t)SN83_NLEV * h->q_cap * sizeof(Job))) !=
        cudaSuccess)
        return fail("cudaMalloc ring");
    if ((e = cudaMalloc(&h->d_qcnt, 3 * SN83_NLEV * sizeof(u32))) != cudaSuccess)
        return fail("cudaMalloc qcnt");
    if ((e = cudaMalloc(&h->d_fp, (size_t)h->fpcap * sizeof(u64))) != cudaSuccess)
        return fail("cudaMalloc fp");
    if ((e = cudaMalloc(&h->d_res_size, (size_t)h->res_cap * sizeof(int))) != cudaSuccess)
        return fail("cudaMalloc res_size");
    if ((e = cudaMalloc(&h->d_res_vert,
                        (size_t)h->res_cap * SN83_KMAX * sizeof(short))) != cudaSuccess)
        return fail("cudaMalloc res_vert");
    if ((e = cudaMalloc(&h->d_n_res, sizeof(u32))) != cudaSuccess)
        return fail("cudaMalloc n_res");
    if ((e = cudaMalloc(&h->d_target, sizeof(int))) != cudaSuccess)
        return fail("cudaMalloc target");
    if ((e = cudaMalloc(&h->d_epoch, sizeof(u32))) != cudaSuccess)
        return fail("cudaMalloc epoch");
    if ((e = cudaMalloc(&h->d_done, sizeof(int))) != cudaSuccess)
        return fail("cudaMalloc done");
    if ((e = cudaMalloc(&h->d_ctr, C_NCTR * sizeof(u64))) != cudaSuccess)
        return fail("cudaMalloc ctr");
    if ((e = cudaMalloc(&h->d_seedv, SN83_KMAX * sizeof(short))) != cudaSuccess)
        return fail("cudaMalloc seedv");
    return h;
}

void sn83_gpu_close(void *hv) {
    Handle *h = (Handle *)hv;
    if (!h) return;
    cudaFree(h->d_bits);
    cudaFree(h->d_deg);
    cudaFree(h->d_age);
    cudaFree(h->d_bestv);
#if SN83_CANDS_PREFIX
    cudaFree(h->d_ps);
#endif
    if (h->h_stop) cudaFreeHost(h->h_stop);
    cudaFree(h->d_ring);
    cudaFree(h->d_qcnt);
    cudaFree(h->d_fp);
    cudaFree(h->d_res_size);
    cudaFree(h->d_res_vert);
    cudaFree(h->d_n_res);
    cudaFree(h->d_target);
    cudaFree(h->d_epoch);
    cudaFree(h->d_done);
    cudaFree(h->d_ctr);
    cudaFree(h->d_seedv);
    if (h->stream) cudaStreamDestroy(h->stream);
    delete h;
}

int sn83_gpu_walkers(void *hv) {
    Handle *h = (Handle *)hv;
    return h ? h->n_walkers : 0;
}



int sn83_gpu_probe(void *hv, int n_walkers, int max_steps, unsigned long long seed,
                   double *out_secs, long long *out_steps, int *out_best) {
    Handle *h = (Handle *)hv;
    if (!h) return -1;
    if (n_walkers <= 0 || n_walkers > h->n_walkers) n_walkers = h->n_walkers;
    int blocks = (n_walkers + SN83_WPB - 1) / SN83_WPB;
    int *d_best = nullptr;
    long long *d_steps = nullptr;
    CK(cudaMalloc(&d_best, (size_t)n_walkers * sizeof(int)));
    CK(cudaMalloc(&d_steps, (size_t)n_walkers * sizeof(long long)));
    cudaMemset(d_best, 0, (size_t)n_walkers * sizeof(int));
    cudaMemset(d_steps, 0, (size_t)n_walkers * sizeof(long long));
    *h->h_stop = 0;
    CK(cudaDeviceSynchronize());
    double t0 = now_s();
    probe_kernel<<<blocks, SN83_BLOCK>>>(dgraph(*h), default_cfg(), wmem(*h), seed,
                                         max_steps, n_walkers, d_best, d_steps,
                                         h->d_stop);
    CK(cudaDeviceSynchronize());
    double t1 = now_s();
    CK(cudaGetLastError());
    if (out_secs) *out_secs = t1 - t0;
    cudaMemcpy(out_best, d_best, (size_t)n_walkers * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_steps, d_steps, (size_t)n_walkers * sizeof(long long),
               cudaMemcpyDeviceToHost);
    cudaFree(d_best);
    cudaFree(d_steps);
    return n_walkers;
}



int sn83_gpu_check_cands(void *hv, unsigned long long seed, int trials,
                         int max_clique, int n_bans, int *out_bad) {
    Handle *h = (Handle *)hv;
    if (!h) return -1;
    if (trials <= 0) return 0;
    if (trials > h->n_walkers) trials = h->n_walkers;
    if (max_clique > SN83_KMAX) max_clique = SN83_KMAX;
    if (max_clique < 1) max_clique = 1;

    std::vector<short> Cs((size_t)trials * SN83_KMAX, 0);
    std::vector<int> ks(trials, 0);
    std::vector<u64> allows((size_t)trials * SN83_MAXW, 0ull);
    std::vector<std::vector<int> > hostC(trials);
    std::vector<std::vector<u64> > hostAllow(trials);

    u64 rng = seed;
    for (int t = 0; t < trials; ++t) {
        std::vector<u64> allow(h->W, 0ull);
        for (int w = 0; w < h->W; ++w) allow[w] = (w == h->W - 1) ? h->tail : ~0ull;
        for (int b = 0; b < n_bans; ++b) {
            int v = (int)(sm64(rng) % (unsigned)h->n);
            allow[v >> 6] &= ~(1ull << (v & 63));
        }
        // Half the trials grow a real clique (the shape the search sees), half
        // take an arbitrary vertex subset. slack is defined either way, and an
        // arbitrary set is the harsher test of the carry recurrence.
        std::vector<int> C;
        const bool want_clique = (t & 1) == 0;
        const int want = 1 + (int)(sm64(rng) % (unsigned)max_clique);
        for (int guard = 0; guard < 12 * want && (int)C.size() < want; ++guard) {
            int v = (int)(sm64(rng) % (unsigned)h->n);
            if (!((allow[v >> 6] >> (v & 63)) & 1ull)) continue;
            bool skip = false;
            for (size_t i = 0; i < C.size(); ++i)
                if (C[i] == v) skip = true;
            if (skip) continue;
            if (want_clique)
                for (size_t i = 0; i < C.size() && !skip; ++i)
                    if (!h->rg.adj(C[i], v)) skip = true;
            if (skip) continue;
            C.push_back(v);
        }
        ks[t] = (int)C.size();
        for (size_t i = 0; i < C.size(); ++i) Cs[(size_t)t * SN83_KMAX + i] = (short)C[i];
        for (int w = 0; w < h->W; ++w) allows[(size_t)t * SN83_MAXW + w] = allow[w];
        hostC[t] = C;
        hostAllow[t] = allow;
    }

    short *d_Cs = nullptr;
    int *d_ks = nullptr;
    u64 *d_allow = nullptr, *d_o0 = nullptr, *d_o1 = nullptr;
    CK(cudaMalloc(&d_Cs, Cs.size() * sizeof(short)));
    CK(cudaMalloc(&d_ks, ks.size() * sizeof(int)));
    CK(cudaMalloc(&d_allow, allows.size() * sizeof(u64)));
    CK(cudaMalloc(&d_o0, (size_t)trials * SN83_MAXW * sizeof(u64)));
    CK(cudaMalloc(&d_o1, (size_t)trials * SN83_MAXW * sizeof(u64)));
    cudaMemcpy(d_Cs, Cs.data(), Cs.size() * sizeof(short), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ks, ks.data(), ks.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_allow, allows.data(), allows.size() * sizeof(u64),
               cudaMemcpyHostToDevice);
    cudaMemset(d_o0, 0, (size_t)trials * SN83_MAXW * sizeof(u64));
    cudaMemset(d_o1, 0, (size_t)trials * SN83_MAXW * sizeof(u64));
    *h->h_stop = 0;
    int blocks = (trials + SN83_WPB - 1) / SN83_WPB;
    cand_dump_kernel<<<blocks, SN83_BLOCK>>>(dgraph(*h), default_cfg(), wmem(*h), d_Cs,
                                             d_ks, d_allow, trials, d_o0, d_o1,
                                             h->d_stop);
    CK(cudaDeviceSynchronize());
    CK(cudaGetLastError());
    std::vector<u64> o0((size_t)trials * SN83_MAXW), o1((size_t)trials * SN83_MAXW);
    cudaMemcpy(&o0[0], d_o0, o0.size() * sizeof(u64), cudaMemcpyDeviceToHost);
    cudaMemcpy(&o1[0], d_o1, o1.size() * sizeof(u64), cudaMemcpyDeviceToHost);
    cudaFree(d_Cs);
    cudaFree(d_ks);
    cudaFree(d_allow);
    cudaFree(d_o0);
    cudaFree(d_o1);

    int bad = 0;
    for (int t = 0; t < trials; ++t) {
        std::vector<u64> r0, r1;
        ref::cands_bruteforce(h->rg, hostAllow[t], hostC[t], r0, r1);
        for (int w = 0; w < h->W; ++w)
            if (o0[(size_t)t * SN83_MAXW + w] != r0[w] ||
                o1[(size_t)t * SN83_MAXW + w] != r1[w]) {
                ++bad;
                break;
            }
    }
    if (out_bad) *out_bad = bad;
    return trials;
}

// bans: n_jobs rows of SN83_MAXBANS int32 (-1 unused).
// out_vert: n_jobs rows of SN83_KMAX int32.

int sn83_gpu_solve_batch(void *hv, const unsigned long long *seeds, const int *bans,
                         const int *n_bans, int n_jobs, int max_steps,
                         double time_limit, int *out_size, int *out_vert,
                         unsigned long long *out_traj, long long *out_ctr) {
    Handle *h = (Handle *)hv;
    if (!h || n_jobs <= 0) return -1;

    std::vector<Job> jobs(n_jobs);
    for (int i = 0; i < n_jobs; ++i) {
        Job &j = jobs[i];
        memset(&j, 0, sizeof(Job));
        j.seed = seeds[i];
        j.max_steps = (u32)max_steps;
        j.epoch = 0;
        j.level = 0;
        int nb = n_bans ? n_bans[i] : 0;
        if (nb > SN83_MAXBANS) nb = SN83_MAXBANS;
        j.n_bans = (unsigned char)nb;
        for (int b = 0; b < nb; ++b) j.bans[b] = (unsigned short)bans[(size_t)i * SN83_MAXBANS + b];
        j.level = (unsigned char)nb;
    }

    Job *d_jobs = nullptr;
    u32 *d_cursor = nullptr;
    int *d_size = nullptr;
    short *d_vert = nullptr;
    u64 *d_fp = nullptr, *d_traj = nullptr;
    CK(cudaMalloc(&d_jobs, (size_t)n_jobs * sizeof(Job)));
    CK(cudaMalloc(&d_cursor, sizeof(u32)));
    CK(cudaMalloc(&d_size, (size_t)n_jobs * sizeof(int)));
    CK(cudaMalloc(&d_vert, (size_t)n_jobs * SN83_KMAX * sizeof(short)));
    CK(cudaMalloc(&d_fp, (size_t)n_jobs * sizeof(u64)));
    CK(cudaMalloc(&d_traj, (size_t)n_jobs * sizeof(u64)));
    cudaMemcpy(d_jobs, &jobs[0], (size_t)n_jobs * sizeof(Job), cudaMemcpyHostToDevice);
    cudaMemset(d_cursor, 0, sizeof(u32));
    cudaMemset(d_size, 0, (size_t)n_jobs * sizeof(int));
    cudaMemset(d_vert, 0, (size_t)n_jobs * SN83_KMAX * sizeof(short));
    cudaMemset(h->d_ctr, 0, C_NCTR * sizeof(u64));
    *h->h_stop = 0;

    Cfg cfg = make_cfg(64, 4000, 4, 1, 1 << 20, 0, 0);
    int blocks = h->n_blocks;
    if (blocks > (n_jobs + SN83_WPB - 1) / SN83_WPB)
        blocks = (n_jobs + SN83_WPB - 1) / SN83_WPB;
    batch_kernel<<<blocks, SN83_BLOCK, 0, h->stream>>>(
        dgraph(*h), cfg, wmem(*h), d_jobs, n_jobs, d_cursor, d_size, d_vert, d_fp,
        d_traj, h->d_ctr, h->d_stop);

    if (time_limit > 0.0) {
        double deadline = now_s() + time_limit;
        while (cudaStreamQuery(h->stream) == cudaErrorNotReady) {
            if (now_s() >= deadline) {
                *h->h_stop = 1;
                break;
            }
            struct timespec ts = {0, 2 * 1000 * 1000};
            nanosleep(&ts, nullptr);
        }
    }
    CK(cudaStreamSynchronize(h->stream));
    CK(cudaGetLastError());

    std::vector<short> verts((size_t)n_jobs * SN83_KMAX);
    cudaMemcpy(out_size, d_size, (size_t)n_jobs * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&verts[0], d_vert, verts.size() * sizeof(short), cudaMemcpyDeviceToHost);
    if (out_traj)
        cudaMemcpy(out_traj, d_traj, (size_t)n_jobs * sizeof(u64), cudaMemcpyDeviceToHost);
    if (out_ctr) {
        u64 ctr[C_NCTR];
        cudaMemcpy(ctr, h->d_ctr, C_NCTR * sizeof(u64), cudaMemcpyDeviceToHost);
        for (int i = 0; i < C_NCTR; ++i) out_ctr[i] = (long long)ctr[i];
    }
    for (size_t i = 0; i < verts.size(); ++i) out_vert[i] = verts[i];
    cudaFree(d_jobs);
    cudaFree(d_cursor);
    cudaFree(d_size);
    cudaFree(d_vert);
    cudaFree(d_fp);
    cudaFree(d_traj);
    return n_jobs;
}

// One-walker trajectory diff against the host mirror.

int sn83_gpu_check_trajectory(void *hv, unsigned long long seed, int max_steps,
                              int n_bans, long long *out) {
    // out: [dev_size, ref_size, dev_traj, ref_traj, ref_steps, same_clique]
    Handle *h = (Handle *)hv;
    if (!h) return -1;

    Job job;
    memset(&job, 0, sizeof(job));
    job.seed = seed;
    job.max_steps = (u32)max_steps;
    if (n_bans > SN83_MAXBANS) n_bans = SN83_MAXBANS;
    job.n_bans = (unsigned char)n_bans;
    u64 r = seed ^ 0x1234567ull;
    std::vector<int> banlist;
    for (int i = 0; i < n_bans; ++i) {
        int v = (int)(sm64(r) % (unsigned)h->n);
        job.bans[i] = (unsigned short)v;
        banlist.push_back(v);
    }

    std::vector<int> hb(SN83_MAXBANS, -1);
    for (int i = 0; i < n_bans; ++i) hb[i] = banlist[i];
    std::vector<unsigned long long> seeds(1, seed);
    std::vector<int> nb(1, n_bans);
    std::vector<int> dsize(1, 0), dvert(SN83_KMAX, 0);
    std::vector<unsigned long long> dtraj(1, 0);
    if (sn83_gpu_solve_batch(hv, &seeds[0], &hb[0], &nb[0], 1, max_steps, 0.0,
                             &dsize[0], &dvert[0], &dtraj[0], nullptr) < 0)
        return -1;

    ref::State st;
    std::vector<int> rc;
    u64 rtraj = 0;
    long long rsteps = 0;
    ref::run_job(h->rg, make_cfg(64, 4000, 4, 1, 1 << 20, 0, 0), job, st, &rc, &rtraj,
                 &rsteps);

    std::vector<int> dv(dvert.begin(), dvert.begin() + dsize[0]);
    std::sort(dv.begin(), dv.end());
    std::vector<int> rv(rc);
    std::sort(rv.begin(), rv.end());

    out[0] = dsize[0];
    out[1] = (long long)rc.size();
    out[2] = (long long)dtraj[0];
    out[3] = (long long)rtraj;
    out[4] = rsteps;
    out[5] = (dv == rv) ? 1 : 0;
    return 0;
}

// out_vert: max_out rows of SN83_KMAX int32.  Returns the result count, or -1.
// Every result is maximal in the FULL graph: bans constrain the search, never
// the answer.

int sn83_gpu_harvest(void *hv, double time_limit, unsigned long long seed,
                     int max_steps, int n_boot, int boot_steps, int max_steps_cap,
                     int spare_margin, const int *init_clique, int init_size,
                     int *out_size, int *out_vert, int max_out, long long *out_ctr) {
    Handle *h = (Handle *)hv;
    if (!h) return -1;
    if (max_steps <= 0) max_steps = 20000;
    if (boot_steps <= 0) boot_steps = max_steps;
    if (n_boot <= 0) n_boot = h->n_walkers;
    if ((u32)n_boot > h->q_cap) n_boot = (int)h->q_cap;
    if (max_steps_cap <= 0) max_steps_cap = 1 << 20;
    if (spare_margin < 0) spare_margin = 1;

    cudaMemset(h->d_qcnt, 0, 3 * SN83_NLEV * sizeof(u32));
    cudaMemset(h->d_fp, 0, (size_t)h->fpcap * sizeof(u64));
    cudaMemset(h->d_n_res, 0, sizeof(u32));
    // Zeroed too, not just n_res: a synthesizing warp may read a slot another
    // warp reserved but has not filled, and a stale size would send it reading
    // uninitialised vertices.  Bans are range-checked, so this was never an
    // out-of-bounds read -- but a ban drawn from garbage is a wasted job.
    cudaMemset(h->d_res_size, 0, (size_t)h->res_cap * sizeof(int));
    cudaMemset(h->d_target, 0, sizeof(int));
    cudaMemset(h->d_epoch, 0, sizeof(u32));
    cudaMemset(h->d_done, 0, sizeof(int));
    cudaMemset(h->d_ctr, 0, C_NCTR * sizeof(u64));
    *h->h_stop = 0;

    // Bootstrap: a burst of unbanned walkers on distinct seeds.  Yields omega
    // from more than one walker, and several distinct omega-cliques at once so
    // level 1 starts wide instead of with one parent's children.
    std::vector<Job> boot(n_boot);
    for (int i = 0; i < n_boot; ++i) {
        memset(&boot[i], 0, sizeof(Job));
        boot[i].seed = mix64(seed + 0x9E3779B97F4A7C15ull * (u64)(i + 1));
        boot[i].max_steps = (u32)boot_steps;
        boot[i].epoch = 0;
        boot[i].n_bans = 0;
        boot[i].level = 0;
    }
    CK(cudaMemcpy(h->d_ring, &boot[0], (size_t)n_boot * sizeof(Job),
                  cudaMemcpyHostToDevice));
    u32 cnts[3] = {0u, (u32)n_boot, (u32)n_boot};   // head[0], ready[0], tail[0]
    CK(cudaMemcpy(h->d_qcnt + 0, &cnts[0], sizeof(u32), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(h->d_qcnt + SN83_NLEV, &cnts[1], sizeof(u32), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(h->d_qcnt + 2 * SN83_NLEV, &cnts[2], sizeof(u32),
                  cudaMemcpyHostToDevice));
    u64 enq = (u64)n_boot;
    CK(cudaMemcpy(h->d_ctr + C_ENQ, &enq, sizeof(u64), cudaMemcpyHostToDevice));

    Cfg cfg = make_cfg(64, 4000, 4, spare_margin, max_steps_cap, 1, 1);

    if (init_clique && init_size > 0) {
        int sz = init_size > SN83_KMAX ? SN83_KMAX : init_size;
        std::vector<short> sv(sz);
        for (int i = 0; i < sz; ++i) sv[i] = (short)init_clique[i];
        CK(cudaMemcpy(h->d_seedv, &sv[0], (size_t)sz * sizeof(short),
                      cudaMemcpyHostToDevice));
        seed_kernel<<<1, 32, 0, h->stream>>>(cfg, queues(*h), poolof(*h), h->d_seedv, sz,
                                             mix64(seed ^ 0xBEEFull), max_steps);
    }

    harvest_kernel<<<h->n_blocks, SN83_BLOCK, 0, h->stream>>>(dgraph(*h), cfg, wmem(*h),
                                                              queues(*h), poolof(*h));

    const double deadline = now_s() + time_limit;
    for (;;) {
        cudaError_t q = cudaStreamQuery(h->stream);
        if (q != cudaErrorNotReady) break;
        if (now_s() >= deadline) {
            *h->h_stop = 1;
            break;
        }
        struct timespec ts = {0, 2 * 1000 * 1000};
        nanosleep(&ts, nullptr);
    }
    CK(cudaStreamSynchronize(h->stream));
    CK(cudaGetLastError());

    u32 n_res = 0;
    CK(cudaMemcpy(&n_res, h->d_n_res, sizeof(u32), cudaMemcpyDeviceToHost));
    if (n_res > h->res_cap) n_res = h->res_cap;
    int take = (int)n_res;
    if (take > max_out) take = max_out;
    if (take > 0) {
        std::vector<short> verts((size_t)take * SN83_KMAX);
        CK(cudaMemcpy(out_size, h->d_res_size, (size_t)take * sizeof(int),
                      cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(&verts[0], h->d_res_vert, verts.size() * sizeof(short),
                      cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < verts.size(); ++i) out_vert[i] = verts[i];
    }
    if (out_ctr) {
        u64 ctr[C_NCTR];
        CK(cudaMemcpy(ctr, h->d_ctr, C_NCTR * sizeof(u64), cudaMemcpyDeviceToHost));
        for (int i = 0; i < C_NCTR; ++i) out_ctr[i] = (long long)ctr[i];
    }
    return take;
}

}  // extern "C"
