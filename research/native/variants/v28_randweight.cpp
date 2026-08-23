// SN83 maximum-clique core: bitset adjacency + LSCC-style local search.
//
// Why C++: the field's margin is one vertex on graphs up to |V|=900 inside a 6s
// deadline, and a numpy plateau search saturates long before it gets there
// (measured: 10x the budget bought exactly zero vertices). The wins come from
// iteration count and from the two mechanisms below, both of which want tight
// loops over 64-bit words.
//
// The two mechanisms that separate this from a naive plateau search:
//
//   * strong configuration checking (SCC) — a vertex swapped out of C may not
//     come back until one of its neighbours is genuinely *added*. Plain CC
//     frees a vertex whenever any neighbour enters, which on a density-0.9
//     graph means "always", so the restriction has to be the strong variant to
//     bite at all here.
//   * stagnation-driven perturbation — the check runs every step, not only when
//     the search runs out of moves. An earlier revision gated it behind "no
//     legal move" and measured 3.1M swaps against 2 adds: on a dense graph a
//     plateau move is essentially always available, so the search never
//     restarted and never improved.
//
// VARIANT v7_fastscan: replace the O(n) per-step candidate scan with incremental
// bucket maintenance over the COMPLEMENT graph.
//
// The champion rebuilds its add/swap candidate lists by scanning all n vertices
// every step. Track slack(v) = |C| - |N(v) cap C| instead and the update becomes
// tiny, because of an identity that only helps at high density:
//
//   adding u:  |C| += 1        -> slack += 1 for EVERY vertex
//              cnt += 1 for N(u) -> slack -= 1 for every neighbour
//   net:       slack unchanged for neighbours, +1 for NON-neighbours only.
//
// At density 0.9 a vertex has ~90 non-neighbours and ~810 neighbours, so the work
// per move drops from ~n to ~n/10. Add-candidates are exactly slack==0 and
// swap-candidates exactly slack==1, so they can be kept in two O(1) indexed sets
// instead of being rediscovered by a scan. This is MoMC's complement-representation
// trick (http://www.mis.u-picardie.fr/~cli/MoMC2016.c) applied to local search.
//
// v24 = v7 plus an UNGUIDED PLATEAU RANDOM WALK before returning.
//
// The problem, restated from the measurements: we are at the 36th percentile of the
// field on replayed reward despite 99.8% parity, because we collide on ~65% of tasks
// while the field's coordinated miners collide on ~25%. Optimality is pinned; the
// entire deficit is uniqueness, and it needs the fresh rate to roughly double.
//
// Why our answers cluster with everyone else's: the search is DRAWN to cliques with
// large basins of attraction, and a large basin is exactly what makes a clique easy
// for every other solver to find. Our plateau moves are guided — scored by candidate
// overlap and filtered by configuration checking — so they keep us inside the
// attractive region rather than exploring the plateau.
//
// This walks the plateau WITHOUT guidance: once the best size is known, repeatedly
// take a uniformly random (1,1) swap at that size, ignoring scores and conf. A
// guided walk concentrates on high-basin cliques; an unguided one diffuses toward a
// uniform sample of the plateau, which is where the un-taken cliques live.
//
// Safety: a (1,1) swap preserves |C| and preserves cliqueness by construction, and
// the final maximality pass still runs, so optimality cannot move. If a swap ever
// exposes an addable vertex we take it — that is a strictly bigger clique.
//
// SN83_WALK = number of random plateau swaps (0 = exactly v7).
// SN83_WALKPCT = percent of the deadline RESERVED for the walk. Without this the
// walk is dead code: the search runs to the deadline, so a walk that starts
// afterwards has no time and exits on its first clock check.
//
// Exposed through a flat C ABI so ctypes can call it with no build-time
// dependency beyond g++ itself.
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

using u64 = uint64_t;
using Clock = std::chrono::steady_clock;

namespace {

struct Graph {
    int n = 0, W = 0;
    std::vector<u64> bits;          // n rows of W words
    std::vector<u64> co;            // complement rows, self bit clear
    std::vector<int> deg;

    const u64 *row(int v) const { return &bits[(size_t)v * W]; }
    const u64 *corow(int v) const { return &co[(size_t)v * W]; }
    bool adj(int u, int v) const { return (bits[(size_t)u * W + (v >> 6)] >> (v & 63)) & 1ULL; }
};

template <typename F>
inline void for_bits(const u64 *r, int W, F f) {
    for (int w = 0; w < W; ++w) {
        u64 m = r[w];
        while (m) {
            int b = __builtin_ctzll(m);
            m &= m - 1;
            f((w << 6) | b);
        }
    }
}

template <typename F>
inline void for_neighbours(const Graph &g, int v, F f) { for_bits(g.row(v), g.W, f); }

template <typename F>
inline void for_non_neighbours(const Graph &g, int v, F f) { for_bits(g.corow(v), g.W, f); }

// Swap-to-end indexed set: O(1) insert, O(1) erase, contiguous for scanning.
struct IndexSet {
    std::vector<int> items, pos;
    void init(int n) { items.clear(); pos.assign(n, -1); }
    bool has(int v) const { return pos[v] >= 0; }
    void insert(int v) {
        if (pos[v] >= 0) return;
        pos[v] = (int)items.size();
        items.push_back(v);
    }
    void erase(int v) {
        int p = pos[v];
        if (p < 0) return;
        int last = items.back();
        items[p] = last;
        pos[last] = p;
        items.pop_back();
        pos[v] = -1;
    }
    size_t size() const { return items.size(); }
};

// |N(v) cap S| for a set S held as a bitset.
inline int overlap(const Graph &g, int v, const std::vector<u64> &S) {
    const u64 *r = g.row(v);
    int c = 0;
    for (int w = 0; w < g.W; ++w) c += __builtin_popcountll(r[w] & S[w]);
    return c;
}

// The incumbent shared by the whole thread portfolio.
struct Shared {
    std::atomic<int> best{0};
    std::atomic<bool> stop{false};
    std::mutex mu;
    std::vector<int> clique;
};

struct Params {
    int bms_k = 64;             // BMS sample size when picking a move
    int pen_mode = 0;           // 1 = DLS-MC penalty selection instead of BMS overlap
    int pen_delay = 2;          // decrement all non-zero penalties every this many
                                // plateau ends -- DLS-MC's `pd`, the forgetting rate
    int pen_spike = 0;          // extra penalty on the vertices of an EMITTED clique
    int rw_mode = 0;            // 1 = per-chain RANDOM vertex weights steer the harvest
    int restart_steps = 4000;   // non-improving steps before a perturbation
    int perturb_keep = 4;       // vertices carried over from the incumbent
};

// One independent local search. Threads run these as a portfolio and publish
// into the shared incumbent.
struct Search {
    const Graph &g;
    Params p;
    std::mt19937_64 rng;

    std::vector<int> slack;     // |C| - |N(v) cap C|; 0 == addable, 1 == swappable
    IndexSet cand0, cand1;      // the slack==0 and slack==1 sets, kept incrementally
    std::vector<char> inC;
    std::vector<char> conf;     // SCC: 0 = forbidden to (re-)enter C
    std::vector<int> age;       // step at which v last moved, for tie-breaking
    std::vector<int> C;
    std::vector<int> addBuf, swapBuf;
    std::vector<u64> mask;      // scratch bitset for candidate scoring
    // ---- multi-solution collection -------------------------------------------
    // The plateau walk visits many distinct maximum cliques; keep them so one solve
    // can supply a whole fleet of miners with DISTINCT answers.
    std::vector<std::vector<int>> pool;
    std::vector<u64> pool_keys;
    std::vector<unsigned char> seen;
    u64 chash = 0;
    int pool_cap = 0;
    // DLS-MC (Pullan & Hoos, JAIR 25:159-185 2006). A persistent INTEGER penalty per
    // vertex, where selection is uniform-random among the minimum-penalty candidates
    // and nothing else. Our SCC is a single BIT used as a legality filter with the
    // choice still made by delta; a penalty is a MAGNITUDE that reshapes the objective,
    // and JAIR Figs 4/8 show it flattens the degree/membership correlation to ~0. That
    // is a different basin partition, which is what every mechanism refuted here lacked.
    std::vector<int> pen;
    long long pen_events = 0;
    // Per-chain random vertex weights. MEASURED TARGET: our submitted clique carries
    // 2.46 holders against the field's 1.85, worse on 95% of rounds (p=0.0000), and
    // v25/v26-pinned/v27-penalty are indistinguishable on this (2.456/2.466/2.469).
    // They are one search. A chain that maximises a RANDOM objective rather than the
    // shared greedy one should converge somewhere else while still landing on a
    // maximum clique, because every maximum clique is optimal under some weighting.
    std::vector<float> rw;

    static inline u64 vmix(int v) {
        u64 x = (u64)(v + 1) * 0x9e3779b97f4a7c15ULL;
        x ^= x >> 29; x *= 0xbf58476d1ce4e5b9ULL; x ^= x >> 32;
        return x;
    }
    void pool_offer() {
        if (pool_cap <= 0 || (int)pool.size() >= pool_cap) return;
        // Spiking here is the cheap add-on the JAIR paper's machinery already supports:
        // once a clique is EMITTED, push the next expansion phase out of its basin
        // rather than merely discouraging the last plateau.
        if (p.pen_mode && p.pen_spike) for (int v : C) pen[v] += p.pen_spike;
        unsigned char &flag = seen[chash & 4095];
        if (flag) {
            for (u64 k2 : pool_keys)
                if (k2 == chash) return;
        }
        flag = 1;
        pool.push_back(C);
        pool_keys.push_back(chash);
    }

    int k = 0;                  // |C|
    long long step = 0;
    long long n_add = 0, n_swap = 0, n_drop = 0, n_restart = 0;
    std::vector<int> best;

    Search(const Graph &gg, Params pp, u64 seed)
        : g(gg), p(pp), rng(seed), slack(gg.n, 0), inC(gg.n, 0), conf(gg.n, 1),
          age(gg.n, 0), mask(gg.W, 0ULL), seen(4096, 0), pen(gg.n, 0) {
        C.reserve(gg.n);
        addBuf.reserve(gg.n);
        swapBuf.reserve(gg.n);
        cand0.init(gg.n);
        cand1.init(gg.n);
        for (int v = 0; v < gg.n; ++v) cand0.insert(v);   // empty clique: all addable
    }

    // Put v in whichever bucket its slack says, or none. Called only for vertices
    // whose slack or membership actually changed.
    inline void refresh(int v) {
        int want = inC[v] ? -1 : (slack[v] == 0 ? 0 : (slack[v] == 1 ? 1 : -1));
        if (want != 0 && cand0.has(v)) cand0.erase(v);
        if (want != 1 && cand1.has(v)) cand1.erase(v);
        if (want == 0) cand0.insert(v);
        else if (want == 1) cand1.insert(v);
    }

    int rnd(int hi) { return (int)(rng() % (u64)hi); }

    // `free_neighbours` is what separates an add from a swap under SCC.
    void add(int v, bool free_neighbours) {
        inC[v] = 1;
        C.push_back(v);
        ++k;
        age[v] = (int)step;
        // slack moves only for non-neighbours (and for v itself, which is not in
        // its own complement row) — that is the whole point of this variant.
        chash ^= vmix(v);
        ++slack[v];
        for_non_neighbours(g, v, [&](int u) { ++slack[u]; refresh(u); });
        refresh(v);
        if (free_neighbours)
            for_neighbours(g, v, [&](int u) { conf[u] = 1; });
    }

    void drop(int v) {
        inC[v] = 0;
        for (size_t i = 0; i < C.size(); ++i)
            if (C[i] == v) { C[i] = C.back(); C.pop_back(); break; }
        --k;
        age[v] = (int)step;
        conf[v] = 0;
        chash ^= vmix(v);
        --slack[v];
        for_non_neighbours(g, v, [&](int u) { --slack[u]; refresh(u); });
        refresh(v);
    }

    void clear() {
        for (int v : C) inC[v] = 0;
        C.clear();
        chash = 0;
        k = 0;
        std::fill(slack.begin(), slack.end(), 0);
        cand0.init(g.n);
        cand1.init(g.n);
        for (int v = 0; v < g.n; ++v) cand0.insert(v);
    }

    void pen_init() { pen.assign(g.n, 0); }
    void rw_init(u64 sd) {
        rw.assign(g.n, 1.0f);
        u64 st = sd * 0x9e3779b97f4a7c15ULL + 12345ULL;
        for (int i = 0; i < g.n; ++i) {
            st ^= st << 13; st ^= st >> 7; st ^= st << 17;
            rw[i] = 0.5f + (float)((st >> 11) % 1000) / 1000.0f;   // U[0.5, 1.5)
        }
    }

    // Pick from `cand` by BMS: sample at most bms_k of them, keep the one with
    // the largest overlap with the candidate set itself (i.e. the choice that
    // destroys the fewest future options), oldest wins ties.
    // Penalise the working clique at a plateau end, and forget periodically.
    void pen_bump(int amount) {
        if (!p.pen_mode) return;
        for (int v : C) pen[v] += amount;
        if (++pen_events % (long long)p.pen_delay == 0)
            for (int i = 0; i < g.n; ++i)
                if (pen[i] > 0) --pen[i];
    }

    int pick(const std::vector<int> &cand) {
        int m = (int)cand.size();
        if (m == 1) return cand[0];
        if (p.pen_mode) {
            // uniform-random among minimum penalty, reservoir-sampled so ties need no
            // allocation. No degree term and no gain term -- that absence is the point.
            int tries = m <= p.bms_k ? m : p.bms_k;
            int bestv = -1, bestpen = INT32_MAX, seen = 0;
            for (int i = 0; i < tries; ++i) {
                int v = (m <= p.bms_k) ? cand[i] : cand[rnd(m)];
                int pv = pen[v];
                if (pv < bestpen) { bestpen = pv; bestv = v; seen = 1; }
                else if (pv == bestpen && rnd(++seen) == 0) bestv = v;
            }
            return bestv;
        }
        std::fill(mask.begin(), mask.end(), 0ULL);
        for (int v : cand) mask[v >> 6] |= 1ULL << (v & 63);
        int tries = m <= p.bms_k ? m : p.bms_k;
        int bestv = -1, bestscore = -1;
        for (int i = 0; i < tries; ++i) {
            int v = (m <= p.bms_k) ? cand[i] : cand[rnd(m)];
            int sc = overlap(g, v, mask);
            if (sc > bestscore || (sc == bestscore && age[v] < age[bestv])) {
                bestv = v;
                bestscore = sc;
            }
        }
        return bestv;
    }

    // The one clique member v is not adjacent to (v must have cnt[v] == k-1).
    int blocker(int v) {
        for (int u : C)
            if (!g.adj(v, u)) return u;
        return -1;
    }

    // Greedy construction: repeatedly take the candidate that keeps the most
    // candidates alive. This is the numpy baseline's rule, which is markedly
    // better than static degree, done over bitsets.
    void construct() {
        for (;;) {
            if (cand0.size() == 0) return;
            addBuf.assign(cand0.items.begin(), cand0.items.end());
            add(pick(addBuf), true);
        }
    }

    void seed_from(int v) {
        clear();
        std::fill(conf.begin(), conf.end(), 1);
        add(v, true);
    }

    // Perturbation: usually keep a few vertices of the incumbent and rebuild
    // around them; occasionally restart cold so the portfolio does not collapse
    // onto one basin.
    void perturb() {
        ++n_restart;
        if (best.empty() || (rnd(10) == 0)) {
            seed_from(rnd(g.n));
            construct();
            return;
        }
        std::vector<int> keep(best.begin(), best.end());
        int m = std::min<int>(p.perturb_keep, (int)keep.size());
        for (int i = 0; i < m; ++i) std::swap(keep[i], keep[i + rnd((int)keep.size() - i)]);
        keep.resize(m);
        clear();
        std::fill(conf.begin(), conf.end(), 1);
        for (int v : keep)
            if (slack[v] == 0) add(v, true);
        if (k == 0) seed_from(rnd(g.n));
        construct();
    }

    void run(Clock::time_point deadline, Shared &sh) {
        seed_from(rnd(g.n));
        construct();
        best = C;
        long long last_gain = 0;

        for (;;) {
            if ((step & 63) == 0) {
                if (sh.stop.load(std::memory_order_relaxed)) break;
                if (Clock::now() >= deadline) break;
            }
            ++step;

            // Checked every step, not only when the search runs out of moves.
            if (step - last_gain > p.restart_steps) {
                pen_bump(1);                 // plateau end: penalise where we sat
                perturb();
                last_gain = step;
                record(sh);
                continue;
            }

            // Both candidate sets are already maintained; only the configuration
            // filter is applied here, over sets that are small (tens of vertices).
            addBuf.clear();
            for (int v : cand0.items)
                if (conf[v]) addBuf.push_back(v);
            swapBuf.clear();
            if (addBuf.empty())
                for (int v : cand1.items)
                    if (conf[v]) swapBuf.push_back(v);

            if (!addBuf.empty()) {
                ++n_add;
                add(pick(addBuf), true);
                if (k > (int)best.size()) {
                    best = C;
                    last_gain = step;
                    record(sh);
                }
                continue;
            }
            if (!swapBuf.empty()) {
                ++n_swap;
                int v = pick(swapBuf);
                int u = blocker(v);
                if (u >= 0) {
                    drop(u);
                    add(v, false);          // SCC: a swap does not free neighbours
                }
                continue;
            }
            if (k > 0) { ++n_drop; drop(C[rnd(k)]); }
            else seed_from(rnd(g.n));
        }
        construct();                        // never return an extendable clique
        if (k > (int)best.size()) best = C;
    }

    // Uniformly random plateau walk at the current size. No scoring, no conf filter:
    // the point is to stop being pulled toward the basins everyone else falls into.
    int target_size = 0;
    void plateau_walk(int steps, Clock::time_point deadline) {
        for (int i = 0; i < steps; ++i) {
            if ((i & 63) == 0 && Clock::now() >= deadline) return;
            if (cand0.size() > 0) {                 // a free improvement, always take it
                add(cand0.items[rnd((int)cand0.size())], true);
                continue;
            }
            if (cand1.size() == 0) return;          // plateau has no lateral move
            // The POOL comes from this walk, not from run(), so a penalty that only
            // steers run() cannot change pool composition -- which is why SN83_SPIKE
            // was a measured no-op before this. Steer the lateral move too: least
            // penalised first, uniform among ties, reservoir-sampled.
            int v;
            if (p.rw_mode && !rw.empty()) {
                // pick the highest-weight candidate; ties by reservoir. Each chain has
                // its own weighting, so chains converge to different maximum cliques.
                float bw = -1.0f; int seen2 = 0; v = cand1.items[0];
                for (int j = 0; j < (int)cand1.size(); ++j) {
                    int w = cand1.items[j];
                    if (rw[w] > bw) { bw = rw[w]; v = w; seen2 = 1; }
                    else if (rw[w] == bw && rnd(++seen2) == 0) v = w;
                }
            } else if (p.pen_mode) {
                int bestp = INT32_MAX, seen2 = 0; v = cand1.items[0];
                for (int j = 0; j < (int)cand1.size(); ++j) {
                    int w = cand1.items[j];
                    if (pen[w] < bestp) { bestp = pen[w]; v = w; seen2 = 1; }
                    else if (pen[w] == bestp && rnd(++seen2) == 0) v = w;
                }
            } else {
                v = cand1.items[rnd((int)cand1.size())];
            }
            int u = blocker(v);
            if (u < 0) return;
            drop(u);
            add(v, true);
            if (pool_cap > 0 && k == target_size) pool_offer();
        }
    }

    void record(Shared &sh) {
        if ((int)best.size() <= sh.best.load(std::memory_order_relaxed)) return;
        std::lock_guard<std::mutex> lock(sh.mu);       // taken only on improvement
        if ((int)best.size() > sh.best.load(std::memory_order_relaxed)) {
            sh.clique = best;
            sh.best.store((int)best.size(), std::memory_order_relaxed);
        }
    }
};

}  // namespace

extern "C" {

// adj: n*n row-major uint8, symmetric, zero diagonal.
// out: caller-allocated int32[n]; returns the clique size written.
int sn83_solve(const uint8_t *adj, int n, double time_limit, uint64_t seed,
               int n_threads, int bms_k, int restart_steps, int32_t *out) {
    Clock::time_point deadline =
        Clock::now() + std::chrono::duration_cast<Clock::duration>(
                           std::chrono::duration<double>(time_limit));
    if (n <= 0) return 0;

    Graph g;
    g.n = n;
    g.W = (n + 63) / 64;
    g.bits.assign((size_t)n * g.W, 0ULL);
    g.co.assign((size_t)n * g.W, 0ULL);
    g.deg.assign(n, 0);
    for (int i = 0; i < n; ++i) {
        const uint8_t *r = adj + (size_t)i * n;
        int d = 0;
        for (int j = 0; j < n; ++j) {
            if (j == i) continue;
            if (r[j]) { g.bits[(size_t)i * g.W + (j >> 6)] |= 1ULL << (j & 63); ++d; }
            else { g.co[(size_t)i * g.W + (j >> 6)] |= 1ULL << (j & 63); }
        }
        g.deg[i] = d;
    }

    // Params come from the ABI args when set, else from the environment. The env
    // path is what lets the autoresearch harness sweep a parameter without a
    // recompile — one variant binary, many configurations.
    Params p;
    if (const char *e = getenv("SN83_BMS_K")) p.bms_k = atoi(e);
    if (const char *e = getenv("SN83_PEN")) p.pen_mode = atoi(e);
    if (const char *e = getenv("SN83_PD")) p.pen_delay = std::max(1, atoi(e));
    if (const char *e = getenv("SN83_SPIKE")) p.pen_spike = atoi(e);
    if (const char *e = getenv("SN83_RW")) p.rw_mode = atoi(e);
    if (const char *e = getenv("SN83_RESTART")) p.restart_steps = atoi(e);
    if (const char *e = getenv("SN83_KEEP")) p.perturb_keep = atoi(e);
    if (bms_k > 0) p.bms_k = bms_k;
    if (restart_steps > 0) p.restart_steps = restart_steps;

    if (n_threads < 1) n_threads = 1;
    // Defaults are the measured optimum of the dose-response sweep on recent_val(250):
    //   0%  (control)  reward 2.4294  diversity 0.5780  parity 99.60%
    //   10%            reward 2.4341  diversity 0.5826  parity 99.60%
    //   25%            reward 2.4458  diversity 0.5952  parity 99.20%   <- peak
    //   40%            reward 2.4413  diversity 0.5906  parity 99.20%
    // 25% is the only arm significant against its own control (11 better / 2 worse,
    // sign test p = 0.0225). SN83_WALK=0 restores the previous solver exactly.
    int walk = 100000;
    if (const char *e = getenv("SN83_WALK")) walk = atoi(e);
    int walkpct = 25;
    if (const char *e = getenv("SN83_WALKPCT")) walkpct = atoi(e);
    Shared sh;
    std::vector<std::thread> pool;
    std::vector<std::vector<int>> locals(n_threads);
    std::vector<std::array<long long, 5>> stats(n_threads, {0, 0, 0, 0, 0});

    for (int t = 0; t < n_threads; ++t) {
        pool.emplace_back([&, t] {
            Search s(g, p, seed + 0x9e3779b97f4a7c15ULL * (u64)(t + 1));
            // Reserve the tail of the budget for the walk, otherwise it never runs.
            Clock::time_point split = deadline;
            if (walk > 0 && walkpct > 0) {
                auto now = Clock::now();
                split = now + ((deadline - now) * (100 - walkpct)) / 100;
            }
            s.run(split, sh);
            if (walk > 0 && !s.best.empty()) {
                // restart the state at `best`, then diffuse across the plateau
                s.clear();
                std::fill(s.conf.begin(), s.conf.end(), 1);
                for (int v : s.best)
                    if (s.slack[v] == 0) s.add(v, true);
                if ((int)s.C.size() == (int)s.best.size()) {
                    s.plateau_walk(walk, deadline);
                    if (s.C.size() >= s.best.size()) s.best = s.C;
                }
            }
            locals[t] = s.best;
            stats[t] = {s.step, s.n_add, s.n_swap, s.n_drop, s.n_restart};
        });
    }
    for (auto &th : pool) th.join();
    if (getenv("SN83_DEBUG")) {
        for (int t = 0; t < n_threads; ++t)
            fprintf(stderr, "  thread %d: steps=%lld add=%lld swap=%lld drop=%lld "
                            "perturb=%lld best=%d\n", t, stats[t][0], stats[t][1],
                    stats[t][2], stats[t][3], stats[t][4], (int)locals[t].size());
    }

    std::vector<int> best = sh.clique;
    for (auto &l : locals)
        if (l.size() > best.size()) best = l;

    // Final maximality pass on the winner: the incumbent may have been recorded
    // mid-expansion, and a non-maximal answer scores zero.
    std::vector<char> in(n, 0);
    std::vector<int> cnt(n, 0);
    for (int v : best) {
        in[v] = 1;
        for_neighbours(g, v, [&](int u) { ++cnt[u]; });
    }
    for (;;) {
        int pickv = -1;
        for (int v = 0; v < n; ++v)
            if (!in[v] && cnt[v] == (int)best.size() &&
                (pickv < 0 || g.deg[v] > g.deg[pickv]))
                pickv = v;
        if (pickv < 0) break;
        in[pickv] = 1;
        best.push_back(pickv);
        for_neighbours(g, pickv, [&](int u) { ++cnt[u]; });
    }

    for (size_t i = 0; i < best.size(); ++i) out[i] = best[i];
    return (int)best.size();
}


int sn83_solve_many(const uint8_t *adj, int n, double time_limit, uint64_t seed,
                    int n_threads, int want, int32_t *out, int32_t *sizes) {
    Clock::time_point deadline =
        Clock::now() + std::chrono::duration_cast<Clock::duration>(
                           std::chrono::duration<double>(time_limit));
    if (n <= 0 || want <= 0) return 0;

    Graph g;
    g.n = n;
    g.W = (n + 63) / 64;
    g.bits.assign((size_t)n * g.W, 0ULL);
    g.co.assign((size_t)n * g.W, 0ULL);
    g.deg.assign(n, 0);
    for (int i = 0; i < n; ++i) {
        const uint8_t *r = adj + (size_t)i * n;
        int d = 0;
        for (int j = 0; j < n; ++j) {
            if (j == i) continue;
            if (r[j]) { g.bits[(size_t)i * g.W + (j >> 6)] |= 1ULL << (j & 63); ++d; }
            else { g.co[(size_t)i * g.W + (j >> 6)] |= 1ULL << (j & 63); }
        }
        g.deg[i] = d;
    }

    Params p;
    if (const char *e = getenv("SN83_BMS_K")) p.bms_k = atoi(e);
    if (const char *e = getenv("SN83_PEN")) p.pen_mode = atoi(e);
    if (const char *e = getenv("SN83_PD")) p.pen_delay = std::max(1, atoi(e));
    if (const char *e = getenv("SN83_SPIKE")) p.pen_spike = atoi(e);
    if (const char *e = getenv("SN83_RW")) p.rw_mode = atoi(e);
    if (const char *e = getenv("SN83_RESTART")) p.restart_steps = atoi(e);
    if (n_threads < 1) n_threads = 1;
    // Split the budget: find the best size first, then spend the rest harvesting
    // distinct cliques of that size off the plateau.
    int harvestpct = 50;
    if (const char *e = getenv("SN83_HARVESTPCT")) harvestpct = atoi(e);

    Shared sh;
    std::vector<std::thread> pool;
    std::vector<std::vector<std::vector<int>>> found(n_threads);
    std::vector<std::vector<int>> locals(n_threads);
    Clock::time_point now0 = Clock::now();
    Clock::time_point split = now0 + ((deadline - now0) * (100 - harvestpct)) / 100;

    for (int t = 0; t < n_threads; ++t) {
        pool.emplace_back([&, t] {
            Search s(g, p, seed + 0x9e3779b97f4a7c15ULL * (u64)(t + 1));
            s.run(split, sh);
            locals[t] = s.best;
        });
    }
    for (auto &th : pool) th.join();

    int best_size = (int)sh.clique.size();
    for (auto &l : locals) best_size = std::max(best_size, (int)l.size());
    if (best_size <= 0) return 0;

    // Harvest phase: every thread walks the plateau at best_size, collecting
    // distinct cliques into its own pool.
    pool.clear();
    int per_thread = (want + n_threads - 1) / n_threads + 2;
    for (int t = 0; t < n_threads; ++t) {
        pool.emplace_back([&, t] {
            Search s(g, p, seed + 0x51ed270b7f4a7c15ULL * (u64)(t + 7));
            if (p.rw_mode) s.rw_init(seed + 0x2545F4914F6CDD1DULL * (u64)(t + 3));
            s.pool_cap = per_thread;
            s.target_size = best_size;
            std::vector<int> seedc = locals[t].size() == (size_t)best_size
                                     ? locals[t] : sh.clique;
            if ((int)seedc.size() != best_size) return;
            s.clear();
            std::fill(s.conf.begin(), s.conf.end(), 1);
            for (int v : seedc)
                if (s.slack[v] == 0) s.add(v, true);
            if ((int)s.C.size() != best_size) return;
            s.pool_offer();
            s.plateau_walk(1 << 30, deadline);
            found[t].swap(s.pool);
        });
    }
    for (auto &th : pool) th.join();

    // Merge, deduplicate, and keep only cliques that are genuinely maximal.
    std::vector<std::vector<int>> uniq;
    std::vector<u64> keys;
    for (auto &fv : found) {
        for (auto &c : fv) {
            if ((int)c.size() != best_size) continue;
            u64 h = 0;
            for (int v : c) h ^= Search::vmix(v);
            bool dup = false;
            for (u64 k2 : keys) if (k2 == h) { dup = true; break; }
            if (dup) continue;
            std::vector<char> in(n, 0);
            std::vector<int> cnt(n, 0);
            for (int v : c) { in[v] = 1; for_neighbours(g, v, [&](int u) { ++cnt[u]; }); }
            bool maximal = true;
            for (int v = 0; v < n && maximal; ++v)
                if (!in[v] && cnt[v] == best_size) maximal = false;
            if (!maximal) continue;
            keys.push_back(h);
            uniq.push_back(c);
            if ((int)uniq.size() >= want) break;
        }
        if ((int)uniq.size() >= want) break;
    }
    if (uniq.empty()) {
        std::vector<int> b = sh.clique;
        for (auto &l : locals) if (l.size() > b.size()) b = l;
        if (b.empty()) return 0;
        uniq.push_back(b);
    }
    int w = 0, off = 0;
    for (auto &c : uniq) {
        if (w >= want) break;
        for (int v : c) out[off++] = v;
        sizes[w++] = (int32_t)c.size();
    }
    return w;
}

}  // extern "C"
