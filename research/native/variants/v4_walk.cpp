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
    std::vector<int> deg;

    const u64 *row(int v) const { return &bits[(size_t)v * W]; }
    bool adj(int u, int v) const { return (bits[(size_t)u * W + (v >> 6)] >> (v & 63)) & 1ULL; }
};

template <typename F>
inline void for_neighbours(const Graph &g, int v, F f) {
    const u64 *r = g.row(v);
    for (int w = 0; w < g.W; ++w) {
        u64 m = r[w];
        while (m) {
            int b = __builtin_ctzll(m);
            m &= m - 1;
            f((w << 6) | b);
        }
    }
}

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
    int restart_steps = 4000;   // non-improving steps before a perturbation
    int perturb_keep = 4;       // vertices carried over from the incumbent
};

// One independent local search. Threads run these as a portfolio and publish
// into the shared incumbent.
struct Search {
    const Graph &g;
    Params p;
    std::mt19937_64 rng;

    std::vector<int> cnt;       // |N(v) cap C| for every v
    std::vector<char> inC;
    std::vector<char> conf;     // SCC: 0 = forbidden to (re-)enter C
    std::vector<int> age;       // step at which v last moved, for tie-breaking
    std::vector<int> C;
    std::vector<int> addBuf, swapBuf;
    std::vector<u64> mask;      // scratch bitset for candidate scoring

    int k = 0;                  // |C|
    long long step = 0;
    long long n_add = 0, n_swap = 0, n_drop = 0, n_restart = 0;
    std::vector<int> best;

    Search(const Graph &gg, Params pp, u64 seed)
        : g(gg), p(pp), rng(seed), cnt(gg.n, 0), inC(gg.n, 0), conf(gg.n, 1),
          age(gg.n, 0), mask(gg.W, 0ULL) {
        C.reserve(gg.n);
        addBuf.reserve(gg.n);
        swapBuf.reserve(gg.n);
    }

    int rnd(int hi) { return (int)(rng() % (u64)hi); }

    // `free_neighbours` is what separates an add from a swap under SCC.
    void add(int v, bool free_neighbours) {
        inC[v] = 1;
        C.push_back(v);
        ++k;
        age[v] = (int)step;
        if (free_neighbours) {
            for_neighbours(g, v, [&](int u) { ++cnt[u]; conf[u] = 1; });
        } else {
            for_neighbours(g, v, [&](int u) { ++cnt[u]; });
        }
    }

    void drop(int v) {
        inC[v] = 0;
        for (size_t i = 0; i < C.size(); ++i)
            if (C[i] == v) { C[i] = C.back(); C.pop_back(); break; }
        --k;
        age[v] = (int)step;
        conf[v] = 0;
        for_neighbours(g, v, [&](int u) { --cnt[u]; });
    }

    void clear() {
        for (int v : C) inC[v] = 0;
        C.clear();
        k = 0;
        std::fill(cnt.begin(), cnt.end(), 0);
    }

    // Pick from `cand` by BMS: sample at most bms_k of them, keep the one with
    // the largest overlap with the candidate set itself (i.e. the choice that
    // destroys the fewest future options), oldest wins ties.
    int pick(const std::vector<int> &cand) {
        int m = (int)cand.size();
        if (m == 1) return cand[0];
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
            addBuf.clear();
            for (int v = 0; v < g.n; ++v)
                if (!inC[v] && cnt[v] == k) addBuf.push_back(v);
            if (addBuf.empty()) return;
            add(pick(addBuf), true);
        }
    }

    void seed_from(int v) {
        clear();
        std::fill(conf.begin(), conf.end(), 1);
        add(v, true);
    }

    // WALK perturbation (this variant's one change): pick a random vertex v, force
    // it in, and keep only the current clique members adjacent to it.
    //
    // This is the rule DLS-MC's own tuning converges to on dense random graphs. Its
    // penalty-delay parameter pd is tuned to 1 for EVERY C*.9 / gen400* / p_hat*-3
    // instance, and pd=1 means penalties are incremented then immediately decayed,
    // i.e. switched off — leaving exactly "plateau search + this perturbation"
    // (https://arxiv.org/pdf/1109.5717, Tables 1-2). SCCWalk arrives at the same
    // move independently. The champion instead keeps k random vertices of the
    // incumbent, which is a structured-graph heuristic.
    void perturb() {
        ++n_restart;
        int v = rnd(g.n);
        std::vector<int> keep;
        keep.reserve(C.size());
        for (int u : C)
            if (g.adj(u, v)) keep.push_back(u);
        clear();
        std::fill(conf.begin(), conf.end(), 1);
        add(v, true);
        for (int u : keep)
            if (cnt[u] == k) add(u, true);
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
                perturb();
                last_gain = step;
                record(sh);
                continue;
            }

            addBuf.clear();
            swapBuf.clear();
            for (int v = 0; v < g.n; ++v) {
                if (inC[v] || !conf[v]) continue;
                if (cnt[v] == k) addBuf.push_back(v);
                else if (cnt[v] == k - 1) swapBuf.push_back(v);
            }

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
    g.deg.assign(n, 0);
    for (int i = 0; i < n; ++i) {
        const uint8_t *r = adj + (size_t)i * n;
        int d = 0;
        for (int j = 0; j < n; ++j)
            if (r[j] && j != i) { g.bits[(size_t)i * g.W + (j >> 6)] |= 1ULL << (j & 63); ++d; }
        g.deg[i] = d;
    }

    // Params come from the ABI args when set, else from the environment. The env
    // path is what lets the autoresearch harness sweep a parameter without a
    // recompile — one variant binary, many configurations.
    Params p;
    if (const char *e = getenv("SN83_BMS_K")) p.bms_k = atoi(e);
    if (const char *e = getenv("SN83_RESTART")) p.restart_steps = atoi(e);
    if (const char *e = getenv("SN83_KEEP")) p.perturb_keep = atoi(e);
    if (bms_k > 0) p.bms_k = bms_k;
    if (restart_steps > 0) p.restart_steps = restart_steps;

    if (n_threads < 1) n_threads = 1;
    Shared sh;
    std::vector<std::thread> pool;
    std::vector<std::vector<int>> locals(n_threads);
    std::vector<std::array<long long, 5>> stats(n_threads, {0, 0, 0, 0, 0});

    for (int t = 0; t < n_threads; ++t) {
        pool.emplace_back([&, t] {
            Search s(g, p, seed + 0x9e3779b97f4a7c15ULL * (u64)(t + 1));
            s.run(deadline, sh);
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

}  // extern "C"
