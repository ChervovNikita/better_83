// v11 = v7's incremental candidate sets + an AMTS k-fixed tabu contingent.
//
// Two levers stacked, each confirmed separately:
//   * v7 buys 2-4x more search steps per second by tracking slack over the
//     complement, so candidate sets are maintained instead of rescanned.
//   * AMTS searches a different space entirely (fix |S|=k, minimise missing edges)
//     and is documented to reach sizes the grow-and-plateau family never reaches
//     on the same instances.
// Since the misses are near-misses, the pool should differ in WHERE it looks, not
// just in its random seed. SN83_AMTS_FRAC (in eighths) sets the split.
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
    int restart_steps = 4000;   // non-improving steps before a perturbation
    int perturb_keep = 4;       // vertices carried over from the incumbent
};

// One independent local search. Threads run these as a portfolio and publish
// into the shared incumbent.
struct Amts {
    const Graph &g;
    std::mt19937_64 rng;

    std::vector<int> S;          // the current fixed-size set
    std::vector<char> inS;
    std::vector<int> bad;        // |nonN(v) ∩ S| — conflicts with S
    std::vector<long long> tabu; // iteration until which a vertex is tabu
    std::vector<int> freq;       // long-term move frequency, seeds restarts
    long long iter = 0;
    int k = 0;                   // target size
    int f_bad = 0;               // missing edges inside S; 0 == S is a clique
    std::vector<int> best;       // best clique found by this thread

    Amts(const Graph &gg, u64 seed)
        : g(gg), rng(seed), inS(gg.n, 0), bad(gg.n, 0), tabu(gg.n, 0), freq(gg.n, 0) {
        S.reserve(gg.n);
    }

    int rnd(int hi) { return hi <= 0 ? 0 : (int)(rng() % (u64)hi); }

    void add(int v) {
        inS[v] = 1;
        S.push_back(v);
        f_bad += bad[v];
        for_bits(g.corow(v), g.W, [&](int u) { ++bad[u]; });
        ++freq[v];
    }

    void remove_at(size_t i) {
        int v = S[i];
        S[i] = S.back();
        S.pop_back();
        inS[v] = 0;
        for_bits(g.corow(v), g.W, [&](int u) { --bad[u]; });
        f_bad -= bad[v];
        ++freq[v];
    }

    void clear() {
        for (int v : S) inS[v] = 0;
        S.clear();
        std::fill(bad.begin(), bad.end(), 0);
        f_bad = 0;
    }

    // Build a size-k start. `seed` is carried over from the clique just found, so
    // climbing from k to k+1 keeps the k vertices already known to work instead of
    // rebuilding from scratch — that was worth several vertices in testing.
    void init_k(int kk, const std::vector<int> *seed = nullptr) {
        clear();
        k = kk;
        if (seed && !seed->empty()) {
            for (int v : *seed) {
                if ((int)S.size() >= k) break;
                add(v);
            }
        } else {
            int seed_v = 0;
            for (int v = 1; v < g.n; ++v)
                if (freq[v] < freq[seed_v] || (freq[v] == freq[seed_v] && rnd(4) == 0)) seed_v = v;
            add(seed_v);
        }
        while ((int)S.size() < k) {
            int bestv = -1, bestbad = 1 << 30;
            for (int t = 0; t < 64; ++t) {           // BMS sample, not a full scan
                int v = rnd(g.n);
                if (inS[v]) continue;
                if (bad[v] < bestbad) { bestbad = bad[v]; bestv = v; }
            }
            if (bestv < 0) {
                for (int v = 0; v < g.n && bestv < 0; ++v)
                    if (!inS[v]) bestv = v;
            }
            add(bestv);
        }
        std::fill(tabu.begin(), tabu.end(), 0);
    }

    // One constrained swap: worst-connected member out, best-connected outsider in.
    // Ties broken at random; a tabu move is allowed only if it would beat the best.
    void step() {
        ++iter;
        int u = -1, ubad = -1, uties = 0;
        for (size_t i = 0; i < S.size(); ++i) {
            int v = S[i];
            if (bad[v] == 0) continue;               // only conflicted members can help
            bool ok = tabu[v] <= iter;
            if (!ok) continue;
            if (bad[v] > ubad) { ubad = bad[v]; u = (int)i; uties = 1; }
            else if (bad[v] == ubad && rnd(++uties) == 0) u = (int)i;
        }
        if (u < 0) {                                  // everything tabu: force one out
            for (size_t i = 0; i < S.size(); ++i)
                if (bad[S[i]] > 0) { u = (int)i; break; }
            if (u < 0) return;
        }
        int uv = S[u];

        int v = -1, vbad = 1 << 30, vties = 0;
        for (int w = 0; w < g.n; ++w) {
            if (inS[w]) continue;
            if (tabu[w] > iter && bad[w] >= vbad) continue;
            int b = bad[w] - (g.adj(w, uv) ? 0 : 1);  // uv is about to leave
            if (b < vbad) { vbad = b; v = w; vties = 1; }
            else if (b == vbad && rnd(++vties) == 0) v = w;
        }
        if (v < 0) return;

        remove_at(u);
        add(v);
        // Tenures from the paper: proportional to how far S is from being a clique.
        int l = std::min(f_bad, 10);
        int C = std::max(k / 40, 6);
        tabu[uv] = iter + l + rnd(C);
        tabu[v] = iter + (int)(0.6 * l) + rnd(std::max(1, (int)(0.6 * C)));
    }

    void record(Shared &sh) {
        if ((int)best.size() <= sh.best.load(std::memory_order_relaxed)) return;
        std::lock_guard<std::mutex> lock(sh.mu);
        if ((int)best.size() > sh.best.load(std::memory_order_relaxed)) {
            sh.clique = best;
            sh.best.store((int)best.size(), std::memory_order_relaxed);
        }
    }

    // Greedy maximal clique, used to get the first k and to extend any k-clique found.
    std::vector<int> greedy(std::vector<int> seedset) {
        std::vector<char> in(g.n, 0);
        std::vector<int> cnt(g.n, 0), C;
        for (int v : seedset) {
            in[v] = 1;
            C.push_back(v);
            for_bits(g.row(v), g.W, [&](int u) { ++cnt[u]; });
        }
        for (;;) {
            int pick = -1, sc = -1;
            for (int v = 0; v < g.n; ++v)
                if (!in[v] && cnt[v] == (int)C.size() && g.deg[v] > sc) { sc = g.deg[v]; pick = v; }
            if (pick < 0) break;
            in[pick] = 1;
            C.push_back(pick);
            for_bits(g.row(pick), g.W, [&](int u) { ++cnt[u]; });
        }
        return C;
    }

    void run(Clock::time_point deadline, Shared &sh) {
        best = greedy({rnd(g.n)});
        record(sh);
        int target = (int)best.size() + 1;
        long long L = 0, since = 0;
        init_k(target, &best);
        L = (long long)g.n * target;                  // paper: L = |V|*k on random graphs
        int f_local = f_bad;

        for (;;) {
            if ((iter & 255) == 0) {
                if (sh.stop.load(std::memory_order_relaxed)) break;
                if (Clock::now() >= deadline) break;
            }
            step();
            if (f_bad == 0) {                          // S is a clique of size k
                std::vector<int> ext = greedy(S);
                if (ext.size() > best.size()) {
                    best = ext;
                    record(sh);
                }
                target = (int)best.size() + 1;
                init_k(target, &best);          // climb from what we just proved works
                L = (long long)g.n * target;
                f_local = f_bad;
                since = 0;
                continue;
            }
            if (f_bad < f_local) { f_local = f_bad; since = 0; }
            else if (++since > L) {                    // stagnation: fresh start at same k
                int shared = sh.best.load(std::memory_order_relaxed);
                target = std::max(target, shared + 1);
                init_k(target);                 // stagnated: cold start at this k
                L = (long long)g.n * target;
                f_local = f_bad;
                since = 0;
            }
        }
    }
};


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

    int k = 0;                  // |C|
    long long step = 0;
    long long n_add = 0, n_swap = 0, n_drop = 0, n_restart = 0;
    std::vector<int> best;

    Search(const Graph &gg, Params pp, u64 seed)
        : g(gg), p(pp), rng(seed), slack(gg.n, 0), inC(gg.n, 0), conf(gg.n, 1),
          age(gg.n, 0), mask(gg.W, 0ULL) {
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
        --slack[v];
        for_non_neighbours(g, v, [&](int u) { --slack[u]; refresh(u); });
        refresh(v);
    }

    void clear() {
        for (int v : C) inC[v] = 0;
        C.clear();
        k = 0;
        std::fill(slack.begin(), slack.end(), 0);
        cand0.init(g.n);
        cand1.init(g.n);
        for (int v = 0; v < g.n; ++v) cand0.insert(v);
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
    if (const char *e = getenv("SN83_RESTART")) p.restart_steps = atoi(e);
    if (const char *e = getenv("SN83_KEEP")) p.perturb_keep = atoi(e);
    if (bms_k > 0) p.bms_k = bms_k;
    if (restart_steps > 0) p.restart_steps = restart_steps;

    if (n_threads < 1) n_threads = 1;
    int frac = 4;                                  // eighths of the pool on AMTS
    if (const char *e = getenv("SN83_AMTS_FRAC")) frac = atoi(e);
    int n_amts = (n_threads * frac) / 8;
    if (n_threads > 1) n_amts = std::min(std::max(n_amts, 0), n_threads - 1);
    else n_amts = 0;
    Shared sh;
    std::vector<std::thread> pool;
    std::vector<std::vector<int>> locals(n_threads);
    std::vector<std::array<long long, 5>> stats(n_threads, {0, 0, 0, 0, 0});

    for (int t = 0; t < n_threads; ++t) {
        pool.emplace_back([&, t] {
            u64 sd = seed + 0x9e3779b97f4a7c15ULL * (u64)(t + 1);
            if (t < n_amts) {
                Amts a(g, sd);
                a.run(deadline, sh);
                locals[t] = a.best;
                stats[t] = {a.iter, 0, 0, 0, 0};
            } else {
                Search s(g, p, sd);
                s.run(deadline, sh);
                locals[t] = s.best;
                stats[t] = {s.step, s.n_add, s.n_swap, s.n_drop, s.n_restart};
            }
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
