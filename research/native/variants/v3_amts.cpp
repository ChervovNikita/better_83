// AMTS: adaptive multistart tabu search on the FIXED-k feasibility formulation.
//
// Why a different formulation at all. Our champion is grow-and-plateau (LSCC
// lineage): expand the clique, swap on the plateau, perturb when stuck. That
// family has a documented ceiling — on MANN_a45 (omega=345) DLS-MC reaches 344 in
// 100/100 runs and 345 in none, while AMTS reaches 345 in 4/100. Our loss profile
// is exactly that signature: 88% ties, and the rest short by exactly one vertex.
//   Wu & Hao, JOCO 2013: https://leria-info.univ-angers.fr/~jinkao.hao/papers/JOCONov2011.pdf
//
// The formulation: fix |S| = k and MAXIMISE the edges induced by S. S is a clique
// exactly when it induces k(k-1)/2 edges. Moves are constrained swaps (drop the
// worst-connected member, add the best-connected outsider) under a tabu list.
// Instead of growing a clique we ask a feasibility question, and that changes the
// landscape the search moves through.
//
// Everything is done on the COMPLEMENT graph, per the paper's own advice for
// density > 0.5. At density 0.9 a complement row holds ~90 bits instead of ~810,
// so the incremental update after a swap costs a ninth of what it would on the
// graph itself. We count MISSING edges and drive that to zero.
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
    std::vector<u64> bits;      // adjacency
    std::vector<u64> co;        // complement (no self loop)
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

struct Shared {
    std::atomic<int> best{0};
    std::atomic<bool> stop{false};
    std::mutex mu;
    std::vector<int> clique;
};

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

}  // namespace

extern "C" {

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

    if (n_threads < 1) n_threads = 1;
    Shared sh;
    std::vector<std::thread> pool;
    std::vector<std::vector<int>> locals(n_threads);
    for (int t = 0; t < n_threads; ++t) {
        pool.emplace_back([&, t] {
            Amts a(g, seed + 0x9e3779b97f4a7c15ULL * (u64)(t + 1));
            a.run(deadline, sh);
            locals[t] = a.best;
        });
    }
    for (auto &th : pool) th.join();

    std::vector<int> best = sh.clique;
    for (auto &l : locals)
        if (l.size() > best.size()) best = l;

    std::vector<char> in(n, 0);
    std::vector<int> cnt(n, 0);
    for (int v : best) {
        in[v] = 1;
        for_bits(g.row(v), g.W, [&](int u) { ++cnt[u]; });
    }
    for (;;) {
        int pick = -1;
        for (int v = 0; v < n; ++v)
            if (!in[v] && cnt[v] == (int)best.size() &&
                (pick < 0 || g.deg[v] > g.deg[pick])) pick = v;
        if (pick < 0) break;
        in[pick] = 1;
        best.push_back(pick);
        for_bits(g.row(pick), g.W, [&](int u) { ++cnt[u]; });
    }
    for (size_t i = 0; i < best.size(); ++i) out[i] = best[i];
    return (int)best.size();
}

}  // extern "C"
