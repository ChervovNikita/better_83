"""H-REPLICATOR: Motzkin-Straus replicator dynamics with a RANDOMISED regulariser.

Bomze 1997 / Pelillo NeuralComp 1999 Thm 2: on the simplex with f(x) = x'(A + cI)x,
maximal cliques correspond 1-1 to strict local maximisers, for ANY c in (0,1). Pelillo
p.1940: c "only affects the basins of attraction around local optima". So randomising c
per restart repartitions the simplex into basins WITHOUT changing which points are
optima -- a region generator, which every refuted mechanism here lacked.

Batched as a GEMM over B columns so the restart count is affordable.
"""
import collections, json, os, sys, time
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import numpy as np
sys.path.insert(0, '/workspace/better_83')
sys.path.insert(0, '/workspace/better_83/research')
from CliqueAI.graph.codec import GraphCodec

def extend_maximal(M, verts, order):
    """Greedily extend a clique to maximal, trying vertices in `order`."""
    C = list(verts)
    for v in order:
        if v in C:
            continue
        if all(M[v, u] for u in C):
            C.append(v)
    return tuple(sorted(C))

def replicate(M, B, iters, rng, cmode):
    n = M.shape[0]
    A = M.astype(np.float64)
    X = rng.random((n, B)) + 1e-3
    X /= X.sum(axis=0, keepdims=True)
    if cmode == 'fixed':
        c = np.full(B, 0.4)                    # the old naive setting: ONE basin split
    else:
        c = rng.uniform(0.05, 0.95, size=B)    # randomised: a different split each time
    for _ in range(iters):
        P = A @ X + c[None, :] * X
        X *= P
        s = X.sum(axis=0, keepdims=True)
        s[s == 0] = 1.0
        X /= s
    return X

def harvest(M, X, thresh=1e-3):
    n = M.shape[0]
    deg = M.sum(axis=1)
    order = list(np.argsort(-deg))
    out = []
    for b in range(X.shape[1]):
        sup = np.flatnonzero(X[:, b] > thresh)
        if len(sup) == 0:
            continue
        # the support need not be a clique before convergence; take a greedy clique
        # inside it, ordered by the dynamics' own weights, then extend to maximal
        sup = sorted(sup, key=lambda v: -X[v, b])
        C = []
        for v in sup:
            if all(M[v, u] for u in C):
                C.append(v)
        out.append(extend_maximal(M, C, order))
    return out

if __name__ == '__main__':
    B = int(sys.argv[1]); iters = int(sys.argv[2]); nrounds = int(sys.argv[3])
    cmode = sys.argv[4]; out = sys.argv[5]
    cache = {}
    for line in open('/workspace/better_83/research/data/sim_ts.jsonl'):
        r = json.loads(line)
        cache[r['uuid']] = [tuple(sorted(map(int, c))) for c in r['cliques']]
    rows = []
    for line in open('/workspace/better_83/research/data/sim_rounds.jsonl'):
        r = json.loads(line)
        if r.get('answers') and r['uuid'] in cache and cache[r['uuid']]:
            rows.append(r)
    rows.sort(key=lambda r: r['uuid'])
    rows = rows[:nrounds]
    cd = GraphCodec()
    rng = np.random.default_rng(20260823)
    pools = {}
    agg = collections.defaultdict(list)
    for r in rows:
        M = np.array(cd.decode_matrix(r['matrix_b92']), dtype=np.uint8)
        lscc = cache[r['uuid']]
        mx = max(len(c) for c in lscc)
        t0 = time.time()
        X = replicate(M, B, iters, rng, cmode)
        cl = harvest(M, X)
        dt = time.time() - t0
        good = [c for c in set(cl)
                if len(c) == mx and all(M[a, b] for a in c for b in c if a != b)]
        L = [c for c in lscc if len(c) == mx]
        seen = set(); merged = []
        for c in good + L:
            if c not in seen:
                seen.add(c); merged.append(c)
        pools[r['uuid']] = [list(c) for c in merged]
        agg['t'].append(dt); agg['rep'].append(len(good)); agg['lscc'].append(len(L))
        agg['repnew'].append(len(set(good) - set(L)))
        agg['best'].append(max((len(c) for c in set(cl)), default=0) - mx)
    json.dump(pools, open(out, 'w'))
    import statistics as st
    print("B=%d iters=%d c=%s rounds=%d  %.2fs/round"
          % (B, iters, cmode, len(rows), st.mean(agg['t'])))
    print("   replicator maxima %.1f | LSCC %.1f | replicator-only %.1f"
          % (st.mean(agg['rep']), st.mean(agg['lscc']), st.mean(agg['repnew'])))
    print("   best-size vs omega: mean %+.2f (0 = matched omega)" % st.mean(agg['best']))
    print("   pool -> %s" % out)
