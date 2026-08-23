"""H-PERMUTE gate: does relabelling the hull change WHICH cliques a truncated
enumeration returns?

Block J showed MoMC finds 108.5 maxima in our 180-hull at 20s but 30.4 at 4.78s, and
the first 30 are the contested ones -- search order starts in the attractor. We write
the DIMACS file, so we choose the labelling, and MoMC's branching follows it.

The gate: run K permutations on the SAME hull at the SAME budget and measure pairwise
Jaccard of the returned clique sets. High Jaccard (>0.8) means labelling does not steer
the search and the idea is dead before any reward measurement. Low Jaccard means the
union of K parallel short runs can cover what one long run covers -- and a single MoMC
process is single-threaded, so K of them cost one run's wall clock on K cores.
"""
import collections, itertools, json, os, subprocess, sys, time
import numpy as np
sys.path.insert(0, '/workspace/better_83')
sys.path.insert(0, '/workspace/better_83/research')
sys.path.insert(0, '/home/dev/.claude/jobs/5443dea7/tmp')
from CliqueAI.graph.codec import GraphCodec
from hull import run_momc, hull_of

MOMC = '/home/dev/.claude/jobs/5443dea7/tmp/momc'

def write_perm(M, verts, path, perm):
    """DIMACS for the induced subgraph, with vertices written in `perm` order."""
    order = [verts[i] for i in perm]
    idx = {v: i + 1 for i, v in enumerate(order)}
    edges = []
    for a in range(len(order)):
        for b in range(a + 1, len(order)):
            if M[order[a], order[b]]:
                edges.append((idx[order[a]], idx[order[b]]))
    with open(path, 'w') as f:
        f.write("p edge %d %d\n" % (len(order), len(edges)))
        for u, v in edges:
            f.write("e %d %d\n" % (u, v))
    return order

if __name__ == '__main__':
    S = int(sys.argv[1]); K = int(sys.argv[2]); budget = float(sys.argv[3])
    nrounds = int(sys.argv[4])
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
    J = []; sizes = []; unions = []; singles = []
    for r in rows:
        M = np.array(cd.decode_matrix(r['matrix_b92']), dtype=np.uint8)
        lscc = cache[r['uuid']]
        mx = max(len(c) for c in lscc)
        H = hull_of(lscc, mx, M, S)
        sets = []
        for k in range(K):
            perm = list(range(len(H))) if k == 0 else list(rng.permutation(len(H)))
            p = '/home/dev/.claude/jobs/5443dea7/tmp/pm_%d_%d.col' % (os.getpid(), k)
            order = write_perm(M, H, p, perm)
            cl, to = run_momc(p, 2000, budget)
            if to or not cl:
                continue
            out = set()
            for c in cl:
                if not c or max(c) > len(order):
                    continue
                v = tuple(sorted(order[i - 1] for i in c))
                if len(v) == mx and all(M[a, b] for a in v for b in v if a != b):
                    out.add(v)
            if out:
                sets.append(out)
        if len(sets) < 2:
            continue
        for a, b in itertools.combinations(range(len(sets)), 2):
            inter = len(sets[a] & sets[b]); uni = len(sets[a] | sets[b])
            if uni:
                J.append(inter / uni)
        u = set()
        for s_ in sets:
            u |= s_
        unions.append(len(u)); singles.append(sum(len(s_) for s_ in sets) / len(sets))
        sizes.append(len(sets))
    import statistics as st
    if not J:
        print("no usable rounds"); sys.exit()
    print("S=%d K=%d budget=%.1fs  %d rounds, %d pairs" % (S, K, budget, len(unions), len(J)))
    print("   pairwise Jaccard   mean %.3f  median %.3f  max %.3f"
          % (st.mean(J), st.median(J), max(J)))
    print("   cliques per single run   %.1f" % st.mean(singles))
    print("   union of the %d runs      %.1f  (%.2fx a single run)"
          % (K, st.mean(unions), st.mean(unions) / max(st.mean(singles), 1e-9)))
    print()
    print("   GATE: Jaccard > 0.8 kills H-PERMUTE. Observed %.3f -> %s"
          % (st.mean(J), "DEAD" if st.mean(J) > 0.8 else "PROCEED"))
