"""Are the unclaimed maxima reachable from an OUR-POOL-ONLY hull, if we make it bigger?

The oracle found 24.6 maxima/round that nobody reaches, inside the UNION hull (which
needs field data and is not deployable). H-HULL, the same enumeration inside a hull
built from our pool alone, was null -- but only ever at hull <= 130, under a budget
later shown to be too small.

For each unclaimed clique this asks the one question that decides it: are all of its
vertices inside an our-pool-only hull of size S? If yes at a feasible S, broadening our
own hull reaches it with no field data at all.
"""
import collections, json, os, sys, time
import numpy as np
sys.path.insert(0, '/workspace/better_83')
sys.path.insert(0, '/workspace/better_83/research')
sys.path.insert(0, '/home/dev/.claude/jobs/5443dea7/tmp')
from CliqueAI.graph.codec import GraphCodec
from hull import write_dimacs, run_momc, hull_of

SIZES = [130, 180, 240, 320]

if __name__ == '__main__':
    nrounds = int(sys.argv[1]); tmo = float(sys.argv[2])
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
    A = collections.defaultdict(list)
    tos = 0
    for r in rows:
        M = np.array(cd.decode_matrix(r['matrix_b92']), dtype=np.uint8)
        lscc = cache[r['uuid']]
        F = {tuple(sorted(map(int, a['clique'])))
             for a in r['answers'] if a.get('opt', 0) > 0}
        if not F:
            continue
        mx = max([len(c) for c in lscc] + [len(c) for c in F])
        L = {c for c in lscc if len(c) == mx}
        FM = {c for c in F if len(c) == mx}
        UH = set()
        for c in list(L) + list(FM):
            UH.update(c)
        UH = sorted(UH)
        p = '/home/dev/.claude/jobs/5443dea7/tmp/cv_%d.col' % os.getpid()
        write_dimacs(M, UH, p)
        cl, timedout = run_momc(p, 2000, tmo)
        if timedout:
            tos += 1
            continue
        found = set()
        for c in cl:
            if not c or max(c) > len(UH):
                continue
            v = tuple(sorted(UH[i - 1] for i in c))
            if len(v) == mx and all(M[a, b] for a in v for b in v if a != b):
                found.add(v)
        unclaimed = found - L - FM
        if not unclaimed:
            continue
        A['unclaimed'].append(len(unclaimed))
        for S in SIZES:
            H = set(hull_of(list(L), mx, M, S))
            cov = sum(1 for c in unclaimed if set(c) <= H)
            A['cov%d' % S].append(cov)
        A['nL'].append(len(L))
    import statistics as st
    if not A['unclaimed']:
        print("no rounds with unclaimed cliques (timeouts=%d)" % tos); sys.exit()
    n = len(A['unclaimed'])
    tot = sum(A['unclaimed'])
    print("%d rounds with unclaimed maxima, %d timeouts" % (n, tos))
    print("   unclaimed maxima            %.1f/round  (%d total)" % (st.mean(A['unclaimed']), tot))
    print()
    print("   our-pool-only hull size   unclaimed FULLY inside it")
    for S in SIZES:
        c = sum(A['cov%d' % S])
        print("       %4d                  %6.1f/round   %5.1f%% of them" %
              (S, st.mean(A['cov%d' % S]), 100.0 * c / tot))
    print()
    print("   (ratio of totals, not a mean of per-round ratios)")
