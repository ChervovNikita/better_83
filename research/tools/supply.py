"""How many maximum cliques actually EXIST? An ORACLE measurement, not a mechanism.

The decisive open question: is the remaining -0.170 closable by better reach, or is the
optimum supply nearly exhausted so the novel cliques simply are not there?

Method: enumerate exactly (MoMC) inside the UNION hull -- the vertices spanned by our
max cliques AND the field's submitted max cliques. Any maximum clique living in the
combined region is found. This uses field answers, so it is NOT deployable; it exists
only to bound supply from below far more tightly than our pool alone can.

Read the result as: supply >= what this finds. If it returns roughly the union we
already observe (31.3/round), supply is close to exhausted and the gap is structural.
If it returns much more, reach is worth continuing to attack.
"""
import collections, json, os, sys, time
import numpy as np
sys.path.insert(0, '/workspace/better_83')
sys.path.insert(0, '/workspace/better_83/research')
sys.path.insert(0, '/home/dev/.claude/jobs/5443dea7/tmp')
from CliqueAI.graph.codec import GraphCodec
from hull import write_dimacs, run_momc

if __name__ == '__main__':
    target = int(sys.argv[1]); nrounds = int(sys.argv[2]); tmo = float(sys.argv[3])
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
        H = set()
        for c in list(L) + list(FM):
            H.update(c)
        if len(H) < target:
            score = M[:, sorted(H)].sum(axis=1)
            for v in np.argsort(-score):
                if len(H) >= target:
                    break
                H.add(int(v))
        H = sorted(H)
        p = '/home/dev/.claude/jobs/5443dea7/tmp/sup_%d.col' % os.getpid()
        write_dimacs(M, H, p)
        t0 = time.time()
        cl, timedout = run_momc(p, 2000, tmo)
        dt = time.time() - t0
        if timedout:
            tos += 1
            continue
        good = set()
        for c in cl:
            if not c or max(c) > len(H):
                continue
            v = tuple(sorted(H[i - 1] for i in c))
            if len(v) == mx and all(M[a, b] for a in v for b in v if a != b):
                good.add(v)
        if not good:
            continue
        A['hull'].append(len(H)); A['t'].append(dt)
        A['found'].append(len(good))
        A['union'].append(len(L | FM))
        A['ours'].append(len(L)); A['field'].append(len(FM))
        A['unclaimed'].append(len(good - L - FM))
        # NOTE: deliberately NOT storing a per-round ratio. Averaging ratios across
        # rounds of unequal size is the single most repeated error in this project
        # (seven wrong answers). The share is computed from the TOTALS below.
    import statistics as st
    if not A['found']:
        print("no usable rounds (timeouts=%d)" % tos); sys.exit()
    print("union-hull oracle: %d rounds, %d timeouts, hull %.0f, %.2fs/round"
          % (len(A['found']), tos, st.mean(A['hull']), st.mean(A['t'])))
    print("   maxima that EXIST in the union region  %7.1f" % st.mean(A['found']))
    print("   union we already observe (ours+field)  %7.1f" % st.mean(A['union']))
    print("   our pool                               %7.1f  (%.1f%% of what exists)"
          % (st.mean(A['ours']), 100.0 * sum(A['ours']) / sum(A['found'])))
    print("   field                                  %7.1f" % st.mean(A['field']))
    print("   reached by NOBODY                      %7.1f" % st.mean(A['unclaimed']))
