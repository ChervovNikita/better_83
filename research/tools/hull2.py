"""H-HULL, measured honestly: real deadlines, timeouts counted, pool written for scoring.

Differences from the first pass, each closing a way the first pass could flatter itself:
  * the MoMC budget is a fraction of the ROUND'S OWN time limit, not a flat cap
  * a timeout is a FALLBACK to the LSCC pool and stays in the denominator
  * the emitted pool is MoMC-first then LSCC backfill, so it is never worse than today
  * the pool is written to disk and scored through the reward reference separately
"""
import collections, json, os, subprocess, sys, time
import numpy as np
sys.path.insert(0, '/workspace/better_83')
sys.path.insert(0, '/workspace/better_83/research')
sys.path.insert(0, '/home/dev/.claude/jobs/5443dea7/tmp')
from CliqueAI.graph.codec import GraphCodec
from hull import write_dimacs, run_momc, hull_of

if __name__ == '__main__':
    target = int(sys.argv[1]); nrounds = int(sys.argv[2])
    frac = float(sys.argv[3]); out = sys.argv[4]
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
    agg = collections.defaultdict(list)
    pools = {}
    for r in rows:
        M = np.array(cd.decode_matrix(r['matrix_b92']), dtype=np.uint8)
        lscc = cache[r['uuid']]
        mx = max(len(c) for c in lscc)
        F = {tuple(sorted(map(int, a['clique'])))
             for a in r['answers'] if a.get('opt', 0) > 0}
        fmx = max([mx] + [len(c) for c in F])
        tl = float(r.get('time_limit') or 10.0)
        budget = max(0.5, tl * frac)
        H = hull_of(lscc, mx, M, target)
        p = os.path.join('/home/dev/.claude/jobs/5443dea7/tmp', 'h2_%d.col' % os.getpid())
        write_dimacs(M, H, p)
        t0 = time.time()
        cl, timedout = run_momc(p, 400, budget)
        dt = time.time() - t0
        good = []
        if not timedout and cl:
            for c in cl:
                if max(c) > len(H):
                    continue
                v = tuple(sorted(H[i - 1] for i in c))
                if len(v) != fmx:
                    continue
                if all(M[a, b] for a in v for b in v if a != b):
                    good.append(v)
        agg['timeout'].append(1 if timedout else 0)
        agg['t'].append(dt); agg['budget'].append(budget)
        seen = set(); merged = []
        for c in good + [c for c in lscc if len(c) == fmx]:
            if c not in seen:
                seen.add(c); merged.append(c)
        pools[r['uuid']] = [list(c) for c in merged]
        L = {c for c in lscc if len(c) == fmx}
        agg['momc'].append(len(good)); agg['lscc'].append(len(L))
        agg['pool'].append(len(merged))
        agg['novel_lscc'].append(len(L - F) / max(len(L), 1))
        agg['novel_pool'].append(len(set(merged) - F) / max(len(merged), 1))
        agg['novel_pre8'].append(sum(1 for c in merged[:8] if c not in F) / min(8, len(merged)))
    json.dump(pools, open(out, 'w'))
    import statistics as st
    print("hull=%-4d frac=%.2f  rounds=%d  timeouts=%d (%.0f%%)  time %.2fs / budget %.2fs"
          % (target, frac, len(rows), sum(agg['timeout']),
             100 * st.mean(agg['timeout']), st.mean(agg['t']), st.mean(agg['budget'])))
    print("   maxima: MoMC %.1f | LSCC %.1f | merged pool %.1f"
          % (st.mean(agg['momc']), st.mean(agg['lscc']), st.mean(agg['pool'])))
    print("   novel share: LSCC %.1f%% -> pool %.1f%%  (first-8 prefix %.1f%%)"
          % (100 * st.mean(agg['novel_lscc']), 100 * st.mean(agg['novel_pool']),
             100 * st.mean(agg['novel_pre8'])))
    print("   pool written to %s" % out)
