"""H-REGION: the hull's PADDING RULE decides which region we enumerate.

Everything refuted so far enumerates the same region more thoroughly. The union hull
(ours + field) does contain unclaimed cliques, which means the field's members favour
vertices ours does not. `hull_of` pads by "most connected into the hull" -- a greedy,
degree-like rule, i.e. exactly the preference every solver shares. That is why our hull
is everyone's hull.

Arms change only the padding rule, holding hull size and budget fixed:
  high   most connected into the hull  (the current rule, exact control)
  low    LEAST connected into the hull, among vertices still viable (deg >= omega-1)
  anti   vertices absent from all our max cliques, by descending degree
  rand   uniform among viable vertices
"""
import collections, json, os, sys, time
import numpy as np
sys.path.insert(0, '/workspace/better_83')
sys.path.insert(0, '/workspace/better_83/research')
sys.path.insert(0, '/home/dev/.claude/jobs/5443dea7/tmp')
from CliqueAI.graph.codec import GraphCodec
from hull import write_dimacs, run_momc

def hull_mode(pool, mx, M, target, mode, rng):
    H = set()
    for c in pool:
        if len(c) == mx:
            H.update(c)
    core = set(H)
    if len(H) >= target:
        return sorted(H)
    n = M.shape[0]
    deg = M.sum(axis=1)
    viable = [v for v in range(n) if v not in H and deg[v] >= mx - 1]
    if not viable:
        viable = [v for v in range(n) if v not in H]
    Hl = sorted(H)
    conn = M[:, Hl].astype(np.int64).sum(axis=1)   # see hull.py: unsigned negate
    if mode == 'high':
        viable.sort(key=lambda v: -conn[v])
    elif mode == 'low':
        viable.sort(key=lambda v: conn[v])
    elif mode == 'anti':
        viable = [v for v in viable if v not in core]
        viable.sort(key=lambda v: -deg[v])
    elif mode == 'rand':
        rng.shuffle(viable)
    for v in viable:
        if len(H) >= target:
            break
        H.add(int(v))
    return sorted(H)

if __name__ == '__main__':
    S = int(sys.argv[1]); nrounds = int(sys.argv[2]); frac = float(sys.argv[3])
    mode = sys.argv[4]; out = sys.argv[5]
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
    cd = GraphCodec(); rng = np.random.default_rng(20260823)
    A = collections.defaultdict(list); pools = {}
    for r in rows:
        M = np.array(cd.decode_matrix(r['matrix_b92']), dtype=np.uint8)
        lscc = cache[r['uuid']]
        mx = max(len(c) for c in lscc)
        F = {tuple(sorted(map(int, a['clique'])))
             for a in r['answers'] if a.get('opt', 0) > 0}
        if not F:
            continue
        fmx = max([mx] + [len(c) for c in F])
        H = hull_mode(lscc, mx, M, S, mode, rng)
        p = os.path.join('/home/dev/.claude/jobs/5443dea7/tmp', 'h3_%d.col' % os.getpid())
        write_dimacs(M, H, p)
        cl, to = run_momc(p, 2000, max(0.5, float(r.get('time_limit') or 10.0) * frac))
        good = []
        if not to and cl:
            for c in cl:
                if not c or max(c) > len(H):
                    continue
                v = tuple(sorted(H[i - 1] for i in c))
                if len(v) == fmx and all(M[a, b] for a in v for b in v if a != b):
                    good.append(v)
        L = [c for c in lscc if len(c) == fmx]
        seen = set(); merged = []
        for c in good + L:
            if c not in seen:
                seen.add(c); merged.append(c)
        if not merged:
            continue
        pools[r['uuid']] = [list(c) for c in merged]
        A['to'].append(1 if to else 0); A['momc'].append(len(good))
        A['pool'].append(len(merged))
        A['novel'].append(len(set(merged) - F) / len(merged))
        A['pre8'].append(sum(1 for c in merged[:8] if c not in F) / min(8, len(merged)))
    json.dump(pools, open(out, 'w'))
    import statistics as st
    print("mode=%-5s S=%d frac=%.2f  rounds=%d  timeouts=%.0f%%  MoMC %.1f  pool %.1f"
          % (mode, S, frac, len(A['pool']), 100*st.mean(A['to']),
             st.mean(A['momc']), st.mean(A['pool'])))
    print("   novel share  pool %.1f%%   first-8 prefix %.1f%%"
          % (100*st.mean(A['novel']), 100*st.mean(A['pre8'])))
