"""Why does the UNION hull surface unclaimed cliques when OUR hull of the same size does not?

Two candidate causes, and they imply opposite next steps:
  (A) our hull does not CONTAIN those cliques  -> the vertices are the barrier
  (B) it contains them but MoMC does not RETURN them -> enumeration is incomplete and
      returns a search-order prefix biased to the attractor

Test: find the unclaimed cliques via the union hull, then for each one ask both
questions separately on an our-pool-only hull of the same size.
"""
import collections, json, os, sys
import numpy as np
sys.path.insert(0, '/workspace/better_83')
sys.path.insert(0, '/workspace/better_83/research')
sys.path.insert(0, '/home/dev/.claude/jobs/5443dea7/tmp')
from CliqueAI.graph.codec import GraphCodec
from hull import write_dimacs, run_momc, hull_of

if __name__ == '__main__':
    S = int(sys.argv[1]); nrounds = int(sys.argv[2]); tmo = float(sys.argv[3])
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
    A = collections.defaultdict(int)
    per = collections.defaultdict(list)
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
        if not L:
            continue
        def enum(H):
            p = '/home/dev/.claude/jobs/5443dea7/tmp/dg_%d.col' % os.getpid()
            write_dimacs(M, H, p)
            cl, to = run_momc(p, 2000, tmo)
            if to:
                return None
            out = set()
            for c in cl:
                if not c or max(c) > len(H):
                    continue
                v = tuple(sorted(H[i - 1] for i in c))
                if len(v) == mx and all(M[a, b] for a in v for b in v if a != b):
                    out.add(v)
            return out
        UH = set()
        for c in list(L) + list(FM):
            UH.update(c)
        u = enum(sorted(UH))
        if u is None:
            A['union_timeout'] += 1; continue
        unclaimed = u - L - FM
        if not unclaimed:
            continue
        OH = set(hull_of(list(L), mx, M, S))
        o = enum(sorted(OH))
        if o is None:
            A['ours_timeout'] += 1; continue
        inside = {c for c in unclaimed if set(c) <= OH}
        A['unclaimed'] += len(unclaimed)
        A['inside_our_hull'] += len(inside)
        A['returned_by_momc'] += len(inside & o)
        A['rounds'] += 1
        per['ourhull'].append(len(OH)); per['unionhull'].append(len(UH))
        per['ours_found'].append(len(o)); per['union_found'].append(len(u))
    import statistics as st
    if not A['rounds']:
        print("no usable rounds (union timeouts %d, ours %d)"
              % (A['union_timeout'], A['ours_timeout'])); sys.exit()
    print("S=%d  %d rounds with unclaimed cliques (timeouts: union %d, ours %d)"
          % (S, A['rounds'], A['union_timeout'], A['ours_timeout']))
    print("   hull sizes: ours %.0f | union %.0f" % (st.mean(per['ourhull']), st.mean(per['unionhull'])))
    print("   maxima found: ours %.1f | union %.1f" % (st.mean(per['ours_found']), st.mean(per['union_found'])))
    print()
    print("   unclaimed cliques (via union hull)      %5d" % A['unclaimed'])
    print("     (A) inside our own hull               %5d  = %.1f%%"
          % (A['inside_our_hull'], 100.0 * A['inside_our_hull'] / A['unclaimed']))
    print("     (B) AND actually returned by MoMC     %5d  = %.1f%% of those inside"
          % (A['returned_by_momc'],
             100.0 * A['returned_by_momc'] / max(A['inside_our_hull'], 1)))
    print()
    print("   ratios of totals, not means of per-round ratios")
