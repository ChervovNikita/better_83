"""H-HULL: exact enumeration of maximum cliques inside the hull of our LSCC pool.

Full-graph exact enumeration is intractable here (measured: MoMC times out >30s even
with omega injected). But the union of vertices spanned by the cliques our local search
already found is a few hundred vertices, and exact enumeration inside THAT is fast --
and returns maximum cliques of the FULL graph, because any clique inside an induced
subgraph is a clique of G, and it is maximum whenever it hits omega.
"""
import collections, json, os, subprocess, sys, time
import numpy as np
sys.path.insert(0, '/workspace/better_83')
sys.path.insert(0, '/workspace/better_83/research')
from CliqueAI.graph.codec import GraphCodec

MOMC = '/home/dev/.claude/jobs/5443dea7/tmp/momc'
TMP = '/home/dev/.claude/jobs/5443dea7/tmp'

def write_dimacs(M, verts, path):
    idx = {v: i + 1 for i, v in enumerate(verts)}
    edges = []
    for a in range(len(verts)):
        for b in range(a + 1, len(verts)):
            if M[verts[a], verts[b]]:
                edges.append((idx[verts[a]], idx[verts[b]]))
    with open(path, 'w') as f:
        f.write("p edge %d %d\n" % (len(verts), len(edges)))
        for u, v in edges:
            f.write("e %d %d\n" % (u, v))

def run_momc(path, want, timeout):
    try:
        r = subprocess.run([MOMC, path, '-a', str(want)], capture_output=True,
                           text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, True
    out = []
    for line in r.stdout.splitlines():
        if line.startswith('M '):
            # DIMACS is 1-indexed and MoMC pads with 0; H[0-1] would WRAP to the
            # last hull vertex and fabricate a clique that is not one.
            vs = [int(x) for x in line.split()[1:] if x.isdigit()]
            vs = [v for v in vs if v >= 1]
            if vs:
                out.append(tuple(sorted(set(vs))))
    return out, False

def hull_of(pool, mx, M, target):
    """Vertices spanned by our max-size cliques, grown by best connection to the hull."""
    H = set()
    for c in pool:
        if len(c) == mx:
            H.update(c)
    if len(H) >= target:
        return sorted(H)
    n = M.shape[0]
    Hl = sorted(H)
    # add the vertices most connected INTO the hull -- they are the ones that could
    # complete a clique with hull members
    # int64: M is uint8 so the sum is unsigned, and np.argsort(-score) then negates
    # an unsigned value. Harmless here only because no vertex has ZERO connection to
    # the hull in a graph this dense (verified: 0 of 4372 vertices over 6 rounds, and
    # the pad order is identical either way) -- but wrong, so it is not left in.
    score = M[:, Hl].astype(np.int64).sum(axis=1)
    for v in np.argsort(-score):
        if len(H) >= target:
            break
        H.add(int(v))
    return sorted(H)

if __name__ == '__main__':
    target = int(sys.argv[1]); nrounds = int(sys.argv[2]); tmo = float(sys.argv[3])
    want = int(sys.argv[4]) if len(sys.argv) > 4 else 200
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
    tos = 0
    for r in rows:
        M = np.array(cd.decode_matrix(r['matrix_b92']), dtype=np.uint8)
        pool = cache[r['uuid']]
        mx = max(len(c) for c in pool)
        F = set()
        for a in r['answers']:
            if a.get('opt', 0) > 0:
                F.add(tuple(sorted(map(int, a['clique']))))
        fmx = max([mx] + [len(c) for c in F])
        H = hull_of(pool, mx, M, target)
        p = os.path.join(TMP, 'h.col')
        write_dimacs(M, H, p)
        t0 = time.time()
        cl, timedout = run_momc(p, want, tmo)
        dt = time.time() - t0
        if timedout:
            tos += 1
            continue
        back = [tuple(sorted(H[i - 1] for i in c)) for c in cl]
        back = [c for c in back if len(c) == fmx]
        # VERIFY every clique against G rather than trusting the parse. MoMC's
        # multi-clique output packs cliques into one stack separated by sentinels and
        # its Count is known to run +1; a mis-split there would silently hand us a
        # vertex set that is not a clique, which the validator would reject and score 0.
        good = [c for c in back
                if all(M[a, b] for a in c for b in c if a != b)]
        agg['dropped'].append(len(back) - len(good))
        back = good
        ours = {c for c in pool if len(c) == fmx}
        agg['n'].append(len(back)); agg['t'].append(dt)
        agg['hull'].append(len(H)); agg['valid'].append(1)
        agg['lscc'].append(len(ours))
        agg['field'].append(len({c for c in F if len(c) == fmx}))
        agg['new_vs_lscc'].append(len(set(back) - ours))
        agg['novel'].append(len(set(back) - F))
        agg['unclaimed'].append(len(set(back) - F - ours))
    import statistics as st
    if not agg['n']:
        print("all %d rounds timed out at hull=%d" % (len(rows), target)); sys.exit()
    print("hull=%-4d rounds=%-3d timeouts=%-3d  time %.3fs (max %.2fs)  parse-rejects %.2f/round"
          % (target, len(agg['n']), tos, st.mean(agg['t']), max(agg['t']),
             st.mean(agg['dropped'])))
    print("   maxima found by MoMC   %7.1f   (our LSCC pool %.1f, field %.1f)"
          % (st.mean(agg['n']), st.mean(agg['lscc']), st.mean(agg['field'])))
    print("   new vs our LSCC pool   %7.1f" % st.mean(agg['new_vs_lscc']))
    print("   NOVEL (field lacks)    %7.1f   share %.1f%%"
          % (st.mean(agg['novel']), 100 * st.mean(agg['novel']) / max(st.mean(agg['n']), 1e-9)))
