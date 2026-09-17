#!/usr/bin/env python3
"""Order the omega pool by PREDICTED rival load, then play the normal picker.

The trained ranker (load_model.json, fitted on 399 rounds with every latest100
round excluded) scores each omega clique. allocate() spreads one hotkey per clique
over the front of the pool, so ordering the pool by ascending predicted load makes
the picker take the cliques the model thinks are least crowded.

Assignment is held FIXED. Reordering a pool also permutes which hotkey sends which
answer, and that permutation has zero long-run effect but a large finite-window
effect on the bottom-10% metric -- it is what made the earlier degree-ordering
table swing +-10 points. Here the emitted answers are re-assigned by a rule that
depends only on the answer SET and the round id, so the comparison isolates
selection.
"""
import argparse, collections, hashlib, json, os, statistics, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import paths, simulate, solver, pick_derived
from CliqueAI.graph.codec import GraphCodec

RD = "/home/dev/autoresearch-runs/sn83-occupancy/"
MODEL = json.load(open(RD + "load_model.json"))
W = np.asarray(MODEL["w"]); FEATS = MODEL["feats"]
GRAPH = {}
ORDER = "none"
_orig = pick_derived.picker


def clique_features(A, tops, hits):
    """The same 25 features the model was trained on, then within-round percentile."""
    n = A.shape[0]
    deg = A.sum(axis=1).astype(np.float64)
    dens = float(A.sum()) / (n * (n - 1))
    w = len(tops[0])
    members = collections.Counter()
    for c in tops:
        for v in c:
            members[v] += 1
    S = len(tops)
    core = deg.copy()
    alive = np.ones(n, dtype=bool); dd = deg.copy()
    for _ in range(200):
        if not alive.any():
            break
        lo = dd[alive].min()
        peel = alive & (dd <= lo)
        if not peel.any():
            break
        core[peel] = lo
        for v in np.flatnonzero(peel):
            dd -= A[v]
        alive &= ~peel
        dd[~alive] = 1e9
    order = sorted(range(S), key=lambda i: -hits[i])
    rank = {order[i]: i for i in range(S)}
    best = max(range(S), key=lambda i: hits[i]); btop = set(tops[best])
    rows = []
    for i, c in enumerate(tops):
        idx = np.asarray(c); d = deg[idx]
        adj = A[:, idx].sum(axis=1)
        vf = np.asarray([members[v] for v in c], dtype=np.float64)
        cs = core[idx]
        rows.append({
            "deg_sum": d.sum(), "deg_min": d.min(), "deg_max": d.max(),
            "deg_mean": d.mean(), "deg_std": d.std(),
            "ext_near": float((adj >= w - 1).sum() - w),
            "overlap": float(sum(members[v] for v in c)),
            "basin": float(hits[i]), "basin_rank": float(rank[i]),
            "basin_frac": float(rank[i]) / max(1, S - 1),
            "tri_mean": float(adj[idx].mean()),
            "cross_deg": float(adj.sum() - w * (w - 1)),
            "deg_sum_z": 0.0, "ext_near_z": 0.0, "overlap_z": 0.0,
            "vf_min": vf.min(), "vf_max": vf.max(), "vf_mean": vf.mean(),
            "core_min": cs.min(), "core_mean": cs.mean(),
            "vid_min": float(idx.min()), "vid_mean": float(idx.mean()),
            "jac_top": len(btop & set(c)) / float(len(btop | set(c))),
            "ext_full": float((adj >= w).sum()),
            "nbr_cover": float((A[idx].sum(axis=0) > 0).sum()) / n,
        })
    M = np.asarray([[r[f] for f in FEATS] for r in rows], dtype=np.float64)
    for src, dst in (("deg_sum", "deg_sum_z"), ("ext_near", "ext_near_z"),
                     ("overlap", "overlap_z")):
        if src in FEATS and dst in FEATS:
            col = M[:, FEATS.index(src)]
            s = col.std()
            M[:, FEATS.index(dst)] = (col - col.mean()) / s if s > 0 else 0.0
    pct = np.empty_like(M)
    for j in range(M.shape[1]):
        o = M[:, j].argsort(kind="stable")
        rk = np.empty(len(o)); rk[o] = np.arange(len(o))
        pct[:, j] = rk / max(1, len(o) - 1)
    return pct


def canonical_emit(uuid, hotkeys, answers, pool_index):
    """Assign the SAME answer multiset to hotkeys by a rule that ignores pool order."""
    keyed = sorted(answers, key=lambda a: (pool_index.get(tuple(sorted(a)), 1 << 30),
                                           tuple(sorted(a))))
    off = int(hashlib.sha1(str(uuid).encode()).hexdigest()[:8], 16)
    return [list(keyed[(i + off) % len(keyed)]) for i in range(len(hotkeys))]


def picker(pool, uuid, hotkeys, **kw):
    pool_index = {tuple(sorted(c)): i for i, c in enumerate(pool)}
    if ORDER != "none":
        A = GRAPH.get(str(uuid))
        w = max(len(c) for c in pool)
        tops = [list(c) for c in pool if len(c) == w]
        rest = [list(c) for c in pool if len(c) != w]
        if A is not None and len(tops) >= 2:
            hits = kw.get("hits") or [0] * len(pool)
            hmap = {tuple(sorted(c)): hits[i] if i < len(hits) else 0
                    for i, c in enumerate(pool)}
            th = [hmap[tuple(sorted(c))] for c in tops]
            score = clique_features(A, [tuple(sorted(c)) for c in tops], th) @ W
            sgn = 1.0 if ORDER == "asc" else -1.0
            tops = [tops[i] for i in np.argsort(sgn * score, kind="stable")]
            pool = tops + rest
    ans = _orig(pool, uuid, hotkeys, **kw)
    return canonical_emit(uuid, hotkeys, ans, pool_index)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-N", type=int, required=True)
    ap.add_argument("--order", default="none", choices=("none", "asc", "desc"))
    ap.add_argument("--only", default="latest100.txt")
    ap.add_argument("--pool-cache",
                    default=os.path.join(paths.CACHE, "cache_latest100.jsonl"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    global ORDER
    ORDER = args.order
    pick_derived.picker = picker
    solver.configure(fleet_n=args.N, pool_cache=args.pool_cache)
    meta = json.load(open(paths.METAGRAPH_JSON))
    victims = simulate.pick_victims(meta, args.N)
    rows = simulate.load_rounds(paths.ROUNDS_JSON, 100000, paths.rounds_list(args.only))
    codec = GraphCodec()
    for _, rid, rec in rows:
        GRAPH[str(rid)] = np.asarray(codec.decode_matrix(rec["encoded_matrix"]),
                                     dtype=np.uint8)
    out, *_ = simulate.run(rows, victims)
    json.dump(out, open(args.out, "w"))
    print("wrote %s" % args.out)


if __name__ == "__main__":
    main()
