#!/usr/bin/env python3
"""Does ordering the omega pool by a clique feature change who we collide with?

The floor study measured AUC(feature, field-covers-this-clique) over 9809 cliques
and found deg_sum at 0.4544 +- 0.0128 -- 3.6 SE BELOW 0.5, i.e. real signal in the
INVERTED direction: high-degree cliques are LESS likely to be occupied.  The
selection experiment that followed only ever reordered by basin, so this was never
tested.  allocate() spreads one hotkey per clique over the front of the pool, so
the pool's order decides which cliques we take.

    python research_manual/eda/exp_pool_order.py -N 70 --order deg_desc
"""
import argparse
import collections
import json
import os
import statistics
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

import paths
import simulate
import solver
import pick_derived
from CliqueAI.graph.codec import GraphCodec

DEG = {}
ORDER = "none"


def _key(clique, deg):
    s = float(sum(deg[v] for v in clique))
    if ORDER == "deg_desc":
        return -s
    if ORDER == "deg_asc":
        return s
    return 0.0


_orig = pick_derived.picker


def picker(pool, uuid, hotkeys, **kw):
    deg = DEG.get(str(uuid))
    if deg is not None and ORDER != "none":
        pool = sorted(pool, key=lambda c: (-len(c), _key(c, deg)))
    return _orig(pool, uuid, hotkeys, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-N", type=int, required=True)
    ap.add_argument("--order", default="none",
                    choices=("none", "deg_desc", "deg_asc"))
    ap.add_argument("--only", default="latest100.txt")
    ap.add_argument("--pool-cache", default=os.path.join(paths.CACHE, "cache_latest100.jsonl"))
    ap.add_argument("--pool-k-mult", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    global ORDER
    ORDER = args.order
    pick_derived.picker = picker

    solver.configure(fleet_n=args.N, pool_cache=args.pool_cache,
                     pool_k_mult=args.pool_k_mult)
    with open(paths.METAGRAPH_JSON) as handle:
        meta = json.load(handle)
    victims = simulate.pick_victims(meta, args.N)
    rows = simulate.load_rounds(paths.ROUNDS_JSON, 100000, paths.rounds_list(args.only))

    codec = GraphCodec()
    for _, rid, rec in rows:
        A = np.asarray(codec.decode_matrix(rec["encoded_matrix"]), dtype=np.uint8)
        DEG[str(rid)] = A.sum(axis=1).astype(np.int64)

    out, scores, coldkey_of, n_queried, n_late = simulate.run(rows, victims)
    with open(args.out, "w") as handle:
        json.dump(out, handle)

    ours = collections.defaultdict(list)
    field = collections.defaultdict(list)
    alone = tot = 0
    for rec in out.values():
        cnt = collections.Counter(tuple(sorted(a[3])) for a in rec["answers"] if a[3])
        for a, s in zip(rec["answers"], rec["scores"]):
            (ours if a[1].startswith("our_") else field)[a[1]].append(s)
            if a[1].startswith("our_") and a[3]:
                tot += 1
                if cnt[tuple(sorted(a[3]))] == 1:
                    alone += 1
    om = [statistics.mean(v) for v in ours.values() if v]
    fm = sorted(statistics.mean(v) for v in field.values() if v)
    i = 0.10 * (len(fm) - 1)
    lo = int(i)
    hi = min(lo + 1, len(fm) - 1)
    cut = fm[lo] + (i - lo) * (fm[hi] - fm[lo])
    print("N=%-4d order=%-9s share %5.1f%%  margin %+0.5f  edge %+0.5f  alone %.3f"
          % (args.N, args.order,
             100.0 * sum(1 for m in om if m <= cut) / len(om),
             statistics.mean(om) - cut,
             statistics.mean(om) - statistics.mean(fm),
             alone / max(1, tot)))


if __name__ == "__main__":
    main()
