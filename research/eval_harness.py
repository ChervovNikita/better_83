#!/usr/bin/env python3
"""Score a candidate SN83 solver against the real field, offline.

Runs your solver on each instance under that round's real time limit, then
replays the validator's scoring with your answer inserted into the actual set of
miner answers. Reports the mean reward you would have earned and, via the
observed score-to-income curve, the emission that implies.

Every instance carries its own `time_limit` (6 / 7.5 / 10 / 15 / 30 s), so the
per-deadline breakdown tells you which constraint your solver is losing under.

  python eval_harness.py data/v0.0.17/2026-08-20.jsonl --solver solver:solve
  python eval_harness.py bench.jsonl --solver mysolver:run --time-limit 7.5
"""
import argparse
import collections
import importlib
import json
import math
import os
import sys
import time

import numpy as np

from _common import DATA_DIR  # noqa: F401  (also puts the repo on sys.path)

# Emission share as a function of mean reward, measured by replaying
# BaseValidatorNeuron.set_weights() against the live 248-miner field.
CURVE = [(1.690, 0.0), (1.938, 0.0), (2.200, 0.0000041), (2.350, 0.00022),
         (2.450, 0.00196), (2.470, 0.00291), (2.500, 0.00516), (2.550, 0.01228),
         (2.591, 0.02290), (2.700, 0.07867), (2.845, 0.17588)]
MINER_ALPHA_DAY = 2951.6
ALPHA_TAO = 0.0106


def emission_share(mean_reward):
    xs = [c[0] for c in CURVE]
    if mean_reward <= xs[0]:
        return 0.0
    if mean_reward >= xs[-1]:
        return CURVE[-1][1]
    i = max(j for j in range(len(xs)) if xs[j] <= mean_reward)
    if i == len(xs) - 1:
        return CURVE[-1][1]
    (x0, y0), (x1, y1) = CURVE[i], CURVE[i + 1]
    t = (mean_reward - x0) / (x1 - x0)
    lo, hi = math.log(max(y0, 1e-12)), math.log(max(y1, 1e-12))
    return math.exp(lo + t * (hi - lo))        # the curve is exponential, interpolate in log


def decode(b92):
    from CliqueAI.graph.codec import GraphCodec
    return np.array(GraphCodec().decode_matrix(b92), dtype=np.uint8)


def check(A, clique):
    """Exactly CliqueScoreCalculator.is_valid_maximum_clique: clique, no repeats, maximal."""
    S = list(clique)
    if not S or len(set(S)) != len(S):
        return False
    n = A.shape[0]
    if any(v < 0 or v >= n for v in S):
        return False
    idx = np.array(S)
    if A[np.ix_(idx, idx)].sum() != len(S) * (len(S) - 1):
        return False
    cnt = A[idx].sum(axis=0)
    inC = np.zeros(n, dtype=bool)
    inC[idx] = True
    return not np.any((cnt == len(S)) & (~inC))


def replay_reward(rec, our_size, our_mult):
    sizes = []
    for k, v in rec["size_hist"].items():
        sizes += [int(k)] * v
    sizes.append(our_size)
    s = np.array(sizes, dtype=float)
    if s.max() <= 0 or our_size <= 0:
        return 0.0
    rel = s / s.max()
    pr = np.array([np.sum(s > x) / len(s) for x in s])
    om = np.exp(-pr / rel)
    om_n = om / om.max()
    max_delta = 1.0 if (rec["any_unique"] or our_mult == 1) else \
        max(1.0 / max(rec["best_clique_counts"] or [1]), 1.0 / our_mult)
    return float(om_n[-1] * (1 + rec["difficulty"]) + (1.0 / our_mult) / max_delta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--solver", default="solver:solve", help="module:function")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--time-scale", type=float, default=0.88,
                    help="fraction of the round's deadline to solve in; the rest is "
                         "network headroom. 1.0 solves to the wire and will overrun.")
    ap.add_argument("--time-limit", type=float, default=None,
                    help="only evaluate instances with this deadline")
    ap.add_argument("--max-n", type=int, default=None)
    args = ap.parse_args()

    mod, fn = args.solver.split(":")
    solve = getattr(importlib.import_module(mod), fn)

    recs = []
    with open(args.dataset) as f:
        for line in f:
            r = json.loads(line)
            if args.time_limit and r["time_limit"] != args.time_limit:
                continue
            if args.max_n and r["n"] > args.max_n:
                continue
            recs.append(r)
            if len(recs) >= args.limit:
                break
    if not recs:
        print("no instances matched the filters", file=sys.stderr)
        return 1

    rows = []
    for i, rec in enumerate(recs, 1):
        A = decode(rec["matrix_b92"])
        budget = rec["time_limit"] * args.time_scale
        t0 = time.time()
        out = solve(A, budget)
        elapsed = time.time() - t0
        clique = list(out[2]) if isinstance(out, tuple) else list(out)
        valid = check(A, clique)
        size = len(clique) if valid else 0
        mult = 1
        if valid and size == rec["best_size"]:
            key = tuple(sorted(clique))
            for t, c in zip(rec["best_cliques"], rec["best_clique_counts"]):
                if tuple(t) == key:
                    mult = c + 1
                    break
        rows.append(dict(n=rec["n"], tl=rec["time_limit"], best=rec["best_size"],
                         ours=size, valid=valid, mult=mult, elapsed=elapsed,
                         reward=replay_reward(rec, size, mult),
                         over=elapsed > rec["time_limit"] * 1.02 + 0.25))
        if i % 10 == 0:
            print(f"  {i}/{len(recs)}", file=sys.stderr, flush=True)

    d = np.array([r["ours"] - r["best"] for r in rows])
    rw = np.array([r["reward"] for r in rows])
    mean = rw.mean()
    share = emission_share(mean)

    print()
    print(f"instances             {len(rows):>7}")
    print(f"invalid / non-maximal {sum(1 for r in rows if not r['valid']):>7}")
    print(f"overran the deadline  {sum(1 for r in rows if r['over']):>7}   (>2% + 0.25s over)")
    print(f"matched field best    {int((d == 0).sum()):>7}  ({(d == 0).mean():.1%})")
    print(f"one short             {int((d == -1).sum()):>7}")
    print(f"two or more short     {int((d <= -2).sum()):>7}")
    print(f"beat the field        {int((d > 0).sum()):>7}")
    print(f"collided at best      {sum(1 for r in rows if r['mult'] > 1):>7}")

    print("\nby deadline — where the time constraint actually bites:")
    print(f"  {'limit':>6} {'n':>5} {'match':>7} {'-1':>5} {'<=-2':>5} {'mean rwd':>9} {'p50 solve':>10}")
    by = collections.defaultdict(list)
    for r in rows:
        by[r["tl"]].append(r)
    for tl in sorted(by):
        g = by[tl]
        gd = np.array([x["ours"] - x["best"] for x in g])
        print(f"  {tl:6.1f} {len(g):5d} {(gd == 0).mean():6.0%} "
              f"{int((gd == -1).sum()):5d} {int((gd <= -2).sum()):5d} "
              f"{np.mean([x['reward'] for x in g]):9.3f} "
              f"{np.median([x['elapsed'] for x in g]):9.2f}s")

    print("\nby graph size:")
    print(f"  {'|V|':>6} {'n':>5} {'match':>7} {'mean rwd':>9}")
    by_n = collections.defaultdict(list)
    for r in rows:
        by_n[r["n"] // 100 * 100].append(r)
    for nb in sorted(by_n):
        g = by_n[nb]
        gd = np.array([x["ours"] - x["best"] for x in g])
        print(f"  {nb:6d} {len(g):5d} {(gd == 0).mean():6.0%} "
              f"{np.mean([x['reward'] for x in g]):9.3f}")

    print()
    print(f"MEAN REWARD           {mean:.4f}   (field 2.526 | best UID 2.591 | perfect 2.845)")
    print(f"implied emission      {share:.4%}  ->  {share * MINER_ALPHA_DAY:.1f} alpha/day/UID "
          f"({share * MINER_ALPHA_DAY * ALPHA_TAO:.3f} TAO)")
    if mean < 2.35:
        print("VERDICT: dead zone — this solver would earn ~nothing. Do not register.")
    elif mean < 2.47:
        print("VERDICT: below the field median. Marginal at best.")
    elif mean < 2.60:
        print("VERDICT: competitive with today's field.")
    else:
        print("VERDICT: above the current best UID.")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.exit(main())
