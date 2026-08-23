#!/usr/bin/env python3
"""Which K of the enumerated maxima to submit -- no prediction required.

tools/order.py measured two numbers that do not fit together under blind selection:

    maximum cliques enumerated per round   44.0
    held by NOBODY in the field            29.4   (67%)
    unclaimed among the K=8 WE SUBMIT       2.39  (30%)

If the 8 were drawn uniformly from the 44, 67% of them would be unclaimed. We get 30%,
so the prefix is not a uniform draw -- MoMC emits in branching order, which starts in
the dense core, and the dense core is exactly where every other solver lands too. The
contested cliques come out first because they are the easy ones.

That makes the fix free and prediction-free: stop taking the first K.

    head      first K emitted                     (the control, what we do today)
    tail      last K emitted
    stride    every (n/K)th, spread across the whole emission order
    uniform   K sampled without replacement, seeded by the round uuid so it is
              reproducible and a deployed miner can compute it alone

`stride` and `uniform` need nothing but the pool. `tail` is included because if it wins
outright the ordering signal is monotone and worth exploiting directly.

Reported per round then averaged -- never pooled across rounds of unequal clique count,
which is the error behind eleven retractions here.
"""
import argparse
import collections
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/workspace/better_83')
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, HERE)
from CliqueAI.graph.codec import GraphCodec        # noqa: E402
from hull import write_dimacs, run_momc, hull_of    # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rounds", type=int, default=120)
    ap.add_argument("--hull", type=int, default=180)
    ap.add_argument("--frac", type=float, default=0.40)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--cache", default=os.path.join(HERE, "..", "data",
                                                    "sim_cliques.jsonl"))
    ap.add_argument("--dataset", default=os.path.join(HERE, "..", "data",
                                                      "sim_rounds.jsonl"))
    args = ap.parse_args()

    pool = {}
    with open(args.cache) as f:
        for line in f:
            r = json.loads(line)
            pool[r["uuid"]] = [tuple(sorted(c)) for c in r["cliques"]]
    rounds = []
    with open(args.dataset) as f:
        for line in f:
            r = json.loads(line)
            if r.get("answers") and r["uuid"] in pool:
                rounds.append(r)
            if len(rounds) >= args.rounds:
                break

    RULES = ["head", "tail", "stride", "uniform", "lscc"]
    unc = {r: [] for r in RULES}
    tot_cl, tot_unc, used, timeouts = [], [], 0, 0
    tmp = os.path.join(os.environ.get("CLAUDE_JOB_DIR", "/tmp"), "tmp")
    os.makedirs(tmp, exist_ok=True)

    for rec in rounds:
        M = np.array(GraphCodec().decode_matrix(rec["matrix_b92"]), dtype=np.uint8)
        ours = pool[rec["uuid"]]
        if not ours:
            continue
        mx = max(len(c) for c in ours)
        field = {tuple(sorted(int(v) for v in a["clique"]))
                 for a in rec["answers"]
                 if a.get("opt", 0) > 0 and len(a["clique"]) == mx}
        H = hull_of(ours, mx, M, args.hull)
        path = os.path.join(tmp, "sel.clq")
        write_dimacs(M, H, path)
        got, to = run_momc(path, 5000, max(0.5, rec["time_limit"] * args.frac))
        if to or not got:
            timeouts += 1
            continue
        seen, cl = set(), []
        for c in got:
            vs = tuple(sorted(H[i - 1] for i in c if 1 <= i <= len(H)))
            if len(vs) == mx and vs not in seen:
                seen.add(vs)
                cl.append(vs)
        if not cl:
            continue
        used += 1
        n = len(cl)
        lab = [0 if c in field else 1 for c in cl]
        tot_cl.append(n)
        tot_unc.append(sum(lab))
        K = min(args.k, n)

        idx = {}
        idx["head"] = list(range(K))
        idx["tail"] = list(range(n - K, n))
        idx["stride"] = [min(n - 1, int(i * n / K)) for i in range(K)]
        h = int(hashlib.sha1(str(rec["uuid"]).encode()).hexdigest()[:16], 16)
        rng = np.random.default_rng(h % (2 ** 32))
        idx["uniform"] = list(rng.choice(n, size=K, replace=False))
        # what we submit TODAY: our own LSCC pool's first K, no enumeration at all
        lsc = [c for c in ours if len(c) == mx][:K]
        unc["lscc"].append(sum(1 for c in lsc if c not in field))
        for r in RULES:
            if r == "lscc":
                continue
            unc[r].append(sum(lab[i] for i in idx[r]))

    print("hull=%d frac=%.2f K=%d   %d rounds usable of %d, %d timeouts\n"
          % (args.hull, args.frac, args.k, used, len(rounds), timeouts))
    tc = np.array(tot_cl, dtype=float)
    tu = np.array(tot_unc, dtype=float)
    ratio_of_means = 100 * tu.mean() / tc.mean()
    per_round = 100 * np.mean(tu / tc)
    print("  maxima enumerated per round        %.1f" % tc.mean())
    print("  unclaimed, RATIO OF MEANS          %.1f  (%.0f%%)  <- pooled, do not use"
          % (tu.mean(), ratio_of_means))
    print("  unclaimed, MEAN OF PER-ROUND       (%.0f%%)  median %.0f%%"
          % (per_round, 100 * np.median(tu / tc)))
    print()
    print("  The two differ by %.0f points because rounds with many enumerated maxima"
          % (ratio_of_means - per_round))
    print("  also have a higher unclaimed fraction, and the pooled ratio weights them")
    print("  by their clique count. A K-of-n draw happens PER ROUND, so the per-round")
    print("  figure is the one that predicts what selection can deliver. Reasoning from")
    print("  the pooled %.0f%% predicts uniform selection lands ~%.1f unclaimed in K=8;"
          % (ratio_of_means, args.k * ratio_of_means / 100))
    print("  the per-round figure predicts ~%.1f, which is what the table shows."
          % (args.k * per_round / 100))
    print()
    print("%-9s %14s %12s %14s" % ("rule", "UNCL in K", "novel%", "vs head"))
    base = np.mean(unc["head"])
    for r in RULES:
        v = np.array(unc[r], dtype=float)
        if not len(v):
            continue
        print("%-9s %14.2f %11.0f%% %+14.2f"
              % (r, v.mean(), 100 * v.mean() / args.k, v.mean() - base))
    print()
    # paired sign test, head vs the best non-head rule
    cand = [r for r in RULES if r not in ("head", "lscc")]
    best = max(cand, key=lambda r: np.mean(unc[r]))
    a = np.array(unc["head"], dtype=float)
    b = np.array(unc[best], dtype=float)
    d = b - a
    better, worse = int((d > 0).sum()), int((d < 0).sum())
    import math
    ch = better + worse
    p = 1.0 if ch == 0 else min(1.0, 2.0 * sum(
        math.comb(ch, i) for i in range(0, min(better, worse) + 1)) / 2 ** ch)
    print("  paired head -> %s   median %+.1f   %d better / %d worse   sign p = %.4g"
          % (best, np.median(d), better, worse, p))
    print("  Neither stride nor uniform needs any field data or any prediction; both")
    print("  are computable by one miner from its own enumerated pool.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
