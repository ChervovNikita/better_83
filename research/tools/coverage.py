#!/usr/bin/env python3
"""How many DISTINCT maximum cliques does the field cover, and are they in our hull?

Two measured facts do not fit together:

  * the field's mean diversity is 0.7487 against our 0.5141, so a typical field answer
    is nearly unique -- roughly 1.33 holders
  * exact enumeration inside our own 180-vertex hull returns ~44 maximum cliques of
    which only ~14.6 are held by anyone in the field

With ~50 field answers per round, 14.6 distinct cliques would mean ~3.4 holders each and
a diversity far below what they actually earn. So either the field covers many more
distinct cliques than our hull contains, or our hull is not where the field is.

This distinguishes the two. Per round, and per round ONLY -- every quantity here is a
ratio, and pooling ratios across rounds of unequal answer count is the error that has
produced twelve retractions in this project.

    distinct        how many distinct max-size cliques the field submits
    in-hull         how many of those lie entirely inside our 180-vertex hull
    enumerated      how many our MoMC run inside that hull actually returns
    field-outside   distinct field cliques with at least one vertex outside the hull

If field-outside is large, our hull is NOT everyone's hull, the reach line was closed
against the wrong region, and the whole enumeration result needs rereading.
"""
import argparse
import collections
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
    ap.add_argument("--cache", default=os.path.join(HERE, "..", "data",
                                                    "sim_cliques.jsonl"))
    ap.add_argument("--dataset", default=os.path.join(HERE, "..", "data",
                                                      "sim_rounds.jsonl"))
    ap.add_argument("--enumerate", action="store_true",
                    help="also run MoMC; without it only the hull-membership test runs, "
                         "which needs no solver and is exact")
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

    S = collections.defaultdict(list)
    tmp = os.path.join(os.environ.get("CLAUDE_JOB_DIR", "/tmp"), "tmp")
    os.makedirs(tmp, exist_ok=True)

    for rec in rounds:
        ours = pool[rec["uuid"]]
        if not ours:
            continue
        mx_ours = max(len(c) for c in ours)
        fa = [a for a in rec["answers"] if a.get("opt", 0) > 0]
        if not fa:
            continue
        mx = max(mx_ours, max(len(a["clique"]) for a in fa))
        fmax = [tuple(sorted(int(v) for v in a["clique"])) for a in fa
                if len(a["clique"]) == mx]
        if not fmax:
            continue
        M = np.array(GraphCodec().decode_matrix(rec["matrix_b92"]), dtype=np.uint8)
        H = set(hull_of(ours, mx_ours, M, args.hull))
        distinct = set(fmax)
        inside = {c for c in distinct if set(c) <= H}

        S["answers"].append(len(fmax))
        S["distinct"].append(len(distinct))
        S["holders_per_clique"].append(len(fmax) / len(distinct))
        S["in_hull"].append(len(inside))
        S["frac_in_hull"].append(len(inside) / len(distinct))
        op = [c for c in ours if len(c) == mx_ours]
        S["our_pool"].append(len(op))
        # The direct test of "our attractor is the field's attractor": how much of the
        # pool we actually draw from is already held by somebody. This is the quantity
        # that sets our diversity, and it is NOT the same as how much of the FIELD's
        # coverage we reproduce -- the pools are different sizes, so the two fractions
        # answer different questions and only this one predicts our reward.
        if op:
            S["pool_taken"].append(len([c for c in op if c in distinct]))
            S["frac_pool_taken"].append(len([c for c in op if c in distinct]) / len(op))
        sub = op[:8]
        if sub:
            S["frac_sub_taken"].append(len([c for c in sub if c in distinct]) / len(sub))
        S["omega_gap"].append(mx - mx_ours)

        if args.enumerate:
            Hl = sorted(H)
            path = os.path.join(tmp, "cov.clq")
            write_dimacs(M, Hl, path)
            got, to = run_momc(path, 5000, max(0.5, rec["time_limit"] * args.frac))
            if not to and got:
                seen = set()
                for c in got:
                    vs = tuple(sorted(Hl[i - 1] for i in c if 1 <= i <= len(Hl)))
                    if len(vs) == mx_ours:
                        seen.add(vs)
                S["enumerated"].append(len(seen))
                S["enum_field"].append(len(seen & distinct))

    n = len(S["distinct"])
    print("hull=%d   %d rounds usable of %d\n" % (args.hull, n, len(rounds)))
    print("%-22s %10s %10s %10s" % ("per round", "mean", "median", "p90"))
    for k in ["answers", "distinct", "holders_per_clique", "in_hull", "our_pool",
              "pool_taken", "enumerated", "enum_field"]:
        if not S[k]:
            continue
        v = np.array(S[k], dtype=float)
        print("%-22s %10.2f %10.2f %10.2f"
              % (k, v.mean(), np.median(v), np.percentile(v, 90)))
    print()
    f = np.array(S["frac_in_hull"], dtype=float)
    print("  field cliques inside OUR hull, MEAN OF PER-ROUND FRACTIONS  %.1f%%"
          % (100 * f.mean()))
    print("  same, median                                                %.1f%%"
          % (100 * np.median(f)))
    print("  same, ratio of means (POOLED, shown only for contrast)       %.1f%%"
          % (100 * np.mean(S["in_hull"]) / np.mean(S["distinct"])))
    if S["frac_pool_taken"]:
        pt = np.array(S["frac_pool_taken"], dtype=float)
        st = np.array(S["frac_sub_taken"], dtype=float)
        print()
        print("  OUR POOL already held by the field, mean of per-round   %.1f%%"
              % (100 * pt.mean()))
        print("  same, median                                            %.1f%%"
              % (100 * np.median(pt)))
        print("  our SUBMITTED first-8 already held, mean of per-round    %.1f%%"
              % (100 * st.mean()))
        print("  -> the gap between these two is what better SELECTION inside our own")
        print("     pool could win; the level of the first is what a different SEARCH")
        print("     would have to move.")
    g = np.array(S["omega_gap"], dtype=float)
    print()
    print("  rounds where the FIELD found a bigger clique than us: %d of %d"
          % (int((g > 0).sum()), n))
    print("  (that would break the enumeration comparison, since our hull is built")
    print("   from OUR max size)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
