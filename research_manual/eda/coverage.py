#!/usr/bin/env python3
"""Does our harvest contain every omega-clique the whole field found?

    .venv/bin/python research_manual/eda/coverage.py --rounds 30

Counting cliques and finding "we hold more than they submit" is close to
automatic: we run thousands of walker jobs, each miner submits one answer.  The
real test is SET CONTAINMENT against the union over every coldkey -- ~50
independent solvers, which between them are a far stronger enumerator than any
one of them.  Anything they found and we did not is a clique our search misses.

Tuning set only.  Reruns the harvest rather than reading pools.json, because
that cache keeps only the top 64 cliques per round and containment needs all of
them.
"""

import argparse
import collections
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
ROOT = os.path.dirname(PARENT)
for _p in (ROOT, PARENT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--mult", type=float, default=1.0,
                    help="budget multiplier; 1.0 is what we actually ship")
    ap.add_argument("--fleet", type=int, default=40)
    ap.add_argument("--lo", type=int, default=0,
                    help="only rounds whose cached n_top is >= lo")
    ap.add_argument("--hi", type=int, default=10 ** 9)
    args = ap.parse_args()

    from CliqueAI.graph.codec import GraphCodec
    import fleet_solver_gpu as fg
    import gpu_lib
    import fit_field

    tune = json.load(open(os.path.join(HERE, "tuning_data.json")))
    assert not (set(tune) & set(json.load(open(os.path.join(PARENT, "artifacts", "data", "rounds.json"))))), \
        "eval leak"
    meta = json.load(open(os.path.join(PARENT, "artifacts", "data", "metagraph.json")))
    dropped = fit_field.victims(meta, args.fleet)

    rows = sorted(tune.items(), key=lambda kv: kv[1]["timestamp"])
    if args.lo or args.hi < 10 ** 9:
        feats = {r["uuid"]: r for r in
                 json.load(open(os.path.join(HERE, "field_features.json")))}
        rows = [(rid, rec) for rid, rec in rows
                if rid in feats and args.lo <= feats[rid]["n_top"] <= args.hi]
    rows = rows[:args.rounds]
    codec = GraphCodec()

    print("%-14s %5s %7s %7s %8s %7s %7s"
          % ("round", "omega", "field", "ours", "covered", "missed", "our_om"))
    print("%-14s %5s %7s %7s %8s %7s %7s"
          % ("-" * 14, "-" * 5, "-" * 7, "-" * 7, "-" * 8, "-" * 7, "-" * 7))
    tot_field = tot_missed = 0
    worse_omega = 0
    missed_rounds = []
    for rid, rec in rows:
        A = np.array(codec.decode_matrix(rec["encoded_matrix"]), dtype=np.uint8)
        field = [tuple(sorted(cl)) for uid, _h, _c, cl in rec["answers"]
                 if cl and uid not in dropped]
        if not field:
            continue
        budget = (rec["time_limit"] - 2.0) * args.mult
        champ = sorted(fg.fleet_solver._solve_one(A, budget * fg.CHAMPION_SHARE,
                                                  seed=1))
        with gpu_lib.GpuClique(A) as g:
            pool, _, _hits = g.harvest(budget * (1 - fg.CHAMPION_SHARE), seed=1,
                                max_steps=fg.STEPS, boot_steps=fg.BOOT_STEPS,
                                init_clique=champ, max_out=8192)
        our_om = max(len(c) for c in pool)
        om = max(our_om, max(len(c) for c in field))
        ours = {tuple(c) for c in pool if len(c) == om}
        theirs = {c for c in field if len(c) == om}
        missed = theirs - ours
        tot_field += len(theirs)
        tot_missed += len(missed)
        if our_om < om:
            worse_omega += 1
        if missed:
            missed_rounds.append((rec["number_of_nodes"], len(theirs), len(missed),
                                  len(ours)))
        print("%-14s %5d %7d %7d %8d %7d %7d"
              % ("n=%d tl=%g" % (rec["number_of_nodes"], rec["time_limit"]),
                 om, len(theirs), len(ours), len(theirs & ours), len(missed),
                 our_om), flush=True)

    print()
    print("field omega-cliques over %d rounds : %d" % (len(rows), tot_field))
    print("of those, NOT in our harvest       : %d (%.2f%%)"
          % (tot_missed, 100.0 * tot_missed / max(1, tot_field)))
    print("rounds where we missed any         : %d of %d"
          % (len(missed_rounds), len(rows)))
    print("rounds where our omega was smaller : %d" % worse_omega)
    if missed_rounds:
        print()
        print("  %-8s %8s %8s %8s" % ("n", "field", "ours", "missed"))
        for n_, tf, mi, ou in sorted(missed_rounds, key=lambda r: -r[2])[:10]:
            print("  %-8d %8d %8d %8d" % (n_, tf, ou, mi))


if __name__ == "__main__":
    main()
