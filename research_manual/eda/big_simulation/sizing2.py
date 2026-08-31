"""Sizes the deadline from measured per-round cost, not an assumed rate."""

import argparse
import json
import os
import random
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import bound
import partial as partial_module
import responders
import rounds as rounds_module
import strategies
from scoring import score


def main():
    """Prints estimated total runtime and coverage for a range of deadlines."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", default="135,140,150,175,200")
    parser.add_argument("--sample", type=int, default=30)
    parser.add_argument("--n", type=int, default=249)
    parser.add_argument("--seed", type=int, default=5)
    args = parser.parse_args()

    all_rounds = rounds_module.load()
    total = len(all_rounds)
    rng = random.Random(args.seed)
    per_split = {}
    for g in [int(x) for x in args.splits.split(",")]:
        o = args.n - g
        rows = []
        for rnd in rng.sample(all_rounds, args.sample):
            q_a, q_b = rounds_module.split_queries(rnd, g, o)
            if q_a == 0 or q_b == 0:
                continue
            rnd.q_a, rnd.fleet_a, rnd.fleet_b = q_a, g, o
            t = time.time()
            board = strategies.REGISTRY["maximin"](rnd, q_a, score)
            board_s = time.time() - t
            view = responders.build_view(board, rnd)
            probe = bound._probe_seconds(view, rnd)
            n_plans = partial_module.count_candidates(view, q_b)
            rows.append((board_s, n_plans * probe))
        per_split[g] = rows
        sys.stderr.write("\r  sampled A=%d (%d rounds)   " % (g, len(rows)))
        sys.stderr.flush()
    sys.stderr.write("\n")

    print("estimated cost per round = measured board + n_plans * measured probe")
    print("%8s | %9s | %s" % ("deadline", "coverage", "core-hours (all splits)"))
    for d in (10, 30, 60, 120, 300, 600, 1800):
        hours = 0.0
        covered = done = 0
        for g, rows in per_split.items():
            mean = sum(b + (e if e <= d else 0.0) for b, e in rows) / len(rows)
            hours += total * mean / 3600.0
            covered += sum(1 for _b, e in rows if e <= d)
            done += len(rows)
        print("%8ds | %8.0f%% | %8.1f%s"
              % (d, 100.0 * covered / done, hours,
                 "   <= fits 5h x 8 = 40" if hours <= 40 else ""))
    print()
    for g, rows in sorted(per_split.items()):
        est = sorted(e for _b, e in rows)
        print("  A=%3d  board mean %.2fs  est solve: median %.2gs  p75 %.2gs  p90 %.2gs"
              % (g, sum(b for b, _e in rows) / len(rows), est[len(est) // 2],
                 est[int(0.75 * len(est))], est[int(0.9 * len(est))]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
