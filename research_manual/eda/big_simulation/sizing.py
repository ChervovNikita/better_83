"""Estimates exact-partial coverage against a wall-clock budget."""

import argparse
import os
import random
import statistics
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import partial as partial_module
import responders
import rounds as rounds_module
import strategies
from scoring import score

BUDGETS = (1e4, 1e5, 1e6, 1e7, 1e8, 1e9)


def main():
    """Samples rounds per split and prints coverage against a time budget."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", default="135,140,150,175,200")
    parser.add_argument("--sample", type=int, default=40)
    parser.add_argument("--n", type=int, default=249)
    parser.add_argument("--hours", type=float, default=6.0)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--rate-us", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    all_rounds = rounds_module.load()
    total_rounds = len(all_rounds)
    rate = args.rate_us * 1e-6
    splits = [int(x) for x in args.splits.split(",")]
    rng = random.Random(args.seed)
    per_split = {}

    for g in splits:
        o = args.n - g
        sample = rng.sample(all_rounds, args.sample)
        counts, board_times = [], []
        for rnd in sample:
            q_a, q_b = rounds_module.split_queries(rnd, g, o)
            rnd.q_a, rnd.fleet_a, rnd.fleet_b = q_a, g, o
            if q_a == 0 or q_b == 0:
                continue
            t = time.time()
            board = strategies.REGISTRY["maximin"](rnd, q_a, score)
            board_times.append(time.time() - t)
            view = responders.build_view(board, rnd)
            counts.append(partial_module.count_candidates(view, q_b))
        per_split[g] = (counts, statistics.mean(board_times))
        sys.stderr.write("\r  sampled A=%d (%d rounds, board %.2fs)      "
                         % (g, len(counts), per_split[g][1]))
        sys.stderr.flush()
    sys.stderr.write("\n")

    wall = args.hours * 3600 * args.workers
    print("budget: %.1f h wall x %d workers = %.0f core-seconds"
          % (args.hours, args.workers, wall))
    print("cost model: %.0f us per candidate, %d rounds per split, splits %s"
          % (args.rate_us, total_rounds, splits))
    print()
    print("%10s | %8s | %s" % ("cap", "coverage", "core-hours"))
    for cap in BUDGETS:
        total = 0.0
        covered = done = 0
        for g in splits:
            counts, board_t = per_split[g]
            solve = statistics.mean(min(c, cap) for c in counts) * rate
            total += total_rounds * (board_t + solve)
            covered += sum(1 for c in counts if c <= cap)
            done += len(counts)
        flag = "  <= FITS" if total <= wall else ""
        print("%10.0e | %7.0f%% | %10.1f%s"
              % (cap, 100.0 * covered / done, total / 3600.0, flag))
    print()
    print("A-board cost alone: %.1f core-hours (unavoidable, all splits)"
          % (sum(total_rounds * per_split[g][1] for g in splits) / 3600.0))
    for g in splits:
        counts, board_t = per_split[g]
        counts = sorted(counts)
        print("  A=%3d board %.2fs  count median %.2g  p90 %.2g  max %.2g"
              % (g, board_t, counts[len(counts) // 2],
                 counts[int(0.9 * len(counts))], counts[-1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
