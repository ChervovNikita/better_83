"""Certifies A's board against exhaustive search on rounds that can afford it."""

import argparse
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import exhaustive
import native
import rounds as rounds_module
import strategies


def main():
    """Compares maximin to exhaustive search over every A board."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--g", type=int, default=125)
    parser.add_argument("--n", type=int, default=249)
    parser.add_argument("--budget", type=int, default=20000)
    parser.add_argument("--seconds", type=int, default=3000)
    args = parser.parse_args()
    all_rounds = rounds_module.load()
    o = args.n - args.g
    started = time.time()
    done = skipped = below = 0
    worst = 0.0
    boards = 0
    for rnd in all_rounds:
        if time.time() - started > args.seconds:
            break
        q_a, q_b = rounds_module.split_queries(rnd, args.g, o)
        rnd.q_a, rnd.fleet_a, rnd.fleet_b = q_a, args.g, o
        if q_a == 0:
            continue
        got = exhaustive.best_a_board(rnd, q_a, q_b, args.budget)
        if got is None:
            skipped += 1
            continue
        best, _board, count = got
        boards += count
        ours = native.maximin(rnd, q_a, strategies.estimate_q_b(rnd))
        _t, mean_a, mean_b = native.best_response(ours, rnd, q_b)
        value = q_a / float(args.g) * mean_a - q_b / float(o) * mean_b
        done += 1
        if best - value > 1e-9:
            below += 1
            worst = max(worst, best - value)
        if done % 25 == 0:
            sys.stderr.write("\r  certified %d, skipped %d, below %d (%.0fs)  "
                             % (done, skipped, below, time.time() - started))
            sys.stderr.flush()
    sys.stderr.write("\n")
    print("A=%d B=%d  budget %d boards/round" % (args.g, o, args.budget))
    print("certified %d rounds (%d boards), skipped %d as too large"
          % (done, boards, skipped))
    print("maximin below exhaustive on %d/%d, worst %.9f" % (below, done, worst))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
