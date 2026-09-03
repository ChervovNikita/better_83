"""Measures the fast partial responder against the exact one where affordable."""

import argparse
import os
import random
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


def main():
    """Compares fast and exact partial responses on affordable rounds."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--g", type=int, default=150)
    parser.add_argument("--n", type=int, default=249)
    parser.add_argument("--budget", type=int, default=20000)
    parser.add_argument("--strategy", default="even")
    parser.add_argument("--seconds", type=int, default=1500)
    args = parser.parse_args()

    all_rounds = rounds_module.load()
    random.Random(0).shuffle(all_rounds)
    o = args.n - args.g
    started = time.time()
    gaps = []
    skipped = 0
    for rnd in all_rounds:
        if time.time() - started > args.seconds:
            break
        q_a, q_b = rounds_module.split_queries(rnd, args.g, o)
        if q_a == 0 or q_b == 0:
            continue
        rnd.q_a, rnd.fleet_a, rnd.fleet_b = q_a, args.g, o
        board = strategies.REGISTRY[args.strategy](rnd, q_a, score)
        view = responders.build_view(board, rnd)
        if partial_module.count_candidates(view, q_b) > args.budget:
            skipped += 1
            continue
        fast = partial_module.best_response_fast(view, rnd.difficulty, q_b,
                                                 rnd.omega)[0]
        exact = partial_module.best_response(view, rnd.difficulty, q_b,
                                             rnd.omega)[0]
        gaps.append(exact - fast)
        if len(gaps) % 10 == 0:
            worse = sum(1 for g in gaps if g > 1e-9)
            sys.stderr.write("\r  %d compared, %d skipped, %d where exact wins "
                             "(%.0fs)  " % (len(gaps), skipped, worse,
                                            time.time() - started))
            sys.stderr.flush()
    sys.stderr.write("\n")
    worse = [g for g in gaps if g > 1e-9]
    print("A=%d B=%d  strategy=%s  budget=%d candidates"
          % (args.g, o, args.strategy, args.budget))
    print("compared %d rounds, skipped %d as too large (%.0f%% covered)"
          % (len(gaps), skipped, 100.0 * len(gaps) / max(1, len(gaps) + skipped)))
    print("exact strictly better on %d/%d rounds" % (len(worse), len(gaps)))
    if worse:
        worse.sort()
        print("  gap: median %.9f  max %.9f" % (worse[len(worse) // 2], worse[-1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
