"""Runs the fleet-split sweep over the solved rounds and writes the images."""

import argparse
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import plots
import responders
import rounds as rounds_module
import strategies
import sweep as sweep_module


def main():
    """Parses arguments, runs the sweep and writes its outputs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=249)
    parser.add_argument("--strategy", default="greedy",
                        choices=sorted(strategies.REGISTRY))
    parser.add_argument("--responder", default="full",
                        choices=sorted(responders.REGISTRY))
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pool", default=None)
    parser.add_argument("--rounds", default=None)
    parser.add_argument("--out", default=os.path.join(_HERE, "out"))
    args = parser.parse_args()
    assert args.n >= 4 and args.step > 0

    all_rounds = rounds_module.load(args.pool, args.rounds)
    if args.limit:
        all_rounds = all_rounds[:args.limit]
    started = time.time()

    def tick(result):
        sys.stderr.write("\r  A=%d B=%d  A %.4f  B %.4f  (%.0fs)   "
                         % (result["g"], result["o"], result["a_mean"],
                            result["b_mean"], time.time() - started))
        sys.stderr.flush()

    result = sweep_module.sweep(args.n, args.strategy, args.responder,
                                all_rounds, args.seed, args.step, tick)
    sys.stderr.write("\n")
    assert result

    os.makedirs(args.out, exist_ok=True)
    tag = "%s_%s" % (args.strategy, args.responder)
    sweep_module.save(result, os.path.join(args.out, "sweep_%s.json" % tag))
    made = plots.plot_all(result, args.out, tag, args.n, len(all_rounds))

    print("splits %d   rounds %d   elapsed %.1fs"
          % (len(result), len(all_rounds), time.time() - started))
    print("%8s %8s %10s %10s %10s"
          % ("A", "B", "A mean", "B mean", "margin"))
    for row in result[::max(1, len(result) // 12)]:
        print("%8d %8d %10.4f %10.4f %+10.4f"
              % (row["g"], row["o"], row["a_mean"], row["b_mean"],
                 row["b_mean"] - row["a_mean"]))
    for path in made:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
