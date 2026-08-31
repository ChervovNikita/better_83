"""Lower-bounds A's margin against a partial-information opponent.

Per round the opponent answers with the EXACT partial best response when the
plan count is affordable, and with the full-sight best response otherwise.
Full sight dominates partial information -- it can reproduce any multiset the
partial player picks and then choose the matching instead of drawing one -- so
the hybrid is everywhere at least as strong as a true partial oracle, and the
margin it concedes to A is a lower bound on A's real margin.
"""

import argparse
import json
import os
import random
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import native
import partial as partial_module
import responders
import rounds as rounds_module
import strategies
from scoring import score


def _probe_seconds(view, rnd):
    """Returns the measured cost of evaluating one plan on this view."""
    sizes = sorted(view, reverse=True)
    plan = {"fresh": []}
    for size in sizes:
        counts = view[size]["counts"]
        if counts:
            plan[size] = (counts, [1] + [0] * (len(counts) - 1))
    if len(plan) == 1:
        return 0.0
    started = time.time()
    native.expected_scores(plan, rnd.difficulty, rnd.omega)
    return max(time.time() - started, 1e-6)


def play(rnd, g, o, cap, rng, strategy, deadline):
    """Returns one round's answer counts, means, and which solver replied."""
    q_a, q_b = rounds_module.split_queries(rnd, g, o)
    rnd.q_a, rnd.fleet_a, rnd.fleet_b = q_a, g, o
    rnd.q_b_oracle = q_b
    if hasattr(rnd, "q_b"):
        del rnd.q_b
    board = strategies.REGISTRY[strategy](rnd, q_a, score)
    rnd.q_b = q_b
    if q_b == 0:
        mean_a, _mean_b = score(board, rnd.difficulty)
        return q_a, q_b, mean_a, 0.0, "none"

    view = responders.build_view(board, rnd)
    # Cost per candidate is NOT uniform: expected_scores enumerates contingency
    # tables, and one call runs from 20us to seconds depending on how many
    # distinct occupancy values the view carries. So time one representative
    # plan and multiply by the exact plan count, rather than assuming a rate.
    # A single candidate can outlast the deadline on its own, which is why the
    # in-search deadline alone could not bound the round.
    probe = _probe_seconds(view, rnd)
    # The cap only skips rounds too large to be worth starting; the deadline is
    # what actually bounds the run, and a missed deadline falls back to the
    # full-sight solver, which is the stronger opponent.
    result = None
    n_plans = partial_module.count_candidates(view, q_b)
    if n_plans * probe <= deadline and n_plans <= cap:
        result = partial_module.best_response(
            view, rnd.difficulty, q_b, rnd.omega, fleet_a=g, fleet_b=o,
            deadline=time.time() + deadline)
    if result is not None:
        trial = partial_module.realise(result[1], rng)
        mean_a, mean_b = score(trial, rnd.difficulty)
        return q_a, q_b, mean_a, mean_b, "partial"

    _trial, mean_a, mean_b = native.best_response_weighted(
        board, rnd, q_b, q_a / float(g), q_b / float(o))
    return q_a, q_b, mean_a, mean_b, "full"


def main():
    """Runs one split and writes its per-round rows."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--g", type=int, required=True)
    parser.add_argument("--n", type=int, default=249)
    parser.add_argument("--cap", type=float, default=1e9)
    parser.add_argument("--deadline", type=float, default=30.0)
    parser.add_argument("--strategy", default="maximin")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--nshards", type=int, default=1)
    parser.add_argument("--out", default=os.path.join(_HERE, "out"))
    args = parser.parse_args()

    all_rounds = rounds_module.load()
    if args.limit:
        all_rounds = all_rounds[:args.limit]
    o = args.n - args.g
    rng = random.Random(args.seed)
    started = time.time()
    rows = []
    # Shard by round, not by split: the near-even splits cost an order of
    # magnitude more than the lopsided ones, so one worker per split would be
    # bounded by the slowest split rather than by the total work.
    mine = [r for i, r in enumerate(all_rounds) if i % args.nshards == args.shard]
    for index, rnd in enumerate(mine):
        rows.append(play(rnd, args.g, o, int(args.cap), rng, args.strategy,
                         args.deadline))
        if (index + 1) % 25 == 0:
            done = sum(1 for r in rows if r[4] == "partial")
            sys.stderr.write("\r  %d/%d rounds, %d exact-partial (%.0f%%), "
                             "%.0fs  " % (index + 1, len(mine), done,
                                          100.0 * done / len(rows),
                                          time.time() - started))
            sys.stderr.flush()
    sys.stderr.write("\n")

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "bound_%s_%d_%02d.json"
                        % (args.strategy, args.g, args.shard))
    with open(path, "w") as handle:
        json.dump({"g": args.g, "o": o, "cap": args.cap,
                   "strategy": args.strategy, "rows": rows}, handle)
    covered = sum(1 for r in rows if r[4] == "partial")
    qa = sum(r[0] for r in rows)
    qb = sum(r[1] for r in rows)
    pooled = (sum(r[0] * r[2] for r in rows) / qa
              - sum(r[1] * r[3] for r in rows) / qb)
    print("A=%d B=%d  %s  cap=%.0e" % (args.g, o, args.strategy, args.cap))
    print("  %d rounds, %d exact-partial (%.0f%%), %d full fallback"
          % (len(rows), covered, 100.0 * covered / len(rows),
             sum(1 for r in rows if r[4] == "full")))
    print("  pooled A - B = %+.5f   (%.0f min)"
          % (pooled, (time.time() - started) / 60.0))
    print("  wrote %s" % path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
