"""Holds A's board fixed and varies only B, isolating the responder's effect."""

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
import native
import partial as partial_module
import responders
import rounds as rounds_module
import strategies
from scoring import score


def main():
    """Scores one A board against both the hybrid and the full-sight responder."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--g", type=int, default=150)
    parser.add_argument("--n", type=int, default=249)
    parser.add_argument("--deadline", type=float, default=60.0)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--nshards", type=int, default=1)
    parser.add_argument("--seconds", type=int, default=2400)
    parser.add_argument("--out", default=os.path.join(_HERE, "out"))
    args = parser.parse_args()

    all_rounds = rounds_module.load()
    o = args.n - args.g
    rng = random.Random(0)
    started = time.time()
    rows = []
    mine = [r for i, r in enumerate(all_rounds)
            if i % args.nshards == args.shard]
    for rnd in mine:
        if time.time() - started > args.seconds:
            break
        q_a, q_b = rounds_module.split_queries(rnd, args.g, o)
        rnd.q_a, rnd.fleet_a, rnd.fleet_b = q_a, args.g, o
        rnd.q_b_oracle = q_b
        if hasattr(rnd, "q_b"):
            del rnd.q_b
        board = strategies.REGISTRY["maximin"](rnd, q_a, score)
        rnd.q_b = q_b
        if q_b == 0:
            continue
        w_a, w_b = q_a / float(args.g), q_b / float(o)

        # B = full sight, on this exact board
        _t, full_a, full_b = native.best_response_weighted(
            board, rnd, q_b, w_a, w_b)

        # B = the hybrid, on the SAME board
        view = responders.build_view(board, rnd)
        probe = bound._probe_seconds(view, rnd, q_b)
        n_plans = partial_module.count_candidates(view, q_b)
        path = "full"
        hyb_a, hyb_b = full_a, full_b
        if (probe <= args.deadline / 1000.0
                and n_plans * probe <= args.deadline):
            result = partial_module.best_response(
                view, rnd.difficulty, q_b, rnd.omega, fleet_a=args.g,
                fleet_b=o, deadline=time.time() + args.deadline)
            if result is not None:
                trial = partial_module.realise(result[1], rng)
                hyb_a, hyb_b = score(trial, rnd.difficulty)
                path = "partial"
        rows.append((q_a, q_b, full_a, full_b, hyb_a, hyb_b, path))

    path = os.path.join(args.out, "ab_%d_%02d.json" % (args.g, args.shard))
    with open(path, "w") as handle:
        json.dump({"g": args.g, "o": o, "rows": rows}, handle)

    exact = [r for r in rows if r[6] == "partial"]
    print("A=%d B=%d  %d rounds, %d answered by exact partial"
          % (args.g, o, len(rows), len(exact)))
    if exact:
        # On the SAME board a weaker B cannot help itself: B maximises
        # w_b*mean_b - w_a*mean_a, so the hybrid's value must not exceed the
        # full-sight value. A violation is a bug, not a finding about the game.
        viol = [r for r in exact
                if (r[5] * (r[1] / float(args.n - args.g))
                    - r[4] * (r[0] / float(args.g)))
                > (r[3] * (r[1] / float(args.n - args.g))
                   - r[2] * (r[0] / float(args.g))) + 1e-9]
        print("  hybrid ABOVE full-sight on B's own objective: %d/%d %s"
              % (len(viol), len(exact), "<-- BUG" if viol else "(none, as required)"))
        qa = sum(r[0] for r in rows)
        qb = sum(r[1] for r in rows)
        pf = (sum(r[0] * r[2] for r in rows) / qa
              - sum(r[1] * r[3] for r in rows) / qb)
        ph = (sum(r[0] * r[4] for r in rows) / qa
              - sum(r[1] * r[5] for r in rows) / qb)
        print("  pooled A-B vs full  : %+.5f" % pf)
        print("  pooled A-B vs hybrid: %+.5f" % ph)
        print("  hybrid minus full   : %+.5f  (must be >= 0)" % (ph - pf))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
