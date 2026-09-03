"""Runs one fleet split through both strategies and both responders."""

import argparse
import json
import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import responders
import rounds as rounds_module
import strategies
from scoring import score

STRATEGIES = ("greedy", "maximin")
RESPONDERS = ("full_pooled", "partial")


def run(g, o, all_rounds, seed):
    """Returns per-round means and answer counts for every strategy pair."""
    out = {}
    for name in STRATEGIES:
        commit = strategies.REGISTRY[name]
        boards, qs = [], []
        for rnd in all_rounds:
            q_a, q_b = rounds_module.split_queries(rnd, g, o)
            rnd.q_a, rnd.fleet_a, rnd.fleet_b = q_a, g, o
            rnd.q_b_oracle = q_b
            if hasattr(rnd, "q_b"):
                del rnd.q_b
            boards.append(commit(rnd, q_a, score))
            qs.append((q_a, q_b))
        for responder in RESPONDERS:
            reply = responders.REGISTRY[responder]
            rng = random.Random(seed)
            rows = []
            for rnd, board, (q_a, q_b) in zip(all_rounds, boards, qs):
                rnd.q_a, rnd.q_b = q_a, q_b
                rnd.fleet_a, rnd.fleet_b = g, o
                _trial, mean_a, mean_b = reply(board, rnd, q_b, rng)
                rows.append((q_a, q_b, mean_a, mean_b))
            out["%s_%s" % (name, responder)] = rows
    return out


def main():
    """Runs one split and writes its per-round rows."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--g", type=int, required=True)
    parser.add_argument("--n", type=int, default=249)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=os.path.join(_HERE, "out"))
    args = parser.parse_args()
    all_rounds = rounds_module.load()
    result = run(args.g, args.n - args.g, all_rounds, args.seed)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "split_%d.json" % args.g)
    with open(path, "w") as handle:
        json.dump({"g": args.g, "o": args.n - args.g, "rows": result}, handle)
    print("wrote %s" % path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
