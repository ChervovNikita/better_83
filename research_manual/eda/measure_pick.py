#!/usr/bin/env python3
"""Score picker variants against the validator's own calculator.

    .venv/bin/python research_manual/eda/measure_pick.py

Reads pools.json (written by fit_field.py --build), so it re-scores in seconds
instead of re-solving for 21 minutes.  Arms:

    shipped   fleet_pick.picker, SPARE_CAP = 2
    static    pick_static.picker, hold back when P < q
    oracle    oracle_pick.oracle_slots -- reads the field, upper bound only

fleet_pick's own comment records that converting the whole shortfall to spares
measured worse than converting a little.  This is the measurement that says
whether that holds once the choice is conditioned on P.
"""

import argparse
import collections
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
ROOT = os.path.dirname(PARENT)
for _p in (ROOT, PARENT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

POOLS = os.path.join(HERE, "pools.json")
TUNING = os.path.join(HERE, "tuning_data.json")


class _Graph(object):
    def __init__(self, n, adjacency_list):
        self.number_of_nodes = n
        self.adjacency_list = adjacency_list


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", default=POOLS)
    ap.add_argument("--dump", default=TUNING)
    ap.add_argument("--fleet", type=int, default=40)
    ap.add_argument("--limit", type=int, default=0, help="first N rounds only")
    ap.add_argument("--estimate-difficulty", action="store_true",
                    help="hide difficulty from pick_value, as solver.solve does, "
                         "so it must estimate p from its own queried count")
    ap.add_argument("--victims", type=int, default=0,
                    help="recompute the field after dropping this many miners; "
                         "pools.json caches it for 40, and entering with N "
                         "hotkeys displaces N, so the field differs by fleet")
    ap.add_argument("--beta", type=float, nargs="*",
                    default=[0.0, 0.25, 0.5, 1.0, 2.0],
                    help="basin-size exponents to sweep for the hits arm")
    ap.add_argument("--fleet-size", type=int, default=0,
                    help="override pick_value's N_FLEET, to test other fleets")
    args = ap.parse_args()

    from CliqueAI.graph.codec import GraphCodec
    from CliqueAI.scoring.clique_scoring import CliqueScoreCalculator

    class Memo(CliqueScoreCalculator):
        """The validator's calculator with validity memoised per round.

        is_valid_maximum_clique is O(n*k) and the base class calls it inside both
        optimality() and diversity(), for every arm -- so the field's answers get
        revalidated 8x a round. The logic is inherited untouched; only repeat
        work is skipped.
        """

        cache = {}

        def is_valid_maximum_clique(self, nodes):
            key = tuple(sorted(nodes))
            hit = Memo.cache.get(key)
            if hit is None:
                hit = CliqueScoreCalculator.is_valid_maximum_clique(self, nodes)
                Memo.cache[key] = hit
            return hit
    import fleet_pick
    import pick_static
    import pick_value
    import strategy
    from oracle_pick import oracle_slots

    # Field-blind prediction of how many other answers reach omega: the step
    # model fitted on tuning (a/n_others = 0.06 below n_top 5, 1.00 above).
    def predict_a(n_top, n_others):
        return (0.06 if n_top <= 5 else 1.00) * n_others

    def nodup_slots(pool, q):
        """Always prefer omega, but never repeat: fill the rest with spares.

        This is what the deployed path actually did, because solve_many returns
        at most k cliques and the picker backfilled with the omega cliques it
        meant to decline. Isolates "stop self-colliding" from "hold back".
        """
        omega = max(len(c) for c in pool)
        top = [c for c in pool if len(c) == omega]
        spare = sorted((c for c in pool if len(c) < omega), key=len, reverse=True)
        use = list(top[:q])
        use.extend(spare[:max(0, q - len(use))])
        i = 0
        while len(use) < q:
            use.append(use[i % len(use)])
            i += 1
        return use[:q]

    if args.fleet_size:
        pick_value.N_FLEET = args.fleet_size
    refield = None
    if args.victims:
        import fit_field
        meta = json.load(open(os.path.join(PARENT, "metagraph.json")))
        dropped = fit_field.victims(meta, args.victims)
        refield = dropped
    with open(args.pools) as handle:
        pools = json.load(handle)
    with open(args.dump) as handle:
        payload = json.load(handle)
    codec = GraphCodec()

    arms = collections.defaultdict(list)
    dup = collections.Counter()
    cols = ("round", "q", "n_top", "n_spare", "shipped", "nodup", "static",
            "value", "greedy", "oracle")
    fmt = "%-14s %3s %6s %7s  %8s %8s %8s %8s %8s %8s"
    print(fmt % cols)
    print(fmt % tuple("-" * len(c) for c in cols))
    ordered = sorted(pools.items(), key=lambda kv: payload[kv[0]]["timestamp"])
    if args.limit:
        ordered = ordered[:args.limit]
    for rid, cached in ordered:
        rec = payload[rid]
        if refield is None:
            field = [tuple(c) for c in cached["field"]]
        else:
            field = [tuple(sorted(cl)) for uid, _h, _c, cl in rec["answers"]
                     if cl and uid not in refield]
        top = [tuple(c) for c in cached["top"]]
        spare = [tuple(c) for c in cached["spare"]]
        if not field or not top:
            continue
        pool = top + spare
        p = 1.0 - np.exp(-max(0.0, np.sqrt(2.5) - rec["difficulty"] - 0.5))
        q = max(1, int(round(args.fleet * p)))

        matrix = codec.decode_matrix(rec["encoded_matrix"])
        graph = _Graph(rec["number_of_nodes"], codec.matrix_to_list(matrix))
        Memo.cache = {}                      # per round: the graph changes

        th = cached.get("top_hits")
        sh = cached.get("spare_hits")
        picks = {
            "shipped": fleet_pick.slots(pool, list(range(q))),
            "nodup": nodup_slots(pool, q),
            "static": pick_static.slots(pool, list(range(q))),
            "value": pick_value.slots(
                pool, list(range(q)),
                difficulty=None if args.estimate_difficulty else rec["difficulty"]),
            "greedy": strategy.slots(pool, q, len(field),
                                     predict_a(len(top), len(field)),
                                     rec["difficulty"],
                                     b_hat=len(field) - predict_a(len(top), len(field))),
            "oracle": oracle_slots(pool, field, q, rec["difficulty"]),
        }
        if th is not None and sh is not None:
            n_others = len(field)
            a_hat = (0.06 if len(top) <= 5 else 1.00) * n_others
            for b in args.beta:
                picks["b=%g" % b] = strategy.slots_hits(
                    pool, list(th) + list(sh), q, n_others, a_hat,
                    rec["difficulty"], b_hat=n_others - a_hat, beta=b)
        row = {}
        for name, sel in picks.items():
            calc = Memo(
                graph=graph, difficulty=rec["difficulty"],
                responses=[list(c) for c in field] + [list(c) for c in sel])
            *_, rewards = calc.get_scores()
            row[name] = float(np.mean(rewards[len(field):]))
            arms[name].append(row[name])
            if len(set(tuple(sorted(c)) for c in sel)) < len(sel):
                dup[name] += 1
        print("%-14s %3d %6d %7d  %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f"
              % ("n=%d tl=%g" % (rec["number_of_nodes"], rec["time_limit"]),
                 q, len(top), len(spare), row["shipped"], row["nodup"],
                 row["static"], row["value"], row["greedy"], row["oracle"]),
              flush=True)

    n = len(arms["shipped"])
    print()
    print("%-10s %8s %10s %10s" % ("arm", "mean", "vs shipped", "rounds w/ dup"))
    base = np.array(arms["shipped"])
    order = ["shipped", "nodup", "static", "value", "greedy"]
    order += ["b=%g" % b for b in args.beta if ("b=%g" % b) in arms]
    order += ["oracle"]
    for name in order:
        v = np.array(arms[name])
        print("%-10s %8.4f %+10.4f %10d"
              % (name, v.mean(), v.mean() - base.mean(), dup[name]))

    changed = [(a, b) for a, b in zip(arms["static"], arms["value"])
               if abs(a - b) > 1e-9]
    better = sum(1 for a, b in changed if b > a)
    worse = len(changed) - better
    print()
    print("value vs STATIC: %d changed rounds, %d better / %d worse"
          % (len(changed), better, worse))
    if changed:
        from math import comb
        k = better + worse
        pv = sum(comb(k, i) for i in range(min(better, worse) + 1)) / 2.0 ** k * 2
        print("two-sided sign test over changed rounds: p = %.4g" % pv)
    print("rounds %d" % n)


if __name__ == "__main__":
    main()
