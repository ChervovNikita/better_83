#!/usr/bin/env python3
"""Optimal fleet allocation given the field's answers.

    oracle_slots(pool, field, q, difficulty) -> list of q cliques

This is an ORACLE: it reads the field's submissions, which a miner cannot do at
solve time. It exists to put a ceiling on any implementable picker -- run it and
the shipped picker on the same rounds, and the gap is the headroom a real
heuristic could compete for.

    .venv/bin/python research_manual/eda/oracle_pick.py --rounds 60

Rule.  With difficulty D, all answers R, max valid size M:

    rel(s) = s / M
    pr(s)  = |{answers strictly larger than s}| / |R|
    opt(s) = exp(-pr(s) / rel(s))

max_omega is 1 whenever any answer reaches M, so optimality is opt(s) as is.
With f(c) field miners on clique c and m of ours, count(c) = f+m and our total
diversity from c is m/(f+m).  Maximising

    sum_c [ m_c * opt(s(c)) * (1+D) + m_c/(f(c)+m_c) ],   sum_c m_c = q

is a separable concave problem under a cardinality constraint, so greedy on the
marginal is exactly optimal.

opt(s) depends on the sizes WE submit, so the assignment is iterated to a fixed
point.  That feedback is why holding back compounds: vacate omega when we were
its only finder and M drops, every omega-1 answer gets pr = 0, and the optimality
cost of holding back goes to zero.
"""

import argparse
import collections
import heapq
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
ROOT = os.path.dirname(PARENT)
for _p in (ROOT, PARENT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

TUNING = os.path.join(HERE, "tuning_data.json")


def _delta_div(f, m):
    """Diversity gained by the m-th of our hotkeys on a clique f others hold."""
    if f == 0:
        return 1.0 if m == 1 else 0.0
    return f / float((f + m) * (f + m - 1))


def _opt_table(sizes, n_answers, max_size):
    """opt(s) for every size present, from the full answer multiset."""
    counts = collections.Counter(sizes)
    larger = {}
    running = 0
    for s in sorted(counts, reverse=True):
        larger[s] = running
        running += counts[s]
    return {s: math.exp(-(larger[s] / float(n_answers)) / (s / float(max_size)))
            for s in counts}


def _greedy(pool, held, q, difficulty, opt_of):
    """Optimal allocation for a FIXED opt(s), by the concave-marginal argument."""
    heap = []
    for c in pool:
        base = opt_of(len(c))
        heapq.heappush(heap, (-(base * (1.0 + difficulty) + _delta_div(held[c], 1)),
                              c, base))
    out = []
    taken = collections.Counter()
    while heap and len(out) < q:
        _neg, c, base = heapq.heappop(heap)
        out.append(c)
        taken[c] += 1
        heapq.heappush(heap, (-(base * (1.0 + difficulty)
                                + _delta_div(held[c], taken[c] + 1)), c, base))
    while len(out) < q:
        out.append(out[len(out) % len(pool)])
    return out


def _fleet_value(alloc, field, held, difficulty):
    """Mean reward of our answers under CliqueScoreCalculator's exact formula."""
    sizes = [len(c) for c in field] + [len(c) for c in alloc]
    n = len(sizes)
    max_size = max(sizes)
    counts = collections.Counter(field)
    counts.update(alloc)
    bigger = {s: sum(1 for x in sizes if x > s) for s in set(sizes)}
    omega = {s: math.exp(-(bigger[s] / float(n)) / (s / float(max_size)))
             for s in set(sizes)}
    max_omega = max(omega.values())
    max_delta = max(1.0 / counts[c] for c in counts)
    total = 0.0
    for c in alloc:
        total += (omega[len(c)] / max_omega) * (1.0 + difficulty) \
                 + (1.0 / counts[c]) / max_delta
    return total / len(alloc)


def oracle_slots(pool, field, q, difficulty, passes=6):
    """The q cliques our fleet should submit, knowing the field exactly.

    pool  : our distinct valid maximal cliques, as sorted tuples
    field : the field's answers, as sorted tuples (assumed valid)

    The opt(s) <-> allocation coupling has several fixed points and the map is
    self-confirming -- seeding it with "everyone at omega" inflates pr for every
    smaller size, which then justifies staying at omega.  So it is iterated from
    one seed per distinct size in the pool and the exactly-scored best is kept.
    """
    assert pool and q > 0
    pool = list(dict.fromkeys(tuple(c) for c in pool))
    field = [tuple(c) for c in field]
    held = collections.Counter(field)
    field_sizes = [len(c) for c in field]

    best_alloc, best_value = None, -1.0
    for seed_size in sorted({len(c) for c in pool}, reverse=True):
        seed_pool = [c for c in pool if len(c) == seed_size]
        chosen = [seed_pool[i % len(seed_pool)] for i in range(q)]
        for _ in range(passes):
            sizes = field_sizes + [len(c) for c in chosen]
            max_size = max(sizes)
            table = _opt_table(sizes, len(sizes), max_size)

            def opt_of(s, _t=table, _sz=sizes, _m=max_size):
                if s in _t:
                    return _t[s]
                bigger = sum(1 for x in _sz if x > s)
                return math.exp(-(bigger / float(len(_sz))) / (s / float(_m)))

            nxt = _greedy(pool, held, q, difficulty, opt_of)
            if nxt == chosen:
                break
            chosen = nxt
        value = _fleet_value(chosen, field, held, difficulty)
        if value > best_value:
            best_alloc, best_value = chosen, value
    return [list(c) for c in best_alloc]


# --------------------------------------------------------------------- eval

def _score(graph, difficulty, responses):
    from CliqueAI.scoring.clique_scoring import CliqueScoreCalculator
    calc = CliqueScoreCalculator(graph=graph, difficulty=difficulty,
                                 responses=[list(r) for r in responses])
    *_, rewards = calc.get_scores()
    return rewards


class _Graph(object):
    def __init__(self, n, adjacency_list):
        self.number_of_nodes = n
        self.adjacency_list = adjacency_list


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default=TUNING)
    ap.add_argument("--rounds", type=int, default=40)
    ap.add_argument("--fleet", type=int, default=40)
    args = ap.parse_args()

    from CliqueAI.graph.codec import GraphCodec
    import fleet_pick
    import fleet_solver_gpu as fg
    import gpu_lib

    codec = GraphCodec()
    with open(args.dump) as handle:
        payload = json.load(handle)
    rows = sorted(payload.items(), key=lambda kv: kv[1]["timestamp"])[:args.rounds]

    ship, oracle = [], []
    print("%-22s %3s %6s %7s %9s %9s %8s"
          % ("round", "k", "omega", "pool", "shipped", "oracle", "gain"))
    for _rid, rec in rows:
        matrix = codec.decode_matrix(rec["encoded_matrix"])
        A = np.array(matrix, dtype=np.uint8)
        graph = _Graph(rec["number_of_nodes"], codec.matrix_to_list(matrix))
        field = [tuple(sorted(cl)) for _u, _h, _c, cl in rec["answers"] if cl]
        if not field:
            continue
        p = 1.0 - np.exp(-max(0.0, np.sqrt(2.5) - rec["difficulty"] - 0.5))
        q = max(1, int(round(args.fleet * p)))

        budget = rec["time_limit"] - 2.0
        champ = sorted(fg.fleet_solver._solve_one(A, budget * fg.CHAMPION_SHARE,
                                                  seed=1))
        with gpu_lib.GpuClique(A) as gpu:
            raw, _ = gpu.harvest(budget * (1 - fg.CHAMPION_SHARE) - fg.RESERVE_S,
                                 seed=1, max_steps=fg.STEPS,
                                 boot_steps=fg.BOOT_STEPS, init_clique=champ,
                                 max_out=4096)
        pool = [tuple(c) for c in raw if all(gpu_lib.verify(A, c))]
        pool = sorted(set(pool), key=len, reverse=True)
        if not pool:
            continue

        a = fleet_pick.slots(pool, list(range(q)))
        b = oracle_slots(pool, field, q, rec["difficulty"])
        ra = float(np.mean(_score(graph, rec["difficulty"], field + a)[len(field):]))
        rb = float(np.mean(_score(graph, rec["difficulty"], field + b)[len(field):]))
        ship.append(ra)
        oracle.append(rb)
        om = len(pool[0])
        print("n=%3d tl=%4.1f %6d %6d %7d %9.4f %9.4f %8.4f"
              % (rec["number_of_nodes"], rec["time_limit"], q, om,
                 sum(1 for c in pool if len(c) == om), ra, rb, rb - ra),
              flush=True)

    print()
    print("rounds %d | shipped %.4f | oracle %.4f | headroom %+.4f per answer"
          % (len(ship), np.mean(ship), np.mean(oracle),
             np.mean(oracle) - np.mean(ship)))
    wins = sum(1 for a, b in zip(ship, oracle) if b > a + 1e-9)
    print("oracle better on %d of %d rounds" % (wins, len(ship)))


if __name__ == "__main__":
    main()
