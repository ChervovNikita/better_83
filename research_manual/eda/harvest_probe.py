#!/usr/bin/env python3
"""Instrumentation for the omega-clique harvest.

Answers two questions the pool files alone cannot:

  1. Once the device has found an omega-clique, what does it do next? The
     discovery curve (distinct omega vs budget) and the device's own counters
     say whether the remaining time buys new cliques or re-derives the same one.

  2. Are the omega-cliques we miss reachable from the ones we hold? A missed
     clique that differs from a held one by a single vertex is not a search
     failure -- it is a neighbourhood the harvest never enumerated, and closing
     it is deterministic and cheap.

Run as a script for a summary over research_manual/bench_d10.jsonl.
"""
import collections
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
REPO = os.path.dirname(PARENT)
for _p in (REPO, PARENT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from CliqueAI.graph.codec import GraphCodec

import fleet_solver_gpu

one_swap_closure = fleet_solver_gpu.one_swap_closure

ROUNDS = os.path.join(PARENT, "rounds.json")
BENCH = os.path.join(PARENT, "bench_d10.jsonl")


def adjacency(rec):
    A = np.ascontiguousarray(GraphCodec().decode_matrix(rec["encoded_matrix"]),
                             dtype=np.uint8)
    np.fill_diagonal(A, 0)
    return A


def is_clique(A, verts):
    v = list(verts)
    for i in range(len(v)):
        for j in range(i + 1, len(v)):
            if not A[v[i], v[j]]:
                return False
    return True


def curve(A, budgets, lanes=32, max_out=8192, champion=None):
    """Distinct omega found at each budget, with the device counters."""
    import gpu_lib
    gpu_lib.load(lanes, False)
    rows = []
    for b in budgets:
        t0 = time.time()
        with gpu_lib.GpuClique(A, lanes=lanes) as gpu:
            pool, ctr, hits = gpu.harvest(
                time_limit=b, seed=1, max_steps=20000, boot_steps=60000,
                max_steps_cap=1 << 20, spare_margin=1,
                init_clique=champion, max_out=max_out)
        keyed = {tuple(sorted(int(v) for v in c)): int(h)
                 for c, h in zip(pool, hits)}
        omega = max((len(c) for c in keyed), default=0)
        top = {c: h for c, h in keyed.items() if len(c) == omega}
        rows.append(dict(budget=b, wall=time.time() - t0, omega=omega,
                         n_top=len(top), n_pool=len(pool),
                         singletons=sum(1 for h in top.values() if h == 1),
                         stall=gpu_lib.stall(ctr), jobs=ctr["jobs"],
                         newmax=ctr["newmax"], dupmax=ctr["dupmax"],
                         overflow=ctr["overflow"]))
    return rows


def main():
    rounds = json.load(open(ROUNDS))
    rows = [json.loads(l) for l in open(BENCH)]
    tot = collections.Counter()
    per_round = []
    for r in rows:
        ours = {tuple(c) for c in r["ours_cliques"]}
        truth = {tuple(c) for c in r["truth_cliques"]}
        miss = truth - ours
        if not ours:
            continue
        A = adjacency(rounds[r["uuid"]])
        closure = one_swap_closure(A, ours, r["omega"])
        recovered = miss & closure
        tot["rounds"] += 1
        tot["missed"] += len(miss)
        tot["recovered"] += len(recovered)
        tot["closure"] += len(closure - ours)
        if miss and not (miss - recovered):
            tot["rounds_fully_closed"] += 1
        if miss:
            tot["rounds_short"] += 1
        per_round.append((r["uuid"], r["omega"], len(ours), len(miss),
                          len(recovered), len(closure - ours)))
    print("one-swap closure over bench_d10 (%d rounds)" % tot["rounds"])
    print("  omega-cliques missed            : %d" % tot["missed"])
    print("  recovered by one-swap closure   : %d (%.1f%%)"
          % (tot["recovered"],
             100.0 * tot["recovered"] / max(1, tot["missed"])))
    print("  rounds short                    : %d" % tot["rounds_short"])
    print("  rounds fully closed by one swap : %d" % tot["rounds_fully_closed"])
    print("  new cliques the closure proposes: %d (%.1f per round)"
          % (tot["closure"], tot["closure"] / max(1, tot["rounds"])))
    print("\nworst rounds (omega, held, missed, recovered, closure size)")
    per_round.sort(key=lambda t: -(t[3] - t[4]))
    for uuid, om, held, miss, rec, clo in per_round[:10]:
        print("  %s w=%3d held=%3d missed=%3d recovered=%3d closure=%5d"
              % (uuid[:8], om, held, miss, rec, clo))


if __name__ == "__main__":
    main()
