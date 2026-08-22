#!/usr/bin/env python3
"""Fleet solver that harvests across RELABELLINGS. Drop-in: (A, time_limit, k).

Why. For an N-hotkey fleet, fleet_sim serves hotkey j the j-th clique in the pool;
a pool smaller than N leaves the surplus queried, silent and scoring ZERO. Measured
on the current harvester: 17.7 distinct optima per round, so at N=40 the average
hotkey is idle 55.7% of the time. Yield IS the objective.

Permuting the vertex labels changes every deterministic tie-break in the search
(scan order, first-index-wins) while leaving the algorithm bit-identical. Measured
separately: 3.34x more distinct maxima at the SAME total budget (45.8 -> 166.2).

That result was filed as "does not help" because it moved single-miner freshness by
-1.7%. That verdict stands for THAT objective. This is a different objective: a
fleet needs many distinct optima, not one rare one, and 17.7 x 3.34 clears 40.

The budget is SPLIT across permutations, not multiplied — each of the R passes gets
time_limit/R, so this costs exactly what one solve costs.
"""
import os

import numpy as np


def solve_many(A, time_limit, k, seed=0):
    """Up to k distinct maximum cliques, harvested across R relabellings."""
    import fleetsolver

    A = np.ascontiguousarray(np.asarray(A, dtype=np.uint8))
    n = A.shape[0]
    R = int(os.environ.get("SN83_RELABELS", "6"))
    threads = int(os.environ.get("SN83_THREADS", "14"))
    per = max(0.4, float(time_limit) / max(R, 1))
    rng = np.random.default_rng(seed or 8383)

    best_size, pool = 0, {}
    for i in range(R):
        if i == 0:
            perm = np.arange(n)
            B = A
        else:
            perm = rng.permutation(n)
            B = np.ascontiguousarray(A[np.ix_(perm, perm)])
        try:
            cl = fleetsolver.solve_many(B, per, k, threads=threads)
        except Exception:
            continue
        if not cl:
            continue
        m = max(len(c) for c in cl)
        if m < best_size:
            continue                      # a worse pass never displaces a better one
        if m > best_size:
            best_size, pool = m, {}       # a bigger clique invalidates the old pool
        for c in cl:
            if len(c) == m:
                key = tuple(sorted(int(perm[v]) for v in c))
                if key not in pool:
                    pool[key] = None
                    if len(pool) >= k:
                        return [list(t) for t in pool]
    return [list(t) for t in pool]


if __name__ == "__main__":
    import json
    import sys
    import time
    HERE = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, HERE)
    sys.path.insert(0, os.path.dirname(HERE))
    from CliqueAI.graph.codec import GraphCodec
    from _common import DATA_DIR
    rs = [json.loads(l) for l in open(os.path.join(DATA_DIR, "sim_rounds.jsonl"))
          if l.strip()][:3]
    for r in rs:
        A = np.array(GraphCodec().decode_matrix(r["matrix_b92"]), dtype=np.uint8)
        t = time.time()
        cl = solve_many(A, max(0.5, r["time_limit"] - 2.0), 40)
        mx = max((len(c) for c in cl), default=0)
        print(f"n={r['n']:>4} tl={r['time_limit']:>4} best={r['best_size']:>3} -> "
              f"{len(cl):>3} distinct at size {mx} in {time.time()-t:.1f}s")
