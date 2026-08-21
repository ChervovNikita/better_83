#!/usr/bin/env python3
"""Fleet answer generator: k independent seeded runs, one per hotkey.

Drop-in for fleet_sim's --solver, same (A, time_limit, k) signature as
fleet_solver.solve_many and fleetsolver.solve_many.

Why this exists. Measured on recent_val, the mean pairwise vertex-swap distance
between answers is:

    independent seeds   20.33
    one harvested plateau   12.01
    (for scale, the field cliques we cannot reach sit 26.95 away)

Harvesting a single plateau covers barely half the ground independent seeds do, and
both fleetsolver.py and the fleet simulation harvest. Two consequences:

  * coverage -- a fleet built by harvesting clusters more tightly than one built from
    independent seeds, so it self-collides more and earns less diversity;
  * starvation -- harvesting yields a variable number of cliques (mean ~14 at k=40,
    median 8, sometimes 1). Hotkeys past the pool are queried, return nothing and
    score ZERO.

    Seeds help here but do NOT eliminate it, and an earlier draft of this docstring
    wrongly claimed they did. Measured: on a low-yield instance (n=891, best=27) five
    seeds produced only TWO distinct answers -- different starts still converge when
    the instance has few maximum cliques. So seeds return AT MOST k distinct answers
    and often fewer, exactly like harvesting; the claim worth making is that they
    spread wider (20.33 vs 12.01), not that they are always fully served.

The cost is honest and worth stating: this runs k full-budget solves instead of one,
which is only free because the deployment is one pod per hotkey. If a single box had
to serve k hotkeys, harvesting would be the only option.
"""
import os

import numpy as np


def solve_many(A, time_limit, k, seed=0):
    """k answers from k independent full-budget runs, distinct where possible."""
    from fastsolver import solve as solve_one

    A = np.ascontiguousarray(np.asarray(A, dtype=np.uint8))
    out, seen = [], set()
    for i in range(int(k)):
        c = solve_one(A, time_limit, seed=int(seed) + 7919 * (i + 1))
        t = tuple(sorted(int(v) for v in c))
        if not t:
            continue
        if t not in seen:
            seen.add(t)
            out.append(list(t))
    # Fleet answers must be DISTINCT to avoid self-collision. If seeds repeated an
    # answer we return fewer than k rather than duplicates -- fleet_sim scores a
    # hotkey with no clique as zero, which is the honest cost of the collision.
    return out


if __name__ == "__main__":
    import json
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from CliqueAI.graph.codec import GraphCodec
    from _common import DATA_DIR
    r = json.loads(open(os.path.join(DATA_DIR, "sets/recent_val.jsonl")).readline())
    A = np.array(GraphCodec().decode_matrix(r["b92"]), dtype=np.uint8)
    cl = solve_many(A, r["tl"] * 0.88, 5)
    print(f"n={r['n']} best={r['best']} -> {len(cl)} distinct, "
          f"sizes {sorted({len(c) for c in cl})}")
