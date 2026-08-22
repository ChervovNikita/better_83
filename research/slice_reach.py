#!/usr/bin/env python3
"""Short independent restarts -- the only diversification the landscape allows.

Two measured facts force this design:

  * the (1,1)-swap neighbourhood of a maximum clique is EMPTY (0.00-0.14 other
    omega-cliques within 1-8 vertex drops). Maximum cliques are isolated points, so
    walking a plateau at omega has nowhere to go -- which is why the plateau walk
    measured as a no-op and was removed;
  * exhaustive enumeration is intractable at our density (MoMC, the best tool for
    the named problem, cannot finish one n=300 d=0.82 instance in 120s).

Independent restarts JUMP between isolated optima, which local moves cannot. Slice
length is the whole parameter and shorter is better wherever the solver still reaches
omega: measured upstream at n=696 d=0.86, a 0.2s slice gave 18 distinct cliques where
a 4.0s slice gave 2.

The failure mode is the other end: on n>=890 at d>=0.89 a short slice reaches omega
only 0-2 times in 38, so the slice must adapt. This keeps the best size found and
discards any slice that falls short of it.
"""
import os

import numpy as np


def reach(A, time_limit, k, slice_s=None, threads=None, seed=0):
    """Distinct maximum cliques from repeated short independent solves."""
    import fleetsolver
    A = np.ascontiguousarray(np.asarray(A, dtype=np.uint8))
    threads = threads or int(os.environ.get("SN83_THREADS", "13"))
    n = A.shape[0]
    if slice_s is None:
        # adapt: dense/large instances need a longer slice just to reach omega
        slice_s = float(os.environ.get("SN83_SLICE", "0.35"))
        dens = float(A.sum()) / max(n * (n - 1), 1)
        if n >= 850 and dens >= 0.88:
            slice_s = max(slice_s, time_limit * 0.25)
    best, pool = 0, {}
    spent, i = 0.0, 0
    import time
    t0 = time.time()
    while time.time() - t0 < time_limit and len(pool) < k:
        i += 1
        remain = time_limit - (time.time() - t0)
        s = min(slice_s, remain)
        if s <= 0.05:
            break
        try:
            cl = fleetsolver.solve_many(A, s, k, threads=threads, seed=seed + 7919 * i)
        except TypeError:
            cl = fleetsolver.solve_many(A, s, k, threads=threads)
        except Exception:
            continue
        if not cl:
            continue
        m = max(len(c) for c in cl)
        if m > best:
            best, pool = m, {}
        if m < best:
            continue                       # a short slice that missed omega is dropped
        for c in cl:
            if len(c) == best:
                pool.setdefault(tuple(sorted(int(v) for v in c)), None)
    return [list(t) for t in pool]
