#!/usr/bin/env python3
"""Motzkin-Straus replicator dynamics -- the last solver family with different maths.

Every combinatorial mechanism tried (relabelling, exclusion, short restarts, deeper
harvesting, 16x compute) produced MORE cliques that were EQUALLY contested. That says
the set reachable by LSCC-family search is shared with the field.

Motzkin-Straus is not a search over vertex sets at all. It maximises x'Ax over the
probability simplex; the theorem says the maximum equals 1/2(1 - 1/omega), and the
support of a maximiser is a maximum clique. Replicator dynamics

    x_i <- x_i (Ax)_i / (x'Ax)

is a continuous flow whose fixed points are cliques. Different starting points and
different vertex weightings land on different fixed points, and the basins are those
of a dynamical system rather than of a local search -- a genuinely different
distribution over maxima, which is the property nothing else tried has had.

Regularisation: the plain QP has spurious non-clique maximisers, so we add the
standard alpha*x'x term (Bomze) with alpha < 1/2 to make characteristic vectors of
maximal cliques the only strict local maximisers.
"""
import os

import numpy as np


def _extract(A, x, thresh=1e-4):
    """Greedy maximal clique from the support of x, largest weight first."""
    order = np.argsort(-x)
    order = [int(v) for v in order if x[v] > thresh]
    cl = []
    for v in order:
        if all(A[v, u] for u in cl):
            cl.append(v)
    return cl


def reach(A, time_limit, k, omega=None, threads=None, seed=0):
    """Distinct cliques at size omega found by replicator dynamics from many starts."""
    import time
    A = np.ascontiguousarray(np.asarray(A, dtype=np.float64))
    n = A.shape[0]
    Ab = np.ascontiguousarray(np.asarray(A, dtype=np.uint8))
    alpha = float(os.environ.get("SN83_MS_ALPHA", "0.4"))
    M = A + alpha * np.eye(n)
    rng = np.random.default_rng(seed or 12345)
    iters = int(os.environ.get("SN83_MS_ITERS", "180"))
    pool, best = {}, (omega or 0)
    t0 = time.time()
    while time.time() - t0 < time_limit and len(pool) < k:
        # a random point on the simplex, mildly concentrated so runs differ a lot
        x = rng.dirichlet(np.full(n, float(os.environ.get("SN83_MS_CONC", "0.6"))))
        for _ in range(iters):
            Mx = M @ x
            d = float(x @ Mx)
            if d <= 0:
                break
            x = x * Mx / d
            s = x.sum()
            if s <= 0:
                break
            x /= s
            if time.time() - t0 > time_limit:
                break
        cl = _extract(Ab, x)
        if not cl:
            continue
        if len(cl) > best:
            best, pool = len(cl), {}
        if len(cl) == best:
            pool.setdefault(tuple(sorted(cl)), None)
    return [list(t) for t in pool]
