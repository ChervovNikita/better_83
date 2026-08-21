"""Multi-answer solver interface: one solve, many distinct cliques.

A fleet does not get N solve budgets. The validator sends the SAME graph to every
hotkey it queries, so one search has to produce every answer the fleet returns —
which is also why collision avoidance is nearly free: a plateau search already
walks through many maximum cliques, it just throws all but one away.

    solve_many(A, time_limit, k) -> list[list[int]]

Up to k DISTINCT maximal cliques, best first, produced within one time_limit.
Replace this module with the real implementation; the signature is the contract.

The default below wraps the single-answer native solver: it spends most of the
budget finding the champion, then walks the plateau around it collecting other
maximal cliques of the same size. It is a placeholder good enough to make the
simulation run, not a tuned picker.
"""
import os
import time

import numpy as np


def _extend_to_maximal(A, members, rng):
    """Grow a clique until no vertex can be added. Order is randomised so repeated
    calls from the same seed set land on different maximal cliques."""
    n = A.shape[0]
    inC = np.zeros(n, dtype=bool)
    inC[list(members)] = True
    cnt = A[list(members)].sum(axis=0) if members else np.zeros(n, dtype=np.int64)
    k = len(members)
    while True:
        cand = np.flatnonzero((cnt == k) & (~inC))
        if cand.size == 0:
            break
        v = int(cand[rng.integers(cand.size)])
        inC[v] = True
        cnt += A[v]
        k += 1
    return sorted(np.flatnonzero(inC).tolist())


def solve_many(A, time_limit, k, seed=0):
    """Default: native champion, then a plateau walk for alternates."""
    from fastsolver import solve as solve_one

    t0 = time.time()
    share = float(os.environ.get("SN83_CHAMPION_SHARE", "0.75"))
    best = solve_one(A, time_limit * share, seed=seed)
    out = [sorted(int(v) for v in best)]
    if k <= 1:
        return out

    rng = np.random.default_rng(seed or 1)
    seen = {tuple(out[0])}
    target = len(out[0])
    deadline = t0 + time_limit
    # drop a few vertices at random and re-extend: cheap, and it lands on other
    # maximal cliques in the same plateau
    while len(out) < k and time.time() < deadline:
        base = list(out[0])
        drop = max(1, len(base) // 8)
        keep = list(rng.choice(base, size=len(base) - drop, replace=False))
        cand = _extend_to_maximal(A, keep, rng)
        key = tuple(cand)
        if key in seen or len(cand) < target:
            continue
        seen.add(key)
        out.append(cand)
    out.sort(key=len, reverse=True)
    return out[:k]
