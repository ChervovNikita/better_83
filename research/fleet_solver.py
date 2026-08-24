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
    _th = os.environ.get("SN83_SOLVE_THREADS")
    best = (solve_one(A, time_limit * share, seed=seed, threads=int(_th))
            if _th else solve_one(A, time_limit * share, seed=seed))
    out = [sorted(int(v) for v in best)]
    if k <= 1:
        return out

    rng = np.random.default_rng(seed or 1)
    seen = {tuple(out[0])}
    target = len(out[0])
    deadline = t0 + time_limit
    # Cliques the walk reaches that are MAXIMAL but smaller than omega. They used to
    # be dropped on the floor. They are valid answers -- the validator tests
    # maximality, not maximumness -- and the fleet needs them: on 31% of rounds the
    # max-size pool is shorter than the number of hotkeys queried, and the picker
    # then hands two siblings the same clique. Measured over the dumped submissions,
    # our own siblings account for 2.554 of the 3.727 holders on our cliques, more
    # than double the field's 1.173, and replacing each repeat with a unique
    # omega-1 answer wins on 87.2% of the affected rounds (median +3.77 fleet reward).
    #
    # They are collected SEPARATELY so the max-size search runs for exactly as many
    # iterations as before: `out` still grows only on max-size hits, so the loop
    # condition, the RNG stream and the resulting omega pool are byte-identical to
    # solve_many. That makes SN83_BACKFILL=0 an exact control, not an approximation.
    # SN83_POOL=ban -- the user's mechanism, measured 2026-08-24 and the first thing
    # in this project that produces genuinely different maximum cliques.
    #
    # DELETE a few vertices of a clique we already hold from the GRAPH (zero their
    # rows and columns) and re-run the full solver on what is left. The clique we
    # held is now impossible, so the solver must land somewhere else. Contrast with
    # the perturbation loop below, which KEEPS 7/8 of the clique and greedily
    # re-extends: that was measured at 26,000-32,000 attempts on one round producing
    # ZERO new maximum cliques, because a greedy extension of 25 of a 36-clique's
    # vertices returns that same clique essentially always.
    #
    #   round (field distinct)   keep-and-extend      delete-and-resolve
    #   4  (22)                  1  jac  n/a          10  jac 0.291
    #   5  (23)                  2  jac 0.968          8  jac 0.237
    #   8  (37)                  2  jac 0.929         16  jac 0.190
    #
    # The field's own distinct maxima sit at Jaccard 0.4669, so 0.19-0.29 is not
    # merely better than the walk, it is more spread out than the field.
    #
    # ADAPTIVE SLICE. Each re-solve gets SN83_BAN_FRAC of the budget, doubled
    # whenever a solve comes back below the size we already know is reachable. A
    # fixed 5% slice found 9 maxima on n=698 but only 1 on n=894, where 14 of 15
    # re-solves never reached omega at all. The doubling makes the arm safe on big
    # graphs without giving up the many-cheap-solves behaviour on small ones.
    if os.environ.get("SN83_POOL") == "ban":
        nban = int(os.environ.get("SN83_BAN_N", "3"))
        frac = float(os.environ.get("SN83_BAN_FRAC", "0.05"))
        ban_spare = int(os.environ.get("SN83_BACKFILL", "1"))
        ban_margin = int(os.environ.get("SN83_BF_MARGIN", "1"))
        spare = []
        while len(out) < k and time.time() < deadline:
            src = out[int(rng.integers(len(out)))]
            drop = rng.choice(list(src), size=min(nban, len(src)), replace=False)
            B = A.copy()
            for v in drop:
                B[int(v), :] = 0
                B[:, int(v)] = 0
            left = deadline - time.time()
            if left < 0.2:
                break
            cand = solve_one(B, min(left, time_limit * frac),
                             seed=int(rng.integers(1 << 30)))
            cand = tuple(sorted(int(v) for v in cand))
            if not cand:
                continue
            if len(cand) < target:
                # Keep it as a spare BEFORE deciding what to do about the slice. The
                # first version doubled and `continue`d here, which consumed every
                # sub-omega result and left `spare` empty -- the same bug the spread
                # path would have hit, one layer down.
                if (ban_spare and len(cand) >= target - ban_margin
                        and cand not in seen):
                    seen.add(cand)
                    spare.append(list(cand))
                if frac < 0.5:
                    frac *= 2.0      # the slice was too short to reach omega
                    continue
            if len(cand) > target:   # a banned graph cannot beat the full one, but
                target = len(cand)   # be safe rather than silently mis-size the pool
            if cand in seen:
                continue
            seen.add(cand)
            if len(cand) == target:
                out.append(list(cand))
            elif ban_spare and len(cand) >= target - ban_margin:
                # INTEGRATION BUG, caught before shipping: the spread path in
                # native_algorithm_shim.pick_spread needs `spare = [c for c in pool if
                # len(c) == mx - 1]`. Ban mode returned max-size cliques only, so
                # `spare` would be empty, pick_spread would return None, and the fleet
                # would silently never spread -- turning the search fix into a
                # regression on exactly the scarce-omega rounds spreading is for.
                spare.append(list(cand))
        out.sort(key=len, reverse=True)
        if ban_spare and len(out) < k:
            spare.sort(key=len, reverse=True)
            out.extend(spare[:k - len(out)])
        return out[:k]

    backfill = int(os.environ.get("SN83_BACKFILL", "1"))
    margin = int(os.environ.get("SN83_BF_MARGIN", "1"))   # keep sizes >= omega-margin
    spare = []
    spare_seen = set()
    # SN83_WALK: where the next perturbation starts from.
    #   0 = out[0] every time (the shipped behaviour, kept as an EXACT control:
    #       it draws from `rng` in the same order, so the stream is unchanged)
    #   1 = a random member of `out` -- a walk over the plateau instead of a star
    #   2 = walk, and randomise the drop size too
    # Measured 2026-08-23: at walk=0 our max-size cliques have mean pairwise
    # Jaccard 0.9166 (they differ by ~1.2 vertices) against the field's 0.4669.
    # Every alternate was a 1-vertex variant of the same champion clique, which is
    # why the pool held a median of 4 distinct maxima where the round holds 17.
    walk = int(os.environ.get("SN83_WALK", "0"))
    dmax = float(os.environ.get("SN83_DROP_MAX", "0.25"))
    while len(out) < k and time.time() < deadline:
        base = list(out[int(rng.integers(len(out)))] if walk else out[0])
        if walk >= 2:
            hi = max(2, int(len(base) * dmax) + 1)
            drop = int(rng.integers(1, hi))
        else:
            drop = max(1, len(base) // 8)
        keep = list(rng.choice(base, size=len(base) - drop, replace=False))
        cand = _extend_to_maximal(A, keep, rng)
        key = tuple(cand)
        if key in seen or len(cand) < target:
            if (backfill and len(cand) >= target - margin
                    and key not in spare_seen and key not in seen):
                spare_seen.add(key)
                spare.append(cand)
            continue
        seen.add(key)
        out.append(cand)
    out.sort(key=len, reverse=True)
    if backfill and len(out) < k:
        spare.sort(key=len, reverse=True)
        out.extend(spare[:k - len(out)])
    return out[:k]
