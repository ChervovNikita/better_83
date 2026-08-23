#!/usr/bin/env python3
"""Assign one clique per queried hotkey, and size the solve to what is actually needed.

Two measured facts drive this module.

**Never stay silent.** When the pool holds fewer cliques than the fleet has queried
hotkeys, the surplus must still submit something. Returning [] is scored invalid and
earns a hard 0; resubmitting a sibling's clique earns the full optimality term (~1.76)
and merely shares the diversity term. Measured over 81 affected rounds (29.5% of all
rounds, where 6.6 hotkeys are short on average): duplicating is worth **+12.15 total
fleet reward per round and wins on 100% of them**. In the simulator this lifts an
N=40 fleet's median from 1.8265 to 2.2502.

**Do not solve for 40 when 9 are asked.** The validator samples each uid independently
at P(difficulty), so a fleet of N gets Binomial(N, p) queries -- mean 8.9 for N=40,
but ranging 0..19 and swinging with difficulty (12.1 at d=0.7, 3.3 at d=1.0). Asking
the harvester for 40 distinct cliques on a round that will only query 3 wastes the
deadline. `needed()` sizes the request from the difficulty the task itself carries.
"""
import hashlib
import math

REFERENCE_R = 1.5          # MinerSelector.reference_r


def selection_p(difficulty):
    """MinerSelector.miner_selection_probabilities -- uniform across uids."""
    x_m = math.sqrt(1.0 + REFERENCE_R)
    return 1.0 - math.exp(-max(0.0, x_m - float(difficulty) - 0.5))


def needed(fleet_size, difficulty, safety=2.0, cap=None):
    """How many DISTINCT cliques this round plausibly needs.

    Binomial(fleet_size, p) queries arrive, so aim at the mean plus `safety` standard
    deviations rather than at fleet_size. Overshooting wastes deadline; undershooting
    forces duplicates, which cost only shared diversity -- so the asymmetry favours a
    modest margin, not a large one.
    """
    n = max(int(fleet_size), 1)
    p = selection_p(difficulty)
    mean = n * p
    sd = math.sqrt(max(n * p * (1.0 - p), 0.0))
    k = int(math.ceil(mean + safety * sd))
    k = max(1, min(k, n))
    return min(k, cap) if cap else k


def pick(pool, uuid, hotkey, rank=None):
    """This hotkey's clique. Wraps on a short pool rather than returning nothing."""
    if not pool:
        return []
    if rank is None:
        h = hashlib.sha1(f"{hotkey}|{uuid}".encode()).hexdigest()
        return pool[int(h[:8], 16) % len(pool)]
    # Rotate by the round so hotkey 0 does not take the crowded natural endpoint
    # every time and carry the whole collision penalty alone.
    off = int(hashlib.sha1(str(uuid).encode()).hexdigest()[:8], 16)
    return pool[(rank + off) % len(pool)]


def assign(pool, uuid, hotkeys, ranks=None):
    """Whole-fleet assignment: distinct while the pool lasts, then duplicates.

    `hotkeys` should be the ones actually QUERIED this round. `ranks` lets a
    coordinator pass stable per-hotkey indices; otherwise position is used.
    """
    if not pool:
        return {hk: [] for hk in hotkeys}
    out = {}
    for i, hk in enumerate(hotkeys):
        r = ranks[i] if ranks is not None else i
        out[hk] = pick(pool, uuid, hk, rank=r)
    return out


def picker(pool, uuid, hotkeys):
    """fleet_sim --picker entry point: one answer per queried hotkey, in order.

    The SOLVER owns this decision, not the harness. Returning an empty list for a
    hotkey means it deliberately answers nothing, which the validator scores as 0 --
    that remains available, it is simply never the better choice here: a repeat still
    earns the full optimality term while silence earns nothing.
    """
    a = assign(pool, uuid, list(hotkeys))
    return [a[hk] for hk in hotkeys]


def picker_silent(pool, uuid, hotkeys):
    """Control: distinct while the pool lasts, then genuinely silent.

    This is what fleet_sim used to hardcode. Kept as a picker so the two strategies
    can be compared through the same interface instead of by patching the harness.
    """
    out = []
    for i, hk in enumerate(hotkeys):
        out.append(list(pool[i]) if i < len(pool) else [])
    return out


def duplicates(assignment):
    """How many hotkeys share a clique with a sibling -- the cost of a short pool."""
    seen = {}
    for hk, c in assignment.items():
        seen.setdefault(tuple(c), []).append(hk)
    return sum(len(v) - 1 for v in seen.values() if len(v) > 1)


def picker_backfill(pool, uuid, hotkeys):
    """Distinct max-size cliques first; dip into sub-omega ones only when short.

    The pool may now contain maximal cliques below omega (see fleet_solver's
    SN83_BACKFILL). Handing those out indiscriminately would be a loss: the
    optimality term is multiplied by (1 + difficulty), so dropping a vertex costs
    up to 2x what the diversity term can pay back. They are worth submitting in
    exactly one situation -- when the alternative is repeating a sibling's clique,
    which earns full optimality but splits the diversity term two ways.

    So: rotate within the max-size prefix while it is long enough to give every
    queried hotkey a distinct answer, and only extend into the smaller ones for the
    surplus. Wrapping remains the last resort, for when even that runs out.
    """
    q = len(hotkeys)
    if not pool or q == 0:
        return [[] for _ in hotkeys]
    mx = max(len(c) for c in pool)
    top = [c for c in pool if len(c) == mx]
    if len(top) >= q:
        use = top
    else:
        spare = sorted((c for c in pool if len(c) < mx), key=len, reverse=True)
        use = top + spare[:q - len(top)]
    a = assign(use, uuid, list(hotkeys))
    return [a[hk] for hk in hotkeys]


def picker_maxonly(pool, uuid, hotkeys):
    """Exact control for picker_backfill: the behaviour before spares existed.

    solve_many now returns sub-omega cliques too, so plain `picker` would hand them
    out through the modular wrap and would NOT be the old behaviour. Filtering to the
    max-size prefix reproduces it exactly, which makes the two pickers a paired
    comparison on one solve rather than two runs of a nondeterministic solver.
    """
    if not pool:
        return [[] for _ in hotkeys]
    mx = max(len(c) for c in pool)
    return picker([c for c in pool if len(c) == mx], uuid, hotkeys)
