"""The researched champion (research/native/clique.cpp), wired for the live miner.

Everything in research/ was measured against a solver that the deployed miner never
called: CliqueAI/miner.py runs `nx.approximation.max_clique`, a greedy approximation
that does not reach omega. "Promoted to native/clique.cpp" meant promoted within the
research harness. This module is the missing link.

Three properties matter more than speed here, because a bad answer scores a hard zero
while the upstream answer at least scores something:

  1. FALLBACK. Any failure -- missing .so, build error, ctypes fault, timeout overrun --
     returns the upstream result instead of raising. The miner must always answer.
  2. VALIDATION. The returned vertex set is checked to be a clique AND maximal before it
     is used. The validator tests maximality, not maximumness, so a non-maximal answer
     is worth zero even when it is large.
  3. DEADLINE. The solver is given the synapse timeout minus a round-trip reserve. A
     late answer is a zero on chain. The 2 s reserve was measured: parity 99.800% ->
     99.600%, McNemar p=1.0000, reward -0.0001, 0 invalid, 0 over budget.
"""
import os
import sys
import threading
import time

import numpy as np

_RESEARCH = os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "research")

# Seven validators query this subnet, so concurrent requests are not hypothetical, and
# the solver spawns SN83_THREADS workers (default 8) against a 4-8 core miner. The first
# design serialised solves behind a semaphore. Measured, that was worse than the problem:
# with four simultaneous requests the two that waited ran out of budget and fell back,
# and a fallback is worth about 76% of omega while a thread-starved native solve still
# reaches omega. So concurrency is admitted and the THREAD BUDGET is shared instead.
def available_cores():
    """Cores this process may actually use, honouring the cgroup CPU quota.

    os.cpu_count() reports the HOST's cores, and in a container that is routinely wrong
    by an order of magnitude: the box this was developed on reports 128 against a quota
    of 15. A solver sizing its pool from cpu_count there spawns 128 threads for 15 cores,
    which does not error -- every thread is simply throttled, and the search does less
    work per second of a wall-clock-bounded budget. Miners on a VPS or in k8s are the
    normal case, not the exception.
    """
    n = os.cpu_count() or 1
    try:                                                  # cgroup v2
        with open("/sys/fs/cgroup/cpu.max") as f:
            quota, period = f.read().split()
        if quota != "max":
            n = min(n, max(1, int(int(quota) / int(period))))
    except Exception:
        pass
    try:                                                  # cgroup v1
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as f:
            q = int(f.read())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f:
            per = int(f.read())
        if q > 0 and per > 0:
            n = min(n, max(1, q // per))
    except Exception:
        pass
    try:                                                  # explicit pinning
        n = min(n, len(os.sched_getaffinity(0)))
    except Exception:
        pass
    return max(1, n)


# SN83_THREADS still wins when set, so an operator running several hotkeys on one box
# can divide the budget between them explicitly. Unset, the default is the real core
# count capped at 8, which is what min_compute.yml calls a recommended miner.
TOTAL_THREADS = int(os.environ.get("SN83_THREADS", "0")) or min(8, available_cores())
_active = 0
_active_lock = threading.Lock()
_WARNED = False
# Below this many seconds of remaining budget the native solve is not worth starting.
MIN_BUDGET_S = float(os.environ.get("SN83_MIN_BUDGET_S", "0.75"))

# Seconds held back from the deadline for the trip home through the dendrite.
ROUND_TRIP_S = float(os.environ.get("SN83_ROUND_TRIP_S", "2.0"))
# Used only when the synapse carries no timeout. The observed range is 6-30 s, so
# this is deliberately at the bottom of it: overrunning scores zero, finishing early
# only costs unused search.
DEFAULT_TIMEOUT_S = float(os.environ.get("SN83_DEFAULT_TIMEOUT_S", "6.0"))


def _is_valid_maximal_clique(A, verts):
    """The validator's own test: a clique, and extendable by no vertex."""
    if not verts:
        return False
    v = np.asarray(sorted(set(int(x) for x in verts)), dtype=np.int64)
    if len(v) != len(verts):
        return False                       # duplicate vertices
    if v.min() < 0 or v.max() >= A.shape[0]:
        return False
    sub = A[np.ix_(v, v)]
    if not np.array_equal(sub, sub.T):
        return False
    k = len(v)
    if sub.sum() != k * (k - 1):           # every pair adjacent, zero diagonal
        return False
    inC = np.zeros(A.shape[0], dtype=bool)
    inC[v] = True
    # maximal: no outside vertex is adjacent to all k members
    if (A[v].sum(axis=0)[~inC] == k).any():
        return False
    return True


def solver_seed(hotkey, uuid):
    """A seed unique to this hotkey AND this task.

    The solver is reproducible in practice: with the default seed it returned the same
    clique on 5 of 5 runs on every round tested. An operator running N hotkeys as N
    miner processes therefore has all N submit the IDENTICAL clique, and the scorer pays
    diversity = 1 / holders. Measured over 40 rounds with K=8 hotkeys inserted into the
    real round, that costs -0.8954 per answer against distinct omega cliques.

    Seeding by hotkey alone would fix the collision but pin each hotkey to one basin
    across every task; mixing in the task uuid re-randomises the assignment each round,
    which is also what fleet_pick.pick's per-round offset does and for the same reason.
    """
    import hashlib
    h = hashlib.sha1(("%s|%s" % (hotkey, uuid)).encode()).hexdigest()
    return int(h[:16], 16) & 0x7FFFFFFFFFFFFFFF


# Spread: submit a maximal omega-1 clique instead of omega when the omega pool is too
# small for every queried sibling to hold a distinct one. This is what the FIELD does,
# measured over 399 rounds -- the share of field answers at omega-1 is 94.5% when only
# one maximum clique exists, 0.9% when 16 or more do, correlation -0.640 with the
# distinct-omega count. It works because score_round's `pr` counts answers STRICTLY
# larger: when almost nobody holds omega, an omega-1 answer keeps optimality ~0.947 and
# buys a full diversity term. On one measured round a unique omega-1 was worth 2.705
# against 1.925 for one of eight identical omega cliques.
#
# A lone miner can decide this without any coordinator: it knows its own harvested pool,
# the difficulty, and its operator's fleet size, so it can estimate how many siblings
# will be queried and whether the pool covers them.
SPREAD = int(os.environ.get("SN83_SPREAD", "0"))
FLEET_SIZE = int(os.environ.get("SN83_FLEET_SIZE", "1"))


def _spread_pick(pool, hotkey, uuid, difficulty, fleet_size):
    """Choose omega or a distinct omega-1 from a harvested pool, using local state only.

    Returns None when the pool cannot support a decision, so the caller falls back to
    the plain single-clique path.
    """
    if not pool:
        return None
    mx = max(len(c) for c in pool)
    top = [c for c in pool if len(c) == mx]
    spare = [c for c in pool if len(c) == mx - 1]
    import hashlib
    import math
    # Expected number of our hotkeys queried this round. MinerSelector samples each uid
    # independently at p(difficulty), uniform across uids.
    x_m = math.sqrt(1.0 + 1.5)
    p = 1.0 - math.exp(-max(0.0, x_m - float(difficulty) - 0.5))
    q = max(1, int(round(fleet_size * p)))
    if len(top) >= q or not spare:
        slots = top                      # pool covers the fleet: everyone takes omega
    else:
        slots = top + spare[:q - len(top)]
    h = int(hashlib.sha1(("%s|%s" % (hotkey, uuid)).encode()).hexdigest()[:16], 16)
    return slots[h % len(slots)]


def native_algorithm(number_of_nodes, adjacency_list, adjacency_matrix=None,
                     timeout=None, fallback=None, seed=0, hotkey=None, uuid=None,
                     difficulty=None):
    """Return a maximal clique, falling back to `fallback` on any problem.

    `fallback` is called with no arguments and must return a vertex list; pass the
    upstream algorithm. If it is None and the native path fails, an empty list is
    returned, which the validator scores as zero -- so always pass one.
    """
    def _fb(reason):
        # Loud ONCE, then quiet. A silent fallback is the worst outcome here: the miner
        # looks healthy, answers every request, and earns the approximation's reward
        # forever. The deployment checklist says to confirm the first logged clique size
        # is in the 20-60 range rather than single digits -- this is what makes that
        # check possible.
        global _WARNED
        if not _WARNED:
            _WARNED = True
            try:
                import bittensor as bt
                bt.logging.warning(
                    "SN83 native solver unavailable (%s); falling back to the upstream "
                    "approximation. This costs roughly 0.73 reward per answer. Check "
                    "that research/native/ ships with the deployment and that g++ is "
                    "present for build.sh." % reason)
            except Exception:
                print("SN83 native solver unavailable (%s); using fallback" % reason,
                      file=sys.stderr)
        try:
            return list(fallback()) if fallback is not None else []
        except Exception:
            return []

    try:
        if adjacency_matrix is None:
            A = np.zeros((number_of_nodes, number_of_nodes), dtype=np.uint8)
            for i, nbrs in enumerate(adjacency_list):
                for j in nbrs:
                    A[i, int(j)] = 1
        else:
            A = np.ascontiguousarray(adjacency_matrix, dtype=np.uint8)
        np.fill_diagonal(A, 0)

        deadline = time.monotonic() + (float(timeout) if timeout
                                       else DEFAULT_TIMEOUT_S) - ROUND_TRIP_S

        if _RESEARCH not in sys.path:
            sys.path.insert(0, _RESEARCH)
        from fastsolver import solve as solve_one       # builds the .so if stale

        budget = deadline - time.monotonic()
        if budget < MIN_BUDGET_S:
            return _fb("only %.2fs of budget left" % budget)
        # Share the cores rather than queue for them. Each in-flight solve takes an
        # equal slice, floor 1, so N simultaneous requests never ask the box for more
        # than TOTAL_THREADS between them.
        global _active
        with _active_lock:
            _active += 1
            share = max(1, TOTAL_THREADS // _active)
        try:
            if SPREAD and hotkey is not None and uuid is not None:
                # Harvest a pool rather than a single clique, then decide locally.
                # solve_many keeps maximal sub-omega cliques as spares (SN83_BACKFILL).
                from fleet_solver import solve_many
                want = max(2, FLEET_SIZE)
                pool = solve_many(A, budget, want, seed=seed)
                pick = _spread_pick([tuple(c) for c in pool], hotkey, uuid,
                                    difficulty if difficulty is not None else 0.8,
                                    FLEET_SIZE)
                clique = list(pick) if pick else solve_one(A, budget, seed=seed,
                                                           threads=share)
            else:
                clique = solve_one(A, budget, seed=seed, threads=share)
        finally:
            with _active_lock:
                _active -= 1

        clique = [int(x) for x in clique]
        if _is_valid_maximal_clique(A, clique):
            return sorted(clique)
        return _fb("solver returned a clique that failed the maximality check")
    except Exception as e:
        return _fb("%s: %s" % (type(e).__name__, e))
