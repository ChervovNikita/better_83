#!/usr/bin/env python3

import importlib.util
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
for _p in (HERE, PARENT):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _load_sibling(name, path):
    key = "sn83_" + name
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    return sys.modules[key]


fleet_solver = _load_sibling("fleet_solver", os.path.join(PARENT, "fleet_solver.py"))

CHAMPION_SHARE = fleet_solver.CHAMPION_SHARE
SPARE_MARGIN = fleet_solver.SPARE_MARGIN
RESERVE_S = fleet_solver.RESERVE_S

GPU_RESERVE_S = float(os.environ.get("SN83_GPU_RESERVE", "0.35"))
GPU_RESERVE_FRAC = float(os.environ.get("SN83_GPU_RESERVE_FRAC", "0.03"))

STEPS = int(os.environ.get("SN83_GPU_STEPS", "20000"))
BOOT_STEPS = int(os.environ.get("SN83_GPU_BOOT_STEPS", "60000"))
STEPS_CAP = int(os.environ.get("SN83_GPU_STEPS_CAP", str(1 << 20)))
LANES = int(os.environ.get("SN83_GPU_LANES", "32"))
PREFIX_ARM = os.environ.get("SN83_GPU_PREFIX", "0") == "1"
GPU_ONLY = os.environ.get("SN83_GPU_ONLY", "0") == "1"
DEBUG = os.environ.get("SN83_GPU_DEBUG", "0") == "1"

MAXN = None
_last_stats = {}


def _init_gpu():
    global MAXN
    import gpu_lib
    gpu_lib.load(LANES, PREFIX_ARM)
    probe = np.ones((8, 8), dtype=np.uint8)
    np.fill_diagonal(probe, 0)
    with gpu_lib.GpuClique(probe, lanes=LANES, prefix=PREFIX_ARM, walkers=4) as gpu:
        MAXN = gpu.maxn
        if DEBUG:
            sys.stderr.write("gpu: %s\n" % gpu.info())


_init_gpu()


def last_stats():
    return dict(_last_stats)


# Research only: cap on cliques stashed for SN83_POOL_DUMP. 0 disables.
FULL_POOL = int(os.environ.get("SN83_FULL_POOL", "0"))

# The device result table holds res_cap = 1<<13 entries and harvest copies the
# FIRST max_out of them, unsorted -- so a small max_out drops omega-cliques that
# were found after the buffer filled. Default is unchanged; raise it to 8192 to
# take the whole table when harvesting a research pool.
MAX_OUT = int(os.environ.get("SN83_GPU_MAX_OUT", "1024"))

# One-swap closure: after the harvest, every omega-clique that differs from a
# harvested one by a single vertex is added without searching for it. The
# harvest reaches omega and then either re-derives what it holds or runs out of
# clock, so the siblings one vertex away are never enumerated. CLOSURE_MAX caps
# it, and the work is bounded by CLOSURE_S / CLOSURE_CAP rather than by a cap on
# how many omega-cliques the round has: with the time bound in place a rich
# round costs no more than a poor one, it just stops sooner.
CLOSURE = os.environ.get("SN83_GPU_CLOSURE", "1") == "1"
# 0 disables the count gate entirely; the time and size bounds do the limiting.
CLOSURE_MAX = int(os.environ.get("SN83_GPU_CLOSURE_MAX", "0"))
# The closure's own output has to be expanded too: a clique two swaps from the
# harvest is one swap from something the first pass created, and a single pass
# never looks there. Iterate to a fixpoint, bounded three ways so a pathological
# round cannot run away with the deadline.
CLOSURE_ITERS = int(os.environ.get("SN83_GPU_CLOSURE_ITERS", "8"))
CLOSURE_CAP = int(os.environ.get("SN83_GPU_CLOSURE_CAP", "4096"))
# Measured: rounds with n_top <= 127 finish the fixpoint in under 0.07s and
# never hit this bound, while rounds above it want 0.6-0.9s and get truncated.
# That is the right place to truncate -- a round with hundreds of omega-cliques
# spreads occupancy thin enough that the cliques we do not reach change nothing.
CLOSURE_S = float(os.environ.get("SN83_GPU_CLOSURE_S", "0.20"))
CLOSURE_CHUNK = int(os.environ.get("SN83_GPU_CLOSURE_CHUNK", "128"))


def one_swap_closure(A, cliques, omega):
    """Every omega-clique one vertex away from `cliques`.

    For a clique C and a dropped vertex v, any u adjacent to all of C\\{v}
    yields another omega-clique. Rather than intersecting neighbourhoods once
    per drop, count how many members of C each vertex is adjacent to -- one
    omega x n pass -- then u survives the drop of v exactly when

        cnt[u] - A[v, u] == omega - 1

    so each drop costs a single length-n comparison over that shared count.
    O(omega * n) per clique instead of O(omega^2 * n). Every result is a clique
    of size omega, and omega is the maximum, so every result is also maximal.
    """
    out = set()
    need = omega - 1
    for c in cliques:
        if len(c) != omega:
            continue
        idx = np.fromiter(c, dtype=np.intp, count=omega)
        cnt = A[idx].sum(axis=0, dtype=np.int32)
        for v in idx:
            keep = np.nonzero(cnt - A[v] == need)[0]
            if keep.size <= 1:
                continue
            rest = [int(x) for x in idx if x != v]
            for u in keep:
                u = int(u)
                if u != int(v):
                    out.add(tuple(sorted(rest + [u])))
    return out


def closure_fixpoint(A, cliques, omega, deadline):
    """Expand, then expand what the expansion produced, until nothing is new.

    Stops on any of: no growth, CLOSURE_CAP cliques, CLOSURE_ITERS rounds, or
    the wall-clock deadline. Returns (cliques, iterations).
    """
    cur = set(cliques)
    frontier = list(cur)
    for i in range(1, CLOSURE_ITERS + 1):
        # Only the frontier needs expanding: a clique already expanded has its
        # whole closure inside `cur`, so re-expanding it can yield nothing new.
        # The frontier is walked in chunks so the deadline is checked inside a
        # pass as well as between passes -- one pass over a large frontier is
        # itself long enough to overrun the round.
        fresh = set()
        stopped = False
        for j in range(0, len(frontier), CLOSURE_CHUNK):
            # Subtract `cur` per chunk, not once at the end: the raw closure is
            # mostly cliques already held, so testing the cap against it counts
            # those twice and halts on a cap that has not been reached.
            fresh |= one_swap_closure(A, frontier[j:j + CLOSURE_CHUNK], omega) - cur
            if time.monotonic() >= deadline or len(cur) + len(fresh) >= CLOSURE_CAP:
                stopped = True
                break
        if not fresh:
            return cur, i
        cur |= fresh
        frontier = list(fresh)
        if stopped:
            return cur, i
    return cur, CLOSURE_ITERS


def solve_many(adjacency_matrix, time_limit, k):
    assert k > 0
    assert time_limit > 0
    A = np.ascontiguousarray(adjacency_matrix, dtype=np.uint8)
    n = A.shape[0]
    assert A.shape == (n, n)
    assert 0 < n <= MAXN, "n=%d exceeds SN83_MAXN=%d" % (n, MAXN)

    import gpu_lib
    reserve = max(RESERVE_S, GPU_RESERVE_S, GPU_RESERVE_FRAC * time_limit)
    # The closure runs after the harvest returns, so its budget has to come out
    # of the harvest's, not out of the caller's margin.
    if CLOSURE:
        reserve += CLOSURE_S
    deadline = time.monotonic() + time_limit - reserve

    champion = None
    if not GPU_ONLY:
        champion = fleet_solver._solve_one(A, time_limit * CHAMPION_SHARE, seed=1)
        assert champion
        champion = sorted(int(v) for v in champion)

    with gpu_lib.GpuClique(A, lanes=LANES, prefix=PREFIX_ARM) as gpu:
        left = deadline - time.monotonic()
        assert left > 0.05, "%.3fs left of a %.3fs budget" % (left, time_limit)
        pool, counters, hits = gpu.harvest(
            time_limit=left,
            seed=1,
            max_steps=STEPS,
            boot_steps=BOOT_STEPS,
            max_steps_cap=STEPS_CAP,
            spare_margin=SPARE_MARGIN,
            init_clique=champion,
            max_out=max(4 * k, MAX_OUT))
    # hits align with harvest's pool, so capture them before it is mutated
    hit_of = {tuple(sorted(c)): int(h) for c, h in zip(pool, hits)}
    if champion:
        pool.append(champion)
    _closure_added = 0
    _closure_iters = 0
    if CLOSURE:
        omega_now = max(len(c) for c in pool)
        top_now = {tuple(sorted(c)) for c in pool if len(c) == omega_now}
        if not CLOSURE_MAX or len(top_now) <= CLOSURE_MAX:
            grown, _closure_iters = closure_fixpoint(
                A, top_now, omega_now, time.monotonic() + CLOSURE_S)
            added = grown - top_now
            pool.extend(list(c) for c in added)
            _closure_added = len(added)
    pool.sort(key=len, reverse=True)

    seen = set()
    out = []
    spare = []
    for clique in pool:
        if len(out) >= k and len(spare) >= k:
            break
        key = tuple(sorted(clique))
        if key in seen:
            continue
        is_clique, maximal = gpu_lib.verify(A, key)
        if not (is_clique and maximal):
            continue
        seen.add(key)
        if not out or len(key) == len(out[0]):
            if len(out) < k:
                out.append(key)
        elif len(key) >= len(out[0]) - SPARE_MARGIN and len(spare) < k:
            spare.append(key)
    assert out, "harvest returned %d cliques, none valid" % len(pool)

    target = len(out[0])
    # Count DISTINCT cliques: champion is both seeded into harvest and appended
    # to pool, so a raw count reports one omega-clique more than exists and the
    # picker spreads over a clique it does not hold.
    distinct = {tuple(sorted(c)) for c in pool}
    n_top_true = sum(1 for c in distinct if len(c) == target)
    n_spare_true = sum(1 for c in distinct if len(c) == target - 1)
    out = sorted(list(c) for c in out)
    spare = [list(c) for c in spare]

    global _last_stats
    _last_stats = dict(counters)
    _last_stats["stall"] = gpu_lib.stall(counters)
    _last_stats["pool"] = len(pool)
    _last_stats["distinct_max"] = len(out)
    _last_stats["spares"] = len(spare)
    _last_stats["n_top_true"] = n_top_true
    _last_stats["n_spare_true"] = n_spare_true
    _last_stats["closure_added"] = _closure_added
    _last_stats["closure_iters"] = _closure_iters
    _last_stats["hits"] = [hit_of.get(tuple(c), 0) for c in out + spare]
    if FULL_POOL:
        # Research dump: every distinct clique the harvest produced, ranked by
        # size then basin. NOT put through gpu_lib.verify -- only out/spare are.
        # Validate offline before drawing any conclusion from these.
        ranked = sorted(distinct, key=lambda c: (-len(c), -hit_of.get(c, 0)))
        ranked = ranked[:FULL_POOL]
        _last_stats["full_pool_unverified"] = [list(c) for c in ranked]
        _last_stats["full_hits"] = [hit_of.get(c, 0) for c in ranked]
    if DEBUG:
        sys.stderr.write("gpu: omega=%d distinct=%d spares=%d jobs=%d stall=%.2f\n"
                         % (target, len(out), len(spare), counters["jobs"],
                            _last_stats["stall"]))

    return out + spare
