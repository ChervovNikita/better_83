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


def solve_many(adjacency_matrix, time_limit, k):
    assert k > 0
    assert time_limit > 0
    A = np.ascontiguousarray(adjacency_matrix, dtype=np.uint8)
    n = A.shape[0]
    assert A.shape == (n, n)
    assert 0 < n <= MAXN, "n=%d exceeds SN83_MAXN=%d" % (n, MAXN)

    import gpu_lib
    reserve = max(RESERVE_S, GPU_RESERVE_S, GPU_RESERVE_FRAC * time_limit)
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
            max_out=max(4 * k, 1024))
    # hits align with harvest's pool, so capture them before it is mutated
    hit_of = {tuple(sorted(c)): int(h) for c, h in zip(pool, hits)}
    if champion:
        pool.append(champion)
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
