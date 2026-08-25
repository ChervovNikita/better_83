#!/usr/bin/env python3
"""GPU harvester behind fleet_solver.solve_many's signature."""

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
    """Imports a module by path, under a name a bare import cannot shadow."""
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
    """Builds the .so and creates the CUDA context at import, never in a solve."""
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
    """Device counters from the most recent solve_many."""
    return dict(_last_stats)


def solve_many(adjacency_matrix, time_limit, k):
    """Returns up to k distinct maximal cliques, largest first."""
    assert k > 0
    assert time_limit > 0
    A = np.ascontiguousarray(adjacency_matrix, dtype=np.uint8)
    n = A.shape[0]
    assert A.shape == (n, n)
    assert 0 < n <= MAXN, "n=%d exceeds SN83_MAXN=%d" % (n, MAXN)

    import gpu_lib
    deadline = time.monotonic() + time_limit - RESERVE_S

    champion = None
    if not GPU_ONLY:
        champion = fleet_solver._solve_one(A, time_limit * CHAMPION_SHARE, seed=1)
        assert champion
        champion = sorted(int(v) for v in champion)

    left = deadline - time.monotonic()
    assert left > 0.05, "%.3fs left of a %.3fs budget" % (left, time_limit)

    with gpu_lib.GpuClique(A, lanes=LANES, prefix=PREFIX_ARM) as gpu:
        pool, counters = gpu.harvest(
            time_limit=left,
            seed=1,
            max_steps=STEPS,
            boot_steps=BOOT_STEPS,
            max_steps_cap=STEPS_CAP,
            spare_margin=SPARE_MARGIN,
            init_clique=champion,
            max_out=max(4 * k, 1024))
    if champion:
        pool.append(champion)

    # Exact re-check: a 64-bit fingerprint collision must cost a lost clique,
    # never an invalid answer.
    seen = set()
    verified = []
    for clique in pool:
        key = tuple(sorted(clique))
        if key in seen:
            continue
        is_clique, maximal = gpu_lib.verify(A, key)
        if is_clique and maximal:
            seen.add(key)
            verified.append(key)
    assert verified, "harvest returned %d cliques, none valid" % len(pool)

    target = max(len(c) for c in verified)
    out = sorted(list(c) for c in verified if len(c) == target)
    spare = sorted((list(c) for c in verified
                    if target - SPARE_MARGIN <= len(c) < target),
                   key=len, reverse=True)

    global _last_stats
    _last_stats = dict(counters)
    _last_stats["stall"] = gpu_lib.stall(counters)
    _last_stats["distinct_max"] = len(out)
    _last_stats["spares"] = len(spare)
    if DEBUG:
        sys.stderr.write("gpu: omega=%d distinct=%d spares=%d jobs=%d stall=%.2f\n"
                         % (target, len(out), len(spare), counters["jobs"],
                            _last_stats["stall"]))

    out.extend(spare)
    return out[:k]
