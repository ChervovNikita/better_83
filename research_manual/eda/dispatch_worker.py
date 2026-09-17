#!/usr/bin/env python3
"""One solver worker, pinned to one GPU, in its own process.

The dispatcher owns a pool of these. A worker is a process, not a thread,
because each needs its own CUDA_VISIBLE_DEVICES and therefore its own context --
two contexts in one process time-slice the same device and give up the point.

Backends
--------
gpu   fleet_solver_gpu.solve_many, one CUDA device per worker
cpu   fleet_solver.solve_many, no device at all
fake  a deterministic stub returning synthetic cliques

`fake` is what makes the dispatcher testable on a machine with no GPU: it
exercises routing, admission, sibling batching and deadlines without a solver.
"""

import os
import time


class Backend(object):
    """Solves one task. Constructed inside the worker process, after pinning."""

    def __init__(self, kind, threads=None):
        self.kind = kind
        self.threads = threads
        self._solve_many = None

    def warm(self):
        """Build, load and touch the device now, never inside a request.

        An nvcc build takes tens of seconds and CUDA context creation a few
        hundred milliseconds. Either inside a request spends the round's
        deadline, and a late answer scores zero.
        """
        if self.kind == "fake":
            return "fake backend, no solver"
        import sys
        here = os.path.dirname(os.path.abspath(__file__))
        for p in (os.path.dirname(here), here):
            if p not in sys.path:
                sys.path.insert(0, p)
        if self.kind == "cpu":
            import fleet_solver
            if self.threads:
                fleet_solver.THREADS = self.threads
            self._solve_many = fleet_solver.solve_many
            return "cpu backend, THREADS=%s" % fleet_solver.THREADS
        import fleet_solver_gpu
        if self.threads:
            fleet_solver_gpu.fleet_solver.THREADS = self.threads
        self._solve_many = fleet_solver_gpu.solve_many
        import gpu_lib
        probe = [[1] * 8 for _ in range(8)]
        for i in range(8):
            probe[i][i] = 0
        with gpu_lib.GpuClique(probe, walkers=4) as gpu:
            info = gpu.info()
        return "gpu backend, THREADS=%s, %s" % (
            fleet_solver_gpu.fleet_solver.THREADS, info)

    def solve(self, matrix, budget, k):
        # A base92 string rather than a matrix: decode it HERE, in the worker's
        # own process, so the dispatcher's HTTP thread never holds the GIL for
        # it. The decode comes out of this round's budget because that is where
        # it is really being spent.
        if isinstance(matrix, str):
            started = time.monotonic()
            matrix = _decode(matrix)
            budget -= time.monotonic() - started
        assert budget > 0, "no budget left after decoding"
        if self.kind == "fake":
            pool = _fake_pool(len(matrix), budget, k)
            return pool, _counts(pool)
        pool = self._solve_many(matrix, budget, k)
        if self.kind == "gpu":
            import fleet_solver_gpu
            stats = fleet_solver_gpu.last_stats()
            # Only these three cross the process boundary. The rest of
            # last_stats() is device counters for research; shipping them back
            # per round costs pickling time inside the deadline.
            return pool, {
                "n_top_true": stats.get("n_top_true", 0),
                "n_spare_true": stats.get("n_spare_true", 0),
                "hits": [int(h) for h in stats.get("hits", [])],
            }
        return pool, _counts(pool)


def _decode(encoded):
    """base92 -> adjacency matrix, importing CliqueAI lazily.

    Lazy because a worker that never sees an encoded payload should not pay for
    the import, and because this runs after the process has been pinned.
    """
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(here))
    if root not in sys.path:
        sys.path.insert(0, root)
    from CliqueAI.graph.codec import GraphCodec
    return GraphCodec().decode_matrix(encoded)


def _counts(pool):
    if not pool:
        return {"n_top_true": 0, "n_spare_true": 0}
    omega = max(len(c) for c in pool)
    return {
        "n_top_true": sum(1 for c in pool if len(c) == omega),
        "n_spare_true": sum(1 for c in pool if len(c) < omega),
    }


def _fake_pool(n, budget, k):
    """Synthetic pool: k omega-cliques then k one shorter, deterministic.

    Sleeps for the budget so deadline and admission logic are exercised the way
    a real solve exercises them.
    """
    time.sleep(min(budget, 0.05))
    omega = max(3, min(20, n // 10))
    out = [list(range(i, i + omega)) for i in range(k)]
    out += [list(range(100 + i, 100 + i + omega - 1)) for i in range(k)]
    return out


def pin_cores(cores):
    """Restrict this process to `cores`, enforced by the scheduler.

    The shim divides the CPU budget by arithmetic -- available_cores() // fleet
    -- but that only works between processes that can see each other. A miner
    falling back locally computes its share from the cgroup quota and has no
    idea the dispatcher's workers hold most of it. Affinity makes the split a
    fact rather than an agreement.
    """
    if not cores:
        return None
    try:
        os.sched_setaffinity(0, set(cores))
        return sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return None


def worker_main(kind, device, threads, req_q, res_q, ready_q, cores=None):
    """Process entry point: pin, warm, then serve one task at a time."""
    if kind == "gpu" and device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    pinned = pin_cores(cores)
    backend = Backend(kind, threads)
    try:
        ready_q.put(("ok", "%s cores=%s" % (backend.warm(), pinned)))
    except Exception as exc:                      # a worker that cannot warm is
        ready_q.put(("error", repr(exc)))         # useless; the pool drops it
        return
    while True:
        job = req_q.get()
        if job is None:
            return
        job_id, matrix, budget, k = job
        started = time.monotonic()
        try:
            pool = backend.solve(matrix, budget, k)
            res_q.put((job_id, "ok", pool, time.monotonic() - started))
        except Exception as exc:
            res_q.put((job_id, "error", repr(exc), time.monotonic() - started))
