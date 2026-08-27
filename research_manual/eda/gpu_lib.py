#!/usr/bin/env python3
"""Build and ctypes bridge for clique_gpu.cu."""

import ctypes
import os
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "clique_gpu.cu")

NVCC = os.environ.get("SN83_NVCC") or next(
    (p for p in ("/usr/local/cuda/bin/nvcc", "/usr/bin/nvcc",
                 "/usr/local/cuda-12/bin/nvcc") if os.path.exists(p)), "nvcc")
ARCH = os.environ.get("SN83_GPU_ARCH", "86")

# Mirrors the counter enum in clique_gpu.cu.
CTR_NAMES = [
    "jobs", "new", "dup", "short", "drop", "banback", "steps", "synth",
    "enq", "done", "overflow", "kmaxhit", "dupmax", "newmax", "_14", "_15",
]

MAX_BANS = 8

_libs = {}


def lib_path(lanes=32, prefix=False):
    return os.path.join(
        HERE, "libcliquegpu_l%d%s.so" % (lanes, "_prefix" if prefix else ""))


def build(lanes=32, prefix=False):
    """Compiles one arm, keeping the -Xptxas -v report beside the .so."""
    out = lib_path(lanes, prefix)
    cmd = [
        NVCC, "-O3", "-std=c++14",
        "-gencode", "arch=compute_%s,code=[sm_%s,compute_%s]" % (ARCH, ARCH, ARCH),
        "-DSN83_LANES=%d" % lanes,
        "-DSN83_CANDS_PREFIX=%d" % (1 if prefix else 0),
        "-Xcompiler", "-fPIC", "-Xptxas", "-v",
        "-shared", SRC, "-o", out,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    with open(out + ".ptxas", "w") as handle:
        handle.write(" ".join(cmd) + "\n\n" + proc.stderr)
    assert proc.returncode == 0, "nvcc failed:\n" + proc.stderr
    return out


def load(lanes=32, prefix=False):
    """Returns the .so for one arm, rebuilding it if the source is newer."""
    key = (lanes, bool(prefix))
    if key in _libs:
        return _libs[key]
    path = lib_path(lanes, prefix)
    if not os.path.exists(path) or os.path.getmtime(SRC) > os.path.getmtime(path):
        build(lanes, prefix)
    lib = ctypes.CDLL(path)
    _bind(lib)
    _libs[key] = lib
    return lib


def _bind(lib):
    c = ctypes
    u64 = c.c_uint64
    lib.sn83_gpu_open.restype = c.c_void_p
    lib.sn83_gpu_open.argtypes = [c.POINTER(c.c_uint8), c.c_int, c.c_int]
    lib.sn83_gpu_close.restype = None
    lib.sn83_gpu_close.argtypes = [c.c_void_p]
    lib.sn83_gpu_walkers.restype = c.c_int
    lib.sn83_gpu_walkers.argtypes = [c.c_void_p]
    lib.sn83_gpu_last_error.restype = c.c_int
    lib.sn83_gpu_last_error.argtypes = [c.c_char_p, c.c_int]
    lib.sn83_gpu_device_info.restype = c.c_int
    lib.sn83_gpu_device_info.argtypes = [c.c_char_p, c.c_int]
    lib.sn83_gpu_config.restype = c.c_int
    lib.sn83_gpu_config.argtypes = [c.POINTER(c.c_int)] * 6
    lib.sn83_gpu_probe.restype = c.c_int
    lib.sn83_gpu_probe.argtypes = [
        c.c_void_p, c.c_int, c.c_int, u64, c.POINTER(c.c_double),
        c.POINTER(c.c_longlong), c.POINTER(c.c_int)]
    lib.sn83_gpu_check_cands.restype = c.c_int
    lib.sn83_gpu_check_cands.argtypes = [
        c.c_void_p, u64, c.c_int, c.c_int, c.c_int, c.POINTER(c.c_int)]
    lib.sn83_gpu_check_trajectory.restype = c.c_int
    lib.sn83_gpu_check_trajectory.argtypes = [
        c.c_void_p, u64, c.c_int, c.c_int, c.POINTER(c.c_longlong)]
    lib.sn83_gpu_solve_batch.restype = c.c_int
    lib.sn83_gpu_solve_batch.argtypes = [
        c.c_void_p, c.POINTER(u64), c.POINTER(c.c_int), c.POINTER(c.c_int),
        c.c_int, c.c_int, c.c_double, c.POINTER(c.c_int), c.POINTER(c.c_int),
        c.POINTER(u64), c.POINTER(c.c_longlong)]
    lib.sn83_gpu_harvest.restype = c.c_int
    lib.sn83_gpu_harvest.argtypes = [
        c.c_void_p, c.c_double, u64, c.c_int, c.c_int, c.c_int, c.c_int,
        c.c_int, c.POINTER(c.c_int), c.c_int, c.POINTER(c.c_int),
        c.POINTER(c.c_int), c.POINTER(c.c_int), c.c_int,
        c.POINTER(c.c_longlong)]


def verify(A, clique):
    """Returns (is_clique, is_maximal_in_A)."""
    vs = list(clique)
    m = len(vs)
    if m == 0 or len(set(vs)) != m:
        return False, False
    if min(vs) < 0 or max(vs) >= A.shape[0]:
        return False, False
    sub = A[np.ix_(vs, vs)]
    if int(sub.sum()) != m * (m - 1) or int(np.trace(sub)) != 0:
        return False, False
    cnt = A[vs].sum(axis=0, dtype=np.int32)
    in_c = np.zeros(A.shape[0], dtype=bool)
    in_c[vs] = True
    return True, not bool(np.any((cnt == m) & ~in_c))


def stall(counters):
    """Duplicate fraction among omega-sized results, the design's §6 detector.

    Counted over omega only: the omega-1 spares keep arriving new long after the
    omega pool is exhausted, which drags the whole-band ratio down.
    """
    hits = counters.get("dupmax", 0) + counters.get("newmax", 0)
    return counters.get("dupmax", 0) / float(hits) if hits else 0.0


class GpuClique(object):
    """One graph resident on the device, many searches over it."""

    def __init__(self, adjacency_matrix, lanes=32, prefix=False, walkers=0):
        self.lib = load(lanes, prefix)
        self.lanes = lanes
        self.prefix = bool(prefix)
        A = np.ascontiguousarray(adjacency_matrix, dtype=np.uint8)
        assert A.ndim == 2 and A.shape[0] == A.shape[1], A.shape
        self.n = int(A.shape[0])
        self._A = A
        self.h = self.lib.sn83_gpu_open(
            A.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), self.n, int(walkers))
        assert self.h, "sn83_gpu_open: " + self.last_error()
        cfg = [ctypes.c_int(0) for _ in range(6)]
        self.lib.sn83_gpu_config(*[ctypes.byref(x) for x in cfg])
        self.kmax = int(cfg[2].value)
        self.maxn = int(cfg[3].value)
        self.n_ctr = int(cfg[5].value)
        self.walkers = int(self.lib.sn83_gpu_walkers(self.h))

    def close(self):
        if getattr(self, "h", None):
            self.lib.sn83_gpu_close(self.h)
            self.h = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def last_error(self):
        buf = ctypes.create_string_buffer(512)
        self.lib.sn83_gpu_last_error(buf, 512)
        return buf.value.decode()

    def info(self):
        buf = ctypes.create_string_buffer(512)
        self.lib.sn83_gpu_device_info(buf, 512)
        return buf.value.decode()

    def verify(self, clique):
        return verify(self._A, clique)

    def probe(self, n_walkers=0, max_steps=20000, seed=1):
        """Runs one job per walker at fixed max_steps. Returns (secs, steps, best)."""
        w = n_walkers or self.walkers
        steps = np.zeros(w, dtype=np.int64)
        best = np.zeros(w, dtype=np.int32)
        secs = ctypes.c_double(0.0)
        got = self.lib.sn83_gpu_probe(
            self.h, w, int(max_steps), ctypes.c_uint64(seed), ctypes.byref(secs),
            steps.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)),
            best.ctypes.data_as(ctypes.POINTER(ctypes.c_int)))
        assert got >= 0, "probe: " + self.last_error()
        return secs.value, steps[:got], best[:got]

    def check_cands(self, trials=512, max_clique=40, n_bans=0, seed=7):
        """Diffs device cand0/cand1 against brute force. Returns (trials, bad)."""
        bad = ctypes.c_int(0)
        got = self.lib.sn83_gpu_check_cands(
            self.h, ctypes.c_uint64(seed), int(trials), int(max_clique),
            int(n_bans), ctypes.byref(bad))
        assert got >= 0, "check_cands: " + self.last_error()
        return got, bad.value

    def check_trajectory(self, seed=1, max_steps=2000, n_bans=0):
        """Diffs a one-walker device run against the host mirror."""
        out = (ctypes.c_longlong * 6)()
        rc = self.lib.sn83_gpu_check_trajectory(
            self.h, ctypes.c_uint64(seed), int(max_steps), int(n_bans), out)
        assert rc == 0, "check_trajectory: " + self.last_error()
        mask = (1 << 64) - 1
        return {
            "dev_size": int(out[0]), "ref_size": int(out[1]),
            "dev_traj": int(out[2]) & mask, "ref_traj": int(out[3]) & mask,
            "ref_steps": int(out[4]), "same_clique": bool(out[5]),
        }

    def solve_batch(self, seeds, bans=None, max_steps=20000, time_limit=0.0):
        """Runs a static job list with no dynamic enqueue."""
        seeds = np.ascontiguousarray(seeds, dtype=np.uint64)
        n_jobs = int(seeds.size)
        ban_arr = np.full((n_jobs, MAX_BANS), -1, dtype=np.int32)
        n_bans = np.zeros(n_jobs, dtype=np.int32)
        for i, ban_list in enumerate(bans or []):
            ban_list = list(ban_list)[:MAX_BANS]
            n_bans[i] = len(ban_list)
            ban_arr[i, :len(ban_list)] = ban_list
        sizes = np.zeros(n_jobs, dtype=np.int32)
        verts = np.zeros((n_jobs, self.kmax), dtype=np.int32)
        traj = np.zeros(n_jobs, dtype=np.uint64)
        ctr = np.zeros(self.n_ctr, dtype=np.int64)
        got = self.lib.sn83_gpu_solve_batch(
            self.h, seeds.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            ban_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            n_bans.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), n_jobs,
            int(max_steps), float(time_limit),
            sizes.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            verts.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            traj.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            ctr.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)))
        assert got >= 0, "solve_batch: " + self.last_error()
        out = [sorted(int(v) for v in verts[i, :sizes[i]]) for i in range(n_jobs)]
        return out, dict(zip(CTR_NAMES, ctr.tolist()))

    def harvest(self, time_limit, seed=1, max_steps=20000, n_boot=0,
                boot_steps=0, max_steps_cap=1 << 20, spare_margin=1,
                init_clique=None, max_out=4096):
        """Runs to the deadline. Returns (cliques, counters, hits).

        Every clique is maximal in the full graph. There is no early exit: the
        budget is fixed, latency earns nothing, and stopping early costs both
        omega-cliques the fleet still needs and any chance of finding omega+1.

        `hits` is each clique's BASIN SIZE -- how many independent jobs converged
        on it. A large basin is what makes a clique easy for every other solver
        to find too, so this is the device's own estimate of how crowded that
        clique will be in the field.
        """
        sizes = np.zeros(max_out, dtype=np.int32)
        verts = np.zeros((max_out, self.kmax), dtype=np.int32)
        hits = np.zeros(max_out, dtype=np.int32)
        ctr = np.zeros(self.n_ctr, dtype=np.int64)
        init = np.ascontiguousarray(list(init_clique or []), dtype=np.int32)
        got = self.lib.sn83_gpu_harvest(
            self.h, float(time_limit), ctypes.c_uint64(seed), int(max_steps),
            int(n_boot), int(boot_steps), int(max_steps_cap),
            int(spare_margin),
            init.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), int(init.size),
            sizes.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            verts.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            hits.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), int(max_out),
            ctr.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)))
        assert got >= 0, "harvest: " + self.last_error()
        rows = [(sorted(int(v) for v in verts[i, :sizes[i]]), int(hits[i]))
                for i in range(got)]
        rows.sort(key=lambda r: (-len(r[0]), -r[1]))
        return ([c for c, _h in rows], dict(zip(CTR_NAMES, ctr.tolist())),
                [h for _c, h in rows])
