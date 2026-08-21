"""Generate N DISTINCT maximal cliques from one solve — one per hotkey.

    from fleetsolver import solve_many
    cliques = solve_many(adjacency, time_limit, want=100)

Returns a list of vertex-index lists. Every clique is maximal, every clique is at the
best size the search found, and no two are identical — so assigning one per hotkey
guarantees the fleet never self-collides.

Sizing, measured on recent_val: the number of DISTINCT maximum cliques reachable in
one deadline is roughly

    8 chains -> ~5     32 chains -> ~14     64 chains -> ~23

so `want` beyond a few dozen will usually come back short. That is a property of the
graphs, not of the budget: going 8 -> 64 chains finds 4x more distinct optima and
zero additional cliques the field had not already found.

WHEN SHORT, DUPLICATE — DO NOT SHRINK. Two miners sharing a max-size clique score
1.85 + 0.5 = 2.35 each. A unique clique one vertex smaller scores about
0.4*1.85 + 1.0 = 1.74, because `pr` is measured against a field that is overwhelmingly
at max size. Spread duplicates evenly: d miners on one clique each get 1/d.
"""
import ctypes
import fcntl
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "native", "variants", "v25_multi.cpp")
LIB = os.path.join(HERE, "native", "variants", "lib_v25_multi.so")

DEFAULT_THREADS = int(os.environ.get("SN83_THREADS", "8"))
GUARD_S = float(os.environ.get("SN83_GUARD", "0.03"))

_lib = None


def _load():
    global _lib
    if _lib is not None:
        return _lib
    with open(os.path.join(HERE, "native", "variants", ".v25.lock"), "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        if (not os.path.exists(LIB)) or os.path.getmtime(SRC) > os.path.getmtime(LIB):
            subprocess.run(
                f"g++ -O3 -march=native -funroll-loops -std=c++17 -pthread "
                f"-shared -fPIC {SRC} -o {LIB}",
                shell=True, check=True, stdout=sys.stderr.fileno())
    lib = ctypes.CDLL(LIB)
    lib.sn83_solve_many.restype = ctypes.c_int
    lib.sn83_solve_many.argtypes = [
        ctypes.POINTER(ctypes.c_uint8), ctypes.c_int, ctypes.c_double, ctypes.c_uint64,
        ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32)]
    _lib = lib
    return lib


def solve_many(adjacency, time_limit, want, seed=0, threads=None):
    """Up to `want` distinct maximal cliques, all at the best size found."""
    lib = _load()
    A = np.ascontiguousarray(adjacency, dtype=np.uint8)
    n = A.shape[0]
    out = np.zeros(n * max(want, 1), dtype=np.int32)
    sizes = np.zeros(max(want, 1), dtype=np.int32)
    budget = max(0.01, float(time_limit) - GUARD_S)
    got = lib.sn83_solve_many(
        A.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), n, budget,
        ctypes.c_uint64(seed), int(DEFAULT_THREADS if threads is None else threads),
        int(want), out.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        sizes.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
    res, off = [], 0
    for i in range(got):
        k = int(sizes[i])
        res.append(out[off:off + k].tolist())
        off += k
    return res


def assign(cliques, n_hotkeys):
    """Map `n_hotkeys` miners onto the available cliques, spreading duplicates evenly.

    Returns a list of length n_hotkeys. When there are fewer cliques than hotkeys the
    load is balanced, so the worst-case group size is ceil(n_hotkeys/len(cliques))
    rather than everyone piling onto one answer.
    """
    if not cliques:
        return []
    return [cliques[i % len(cliques)] for i in range(n_hotkeys)]


def check(A, cliques):
    """Every clique valid and maximal, and all pairwise distinct."""
    n = A.shape[0]
    seen = set()
    for c in cliques:
        if not c or len(set(c)) != len(c):
            return False, "empty or repeated vertex"
        idx = np.array(c, dtype=int)
        if idx.min() < 0 or idx.max() >= n:
            return False, "vertex out of range"
        if A[np.ix_(idx, idx)].sum() != len(c) * (len(c) - 1):
            return False, "not a clique"
        cnt = A[idx].sum(axis=0)
        inC = np.zeros(n, dtype=bool)
        inC[idx] = True
        if np.any((cnt == len(c)) & (~inC)):
            return False, "not maximal"
        key = tuple(sorted(c))
        if key in seen:
            return False, "duplicate clique returned"
        seen.add(key)
    return True, "ok"
