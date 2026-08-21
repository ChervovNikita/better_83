"""Solver entry point: thin ctypes bridge onto `native/libclique.so`.

    python3 score_submission.py --solver fastsolver:solve

The shared library is rebuilt automatically when `clique.cpp` is newer than it,
so the edit-run loop is just "edit the C++ and re-run the scorer".

Thread count matters for honesty, not just speed. `min_compute.yml` puts a real
miner at 4 recommended / 8 cores, so that is the default here; this box has 128
and using them all would produce validation numbers that do not transfer to
chain. Override with SN83_THREADS when deliberately measuring something else.
"""
import ctypes
import fcntl
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "native", "clique.cpp")
LIB = os.path.join(HERE, "native", "libclique.so")

DEFAULT_THREADS = int(os.environ.get("SN83_THREADS", "8"))
# Time the C++ side is told to leave for the final maximality pass and the trip
# back through ctypes. A late answer is a zero on chain.
GUARD_S = float(os.environ.get("SN83_GUARD", "0.03"))

_lib = None


def _load(force=False):
    global _lib
    if _lib is not None and not force:
        return _lib
    # Hold a lock across the build: the train harness imports this in a dozen
    # worker processes at once, and they must not write the .so on top of
    # each other.
    with open(os.path.join(HERE, "native", ".build.lock"), "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        if (not os.path.exists(LIB)) or os.path.getmtime(SRC) > os.path.getmtime(LIB):
            subprocess.run(["bash", os.path.join(HERE, "native", "build.sh")],
                           check=True, stdout=sys.stderr.fileno())
    lib = ctypes.CDLL(LIB)
    lib.sn83_solve.restype = ctypes.c_int
    lib.sn83_solve.argtypes = [
        ctypes.POINTER(ctypes.c_uint8), ctypes.c_int, ctypes.c_double,
        ctypes.c_uint64, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int32)]
    _lib = lib
    return lib


def solve(adjacency, time_limit, seed=0, threads=None, bms_k=0, restart_steps=0):
    """Return the vertex indices of a maximal clique.

    adjacency: n x n uint8, symmetric, zero diagonal.  time_limit: seconds.
    """
    lib = _load()
    A = np.ascontiguousarray(adjacency, dtype=np.uint8)
    n = A.shape[0]
    out = np.zeros(n, dtype=np.int32)
    budget = max(0.01, float(time_limit) - GUARD_S)
    size = lib.sn83_solve(
        A.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), n, budget,
        ctypes.c_uint64(seed), int(DEFAULT_THREADS if threads is None else threads),
        int(bms_k), int(restart_steps),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
    return out[:size].tolist()


def check(A, clique):
    """Same validity test the scorer applies: clique, maximal, no duplicates."""
    S = list(clique)
    if not S or len(set(S)) != len(S):
        return False, "empty or repeated vertex"
    idx = np.array(S, dtype=int)
    if idx.min() < 0 or idx.max() >= A.shape[0]:
        return False, "vertex out of range"
    if A[np.ix_(idx, idx)].sum() != len(S) * (len(S) - 1):
        return False, "not a clique"
    cnt = A[idx].sum(axis=0)
    inC = np.zeros(A.shape[0], dtype=bool)
    inC[idx] = True
    if np.any((cnt == len(S)) & (~inC)):
        return False, "not maximal (a vertex can still be added)"
    return True, "ok"


if __name__ == "__main__":                     # tiny self-test on random graphs
    rng = np.random.default_rng(0)
    for n, d in [(300, 0.85), (900, 0.75)]:
        M = (rng.random((n, n)) < d).astype(np.uint8)
        M = np.triu(M, 1)
        M = M + M.T
        c = solve(M, 1.0)
        print(n, d, "size", len(c), check(M, c))
