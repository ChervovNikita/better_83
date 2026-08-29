"""ctypes bridge to the C++ solvers."""

import ctypes
import os
import subprocess

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "native.cpp")
_LIB = os.path.join(_HERE, "libbs.so")
_MAX_BOARD = 4096


def _build():
    stale = (not os.path.exists(_LIB)
             or os.path.getmtime(_LIB) < os.path.getmtime(_SRC))
    if stale:
        subprocess.check_call(["g++", "-O3", "-march=native", "-shared",
                               "-fPIC", "-o", _LIB, _SRC])


_build()
_lib = ctypes.CDLL(_LIB)
_lib.bs_score.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int,
                          ctypes.c_double, ctypes.POINTER(ctypes.c_double)]
_lib.bs_score.restype = None
_lib.bs_best_response.argtypes = [
    ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_double,
    ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_double)]
_lib.bs_best_response.restype = ctypes.c_double


def score(board, difficulty):
    """Returns the mean reward of each player."""
    flat = []
    for entry in board:
        mult = entry[3] if len(entry) > 3 else 1
        flat += [entry[0], entry[1], entry[2], mult]
    buf = (ctypes.c_int * len(flat))(*flat)
    out = (ctypes.c_double * 2)()
    _lib.bs_score(buf, len(board), difficulty, out)
    return out[0], out[1]


def best_response(board, rnd, q):
    """Returns the board and both means after the second player answers."""
    assert q >= 0
    if q == 0:
        mean_a, mean_b = score(board, rnd.difficulty)
        return list(board), mean_a, mean_b
    flat = []
    for size, n_a, n_b in board:
        flat += [size, n_a, n_b]
    buf = (ctypes.c_int * len(flat))(*flat)
    out_board = (ctypes.c_int * (3 * _MAX_BOARD))()
    out_n = ctypes.c_int(0)
    out_means = (ctypes.c_double * 2)()
    _lib.bs_best_response(buf, len(board), q, rnd.omega, rnd.n_top,
                          rnd.n_spare, rnd.difficulty, out_board,
                          ctypes.byref(out_n), out_means)
    n = out_n.value
    trial = [(out_board[3 * i], out_board[3 * i + 1], out_board[3 * i + 2])
             for i in range(n)]
    return trial, out_means[0], out_means[1]


_LIBP = os.path.join(_HERE, "libbsp.so")
_SRCP = os.path.join(_HERE, "native_partial.cpp")

if (not os.path.exists(_LIBP)
        or os.path.getmtime(_LIBP) < os.path.getmtime(_SRCP)):
    subprocess.check_call(["g++", "-O3", "-march=native", "-shared", "-fPIC",
                           "-o", _LIBP, _SRCP])

_libp = ctypes.CDLL(_LIBP)
_IP = ctypes.POINTER(ctypes.c_int)
_libp.bsp_expected.argtypes = [_IP, _IP, ctypes.c_int, _IP, _IP, ctypes.c_int,
                               _IP, _IP, ctypes.c_int, ctypes.c_int,
                               ctypes.c_double, ctypes.POINTER(ctypes.c_double)]
_libp.bsp_expected.restype = None


def _ints(values):
    return (ctypes.c_int * max(1, len(values)))(*(values or [0]))


def expected_scores(plan, difficulty, omega):
    """Returns the expected mean of each player over the random matching."""
    top = plan.get(omega, ([], []))
    spare = plan.get(omega - 1, ([], []))
    fresh = plan.get("fresh", [])
    out = (ctypes.c_double * 2)()
    _libp.bsp_expected(_ints(list(top[0])), _ints(list(top[1])), len(top[0]),
                       _ints(list(spare[0])), _ints(list(spare[1])),
                       len(spare[0]),
                       _ints([s for s, _d in fresh]),
                       _ints([d for _s, d in fresh]), len(fresh),
                       omega, difficulty, out)
    return out[0], out[1]
