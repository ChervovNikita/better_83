"""Exhaustive search over every board A can commit, where it is affordable."""

import ctypes
import math
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_LIBX = ctypes.CDLL(os.path.join(_HERE, "libbsx.so"))
_LIBX.bs_exhaustive_a.argtypes = [
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_long,
    ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_long)]
_LIBX.bs_exhaustive_a.restype = ctypes.c_double


def best_a_board(rnd, q_a, q_b, budget):
    """Returns (objective, board, boards_enumerated), or None if over budget."""
    out_board = (ctypes.c_int * (3 * 4096))()
    out_n = ctypes.c_int(0)
    out_count = ctypes.c_long(0)
    value = _LIBX.bs_exhaustive_a(
        q_a, q_b, rnd.omega, rnd.n_top, rnd.n_spare, rnd.difficulty,
        float(rnd.fleet_a), float(rnd.fleet_b), budget, out_board,
        ctypes.byref(out_n), ctypes.byref(out_count))
    if math.isnan(value):
        return None
    board = [(out_board[3 * i], out_board[3 * i + 1], out_board[3 * i + 2])
             for i in range(out_n.value)]
    return value, board, out_count.value
