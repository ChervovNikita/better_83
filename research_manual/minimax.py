#!/usr/bin/env python3
"""Maximin allocation -- a thin adapter over the solver in eda/big_simulation.

The maximin board is NOT computed here. It comes from
eda/big_simulation/strategies.py + native.py, which is the implementation under
test by the bound sweep (bound.py --strategy maximin) and is the one whose
crossover point decides when this picker should be used at all. Writing a second
implementation here means auditing two; this file only translates between the
picker's contract and that solver's.

What the borrowed solver does, and why it is the one to trust:

  a_candidates() sweeps every (split, width) pair on the width/minimum frontier.
  A's diversity depends on its own board only through how many distinct cliques
  it occupies and the smallest count on any of them, so sweeping the target
  minimum covers the frontier -- width is searched, not assumed.

  native.best_response() computes the opponent's exact reply in C++, and
  native.maximin() maximises the fleet-weighted margin w_a*mean_a - w_b*mean_b
  over that family.

  estimate_q_b() estimates the opponent's answer count as
  Binomial(fleet_b, selection_p(D)) -- measured 12% relative error against 23%
  for the p*METAGRAPH - q_a alternative.
"""
import ctypes
import os
import subprocess
import sys
import threading
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_BS = os.path.join(_HERE, "eda", "big_simulation")
_SRC = os.path.join(_HERE, "minimax_native.cpp")
_LIB = os.path.join(_HERE, "libminimax.so")
_MAX_BOARD = 4096
_out_board = (ctypes.c_int * (3 * _MAX_BOARD))()
_out_n = ctypes.c_int(0)
# the output buffers are module-level, so one call at a time. simulate.py is
# sequential today, but solver.py already holds locks for threaded use.
_call_lock = threading.Lock()
# how often the cost guard handed the round back to the derived picker,
# so a run reports what actually played rather than what was requested
STATS = {"maximin": 0, "fallback": 0}


def _build():
    """Our own build of the big_simulation solver, parallelised.

    The algorithm is theirs, unchanged; minimax_native.cpp differs from
    eda/big_simulation/native.cpp only in that the candidate sweep and the hill
    climb evaluate in parallel, picking the best by (value, index) so the result
    does not depend on thread count. We compile our own copy rather than editing
    theirs because their bound sweep compiles that file on demand.
    """
    if (not os.path.exists(_LIB)
            or os.path.getmtime(_LIB) < os.path.getmtime(_SRC)):
        subprocess.check_call(["g++", "-O3", "-march=native", "-fopenmp",
                               "-shared", "-fPIC", "-o", _LIB, _SRC])
    lib = ctypes.CDLL(_LIB)
    lib.bs_maximin.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ctypes.c_double, ctypes.c_int, ctypes.c_int]
    lib.bs_maximin.restype = ctypes.c_double
    return lib


# Built at IMPORT, not on first use: the first call happens inside solver.solve(),
# on the round's critical path, and g++ takes ~19s. libminimax.so is gitignored,
# so a fresh deploy would pay that on its first round and answer nothing.
_lib = _build()


class _Round(object):
    """The fields native.maximin reads off a big_simulation Round."""

    __slots__ = ("uuid", "difficulty", "omega", "n_top", "n_spare",
                 "n_answers", "q_a", "q_b", "q_b_oracle", "fleet_a", "fleet_b")


def _estimate_q_b(difficulty, fleet_b):
    """Their estimator, imported so there is one definition of it."""
    if _BS not in sys.path:
        sys.path.insert(0, _BS)
    import strategies

    class _R(object):
        pass
    r = _R()
    r.difficulty = float(difficulty)
    r.fleet_b = float(fleet_b)
    return strategies.estimate_q_b(r)


def board(difficulty, omega, n_top, n_spare, q_a, fleet_a, fleet_b, deadline_s=0.0):
    """(level, our_count, opponent_count) triples from the borrowed solver.

    deadline_s bounds the search: the candidate sweep stops early and the climb is
    skipped once it is exceeded, returning the best board found so far. The sweep
    alone is a valid allocation, so a deadline costs quality, never correctness.
    0 means unbounded (measurement only -- never on the round's critical path).
    """
    q_b = _estimate_q_b(difficulty, max(1, fleet_b))
    with _call_lock:
        return _call(difficulty, omega, n_top, n_spare, q_a, fleet_a, fleet_b,
                     q_b, deadline_s)


CLIMB_ITERS = 64     # full climb; the search is exact
COARSE = 1           # score EVERY candidate. Coarse sampling was tried and
                     # rejected: it lost up to 0.217 of objective, ~200x the
                     # entire maximin-over-derived advantage.          # 1 = score every candidate; k = score every k-th, then refine


def _call(difficulty, omega, n_top, n_spare, q_a, fleet_a, fleet_b, q_b, deadline_s):
    _lib.bs_maximin(int(q_a), int(q_b), int(omega), int(n_top), int(n_spare),
                    float(difficulty), float(max(1, fleet_a)),
                    float(max(1, fleet_b)), _out_board, ctypes.byref(_out_n),
                    float(deadline_s), int(COARSE), int(CLIMB_ITERS))
    return [(_out_board[3 * i], _out_board[3 * i + 1], _out_board[3 * i + 2])
            for i in range(_out_n.value)]


def picker(pool, uuid, hotkeys, difficulty=None, n_nodes=None, hits=None,
           n_top_true=0, n_spare_true=0, fleet_n=0, deadline_s=None):
    """Same contract as pick_derived.picker, maximin instead of best response."""
    import pick_derived as pd
    if difficulty is None:
        difficulty = pd.difficulty_from_n(n_nodes)
    a = len(hotkeys)
    omega, top, spare = pd._levels(pool)
    n_top = max(int(n_top_true), len(top))
    n_spare = max(int(n_spare_true), len(spare))
    fleet_a = fleet_n or pd.infer_fleet_n(a, difficulty)
    fleet_b = max(1, sum(pd.fleet_profile(fleet_a).values()))

    # Always compute the derived allocation: it costs milliseconds and it is the
    # answer we fall back to. Then run the maximin solve on a watchdog thread. If
    # it does not finish inside the budget we abandon it and return the derived
    # board, so the picker CANNOT overrun -- no cost model, no prediction. A cost
    # model was tried and failed: it has no pool-shape term and under-predicted by
    # 3-10x on real rounds, and bs_maximin only checks its own deadline between
    # whole evals, so it overshot by seconds.
    t0 = time.monotonic()
    # NOTE: pick_derived.picker is itself unbounded. It is the floor cost of any
    # answer and is exactly what the blind path spends, so running it first cannot
    # make us later than blind -- but it does mean the picker's true bound is
    # max(derived_cost, deadline_s), not deadline_s.
    derived = pd.picker(pool, uuid, hotkeys, difficulty=difficulty, n_nodes=n_nodes,
                        hits=hits, n_top_true=n_top_true, n_spare_true=n_spare_true,
                        fleet_n=fleet_n)
    if deadline_s is None:
        budget = None                      # measurement only, never on a live round
    elif deadline_s <= 0.0:
        STATS["fallback"] += 1             # no time left at all
        return derived
    else:
        # the derived fallback has already spent part of the round's remainder
        budget = float(deadline_s) - (time.monotonic() - t0)
        if budget <= 0.0:
            STATS["fallback"] += 1
            return derived

    if not _call_lock.acquire(blocking=False):
        # a previous round's solve is still running and owns the buffers
        STATS["fallback"] += 1
        return derived
    got = {}

    def _work():
        try:
            # deadline 0.0 = let the native solve run to completion. The watchdog
            # already bounds our wall clock, and an unbounded native call is
            # DETERMINISTIC: with a native deadline, which candidates finish before
            # the stop flag depends on thread count and load.
            got["b"] = _call(difficulty, omega, n_top, n_spare, a, fleet_a,
                             fleet_b, _estimate_q_b(difficulty, fleet_b), 0.0)
        except Exception:
            pass
        finally:
            _call_lock.release()

    th = threading.Thread(target=_work, daemon=True)
    th.start()
    th.join(budget)
    if "b" not in got:
        # abandoned: the thread keeps running and will release the lock itself,
        # so the next round falls back rather than blocking
        STATS["fallback"] += 1
        return derived
    brd = got["b"]

    STATS["maximin"] += 1
    at = [0] * len(top)
    asp = [0] * len(spare)
    i_top = i_sp = 0
    for level, count, _opp in brd:
        if count <= 0:
            continue
        if level == omega and i_top < len(at):
            at[i_top] = int(count)
            i_top += 1
        elif level == omega - 1 and i_sp < len(asp):
            asp[i_sp] = int(count)
            i_sp += 1

    # the board is built against n_top/n_spare, which may exceed what this
    # round's pool actually holds; anything that did not fit goes on the first
    # clique we do hold rather than being dropped
    short = a - (sum(at) + sum(asp))
    if short > 0:
        if at:
            at[0] += short
        elif asp:
            asp[0] += short
    return pd._emit(uuid, hotkeys, top, spare, at, asp)
