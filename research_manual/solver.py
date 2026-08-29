#!/usr/bin/env python3

import json
import os
import sys
import threading

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "eda"))

import inspect

LATENCY_S = 2.0

PICKER = os.environ.get("SN83_PICKER", "value").lower()
assert PICKER in ("static", "value", "legacy"), PICKER

if PICKER == "legacy":
    from fleet_pick import picker
elif PICKER == "value":
    from pick_value import picker
else:
    from pick_static import picker

PICKER_WANTS_N = "n_nodes" in inspect.signature(picker).parameters

SOLVER = os.environ.get("SN83_SOLVER", "gpu").lower()
assert SOLVER in ("gpu", "cpu"), SOLVER

_solver_mod = None


def _load_solver():
    global _solver_mod
    if _solver_mod is None:
        if SOLVER == "cpu":
            import fleet_solver as mod
        else:
            import fleet_solver_gpu as mod
        _solver_mod = mod
    return _solver_mod


def solve_many(*args, **kwargs):
    return _load_solver().solve_many(*args, **kwargs)

# The pool handed to the picker is capped at k of each size, but crowding is
# a_hat divided by how many cliques EXIST -- using the capped length inflates it
# by the truncation factor and makes the value picker duplicate needlessly.
PICKER_WANTS_SUPPLY = "n_top_true" in inspect.signature(picker).parameters


POOL_DUMP = os.environ.get("SN83_POOL_DUMP", "")
_dump_lock = threading.Lock()


def _dump_pool(uuid, hotkeys, matrix, time_limit, pool, stats, answers):
    """Append everything the solver found, so pickers can be replayed offline.

    The simulator keeps only the one clique per hotkey that the picker chose, so
    a picker experiment otherwise costs a full GPU re-run.
    """
    omega = max(len(c) for c in pool)
    full = stats.get("full_pool_unverified")
    record = {
        "uuid": str(uuid),
        "n_nodes": int(matrix.shape[0]),
        "time_limit": float(time_limit),
        "hotkeys": list(hotkeys),
        "omega": omega,
        "pool": [[int(v) for v in c] for c in pool],
        "hits": [int(h) for h in stats.get("hits", [])],
        "full_pool_unverified": [[int(v) for v in c] for c in (full or [])],
        "full_hits": [int(h) for h in stats.get("full_hits", [])],
        "n_top_true": int(stats.get("n_top_true", 0)),
        "n_spare_true": int(stats.get("n_spare_true", 0)),
        "answers": [[int(v) for v in a] for a in answers],
    }
    with _dump_lock, open(POOL_DUMP, "a") as handle:
        handle.write(json.dumps(record) + "\n")


def solve(hotkeys, adjacency_matrix, time_limit, uuid):
    assert hotkeys
    budget = time_limit - LATENCY_S
    assert budget > 0, time_limit
    matrix = np.asarray(adjacency_matrix, dtype=np.uint8)
    pool = solve_many(matrix, budget, len(hotkeys))
    kwargs = {}
    if PICKER_WANTS_N:
        kwargs["n_nodes"] = matrix.shape[0]
    if PICKER_WANTS_SUPPLY:
        stats = getattr(_load_solver(), "last_stats", dict)()
        kwargs["n_top_true"] = stats.get("n_top_true", 0)
        kwargs["n_spare_true"] = stats.get("n_spare_true", 0)
    answers = picker(pool, uuid, list(hotkeys), **kwargs)
    assert len(answers) == len(hotkeys)
    if POOL_DUMP:
        _dump_pool(uuid, hotkeys, matrix, time_limit, pool,
                   getattr(_load_solver(), "last_stats", dict)(), answers)
    return answers
