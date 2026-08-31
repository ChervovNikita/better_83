#!/usr/bin/env python3

import importlib
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

# A bare name selects a built-in; "module:function" loads a candidate picker from
# anywhere on the path, which is what lets a search try many without editing this
# file. The contract is identical either way -- the loaded object is called
# exactly as the built-ins are, so a candidate cannot quietly skip a kwarg.
PICKER = os.environ.get("SN83_PICKER", "value")
_BUILTIN = {"legacy": "fleet_pick", "value": "pick_value", "static": "pick_static"}

if ":" in PICKER:
    _mod_name, _fn_name = PICKER.split(":", 1)
    picker = getattr(importlib.import_module(_mod_name), _fn_name)
else:
    assert PICKER.lower() in _BUILTIN, PICKER
    picker = getattr(importlib.import_module(_BUILTIN[PICKER.lower()]), "picker")

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
# Basin size (how many independent jobs converged on a clique) is the only
# measured signal for which cliques the FIELD will avoid: P(free) rises with
# basin in every supply band. It lives in the solver's stats and was not being
# offered to the picker at all.
PICKER_WANTS_HITS = "hits" in inspect.signature(picker).parameters
# Our own fleet size changes WHO the rivals are: registering hotkeys displaces the
# lowest-incentive miners, which are the smaller operators. A picker that models a
# fixed field is modelling opponents that our own growth deregistered.
PICKER_WANTS_FLEET = "fleet_n" in inspect.signature(picker).parameters
FLEET_N = int(os.environ.get("SN83_FLEET_N", "0"))


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
        "closure_added": int(stats.get("closure_added", 0)),
        "closure_iters": int(stats.get("closure_iters", 0)),
        "answers": [[int(v) for v in a] for a in answers],
    }
    with _dump_lock, open(POOL_DUMP, "a") as handle:
        handle.write(json.dumps(record) + "\n")


# Picker research re-runs the same rounds many times while only the picker
# changes. The solver output is a fixed input to that question, so re-solving it
# per run costs a GPU pass and, worse, makes each run see a DIFFERENT pool -- the
# harvest is stochastic, so the comparison would not be paired. A cache pins the
# pool: every candidate picker is scored on identical input.
#
# The cache is keyed by (uuid, k). A pool is truncated to k of each size class,
# so a cache built at one fleet size is not valid at another and is not reused.
# solve_many caps the pool at k cliques of each size, and k defaults to the
# number of hotkeys we must answer for. That makes the picker choose from a
# shortlist the solver happened to return first, which is not the same as
# choosing from what exists. Oversampling costs nothing at solve time (the
# cliques are already found) and gives the picker room to avoid the field.
POOL_K_MULT = int(os.environ.get("SN83_POOL_K_MULT", "1"))
POOL_CACHE = os.environ.get("SN83_POOL_CACHE", "")
_cache = None
_cache_lock = threading.Lock()


def _cache_load():
    global _cache
    if _cache is None:
        _cache = {}
        if POOL_CACHE and os.path.exists(POOL_CACHE):
            with open(POOL_CACHE) as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        # a writer may be mid-append; a torn trailing line is not
                        # a corrupt cache, it is just the next round arriving
                        continue
                    _cache[(rec["uuid"], int(rec["k"]))] = rec
    return _cache


def _cache_put(uuid, k, pool, stats):
    rec = {
        "uuid": str(uuid), "k": int(k),
        "pool": [[int(v) for v in c] for c in pool],
        "n_top_true": int(stats.get("n_top_true", 0)),
        "n_spare_true": int(stats.get("n_spare_true", 0)),
        "hits": [int(h) for h in stats.get("hits", [])],
    }
    with _cache_lock, open(POOL_CACHE, "a") as handle:
        handle.write(json.dumps(rec) + "\n")
    _cache_load()[(rec["uuid"], rec["k"])] = rec
    return rec


def solve(hotkeys, adjacency_matrix, time_limit, uuid):
    assert hotkeys
    budget = time_limit - LATENCY_S
    assert budget > 0, time_limit
    matrix = np.asarray(adjacency_matrix, dtype=np.uint8)
    k = len(hotkeys)
    cached = _cache_load().get((str(uuid), k * POOL_K_MULT)) if POOL_CACHE else None
    if cached is not None:
        pool = [list(c) for c in cached["pool"]]
        stats = {"n_top_true": cached["n_top_true"],
                 "n_spare_true": cached["n_spare_true"],
                 "hits": cached["hits"]}
    else:
        pool = solve_many(matrix, budget, k * POOL_K_MULT)
        stats = getattr(_load_solver(), "last_stats", dict)()
        if POOL_CACHE:
            _cache_put(uuid, k * POOL_K_MULT, pool, stats)
    kwargs = {}
    if PICKER_WANTS_N:
        kwargs["n_nodes"] = matrix.shape[0]
    if PICKER_WANTS_SUPPLY:
        kwargs["n_top_true"] = stats.get("n_top_true", 0)
        kwargs["n_spare_true"] = stats.get("n_spare_true", 0)
    if PICKER_WANTS_HITS:
        kwargs["hits"] = list(stats.get("hits", []))
    if PICKER_WANTS_FLEET:
        kwargs["fleet_n"] = FLEET_N
    answers = picker(pool, uuid, list(hotkeys), **kwargs)
    assert len(answers) == len(hotkeys)
    if POOL_DUMP:
        _dump_pool(uuid, hotkeys, matrix, time_limit, pool, stats, answers)
    return answers
