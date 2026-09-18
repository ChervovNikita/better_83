#!/usr/bin/env python3

import json
import os
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import fleet_solver_gpu
import pick_derived
import minimax

LATENCY_S = 2.0

FLEET_N = 0
POOL_CACHE = ""
POOL_CAP = 8192
PICKER = "blind"
# Fleet size at or above which the maximin allocation replaces the derived one.
# None disables the switch. Set from --minimax-n / configure(minimax_n=...).
MINIMAX_N = None
POOL_DUMP = ""

_cache = None
_cache_lock = threading.Lock()
_dump_lock = threading.Lock()


# Seed for the within-level pool shuffle (pick_derived.shuffle_levels). Fixed
# here so a simulator run is reproducible and paired comparisons hold; the
# dispatcher keys the same shuffle with a secret instead. None = the old
# vertex-id order, kept only to measure against.
SHUFFLE_SEED = 0


def configure(fleet_n, pool_cache="", pool_dump="", picker="blind", minimax_n=None,
              harvest_cap_s=None, solve_budget_s=None, shuffle_seed=0):
    global FLEET_N, POOL_CACHE, POOL_DUMP, PICKER, MINIMAX_N, HARVEST_CAP_S
    global SOLVE_BUDGET_S, SHUFFLE_SEED, _cache
    SHUFFLE_SEED = None if shuffle_seed is None else int(shuffle_seed)
    HARVEST_CAP_S = None if harvest_cap_s is None else float(harvest_cap_s)
    SOLVE_BUDGET_S = None if solve_budget_s is None else float(solve_budget_s)
    MINIMAX_N = None if minimax_n is None else int(minimax_n)
    assert fleet_n > 0, fleet_n
    assert picker in _PICKERS, picker
    FLEET_N = int(fleet_n)
    POOL_CACHE = pool_cache
    POOL_DUMP = pool_dump
    PICKER = picker
    _cache = None


_PICKERS = {
    "blind": lambda *a, **k: pick_derived.picker(*a, **k),
    "oracle": lambda *a, **k: pick_derived.picker_oracle(*a, **k),
    "partial": lambda *a, **k: pick_derived.picker_partial(*a, **k),
    "minimax": lambda *a, **k: minimax.picker(*a, **k),
}


def effective_picker():
    """The picker that will actually run, after the fleet-size switch.

    MINIMAX_N only overrides the derived picker. Asking for oracle/partial with a
    threshold set is a contradiction -- those are measurement baselines, not
    deployable -- so it is refused rather than silently ignored.
    """
    if MINIMAX_N is None:
        return PICKER
    assert PICKER in ("blind", "minimax"), (
        "--minimax-n applies to the deployed picker; it cannot be combined with "
        "--picker %s" % PICKER)
    if PICKER == "minimax":
        return "minimax"
    return "minimax" if FLEET_N >= MINIMAX_N else "blind"


def solve_many(matrix, time_limit, k):
    return fleet_solver_gpu.solve_many(matrix, time_limit, k)


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


def _cache_get(uuid, want):
    c = _cache_load()
    rec = c.get((str(uuid), want))
    if rec is not None:
        return rec
    bigger = sorted(kk for (u, kk) in c if u == str(uuid) and kk >= want)
    if not bigger:
        return None
    src = c[(str(uuid), bigger[0])]
    om = max(len(x) for x in src["pool"]) if src["pool"] else 0
    keep, seen_t, seen_s = [], 0, 0
    for cl in src["pool"]:
        if len(cl) == om and seen_t < want:
            keep.append(cl)
            seen_t += 1
        elif len(cl) == om - 1 and seen_s < want:
            keep.append(cl)
            seen_s += 1
    return dict(src, pool=keep, hits=src["hits"][:len(keep)])


def _dump_pool(uuid, hotkeys, matrix, time_limit, pool, stats, answers):
    full = stats.get("full_pool_unverified")
    record = {
        "uuid": str(uuid),
        "n_nodes": int(matrix.shape[0]),
        "time_limit": float(time_limit),
        "hotkeys": list(hotkeys),
        "omega": max(len(c) for c in pool),
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


# A hard ceiling on the harvest, in seconds, applied to EVERY round. The harvest
# has no early exit, so without this it spends the whole round budget re-finding
# cliques it already holds: measured pool size is identical whether it is given
# 1.5s or 28s. None keeps the old behaviour of spending the whole budget.
HARVEST_CAP_S = None
# Whole-solver wall clock (harvest + picker). None = bounded only by the round.
SOLVE_BUDGET_S = None

PICKER_SAFETY_S = 0.12   # covers watchdog thread start/join granularity
_PICKERS_NEED_DEADLINE = {"minimax": True}


def solve(hotkeys, adjacency_matrix, time_limit, uuid):
    assert hotkeys
    assert FLEET_N > 0, "call solver.configure() first"
    t_start = time.monotonic()
    # SOLVE_BUDGET_S is the whole solver's wall clock -- harvest AND picker -- so
    # the miner answers within SOLVE_BUDGET_S + network latency. The harvest gets
    # HARVEST_CAP_S of it and whatever it leaves goes to the picker.
    budget = time_limit - LATENCY_S
    if SOLVE_BUDGET_S is not None:
        budget = min(budget, float(SOLVE_BUDGET_S))
    harvest_budget = budget
    if HARVEST_CAP_S is not None:
        harvest_budget = min(harvest_budget, float(HARVEST_CAP_S))
    assert harvest_budget > 0, time_limit
    matrix = np.asarray(adjacency_matrix, dtype=np.uint8)
    want = POOL_CAP

    cached = _cache_get(uuid, want) if POOL_CACHE else None
    if cached is not None:
        pool = [list(c) for c in cached["pool"]]
        stats = {"n_top_true": cached["n_top_true"],
                 "n_spare_true": cached["n_spare_true"],
                 "hits": cached["hits"]}
    else:
        pool = solve_many(matrix, harvest_budget, want)
        stats = fleet_solver_gpu.last_stats()
        if POOL_CACHE:
            _cache_put(uuid, want, pool, stats)

    if SHUFFLE_SEED is not None:
        pool, shuffled_hits = pick_derived.shuffle_levels(
            pool, stats.get("hits", []),
            ("sim|%d|%s" % (SHUFFLE_SEED, uuid)).encode())
        stats = dict(stats, hits=shuffled_hits)
    fn = _PICKERS[effective_picker()]
    kw = {}
    if _PICKERS_NEED_DEADLINE.get(effective_picker()):
        # whatever is left of the round after the harvest. The picker must fit in
        # it: going late replaces every one of our answers with [].
        kw["deadline_s"] = max(0.0, budget - (time.monotonic() - t_start)
                               - PICKER_SAFETY_S)
    answers = fn(
        pool, uuid, list(hotkeys),
        n_nodes=matrix.shape[0],
        hits=list(stats.get("hits", [])),
        n_top_true=stats.get("n_top_true", 0),
        n_spare_true=stats.get("n_spare_true", 0),
        fleet_n=FLEET_N,
        **kw
    )
    assert len(answers) == len(hotkeys)
    if POOL_DUMP:
        _dump_pool(uuid, hotkeys, matrix, time_limit, pool, stats, answers)
    return answers
