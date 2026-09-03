#!/usr/bin/env python3

import json
import os
import sys
import threading

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import fleet_solver_gpu
import pick_derived

LATENCY_S = 2.0

FLEET_N = 0
POOL_CACHE = ""
POOL_K_MULT = 1
POOL_DUMP = ""

_cache = None
_cache_lock = threading.Lock()
_dump_lock = threading.Lock()


def configure(fleet_n, pool_cache="", pool_k_mult=1, pool_dump=""):
    global FLEET_N, POOL_CACHE, POOL_K_MULT, POOL_DUMP, _cache
    assert fleet_n > 0, fleet_n
    assert pool_k_mult >= 1, pool_k_mult
    FLEET_N = int(fleet_n)
    POOL_CACHE = pool_cache
    POOL_K_MULT = int(pool_k_mult)
    POOL_DUMP = pool_dump
    _cache = None


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


def solve(hotkeys, adjacency_matrix, time_limit, uuid):
    assert hotkeys
    assert FLEET_N > 0, "call solver.configure() first"
    budget = time_limit - LATENCY_S
    assert budget > 0, time_limit
    matrix = np.asarray(adjacency_matrix, dtype=np.uint8)
    want = len(hotkeys) * POOL_K_MULT

    cached = _cache_get(uuid, want) if POOL_CACHE else None
    if cached is not None:
        pool = [list(c) for c in cached["pool"]]
        stats = {"n_top_true": cached["n_top_true"],
                 "n_spare_true": cached["n_spare_true"],
                 "hits": cached["hits"]}
    else:
        pool = solve_many(matrix, budget, want)
        stats = fleet_solver_gpu.last_stats()
        if POOL_CACHE:
            _cache_put(uuid, want, pool, stats)

    answers = pick_derived.picker(
        pool, uuid, list(hotkeys),
        n_nodes=matrix.shape[0],
        hits=list(stats.get("hits", [])),
        n_top_true=stats.get("n_top_true", 0),
        n_spare_true=stats.get("n_spare_true", 0),
        fleet_n=FLEET_N,
    )
    assert len(answers) == len(hotkeys)
    if POOL_DUMP:
        _dump_pool(uuid, hotkeys, matrix, time_limit, pool, stats, answers)
    return answers
