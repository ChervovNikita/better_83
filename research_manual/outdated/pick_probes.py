"""Falsified research probes. Kept for reference; nothing imports them.

Each was measured against research_manual/simulate.py and lost to picker().
See ~/autoresearch-runs/sn83-floor/RESULTS.md for the numbers.
"""

import os
import sys
import math
import json
import hashlib
import random
import collections

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir))

from pick_derived import *  # noqa: F401,F403
from pick_derived import (_levels, _emit, _best_widths, _field_counter,
                          allocate, eval_J, c_min_of, entity_plan, law_from_plan,
                          expected_cliques, difficulty_from_n, infer_fleet_n, spread)

_hk_state = {}
_score_table = None


FAIR = os.environ.get("SN83_FAIR", "0")


FAIR = os.environ.get("SN83_FAIR", "0")


SCORE_TABLE = os.environ.get("SN83_SCORE_TABLE", "")


FAIR_CUT = float(os.environ.get("SN83_CUT", "0"))


_hk_state = {}


_score_table = None


def _scores_for(uuid):
    global _score_table
    if _score_table is None:
        _score_table = {}
        if SCORE_TABLE:
            with open(SCORE_TABLE) as handle:
                raw = json.load(handle)
            for rid, pairs in raw.items():
                _score_table[rid] = {tuple(k.split(",")): v for k, v in pairs.items()}
    return _score_table.get(str(uuid), {})


def answer_value(difficulty, sigma, cmin, is_top, m, occ):
    """Predicted reward for ONE of our answers -- the validator's R_i, in exactly"""
    opt = 1.0 if is_top else sigma
    if isinstance(occ, dict):
        div = sum(p / float(m + f) for f, p in occ.items())
    else:
        div = 1.0 / float(m + occ)
    return (1.0 + difficulty) * opt + cmin * div


def _emit_feedback(uuid, hotkeys, slots, values, mode):
    """Hand out slots by each hotkey's RUNNING MEAN so far, not by a fixed rank."""
    order = sorted(range(len(slots)), key=lambda i: -values[i])
    def running(h):
        st = _hk_state.get(h)
        return st[0] / st[1] if st and st[1] else 0.0
    concentrate = mode == "2"
    if mode == "3":

        seen = [running(h) for h in hotkeys if _hk_state.get(h, [0, 0])[1]]
        if seen and FAIR_CUT > 0:
            med = sorted(seen)[len(seen) // 2]
            concentrate = med < FAIR_CUT
    rank = sorted(range(len(hotkeys)), key=lambda j: running(hotkeys[j]),
                  reverse=concentrate)
    out = [None] * len(hotkeys)
    realized = _scores_for(uuid) if mode == "3" else {}
    for pos, j in enumerate(rank):
        i = order[pos % len(order)]
        out[j] = list(slots[i])
        st = _hk_state.setdefault(hotkeys[j], [0.0, 0])
        key = tuple(str(v) for v in sorted(slots[i]))
        st[0] += realized.get(key, values[i])
        st[1] += 1
    return out


def _slots_and_values(difficulty, omega, a, top, spare, alloc_top, alloc_sp,
                      occ_top, occ_sp, f_top, f_sp, field_min):
    """The round's answers, each with its predicted reward."""
    a_top = float(sum(alloc_top))
    n = a + f_top + f_sp
    rho = omega / float(omega - 1)
    n_omega = f_top + a_top
    sigma = 1.0 if n_omega <= 0.0 or n <= 0 else math.exp(-(n_omega / n) * rho)
    cmin = c_min_of(alloc_top, alloc_sp, occ_top, occ_sp, field_min)
    slots = []
    values = []
    for alloc, cliques, occ, is_top in ((alloc_top, top, occ_top, True),
                                        (alloc_sp, spare, occ_sp, False)):
        for i, m in enumerate(alloc):
            m = int(m)
            if m <= 0 or i >= len(cliques):
                continue
            o = occ if isinstance(occ, dict) else occ[i]
            v = answer_value(difficulty, sigma, cmin, is_top, m, o)
            for _ in range(m):
                slots.append(list(cliques[i]))
                values.append(v)
    pool = top + spare
    while len(slots) < a:
        slots.append(list(pool[len(slots) % len(pool)]))
        values.append(0.0)
    return slots, values


PRIORITY = os.environ.get("SN83_PRIORITY", "0") == "1"


def _hotkey_rank(hotkey):
    """A hotkey's fixed place in the queue, stable across rounds and rounds-sets."""
    return int(hashlib.sha1(str(hotkey).encode()).hexdigest()[:8], 16)


ATOP_BIAS = int(os.environ.get("SN83_ATOP_BIAS", "0"))


ATOP_EPS = float(os.environ.get("SN83_ATOP_EPS", "0"))


def picker_bias(pool, uuid, hotkeys, difficulty=None, n_nodes=None, hits=None,
                n_top_true=0, n_spare_true=0, fleet_n=0):
    """picker(), then shift the level split by SN83_ATOP_BIAS hotkeys."""
    if difficulty is None:
        difficulty = difficulty_from_n(n_nodes)
    a = len(hotkeys)
    omega, top, spare = _levels(pool)
    n_top = max(int(n_top_true), len(top), 1)
    n_sp = max(int(n_spare_true), len(spare), 1)
    rows = entity_plan(difficulty, n_top, n_sp, fleet_n or infer_fleet_n(a, difficulty))
    f_top = sum(q for lvl, q, _d, _m in rows if lvl == "top")
    f_sp = sum(q for lvl, q, _d, _m in rows if lvl == "spare")
    b = max(1e-9, f_top + f_sp)
    law_t = law_from_plan(rows, "top", n_top)
    law_s = law_from_plan(rows, "spare", n_sp)
    their_cliques = expected_cliques(law_t, n_top, law_s, n_sp)
    occupied = [(f, p * n) for law, n in ((law_t, n_top), (law_s, n_sp))
                for f, p in law.items() if f > 0]
    field_min = float(min((f for f, cnt in occupied if cnt >= 1.0), default=0.0))
    at, asp = allocate(difficulty, omega, a, b, law_t, law_s,
                       f_top, f_sp, their_cliques,
                       len(top), len(spare), field_min)
    a_top = int(round(sum(at) * (1.0 + ATOP_EPS))) + ATOP_BIAS
    a_top = max(0, min(a, a_top))
    a_sp = a - a_top
    at, asp = _best_widths(difficulty, omega, a, b, a_top, a_sp, law_t, law_s,
                           f_top, f_sp, their_cliques,
                           len(top), len(spare), field_min, False)
    return _emit(uuid, hotkeys, top, spare, at, asp)


NOSTACK = os.environ.get("SN83_NOSTACK", "0") == "1"


def picker_nostack(pool, uuid, hotkeys, difficulty=None, n_nodes=None, hits=None,
                   n_top_true=0, n_spare_true=0, fleet_n=0):
    """picker(), but never put a SECOND hotkey on an omega clique."""
    if difficulty is None:
        difficulty = difficulty_from_n(n_nodes)
    a = len(hotkeys)
    omega, top, spare = _levels(pool)
    n_top = max(int(n_top_true), len(top), 1)
    n_sp = max(int(n_spare_true), len(spare), 1)
    rows = entity_plan(difficulty, n_top, n_sp, fleet_n or infer_fleet_n(a, difficulty))
    f_top = sum(q for lvl, q, _d, _m in rows if lvl == "top")
    f_sp = sum(q for lvl, q, _d, _m in rows if lvl == "spare")
    b = max(1e-9, f_top + f_sp)
    law_t = law_from_plan(rows, "top", n_top)
    law_s = law_from_plan(rows, "spare", n_sp)
    their_cliques = expected_cliques(law_t, n_top, law_s, n_sp)
    occupied = [(f, p * n) for law, n in ((law_t, n_top), (law_s, n_sp))
                for f, p in law.items() if f > 0]
    field_min = float(min((f for f, cnt in occupied if cnt >= 1.0), default=0.0))
    at, asp = allocate(difficulty, omega, a, b, law_t, law_s,
                       f_top, f_sp, their_cliques,
                       len(top), len(spare), field_min)
    a_top = min(int(sum(at)), len(top))
    a_sp = a - a_top
    if len(spare) > 0:
        at, asp = _best_widths(difficulty, omega, a, b, a_top, a_sp, law_t, law_s,
                               f_top, f_sp, their_cliques,
                               len(top), len(spare), field_min, False)
    return _emit(uuid, hotkeys, top, spare, at, asp)


def picker_fair(pool, uuid, hotkeys, difficulty=None, n_nodes=None, hits=None,
                n_top_true=0, n_spare_true=0, fleet_n=0):
    """picker(), but the ANSWERS are handed out by each hotkey's running mean."""
    if difficulty is None:
        difficulty = difficulty_from_n(n_nodes)
    a = len(hotkeys)
    omega, top, spare = _levels(pool)
    n_top = max(int(n_top_true), len(top), 1)
    n_sp = max(int(n_spare_true), len(spare), 1)
    rows = entity_plan(difficulty, n_top, n_sp, fleet_n or infer_fleet_n(a, difficulty))
    f_top = sum(q for lvl, q, _d, _m in rows if lvl == "top")
    f_sp = sum(q for lvl, q, _d, _m in rows if lvl == "spare")
    b = max(1e-9, f_top + f_sp)
    law_t = law_from_plan(rows, "top", n_top)
    law_s = law_from_plan(rows, "spare", n_sp)
    their_cliques = expected_cliques(law_t, n_top, law_s, n_sp)
    occupied = [(f, p * n) for law, n in ((law_t, n_top), (law_s, n_sp))
                for f, p in law.items() if f > 0]
    field_min = float(min((f for f, cnt in occupied if cnt >= 1.0), default=0.0))
    at, asp = allocate(difficulty, omega, a, b, law_t, law_s,
                       f_top, f_sp, their_cliques,
                       len(top), len(spare), field_min)
    slots, values = _slots_and_values(difficulty, omega, a, top, spare, at, asp,
                                      law_t, law_s, f_top, f_sp, field_min)
    return _emit_feedback(uuid, list(hotkeys), slots, values, FAIR)


LAM = os.environ.get("SN83_LAMBDA", "")


FLOOR = os.environ.get("SN83_FLOOR", "0") == "1"


def picker_lambda(pool, uuid, hotkeys, difficulty=None, n_nodes=None, hits=None,
                  n_top_true=0, n_spare_true=0, fleet_n=0):
    """picker() with the value function reweighted: J = our_mean - lambda*their_mean."""
    if difficulty is None:
        difficulty = difficulty_from_n(n_nodes)
    a = len(hotkeys)
    omega, top, spare = _levels(pool)
    n_top = max(int(n_top_true), len(top), 1)
    n_sp = max(int(n_spare_true), len(spare), 1)
    rows = entity_plan(difficulty, n_top, n_sp, fleet_n or infer_fleet_n(a, difficulty))
    f_top = sum(q for lvl, q, _d, _m in rows if lvl == "top")
    f_sp = sum(q for lvl, q, _d, _m in rows if lvl == "spare")
    b = max(1e-9, f_top + f_sp)
    law_t = law_from_plan(rows, "top", n_top)
    law_s = law_from_plan(rows, "spare", n_sp)
    their_cliques = expected_cliques(law_t, n_top, law_s, n_sp)
    occupied = [(f, p * n) for law, n in ((law_t, n_top), (law_s, n_sp))
                for f, p in law.items() if f > 0]
    field_min = float(min((f for f, cnt in occupied if cnt >= 1.0), default=0.0))
    at, asp = allocate(difficulty, omega, a, b, law_t, law_s,
                       f_top, f_sp, their_cliques,
                       len(top), len(spare), field_min,
                       lam=(float(LAM) if LAM else None), floor_mode=FLOOR)
    return _emit(uuid, hotkeys, top, spare, at, asp)


POOL_PICK = os.environ.get("SN83_POOL_PICK", "")


def picker_select(pool, uuid, hotkeys, difficulty=None, n_nodes=None, hits=None,
                  n_top_true=0, n_spare_true=0, fleet_n=0):
    """picker(), but CHOOSING which cliques to occupy out of a deeper pool."""
    if difficulty is None:
        difficulty = difficulty_from_n(n_nodes)
    a = len(hotkeys)
    omega, top, spare = _levels(pool)
    if POOL_PICK and hits:
        idx = {tuple(sorted(c)): i for i, c in enumerate(pool)}
        def h_of(c):
            i = idx.get(tuple(sorted(c)))
            return hits[i] if i is not None and i < len(hits) else 0
        if POOL_PICK == "hits_asc":
            top = sorted(top, key=h_of)
            spare = sorted(spare, key=h_of)
        elif POOL_PICK == "random":
            seed = int(hashlib.sha1(str(uuid).encode()).hexdigest()[:8], 16)
            rng = random.Random(seed)
            top = list(top); spare = list(spare)
            rng.shuffle(top); rng.shuffle(spare)
    n_top = max(int(n_top_true), len(top), 1)
    n_sp = max(int(n_spare_true), len(spare), 1)
    rows = entity_plan(difficulty, n_top, n_sp, fleet_n or infer_fleet_n(a, difficulty))
    f_top = sum(q for lvl, q, _d, _m in rows if lvl == "top")
    f_sp = sum(q for lvl, q, _d, _m in rows if lvl == "spare")
    b = max(1e-9, f_top + f_sp)
    law_t = law_from_plan(rows, "top", n_top)
    law_s = law_from_plan(rows, "spare", n_sp)
    their_cliques = expected_cliques(law_t, n_top, law_s, n_sp)
    occupied = [(f, p * n) for law, n in ((law_t, n_top), (law_s, n_sp))
                for f, p in law.items() if f > 0]
    field_min = float(min((f for f, cnt in occupied if cnt >= 1.0), default=0.0))
    at, asp = allocate(difficulty, omega, a, b, law_t, law_s,
                       f_top, f_sp, their_cliques,
                       len(top), len(spare), field_min)
    return _emit(uuid, hotkeys, top, spare, at, asp)


FREE_CAP = float(os.environ.get("SN83_FREE_CAP", "0"))


def picker_free(pool, uuid, hotkeys, difficulty=None, n_nodes=None, hits=None,
                n_top_true=0, n_spare_true=0, fleet_n=0):
    """Take only as many omega cliques as we expect to hold ALONE; spill the rest."""
    if difficulty is None:
        difficulty = difficulty_from_n(n_nodes)
    a = len(hotkeys)
    omega, top, spare = _levels(pool)
    n_top = max(int(n_top_true), len(top), 1)
    n_sp = max(int(n_spare_true), len(spare), 1)
    rows = entity_plan(difficulty, n_top, n_sp, fleet_n or infer_fleet_n(a, difficulty))
    f_top = sum(q for lvl, q, _d, _m in rows if lvl == "top")
    f_sp = sum(q for lvl, q, _d, _m in rows if lvl == "spare")
    b = max(1e-9, f_top + f_sp)
    law_t = law_from_plan(rows, "top", n_top)
    law_s = law_from_plan(rows, "spare", n_sp)
    their_cliques = expected_cliques(law_t, n_top, law_s, n_sp)
    occupied = [(f, p * n) for law, n in ((law_t, n_top), (law_s, n_sp))
                for f, p in law.items() if f > 0]
    field_min = float(min((f for f, cnt in occupied if cnt >= 1.0), default=0.0))
    at, asp = allocate(difficulty, omega, a, b, law_t, law_s,
                       f_top, f_sp, their_cliques,
                       len(top), len(spare), field_min)
    if FREE_CAP > 0 and len(spare) > 0:
        free = int(round(FREE_CAP * min(len(top), n_top) * law_t.get(0, 0.0)))
        a_top = min(int(sum(at)), max(0, free))
        a_sp = a - a_top
        at, asp = _best_widths(difficulty, omega, a, b, a_top, a_sp, law_t, law_s,
                               f_top, f_sp, their_cliques,
                               len(top), len(spare), field_min, False)
    return _emit(uuid, hotkeys, top, spare, at, asp)


def picker_tail(pool, uuid, hotkeys, difficulty=None, n_nodes=None, hits=None,
                n_top_true=0, n_spare_true=0, fleet_n=0):
    """BLIND, but aimed at the field's LOWER TAIL instead of its mean."""
    if difficulty is None:
        difficulty = difficulty_from_n(n_nodes)
    a = len(hotkeys)
    omega, top, spare = _levels(pool)
    n_top = max(int(n_top_true), len(top), 1)
    n_sp = max(int(n_spare_true), len(spare), 1)
    rows = entity_plan(difficulty, n_top, n_sp, fleet_n or infer_fleet_n(a, difficulty))
    f_top = sum(q for lvl, q, _d, _m in rows if lvl == "top")
    f_sp = sum(q for lvl, q, _d, _m in rows if lvl == "spare")
    b = max(1e-9, f_top + f_sp)
    law_t = law_from_plan(rows, "top", n_top)
    law_s = law_from_plan(rows, "spare", n_sp)
    their_cliques = expected_cliques(law_t, n_top, law_s, n_sp)
    occupied = [(f, p * n) for law, n in ((law_t, n_top), (law_s, n_sp))
                for f, p in law.items() if f > 0]
    field_min = float(min((f for f, cnt in occupied if cnt >= 1.0), default=0.0))
    at, asp = allocate(difficulty, omega, a, b, law_t, law_s,
                       f_top, f_sp, their_cliques,
                       len(top), len(spare), field_min, tail=True)
    return _emit(uuid, hotkeys, top, spare, at, asp)


ABSOLUTE = os.environ.get("SN83_ABSOLUTE", "0") == "1"


def picker_absolute(pool, uuid, hotkeys, difficulty=None, n_nodes=None, hits=None,
                    n_top_true=0, n_spare_true=0, fleet_n=0):
    """Maximise OUR OWN mean reward instead of the difference."""
    if difficulty is None:
        difficulty = difficulty_from_n(n_nodes)
    a = len(hotkeys)
    omega, top, spare = _levels(pool)
    n_top = max(int(n_top_true), len(top), 1)
    n_sp = max(int(n_spare_true), len(spare), 1)
    rows = entity_plan(difficulty, n_top, n_sp, fleet_n or infer_fleet_n(a, difficulty))
    f_top = sum(q for lvl, q, _d, _m in rows if lvl == "top")
    f_sp = sum(q for lvl, q, _d, _m in rows if lvl == "spare")
    b = max(1e-9, f_top + f_sp)
    law_t = law_from_plan(rows, "top", n_top)
    law_s = law_from_plan(rows, "spare", n_sp)
    their_cliques = expected_cliques(law_t, n_top, law_s, n_sp)
    occupied = [(f, p * n) for law, n in ((law_t, n_top), (law_s, n_sp))
                for f, p in law.items() if f > 0]
    field_min = float(min((f for f, cnt in occupied if cnt >= 1.0), default=0.0))
    at, asp = allocate(difficulty, omega, a, b, law_t, law_s, f_top, f_sp,
                       their_cliques, len(top), len(spare),
                       field_min, absolute=True)
    return _emit(uuid, hotkeys, top, spare, at, asp)
