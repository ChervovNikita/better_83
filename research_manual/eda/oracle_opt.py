"""Exact best response for a player who can see every other answer."""

import argparse
import collections
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(os.path.dirname(HERE)), os.path.join(
        os.path.dirname(os.path.dirname(HERE)), "research"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fleet_sim import score_round


def _profile_optimality(sizes_all, n_total, difficulty):
    """optimality per distinct size, given every size in the finished round."""
    arr = np.asarray(sizes_all, dtype=float)
    M = arr.max()
    out = {}
    for s in set(int(x) for x in arr):
        pr = float((arr > s).sum()) / n_total
        rel = s / M
        out[s] = np.exp(-pr / max(rel, 1e-12))
    top = max(out.values())
    return {s: v / top * (1.0 + difficulty) for s, v in out.items()}


def _dp(items, budget, floor):
    """Best total over independent cliques."""
    NEG = -1e18
    n = len(items)
    dp = np.full((n + 1, budget + 1), NEG)
    dp[0, 0] = 0.0
    back = [[0] * (budget + 1) for _ in range(n + 1)]
    for i, (lo, gain) in enumerate(items):
        top = min(budget, len(gain) - 1)
        for b in range(budget + 1):
            best, bm = NEG, 0
            for m in range(lo, min(top, b) + 1):
                prev = dp[i, b - m]
                if prev <= NEG / 2:
                    continue
                v = prev + gain[m]
                if v > best:
                    best, bm = v, m
            dp[i + 1, b] = best
            back[i + 1][b] = bm
    b = int(np.argmax(dp[n]))
    if dp[n, b] <= NEG / 2:
        return None
    take = [0] * n
    for i in range(n, 0, -1):
        take[i - 1] = back[i][b]
        b -= back[i][b]
    return float(dp[n, int(np.argmax(dp[n]))]), take


def best_response(field, pool, o, difficulty, target=None, max_xmin=4,
                  sizes_used=2):
    """Our best allocation of `o` hotkeys against a board we can fully see."""
    assert o > 0 and field
    cnt = collections.Counter(c for c, _w in field)
    tgt = collections.Counter(c for c, w in field
                              if target is None or w == target)
    n_tgt = sum(tgt.values())
    assert n_tgt > 0, "target has no answers"
    n_field = len(field)
    n_total = n_field + o

    by_size = collections.defaultdict(list)
    for c in cnt:
        by_size[len(c)].append(c)
    fresh = collections.defaultdict(list)
    for c in pool:
        c = tuple(sorted(c))
        if c not in cnt:
            fresh[len(c)].append(c)

    omega = max(list(by_size) + list(fresh))
    order = sorted(set(list(by_size) + list(fresh)), reverse=True)[:sizes_used]

    best = None
    for split in _splits(o, len(order)):
        for t in range(1, max_xmin + 1):
            got = _solve_fixed(cnt, tgt, by_size, fresh, order, split, t, o,
                               n_tgt)
            if got is None:
                continue
            place = got
            val = _exact(field, place, difficulty, target)
            if best is None or val > best[0] + 1e-12:
                best = (val, place)
    assert best is not None, "no feasible placement"
    return best


def _splits(total, parts):
    if parts == 1:
        yield (total,)
        return
    for i in range(total + 1):
        for rest in _splits(total - i, parts - 1):
            yield (i,) + rest


def _solve_fixed(cnt, tgt, by_size, fresh, order, split, t, o, n_tgt):
    """Placement maximising the diversity gap with x_min pinned at t."""
    place = []
    for s, k in zip(order, split):
        if k == 0:
            continue
        joins = by_size.get(s, [])
        items = []
        for c in joins:
            f = cnt[c]
            lo = max(0, t - f)
            gain = []
            for m in range(0, k + 1):
                if m < lo:
                    gain.append(-1e18)
                else:
                    gain.append(t * (m / (f + m) / o - tgt[c] / (f + m) / n_tgt))
            items.append((0, gain))
        n_fresh = len(fresh.get(s, []))
        if n_fresh:
            cap = min(n_fresh, k // t)
            gain = [-1e18] * (k + 1)
            for j in range(cap + 1):
                if j * t <= k:
                    gain[j * t] = t * (j / float(o))
            items.append((0, gain))
        got = _dp(items, k, t)
        if got is None:
            return None
        _v, take = got
        if sum(take) != k:
            return None
        for c, m in zip(joins, take[:len(joins)]):
            place += [c] * m
        if n_fresh:
            j = take[-1] // t
            for x in range(j):
                place += [fresh[s][x]] * t
    if len(place) != o:
        return None
    return place


def _exact(field, place, difficulty, target):
    """Score the finished round with the validator's own formula."""
    keys = [c for c, _w in field] + list(place)
    sizes = [len(c) for c in keys]
    who = [0] * len(field) + [1] * len(place)
    owners = [w for _c, w in field] + [None] * len(place)
    r = score_round(sizes, [1] * len(keys), keys, difficulty)
    r = np.asarray(r)
    ours = r[len(field):].mean()
    idx = [i for i in range(len(field))
           if target is None or owners[i] == target]
    theirs = r[idx].mean()
    return float(ours - theirs)
