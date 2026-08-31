#!/usr/bin/env python3
"""Pickers derived from the reward algebra in autoresearch-runs/sn83-picker/MATH.md.

The reward a hotkey earns is exactly

    R = (1+D)·exp(-pr/rel) + c_min/cnt

with `cnt` the number of miners on the same clique and `c_min` the smallest such
count over valid answers.  Two consequences drive everything here:

  * a hotkey ALONE on a clique earns the maximum diversity term AND forces
    c_min = 1, which divides every field answer's diversity by whatever c_min
    would otherwise have been (measured 2-9 in 17.7% of rounds);
  * a second hotkey on an already-free clique adds exactly zero diversity,
    because a/(a+0) = 1 for every a.

Field occupancy is forecast from the rules the three real operators were measured
to follow, not assumed uniform.
"""
import collections
import hashlib
import math
import os

# Fleet sizes measured from the metagraph over the logged rounds.
TOP4_HOTKEYS = 178
E2_HOTKEYS = 33
E3_HOTKEYS = 29
OTHER_HOTKEYS = 6

# TOP4's level rule, fitted exactly on 739 rounds.
TOP4_S_STAR = {0.7: 12, 0.8: 8, 0.9: 5, 1.0: 3}
# Entities 2 and 3 run the same fill-and-spill; they differ only in solver reach.
E2_REACH = 0.76
E3_REACH = 0.97

REF_R = 1.5
POISSON_TERMS = 24


def selection_p(difficulty):
    return 1.0 - math.exp(-max(0.0, math.sqrt(1.0 + REF_R) - difficulty - 0.5))


def difficulty_from_n(number_of_nodes):
    n = int(number_of_nodes)
    if 290 <= n <= 300:
        return 0.7
    if 490 <= n <= 500:
        return 0.8
    if 690 <= n <= 700:
        return 0.9
    if 890 <= n <= 900:
        return 1.0
    return 0.8


def _spread(total, parts):
    base, extra = divmod(total, parts)
    return [base + 1] * extra + [base] * (parts - extra)


def forecast(difficulty, n_top, n_spare):
    """Expected field answers and distinct cliques used, per level.

    Applies each operator's measured rule to the round's supply.  Returns
    (F_top, F_spare, d_top, d_spare, n_field) where F_* are answer counts and
    d_* are distinct cliques the field is expected to occupy.
    """
    p = selection_p(difficulty)
    q4 = TOP4_HOTKEYS * p
    q2 = E2_HOTKEYS * p
    q3 = E3_HOTKEYS * p
    q5 = OTHER_HOTKEYS * p

    f_top = 0.0
    f_sp = 0.0
    d_top = 0.0
    d_sp = 0.0

    # TOP4: all-or-nothing on the S* threshold, then spread(q, min(q, supply)).
    star = TOP4_S_STAR.get(round(difficulty, 1), 3)
    if n_top >= star:
        f_top += q4
        d_top += min(q4, max(1, n_top))
    else:
        f_sp += q4
        d_sp += min(q4, max(1, n_spare))

    # Entities 2 and 3: fill omega with distinct cliques, spill the rest.
    for q_i, reach in ((q2, E2_REACH), (q3, E3_REACH)):
        take = min(q_i, max(1, n_top) * reach)
        f_top += take
        d_top += take
        f_sp += q_i - take
        d_sp += q_i - take

    # Everyone else: always omega.
    f_top += q5
    d_top += min(q5, max(1, n_top))

    # Distinct coverage cannot exceed supply, and entities overlap: if each
    # operator picks d_i of S uniformly, the expected covered set is
    # S(1 - prod(1 - d_i/S)), not the sum.
    d_top = min(d_top, float(n_top))
    d_sp = min(d_sp, float(n_spare))
    return f_top, f_sp, d_top, d_sp, q4 + q2 + q3 + q5


def _poisson_share(lam, a):
    """E[a/(a+f)] for f ~ Poisson(lam).

    The mean occupancy is not enough: a/(a+E[f]) overstates the value of a
    crowded clique badly, because 1/(1+f) is convex.  Summing the distribution
    is cheap and removes the bias.
    """
    if lam <= 0:
        return 1.0
    total = 0.0
    weight = math.exp(-lam)
    term = weight
    for f in range(POISSON_TERMS):
        if f:
            term *= lam / f
        total += term * (a / float(a + f))
    return total


def _order(pool, omega, prefer_large_basin, hits):
    """Cliques of each level, best-first.

    Large-basin cliques were measured to carry LESS field occupancy (Q3 0.837
    against Q0 1.034 over 27,729 cliques), so basin descending is the tiebreak.
    """
    top = [c for c in pool if len(c) == omega]
    spare = [c for c in pool if len(c) == omega - 1]
    if prefer_large_basin and hits:
        key = {tuple(sorted(c)): h for c, h in zip(pool, hits)}
        top.sort(key=lambda c: -key.get(tuple(sorted(c)), 0))
        spare.sort(key=lambda c: -key.get(tuple(sorted(c)), 0))
    return top, spare


def _allocate(pool, q, difficulty, n_top_true, n_spare_true,
              prefer_large_basin=True, hits=None, allow_repeats=True):
    """Greedy on exact marginals -- optimal for a separable concave objective.

    Marginal of the (a+1)-th hotkey on a clique whose expected field load is lam:

        (1+D)·opt(level) + [E[(a+1)/(a+1+f)] - E[a/(a+f)]]

    phi (the share of ALL responses sitting at omega) sets opt(omega-1), and phi
    itself depends on how many of our own hotkeys go to omega, so it is iterated
    to a fixed point.
    """
    omega = max(len(c) for c in pool)
    top, spare = _order(pool, omega, prefer_large_basin, hits)
    n_top = max(int(n_top_true) or len(top), 1)
    n_sp = max(int(n_spare_true) or len(spare), 1)

    f_top, f_sp, _d_top, _d_sp, n_field = forecast(difficulty, n_top, n_sp)
    lam_top = f_top / float(n_top)
    lam_sp = f_sp / float(n_sp)
    rho = omega / float(omega - 1) if omega > 1 else 1.0

    a_top = min(q, len(top))
    for _ in range(6):
        n_all = n_field + q
        phi = (f_top + a_top) / max(1.0, n_all)
        opt_sp = math.exp(-phi * rho)
        gain_top = (1.0 + difficulty)
        gain_sp = (1.0 + difficulty) * opt_sp

        counts_top = [0] * len(top)
        counts_sp = [0] * len(spare)
        placed = 0
        while placed < q:
            best = None
            for i in range(len(top)):
                a = counts_top[i]
                if a and not allow_repeats:
                    continue
                d = _poisson_share(lam_top, a + 1) - (_poisson_share(lam_top, a) if a else 0.0)
                v = gain_top + d
                if best is None or v > best[0]:
                    best = (v, "t", i)
            for i in range(len(spare)):
                b = counts_sp[i]
                if b and not allow_repeats:
                    continue
                d = _poisson_share(lam_sp, b + 1) - (_poisson_share(lam_sp, b) if b else 0.0)
                v = gain_sp + d
                if best is None or v > best[0]:
                    best = (v, "s", i)
            if best is None:
                break
            if best[1] == "t":
                counts_top[best[2]] += 1
            else:
                counts_sp[best[2]] += 1
            placed += 1
        new_a = sum(counts_top)
        if new_a == a_top:
            break
        a_top = new_a

    out = []
    for i, c in enumerate(top):
        out.extend([list(c)] * counts_top[i])
    for i, c in enumerate(spare):
        out.extend([list(c)] * counts_sp[i])
    while len(out) < q:
        out.append(list(top[0] if top else spare[0]))
    return out[:q]


def _rotate(uuid, slots, hotkeys):
    """Deal slots to hotkeys, rotating by round so no hotkey is always sacrificed."""
    offset = int(hashlib.sha1(str(uuid).encode()).hexdigest()[:8], 16)
    return [list(slots[(i + offset) % len(slots)]) for i in range(len(hotkeys))]


def picker(pool, uuid, hotkeys, difficulty=None, n_nodes=None, hits=None,
           n_top_true=0, n_spare_true=0):
    """H4 -- the theory-complete candidate: exact greedy on predicted occupancy."""
    assert pool and hotkeys
    if difficulty is None:
        difficulty = difficulty_from_n(n_nodes) if n_nodes else 0.8
    slots = _allocate(pool, len(hotkeys), difficulty, n_top_true, n_spare_true,
                      hits=hits)
    return _rotate(uuid, slots, hotkeys)


def picker_nodup(pool, uuid, hotkeys, difficulty=None, n_nodes=None, hits=None,
                 n_top_true=0, n_spare_true=0):
    """H6 -- never place two hotkeys on the same clique, as entities 2 and 3 do."""
    assert pool and hotkeys
    if difficulty is None:
        difficulty = difficulty_from_n(n_nodes) if n_nodes else 0.8
    slots = _allocate(pool, len(hotkeys), difficulty, n_top_true, n_spare_true,
                      hits=hits, allow_repeats=False)
    return _rotate(uuid, slots, hotkeys)


def picker_nobasin(pool, uuid, hotkeys, difficulty=None, n_nodes=None, hits=None,
                   n_top_true=0, n_spare_true=0):
    """H3 ablation -- same greedy, basin tiebreak removed."""
    assert pool and hotkeys
    if difficulty is None:
        difficulty = difficulty_from_n(n_nodes) if n_nodes else 0.8
    slots = _allocate(pool, len(hotkeys), difficulty, n_top_true, n_spare_true,
                      hits=hits, prefer_large_basin=False)
    return _rotate(uuid, slots, hotkeys)


def picker_solitude(pool, uuid, hotkeys, difficulty=None, n_nodes=None, hits=None,
                    n_top_true=0, n_spare_true=0):
    """H1 -- solitude first: one hotkey per distinct clique, omega then spares.

    No occupancy model at all.  Isolates how much of the gain is just "be alone"
    versus the forecast that H4 layers on top.
    """
    assert pool and hotkeys
    omega = max(len(c) for c in pool)
    top, spare = _order(pool, omega, True, hits)
    order = top + spare
    slots = [list(c) for c in order[:len(hotkeys)]]
    while len(slots) < len(hotkeys):
        slots.append(list(order[len(slots) % len(order)]))
    return _rotate(uuid, slots, hotkeys)


def _rank_slots(slots, pool, difficulty, n_top_true, n_spare_true, hits):
    """Slots best-first by predicted reward, so the deal can be made unfair."""
    omega = max(len(c) for c in pool)
    n_top = max(int(n_top_true) or 1, 1)
    n_sp = max(int(n_spare_true) or 1, 1)
    f_top, f_sp, _dt, _ds, _nf = forecast(difficulty, n_top, n_sp)
    lam = {omega: f_top / float(n_top), omega - 1: f_sp / float(n_sp)}
    mult = collections.Counter(tuple(sorted(c)) for c in slots)
    key = {}
    if hits:
        key = {tuple(sorted(c)): h for c, h in zip(pool, hits)}

    def value(c):
        k = tuple(sorted(c))
        lv = lam.get(len(c), 1.0)
        # our own stacking is known exactly; the field's load is the forecast
        share = _poisson_share(lv, mult[k]) / max(1, mult[k])
        opt = 1.0 if len(c) == omega else math.exp(-0.75)
        return (1.0 + difficulty) * opt + share + 1e-6 * key.get(k, 0)

    return sorted(slots, key=value, reverse=True)


def _deal_ranked(slots, hotkeys, pool, difficulty, n_top_true, n_spare_true,
                 hits=None):
    """Give the best slots to the lowest-numbered hotkeys, every round.

    The metric is a MEDIAN over our hotkeys, so twenty-one consistently-good
    hotkeys beat forty equally-mediocre ones.  Rotating the deal (which is what
    every picker here did before) equalises the fleet and throws that away.
    Ranking by hotkey name is stable across rounds, which is what makes the
    advantage accumulate into the per-hotkey mean rather than averaging out.
    """
    ranked = _rank_slots(slots, pool, difficulty, n_top_true, n_spare_true, hits)
    order = sorted(range(len(hotkeys)), key=lambda i: str(hotkeys[i]))
    out = [None] * len(hotkeys)
    for slot_i, hk_i in enumerate(order):
        out[hk_i] = list(ranked[slot_i % len(ranked)])
    return out


def picker_median(pool, uuid, hotkeys, difficulty=None, n_nodes=None, hits=None,
                  n_top_true=0, n_spare_true=0):
    """H5 -- greedy allocation, dealt unfairly to lift the median."""
    assert pool and hotkeys
    if difficulty is None:
        difficulty = difficulty_from_n(n_nodes) if n_nodes else 0.8
    slots = _allocate(pool, len(hotkeys), difficulty, n_top_true, n_spare_true,
                      hits=hits)
    return _deal_ranked(slots, hotkeys, pool, difficulty, n_top_true, n_spare_true,
                        hits=hits)


def picker_median_value(pool, uuid, hotkeys, difficulty=None, n_nodes=None,
                        n_top_true=0, n_spare_true=0):
    """H5 ablation -- the SHIPPED allocation, dealt unfairly.

    Isolates the deal from the allocation: if this alone recovers the gap, the
    allocation was never the problem and the metric was.
    """
    import pick_value
    assert pool and hotkeys
    if difficulty is None:
        difficulty = difficulty_from_n(n_nodes) if n_nodes else 0.8
    slots = pick_value.picker(pool, uuid, list(hotkeys), difficulty=difficulty,
                              n_nodes=n_nodes, n_top_true=n_top_true,
                              n_spare_true=n_spare_true)
    return _deal_ranked(slots, hotkeys, pool, difficulty, n_top_true, n_spare_true)


def _order_hard(pool, omega, hits):
    """Cliques ordered HARDEST-to-find first.

    Our harvester now covers 100% of the field's omega answers and routinely finds
    more than they do, so the cliques with the smallest basins -- and especially
    the ones the one-swap closure derived that the search never landed on at all
    (hits == 0) -- are the ones a rival solver is least able to reach. A clique no
    opponent can find is free by construction, which is worth a full +1 of
    diversity. This is the exact inverse of the offline P(free)-by-basin
    statistic, which measured the historical field rather than a rival's reach.
    """
    key = {}
    if hits:
        key = {tuple(sorted(c)): h for c, h in zip(pool, hits)}
    top = [c for c in pool if len(c) == omega]
    spare = [c for c in pool if len(c) == omega - 1]
    rank = lambda c: (key.get(tuple(sorted(c)), 0) != 0, key.get(tuple(sorted(c)), 0))
    top.sort(key=rank)
    spare.sort(key=rank)
    return top, spare


def picker_hard(pool, uuid, hotkeys, difficulty=None, n_nodes=None, hits=None,
                n_top_true=0, n_spare_true=0):
    """Take the cliques rivals are least able to find, one hotkey each."""
    assert pool and hotkeys
    omega = max(len(c) for c in pool)
    top, spare = _order_hard(pool, omega, hits)
    order = top + spare
    slots = [list(c) for c in order[:len(hotkeys)]]
    while len(slots) < len(hotkeys):
        slots.append(list(order[len(slots) % len(order)]))
    return _rotate(uuid, slots, hotkeys)


def picker_hard_median(pool, uuid, hotkeys, difficulty=None, n_nodes=None,
                       hits=None, n_top_true=0, n_spare_true=0):
    """Hardest-to-find cliques, dealt unfairly to lift the median."""
    assert pool and hotkeys
    if difficulty is None:
        difficulty = difficulty_from_n(n_nodes) if n_nodes else 0.8
    omega = max(len(c) for c in pool)
    top, spare = _order_hard(pool, omega, hits)
    order = top + spare
    slots = [list(c) for c in order[:len(hotkeys)]]
    while len(slots) < len(hotkeys):
        slots.append(list(order[len(slots) % len(order)]))
    return _deal_ranked(slots, hotkeys, pool, difficulty, n_top_true,
                        n_spare_true, hits=hits)


def _twopoint_share(n_cliques, occupants, d_used, a):
    """E[a/(a+f)] when the field's spread is EVEN, not Poisson.

    TOP4 places spread(q, d) -- every clique they touch gets floor(q/d) or
    ceil(q/d), never a Poisson draw. So occupancy is a two-point mixture:

        with prob d/S   f = q/d   (they took this clique)
        with prob 1-d/S f = 0     (they did not)

    Modelling that as Poisson(q/S) is wrong in the direction that matters: it
    smears the mass and hides the fact that (S - d) cliques are EXACTLY free.
    Solitude is worth a full +1, so mistaking "certainly free" for "probably
    lightly loaded" is the difference between taking it and not.
    """
    if n_cliques <= 0:
        return 1.0
    d = max(0.0, min(float(d_used), float(n_cliques)))
    p_hit = d / float(n_cliques)
    load = (occupants / d) if d > 0 else 0.0
    return (1.0 - p_hit) * 1.0 + p_hit * (a / float(a + load))


def _allocate_even(pool, q, difficulty, n_top_true, n_spare_true, hits=None):
    """Greedy on marginals, with the field's EVEN spread modelled explicitly."""
    omega = max(len(c) for c in pool)
    top = [c for c in pool if len(c) == omega]
    spare = [c for c in pool if len(c) == omega - 1]
    n_top = max(int(n_top_true) or len(top), 1)
    n_sp = max(int(n_spare_true) or len(spare), 1)
    f_top, f_sp, d_top, d_sp, n_field = forecast(difficulty, n_top, n_sp)
    rho = omega / float(omega - 1) if omega > 1 else 1.0

    a_top = min(q, len(top))
    counts_top = [0] * len(top)
    counts_sp = [0] * len(spare)
    for _ in range(6):
        phi = (f_top + a_top) / max(1.0, n_field + q)
        g_top = 1.0 + difficulty
        g_sp = (1.0 + difficulty) * math.exp(-phi * rho)
        counts_top = [0] * len(top)
        counts_sp = [0] * len(spare)
        placed = 0
        while placed < q:
            best = None
            for i in range(len(top)):
                a = counts_top[i]
                d = (_twopoint_share(n_top, f_top, d_top, a + 1)
                     - (_twopoint_share(n_top, f_top, d_top, a) if a else 0.0))
                v = g_top + d
                if best is None or v > best[0]:
                    best = (v, "t", i)
            for i in range(len(spare)):
                b = counts_sp[i]
                d = (_twopoint_share(n_sp, f_sp, d_sp, b + 1)
                     - (_twopoint_share(n_sp, f_sp, d_sp, b) if b else 0.0))
                v = g_sp + d
                if best is None or v > best[0]:
                    best = (v, "s", i)
            if best is None:
                break
            if best[1] == "t":
                counts_top[best[2]] += 1
            else:
                counts_sp[best[2]] += 1
            placed += 1
        new_a = sum(counts_top)
        if new_a == a_top:
            break
        a_top = new_a

    out = []
    for i, c in enumerate(top):
        out.extend([list(c)] * counts_top[i])
    for i, c in enumerate(spare):
        out.extend([list(c)] * counts_sp[i])
    while len(out) < q:
        out.append(list(top[0] if top else spare[0]))
    return out[:q]


def picker_even(pool, uuid, hotkeys, difficulty=None, n_nodes=None, hits=None,
                n_top_true=0, n_spare_true=0):
    """Even-spread occupancy model, rotated deal."""
    assert pool and hotkeys
    if difficulty is None:
        difficulty = difficulty_from_n(n_nodes) if n_nodes else 0.8
    slots = _allocate_even(pool, len(hotkeys), difficulty, n_top_true, n_spare_true)
    return _rotate(uuid, slots, hotkeys)


def picker_even_median(pool, uuid, hotkeys, difficulty=None, n_nodes=None,
                       hits=None, n_top_true=0, n_spare_true=0):
    """Even-spread occupancy model, dealt unfairly to lift the median."""
    assert pool and hotkeys
    if difficulty is None:
        difficulty = difficulty_from_n(n_nodes) if n_nodes else 0.8
    slots = _allocate_even(pool, len(hotkeys), difficulty, n_top_true, n_spare_true)
    return _deal_ranked(slots, hotkeys, pool, difficulty, n_top_true, n_spare_true)


def picker_level(pool, uuid, hotkeys, difficulty=None, n_nodes=None, hits=None,
                 n_top_true=0, n_spare_true=0):
    """Compare the two whole-fleet plans exactly, instead of greedily.

    MATH section 3 gives the level condition in closed form, but phi depends on
    where WE put our own hotkeys, so the marginal greedy evaluates it at a phi
    that its own decision then invalidates. With only two levels the honest thing
    is to price both complete plans under their OWN phi and take the larger:

        all-omega   : phi = (F_top + q)/n      -> q*[(1+D) + E 1/(a+f)]
        all-spare   : phi =  F_top/n           -> q*[(1+D)e^(-phi*rho) + E 1/(b+g)]

    The second plan lowers phi by removing our answers from omega, which raises
    the value of the spare level for us -- an effect no marginal step can see.
    """
    assert pool and hotkeys
    if difficulty is None:
        difficulty = difficulty_from_n(n_nodes) if n_nodes else 0.8
    q = len(hotkeys)
    omega = max(len(c) for c in pool)
    top = [c for c in pool if len(c) == omega]
    spare = [c for c in pool if len(c) == omega - 1]
    if not spare:
        return _rotate(uuid, [list(c) for c in (top * q)[:q]], hotkeys)

    n_top = max(int(n_top_true) or len(top), 1)
    n_sp = max(int(n_spare_true) or len(spare), 1)
    f_top, f_sp, _dt, _ds, n_field = forecast(difficulty, n_top, n_sp)
    rho = omega / float(omega - 1) if omega > 1 else 1.0
    n_all = max(1.0, n_field + q)

    def plan_value(level_top):
        cliques = top if level_top else spare
        supply = n_top if level_top else n_sp
        load = f_top if level_top else f_sp
        k = min(q, len(cliques))
        per = _spread(q, k)                      # our own stacking, exact
        phi = (f_top + (q if level_top else 0)) / n_all
        opt = 1.0 if level_top else math.exp(-phi * rho)
        lam = load / float(supply)
        div = sum(_poisson_share(lam, a) for a in per)
        return (1.0 + difficulty) * q * opt + div, cliques, per

    v_top, c_top, p_top = plan_value(True)
    v_sp, c_sp, p_sp = plan_value(False)
    cliques, per = (c_top, p_top) if v_top >= v_sp else (c_sp, p_sp)
    slots = []
    for i, m in enumerate(per):
        slots.extend([list(cliques[i])] * m)
    return _deal_ranked(slots[:q], hotkeys, pool, difficulty, n_top_true,
                        n_spare_true)
