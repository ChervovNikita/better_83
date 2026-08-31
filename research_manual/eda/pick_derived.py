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
import json
import hashlib
import math
import os

# Operator identity is measured; fleet SIZES are not constants. A hotkey we
# register displaces someone, and the validator's churn takes the lowest-incentive
# miners first -- which are entities 2 and 3, not the dominant operator. So the
# field a picker faces depends on our own fleet size, and hard-coding 178/33/29
# models opponents that may no longer be registered.
#
#   N=40  -> TOP4 178, E2 20, E3  9, other 2
#   N=80  -> TOP4 168, E2  0, E3  0, other 1   <- E2 and E3 are GONE
#   N=120 -> TOP4 128, E2  0, E3  0, other 1
#
# Everything below therefore derives the rival profile from the metagraph at call
# time. The metagraph is public and our own hotkeys are known, so this is
# information a deployed miner genuinely has.

OPERATORS = {
    "top4": ("5HMevt8h", "5Eyh8ePM", "5EfHz7fE", "5Hg2Ps2L"),
    "e2": ("5D7BMeGt",),
    "e3": ("5GghBgin",),
}
# TOP4's level rule, fitted exactly on 739 rounds.
TOP4_S_STAR = {0.7: 12, 0.8: 8, 0.9: 5, 1.0: 3}
# Entities 2 and 3 run the same fill-and-spill; they differ only in solver reach.
REACH = {"e2": 0.76, "e3": 0.97}

METAGRAPH = os.environ.get(
    "SN83_METAGRAPH",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "metagraph.json"))
FLEET_N = int(os.environ.get("SN83_FLEET_N", "0"))
IMMUNITY_BLOCKS = 6000

REF_R = 1.5
POISSON_TERMS = 24
_profile_cache = {}


def _operator_of(coldkey):
    for name, prefixes in OPERATORS.items():
        if coldkey.startswith(prefixes):
            return name
    return "other"


def fleet_profile(fleet_n):
    """Rival hotkey counts once OUR fleet_n hotkeys have displaced the weakest.

    Mirrors the validator's churn order (lowest incentive, then uid) so the
    profile matches who is actually registered alongside us. Falls back to the
    full metagraph when it cannot be read, which is the conservative direction:
    it over-states the opposition rather than inventing an empty field.
    """
    if fleet_n in _profile_cache:
        return _profile_cache[fleet_n]
    counts = collections.Counter()
    try:
        with open(METAGRAPH) as handle:
            meta = json.load(handle)
        block = meta["block"]
        cand = [m for m in meta["miners"]
                if block - m["block_at_registration"] >= IMMUNITY_BLOCKS]
        taken = {m["hotkey"] for m in cand[:max(0, int(fleet_n))]}
        for m in meta["miners"]:
            if m["hotkey"] not in taken:
                counts[_operator_of(m["coldkey"])] += 1
    except (OSError, ValueError, KeyError):
        counts.update({"top4": 178, "e2": 33, "e3": 29, "other": 9})
    _profile_cache[fleet_n] = counts
    return counts


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


def forecast(difficulty, n_top, n_spare, fleet_n=0):
    """Expected field answers and distinct cliques per level, for THIS field.

    Applies each operator's measured rule to the round's supply, with fleet sizes
    taken from the live profile rather than from constants.
    """
    p = selection_p(difficulty)
    prof = fleet_profile(fleet_n)
    q = {k: prof.get(k, 0) * p for k in ("top4", "e2", "e3", "other")}

    f_top = f_sp = d_top = d_sp = 0.0

    # TOP4: all-or-nothing on the S* threshold, then spread(q, min(q, supply)).
    star = TOP4_S_STAR.get(round(difficulty, 1), 3)
    if n_top >= star:
        f_top += q["top4"]
        d_top += min(q["top4"], max(1, n_top))
    else:
        f_sp += q["top4"]
        d_sp += min(q["top4"], max(1, n_spare))

    # Entities 2 and 3: fill omega with distinct cliques, spill the rest.
    for name in ("e2", "e3"):
        take = min(q[name], max(1, n_top) * REACH[name])
        f_top += take
        d_top += take
        f_sp += q[name] - take
        d_sp += q[name] - take

    # Everyone else: always omega.
    f_top += q["other"]
    d_top += min(q["other"], max(1, n_top))

    d_top = min(d_top, float(n_top))
    d_sp = min(d_sp, float(n_spare))
    return f_top, f_sp, d_top, d_sp, sum(q.values())


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


def infer_fleet_n(q, difficulty):
    """Our own fleet size, from the env if set, else from the queried count.

    q ~ Binomial(N, p(D)), so q/p(D) is an unbiased estimate of N. Noisy for one
    round, but the profile only has to be right to the nearest churn boundary,
    and being wrong here is what made the N=40-tuned picker fail at N=80.
    """
    if FLEET_N > 0:
        return FLEET_N
    p = selection_p(difficulty)
    return int(round(q / p)) if p > 1e-9 else q


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
              prefer_large_basin=True, hits=None, allow_repeats=True,
              fleet_n=0):
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

    f_top, f_sp, _d_top, _d_sp, n_field = forecast(difficulty, n_top, n_sp,
                                                   fleet_n or infer_fleet_n(q, difficulty))
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
           n_top_true=0, n_spare_true=0, fleet_n=0):
    """H4 -- the theory-complete candidate: exact greedy on predicted occupancy."""
    assert pool and hotkeys
    if difficulty is None:
        difficulty = difficulty_from_n(n_nodes) if n_nodes else 0.8
    slots = _allocate(pool, len(hotkeys), difficulty, n_top_true, n_spare_true,
                      hits=hits, fleet_n=fleet_n)
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
    f_top, f_sp, _dt, _ds, _nf = forecast(difficulty, n_top, n_sp,
                                          infer_fleet_n(len(slots), difficulty))
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


def _allocate_even(pool, q, difficulty, n_top_true, n_spare_true, hits=None,
                   fleet_n=0):
    """Greedy on marginals, with the field's EVEN spread modelled explicitly."""
    omega = max(len(c) for c in pool)
    top = [c for c in pool if len(c) == omega]
    spare = [c for c in pool if len(c) == omega - 1]
    n_top = max(int(n_top_true) or len(top), 1)
    n_sp = max(int(n_spare_true) or len(spare), 1)
    f_top, f_sp, d_top, d_sp, n_field = forecast(difficulty, n_top, n_sp,
                                                 fleet_n or infer_fleet_n(q, difficulty))
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
    f_top, f_sp, _dt, _ds, n_field = forecast(difficulty, n_top, n_sp,
                                              infer_fleet_n(q, difficulty))
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


def picker_edge(pool, uuid, hotkeys, difficulty=None, n_nodes=None, hits=None,
                n_top_true=0, n_spare_true=0, fleet_n=0):
    """Maximise our_reward MINUS theirs, not our reward alone.

    The greedy pickers are a best response for our OWN score, which is the wrong
    objective when the metric is a difference. They miss that where we stand
    changes what the FIELD earns:

        their opt = exp(-phi*rho),   phi = (F_omega + A) / n_total

    so every hotkey we add at omega raises phi and cuts the optimality of every
    answer sitting at omega-1. The derivative is

        d(their opt)/dA = -(rho / n_total) * exp(-phi*rho)

    and with B of their answers down at omega-1 the damage per hotkey we move is
    B*(1+D)*rho*exp(-phi*rho)/n_total. That term is worth the most in exactly the
    rounds TOP4's own rule makes predictable: when S < S*(D) they put EVERYTHING
    on omega-1, so B is their whole fleet and omega is ours alone. Following them
    down there -- which the plain greedy does, 535 hotkeys against 70 in the
    measured N=120 rounds -- forfeits the attack entirely.

    Placing at omega-1 damages nobody: it lowers phi, which HELPS them.
    """
    assert pool and hotkeys
    if difficulty is None:
        difficulty = difficulty_from_n(n_nodes) if n_nodes else 0.8
    q = len(hotkeys)
    omega = max(len(c) for c in pool)
    top = [c for c in pool if len(c) == omega]
    spare = [c for c in pool if len(c) == omega - 1]
    if not top:
        return _rotate(uuid, [list(c) for c in (spare * q)[:q]], hotkeys)
    if not spare:
        return _rotate(uuid, [list(c) for c in (top * q)[:q]], hotkeys)

    n_top = max(int(n_top_true) or len(top), 1)
    n_sp = max(int(n_spare_true) or len(spare), 1)
    fn = fleet_n or infer_fleet_n(q, difficulty)
    f_top, f_sp, _dt, _ds, n_field = forecast(difficulty, n_top, n_sp, fn)
    rho = omega / float(omega - 1) if omega > 1 else 1.0
    n_all = max(1.0, n_field + q)
    lam_top = f_top / float(n_top)
    lam_sp = f_sp / float(n_sp)

    best = None
    for a_top in range(0, q + 1):
        phi = (f_top + a_top) / n_all
        opt_sp = math.exp(-phi * rho)
        k_t = min(a_top, len(top))
        k_s = min(q - a_top, len(spare))
        ours = 0.0
        if k_t:
            for m in _spread(a_top, k_t):
                ours += m * (1.0 + difficulty) + _poisson_share(lam_top, m)
        if k_s:
            for m in _spread(q - a_top, k_s):
                ours += m * ((1.0 + difficulty) * opt_sp) + _poisson_share(lam_sp, m)
        # what the field earns, per answer, under this phi
        theirs = (f_top * (1.0 + difficulty)
                  + f_sp * (1.0 + difficulty) * opt_sp) / max(1.0, f_top + f_sp)
        score = ours / q - theirs
        if best is None or score > best[0]:
            best = (score, a_top)

    a_top = best[1]
    slots = []
    if a_top:
        for i, m in enumerate(_spread(a_top, min(a_top, len(top)))):
            slots.extend([list(top[i])] * m)
    if q - a_top:
        for i, m in enumerate(_spread(q - a_top, min(q - a_top, len(spare)))):
            slots.extend([list(spare[i])] * m)
    return _rotate(uuid, slots[:q], hotkeys)


OMEGA_DEPTH = int(os.environ.get("SN83_OMEGA_DEPTH", "1"))


def picker_depth(pool, uuid, hotkeys, difficulty=None, n_nodes=None, hits=None,
                 n_top_true=0, n_spare_true=0, fleet_n=0):
    """Fill omega DEPTH-deep before spilling to spares, DEPTH from the env.

    Exists to test the depth claim empirically rather than by argument. The
    marginal algebra says depth 1: a second hotkey on a clique we alone hold takes
    cnt 1 -> 2, so both answers get 1/2 where one got 1/1 and the clique's total
    diversity is unchanged. The second hotkey's whole value is then the (1+D)
    optimality term, against (1+D)*exp(-phi*rho) + 1 on a free spare -- and with
    few of ours at omega, exp(-phi*rho) is ~0.97, so the spare keeps its full +1
    of diversity almost for free.

    If that reasoning is wrong, DEPTH=2 or 3 will beat DEPTH=1 here.
    """
    assert pool and hotkeys
    if difficulty is None:
        difficulty = difficulty_from_n(n_nodes) if n_nodes else 0.8
    q = len(hotkeys)
    omega = max(len(c) for c in pool)
    top = [c for c in pool if len(c) == omega]
    spare = [c for c in pool if len(c) == omega - 1]
    if not top:
        return _rotate(uuid, [list(c) for c in (spare * q)[:q]], hotkeys)

    slots = []
    for _ in range(OMEGA_DEPTH):
        for c in top:
            if len(slots) < q:
                slots.append(list(c))
    i = 0
    while len(slots) < q and spare:
        slots.append(list(spare[i % len(spare)]))
        i += 1
    while len(slots) < q:
        slots.append(list(top[len(slots) % len(top)]))
    return _rotate(uuid, slots[:q], hotkeys)


# ---------------------------------------------------------------------------
# Exact optimum against a single known rival.  Proof: autoresearch-runs/
# sn83-picker/PROOF_2P.md.  Lemmas 1-3 reduce the round to a one-dimensional
# enumeration over A_omega, with a closed-form inner solution.
# ---------------------------------------------------------------------------

def _rival_shape(difficulty, b, n_top, n_spare):
    """(B_omega, d_B, m_top, B_spare, d_S, m_sp) from the rival's measured rule."""
    star = TOP4_S_STAR.get(round(difficulty, 1), 3)
    if n_top >= star:
        d = max(1, min(int(b), n_top))
        return b, d, b / float(d), 0.0, 1, 0.0
    d = max(1, min(int(b), n_spare))
    return 0.0, 1, 0.0, b, d, b / float(d)


def _exp_share(x, q_hit, m):
    """E[x/(x+f)] when f = m with probability q_hit and 0 otherwise."""
    if x <= 0:
        return 0.0
    return (1.0 - q_hit) + q_hit * (x / float(x + m)) if m > 0 else 1.0


def _level_value(units, n_cliques, q_hit, m):
    """Total expected diversity from `units` hotkeys placed optimally on a level.

    By Lemma 2 the optimum is an even spread over min(units, n_cliques) cliques:
    every clique is exchangeable, so the first unit is worth the same everywhere
    and marginals decrease with depth.
    """
    if units <= 0:
        return 0.0
    k = max(1, min(units, int(n_cliques)))
    return sum(_exp_share(d, q_hit, m) for d in _spread(units, k))


def optimal_split(difficulty, a, b, n_top, n_spare, omega,
                  usable_top=None, usable_spare=None):
    """The exact best (A_omega, A_spare) against a single known rival.

    Enumerates A_omega over its whole range -- the only quantity coupling the two
    levels, since phi depends on it -- and evaluates each with the closed-form
    inner optimum. Returns (A_omega, total_value).
    """
    n = max(1.0, float(a + b))
    rho = omega / float(omega - 1) if omega > 1 else 1.0
    b_top, d_b, m_top, b_sp, d_s, m_sp = _rival_shape(difficulty, b, n_top, n_spare)
    q_top = min(1.0, d_b / float(max(1, n_top))) if b_top else 0.0
    q_sp = min(1.0, d_s / float(max(1, n_spare))) if b_sp else 0.0

    best = (0, -1.0)
    for a_top in range(0, a + 1):
        phi = (b_top + a_top) / n
        sigma = math.exp(-phi * rho)
        val = (1.0 + difficulty) * (a_top + (a - a_top) * sigma)
        val += _level_value(a_top, usable_top or n_top, q_top, m_top)
        val += _level_value(a - a_top, usable_spare or n_spare, q_sp, m_sp)
        if val > best[1]:
            best = (a_top, val)
    return best


def picker_optimal2p(pool, uuid, hotkeys, difficulty=None, n_nodes=None, hits=None,
                     n_top_true=0, n_spare_true=0, fleet_n=0):
    """Provably optimal against one known rival (PROOF_2P.md theorem, section 6)."""
    assert pool and hotkeys
    if difficulty is None:
        difficulty = difficulty_from_n(n_nodes) if n_nodes else 0.8
    a = len(hotkeys)
    omega = max(len(c) for c in pool)
    top = [c for c in pool if len(c) == omega]
    spare = [c for c in pool if len(c) == omega - 1]
    n_top = max(int(n_top_true) or len(top), 1)
    n_sp = max(int(n_spare_true) or len(spare), 1)

    fn = fleet_n or infer_fleet_n(a, difficulty)
    prof = fleet_profile(fn)
    b = sum(prof.values()) * selection_p(difficulty)

    # The optimiser may only spread over cliques we were actually handed: the
    # supply n_top counts what EXISTS (often 100+) but the pool carries far fewer.
    # Feeding it the supply lets it believe it can be alone more times than it has
    # cliques, which is the proof being given an infeasible action set.
    a_top, _v = optimal_split(difficulty, a, b, n_top, n_sp, omega,
                              usable_top=len(top), usable_spare=len(spare))
    a_top = min(a_top, len(top))
    slots = []
    if a_top:
        for i, mlt in enumerate(_spread(a_top, min(a_top, len(top)))):
            slots.extend([list(top[i])] * mlt)
    rest = a - len(slots)
    if rest and spare:
        for i, mlt in enumerate(_spread(rest, min(rest, len(spare)))):
            slots.extend([list(spare[i])] * mlt)
    while len(slots) < a:
        slots.append(list((top or spare)[len(slots) % len(top or spare)]))
    return _rotate(uuid, slots[:a], hotkeys)


# ---------------------------------------------------------------------------
# Optimal for the DIFFERENCE objective (our mean - their mean), which is the
# metric that matters: rewards are normalised per round, so absolute score is
# not meaningful.  Derivation in PROOF_2P.md section 10.
#
#   J = (1+D)(alpha - beta)(1 - sigma) + c_min[ Div_A(1/a + 1/b) - C/b ]
#
# with alpha = A_omega/a, beta = B_omega/b, and C the number of distinct cliques
# ANY player uses.  Div_B = C - Div_A identically, because the answers on one
# clique split c_min/(x+f) each and sum to 1.
# ---------------------------------------------------------------------------

def _diff_marginals(units, n_cliques, q_hit, m, a, b, cmin=1.0):
    """Greedy value of placing `units` hotkeys on one level, for the J objective.

    Marginals, all non-increasing, so greedy is exact (PROOF_2P Lemma 1):
        fresh clique   c_min/a                      (Div_A +1, C +1)
        occupied, load f   c_min*delta(x,f)*(1/a+1/b)   (Div_A +delta, C +0)
    """
    if units <= 0:
        return 0.0
    w_join = (1.0 / a + 1.0 / b)
    n_occ = q_hit * n_cliques          # cliques the rival already holds
    n_free = max(0.0, n_cliques - n_occ)
    total = 0.0
    left = units
    # first pass: one hotkey on each occupied clique, and on each free clique
    join_first = (m > 0) and ((1.0 / (1.0 + m)) * w_join > 1.0 / a)
    order = []
    if n_occ > 0:
        order.append(("occ", int(n_occ), (1.0 / (1.0 + m)) * w_join if m > 0 else w_join))
    if n_free > 0:
        order.append(("free", int(n_free), 1.0 / a))
    order.sort(key=lambda t: -t[2])
    for _kind, cap, val in order:
        take = min(left, cap)
        total += take * val * cmin
        left -= take
        if left <= 0:
            break
    # remaining hotkeys stack; depth-2+ marginals on occupied cliques
    if left > 0 and n_occ > 0:
        x = 1
        while left > 0 and x < 40:
            d = m / ((x + m) * (x + 1.0 + m)) if m > 0 else 0.0
            take = min(left, int(n_occ))
            total += take * d * w_join * cmin
            left -= take
            x += 1
        if left > 0:
            total += 0.0
    return total


def optimal_split_diff(difficulty, a, b, n_top, n_spare, omega,
                       usable_top=None, usable_spare=None):
    """Exact best A_omega for J = our mean - their mean. Returns (A_omega, J)."""
    a = max(1, int(a))
    b = max(1e-9, float(b))
    n = float(a) + b
    rho = omega / float(omega - 1) if omega > 1 else 1.0
    b_top, d_b, m_top, b_sp, d_s, m_sp = _rival_shape(difficulty, b, n_top, n_spare)
    ut = usable_top or n_top
    us = usable_spare or n_spare
    q_top = min(1.0, d_b / float(max(1, n_top))) if b_top else 0.0
    q_sp = min(1.0, d_s / float(max(1, n_spare))) if b_sp else 0.0
    beta = b_top / b

    best = (0, -1e18)
    for a_top in range(0, a + 1):
        n_omega = b_top + a_top
        if n_omega <= 0:
            sigma = 1.0            # nobody at omega: omega-1 IS max_size
        else:
            sigma = math.exp(-(n_omega / n) * rho)
        alpha = a_top / float(a)
        j = (1.0 + difficulty) * (alpha - beta) * (1.0 - sigma)
        j += _diff_marginals(a_top, ut, q_top, m_top, a, b)
        j += _diff_marginals(a - a_top, us, q_sp, m_sp, a, b)
        if j > best[1]:
            best = (a_top, j)
    return best


def picker_diff(pool, uuid, hotkeys, difficulty=None, n_nodes=None, hits=None,
                n_top_true=0, n_spare_true=0, fleet_n=0):
    """Optimal for our_mean - their_mean against one known rival."""
    assert pool and hotkeys
    if difficulty is None:
        difficulty = difficulty_from_n(n_nodes) if n_nodes else 0.8
    a = len(hotkeys)
    omega = max(len(c) for c in pool)
    top = [c for c in pool if len(c) == omega]
    spare = [c for c in pool if len(c) == omega - 1]
    n_top = max(int(n_top_true) or len(top), 1)
    n_sp = max(int(n_spare_true) or len(spare), 1)
    fn = fleet_n or infer_fleet_n(a, difficulty)
    b = sum(fleet_profile(fn).values()) * selection_p(difficulty)

    a_top, _j = optimal_split_diff(difficulty, a, b, n_top, n_sp, omega,
                                   usable_top=len(top) or 1,
                                   usable_spare=len(spare) or 1)
    a_top = min(a_top, len(top)) if top else 0
    slots = []
    if a_top:
        for i, mlt in enumerate(_spread(a_top, min(a_top, len(top)))):
            slots.extend([list(top[i])] * mlt)
    rest = a - len(slots)
    if rest > 0 and spare:
        for i, mlt in enumerate(_spread(rest, min(rest, len(spare)))):
            slots.extend([list(spare[i])] * mlt)
    while len(slots) < a:
        pick = top or spare
        slots.append(list(pick[len(slots) % len(pick)]))
    return _rotate(uuid, slots[:a], hotkeys)


def picker_proved(pool, uuid, hotkeys, difficulty=None, n_nodes=None, hits=None,
                  n_top_true=0, n_spare_true=0, fleet_n=0):
    """A_omega = min(a, |omega cliques we hold|), one each; the rest on fresh spares.

    This is the optimum of J = our_mean - their_mean derived in PROOF_2P section 10
    and verified against direct scoring (0 mismatches). Every omega-clique taken
    once is worth c_min/a to J and denies the rival max_size; a SECOND hotkey on
    one is worth nothing (delta(x,0)=0) while still costing every other answer of
    ours optimality through sigma. No search, no forecast, no tuning.
    """
    assert pool and hotkeys
    a = len(hotkeys)
    omega = max(len(c) for c in pool)
    top = [c for c in pool if len(c) == omega]
    spare = [c for c in pool if len(c) == omega - 1]
    slots = [list(c) for c in top[:a]]
    i = 0
    while len(slots) < a and spare:
        slots.append(list(spare[i % len(spare)]))
        i += 1
    while len(slots) < a:
        slots.append(list(top[len(slots) % len(top)]))
    return _rotate(uuid, slots[:a], hotkeys)


def picker_random_top(pool, uuid, hotkeys, difficulty=None, n_nodes=None, hits=None,
                      n_top_true=0, n_spare_true=0, fleet_n=0):
    """Control: every hotkey on a RANDOM omega-clique from the pool."""
    import random as _r
    assert pool and hotkeys
    a = len(hotkeys)
    omega = max(len(c) for c in pool)
    top = [c for c in pool if len(c) == omega] or pool
    rng = _r.Random(int(hashlib.sha1(str(uuid).encode()).hexdigest()[:8], 16))
    return [list(top[rng.randrange(len(top))]) for _ in range(a)]


def picker_random_spare(pool, uuid, hotkeys, difficulty=None, n_nodes=None, hits=None,
                        n_top_true=0, n_spare_true=0, fleet_n=0):
    """Control: every hotkey on a RANDOM omega-1 clique from the pool."""
    import random as _r
    assert pool and hotkeys
    a = len(hotkeys)
    omega = max(len(c) for c in pool)
    sp = [c for c in pool if len(c) == omega - 1] or pool
    rng = _r.Random(int(hashlib.sha1(str(uuid).encode()).hexdigest()[:8], 16))
    return [list(sp[rng.randrange(len(sp))]) for _ in range(a)]


def picker_random_any(pool, uuid, hotkeys, difficulty=None, n_nodes=None, hits=None,
                      n_top_true=0, n_spare_true=0, fleet_n=0):
    """Control: every hotkey on a random clique of ANY level in the pool."""
    import random as _r
    assert pool and hotkeys
    a = len(hotkeys)
    rng = _r.Random(int(hashlib.sha1(str(uuid).encode()).hexdigest()[:8], 16))
    return [list(pool[rng.randrange(len(pool))]) for _ in range(a)]


def _greedy_level(units, n_cliques, q_hit, m, a, b):
    """Exact greedy on the J marginals for one level. Returns (value, depths).

    Marginals (PROOF_2P section 10), all non-increasing within a clique:

        free clique, 1st hotkey    1/a                  (Div_A +1, C +1)
        free clique, 2nd+          0                    (delta(x,0)=0, C unchanged)
        occupied,   (x+1)-th       delta(x,m)*(1/a+1/b) (C unchanged)

    The previous version bucketed these and rounded the clique counts to ints,
    which loses the interleaving between "one more free clique" and "go deeper on
    an occupied one". Here the two streams are merged properly.
    """
    if units <= 0 or n_cliques <= 0:
        return 0.0, []
    w = 1.0 / a + 1.0 / b
    n_occ = int(round(q_hit * n_cliques))
    n_free = max(0, int(n_cliques) - n_occ)
    v_free = 1.0 / a
    total = 0.0
    left = int(units)
    depth = [0] * max(1, n_occ)
    used_free = 0
    while left > 0:
        best_v, best_i = -1.0, None
        if used_free < n_free:
            best_v, best_i = v_free, -1
        if n_occ:
            i = min(range(n_occ), key=lambda j: depth[j])
            x = depth[i]
            d = (m / ((x + m) * (x + 1.0 + m))) if m > 0 else 0.0
            v = (1.0 / (1.0 + m)) * w if x == 0 and m > 0 else d * w
            if v > best_v:
                best_v, best_i = v, i
        if best_i is None or best_v <= 0.0:
            break
        if best_i == -1:
            used_free += 1
        else:
            depth[best_i] += 1
        total += best_v
        left -= 1
    return total, (used_free, depth)


def optimal_split_exact(difficulty, a, b, n_top, n_spare, omega,
                        usable_top=None, usable_spare=None):
    """A_omega maximising J, with the exact greedy inner solution."""
    a = max(1, int(a)); b = max(1e-9, float(b))
    n = float(a) + b
    rho = omega / float(omega - 1) if omega > 1 else 1.0
    b_top, d_b, m_top, b_sp, d_s, m_sp = _rival_shape(difficulty, b, n_top, n_spare)
    ut = int(usable_top or n_top); us = int(usable_spare or n_spare)
    q_top = min(1.0, d_b / float(max(1, n_top))) if b_top else 0.0
    q_sp = min(1.0, d_s / float(max(1, n_spare))) if b_sp else 0.0
    beta = b_top / b
    best = (0, -1e18)
    for a_top in range(0, a + 1):
        n_om = b_top + a_top
        sigma = 1.0 if n_om <= 0 else math.exp(-(n_om / n) * rho)
        j = (1.0 + difficulty) * (a_top / float(a) - beta) * (1.0 - sigma)
        j += _greedy_level(a_top, ut, q_top, m_top, a, b)[0]
        j += _greedy_level(a - a_top, us, q_sp, m_sp, a, b)[0]
        if j > best[1]:
            best = (a_top, j)
    return best


def picker_exact(pool, uuid, hotkeys, difficulty=None, n_nodes=None, hits=None,
                 n_top_true=0, n_spare_true=0, fleet_n=0):
    """Optimal for our_mean - their_mean, exact greedy, conditional on occupancy."""
    assert pool and hotkeys
    if difficulty is None:
        difficulty = difficulty_from_n(n_nodes) if n_nodes else 0.8
    a = len(hotkeys)
    omega = max(len(c) for c in pool)
    top = [c for c in pool if len(c) == omega]
    spare = [c for c in pool if len(c) == omega - 1]
    n_top = max(int(n_top_true) or len(top), 1)
    n_sp = max(int(n_spare_true) or len(spare), 1)
    fn = fleet_n or infer_fleet_n(a, difficulty)
    b = sum(fleet_profile(fn).values()) * selection_p(difficulty)
    a_top, _ = optimal_split_exact(difficulty, a, b, n_top, n_sp, omega,
                                   usable_top=len(top) or 1,
                                   usable_spare=len(spare) or 1)
    a_top = min(a_top, a) if top else 0
    slots = []
    if a_top and top:
        for i, mlt in enumerate(_spread(a_top, min(a_top, len(top)))):
            slots.extend([list(top[i])] * mlt)
    rest = a - len(slots)
    if rest > 0 and spare:
        for i, mlt in enumerate(_spread(rest, min(rest, len(spare)))):
            slots.extend([list(spare[i])] * mlt)
    while len(slots) < a:
        pick = top or spare
        slots.append(list(pick[len(slots) % len(pick)]))
    return _rotate(uuid, slots[:a], hotkeys)


# Above this fleet size we hold a large enough majority that the mirror is
# feasible on every clique, so J >= 0 is GUARANTEED whatever the rival does --
# including an adaptive rival who would otherwise force the draw on us. Below it
# we best-respond to the measured fixed rule, which is strictly better than 0 but
# only because they are not adapting.
MINIMAX_N = int(os.environ.get("SN83_MINIMAX_N", "150"))


def _mirror_split(difficulty, a, b, n_top, n_spare):
    """Our level split that mirrors the rival's, which yields J = 0 exactly."""
    b_top, _d, _m, _bs, _ds, _ms = _rival_shape(difficulty, b, n_top, n_spare)
    return int(round(a * (b_top / max(1e-9, b))))


def picker_unified(pool, uuid, hotkeys, difficulty=None, n_nodes=None, hits=None,
                   n_top_true=0, n_spare_true=0, fleet_n=0):
    """One strategy for every fleet size.

    Best-responds to the rival's measured rule via the exact J optimum, with the
    rival PROFILE derived from the metagraph minus our own hotkeys -- so as N grows
    and our registrations displace the smaller operators, the model follows.

    Guard: the minimax value of this game is 0. A rival who mirrors us
    proportionally (f_j = (b/a)x_j, same fractions per level) forces alpha = beta
    and Div_A(1/a+1/b) = C/b, hence J = 0 exactly. Our edge therefore comes
    entirely from the rival playing a FIXED rule rather than best-responding. When
    the model says no allocation beats 0 -- which is what an adaptive rival would
    enforce -- fall back to mirroring their shape, which concedes nothing.
    """
    assert pool and hotkeys
    if difficulty is None:
        difficulty = difficulty_from_n(n_nodes) if n_nodes else 0.8
    a = len(hotkeys)
    omega = max(len(c) for c in pool)
    top = [c for c in pool if len(c) == omega]
    spare = [c for c in pool if len(c) == omega - 1]
    n_top = max(int(n_top_true) or len(top), 1)
    n_sp = max(int(n_spare_true) or len(spare), 1)
    fn = fleet_n or infer_fleet_n(a, difficulty)
    b = max(1e-9, sum(fleet_profile(fn).values()) * selection_p(difficulty))

    if fn >= MINIMAX_N:
        # majority regime: take the guaranteed draw-or-better rather than a best
        # response that an adaptive rival could punish
        a_top = _mirror_split(difficulty, a, b, n_top, n_sp)
    else:
        a_top, j = optimal_split_exact(difficulty, a, b, n_top, n_sp, omega,
                                       usable_top=len(top) or 1,
                                       usable_spare=len(spare) or 1)
        if j <= 0.0:
            a_top = _mirror_split(difficulty, a, b, n_top, n_sp)
    a_top = min(a_top, a) if top else 0

    slots = []
    if a_top and top:
        for i, mlt in enumerate(_spread(a_top, min(a_top, len(top)))):
            slots.extend([list(top[i])] * mlt)
    rest = a - len(slots)
    if rest > 0 and spare:
        for i, mlt in enumerate(_spread(rest, min(rest, len(spare)))):
            slots.extend([list(spare[i])] * mlt)
    while len(slots) < a:
        pick = top or spare
        slots.append(list(pick[len(slots) % len(pick)]))
    return _rotate(uuid, slots[:a], hotkeys)
