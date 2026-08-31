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
