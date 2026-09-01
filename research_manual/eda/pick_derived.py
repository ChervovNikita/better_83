#!/usr/bin/env python3
"""Picker derived from the SN83 reward algebra.

The validator pays, for response i among the round's n responses,

    R_i = (1+D)*exp(-pr_i/rel_i) + c_min/cnt_i

with cnt_i the number of miners on i's exact clique and c_min the smallest such
count.  Someone is always at the top size, so the optimality normaliser is 1 and

    opt(omega) = 1,   opt(omega-1) = exp(-phi*rho),   phi = n_omega/n, rho = w/(w-1)

The metric is a DIFFERENCE of means (rewards are normalised per round, so an
absolute score means nothing), which makes maximising our own reward the wrong
objective: in rounds the rival vacates omega, placing a few hotkeys there lowers
our mean and lowers theirs further.

Every quantity below comes from the metagraph and each operator's measured rule.
There is exactly one value function (eval_J) and one allocator (allocate); the
pickers differ only in what they are told about the field.  Earlier versions of
this file carried four copies of that algebra and the same two defects kept
reappearing in whichever copy had not been fixed.
"""

import collections
import hashlib
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
METAGRAPH = os.environ.get("SN83_METAGRAPH",
                           os.path.join(os.path.dirname(HERE), "metagraph.json"))
ROUNDS_PATH = os.environ.get("SN83_ROUNDS",
                             os.path.join(os.path.dirname(HERE), "rounds.json"))

REF_R = 1.5
IMMUNITY_BLOCKS = 6000

# Operator identity, measured from coldkey transaction analysis.
OPERATORS = {
    "top4": ("5HMevt8h", "5Eyh8ePM", "5EfHz7fE", "5Hg2Ps2L"),
    "e2": ("5D7BMeGt",),
    "e3": ("5GghBgin",),
}
# TOP4's level rule, fitted exactly on 739 rounds: omega iff supply >= S*(D).
TOP4_S_STAR = {0.7: 12, 0.8: 8, 0.9: 5, 1.0: 3}
# Entities 2 and 3 run the same fill-and-spill and differ only in solver reach.
REACH = {"e2": 0.76, "e3": 0.97}

FLEET_N = int(os.environ.get("SN83_FLEET_N", "0"))

_profile_cache = {}
_victim_cache = {}
_rounds_cache = {}


# --------------------------------------------------------------------------
# round primitives
# --------------------------------------------------------------------------

def selection_p(difficulty):
    """MinerSelector.miner_selection_probabilities, per hotkey per round."""
    return 1.0 - math.exp(-max(0.0, math.sqrt(1.0 + REF_R) - difficulty - 0.5))


def difficulty_from_n(number_of_nodes):
    """Difficulty from the vertex count, as a deployed miner recovers it."""
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


def spread(total, parts):
    """Even split of `total` over `parts`, largest first, max-min <= 1."""
    base, extra = divmod(int(total), int(parts))
    return [base + 1] * extra + [base] * (int(parts) - extra)


def infer_fleet_n(q, difficulty):
    """Our fleet size: from the environment if set, else q/p(D)."""
    if FLEET_N > 0:
        return FLEET_N
    return int(round(q / selection_p(difficulty)))


# --------------------------------------------------------------------------
# who is registered alongside us
# --------------------------------------------------------------------------

def _load_metagraph():
    if "meta" not in _profile_cache:
        with open(METAGRAPH) as handle:
            _profile_cache["meta"] = json.load(handle)
    return _profile_cache["meta"]


def _operator_of(coldkey):
    for name, prefixes in OPERATORS.items():
        if coldkey.startswith(prefixes):
            return name
    return "other"


def _churn_order(meta):
    """Hotkeys in the order the validator deregisters them: weakest first."""
    block = meta["block"]
    return [m for m in meta["miners"]
            if block - m["block_at_registration"] >= IMMUNITY_BLOCKS]


def victim_hotkeys(fleet_n):
    """The hotkeys our fleet_n registrations displace."""
    key = int(fleet_n)
    if key not in _victim_cache:
        cand = _churn_order(_load_metagraph())
        _victim_cache[key] = {m["hotkey"] for m in cand[:max(0, key)]}
    return _victim_cache[key]


def fleet_profile(fleet_n):
    """Rival hotkey counts per operator, after our registrations displace theirs.

    Displacement is not proportional: churn removes the lowest-incentive miners,
    which are the smaller operators, so entities 2 and 3 vanish entirely well
    before the dominant one is touched.  A model with fixed counts describes a
    field our own growth deleted.
    """
    key = int(fleet_n)
    if key in _profile_cache:
        return _profile_cache[key]
    meta = _load_metagraph()
    taken = victim_hotkeys(key)
    counts = collections.Counter()
    for m in meta["miners"]:
        if m["hotkey"] not in taken:
            counts[_operator_of(m["coldkey"])] += 1
    _profile_cache[key] = counts
    return counts


# --------------------------------------------------------------------------
# what the field will do this round
# --------------------------------------------------------------------------

def entity_plan(difficulty, n_top, n_spare, fleet_n):
    """One (level, answers, distinct_cliques, multiplicity) row per operator.

    Each operator's rule is measured; an unknown coldkey takes the generic
    always-omega behaviour of the remainder rather than inheriting a threshold.
    """
    p = selection_p(difficulty)
    star = TOP4_S_STAR[round(difficulty, 1)]
    rows = []
    for name, count in fleet_profile(fleet_n).items():
        q = count * p
        if q <= 0.0:
            continue
        if name == "top4":
            if n_top >= star:
                d = max(1.0, min(q, float(n_top)))
                rows.append(("top", q, d, q / d))
            else:
                d = max(1.0, min(q, float(n_spare)))
                rows.append(("spare", q, d, q / d))
        elif name in REACH:
            take = min(q, n_top * REACH[name])
            if take > 0.0:
                d = max(1.0, min(take, float(n_top)))
                rows.append(("top", take, d, take / d))
            rest = q - take
            if rest > 0.0:
                d = max(1.0, min(rest, float(n_spare)))
                rows.append(("spare", rest, d, rest / d))
        else:
            d = max(1.0, min(q, float(n_top)))
            rows.append(("top", q, d, q / d))
    return rows


def law_from_plan(rows, level, supply):
    """Load distribution on one clique, convolved over independent operators.

    A clique is free only if EVERY operator missed it, P(free) = prod(1-d_i/S).
    A single two-point law is right for one rival and badly wrong for several:
    measured at N=30 it claimed 54.8% of omega-cliques were free against 35.7%.
    """
    law = {0: 1.0}
    for lvl, _q, d, m in rows:
        if lvl != level:
            continue
        p_hit = min(1.0, d / float(max(1, supply)))
        step = max(1, int(round(m)))
        nxt = collections.defaultdict(float)
        for f, pr in law.items():
            nxt[f] += pr * (1.0 - p_hit)
            nxt[f + step] += pr * p_hit
        law = dict(nxt)
    return law


# --------------------------------------------------------------------------
# THE value function and THE allocator
# --------------------------------------------------------------------------

def j_marginal(m, f, a, b):
    """Gain in J from adding one more hotkey to a clique already holding m of
    ours and f of theirs.  THE marginal -- allocate must not carry its own.

    Adding to an occupied clique (f>0) raises our diversity by delta AND lowers
    theirs by exactly the same delta, so it is worth delta*(1/a + 1/b).  A free
    clique gives us delta and costs them nothing: delta/a.  Comparing bare delta
    across cliques -- which an earlier private copy of this did -- maximises OUR
    diversity instead of the difference, and picks fresh cliques when
    PROOF_2P section 10 says joining wins (f < a/b).
    """
    d = (m + 1) / float(m + 1 + f) - (m / float(m + f) if m else 0.0)
    return d * (1.0 / a + 1.0 / b) if f > 0 else d / a


def c_min_of(alloc_top, alloc_sp, occ_top, occ_sp, field_min):
    """Expected smallest answer count over the round -- the diversity numerator.

    The validator pays c_min/cnt_i (clique_scoring.py:99-111), so this is the
    weight between the diversity bracket and the (1+D) optimality term.  Because
    a != b it does NOT cancel out of J, and because the allocator changes it by
    its own choices it moves the argmax, not merely the reported value.

    Two mistakes are easy here and both were made:

      * taking min over the joint SUPPORT of the per-clique laws, which minimises
        together outcomes that cannot co-occur and returns the best case rather
        than the expectation;
      * counting only cliques we allocate to, so an occupied clique sitting in
        our pool with none of our hotkeys on it is missed by this AND by
        field_min (which covers only cliques outside the pool).

    Exact loads give a deterministic minimum.  Laws give E[min] via the survival
    function, treating cliques as independent:  E[c] = sum_k P(c >= k).
    """
    exact = []
    dists = []
    for alloc, occ in ((alloc_top, occ_top), (alloc_sp, occ_sp)):
        if isinstance(occ, dict):
            # every clique of our pool at this level, allocated or not
            for m in alloc:
                dists.append((m, occ))
        else:
            for m, f in zip(alloc, occ):
                if m + f > 0:
                    exact.append(float(m + f))
    if field_min:
        exact.append(float(field_min))

    if not dists:
        return float(min(exact)) if exact else 1.0

    floor = min(exact) if exact else None
    top_k = int(max(m + max(law) for m, law in dists)) + 1
    if floor is not None:
        top_k = min(top_k, int(floor))
    expected = 0.0
    for k in range(1, max(2, top_k + 1)):
        if floor is not None and floor < k:
            break
        p_all = 1.0
        for m, law in dists:
            p_all *= sum(p for f, p in law.items() if m + f >= k or m + f == 0)
        expected += p_all
    return max(1.0, expected)


def eval_J(difficulty, omega, a, b, alloc_top, alloc_sp, occ_top, occ_sp,
           f_top, f_sp, their_cliques, field_min=1.0, absolute=False):
    """J = our_mean - their_mean for one fully specified allocation.

    alloc_* : our multiplicity per clique we use.
    occ_*   : the field's load -- a LIST (exact loads, position-aware) or a
              DICT law f -> P(f) (exchangeable, which is all a blind picker has).
    their_cliques : distinct cliques the field occupies, ours or not.

    On a clique carrying f of theirs, adding m of ours takes their diversity from
    1 to f/(m+f), so THEIR LOSS IS m/(m+f) -- identically our gain there.  A free
    clique gives us 1 and costs them nothing.
    """
    n = float(a) + float(b)
    rho = omega / float(omega - 1)
    a_top = float(sum(alloc_top))
    n_omega = f_top + a_top
    sigma = 1.0 if n_omega <= 0.0 else math.exp(-(n_omega / n) * rho)

    gain = 0.0
    loss = 0.0
    for alloc, occ in ((alloc_top, occ_top), (alloc_sp, occ_sp)):
        if isinstance(occ, dict):
            for m in alloc:
                if m <= 0:
                    continue
                for f, p in occ.items():
                    g = m / float(m + f)
                    gain += p * g
                    if f > 0:
                        loss += p * g
        else:
            for m, f in zip(alloc, occ):
                if m <= 0:
                    continue
                g = m / float(m + f)
                gain += g
                if f > 0:
                    loss += g

    cmin = c_min_of(alloc_top, alloc_sp, occ_top, occ_sp, field_min)
    our_mean = ((1.0 + difficulty) * (a_top + (a - a_top) * sigma)
                + cmin * gain) / float(a)
    if absolute:
        # OUR mean alone. Optimising the difference was shown to win by
        # suppression: it lowers our_median 0.064 to lower theirs 0.069. This
        # asks the other question -- can we raise our own score at all?
        return our_mean
    their_mean = ((1.0 + difficulty) * (f_top + f_sp * sigma)
                  + cmin * max(0.0, their_cliques - loss)) / float(b)
    return our_mean - their_mean


def allocate(difficulty, omega, a, b, occ_top, occ_sp, f_top, f_sp,
             their_cliques, cap_top, cap_sp, field_min=1.0, absolute=False):
    """Best (alloc_top, alloc_sp) under eval_J.

    phi depends on how many of our hotkeys sit at omega, so a_top is the one
    quantity coupling the levels and is enumerated exhaustively.  Given a_top:

      * with a LAW, every clique is exchangeable and E[m/(m+f)] is concave in m,
        so by majorisation the even spread is optimal -- there is nothing to
        search, and searching would price knowledge we do not have;
      * with exact LOADS the cliques genuinely differ, so hotkeys go greedily to
        the largest marginal, which is sound only because the loads are known.
    """
    best_j = None
    best = None
    for a_top in range(0, a + 1):
        a_sp = a - a_top
        if isinstance(occ_top, dict):
            at = spread(a_top, max(1, min(a_top, cap_top))) if a_top else []
            asp = spread(a_sp, max(1, min(a_sp, cap_sp))) if a_sp else []
        else:
            at = [0] * len(occ_top)
            asp = [0] * len(occ_sp)
            for target, occ, units in ((at, occ_top, a_top), (asp, occ_sp, a_sp)):
                if not occ:
                    continue
                for _ in range(units):
                    best_i = 0
                    best_d = -1e18
                    for i, f in enumerate(occ):
                        d = j_marginal(target[i], f, a, b)
                        if d > best_d:
                            best_d = d
                            best_i = i
                    target[best_i] += 1
        j = eval_J(difficulty, omega, a, b, at, asp, occ_top, occ_sp,
                   f_top, f_sp, their_cliques, field_min, absolute)
        if best_j is None or j > best_j:
            best_j = j
            best = (list(at), list(asp))
    return best


def _emit(uuid, hotkeys, top, spare, alloc_top, alloc_sp):
    """Turn multiplicities into one answer per hotkey, rotated by round."""
    slots = []
    for i, m in enumerate(alloc_top):
        slots.extend([list(top[i])] * int(m))
    for i, m in enumerate(alloc_sp):
        slots.extend([list(spare[i])] * int(m))
    pool = top + spare
    while len(slots) < len(hotkeys):
        slots.append(list(pool[len(slots) % len(pool)]))
    offset = int(hashlib.sha1(str(uuid).encode()).hexdigest()[:8], 16)
    return [list(slots[(i + offset) % len(slots)]) for i in range(len(hotkeys))]


def _levels(pool):
    omega = max(len(c) for c in pool)
    return omega, [c for c in pool if len(c) == omega], \
        [c for c in pool if len(c) == omega - 1]


# --------------------------------------------------------------------------
# pickers: identical machinery, different information
# --------------------------------------------------------------------------

def picker(pool, uuid, hotkeys, difficulty=None, n_nodes=None, hits=None,
           n_top_true=0, n_spare_true=0, fleet_n=0):
    """BLIND: the field is modelled from operator counts and measured rules."""
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
    their_cliques = sum(min(d, float(n_top if lvl == "top" else n_sp))
                        for lvl, _q, d, _m in rows)
    law_t = law_from_plan(rows, "top", n_top)
    law_s = law_from_plan(rows, "spare", n_sp)
    # blind bound on c_min: the lightest load the field is expected to leave on
    # at least one clique it holds, outside whatever we take
    occupied = [(f, p * n) for law, n in ((law_t, n_top), (law_s, n_sp))
                for f, p in law.items() if f > 0]
    field_min = float(min((f for f, cnt in occupied if cnt >= 1.0), default=0.0))
    at, asp = allocate(difficulty, omega, a, b, law_t, law_s,
                       f_top, f_sp, their_cliques,
                       max(1, len(top)), max(1, len(spare)), field_min)
    return _emit(uuid, hotkeys, top, spare, at, asp)


def _field_counter(uuid, fleet_n):
    """The field's actual answers this round, excluding hotkeys we displaced."""
    if "r" not in _rounds_cache:
        with open(ROUNDS_PATH) as handle:
            _rounds_cache["r"] = json.load(handle)
    rec = _rounds_cache["r"][str(uuid)]
    victims = victim_hotkeys(fleet_n)
    return collections.Counter(tuple(sorted(x[3])) for x in rec["answers"]
                               if x[3] and x[1] not in victims)


def picker_oracle(pool, uuid, hotkeys, difficulty=None, n_nodes=None, hits=None,
                  n_top_true=0, n_spare_true=0, fleet_n=0):
    """ORACLE: exact per-clique loads.  Not deployable; it bounds the blind play."""
    if difficulty is None:
        difficulty = difficulty_from_n(n_nodes)
    a = len(hotkeys)
    omega, top, spare = _levels(pool)
    field = _field_counter(uuid, fleet_n or infer_fleet_n(a, difficulty))
    f_top = float(sum(v for k, v in field.items() if len(k) == omega))
    f_sp = float(sum(v for k, v in field.items() if len(k) == omega - 1))
    b = max(1e-9, f_top + f_sp)
    their_cliques = float(sum(1 for v in field.values() if v > 0))
    # every clique the field holds, including ones inside our pool that we may
    # give no hotkeys to -- those are covered by neither the alloc loop nor an
    # outside-the-pool minimum, and vanished from c_min entirely
    held = [v for v in field.values() if v > 0]
    field_min = float(min(held)) if held else 0.0
    at, asp = allocate(difficulty, omega, a, b,
                       [float(field[tuple(sorted(c))]) for c in top],
                       [float(field[tuple(sorted(c))]) for c in spare],
                       f_top, f_sp, their_cliques,
                       max(1, len(top)), max(1, len(spare)), field_min)
    return _emit(uuid, hotkeys, top, spare, at, asp)


def picker_partial(pool, uuid, hotkeys, difficulty=None, n_nodes=None, hits=None,
                   n_top_true=0, n_spare_true=0, fleet_n=0):
    """PARTIAL ORACLE: the true occupancy MULTISET, but not which clique is which."""
    if difficulty is None:
        difficulty = difficulty_from_n(n_nodes)
    a = len(hotkeys)
    omega, top, spare = _levels(pool)
    n_top = max(int(n_top_true), len(top), 1)
    n_sp = max(int(n_spare_true), len(spare), 1)
    field = _field_counter(uuid, fleet_n or infer_fleet_n(a, difficulty))
    f_top = float(sum(v for k, v in field.items() if len(k) == omega))
    f_sp = float(sum(v for k, v in field.items() if len(k) == omega - 1))
    b = max(1e-9, f_top + f_sp)
    their_cliques = float(sum(1 for v in field.values() if v > 0))

    def law(level, supply):
        loads = [v for k, v in field.items() if len(k) == level]
        counts = collections.Counter(loads)
        total = max(int(supply), len(loads), 1)
        out = {f: c / float(total) for f, c in counts.items()}
        out[0] = max(0.0, 1.0 - sum(out.values()))
        return out

    # every clique the field holds, including ones inside our pool that we may
    # give no hotkeys to -- those are covered by neither the alloc loop nor an
    # outside-the-pool minimum, and vanished from c_min entirely
    held = [v for v in field.values() if v > 0]
    field_min = float(min(held)) if held else 0.0
    at, asp = allocate(difficulty, omega, a, b, law(omega, n_top),
                       law(omega - 1, n_sp), f_top, f_sp, their_cliques,
                       max(1, len(top)), max(1, len(spare)), field_min)
    return _emit(uuid, hotkeys, top, spare, at, asp)


ABSOLUTE = os.environ.get("SN83_ABSOLUTE", "0") == "1"


def picker_absolute(pool, uuid, hotkeys, difficulty=None, n_nodes=None, hits=None,
                    n_top_true=0, n_spare_true=0, fleet_n=0):
    """Maximise OUR OWN mean reward instead of the difference.

    The difference-optimal picker beats the baseline on edge while scoring worse
    per hotkey (64 better / 1296 worse, p=1e-244) -- it wins by suppression, and
    emissions follow absolute share. This asks whether the same machinery, aimed
    at our own score, can beat the baseline on the quantity that pays.
    """
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
    their_cliques = sum(min(d, float(n_top if lvl == "top" else n_sp))
                        for lvl, _q, d, _m in rows)
    law_t = law_from_plan(rows, "top", n_top)
    law_s = law_from_plan(rows, "spare", n_sp)
    occupied = [(f, p * n) for law, n in ((law_t, n_top), (law_s, n_sp))
                for f, p in law.items() if f > 0]
    field_min = float(min((f for f, cnt in occupied if cnt >= 1.0), default=0.0))
    at, asp = allocate(difficulty, omega, a, b, law_t, law_s, f_top, f_sp,
                       their_cliques, max(1, len(top)), max(1, len(spare)),
                       field_min, absolute=True)
    return _emit(uuid, hotkeys, top, spare, at, asp)
