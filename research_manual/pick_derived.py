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
import sys
import random

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths
METAGRAPH = os.environ.get("SN83_METAGRAPH", paths.METAGRAPH_JSON)
ROUNDS_PATH = os.environ.get("SN83_ROUNDS", paths.ROUNDS_JSON)

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
# Entities 2 and 3 run the same fill-and-spill.  Measured on 209 rounds with
# omega-supply >= 3: e2 fills exactly in 89.5% of them (mean take/min(q,S) 0.964),
# e3 in 98.6% (0.996).  The old REACH["e2"] = 0.76 modelled a solver that reaches
# three quarters of the omega cliques; e2 in fact reaches nearly all of them and
# instead ABSTAINS from omega when supply is tiny -- at S <= 2 it places only 0.42
# of min(q,S) there, and in 55% of those rounds nothing at all.  Modelling that
# abstention as a flat reach over-stated the field at omega in the small-supply
# stratum, which is exactly where it flipped our level split.
REACH = {"e2": 0.96, "e3": 0.97}
# e2's reach is a CURVE in supply, not a constant, and not a two-step threshold.
# Measured take/min(q,S) by supply: S=1 0.364 (n=22), S=2 0.618 (17), S=3 0.762 (7),
# S=4 0.821 (14) -- rising and saturating.  One parameter fits all four within their
# CIs: 1 - (1-p)^S with p = 0.36, giving 0.360 / 0.590 / 0.738 / 0.832.  At S=1 the
# behaviour is a clean Bernoulli (0% partial fills, n=22): e2 either takes the single
# omega clique or abstains entirely.  Modelling that as a flat reach put a fractional
# 0.42 answer on it in every round, which is the mean of two regimes that never occur.
REACH_MISS = {"e2": 0.64}
REACH_CAP = 0.96

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

def operator_reach(name, supply):
    """Fraction of the omega cliques an operator places answers on.

    Constant for e3 (measured 0.996 at supply >= 3).  For e2 it rises with supply
    and saturates -- see REACH_MISS.  The curve matters only at small supply, which
    is exactly the stratum where the level split is decided by a couple of answers.
    """
    miss = REACH_MISS.get(name)
    if miss is None:
        return REACH[name]
    return min(REACH_CAP, 1.0 - miss ** max(1, int(supply)))


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
            take = min(q, n_top * operator_reach(name, n_top))
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
    for lvl, q, d, m in rows:
        if lvl != level:
            continue
        # An operator plays spread(q, d): e cliques carry ceil(m) and d-e carry
        # floor(m). Rounding m to a single step over-states the load whenever the
        # fraction is above a half -- at N=30 TOP4 sits at m=2.56, rounded to 3,
        # a 17% over-statement that made the field look more crowded than it is
        # and cost 0.010-0.018 of edge below N=90.
        d_i = max(1.0, d)
        lo = int(math.floor(m))
        e = q - lo * d_i                      # cliques carrying lo+1
        e = min(max(e, 0.0), d_i)
        s = float(max(1, supply))
        p_hi = min(1.0, e / s)
        p_lo = min(1.0 - p_hi, max(0.0, (d_i - e) / s))
        nxt = collections.defaultdict(float)
        for f, pr in law.items():
            nxt[f] += pr * (1.0 - p_hi - p_lo)
            if p_lo > 0.0 and lo > 0:
                nxt[f + lo] += pr * p_lo
            elif p_lo > 0.0:
                nxt[f] += pr * p_lo
            if p_hi > 0.0:
                nxt[f + lo + 1] += pr * p_hi
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


def expected_cliques(law_top, n_top, law_sp, n_sp):
    """Expected number of DISTINCT cliques the field occupies.

    C is a UNION cardinality, not a sum.  Summing each operator's distinct-clique
    count over rows -- which this did -- counts a clique once per operator that
    lands on it, and the operators overlap heavily because they are all solving
    the same graph and finding the same cliques.  The over-count ran 1.25x
    overall and 1.73x on exactly the rounds where it flipped our level split,
    which was the whole of the blind picker's remaining gap to the partial oracle.

    A clique is occupied iff at least one operator lands on it, which the law
    already gives as 1 - P(f=0), so no new estimate is introduced here.
    """
    return n_top * (1.0 - law_top.get(0, 0.0)) + n_sp * (1.0 - law_sp.get(0, 0.0))


def eval_J(difficulty, omega, a, b, alloc_top, alloc_sp, occ_top, occ_sp,
           f_top, f_sp, their_cliques, field_min=1.0, absolute=False,
           tail=False, lam=None, floor_mode=False):
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
    cnt_hi = 1.0
    for alloc, occ in ((alloc_top, occ_top), (alloc_sp, occ_sp)):
        if isinstance(occ, dict):
            hi = max(occ) if occ else 0.0
            cnt_hi = max(cnt_hi, float(hi) + (max(alloc) if alloc else 0.0))
        else:
            for m, f in zip(alloc, occ):
                cnt_hi = max(cnt_hi, float(m) + float(f))
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
    if floor_mode:
        # The metric counts hotkeys under a line, so the worst answer we send matters
        # more than the average one.  Score the allocation by its weakest slot.
        worst = None
        for alloc, occ, is_top in ((alloc_top, occ_top, True), (alloc_sp, occ_sp, False)):
            for m in alloc:
                m = int(m)
                if m <= 0:
                    continue
                v = answer_value(difficulty, sigma, cmin, is_top, m, occ)
                if worst is None or v < worst:
                    worst = v
        return worst if worst is not None else -1e9
    if tail:
        # Deregistration takes the WORST miners, so the quantity to beat is the
        # field's lower tail, not its mean.  Measured: our hotkey means are tightly
        # clustered (spread 0.30) while the field's span 1.6, and the field's 10th
        # percentile lands almost exactly on our median -- so the metric is the
        # MARGIN between our score and that percentile, and it is decided as much by
        # where the cut sits as by where we sit.  Maximising our own mean instead
        # (picker_absolute) raises our median 0.027 but lifts the cut 0.037, which
        # is why it scores WORSE on the tail: 19.0% vs 17.5%.
        #
        # A field miner in that tail is at omega-1 -- if the field is there at all --
        # on the most crowded clique it holds, so its reward is
        #     (1+D)*sigma + c_min/cnt_hi
        # Both terms are ours to push down: sigma falls as we put more hotkeys at
        # omega (it is exp(-phi*rho) and phi counts OUR omega answers), and cnt_hi
        # rises as we pile onto the cliques the field already crowds.
        weak_sigma = sigma if f_sp > 0.0 else 1.0
        their_weak = (1.0 + difficulty) * weak_sigma + cmin / max(1.0, cnt_hi)
        return our_mean - their_weak
    if absolute:
        # OUR mean alone. Optimising the difference was shown to win by
        # suppression: it lowers our_median 0.064 to lower theirs 0.069. This
        # asks the other question -- can we raise our own score at all?
        return our_mean
    their_mean = ((1.0 + difficulty) * (f_top + f_sp * sigma)
                  + cmin * max(0.0, their_cliques - loss)) / float(b)
    if lam is not None:
        # J_lambda = our_mean - lambda*their_mean.  lambda=1 is the edge objective;
        # lambda=0 is our own score alone.  The bottom-10% metric scores us against a
        # FIELD-ONLY threshold, so suppression is worth only what it moves that
        # threshold -- which is an empirical question, not a modelling one.
        return our_mean - lam * their_mean
    return our_mean - their_mean


def _widths(hi):
    """Candidate spread widths.  J is NOT monotone in k: spread() leaves a ragged
    tail, and a clique carrying a single hotkey drops its answer count to 1, which
    collapses c_min for the whole round.  So the widest spread can be beaten by a
    narrower even one -- 15 hotkeys over 8 cliques scores 0.383 where 7 cliques
    score 0.552.  The old code assumed k = kmax by a majorisation argument that
    holds only at fixed c_min.  A full 1..kmax sweep is too slow when cap is in the
    thousands, so take a geometric grid, every divisor of the load (those give an
    even spread with no tail), and the top end exactly.
    """
    if hi <= 0:
        return [0]
    ks = {1, hi}
    k = 1
    while k < hi:
        ks.add(k)
        k = max(k + 1, int(k * 1.4))
    ks.update(x for x in (hi - 1, hi - 2) if x >= 1)
    return sorted(ks)


def _best_widths(difficulty, omega, a, b, a_top, a_sp, occ_top, occ_sp,
                 f_top, f_sp, their_cliques, cap_top, cap_sp, field_min, absolute,
                 tail=False, lam=None, floor_mode=False):
    """Best (alloc_top, alloc_sp) over spread WIDTHS, for a fixed level split."""
    # A level with no cliques cannot hold hotkeys.  Callers used to pass
    # max(1, len(...)) as the cap, which reports room on an EMPTY level and let
    # _emit index spare[0] on a pool holding no omega-1 cliques.  picker() never
    # reached it because allocate always sent everything to omega there; forcing
    # the split (picker_bias) does.  Caps are now the true lengths, clamped here.
    if cap_sp <= 0 and a_sp > 0:
        a_top, a_sp = a_top + a_sp, 0
    if cap_top <= 0 and a_top > 0:
        a_sp, a_top = a_sp + a_top, 0
    kt = [k for k in _widths(min(a_top, cap_top)) if k] if a_top else [0]
    ks = [k for k in _widths(min(a_sp, cap_sp)) if k] if a_sp else [0]
    best_j = None
    best = ([], [])
    for i in kt:
        at = spread(a_top, i) if a_top else []
        for j in ks:
            asp = spread(a_sp, j) if a_sp else []
            v = eval_J(difficulty, omega, a, b, at, asp, occ_top, occ_sp,
                       f_top, f_sp, their_cliques, field_min, absolute, tail, lam, floor_mode)
            if best_j is None or v > best_j:
                best_j = v
                best = (list(at), list(asp))
    return best


def allocate(difficulty, omega, a, b, occ_top, occ_sp, f_top, f_sp,
             their_cliques, cap_top, cap_sp, field_min=1.0, absolute=False,
             tail=False, lam=None, floor_mode=False):
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
            at, asp = _best_widths(difficulty, omega, a, b, a_top, a_sp,
                                   occ_top, occ_sp, f_top, f_sp, their_cliques,
                                   cap_top, cap_sp, field_min, absolute, tail, lam, floor_mode)
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
                   f_top, f_sp, their_cliques, field_min, absolute, tail, lam, floor_mode)
        if best_j is None or j > best_j:
            best_j = j
            best = (list(at), list(asp))
    return best


FAIR = os.environ.get("SN83_FAIR", "0")
# Realized-score feedback.  A deployed miner reads its own past rewards off the
# chain; the validator scores the ANSWER, not the sender, so a table of
# {round: {clique: score}} harvested from an earlier pass reproduces exactly what
# it would know, with no foresight beyond the previous round.  SN83_SCORE_TABLE
# points at that table; SN83_CUT is the field's 10th percentile, also chain-visible,
# and decides the DIRECTION:
#   our median above the cut -> equalise, and the fleet lands on the good side
#   our median below it      -> concentrate, so at least some clear the bar
# Equalising below the cut is the trap: it moves everyone to the wrong side at once
# (ceiling at N=70: 52.9% -> 75.7%).
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
    """Predicted reward for ONE of our answers -- the validator's R_i, in exactly
    the terms eval_J sums over: (1+D)*opt + c_min/cnt, with opt = 1 at omega and
    sigma below it, and cnt = our multiplicity plus the field's load.
    """
    opt = 1.0 if is_top else sigma
    if isinstance(occ, dict):
        div = sum(p / float(m + f) for f, p in occ.items())
    else:
        div = 1.0 / float(m + occ)
    return (1.0 + difficulty) * opt + cmin * div


def _emit_feedback(uuid, hotkeys, slots, values, mode):
    """Hand out slots by each hotkey's RUNNING MEAN so far, not by a fixed rank.

    The static queue tried earlier could not work: it ranked hotkeys by a hash, so
    it never responded to how anyone was actually doing.  82% of the variance in our
    hotkey means is which ROUNDS a hotkey happened to be queried in -- luck a fixed
    rank cannot see, but a running mean can, because a hotkey that drew poor rounds
    simply shows a low mean and gets compensated in the next one.

    mode "1" equalises: the best slot goes to the hotkey with the lowest running
    mean.  That is the right direction when our median is ABOVE the cut and the
    fleet only loses because it straddles it -- which is the case at N=50-90, where
    our median clears the field's 10th percentile by +0.009 and 38.6% of hotkeys
    still fall below.  Compressing the spread moves them all to the same side.

    mode "2" does the opposite as a control: the best slot goes to the hotkey that
    is already highest, concentrating the damage on a designated few.
    """
    order = sorted(range(len(slots)), key=lambda i: -values[i])
    def running(h):
        st = _hk_state.get(h)
        return st[0] / st[1] if st and st[1] else 0.0
    concentrate = mode == "2"
    if mode == "3":
        # Direction chosen from where our fleet sits relative to the cut.
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


def _emit(uuid, hotkeys, top, spare, alloc_top, alloc_sp):
    """Turn multiplicities into one answer per hotkey.

    The ANSWERS are fixed by allocate(); this only decides which hotkey sends
    which one, so the round's score multiset -- and the edge -- are untouched.
    What it does change is how those scores pile up per hotkey across rounds.

    Two schemes:

    rotation (default): offset by a hash of the round id, so over many rounds every
    hotkey draws the same mix.  That is the FAIR choice and the wrong one for a
    deregistration threshold.  It gives every hotkey the same expected score, so the
    whole fleet lands in a 0.06-wide band on top of the field's 10th percentile
    (margin at N=70 is +0.008) and pure noise decides who falls below -- 38.6% of
    them do.

    priority: rank hotkeys by a hash of the HOTKEY, then hand out slots best-first
    down that fixed queue.  The same hotkeys always take the crowded cliques, so
    they sit clearly below the cut while everyone above them sits clearly above it.
    Concentrating the damage on a designated few is what a threshold metric wants:
    with the mean pinned ON the cut, spreading the loss evenly maximises the number
    of marginal hotkeys, which is exactly the failure being fixed.
    """
    slots = []
    for i, m in enumerate(alloc_top):
        slots.extend([(0, int(m), list(top[i]))] * int(m))
    for i, m in enumerate(alloc_sp):
        slots.extend([(1, int(m), list(spare[i]))] * int(m))
    pool = top + spare
    while len(slots) < len(hotkeys):
        slots.append((2, 99, list(pool[len(slots) % len(pool)])))
    if not PRIORITY:
        offset = int(hashlib.sha1(str(uuid).encode()).hexdigest()[:8], 16)
        return [list(slots[(i + offset) % len(slots)][2]) for i in range(len(hotkeys))]
    # best slot first: omega before omega-1, and within a level the cliques we put
    # fewest of our own hotkeys on (those are the ones that stay uncrowded).
    order = sorted(range(len(slots)), key=lambda i: (slots[i][0], slots[i][1], i))
    queue = sorted(range(len(hotkeys)), key=lambda h: _hotkey_rank(hotkeys[h]))
    out = [None] * len(hotkeys)
    for pos, h in enumerate(queue):
        out[h] = list(slots[order[pos % len(order)]][2])
    return out


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
    law_t = law_from_plan(rows, "top", n_top)
    law_s = law_from_plan(rows, "spare", n_sp)
    their_cliques = expected_cliques(law_t, n_top, law_s, n_sp)
    # blind bound on c_min: the lightest load the field is expected to leave on
    # at least one clique it holds, outside whatever we take
    occupied = [(f, p * n) for law, n in ((law_t, n_top), (law_s, n_sp))
                for f, p in law.items() if f > 0]
    field_min = float(min((f for f, cnt in occupied if cnt >= 1.0), default=0.0))
    at, asp = allocate(difficulty, omega, a, b, law_t, law_s,
                       f_top, f_sp, their_cliques,
                       len(top), len(spare), field_min)
    return _emit(uuid, hotkeys, top, spare, at, asp)


ATOP_BIAS = int(os.environ.get("SN83_ATOP_BIAS", "0"))
# Fractional version.  The absolute knob is far too coarse: the validator queries
# only ~19-30 of our hotkeys in a round, so a shift of 3 is a 16% swing in the
# level split, which is why the first sweep saw only a cliff at 0 and no shape.
ATOP_EPS = float(os.environ.get("SN83_ATOP_EPS", "0"))


def picker_bias(pool, uuid, hotkeys, difficulty=None, n_nodes=None, hits=None,
                n_top_true=0, n_spare_true=0, fleet_n=0):
    """picker(), then shift the level split by SN83_ATOP_BIAS hotkeys.

    A probe, not a strategy: the tail metric is the margin between our score and
    the field's 10th percentile, and both move with phi (our share of omega
    answers).  Measured at N=110, raising phi by 0.167 moved the cut -0.105 but our
    own median -0.187, so the margin's derivative in phi is negative and the
    optimum is at LOWER phi than the difference objective picks.  This sweeps it.
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
    """picker(), but never put a SECOND hotkey on an omega clique.

    Measured at N=70: within every crowding cell our per-answer score matches the
    field's (alone 2.7584 vs 2.7649; cnt=2 2.3111 vs 2.3776), and the whole 0.061
    deficit in our mean is the MIX -- we are alone 58.1% of the time against their
    68.7%.  Giving us their mix would recover 0.045 of it.

    8.3% of our answers are self-collisions, which arise only when a_top exceeds the
    omega cliques we hold and spread() has to double up.  A stacked omega answer is
    worth 2.31; a fresh omega-1 answer is worth 2.67.  So the excess should spill to
    omega-1 rather than pile up.  This is CONDITIONAL on the overflow, which is why
    it is not the same as shifting the level split globally -- that was swept over
    217 rounds at +-8% and +-15% and is sharply worse in both directions.
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
    """picker(), but the ANSWERS are handed out by each hotkey's running mean.

    Identical allocation, so the round's score multiset and the edge are untouched;
    only which hotkey sends which answer changes.  SN83_FAIR=1 equalises, =2
    concentrates.  State is per-process and accumulates across rounds, which a
    deployed miner can do too -- it knows what it sent and can price it.
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
    """picker() with the value function reweighted: J = our_mean - lambda*their_mean.

    SN83_LAMBDA=1 reproduces picker() exactly; 0 optimises our own score alone.
    SN83_FLOOR=1 switches to maximin over our own answers instead.
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
    """picker(), but CHOOSING which cliques to occupy out of a deeper pool.

    allocate() spreads one hotkey per clique over the front of the pool, so the pool's
    ORDER decides which cliques we take.  The default order is basin descending -- the
    most findable cliques first -- and a rival solver lands on exactly those.  Measured
    at N=70: our alone-rate at omega is 0.480 against TOP4's 0.633 and the field top
    decile's 0.748, while omega-RATE is identical (0.82) across all of them.  The gap is
    worst where supply is AMPLE (30-99 cliques: 0.263 vs 0.565), so it is choice, not
    scarcity.  SN83_POOL_PICK reorders: hits_asc takes the least findable, random takes
    an arbitrary subset, hits_desc is the current behaviour.
    """
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
    """Take only as many omega cliques as we expect to hold ALONE; spill the rest.

    The law already gives P(f=0) at omega, so n_top*P(f=0) is the expected number of
    free omega cliques.  Beyond that we are landing on the field, and a crowded omega
    answer (measured 2.317) is worth less than an uncrowded omega-1 one (2.675), of
    which there are ~1230 with almost nobody on them.  SN83_FREE_CAP scales the cap;
    0 disables it and reproduces picker().
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
    """BLIND, but aimed at the field's LOWER TAIL instead of its mean.

    Same field model and same allocator as picker(); only the value function
    differs.  See eval_J's tail branch for why the tail is a different objective
    from both the difference and our own absolute score.
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
    # Only cliques OUTSIDE our pool.  The claim this replaced -- that in-pool
    # cliques we give no hotkeys to are covered by neither the alloc loop nor the
    # outside minimum -- is false: c_min_of's exact branch keeps every pool clique
    # with m + f > 0, m = 0 included.  Passing their raw f here a SECOND time, as a
    # floor, ignores the hotkeys we just piled onto them and can only drag c_min
    # down.  On round 35229818 at N=160 the realised counts were [20..22] and this
    # returned 1.0.  It moved allocate()'s argmax on 29/100 rounds at N=160.
    ours = {tuple(sorted(c)) for c in top} | {tuple(sorted(c)) for c in spare}
    held = [v for k, v in field.items() if v > 0 and k not in ours]
    field_min = float(min(held)) if held else 0.0
    at, asp = allocate(difficulty, omega, a, b,
                       [float(field[tuple(sorted(c))]) for c in top],
                       [float(field[tuple(sorted(c))]) for c in spare],
                       f_top, f_sp, their_cliques,
                       len(top), len(spare), field_min)
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

    # Only cliques OUTSIDE our pool.  The claim this replaced -- that in-pool
    # cliques we give no hotkeys to are covered by neither the alloc loop nor the
    # outside minimum -- is false: c_min_of's exact branch keeps every pool clique
    # with m + f > 0, m = 0 included.  Passing their raw f here a SECOND time, as a
    # floor, ignores the hotkeys we just piled onto them and can only drag c_min
    # down.  On round 35229818 at N=160 the realised counts were [20..22] and this
    # returned 1.0.  It moved allocate()'s argmax on 29/100 rounds at N=160.
    ours = {tuple(sorted(c)) for c in top} | {tuple(sorted(c)) for c in spare}
    held = [v for k, v in field.items() if v > 0 and k not in ours]
    field_min = float(min(held)) if held else 0.0
    at, asp = allocate(difficulty, omega, a, b, law(omega, n_top),
                       law(omega - 1, n_sp), f_top, f_sp, their_cliques,
                       len(top), len(spare), field_min)
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
