#!/usr/bin/env python3
"""How many hotkeys to put on omega, without knowing what the field returns.

    plan(q, omega, n_top, n_spare, n_others, a_hat, difficulty) -> Plan

Step 1 of the two-step scheme: the field's answers are unknown, so they are
modelled.  `a_hat` (how many OTHER answers reach omega) comes from the predictor
in fit_field.py; everything else is known at solve time.

MODEL
-----
Of the `n_others` other answers, `a_hat` are at omega and pick uniformly among
the P distinct omega-cliques, so the count of others on any one clique is
F ~ Binomial(a_hat, 1/P).  Our omega-1 answers contend the same way against the
`b_hat` others at omega-1, spread over the far larger pool of omega-1 cliques.

With t of our q hotkeys on omega (spread as evenly as possible over P cliques):

    N        = n_others + q
    opt(w)   = 1
    opt(w-1) = exp(-((a_hat + t)/N) * w/(w-1))     if a_hat + t > 0
             = 1                                    if a_hat + t == 0

The second branch is not a special case for tidiness: if nobody submits omega,
the max size IS omega-1, every omega-1 answer gets pr = 0, and holding back
costs nothing at all.

Expected value is scanned over t in 0..q.  One dimension, exact.

WHAT THIS DELIBERATELY DOES NOT DO
----------------------------------
It never looks at which cliques the field returned.  oracle_pick.py does, and is
the ceiling this is measured against.

The uniform-pick assumption is the weak point: real solvers are drawn to large
basins, so the field concentrates on the same few cliques far more than uniform.
That makes F under-dispersed here and the value of holding an omega clique
OVERSTATED, so the model is conservative about holding back.  measure() reports
the gap against the oracle so the size of that error is visible.
"""

import argparse
import collections
import math


class Plan(object):
    def __init__(self, t, value, detail):
        self.t = t                  # hotkeys submitting an omega clique
        self.value = value          # expected mean reward per hotkey
        self.detail = detail

    def __repr__(self):
        return "Plan(t=%d, value=%.4f)" % (self.t, self.value)


def _log_binom_pmf(a, p, f):
    if p <= 0.0:
        return 0.0 if f == 0 else float("-inf")
    if p >= 1.0:
        return 0.0 if f == a else float("-inf")
    return (math.lgamma(a + 1) - math.lgamma(f + 1) - math.lgamma(a - f + 1)
            + f * math.log(p) + (a - f) * math.log1p(-p))


def expected_share(a, n_distinct, mine):
    """E[mine / (F + mine)] with F ~ Binomial(a, 1/n_distinct).

    The diversity our `mine` hotkeys collect from one clique, given `a` field
    answers spread uniformly over `n_distinct` cliques.
    """
    if mine <= 0:
        return 0.0
    if n_distinct <= 0:
        return 0.0
    if a <= 0:
        return 1.0
    p = 1.0 / n_distinct
    total = 0.0
    for f in range(a + 1):
        lp = _log_binom_pmf(a, p, f)
        if lp > -60.0:
            total += math.exp(lp) * mine / float(f + mine)
    return total


def _spread(t, n_distinct):
    """How t hotkeys divide over n_distinct cliques, as evenly as possible."""
    if t <= 0 or n_distinct <= 0:
        return []
    base, extra = divmod(t, n_distinct)
    return [base + 1] * extra + [base] * (n_distinct - extra)


def marginal(a, n_distinct, m, A):
    """Gain from the m-th of our hotkeys on one clique. DERIVATION.md, "Marginals".

    A = opt(size) * (1 + difficulty);  the diversity part is
    E[m/(F+m)] - E[(m-1)/(F+m-1)] with F ~ Bin(a, 1/n_distinct).
    """
    assert m >= 1
    return A + expected_share(a, n_distinct, m) - expected_share(a, n_distinct, m - 1)


def _greedy(classes, q):
    """Top-q marginals, which is exactly optimal for fixed A (DERIVATION.md).

    `classes` is [(key, n_distinct, a, A)] -- one entry per size class. Within a
    class every clique has the same marginal under A4, so greedy round-robins and
    the allocation is fully described by how many hotkeys each class receives.
    """
    counts = {k: 0 for k, _n, _a, _A in classes}
    total = 0.0
    for _ in range(q):
        best, best_gain = None, None
        for key, n_distinct, a, A in classes:
            if n_distinct <= 0:
                continue
            taken = counts[key]
            # round-robin: the next hotkey lands on the least-loaded clique,
            # which currently holds floor(taken / n_distinct)
            m = taken // n_distinct + 1
            gain = marginal(a, n_distinct, m, A)
            if best_gain is None or gain > best_gain:
                best, best_gain = key, gain
        if best is None:
            break
        counts[best] += 1
        total += best_gain
    return counts, total


def plan(q, omega, n_top, n_spare, n_others, a_hat, difficulty, b_hat=0):
    """How many hotkeys go to omega, and how many to omega-1.

    Enumerates the size split t (A5: opt depends on it), solving each split
    exactly by greedy, and keeps the best.  Returns a Plan whose detail carries
    the per-class counts.
    """
    assert q > 0 and omega > 0 and n_top > 0
    a_hat = max(0, int(round(a_hat)))
    b_hat = max(0, int(round(b_hat)))
    total_answers = max(1, int(n_others) + q)

    best = None
    for t in range(0, q + 1):
        if t < q and (n_spare <= 0 or omega <= 1):
            continue
        at_omega = a_hat + t
        opt_short = (1.0 if at_omega == 0 else
                     math.exp(-(at_omega / float(total_answers)) * omega / (omega - 1.0)))
        classes = [("top", n_top, a_hat, 1.0 * (1.0 + difficulty))]
        if n_spare > 0 and omega > 1:
            classes.append(("spare", n_spare, b_hat, opt_short * (1.0 + difficulty)))
        counts, value = _greedy(classes, q)
        if counts.get("top", 0) != t:
            continue                    # greedy disagrees with this split; skip
        detail = {"t": t, "counts": counts, "opt_short": opt_short,
                  "top_spread": _spread(counts.get("top", 0), n_top),
                  "spare_spread": _spread(counts.get("spare", 0), max(1, n_spare))}
        detail["distinct"] = sum(1 for m in detail["top_spread"] + detail["spare_spread"] if m)
        detail["duplicated"] = sum(m - 1 for m in detail["top_spread"] + detail["spare_spread"] if m > 1)
        if best is None or value / q > best.value:
            best = Plan(t, value / q, detail)

    if best is None:                    # no split was self-consistent: take greedy's
        classes = [("top", n_top, a_hat, 1.0 * (1.0 + difficulty))]
        if n_spare > 0 and omega > 1:
            classes.append(("spare", n_spare, b_hat, 1.0 + difficulty))
        counts, value = _greedy(classes, q)
        detail = {"t": counts.get("top", 0), "counts": counts, "opt_short": 1.0,
                  "top_spread": _spread(counts.get("top", 0), n_top),
                  "spare_spread": _spread(counts.get("spare", 0), max(1, n_spare))}
        detail["distinct"] = sum(1 for m in detail["top_spread"] + detail["spare_spread"] if m)
        detail["duplicated"] = sum(m - 1 for m in detail["top_spread"] + detail["spare_spread"] if m > 1)
        best = Plan(counts.get("top", 0), value / q, detail)
    return best


def slots(pool, q, n_others, a_hat, difficulty, b_hat=0):
    """The q cliques to submit, from plan()'s per-class counts."""
    assert pool and q > 0
    pool = [tuple(c) for c in pool]
    omega = max(len(c) for c in pool)
    top = [c for c in pool if len(c) == omega]
    spare = sorted((c for c in pool if len(c) < omega), key=len, reverse=True)
    p = plan(q, omega, len(top), len(spare), n_others, a_hat, difficulty, b_hat)

    out = []
    for c, m in zip(top, p.detail["top_spread"]):
        out.extend([c] * m)
    for c, m in zip(spare, p.detail["spare_spread"]):
        out.extend([c] * m)
    assert len(out) == q, (len(out), q, p.detail)
    return [list(c) for c in out]


# ------------------------------------------------------------------ self-test

def _monte_carlo_share(a, n_distinct, mine, trials=200000, seed=0):
    import numpy as np
    rng = np.random.default_rng(seed)
    f = rng.binomial(a, 1.0 / n_distinct, size=trials)
    return float(np.mean(mine / (f + mine)))


def selftest():
    for a, nd, mine in ((0, 3, 1), (5, 1, 1), (5, 5, 1), (20, 4, 2), (50, 30, 3)):
        exact = expected_share(a, nd, mine)
        mc = _monte_carlo_share(a, nd, mine)
        assert abs(exact - mc) < 5e-3, (a, nd, mine, exact, mc)
        print("  E[%d/(F+%d)] a=%-3d P=%-3d exact %.4f  mc %.4f"
              % (mine, mine, a, nd, exact, mc))

    # closed form for mine=1: P(1-(1-1/P)^(a+1))/(a+1)
    for a, nd in ((7, 4), (30, 12)):
        closed = nd * (1 - (1 - 1.0 / nd) ** (a + 1)) / (a + 1)
        assert abs(expected_share(a, nd, 1) - closed) < 1e-9

    print("\n  the three cases the rule has to get right:")
    cases = [
        ("omega nearly unique, field crowds it", dict(
            q=8, omega=30, n_top=2, n_spare=400, n_others=55, a_hat=20, difficulty=0.8)),
        ("omega wide open, field spread thin", dict(
            q=8, omega=30, n_top=60, n_spare=400, n_others=55, a_hat=40, difficulty=0.8)),
        ("we alone found omega", dict(
            q=8, omega=30, n_top=5, n_spare=400, n_others=55, a_hat=0, difficulty=0.8)),
        ("no spares to hold back into", dict(
            q=8, omega=30, n_top=2, n_spare=0, n_others=55, a_hat=20, difficulty=0.8)),
    ]
    for name, kw in cases:
        p = plan(**kw)
        print("  %-38s t=%d/%d  value=%.4f" % (name, p.t, kw["q"], p.value))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    raise SystemExit(selftest() if args.selftest else selftest())
