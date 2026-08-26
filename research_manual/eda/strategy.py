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


def fill(n_top, n_spare, t, q):
    """Multiplicities per distinct clique for `t` hotkeys on omega.

    Returns (top_counts, spare_counts).  Distinct cliques are exhausted at BOTH
    sizes before anything is duplicated: a repeat earns no diversity at all once
    f = 0, while a distinct clique one vertex shorter still earns a full unit as
    long as opt(s) > D/(1+D).
    """
    top_counts = [0] * max(0, n_top)
    for i in range(t):                          # wraps only once t > n_top
        top_counts[i % max(1, n_top)] += 1

    short = q - t
    spare_counts = []
    if short > 0 and n_spare > 0:
        take = min(short, n_spare)
        spare_counts = [1] * take
        short -= take
    if short > 0:
        unused = [i for i, c in enumerate(top_counts) if c == 0]
        for i in unused[:short]:
            top_counts[i] += 1
        short -= min(short, len(unused))
    if short > 0:                               # nothing distinct left anywhere
        for i in range(short):
            if spare_counts:
                spare_counts[i % len(spare_counts)] += 1
            else:
                top_counts[i % max(1, n_top)] += 1
    return top_counts, spare_counts


def _value(top_counts, spare_counts, q, omega, n_top, n_spare, n_others,
           a_hat, b_hat, difficulty):
    """Expected mean reward per hotkey for one concrete fill."""
    at_omega = a_hat + sum(top_counts)
    total = n_others + q
    if at_omega == 0:
        opt_short = 1.0          # nobody at omega -> M drops, no cost at all
    else:
        opt_short = math.exp(-(at_omega / float(total)) * omega / (omega - 1.0))

    value = 0.0
    for m in top_counts:
        if m:
            value += m * (1.0 + difficulty) + expected_share(a_hat, n_top, m)
    for m in spare_counts:
        if m:
            value += m * opt_short * (1.0 + difficulty) \
                     + expected_share(b_hat, max(1, n_spare), m)
    return value / q


def plan(q, omega, n_top, n_spare, n_others, a_hat, difficulty, b_hat=0):
    """Best t, by expected reward over the fill it actually produces.

    q        our queried hotkeys
    omega    the max clique size our solver found
    n_top    distinct omega-cliques we hold
    n_spare  distinct (omega-1)-cliques we hold
    n_others other answers in the round
    a_hat    predicted number of OTHER answers at omega
    b_hat    predicted number of OTHER answers at omega-1
    """
    assert q > 0 and omega > 0 and n_top > 0
    a_hat = max(0, int(round(a_hat)))
    b_hat = max(0, int(round(b_hat)))
    n_others = max(0, int(n_others))

    best = None
    for t in range(0, q + 1):
        if t < q and (n_spare <= 0 or omega <= 1):
            continue                       # nothing to hold back into
        tc, sc = fill(n_top, n_spare, t, q)
        value = _value(tc, sc, q, omega, n_top, n_spare, n_others, a_hat, b_hat,
                       difficulty)
        detail = {"t": t, "top_counts": tc, "spare_counts": sc,
                  "distinct": sum(1 for m in tc + sc if m),
                  "duplicated": sum(m - 1 for m in tc + sc if m > 1)}
        if best is None or value > best.value:
            best = Plan(t, value, detail)
    assert best is not None
    return best


def slots(pool, q, n_others, a_hat, difficulty, b_hat=0):
    """The cliques to submit: plan(), then the same fill it was scored on."""
    assert pool and q > 0
    pool = [tuple(c) for c in pool]
    omega = max(len(c) for c in pool)
    top = [c for c in pool if len(c) == omega]
    spare = sorted((c for c in pool if len(c) < omega), key=len, reverse=True)
    p = plan(q, omega, len(top), len(spare), n_others, a_hat, difficulty, b_hat)
    tc, sc = fill(len(top), len(spare), p.t, q)

    out = []
    for c, m in zip(top, tc):
        out.extend([c] * m)
    for c, m in zip(spare, sc):
        out.extend([c] * m)
    assert len(out) == q, (len(out), q)
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
