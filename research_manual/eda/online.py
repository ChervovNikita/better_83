#!/usr/bin/env python3
"""Online estimate of how many other answers will reach omega.

    model = FieldModel()
    a_hat = model.predict(n_top, expected_answers_by_coldkey)
    ...                                   # round happens
    model.update(n_top, answers_by_coldkey, omega)

A threshold fitted once on history goes stale: coldkeys register and deregister,
and a new operator may always return omega where the incumbents hold back.  So
the propensity is tracked PER COLDKEY and updated every round.

    a_hat = sum over coldkeys of  n_answers(ck) * p(ck, regime)

`regime` is whether our own solver found few or many distinct omega-cliques,
which is the feature the field's own behaviour keys on (measured: the field
abandons omega below n_top ~ 5 and goes all-in above it).  p(ck, regime) is an
EWMA of the fraction of that coldkey's answers that reached omega, shrunk toward
a global prior so a coldkey seen once does not swing the estimate.

Where the observation comes from in deployment: the validator publishes every
round's answers to wandb, which is where rounds.json comes from.  So the field is
observable with a lag of a round or two -- late enough to be useless for the
round in flight, soon enough to track drift.
"""

import argparse
import collections
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
ROOT = os.path.dirname(PARENT)
for _p in (ROOT, PARENT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

FEATURES = os.path.join(HERE, "field_features.json")

# Regime split, from the measured step in the field's behaviour.
N_TOP_SPLIT = 5

# EWMA half-life in rounds, and the shrinkage weight of the global prior. A new
# coldkey needs PRIOR_WEIGHT rounds of its own before it outvotes the prior.
ALPHA = 0.15
PRIOR_WEIGHT = 4.0


def regime(n_top):
    return 0 if n_top <= N_TOP_SPLIT else 1


class FieldModel(object):
    """Per-coldkey propensity to answer at omega, per regime."""

    def __init__(self, prior=(0.06, 1.00)):
        self.prior = list(prior)          # global fallback per regime
        self.prior_n = [1.0, 1.0]
        self.p = collections.defaultdict(lambda: [None, None])
        self.n = collections.defaultdict(lambda: [0.0, 0.0])
        self.seen = collections.Counter()  # answers per coldkey per round

    def propensity(self, coldkey, reg):
        """Shrunk estimate for one coldkey, falling back to the global prior."""
        own = self.p[coldkey][reg]
        cnt = self.n[coldkey][reg]
        if own is None:
            return self.prior[reg]
        return (cnt * own + PRIOR_WEIGHT * self.prior[reg]) / (cnt + PRIOR_WEIGHT)

    def predict(self, n_top, answers_by_coldkey):
        """Expected number of OTHER answers at omega.

        answers_by_coldkey: {coldkey: how many answers they are expected to give}
        """
        reg = regime(n_top)
        return sum(k * self.propensity(ck, reg)
                   for ck, k in answers_by_coldkey.items())

    def update(self, n_top, sizes_by_coldkey, omega):
        """Fold in one observed round.

        sizes_by_coldkey: {coldkey: [answer sizes]}
        """
        reg = regime(n_top)
        hits = total = 0
        for ck, sizes in sizes_by_coldkey.items():
            if not sizes:
                continue
            frac = sum(1 for s in sizes if s >= omega) / float(len(sizes))
            cur = self.p[ck][reg]
            self.p[ck][reg] = frac if cur is None else (1 - ALPHA) * cur + ALPHA * frac
            self.n[ck][reg] += 1.0
            self.seen[ck] = len(sizes)
            hits += sum(1 for s in sizes if s >= omega)
            total += len(sizes)
        if total:
            obs = hits / float(total)
            self.prior[reg] = ((1 - ALPHA) * self.prior[reg] + ALPHA * obs)
            self.prior_n[reg] += 1.0


# ------------------------------------------------------------ walk-forward

def walk_forward(rows, pools_path=None):
    """Predict each round using ONLY the rounds before it."""
    from strategy import plan

    model = FieldModel()
    static_pred, online_pred, truth = [], [], []
    t_static, t_online, t_true = [], [], []

    for row in rows:
        n_top = row["n_top"]
        by_ck = row["answers_by_coldkey"]
        expected = {ck: len(v) for ck, v in by_ck.items()}

        a_online = model.predict(n_top, expected)
        # static: the step model fitted on all of history, frozen
        a_static = (0.06 if n_top <= N_TOP_SPLIT else 1.00) * row["n_others"]

        online_pred.append(a_online)
        static_pred.append(a_static)
        truth.append(row["a"])

        q = max(1, int(round(40 * (1 - np.exp(
            -max(0.0, np.sqrt(2.5) - row["difficulty"] - 0.5))))))
        kw = dict(q=q, omega=row["omega_ours"], n_top=n_top,
                  n_spare=row["n_spare"], n_others=row["n_others"],
                  difficulty=row["difficulty"])
        t_true.append(plan(a_hat=row["a"], b_hat=row["b"], **kw).t)
        t_static.append(plan(a_hat=a_static,
                             b_hat=row["n_others"] - a_static, **kw).t)
        t_online.append(plan(a_hat=a_online,
                             b_hat=row["n_others"] - a_online, **kw).t)

        model.update(n_top, by_ck, row["omega"])

    truth = np.array(truth, float)
    out = {}
    for name, pred, ts in (("static", static_pred, t_static),
                           ("online", online_pred, t_online)):
        out[name] = {
            "mae": float(np.mean(np.abs(np.array(pred) - truth))),
            "agree": int(sum(1 for x, y in zip(ts, t_true) if x == y)),
        }
    out["n"] = len(rows)
    return out, model


def load_rows(features_path, dump_path):
    """Join the solver features with each round's per-coldkey answers."""
    with open(features_path) as handle:
        feats = json.load(handle)
    with open(dump_path) as handle:
        payload = json.load(handle)
    meta_path = os.path.join(PARENT, "metagraph.json")
    with open(meta_path) as handle:
        meta = json.load(handle)
    import fit_field
    dropped = fit_field.victims(meta, 40)

    rows = []
    for row in feats:
        rec = payload[row["uuid"]]
        by_ck = collections.defaultdict(list)
        for uid, _h, ck, clique in rec["answers"]:
            if clique and uid not in dropped:
                by_ck[ck].append(len(clique))
        row = dict(row)
        row["answers_by_coldkey"] = dict(by_ck)
        rows.append(row)
    rows.sort(key=lambda r: payload[r["uuid"]]["timestamp"])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=FEATURES)
    ap.add_argument("--dump", default=os.path.join(HERE, "tuning_data.json"))
    args = ap.parse_args()

    rows = load_rows(args.features, args.dump)
    res, model = walk_forward(rows)
    print("walk-forward over %d rounds (each predicted from earlier rounds only)"
          % res["n"])
    print("%-8s %8s %12s" % ("model", "MAE a", "t* agree"))
    for name in ("static", "online"):
        print("%-8s %8.2f %9d/%d"
              % (name, res[name]["mae"], res[name]["agree"], res["n"]))
    print()
    print("learned propensities (fraction of a coldkey's answers reaching omega)")
    print("%-12s %6s %8s %8s" % ("coldkey", "seen", "few-top", "many-top"))
    for ck, cnt in sorted(model.seen.items(), key=lambda kv: -kv[1])[:10]:
        lo, hi = model.p[ck]
        print("%-12s %6d %8s %8s"
              % (ck[:10], cnt,
                 "-" if lo is None else "%.2f" % lo,
                 "-" if hi is None else "%.2f" % hi))


if __name__ == "__main__":
    main()
