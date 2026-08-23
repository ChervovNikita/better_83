#!/usr/bin/env python3
"""Decompose our diversity loss into SELF-collision and FIELD-collision.

The scorer pays diversity = (1/holders) / max_delta, where `holders` counts the valid
answers in that round carrying the byte-identical vertex set and max_delta normalises
over every valid answer, not just the best-size ones. So a submission's diversity is
destroyed equally by a sibling repeating it and by a rival independently finding it --
but the two have completely different fixes, and until they are separated it is not
possible to say which one is costing more.

  self-collision   our picker wrapped on a short pool and handed the same clique to
                   two of our own hotkeys. Fix: deeper harvest (supply).
  field-collision  a rival submitted the same clique independently. Fix: a solver whose
                   attractor sits elsewhere while still reaching omega (reach).

This tool reports the measured reward we would recover from each, using counterfactuals
that are bounds rather than predictions, and says which direction each bound errs in.

Input is `fleet_sim --dump-submissions` output. See tools/collide.py for why rebuilding
this from data/sim_rounds.jsonl is wrong.
"""
import argparse
import collections
import json
import sys

import numpy as np


def score_round(sizes, valid, keys, difficulty):
    """Byte-for-byte fleet_sim.score_round, duplicated so this tool stands alone."""
    size = np.asarray(sizes, dtype=float) * np.asarray(valid, dtype=float)
    n = len(size)
    if n == 0 or size.max() <= 0:
        return np.zeros(n)
    rel = size / size.max()
    pr = np.array([(size > s).sum() / n for s in size])
    omega = np.where(size > 0, np.exp(-pr / np.maximum(rel, 1e-12)), 0.0)
    optimality = omega / omega.max() if omega.max() > 0 else omega
    counts = collections.Counter(k for k, v in zip(keys, valid) if v)
    delta = np.array([(1.0 / counts[k]) if v else 0.0 for k, v in zip(keys, valid)])
    diversity = delta / delta.max() if delta.max() > 0 else delta
    return optimality * (1 + difficulty) + diversity


def load(path):
    rounds = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rounds.append(json.loads(line))
    return rounds


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("submissions")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    rounds = load(args.submissions)
    per = []          # per-round dicts
    ours_hold = []    # per-submission holder decomposition
    for r in rounds:
        A = r["answers"]
        sizes = [a["size"] for a in A]
        valid = [a["valid"] for a in A]
        keys = [tuple(a["clique"]) if a["clique"] is not None else ("x", a["who"])
                for a in A]
        d = r["difficulty"]
        base = score_round(sizes, valid, keys, d)

        cnt_all = collections.Counter(k for k, v in zip(keys, valid) if v)
        cnt_our = collections.Counter(k for k, v, a in zip(keys, valid, A)
                                      if v and a["ours"])
        our_i = [i for i, a in enumerate(A) if a["ours"] and a["valid"]]
        if not our_i:
            continue

        for i in our_i:
            k = keys[i]
            ours_hold.append((cnt_all[k], cnt_our[k], cnt_all[k] - cnt_our[k]))

        # counterfactual 1 -- NO SELF-COLLISION. Every repeated clique of ours is
        # replaced by a distinct one that no sibling holds. It keeps whatever field
        # holders the original had, so this is a LOWER bound on the gain from supply:
        # a genuinely new clique would usually have fewer field holders too, not the
        # same number. Size and validity are unchanged, so optimality cannot move.
        keys_ns = list(keys)
        seen = collections.Counter()
        for i in our_i:
            k = keys[i]
            seen[k] += 1
            if seen[k] > 1:
                keys_ns[i] = ("split", i)          # unique, so holders -> 1
        # a split clique keeps the field holders it had: re-add them by making the
        # synthetic key collide with exactly the field copies
        ns = score_round(sizes, valid, keys_ns, d)

        # counterfactual 2 -- NO FIELD COLLISION. Our cliques keep their self-holders
        # but no rival holds them. UPPER bound on the gain from moving the attractor:
        # it assumes we can always find an equally-good clique nobody else reached,
        # which the union-hull oracle says is false on most rounds.
        keys_nf = list(keys)
        for i in our_i:
            keys_nf[i] = ("solo", keys[i])         # ours still collide with each other
        nf = score_round(sizes, valid, keys_nf, d)

        # counterfactual 3 -- BACKFILL. Every hotkey that would have repeated a
        # sibling's clique submits a UNIQUE clique one vertex smaller instead. This is
        # the only counterfactual here that a solver can actually execute today: the
        # harvester already produces sub-omega cliques (mean 4.4/round), they are just
        # never used. It is not a bound, it is a proposal, and it can LOSE: dropping to
        # omega-1 cuts the optimality term, which is multiplied by (1+difficulty),
        # while it only buys back a diversity term capped at 1.
        sizes_bf, keys_bf = list(sizes), list(keys)
        seen2 = collections.Counter()
        n_bf = 0
        for i in our_i:
            kk = keys[i]
            seen2[kk] += 1
            if seen2[kk] > 1:
                sizes_bf[i] = sizes[i] - 1
                keys_bf[i] = ("bf", i)
                n_bf += 1
        bf = score_round(sizes_bf, valid, keys_bf, d)

        per.append(dict(
            base=float(sum(base[i] for i in our_i)),
            nself=float(sum(ns[i] for i in our_i)),
            nfield=float(sum(nf[i] for i in our_i)),
            bf=float(sum(bf[i] for i in our_i)),
            n_bf=n_bf,
            n=len(our_i)))

    if not per:
        raise SystemExit("no valid submissions of ours in the dump")

    b = np.array([p["base"] for p in per])
    s = np.array([p["nself"] for p in per])
    f = np.array([p["nfield"] for p in per])
    k = np.array([p["n"] for p in per])

    h_all, h_our, h_field = map(np.array, zip(*ours_hold))
    print(f"{len(per)} rounds with at least one valid submission of ours, "
          f"{len(ours_hold)} submissions\n")
    print("holders on OUR submitted cliques")
    print("  total holders          mean %.3f  median %.1f" % (h_all.mean(),
                                                               np.median(h_all)))
    print("    of which OUR OWN     mean %.3f  median %.1f   (self-collision)"
          % (h_our.mean(), np.median(h_our)))
    print("    of which the FIELD   mean %.3f  median %.1f"
          % (h_field.mean(), np.median(h_field)))
    print("  submissions with a self-holder   %5.1f%%" % (100 * (h_our > 1).mean()))
    print("  submissions with a field holder  %5.1f%%" % (100 * (h_field > 0).mean()))
    print("  submissions held by us ALONE     %5.1f%%" % (100 * (h_all == 1).mean()))
    print()
    print("fleet reward per round, and what each counterfactual recovers")
    print("  %-28s %9s %9s %9s" % ("", "mean", "median", "per hk"))
    print("  %-28s %9.4f %9.4f %9.4f" % ("as submitted", b.mean(), np.median(b),
                                         b.sum() / k.sum()))
    print("  %-28s %9.4f %9.4f %9.4f" % ("no SELF-collision (lower)", s.mean(),
                                         np.median(s), s.sum() / k.sum()))
    print("  %-28s %9.4f %9.4f %9.4f" % ("no FIELD-collision (upper)", f.mean(),
                                         np.median(f), f.sum() / k.sum()))
    print()
    bfa = np.array([p["bf"] for p in per])
    nbf = np.array([p["n_bf"] for p in per])
    ds, df = s - b, f - b
    print("  gain from SUPPLY   median %+.4f/round  %+.4f/hotkey   wins %.1f%% of rounds"
          % (np.median(ds), ds.sum() / k.sum(), 100 * (ds > 1e-9).mean()))
    print("  gain from REACH    median %+.4f/round  %+.4f/hotkey   wins %.1f%% of rounds"
          % (np.median(df), df.sum() / k.sum(), 100 * (df > 1e-9).mean()))
    print()
    dbf = bfa - b
    aff = nbf > 0
    print("  %-28s %9.4f %9.4f %9.4f" % ("BACKFILL omega-1 (real)", bfa.mean(),
                                         np.median(bfa), bfa.sum() / k.sum()))
    print("  gain from BACKFILL median %+.4f/round  %+.4f/hotkey   wins %.1f%% of "
          "the %d AFFECTED rounds"
          % (np.median(dbf[aff]) if aff.any() else 0.0, dbf.sum() / k.sum(),
             100 * (dbf[aff] > 1e-9).mean() if aff.any() else 0.0, aff.sum()))
    print("    (%d of %d rounds have a repeat to replace; %.1f%% of all submissions)"
          % (aff.sum(), len(per), 100 * nbf.sum() / k.sum()))
    print()
    print("  SUPPLY is a lower bound (a new clique would usually also shed field")
    print("  holders); REACH is an upper bound (it assumes an un-taken clique always")
    print("  exists, which the union-hull oracle refutes on most rounds). The true")
    print("  values are therefore both INSIDE these, and supply is the safer one.")

    if args.json:
        json.dump(dict(rounds=len(per), submissions=len(ours_hold),
                       holders_total=float(h_all.mean()),
                       holders_self=float(h_our.mean()),
                       holders_field=float(h_field.mean()),
                       base=float(b.sum() / k.sum()),
                       no_self=float(s.sum() / k.sum()),
                       no_field=float(f.sum() / k.sum())),
                  open(args.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
