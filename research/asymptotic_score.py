#!/usr/bin/env python3
"""Where our score CONVERGES if we mine like this forever — from a short window.

Simulating 2 h and reading fleet_sim's leaderboard does not answer this. In 216
rounds each hotkey is queried ~47 times, and set_weights is a rank amplifier, so
that leaderboard is a noise draw (fleet_sim says so itself). The infinite-horizon
question is different, and much cheaper to answer.

The EMA in update_scores converges to the MEAN of its stream. So "run like this
forever" = replace every identity's EMA with its converged mean, then apply
set_weights once. What is left is only the standard error on those means.

That splits the cost in two, and only one half is expensive:

  * the FIELD's converged means need no solving at all — their answers are already
    in the log, so they are estimated over the whole dataset (~550 samples each);
  * OUR converged mean needs solved rounds, but it is estimated PAIRED against the
    field on the same round, which removes 57% of the variance (measured). 216
    rounds resolve our deficit at ~9 sigma.

Our presence also shifts the field: collisions cost the miners we collide with, and
displacement removes some entirely. That shift is measured on the solved rounds and
applied to the long-window field means, so it is not assumed away.

    python3 asymptotic_score.py --sizes 1 5 10 20 40
"""
import argparse
import collections
import json
import os
import sys

import numpy as np

import fleet_sim as F
from _common import DATA_DIR


def converged(stream):
    """The limit of the debiased EMA: the mean of the stream."""
    return float(np.mean(stream)) if stream else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[1, 5, 10, 20, 40])
    ap.add_argument("--dataset", default=os.path.join(DATA_DIR, "sim_rounds.jsonl"))
    ap.add_argument("--metagraph", default=os.path.join(DATA_DIR, "metagraph.json"))
    ap.add_argument("--cache", default=os.path.join(DATA_DIR, "sim_ts.jsonl"))
    ap.add_argument("--boot", type=int, default=400, help="bootstrap resamples")
    ap.add_argument("--seed", type=int, default=8383)
    args = ap.parse_args()

    rounds = [json.loads(l) for l in open(args.dataset) if l.strip()]
    rounds = [r for r in rounds if r.get("answers") and r.get("timestamp")]
    rounds.sort(key=lambda r: r["timestamp"])
    meta = json.load(open(args.metagraph))
    cache = {}
    with open(args.cache) as f:
        for line in f:
            r = json.loads(line)
            cache[r["uuid"]] = r
    solved = [r for r in rounds if r["uuid"] in cache]
    if not solved:
        raise SystemExit("no solved rounds in the cache; run the solve stage first")
    print(f"{len(rounds)} logged rounds | {len(solved)} solved "
          f"({len(solved)/108:.1f} h of subnet time)", file=sys.stderr)

    validity = {r["uuid"]: F.validate_cliques(r, cache[r["uuid"]]["cliques"])
                for r in solved}

    # ---- the field's converged means, from the WHOLE log. No solving needed.
    long_field = collections.defaultdict(list)
    for r in rounds:
        for a in r["answers"]:
            long_field[a["hk"]].append(a["reward"])
    n_long = {hk: len(v) for hk, v in long_field.items()}
    base = {hk: converged(v) for hk, v in long_field.items()}
    print(f"  field means over the full log: {len(base)} identities, "
          f"median {np.median(list(n_long.values())):.0f} samples each", file=sys.stderr)

    # ---- how much does our presence move the field? measured, not assumed
    st0, *_ = F.replay(solved, cache, meta, 0, args.seed, validity, sample="real")
    short0 = {hk: converged(v) for hk, v in st0.items()}

    print(f"\n{'N':>4} {'OUR MEDIAN':>10} {'95% CI':>17} {'field median':>13} "
          f"{'rank':>10} {'share':>9} {'vs median':>10} {'zeros':>7} {'beat':>4}")
    out = []
    rng = np.random.default_rng(args.seed)
    for N in args.sizes:
        st, vu, oid, alive, ev, hku = F.replay(
            solved, cache, meta, N, args.seed, validity, sample="real")
        # PER-HOTKEY converged scores. The user's target is our MEDIAN against the
        # field's median, and at N=40 our hotkeys are not interchangeable: a pool
        # smaller than N leaves the surplus queried, silent and scoring zero. A
        # pooled mean hides exactly that -- the median is the honest summary of a
        # fleet whose members are served unequally.
        per_hotkey = [converged(st[o]) for o in oid]
        ours_stream = [x for o in oid for x in st[o]]
        if not ours_stream:
            print(f"{N:>4}   no fleet answers in this window (warmup?)")
            continue
        ours = float(np.median(per_hotkey))      # OUR MEDIAN = the target metric
        ours_mean = converged(ours_stream)
        n_zero = sum(1 for v in per_hotkey if v <= 0)

        # shift = what our presence did to each surviving field identity, on the
        # same rounds. Applied to the long-window mean so the field keeps its
        # precision without pretending we were absent.
        shifts = [converged(st[hk]) - short0[hk]
                  for hk in st if not str(hk).startswith("OURS-")
                  and hk in short0 and st[hk] and short0[hk]]
        shift = float(np.mean(shifts)) if shifts else 0.0

        vec, names = [], []
        for hk in alive:
            if str(hk).startswith("OURS-"):
                vec.append(per_hotkey[oid.index(hk)]); names.append("OURS")
            elif hk in base:
                vec.append(base[hk] + shift); names.append(hk)
        vec = np.array(vec)
        ours_idx = np.array([i for i, n in enumerate(names) if n == "OURS"])
        w, gamma, half = F.set_weights(vec)
        share = float(w[ours_idx].sum())
        order = np.argsort(-vec)
        rank = {int(j): i + 1 for i, j in enumerate(order)}
        ranks = sorted(rank[int(i)] for i in ours_idx)
        fmed = float(np.median([v for v, n in zip(vec, names) if n != "OURS"]))

        # bootstrap over ROUNDS: the only real sampling unit
        per_round = collections.defaultdict(list)
        for o in oid:
            for x in st[o]:
                per_round[o].append(x)
        # CI for the MEDIAN ACROSS HOTKEYS, which is the reported statistic. An
        # earlier version bootstrapped the pooled answer stream and reported its
        # MEAN interval next to a median point estimate -- the two are different
        # quantities, and the interval did not even bracket the point.
        ph = np.array(per_hotkey)
        bs = [float(np.median(ph[rng.integers(0, len(ph), len(ph))]))
              for _ in range(args.boot)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        beat = "YES" if ours > fmed else "no"
        print(f"{N:>4} {ours:>10.4f} [{lo:>7.4f},{hi:>7.4f}] {fmed:>13.4f} "
              f"{min(ranks):>4}-{max(ranks):<5} {share:>8.3%} {ours-fmed:>+10.4f} "
              f"{n_zero:>3}/{N:<3} {beat:>4}")
        out.append(dict(N=N, our_median=ours, our_mean=ours_mean,
                        ci=[float(lo), float(hi)], field_median=fmed,
                        beats_field_median=bool(ours > fmed), n_zero_hotkeys=n_zero,
                        ranks=ranks, share=share, shift=shift, n_alive=len(vec),
                        n_our_samples=len(ours_stream)))
    dest = os.path.join(DATA_DIR, "asymptotic_results.json")
    json.dump(out, open(dest, "w"), indent=1)
    print(f"\nconverged means, not a short-window EMA: this is where the score goes,")
    print(f"not where it is after {len(solved)} rounds. -> {dest}")


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.exit(main())
