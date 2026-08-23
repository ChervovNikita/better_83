#!/usr/bin/env python3
"""Should we submit omega-1 even when we HAVE a distinct omega clique to send?

Backfill only fires when the max-size pool is shorter than the queried count -- a third
of rounds. But tools/coverage.py measured that 67.5% of our pool is already held by the
field even when the pool is long (median 100%), so most of our omega submissions are
earning a split diversity term anyway. The trade backfill exploits may not be special to
short pools:

    a CONTESTED omega   full optimality, diversity 1/holders  (holders ~2.17 measured)
    a unique omega-1    reduced optimality, diversity 1.0

This sweeps j = how many of our submissions are converted from omega to a unique
omega-1, applied to ALL rounds rather than only starved ones, and scores each j with the
real scorer. j=0 is the exact control.

It can lose, and the prior says it should: optimality is multiplied by (1 + difficulty)
while diversity is capped at 1, so a needless vertex drop costs roughly twice what
uniqueness repays. tools/holders.py already measured the cliff at omega-2 (87.2% of
affected rounds -> 0.9%). This asks whether omega-1 has any headroom left once the
duplicate case is removed.

UPPER BOUND, not a prediction: every converted clique is assumed unique. Field answers
below max size are median 0.0% per round so that is close to true, but our own converted
submissions must also differ from each other, which needs j distinct omega-1 cliques.
The harvest supplies ~27 per round, so j <= 8 is safe.
"""
import argparse
import collections
import json
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from holders import score_round, load     # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("submissions")
    ap.add_argument("--max-j", type=int, default=8)
    args = ap.parse_args()

    rounds = load(args.submissions)
    J = list(range(0, args.max_j + 1))
    tot = {j: [] for j in J}
    held = []          # per round, fraction of our submissions sharing with the field
    for r in rounds:
        A = r["answers"]
        sizes = [a["size"] for a in A]
        valid = [a["valid"] for a in A]
        keys = [tuple(a["clique"]) if a["clique"] is not None else ("x", a["who"])
                for a in A]
        d = r["difficulty"]
        our_i = [i for i, a in enumerate(A) if a["ours"] and a["valid"]]
        if not our_i:
            continue
        cnt_f = collections.Counter(k for k, a in zip(keys, A)
                                    if a["valid"] and not a["ours"])
        held.append(np.mean([1.0 if cnt_f[keys[i]] else 0.0 for i in our_i]))
        for j in J:
            s2, k2 = list(sizes), list(keys)
            # convert the LAST j of our submissions; which ones is arbitrary because
            # no feature predicts contestedness (predict.py: AUC 0.49-0.54)
            for t, i in enumerate(reversed(our_i)):
                if t >= j:
                    break
                s2[i] = max(1, sizes[i] - 1)
                k2[i] = ("mix", i)
            sc = score_round(s2, valid, k2, d)
            tot[j].append(float(sum(sc[i] for i in our_i)))

    n = len(tot[0])
    print("%d rounds with valid submissions of ours" % n)
    print("  our submissions sharing a clique with the field: %.1f%% (mean of per-round)"
          % (100 * np.mean(held)))
    print()
    print("%-4s %12s %12s %14s %12s" % ("j", "mean", "median", "vs j=0 median",
                                        "wins%"))
    base = np.array(tot[0])
    for j in J:
        v = np.array(tot[j])
        dl = v - base
        print("%-4d %12.4f %12.4f %+14.4f %11.1f%%"
              % (j, v.mean(), np.median(v), np.median(dl),
                 100 * (dl > 1e-9).mean()))
    print()
    best = max(J, key=lambda j: np.median(np.array(tot[j]) - base))
    print("  best j = %d" % best)
    if best == 0:
        print("  No conversion helps. Submitting omega-1 in place of a CONTESTED omega")
        print("  loses, so backfill's gain really is specific to the case where the")
        print("  alternative is repeating a sibling's clique.")
    else:
        dl = np.array(tot[best]) - base
        better = int((dl > 1e-9).sum())
        worse = int((dl < -1e-9).sum())
        import math
        ch = better + worse
        p = 1.0 if ch == 0 else min(1.0, 2.0 * sum(
            math.comb(ch, i) for i in range(0, min(better, worse) + 1)) / 2 ** ch)
        print("  paired j=0 -> j=%d   median %+.4f   %d better / %d worse   sign p = %.4g"
              % (best, np.median(dl), better, worse, p))
    return 0


if __name__ == "__main__":
    sys.exit(main())
