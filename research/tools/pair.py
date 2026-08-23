#!/usr/bin/env python3
"""Paired comparison of two picker arms over the SAME rounds and the SAME pools.

The protocol this project settled on after thirteen retractions, applied here:

  * paired within round -- never a difference of two means over different round sets
  * MEDIAN delta, not mean; one starved round with a huge swing dominates the mean
  * sign test over CHANGED rounds only; rounds where both arms submit the identical
    answers carry no information and inflate n
  * the round sets are INTERSECTED and the count is printed, because two dumps that
    filter rounds differently silently compare different things -- that error cost a
    whole 4-arm block once

Both dumps must come from runs sharing one --cache, so the solver's nondeterminism is
held fixed and the only thing varying is the picker. If they do not, this is measuring
the solver, not the picker, and the header warning says so.

    python3 tools/pair.py control.jsonl treated.jsonl
"""
import argparse
import collections
import json
import math
import sys

import numpy as np


def load(path):
    out = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            ours = [a for a in r["answers"] if a["ours"]]
            out[r["uuid"]] = {
                "reward": sum(a["reward"] for a in ours),
                "n": len(ours),
                "valid": sum(a["valid"] for a in ours),
                "sizes": [a["size"] for a in ours],
                "keys": sorted(tuple(a["clique"]) if a["clique"] else () for a in ours),
                "difficulty": r["difficulty"],
            }
    return out


def sign_test(better, changed):
    """Two-sided exact binomial at p=0.5."""
    if changed == 0:
        return 1.0
    c = math.comb
    tail = sum(c(changed, i) for i in range(0, min(better, changed - better) + 1))
    return min(1.0, 2.0 * tail / (2 ** changed))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("control")
    ap.add_argument("treated")
    ap.add_argument("--label-a", default=None)
    ap.add_argument("--label-b", default=None)
    args = ap.parse_args()

    A, B = load(args.control), load(args.treated)
    la = args.label_a or args.control.split("/")[-1]
    lb = args.label_b or args.treated.split("/")[-1]
    common = sorted(set(A) & set(B))
    print("%s  ->  %s" % (la, lb))
    print("  rounds: %d in control, %d in treated, %d IN COMMON"
          % (len(A), len(B), len(common)))
    if not common:
        raise SystemExit("no rounds in common -- the two runs filtered differently")
    if len(common) < 0.9 * max(len(A), len(B)):
        print("  WARNING: the round sets differ by more than 10%. Two dumps that filter"
              "\n           differently compare different things; check both runs used"
              "\n           the same --rounds and --dataset.", file=sys.stderr)

    d, changed_idx = [], []
    for u in common:
        a, b = A[u], B[u]
        d.append(b["reward"] - a["reward"])
        if a["keys"] != b["keys"]:
            changed_idx.append(len(d) - 1)
    d = np.array(d)
    ch = np.array(changed_idx, dtype=int)

    print()
    print("  fleet reward per round")
    print("    %-10s mean %8.4f   median %8.4f" % (la, np.mean([A[u]["reward"] for u in common]),
                                                   np.median([A[u]["reward"] for u in common])))
    print("    %-10s mean %8.4f   median %8.4f" % (lb, np.mean([B[u]["reward"] for u in common]),
                                                   np.median([B[u]["reward"] for u in common])))
    print()
    print("  paired delta, ALL %d rounds        mean %+8.4f   median %+8.4f"
          % (len(d), d.mean(), np.median(d)))
    if len(ch):
        dc = d[ch]
        better = int((dc > 1e-9).sum())
        worse = int((dc < -1e-9).sum())
        p = sign_test(better, better + worse)
        print("  paired delta, %d CHANGED rounds     mean %+8.4f   median %+8.4f"
              % (len(ch), dc.mean(), np.median(dc)))
        print("  sign test over changed            %d better / %d worse   p = %.4f"
              % (better, worse, p))
        print("  changed rounds are %.1f%% of the common set" % (100 * len(ch) / len(d)))
    else:
        print("  NO ROUND CHANGED -- the two arms are byte-identical. If that is not"
              "\n  expected, the picker was not actually swapped.")

    # per-hotkey view, because the fleet total mixes in how many were queried
    na = sum(A[u]["n"] for u in common)
    nb = sum(B[u]["n"] for u in common)
    print()
    print("  per submitted hotkey              %-10s %.4f   %-10s %.4f   delta %+.4f"
          % (la, sum(A[u]["reward"] for u in common) / max(na, 1),
             lb, sum(B[u]["reward"] for u in common) / max(nb, 1),
             sum(B[u]["reward"] for u in common) / max(nb, 1)
             - sum(A[u]["reward"] for u in common) / max(na, 1)))
    print("  submissions                       %-10s %d   %-10s %d" % (la, na, lb, nb))
    iv_a = na - sum(A[u]["valid"] for u in common)
    iv_b = nb - sum(B[u]["valid"] for u in common)
    print("  INVALID submissions               %-10s %d   %-10s %d" % (la, iv_a, lb, iv_b))
    if iv_b > iv_a:
        print("  WARNING: the treated arm submits MORE invalid answers. A backfilled"
              "\n           clique that is not maximal scores a hard zero.", file=sys.stderr)

    # where the gain sits: by whether the control arm had to repeat
    rep = []
    for u in common:
        k = A[u]["keys"]
        rep.append(len(k) - len(set(k)))
    rep = np.array(rep)
    aff = rep > 0
    if aff.any():
        print()
        print("  by whether the CONTROL arm had to repeat a clique:")
        print("    control repeated   %4d rounds   median delta %+8.4f   mean %+8.4f"
              % (aff.sum(), np.median(d[aff]), d[aff].mean()))
        print("    control did not    %4d rounds   median delta %+8.4f   mean %+8.4f"
              % ((~aff).sum(), np.median(d[~aff]), d[~aff].mean()))
        print("    (a targeted fix shows ~0 on the second row; if it moves there too,"
              "\n     the arms differ for some reason other than the repeat)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
