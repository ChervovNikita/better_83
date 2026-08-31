"""Merges the sharded bound run and reports margins with measured coverage."""

import glob
import json
import math
import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))


def pooled(rows):
    """Returns A's reward per answer minus B's, over the whole split."""
    qa = sum(r[0] for r in rows)
    qb = sum(r[1] for r in rows)
    return (sum(r[0] * r[2] for r in rows) / qa
            - sum(r[1] * r[3] for r in rows) / qb)


def sign_test(rows):
    """Two-sided sign test on the per-round margin, over rounds that differ."""
    up = sum(1 for r in rows if r[2] - r[3] > 1e-12)
    down = sum(1 for r in rows if r[3] - r[2] > 1e-12)
    n = up + down
    if not n:
        return up, down, 1.0
    tail = sum(math.comb(n, k) for k in range(min(up, down) + 1)) / 2.0 ** n
    return up, down, min(1.0, 2 * tail)


def bootstrap(rows, draws=4000, seed=0):
    """Returns the 95% interval and P(margin > 0) by resampling rounds."""
    rng = random.Random(seed)
    out = []
    for _ in range(draws):
        sample = [rows[rng.randrange(len(rows))] for _ in range(len(rows))]
        out.append(pooled(sample))
    out.sort()
    return (out[int(0.025 * draws)], out[int(0.975 * draws)],
            sum(1 for v in out if v > 0) / float(draws))


def main():
    """Prints the merged table."""
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_HERE, "out")
    by_split = {}
    for path in sorted(glob.glob(os.path.join(out_dir, "bound_*.json"))):
        data = json.load(open(path))
        entry = by_split.setdefault(data["g"], {"o": data["o"], "rows": [],
                                                "cap": data["cap"], "shards": 0})
        entry["rows"] += data["rows"]
        entry["shards"] += 1
    assert by_split, out_dir

    print("A vs the HYBRID BOUND: exact partial where affordable, full-sight")
    print("otherwise. Full sight dominates partial information, so this is a")
    print("LOWER bound on A's margin against a real partial-information rival.")
    print()
    print("%5s %5s %7s %8s | %10s %-20s %-8s | %s"
          % ("A", "B", "shards", "rounds", "pooled", "95% CI", "P(>0)",
             "exact-partial"))
    for g in sorted(by_split):
        e = by_split[g]
        rows = e["rows"]
        exact = sum(1 for r in rows if r[4] == "partial")
        full = sum(1 for r in rows if r[4] == "full")
        none = sum(1 for r in rows if r[4] == "none")
        lo, hi, p = bootstrap(rows)
        print("%5d %5d %7d %8d | %+10.5f [%+.5f,%+.5f] %-8.2f | %d (%.0f%%), "
              "full %d, empty %d"
              % (g, e["o"], e["shards"], len(rows), pooled(rows), lo, hi, p,
                 exact, 100.0 * exact / len(rows), full, none))
    print()
    print("sign test on the per-round margin")
    for g in sorted(by_split):
        up, down, p = sign_test(by_split[g]["rows"])
        print("  A=%3d  A wins %4d  B wins %4d  p=%.3g" % (g, up, down, p))
    incomplete = [g for g, e in by_split.items() if e["shards"] != 12]
    if incomplete:
        print()
        print("PARTIAL: splits still running %s" % sorted(incomplete))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
