"""Prints the pooled table over every split written by table.py."""

import json
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
CELLS = ("greedy_full_pooled", "greedy_partial",
         "maximin_full_pooled", "maximin_partial")


def pooled(rows):
    """Returns A's and B's reward per answer over the whole sweep."""
    qa = sum(r[0] for r in rows)
    qb = sum(r[1] for r in rows)
    a = sum(r[0] * r[2] for r in rows) / qa
    b = sum(r[1] * r[3] for r in rows) / qb
    return a, b


def sign_test(rows):
    """Two-sided sign test on the per-round margin, over rounds that differ."""
    up = sum(1 for r in rows if r[2] - r[3] > 1e-12)
    down = sum(1 for r in rows if r[3] - r[2] > 1e-12)
    n = up + down
    if not n:
        return up, down, 1.0
    tail = sum(math.comb(n, k) for k in range(min(up, down) + 1)) / 2.0 ** n
    return up, down, min(1.0, 2 * tail)


def main():
    """Loads every split file and prints the pooled and per-round tables."""
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_HERE, "out")
    files = sorted(f for f in os.listdir(out) if f.startswith("split_"))
    assert files, out
    print("pooled reward per answer, A - B.  A commits first, B replies.")
    print("%5s %5s | %-21s | %-21s" % ("A", "B", "B = full (sees board)",
                                       "B = partial (multiset)"))
    print("%5s %5s | %10s %10s | %10s %10s"
          % ("", "", "greedy", "maximin", "greedy", "maximin"))
    table = []
    for name in files:
        data = json.load(open(os.path.join(out, name)))
        cells = {c: pooled(data["rows"][c]) for c in CELLS}
        table.append((data["g"], data["o"], data, cells))
    for g, o, _data, cells in sorted(table):
        print("%5d %5d | %+10.5f %+10.5f | %+10.5f %+10.5f"
              % (g, o,
                 cells["greedy_full_pooled"][0] - cells["greedy_full_pooled"][1],
                 cells["maximin_full_pooled"][0] - cells["maximin_full_pooled"][1],
                 cells["greedy_partial"][0] - cells["greedy_partial"][1],
                 cells["maximin_partial"][0] - cells["maximin_partial"][1]))
    print()
    print("sign test on the per-round margin (A over B), maximin only")
    print("%5s %5s | %-25s | %-25s" % ("A", "B", "B = full", "B = partial"))
    for g, o, data, _cells in sorted(table):
        line = "%5d %5d |" % (g, o)
        for cell in ("maximin_full_pooled", "maximin_partial"):
            up, down, p = sign_test(data["rows"][cell])
            line += "  A %4d  B %4d  p=%-7.2g |" % (up, down, p)
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
