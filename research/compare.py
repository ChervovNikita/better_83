#!/usr/bin/env python3
"""Paired comparison of two bench runs on the same task set.

Parity rate alone is a blunt instrument: on 500 tasks its standard error is ~1.4
points, so a 1-point "improvement" is noise. But the two runs solved the SAME
graphs, so compare them pairwise — only the tasks where they disagree carry
information. That is McNemar's test, and it resolves differences an unpaired
comparison cannot.

    python3 compare.py runs/champ.json runs/v3.json
"""
import json
import sys
from math import comb


def mcnemar_p(a, b):
    """Two-sided exact binomial p on the discordant pairs (a wins vs b wins)."""
    n = a + b
    if n == 0:
        return 1.0
    k = min(a, b)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def load(path):
    d = json.load(open(path))
    return d, {r["uuid"]: r for r in d["rows"]}


def main(pa, pb):
    da, ra = load(pa)
    db, rb = load(pb)
    shared = sorted(set(ra) & set(rb))
    if not shared:
        print("no shared uuids — different sets?")
        return 1
    name_a = f"{da['variant']}({da.get('seed', 0)})"
    name_b = f"{db['variant']}({db.get('seed', 0)})"
    print(f"A = {name_a}  {pa}")
    print(f"B = {name_b}  {pb}")
    print(f"set = {da['set']} / {db['set']}, {len(shared)} shared tasks\n")

    pa_ok = pb_ok = a_only = b_only = 0
    bigger_a = bigger_b = 0
    size_diff = 0
    hard_rows = []
    for u in shared:
        x, y = ra[u], rb[u]
        ax = x["ok"] and x["delta"] >= 0
        bx = y["ok"] and y["delta"] >= 0
        pa_ok += ax
        pb_ok += bx
        a_only += ax and not bx
        b_only += bx and not ax
        sa = x["ours"] if x["ok"] else 0
        sb = y["ours"] if y["ok"] else 0
        size_diff += sa - sb
        bigger_a += sa > sb
        bigger_b += sb > sa
        if ax != bx:
            hard_rows.append((u, x["n"], x["tl"], x["best"], sa, sb))

    n = len(shared)
    print(f"parity   A {pa_ok/n:7.3%}   B {pb_ok/n:7.3%}   diff {(pb_ok-pa_ok)/n:+.3%}")
    print(f"clique size: B bigger on {bigger_b}, A bigger on {bigger_a}, "
          f"total size delta {-size_diff:+d} vertices in B's favour")
    print(f"\nPAIRED (parity flips only — the tasks that carry information)")
    print(f"  A parity, B not : {a_only}")
    print(f"  B parity, A not : {b_only}")
    p = mcnemar_p(a_only, b_only)
    print(f"  McNemar exact two-sided p = {p:.4f}"
          + ("   <-- significant at 0.05" if p < 0.05 else "   (not significant)"))
    if hard_rows:
        print(f"\n  flipped tasks (uuid, n, tl, best, A, B):")
        for r in hard_rows[:20]:
            print(f"    {r[0][:8]} n={r[1]:<4} tl={r[2]:<5} best={r[3]:<4} A={r[4]:<4} B={r[5]}")
        if len(hard_rows) > 20:
            print(f"    ... {len(hard_rows)-20} more")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))
