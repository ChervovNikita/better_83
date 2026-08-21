#!/usr/bin/env python3
"""Paired comparison of two bench runs on the same task set.

Covers BOTH halves of the objective: parity (McNemar over parity flips) and the
diversity term (paired sign test over the tasks where the answer actually changed).

The diversity half matters because unpaired means mislead here. A 120-task diversity
mean has a standard error near 0.019, and 70-80% of tasks return the identical clique
in both runs, so those tasks contribute nothing but variance. Comparing means diluted
several real comparisons in this run before the paired test was added.

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
    # --- diversity, paired over the answers that actually differ -----------------
    da = [r for r in shared if ra[r].get("diversity") is not None
          and rb[r].get("diversity") is not None]
    if da:
        diffs = [(rb[u]["diversity"] - ra[u]["diversity"]) for u in da]
        changed = [(u, d) for u, d in zip(da, diffs) if abs(d) > 1e-9]
        print(f"\nDIVERSITY (paired, only the {len(changed)} of {len(da)} tasks whose "
              f"answer changed)")
        if changed:
            better = sum(1 for _, d in changed if d > 0)
            worse = len(changed) - better
            mean_all = sum(diffs) / len(diffs)
            print(f"  B better on {better}, A better on {worse}")
            print(f"  mean diversity change over ALL tasks: {mean_all:+.4f}")
            p_div = mcnemar_p(worse, better)
            print(f"  two-sided sign test p = {p_div:.4f}"
                  + ("   <-- significant at 0.05" if p_div < 0.05 else "   (not significant)"))
        else:
            print("  identical answers on every task — nothing to test")

    rw = [u for u in shared if ra[u].get("reward") is not None
          and rb[u].get("reward") is not None]
    if rw:
        d = [rb[u]["reward"] - ra[u]["reward"] for u in rw]
        m = sum(d) / len(d)
        var = sum((x - m) ** 2 for x in d) / max(1, len(d) - 1)
        se = (var / len(d)) ** 0.5
        print(f"\nREWARD (paired over {len(rw)} tasks)")
        print(f"  mean B - A = {m:+.4f}   SE {se:.4f}   t = {(m/se if se else 0):+.2f}")

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
