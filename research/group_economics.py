#!/usr/bin/env python3
"""What does a coordinated group of size N actually earn per UID?

We no longer have to model this: the fetched rounds carry every miner's answer, and
three real operators run coordinated groups of 23, 35, 43, 45 and 75 UIDs. Replay the
validator's scoring per UID and read the scaling curve straight off the field.

Per round the validator computes
    optimality = omega/max(omega),  omega = exp(-pr/rel)
    diversity  = (1/#miners with your exact set) / max over responders
    reward     = optimality*(1+difficulty) + diversity
and weights are per-UID, min-max normalised then rank-amplified — so what matters for
break-even is reward PER UID, and how it degrades as a group adds members.
"""
import collections
import json
import os
import sys

import numpy as np

from _common import DATA_DIR

SRC = os.path.join(DATA_DIR, "identities.jsonl")
DIFF = 0.85          # mean difficulty across the mix


def main():
    rounds = [json.loads(l) for l in open(SRC) if l.strip()]
    ck = {}
    for r in rounds:
        for u, c in zip(r["uids"], r.get("coldkeys") or []):
            if c:
                ck[u] = c
    groups = collections.defaultdict(set)
    for u, c in ck.items():
        groups[c].add(u)
    order = sorted(groups, key=lambda c: -len(groups[c]))
    # W3+W4+W5 proved to be one operator; merge them
    merged = {}
    for i, c in enumerate(order):
        merged[c] = f"W{i+1}"
    op = dict(merged)
    for c in order[2:5]:
        op[c] = "A(W3+W4+W5)"

    per_uid_rw = collections.defaultdict(list)
    per_uid_dv = collections.defaultdict(list)
    for r in rounds:
        ans = [(u, tuple(a)) for u, a in zip(r["uids"], r["answers"]) if a]
        if not ans:
            continue
        sizes = np.array([len(a) for _, a in ans], dtype=float)
        n_resp = len(r["uids"])                      # invalids sit in pr's denominator
        mx = sizes.max()
        if mx <= 0:
            continue
        rel = sizes / mx
        pr = np.array([(sizes > s).sum() / n_resp for s in sizes])
        omega = np.exp(-pr / np.maximum(rel, 1e-9))
        optim = omega / omega.max()
        cnt = collections.Counter(a for _, a in ans)
        delta = np.array([1.0 / cnt[a] for _, a in ans])
        div = delta / delta.max()
        for (u, _), o, d in zip(ans, optim, div):
            per_uid_rw[u].append(o * (1 + DIFF) + d)
            per_uid_dv[u].append(d)

    print(f"{len(rounds)} rounds, {len(per_uid_rw)} UIDs with answers\n")
    print(f"{'operator':>14} {'UIDs':>5} {'mean reward/UID':>17} {'mean diversity':>15} "
          f"{'group total':>12}")
    rows = []
    for c in order:
        us = [u for u in groups[c] if len(per_uid_rw.get(u, [])) >= 50]
        if not us:
            continue
        rw = float(np.mean([np.mean(per_uid_rw[u]) for u in us]))
        dv = float(np.mean([np.mean(per_uid_dv[u]) for u in us]))
        rows.append((op[c], len(us), rw, dv))
    agg = collections.defaultdict(lambda: [0, [], []])
    for name, n, rw, dv in rows:
        agg[name][0] += n
        agg[name][1].append(rw)
        agg[name][2].append(dv)
    for name, (n, rws, dvs) in sorted(agg.items(), key=lambda kv: -kv[1][0]):
        rw, dv = float(np.mean(rws)), float(np.mean(dvs))
        print(f"{name:>14} {n:>5} {rw:>17.4f} {dv:>15.4f} {rw*n:>12.1f}")

    print("\nSCALING — reward per UID against coordinated group size:")
    pts = sorted([(n, rw) for _, n, rw, _ in rows])
    for n, rw in pts:
        print(f"   group of {n:>3} UIDs -> {rw:.4f} reward/UID")
    if len(pts) >= 3:
        x = np.array([p[0] for p in pts], dtype=float)
        y = np.array([p[1] for p in pts])
        a, b = np.polyfit(np.log(x), y, 1)
        print(f"\n   fit: reward/UID = {b:.4f} + {a:.4f}*ln(N)")
        for n in (1, 5, 10, 20, 50, 100):
            print(f"      N={n:>4}: predicted {b + a*np.log(n):.4f}")


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
