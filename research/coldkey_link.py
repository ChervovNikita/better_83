#!/usr/bin/env python3
"""Are the 28 wallets independent operators, or is one operator splitting wallets?

Two fixes over the first attempt:

1. POWER. A per-pair test is hopeless. The base pairwise collision rate is ~0.45%, so
   "never collided in 30 shared rounds" happens with probability 0.9955^30 = 0.87
   under pure independence — no evidence at all. Getting p<0.05 per pair needs ~670
   shared rounds, which no pair has. So aggregate at the GROUP level: wallet A's 75
   UIDs against wallet B's 45 UIDs is 3,375 pairs x ~200 shared rounds each, which is
   hundreds of thousands of pair-rounds and plenty of power.

2. THE RIGHT COMPARISON. Coordination shows up as a collision rate BELOW what
   independence predicts. So compute, for every wallet pair, observed collisions
   against expected-under-independence, and flag the pairs that collide far less than
   chance. Two wallets that systematically avoid each other are one operator.

Also reports WITHIN-wallet rates: if a wallet's own UIDs avoid each other, that is
coordination we can see directly, and it calibrates what a coordinated rate looks like.
"""
import collections
import itertools
import json
import os
import sys

from _common import DATA_DIR

SRC = os.path.join(DATA_DIR, "identities.jsonl")


def main():
    rounds = [json.loads(l) for l in open(SRC) if l.strip()]
    ck_of = {}
    for r in rounds:
        for u, c in zip(r["uids"], r.get("coldkeys") or []):
            if c:
                ck_of[u] = c
    groups = collections.defaultdict(set)
    for u, c in ck_of.items():
        groups[c].add(u)
    # label wallets by size, biggest first
    order = sorted(groups, key=lambda c: -len(groups[c]))
    label = {c: f"W{i+1}({len(groups[c])})" for i, c in enumerate(order)}

    # pair statistics, aggregated by wallet pair
    shared = collections.Counter()
    collide = collections.Counter()
    tot_shared = tot_collide = 0
    for r in rounds:
        present = [(u, tuple(a)) for u, a in zip(r["uids"], r["answers"]) if a]
        for (u1, a1), (u2, a2) in itertools.combinations(present, 2):
            c1, c2 = ck_of.get(u1), ck_of.get(u2)
            if c1 is None or c2 is None:
                continue
            k = (c1, c2) if c1 <= c2 else (c2, c1)
            shared[k] += 1
            tot_shared += 1
            if a1 == a2:
                collide[k] += 1
                tot_collide += 1

    base = tot_collide / max(tot_shared, 1)
    print(f"{len(rounds)} rounds, {len(ck_of)} UIDs, {len(groups)} wallets")
    print(f"OVERALL pairwise collision rate (the independence baseline): {base:.4%}")
    print(f"  {tot_collide} collisions over {tot_shared} pair-rounds\n")

    print("WITHIN each wallet — do its own UIDs avoid each other?")
    print(f"{'wallet':>12} {'pair-rounds':>12} {'collisions':>11} {'rate':>9} {'vs base':>9}")
    for c in order:
        if len(groups[c]) < 2:
            continue
        k = (c, c)
        s, x = shared.get(k, 0), collide.get(k, 0)
        if s < 500:
            continue
        rate = x / s
        print(f"{label[c]:>12} {s:>12} {x:>11} {rate:>8.4%} {rate/base if base else 0:>8.2f}x")

    print("\nBETWEEN wallets — pairs colliding FAR BELOW chance are one operator")
    print(f"{'wallet A':>12} {'wallet B':>12} {'pair-rounds':>12} {'obs':>6} {'exp':>8} "
          f"{'obs/exp':>8}")
    rows = []
    for c1, c2 in itertools.combinations(order, 2):
        k = (c1, c2) if c1 <= c2 else (c2, c1)
        s = shared.get(k, 0)
        if s < 2000:                      # need power
            continue
        x = collide.get(k, 0)
        exp = s * base
        rows.append((x / exp if exp else 1.0, label[c1], label[c2], s, x, exp))
    rows.sort()
    for ratio, a, b, s, x, exp in rows[:12]:
        flag = "  <-- suppressed" if ratio < 0.5 else ""
        print(f"{a:>12} {b:>12} {s:>12} {x:>6} {exp:>8.1f} {ratio:>7.2f}x{flag}")
    print()
    if rows:
        hi = rows[-1]
        print(f"highest ratio (most independent-looking): {hi[1]} vs {hi[2]} "
              f"at {hi[0]:.2f}x")
    print("\nratio ~1.0 = collides exactly as chance predicts -> independent operators")
    print("ratio << 1.0 = systematically avoids each other -> same operator, split wallets")


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
