#!/usr/bin/env python3
"""Who is actually competing? Group the field's UIDs into operators.

Two independent layers, because either one alone can be fooled:

  1. COLDKEY. Every logged round carries `miner_coldkeys`. A coldkey is the on-chain
     wallet, so UIDs sharing one are provably the same operator. This is proof, not
     inference — but an operator who splits wallets defeats it.

  2. COLLISION STRUCTURE. Independent solvers on these graphs collide at a measurable
     base rate (our own solver hits the field's answer on ~65% of rounds). Coordinated
     miners collide at ~zero, because avoiding each other is the whole point. So a
     pair of UIDs that share hundreds of rounds and NEVER return the same vertex set
     is coordinated with overwhelming likelihood — the null hypothesis of independence
     assigns that essentially no probability.

The layers cross-check: coldkey groups should sit INSIDE collision clusters, and any
cluster spanning several coldkeys is an operator deliberately splitting wallets.

    python3 identities.py --fetch 3000      # pull rounds WITH per-UID answers
    python3 identities.py --analyse         # cluster what has been pulled

Needs WANDB_API_KEY (or ~/.netrc). The corpus harvested for solver work used
--keep-answers=False, so it cannot be reused here: it records how many miners took a
clique, not which ones.
"""
import argparse
import collections
import itertools
import json
import os
import sys

from _common import DATA_DIR, KEYS, PROJECT, check_wandb_version, discover_runs

OUT = os.path.join(DATA_DIR, "identities.jsonl")


def fetch(limit, versions):
    import wandb
    check_wandb_version()
    api = wandb.Api()
    runs = discover_runs(api, versions)
    n = 0
    with open(OUT, "w") as f:
        for run in runs:
            for row in run.scan_history(keys=KEYS, page_size=200):
                uids = row.get("miner_uids") or []
                cks = row.get("miner_coldkeys") or []
                ans = row.get("miner_ans") or []
                opt = row.get("miner_optimality") or []
                if not uids:
                    continue
                f.write(json.dumps({
                    "uuid": row.get("uuid"),
                    "uids": list(uids),
                    "coldkeys": list(cks),
                    # canonical vertex sets, empty for invalid answers
                    "answers": [sorted(a) if (o or 0) > 0 else None
                                for a, o in itertools.zip_longest(ans, opt)],
                }) + "\n")
                n += 1
                if n % 200 == 0:
                    print(f"  {n} rounds", file=sys.stderr, flush=True)
                if limit and n >= limit:
                    return n
    return n


def analyse():
    rounds = [json.loads(l) for l in open(OUT) if l.strip()]
    print(f"{len(rounds)} rounds with per-UID answers\n")

    # ---- layer 1: coldkeys ------------------------------------------------
    ck_of = {}
    seen_rounds = collections.Counter()
    for r in rounds:
        for u, c in zip(r["uids"], r.get("coldkeys") or []):
            if c:
                ck_of[u] = c
            seen_rounds[u] += 1
    by_ck = collections.defaultdict(set)
    for u, c in ck_of.items():
        by_ck[c].add(u)
    groups = sorted(by_ck.values(), key=len, reverse=True)
    print(f"LAYER 1 — coldkeys: {len(ck_of)} UIDs across {len(by_ck)} distinct wallets")
    for g in groups[:10]:
        if len(g) > 1:
            print(f"   {len(g):>3} UIDs share one coldkey: {sorted(g)[:12]}")
    solo = sum(1 for g in groups if len(g) == 1)
    print(f"   {solo} wallets hold exactly one UID\n")

    # ---- layer 2: never-collide clusters ----------------------------------
    # count, for each ordered pair, shared rounds and identical answers
    shared = collections.Counter()
    collide = collections.Counter()
    for r in rounds:
        present = [(u, tuple(a)) for u, a in zip(r["uids"], r["answers"]) if a]
        for (u1, a1), (u2, a2) in itertools.combinations(present, 2):
            k = (u1, u2) if u1 < u2 else (u2, u1)
            shared[k] += 1
            if a1 == a2:
                collide[k] += 1

    MIN_SHARED = 30
    pairs = [(k, shared[k], collide[k]) for k in shared if shared[k] >= MIN_SHARED]
    if not pairs:
        print("LAYER 2 — not enough shared rounds per pair; fetch more")
        return
    rates = [c / s for _, s, c in pairs]
    import statistics
    base = statistics.median(rates)
    print(f"LAYER 2 — collision structure over {len(pairs)} UID pairs "
          f"(>= {MIN_SHARED} shared rounds)")
    print(f"   median pairwise collision rate: {base:.3%}")
    never = [(k, s) for k, s, c in pairs if c == 0]
    print(f"   pairs that NEVER collide despite >= {MIN_SHARED} shared rounds: "
          f"{len(never)} of {len(pairs)}")

    # union-find over never-colliding pairs that share many rounds
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for (u1, u2), s in never:
        if s >= MIN_SHARED:
            union(u1, u2)
    clusters = collections.defaultdict(set)
    for u in parent:
        clusters[find(u)].add(u)
    cl = sorted(clusters.values(), key=len, reverse=True)
    print(f"\n   never-collide clusters: {len(cl)}")
    for g in cl[:8]:
        cks = {ck_of.get(u, "?") for u in g}
        print(f"      {len(g):>3} UIDs, {len(cks)} distinct coldkey(s)")
    print("\n   A cluster spanning several coldkeys is one operator splitting wallets.")
    print("   A cluster of one coldkey is simply that wallet's UIDs.")


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", type=int, default=0, help="rounds to pull WITH answers")
    ap.add_argument("--versions", nargs="+", default=["0.0.17"])
    ap.add_argument("--analyse", action="store_true")
    a = ap.parse_args()
    if a.fetch:
        print(f"fetched {fetch(a.fetch, a.versions)} rounds -> {OUT}", file=sys.stderr)
    if a.analyse or not a.fetch:
        analyse()
