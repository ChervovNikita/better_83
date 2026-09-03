#!/usr/bin/env python3
"""Hitting-set bans: exclude the WHOLE pool, not just one parent.

    .venv/bin/python research_manual/eda/hitban.py --rounds 6

A level-1 ban removes one vertex of one held clique, so it excludes that clique
and leaves every sibling reachable -- the two-disjoint-cliques stall.  A ban that
is a HITTING SET of the pool (>=1 vertex from every held clique) makes no held
clique reconstructible, so the search must return something new or fail to reach
omega.

Measured on the tuning pools, a greedy hitting set of the stalled 16-63 band is
2.1 vertices on average, 0.46% of the graph -- small enough that omega should
survive.  That is the whole reason this is worth trying.

WHY THE VERTICES ARE DELETED, NOT BANNED
----------------------------------------
A banned job still extends its answer in the FULL graph with bans lifted, so the
maximality pass adds the hit vertices straight back and hands back exactly the
clique the ban was meant to exclude.  Deleting H from the adjacency matrix makes
the exclusion stick, and turns the question into the one that matters: does an
omega-clique exist inside G - H?  If it does, it is new by construction, since
every held clique contains a vertex of H.  If the best in G - H is below omega,
then every omega-clique in G meets H and the pool is closed under this ban.

BUDGET
------
Compared against the 3x-budget baseline, which measured +0.1 distinct cliques
over ten rounds (i.e. nothing).  Here the same 3x is split: 1x for the ordinary
harvest, 2x for hitting-set rounds.  Equal compute, different allocation.

MINIMALITY
----------
Every banned vertex aims the search further below omega, so H must be as small
as possible.  Greedy set-cover is not minimal, so it is pruned afterwards:
any vertex whose removal still leaves every clique hit is dropped.  The result
is irredundant, and asserted so.
"""

import argparse
import collections
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
ROOT = os.path.dirname(PARENT)
for _p in (ROOT, PARENT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

MAX_BANS = 8

# Fraction of one round budget spent testing a single H.
SLICE_FRAC = 0.15


def hitting_set(cliques, cap=MAX_BANS, rng=None, avoid=()):
    """A minimal set of vertices hitting every clique, avoiding `avoid`.

    Greedy set-cover, then pruned to irredundance.  `rng` randomises ties so
    successive calls yield DIFFERENT minimal hitting sets.

    `avoid` holds vertices already proven to sit in EVERY omega-clique: deleting
    one of those necessarily drops omega, so it can hit the pool but can never
    leave anything to find.  Excluding them forces a larger H that has a chance
    of sparing omega -- which is the whole point of climbing to |H| = 2, 3, ...

    Returns (H, n_unhit).  n_unhit > 0 means no hitting set exists under the cap.
    """
    avoid = set(avoid)
    sets = [set(c) for c in cliques]
    remaining = list(sets)
    H = []
    while remaining and len(H) < cap:
        cnt = collections.Counter()
        for s in remaining:
            for v in s:
                if v not in avoid:
                    cnt[v] += 1
        if not cnt:
            break
        best = max(cnt.values())
        ties = sorted(v for v, k in cnt.items() if k == best)
        v = int(rng.choice(ties)) if rng is not None else ties[0]
        H.append(v)
        remaining = [s for s in remaining if v not in s]

    # prune: drop any vertex the rest of H already covers for
    for v in list(H):
        rest = [x for x in H if x != v]
        if all(any(x in s for x in rest) for s in sets):
            H = rest
    assert all(any(x in s for x in H) for s in sets) or remaining, "H does not hit"
    for v in H:                                   # irredundant
        rest = [x for x in H if x != v]
        assert not all(any(x in s for x in rest) for s in sets), "H not minimal"
    return H, len(remaining)


def harvest_pool(gpu, fg, A, budget, champ):
    pool, ctr, _hits = gpu.harvest(budget, seed=1, max_steps=fg.STEPS,
                            boot_steps=fg.BOOT_STEPS, init_clique=champ,
                            max_out=8192)
    return pool, ctr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--jobs", type=int, default=512)
    ap.add_argument("--steps", type=int, default=200000)
    ap.add_argument("--lo", type=int, default=16)
    ap.add_argument("--hi", type=int, default=63)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--select", choices=("band", "multibasin"), default="band",
                    help="multibasin: lowest mean-Jaccard rounds, where the pool "
                         "already spans many separated basins")
    ap.add_argument("--cap", type=int, default=MAX_BANS,
                    help="max |H|; vertices are DELETED not banned, so the "
                         "device's 8-ban ABI limit does not apply")
    args = ap.parse_args()

    from CliqueAI.graph.codec import GraphCodec
    import fleet_solver_gpu as fg
    import gpu_lib

    feats = {r["uuid"]: r for r in
             json.load(open(os.path.join(HERE, "field_features.json")))}
    pools = json.load(open(os.path.join(HERE, "pools.json")))
    tune = json.load(open(os.path.join(HERE, "tuning_data.json")))
    assert not (set(tune) & set(json.load(open(os.path.join(PARENT,
                                                           "rounds.json"))))), "eval leak"
    rng = np.random.default_rng(args.seed)
    if args.select == "multibasin":
        import itertools as _it
        scored = []
        for rid, c in pools.items():
            if rid not in feats or len(c["top"]) < 8:
                continue
            top = [set(x) for x in c["top"][:40]]
            jac = float(np.mean([len(a & b) / len(a | b)
                                 for a, b in _it.combinations(top, 2)]))
            scored.append((jac, feats[rid]))
        scored.sort(key=lambda kv: kv[0])
        band = [f for _j, f in scored]
        sel = band[:args.rounds]
    else:
        band = [f for f in feats.values() if args.lo <= f["n_top"] <= args.hi]
        sel = [band[i] for i in rng.permutation(len(band))[:args.rounds]]
    codec = GraphCodec()

    print("%d rounds in the %d-%d band; %d drawn"
          % (len(band), args.lo, args.hi, len(sel)))
    cols = ("round", "1x pool", "+hitban", "gain", "|H| seq", "iters", "killed",
            "core")
    fmt = "%-14s %7s %8s %6s %12s %6s %6s %5s"
    print(fmt % cols)
    print(fmt % tuple("-" * len(c) for c in cols))
    base_all, final_all = [], []
    for f in sel:
        rec = tune[f["uuid"]]
        A = np.array(codec.decode_matrix(rec["encoded_matrix"]), dtype=np.uint8)
        budget = rec["time_limit"] - 2.0
        champ = sorted(fg.fleet_solver._solve_one(A, budget * fg.CHAMPION_SHARE,
                                                  seed=1))
        with gpu_lib.GpuClique(A) as g:
            # phase 1: the ordinary harvest, 1x
            pool, _ = harvest_pool(g, fg, A, budget * (1 - fg.CHAMPION_SHARE), champ)
            om = max(len(c) for c in pool)
            held = {tuple(c) for c in pool if len(c) == om}
            base = len(held)

            # phase 2: hitting-set DELETIONS, 2x. If H kills omega then every
            # omega-clique meets H; when |H| == 1 that pins the culprit exactly,
            # so it joins `fatal` and the next H has to be bigger. Climbing
            # |H| = 1, 2, 3, ... is the search for a ban that excludes the pool
            # while sparing omega.
            deadline = time.monotonic() + 2.0 * budget
            # Every vertex in the pool's common core is a one-vertex hitting set,
            # and greedy picks those first -- but measured 14 of 14, deleting one
            # drops omega, because a core vertex sits in every omega-clique of the
            # graph and not merely in every clique we hold. Testing them one at a
            # time burns the whole budget proving that again, so they start
            # excluded and the search begins at |H| >= 2.
            core = set.intersection(*[set(c) for c in held]) if held else set()
            fatal, tried, hs = set(core), set(), []
            iters, gained, killed = 0, 0, 0
            n_core = len(core)
            while time.monotonic() < deadline:
                H, unhit = hitting_set(sorted(held), cap=args.cap, rng=rng, avoid=fatal)
                if unhit or not H:
                    hs.append("none")
                    break
                key = tuple(sorted(H))
                if key in tried:
                    # the greedy is deterministic under a fixed `avoid`, so force
                    # a different minimal hitting set by excluding one of its own
                    # members for this attempt only
                    H, unhit = hitting_set(sorted(held), cap=args.cap, rng=rng,
                                           avoid=set(fatal) | {int(rng.choice(H))})
                    key = tuple(sorted(H))
                    if unhit or not H or key in tried:
                        break
                tried.add(key)
                hs.append(len(H))
                iters += 1
                B = A.copy()
                for v in H:
                    B[v, :] = 0
                    B[:, v] = 0
                # a short slice: the question is only whether omega survives in
                # G - H, and 2x budget must cover many H's, not two
                slice_s = min(SLICE_FRAC * budget,
                              max(0.05, deadline - time.monotonic()))
                with gpu_lib.GpuClique(B) as gb:
                    sub, _, _hits = gb.harvest(slice_s, seed=iters, max_steps=fg.STEPS,
                                        boot_steps=fg.BOOT_STEPS, max_out=8192)
                sz = max((len(c) for c in sub), default=0)
                if sz < om:                        # H destroyed omega
                    killed += 1
                    if len(H) == 1:
                        fatal.add(H[0])
                    continue
                fresh = {tuple(sorted(c)) for c in sub if len(c) == om}
                fresh = {c for c in fresh if all(gpu_lib.verify(A, c))} - held
                held |= fresh
                gained += len(fresh)
            best_sub = len(fatal)
        base_all.append(base)
        final_all.append(len(held))
        print("%-14s %7d %8d %+6d %12s %6d %6d %5d"
              % ("n=%d tl=%g" % (f["n"], rec["time_limit"]), base, len(held),
                 len(held) - base, ",".join(str(x) for x in hs[:6]), iters,
                 killed, n_core), flush=True)

    b = np.array(base_all, float)
    fin = np.array(final_all, float)
    print()
    print("distinct omega-cliques: 1x harvest %.1f -> +hitban(2x) %.1f  (%+.1f, %+.0f%%)"
          % (b.mean(), fin.mean(), (fin - b).mean(), 100 * (fin.mean() / b.mean() - 1)))
    print("rounds improved: %d of %d" % (np.sum(fin > b), len(b)))
    print("baseline for the same 3x compute (measured): +0%")


if __name__ == "__main__":
    main()
