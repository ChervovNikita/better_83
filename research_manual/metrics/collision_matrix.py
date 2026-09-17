#!/usr/bin/env python3
"""Pairwise clique-collision matrix between entities.

For two entities holding A and B distinct omega-cliques out of S available in a
round, independent picking predicts E[|A n B|] = |A||B|/S. The reported ratio is
observed/expected pooled over rounds:

    ~1.00  picking independently of each other
    < 1    avoiding each other -- they partition the clique space
    > 1    drawn to the same cliques

The diagonal is SELF-collision: an entity's own answers landing on a clique it
already holds, as a fraction of what independent picking would give. Coldkeys
belonging to one operator are merged, so TOP4 is a single row and column and its
four keys cannot flatter each other in the off-diagonal.

Saturation confounds the ratio: when one entity already holds most of the
supply, everyone else is forced onto its cliques and every ratio is pulled
toward 1 regardless of intent. --room F drops rounds where the largest entity
holds >= F of the supply, leaving only rounds where partitioning was possible.

Supply matters as much as saturation. Without --pool, S is the number of
distinct cliques the field happened to answer, which understates the true
supply and drags every ratio below 1 -- an independent miner then reads as an
avoider. --pool passes the harvested omega-clique pool, which is the supply the
field actually chose from.

    python research_manual/metrics/collision_matrix.py <sim_out.json> \
        [--pool pools/k8192.jsonl] [--room 0.75] [--split] [--png out.png]
"""
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

# one entry per ENTITY; every coldkey prefix listed maps to that single entity
ENTITIES = {
    "TOP4": ("5HMevt8h", "5Eyh8ePM", "5EfHz7fE", "5Hg2Ps2L"),
    "newA": ("5FcJnoeN",),
    "newB": ("5HHKTW5b",),
    "newC": ("5GKjJP64",),
    "e2":   ("5D7BMeGt",),
    "e3":   ("5GghBgin",),
}
PREFIX = {p: name for name, ps in ENTITIES.items() for p in ps}


SPLIT = False


def entity_of(answer):
    if answer[1].startswith("our_"):
        return "OURS"
    if SPLIT:
        return answer[2][:8]
    return PREFIX.get(answer[2][:8], "other")


def load_rounds(path):
    """A simulator output, or a raw rounds.json (the field with us absent)."""
    d = json.load(open(path))
    first = next(iter(d.values()))
    if "answers" in first and "scores" not in first:
        return d, True          # raw rounds.json: field only, we were never there
    return d, False


def load_pool(path):
    """uuid -> set of omega-cliques the harvester found (the real supply)."""
    out = {}
    for line in open(path):
        r = json.loads(line)
        full = r.get("full_pool_unverified") or r.get("pool") or []
        if not full:
            continue
        w = max(len(c) for c in full)
        out[r["uuid"]] = {tuple(sorted(c)) for c in full if len(c) == w}
    return out


def matrix(path, min_supply=8, room=0.0, pool=None):
    d, _ = load_rounds(path)
    dropped = 0
    pair = collections.defaultdict(lambda: [0, 0.0, 0.0])
    seen = collections.Counter()
    answers = collections.Counter()
    dup = collections.defaultdict(lambda: [0, 0.0, 0.0])
    for uuid, rec in d.items():
        top = [a for a in rec["answers"] if a[3]]
        if not top:
            continue
        w = max(len(a[3]) for a in top)
        by = collections.defaultdict(list)
        for a in top:
            if len(a[3]) == w:
                by[entity_of(a)].append(tuple(sorted(a[3])))
        allc = {c for v in by.values() for c in v}
        if pool is not None:
            supply = pool.get(uuid)
            if supply is None:
                continue
            by = {e: [c for c in v if c in supply] for e, v in by.items()}
            by = {e: v for e, v in by.items() if v}
            allc = supply
        S = len(allc)
        if S < min_supply:
            continue
        if room:
            # saturation is a property of the OPERATOR, not the coldkey: always
            # merge before testing it, or --split lets a fleet's saturated
            # rounds back in one key at a time and every ratio drifts toward 1
            merged = collections.defaultdict(set)
            for a in top:
                if len(a[3]) == w:
                    e = "OURS" if a[1].startswith("our_") else PREFIX.get(a[2][:8], "other")
                    c = tuple(sorted(a[3]))
                    if pool is None or c in allc:
                        merged[e].add(c)
            if merged and max(len(v) for v in merged.values()) >= room * S:
                dropped += 1
                continue
        for e, cs in by.items():
            seen[e] += 1
            answers[e] += len(cs)
            n = len(cs)
            distinct = len(set(cs))
            # self: observed repeats vs what independent picking of n from S gives
            dup[e][0] += n - distinct
            dup[e][1] += n * (n - 1) / (2.0 * S)
            # n answers into S cliques force n-S repeats by pigeonhole
            dup[e][2] += max(0, n - S)
        names = sorted(by)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                A, B = set(by[names[i]]), set(by[names[j]])
                pair[(names[i], names[j])][0] += len(A & B)
                pair[(names[i], names[j])][1] += len(A) * len(B) / float(S)
                # |A|+|B| > S forces an overlap; independence is unreachable there
                pair[(names[i], names[j])][2] += max(0, len(A) + len(B) - S)
    return pair, dup, seen, answers, dropped


LABELS = {"5HMevt8h": "T_a", "5Eyh8ePM": "T_b", "5EfHz7fE": "T_c",
          "5Hg2Ps2L": "T_d", "5FcJnoeN": "newA", "5HHKTW5b": "newB",
          "5GKjJP64": "newC"}


def heatmap(order, ratio, seen, answers, path, subtitle):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    n = len(order)
    M = np.full((n, n), np.nan)
    for i, a in enumerate(order):
        for j, b in enumerate(order):
            r = ratio(a, b)
            if r is not None:
                M[i, j] = r
    names = ["%s\n%s" % (LABELS[e], e) if e in LABELS else e for e in order]
    fig, ax = plt.subplots(figsize=(1.05 * n + 3.2, 1.05 * n + 2.4))
    im = ax.imshow(M, cmap="coolwarm", vmin=0.0, vmax=2.0)
    ax.set_xticks(range(n), names, fontsize=8)
    ax.set_yticks(range(n), names, fontsize=8)
    ax.set_xticks(np.arange(n + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    for i in range(n):
        for j in range(n):
            if np.isnan(M[i, j]):
                ax.text(j, i, "-", ha="center", va="center", color="0.6", fontsize=9)
            else:
                shade = "white" if abs(M[i, j] - 1.0) > 0.72 else "black"
                ax.text(j, i, "%.2f" % M[i, j], ha="center", va="center",
                        color=shade, fontsize=9)
    cb = fig.colorbar(im, ax=ax, shrink=0.72)
    cb.set_label("observed / expected   (1.00 = independent picking)", fontsize=8)
    cb.ax.axhline(1.0, color="black", linewidth=1.2)
    ax.set_title("SN83 clique-collision matrix\n%s" % subtitle, fontsize=10, pad=12)
    fig.text(0.01, 0.015,
             "blue = avoids (partitions the clique space)   red = collides more than chance\n"
             "diagonal = self-collision   cells with expectation < 3 shown as -",
             fontsize=7.5, color="0.35")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=170)
    return path


def main():
    global SPLIT
    argv = sys.argv[1:]
    room = 0.0
    if "--room" in argv:
        i = argv.index("--room")
        room = float(argv[i + 1])
        del argv[i:i + 2]
    raw = False
    if "--raw" in argv:
        raw = True
        argv.remove("--raw")
    if "--split" in argv:
        SPLIT = True
        argv.remove("--split")
    pool = None
    if "--pool" in argv:
        i = argv.index("--pool")
        pool = load_pool(argv[i + 1])
        del argv[i:i + 2]
    png = None
    if "--png" in argv:
        i = argv.index("--png")
        png = argv[i + 1]
        del argv[i:i + 2]
    path = argv[0]
    pair, dup, seen, answers, dropped = matrix(path, room=room, pool=pool)
    known = ("OURS", "TOP4", "newA", "newB", "newC", "e2", "e3", "other")
    order = [e for e in known if seen.get(e, 0) >= 5]
    if SPLIT:
        # cluster first in a fixed order so the block structure is legible,
        # then anyone else by volume
        rank = {k: i for i, k in enumerate(LABELS)}
        order = sorted((e for e in seen if seen[e] >= 5),
                       key=lambda e: (rank.get(e, len(rank)), -answers[e]))

    def ratio(a, b):
        if a == b:
            o, e, f = dup[a]
        else:
            k = (a, b) if (a, b) in pair else (b, a)
            o, e, f = pair.get(k, [0, 0.0, 0.0])
        if raw:
            return o / e if e >= 3 else None
        # excess over the arithmetically forced floor: with no floor this is the
        # plain ratio, and in saturated rounds it credits only avoidance that was
        # actually available. Lets every round be used instead of dropping some.
        return (o - f) / (e - f) if (e - f) >= 3 else None

    order = [a for a in order
             if any(ratio(a, b) is not None for b in order)]
    print("  COLLISION MATRIX -- observed / independent-expectation")
    print("  <1 avoids, ~1 independent, >1 attracted. Diagonal = self-collision.")
    print("  metric: %s" % ("raw observed/expected"
                            if raw else "excess over forced floor: (obs-floor)/(exp-floor)"))
    print("  source: %s   supply: %s" % (os.path.basename(path),
                                        "harvested pool" if pool else "cliques the field answered"))
    if room:
        print("  --room %.2f: dropped %d saturated rounds (one entity held >= %.0f%% of supply)"
              % (room, dropped, room * 100))
    print()
    w = max(6, max(len(e) for e in order))
    head = " " * (w + 2) + "".join("%*s" % (w + 2, e) for e in order)
    print(head)
    print(" " * (w + 2) + "-" * (len(order) * (w + 2)))
    for a in order:
        cells = []
        for b in order:
            r = ratio(a, b)
            cells.append("%*.2f" % (w + 2, r) if r is not None else "%*s" % (w + 2, "-"))
        print("%-*s|%s" % (w + 1, a, "".join(cells)))
    print("\n  rounds present / omega-answers:")
    for e in order:
        print("    %-8s %4d rounds  %6d answers" % (e, seen[e], answers[e]))
    off = [ratio(a, b) for i, a in enumerate(order) for b in order[i + 1:]
           if ratio(a, b) is not None]
    if off:
        print("\n  mean off-diagonal (entity vs entity): %.2f" % (sum(off) / len(off)))
    diag = [(a, ratio(a, a)) for a in order if ratio(a, a) is not None]
    if diag:
        print("  self-collision: " + "  ".join("%s %.2f" % (a, r) for a, r in diag))
    if png:
        _emit_png(png, order, ratio, seen, answers, room, dropped, pool, raw)


def _emit_png(png, order, ratio, seen, answers, room, dropped, pool, raw):
    bits = ["supply: %s" % ("harvested pool" if pool else "field answers")]
    bits.append("all rounds" if not room else
                "%d saturated rounds dropped" % dropped)
    bits.append("raw ratio" if raw else "excess over forced floor")
    out = heatmap(order, ratio, seen, answers, png, "  ·  ".join(bits))
    print("\n  wrote %s" % out)


if __name__ == "__main__":
    main()
