#!/usr/bin/env python3
"""Pairwise collision matrix between ENTITIES, built from the simulator's own output.

A "collision" is two co-occurring answers in the same round carrying the BYTE-IDENTICAL
vertex set. That is the only thing the scorer's diversity term counts:

    diversity = (1 / Counter[tuple(sorted(clique))]) / max_delta

so a pair of answers that differ by one vertex is worth exactly as much as a pair that
share nothing. The matrix therefore reads directly as "how much diversity reward is
each pair of entities burning on each other".

INPUT: the JSONL written by `fleet_sim --dump-submissions`, NOT data/sim_rounds.jsonl.
This matters and is not a stylistic preference:

  * sim_rounds holds every hotkey the log ever saw. Our fleet DEREGISTERS some of them,
    so building the field's rows from sim_rounds counts answers that would not have
    existed and inflates the field's apparent supply.
  * our own row cannot be reconstructed from sim_rounds at all. What each of our
    hotkeys submits is the PICKER's decision (fleet_pick.picker wraps modularly on a
    short pool, so it deliberately repeats), and the queried set comes from the
    validator's per-uid sampling. Slicing `pool[:k]` off a distinct pool -- the obvious
    reconstruction -- makes our self-collision 0.00% BY CONSTRUCTION and is wrong: the
    measured rate under the real picker is an order of magnitude higher.

ENTITY GROUPING. Hotkeys are grouped by coldkey; coldkeys below --min-hotkeys are pooled
as "indep". Coldkeys are then MERGED into one entity when they collide far less often
than independence predicts, because that is the signature of one operator running a
deduplicating fleet across several coldkeys. Expected collisions for a pair of groups
under independence are computed per round from the clique multiset the round actually
produced -- not from a global rate -- since round difficulty drives both the number of
distinct maxima and how many answers land on each.
"""
import argparse
import collections
import itertools
import json
import math
import os
import sys

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


def load_submissions(path):
    rounds = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rounds.append(json.loads(line))
    if not rounds:
        raise SystemExit(f"{path} is empty")
    if not any(a["ours"] for r in rounds for a in r["answers"]):
        print(f"  WARNING: no OURS-* submissions in {path}. The fleet was never "
              f"sampled, so the US row will be absent. Was --sizes 0 used?",
              file=sys.stderr)
    return rounds


def entity_map(metagraph, min_hotkeys):
    """hotkey -> raw group label, before any merging."""
    meta = json.load(open(metagraph))
    hk2ck = {m["hotkey"]: m["coldkey"] for m in meta["miners"]}
    size = collections.Counter(hk2ck.values())
    big = [c for c, n in size.most_common() if n >= min_hotkeys]
    label = {c: "G%d" % (i + 1) for i, c in enumerate(big)}
    return {h: label.get(c, "indep") for h, c in hk2ck.items()}, hk2ck, label


def round_pairs(rounds, group_of):
    """Per round, the (group, clique) list of every SCORED, valid answer.

    Invalid and unserved answers are dropped: they carry no clique, so they cannot
    collide with anything and including them would dilute every rate by a constant
    that differs per entity.
    """
    out = []
    for r in rounds:
        per = []
        for a in r["answers"]:
            if not a["valid"] or a["clique"] is None:
                continue
            g = "US" if a["ours"] else group_of.get(a["who"], "indep")
            per.append((g, tuple(a["clique"])))
        if len(per) >= 2:
            out.append(per)
    return out


def tally(per_round, groups):
    """observed[(g1,g2)] and total[(g1,g2)] over co-occurring answer pairs."""
    obs = collections.Counter()
    tot = collections.Counter()
    for per in per_round:
        for i in range(len(per)):
            gi, ci = per[i]
            for j in range(i + 1, len(per)):
                gj, cj = per[j]
                k = (gi, gj) if gi <= gj else (gj, gi)
                tot[k] += 1
                if ci == cj:
                    obs[k] += 1
    return obs, tot


def expected_within_round(per_round):
    """E[collisions] per group pair if each entity drew from that round's own mix.

    For a round with clique multiset counts {c: n_c} and group sizes {g: m_g}, a
    permutation null that keeps both margins gives, for g != h,

        E = m_g * m_h * sum_c (n_c/M)*((n_c-1)/(M-1))     ... approximately,

    where M is the round's answer count. Using the round's OWN multiset is what makes
    this a null about *who submitted what* rather than about how hard the round was:
    a pooled global rate would attribute round-difficulty structure to the entities.
    """
    exp = collections.Counter()
    for per in per_round:
        M = len(per)
        if M < 2:
            continue
        cnt = collections.Counter(c for _, c in per)
        # P(two distinct answers drawn without replacement match)
        pmatch = sum(n * (n - 1) for n in cnt.values()) / (M * (M - 1))
        gs = collections.Counter(g for g, _ in per)
        for g, h in itertools.combinations_with_replacement(sorted(gs), 2):
            npairs = gs[g] * gs[h] if g != h else gs[g] * (gs[g] - 1) / 2
            if npairs:
                exp[(g, h)] += npairs * pmatch
    return exp


def poisson_tail(obs, exp):
    """P(X <= obs) when X ~ Poisson(exp). The evidence that a pair collides LESS than
    independence predicts.

    A raw floor on `expected` is the wrong test and rejected a real merge: B/D had 0
    observed against 44.3 expected, which is P = e^-44.3 ~ 6e-20, yet a floor of 100
    called it thin evidence. What matters is how improbable the observed count is under
    the null, not how large the null is.
    """
    if exp <= 0:
        return 1.0
    # sum_{k<=obs} e^-exp exp^k / k!, computed in log space for large exp
    tot = 0.0
    term = math.exp(-exp) if exp < 700 else 0.0
    if term == 0.0 and obs == 0:
        return math.exp(-exp) if exp < 700 else 0.0
    for k in range(0, obs + 1):
        if k:
            term *= exp / k
        tot += term
    return min(1.0, tot)


def merge_groups(obs, exp, tot, labels, ratio_max, min_expected):
    """Union coldkey groups whose cross-collisions are far BELOW independence.

    Deduplicating across coldkeys is deliberate coordination; nothing else drives the
    ratio to zero while both groups are large enough for the null to have teeth.
    """
    parent = {g: g for g in labels}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    merges = []
    for g, h in itertools.combinations(sorted(labels), 2):
        k = (g, h) if g <= h else (h, g)
        e = exp.get(k, 0.0)
        o = obs.get(k, 0)
        if e <= 0:
            continue
        r = o / e
        # min_expected is now a Poisson ALPHA, not a count floor. Merge when the
        # observed count is improbably low under independence.
        if poisson_tail(o, e) > min_expected or r > ratio_max:
            continue
        if True:
            merges.append((g, h, obs.get(k, 0), e, r))
            a, b = find(g), find(h)
            if a != b:
                parent[a] = b
    return {g: find(g) for g in labels}, merges


def fmt_matrix(obs, tot, order, title, note=None):
    w = max(9, max(len(o) for o in order) + 2)
    lines = [title, ""]
    lines.append(" " * 8 + "".join("%*s" % (w, o) for o in order))
    for g in order:
        row = "%-8s" % g
        for h in order:
            k = (g, h) if g <= h else (h, g)
            t = tot.get(k, 0)
            row += ("%*.2f%%" % (w - 1, 100.0 * obs.get(k, 0) / t)) if t >= 50 \
                else "%*s" % (w, "-")
        lines.append(row)
    if note:
        lines += ["", note]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("submissions",
                    help="JSONL from `fleet_sim --dump-submissions`")
    ap.add_argument("--metagraph", default=os.path.join(DATA_DIR, "metagraph.json"))
    ap.add_argument("--min-hotkeys", type=int, default=15,
                    help="coldkeys below this are pooled into 'indep' (default 15)")
    ap.add_argument("--merge-ratio", type=float, default=0.35,
                    help="merge two coldkey groups into one entity when their observed "
                         "collisions are below this multiple of the independence "
                         "expectation (default 0.35)")
    ap.add_argument("--min-expected", type=float, default=1e-6, metavar="ALPHA",
                    help="Poisson tail alpha: merge two coldkey groups when P(X <= "
                         "observed) under independence is below this (default 1e-6). "
                         "This replaced a raw floor on the expected count, which "
                         "rejected a pair with 0 observed against 44.3 expected -- "
                         "P ~ 6e-20 -- as 'thin evidence'.")
    ap.add_argument("--no-merge", action="store_true",
                    help="report raw coldkey groups without linkage detection")
    ap.add_argument("--json", default=None, help="also write the matrix here")
    args = ap.parse_args()

    rounds = load_submissions(args.submissions)
    group_of, hk2ck, label = entity_map(args.metagraph, args.min_hotkeys)
    per_round = round_pairs(rounds, group_of)

    raw_groups = sorted(set(label.values()))
    obs, tot = tally(per_round, raw_groups)
    exp = expected_within_round(per_round)

    ck_of = {v: k for k, v in label.items()}
    hk_count = collections.Counter(group_of.values())

    print(f"{len(rounds)} rounds dumped, {len(per_round)} with >=2 valid answers, "
          f"{sum(len(p) for p in per_round)} scored valid submissions")
    print(f"grouping by coldkey, >={args.min_hotkeys} hotkeys named separately\n")

    if args.no_merge:
        final = {g: g for g in raw_groups}
        merges = []
    else:
        final, merges = merge_groups(obs, exp, tot, raw_groups,
                                     args.merge_ratio, args.min_expected)

    print("linkage test -- observed vs independence expectation, coldkey groups")
    print("%-14s %10s %12s %8s" % ("pair", "observed", "expected", "ratio"))
    rows = []
    for g, h in itertools.combinations(sorted(raw_groups), 2):
        k = (g, h) if g <= h else (h, g)
        e = exp.get(k, 0.0)
        if e >= 1.0:
            rows.append((obs.get(k, 0) / e, g, h, obs.get(k, 0), e))
    for r, g, h, o, e in sorted(rows)[:8]:
        mark = "  <- MERGED" if final[g] == final[h] and not args.no_merge else ""
        print("%-14s %10d %12.1f %8.2f%s" % (f"{g}/{h}", o, e, r, mark))
    print()

    # relabel to merged entities, then re-tally from scratch so within-entity pairs
    # that used to be cross-coldkey now land on the diagonal
    comp = collections.defaultdict(list)
    for g in raw_groups:
        comp[final[g]].append(g)
    name = {}
    for i, (root, members) in enumerate(sorted(comp.items(),
                                               key=lambda kv: -sum(hk_count[m] for m in kv[1]))):
        name[root] = "indep" if members == ["indep"] else chr(ord("A") + i)
    merged_of = {h: name[final[g]] if g in final else "indep"
                 for h, g in group_of.items()}
    per_round2 = round_pairs(rounds, merged_of)
    order = sorted({g for per in per_round2 for g, _ in per},
                   key=lambda g: (g == "indep", g == "US", g))
    obs2, tot2 = tally(per_round2, order)

    print(fmt_matrix(obs2, tot2, order,
                     "PAIRWISE COLLISION RATE -- % of co-occurring answer pairs "
                     "submitting the IDENTICAL clique"))
    print()
    print("%-7s %8s  %s" % ("entity", "hotkeys", "composition"))
    for root, members in sorted(comp.items(),
                                key=lambda kv: -sum(hk_count[m] for m in kv[1])):
        n = sum(hk_count[m] for m in members)
        cks = [ck_of[m][:14] for m in members if m in ck_of]
        who = ", ".join(cks) if cks else "all coldkeys below the threshold"
        print("%-7s %8d  %d coldkey%s: %s" % (name[root], n, len(members),
                                              "" if len(members) == 1 else "s", who))
    n_our = len({a["who"] for r in rounds for a in r["answers"] if a["ours"]})
    if n_our:
        print("%-7s %8d  our fleet, as submitted by the picker" % ("US", n_our))

    def rate(g, h):
        k = (g, h) if g <= h else (h, g)
        t = tot2.get(k, 0)
        return (100.0 * obs2.get(k, 0) / t) if t >= 50 else None

    field = [g for g in order if g != "US"]
    cross = [(rate(g, h), g, h) for g, h in itertools.combinations(field, 2)
             if rate(g, h) is not None]
    selfs = [(rate(g, g), g) for g in field if rate(g, g) is not None]
    if cross and "US" in order:
        ours_cross = [(rate("US", g), g) for g in field if rate("US", g) is not None]
        print()
        print("headline")
        print("  field cross-entity   min %.2f%%  median %.2f%%  max %.2f%%  (%s)"
              % (min(c[0] for c in cross),
                 sorted(c[0] for c in cross)[len(cross) // 2],
                 max(c[0] for c in cross),
                 "/".join(max(cross)[1:])))
        print("  field self           " + "  ".join("%s %.2f%%" % (g, r)
                                                    for r, g in sorted(selfs)))
        if ours_cross:
            print("  US cross-entity      min %.2f%%  median %.2f%%  max %.2f%%  (%s)"
                  % (min(c[0] for c in ours_cross),
                     sorted(c[0] for c in ours_cross)[len(ours_cross) // 2],
                     max(c[0] for c in ours_cross), max(ours_cross)[1]))
        us_self = rate("US", "US")
        if us_self is not None:
            print("  US self              %.2f%%   <- the picker repeating on a short "
                  "pool, not a solver failure" % us_self)

    if args.json:
        json.dump({"order": order,
                   "collisions": {f"{g}|{h}": obs2.get((g, h) if g <= h else (h, g), 0)
                                  for g in order for h in order},
                   "pairs": {f"{g}|{h}": tot2.get((g, h) if g <= h else (h, g), 0)
                             for g in order for h in order},
                   "entities": {name[r]: sum(hk_count[m] for m in ms)
                                for r, ms in comp.items()},
                   "merges": [{"a": g, "b": h, "observed": o, "expected": e,
                               "ratio": r} for r, g, h, o, e in
                              sorted((m[4], m[0], m[1], m[2], m[3]) for m in merges)],
                   }, open(args.json, "w"), indent=1)
        print(f"\n  wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
