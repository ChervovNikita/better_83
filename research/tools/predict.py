#!/usr/bin/env python3
"""Can a LOCAL feature tell a contested maximum clique from an unclaimed one?

tools/order.py established that reach is not the binding constraint. Exact enumeration
inside our own 180-vertex hull returns 44.0 maximum cliques per round of which 29.4 are
held by NOBODY in the field -- but only 2.39 of them land in the 8 we submit, because we
pick blind. Perfect selection would take the submitted novel share from ~30% to ~100%.

So the question is no longer "can we find un-taken cliques". It is "given 44 maximum
cliques and no sight of the field, can we rank them so the un-taken ones come first?"

Every feature here is computable from the graph and our own search alone -- nothing uses
a field answer. The label does, because it is the thing being predicted.

    degsum     total degree of the members. A clique of high-degree vertices sits in the
               dense core, which is where every greedy search lands.
    degmin     the member with fewest neighbours -- how "hard" the clique's tightest
               vertex is to stumble onto.
    ext        how many vertices outside the clique are adjacent to ALL BUT ONE member.
               A clique with many near-misses sits on a wide plateau, so independent
               searches funnel into it.
    lsccHits   how many of our own independent restart chains reached it. A clique our
               own multi-start finds repeatedly is one anybody's multi-start finds.

Reported as AUC -- P(a random unclaimed clique ranks above a random contested one) --
computed WITHIN each round and then averaged, never pooled, because rounds differ in how
many cliques and how many field answers they carry, and pooling that has produced eleven
retractions in this project. AUC 0.50 is no signal.
"""
import argparse
import collections
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/workspace/better_83')
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, HERE)
from CliqueAI.graph.codec import GraphCodec        # noqa: E402
from hull import write_dimacs, run_momc, hull_of    # noqa: E402


def auc(scores, labels):
    """P(score of a positive > score of a negative), ties counted as half."""
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return None
    win = 0.0
    for p in pos:
        for n in neg:
            win += 1.0 if p > n else (0.5 if p == n else 0.0)
    return win / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rounds", type=int, default=60)
    ap.add_argument("--hull", type=int, default=180)
    ap.add_argument("--frac", type=float, default=0.40)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--cache", default=os.path.join(HERE, "..", "data",
                                                    "sim_cliques.jsonl"))
    ap.add_argument("--dataset", default=os.path.join(HERE, "..", "data",
                                                      "sim_rounds.jsonl"))
    args = ap.parse_args()

    pool = {}
    with open(args.cache) as f:
        for line in f:
            r = json.loads(line)
            pool[r["uuid"]] = [tuple(sorted(c)) for c in r["cliques"]]

    rounds = []
    with open(args.dataset) as f:
        for line in f:
            r = json.loads(line)
            if r.get("answers") and r["uuid"] in pool:
                rounds.append(r)
            if len(rounds) >= args.rounds:
                break

    FEATS = ["degsum", "degmin", "ext", "lsccHits"]
    per_round = {f: [] for f in FEATS}
    prefix = {f: [] for f in FEATS}
    base_prefix, used, tot_cl, tot_unc = [], 0, [], []
    tmp = os.path.join(os.environ.get("CLAUDE_JOB_DIR", "/tmp"), "tmp")
    os.makedirs(tmp, exist_ok=True)

    for rec in rounds:
        M = np.array(GraphCodec().decode_matrix(rec["matrix_b92"]), dtype=np.uint8)
        ours = pool[rec["uuid"]]
        if not ours:
            continue
        mx = max(len(c) for c in ours)
        field = {tuple(sorted(int(v) for v in a["clique"]))
                 for a in rec["answers"]
                 if a.get("opt", 0) > 0 and len(a["clique"]) == mx}
        H = hull_of(ours, mx, M, args.hull)
        path = os.path.join(tmp, "pred.clq")
        write_dimacs(M, H, path)
        got, to = run_momc(path, 5000, max(0.5, rec["time_limit"] * args.frac))
        if to or not got:
            continue
        seen, cl = set(), []
        for c in got:
            vs = tuple(sorted(H[i - 1] for i in c if 1 <= i <= len(H)))
            if len(vs) == mx and vs not in seen:
                seen.add(vs)
                cl.append(vs)
        if len(cl) < 4:
            continue
        labels = [1 if c not in field else 0 for c in cl]
        if not any(labels) or all(labels):
            continue
        used += 1
        tot_cl.append(len(cl))
        tot_unc.append(sum(labels))
        deg = M.sum(axis=1).astype(np.int64)
        hits = collections.Counter(c for c in ours if len(c) == mx)

        feats = {}
        feats["degsum"] = [-float(deg[list(c)].sum()) for c in cl]
        feats["degmin"] = [-float(deg[list(c)].min()) for c in cl]
        ext = []
        for c in cl:
            sub = M[list(c)].astype(np.int64).sum(axis=0)
            ext.append(-float(((sub == len(c) - 1)).sum()))
        feats["ext"] = ext
        feats["lsccHits"] = [-float(hits.get(c, 0)) for c in cl]

        base_prefix.append(sum(labels[:args.k]))
        for f in FEATS:
            a = auc(feats[f], labels)
            if a is not None:
                per_round[f].append(a)
            order = sorted(range(len(cl)), key=lambda i: -feats[f][i])
            prefix[f].append(sum(labels[i] for i in order[:args.k]))

    print("hull=%d frac=%.2f K=%d   %d rounds usable\n" % (args.hull, args.frac,
                                                           args.k, used))
    print("  maxima enumerated per round   %.1f" % np.mean(tot_cl))
    print("  of those unclaimed            %.1f  (%.0f%%)"
          % (np.mean(tot_unc), 100 * np.mean(tot_unc) / np.mean(tot_cl)))
    print("  unclaimed in the first %d as emitted   %.2f  (%.0f%%)"
          % (args.k, np.mean(base_prefix), 100 * np.mean(base_prefix) / args.k))
    print()
    print("%-10s %10s %14s %14s" % ("feature", "AUC", "UNCL in K", "vs emitted"))
    for f in FEATS:
        if not per_round[f]:
            continue
        a = np.mean(per_round[f])
        p = np.mean(prefix[f])
        print("%-10s %10.3f %14.2f %+14.2f" % (f, a, p, p - np.mean(base_prefix)))
    print()
    print("  AUC 0.50 is no signal. A feature that ranks unclaimed cliques above")
    print("  contested ones lifts 'UNCL in K' toward %d, which is what the submitted"
          % args.k)
    print("  novel share becomes. All four features use only the graph and our own")
    print("  search -- none of them can see a field answer.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
