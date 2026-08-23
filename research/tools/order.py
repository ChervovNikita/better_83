#!/usr/bin/env python3
"""Is the hull's failure a BUDGET problem or an ORDERING problem?

Nine reach mechanisms are null and the novel share does not move off 25.0%. The one
explanation still standing is that exact enumeration inside our own hull DOES contain
the unclaimed cliques -- 87% of their vertices sit in a 180-vertex hull we build with no
field data -- but MoMC emits the CONTESTED ones first, because its search order starts
where the degree is highest, which is the same attractor every solver falls into. Under
a real deadline it never reaches the rest.

If that is true, the fix costs nothing: enumerate the same region in a different vertex
order and the unclaimed cliques come out early enough to submit. If it is false -- if
they come out last no matter the order -- then reach is closed for good and nothing in
this direction is worth further compute.

Arms permute the DIMACS vertex labels, which is what MoMC's branching order keys on:
  ident    hull order as built, core first        (exact control)
  rev      reversed, so the padding is branched first
  anti     core LAST: padding, then core, by descending degree
  shuf     uniform permutation, per round

The metric is not how many maxima each arm finds. It is how many UNCLAIMED ones land in
the first K emitted, because K is all we can submit.
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
from CliqueAI.graph.codec import GraphCodec          # noqa: E402
from hull import write_dimacs, run_momc, hull_of      # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rounds", type=int, default=60)
    ap.add_argument("--hull", type=int, default=180)
    ap.add_argument("--frac", type=float, default=0.40,
                    help="MoMC budget as a fraction of the round's own time limit")
    ap.add_argument("--k", type=int, default=8, help="submitted prefix")
    ap.add_argument("--cache", default=os.path.join(HERE, "..", "data",
                                                    "sim_cliques.jsonl"))
    ap.add_argument("--dataset", default=os.path.join(HERE, "..", "data",
                                                      "sim_rounds.jsonl"))
    ap.add_argument("--seed", type=int, default=8383)
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

    rng = np.random.default_rng(args.seed)
    ARMS = ["ident", "rev", "anti", "shuf"]
    stat = {a: collections.defaultdict(list) for a in ARMS}
    tmp = os.path.join(os.environ.get("CLAUDE_JOB_DIR", "/tmp"), "tmp")
    os.makedirs(tmp, exist_ok=True)
    used = 0

    for rec in rounds:
        M = np.array(GraphCodec().decode_matrix(rec["matrix_b92"]), dtype=np.uint8)
        ours = pool[rec["uuid"]]
        if not ours:
            continue
        mx = max(len(c) for c in ours)
        # what the FIELD submitted at max size this round -- the claim set
        field = {tuple(sorted(int(v) for v in a["clique"]))
                 for a in rec["answers"]
                 if a.get("opt", 0) > 0 and len(a["clique"]) == mx}
        H = hull_of(ours, mx, M, args.hull)
        if len(H) < mx:
            continue
        deg = M.sum(axis=1)
        core = set()
        for c in ours:
            if len(c) == mx:
                core.update(c)
        budget = max(0.5, rec["time_limit"] * args.frac)
        used += 1
        for arm in ARMS:
            if arm == "ident":
                verts = list(H)
            elif arm == "rev":
                verts = list(reversed(H))
            elif arm == "anti":
                pad = [v for v in H if v not in core]
                cor = [v for v in H if v in core]
                verts = sorted(pad, key=lambda v: -int(deg[v])) + \
                    sorted(cor, key=lambda v: -int(deg[v]))
            else:
                verts = list(H)
                rng.shuffle(verts)
            path = os.path.join(tmp, "order_%s.clq" % arm)
            write_dimacs(M, verts, path)
            t0 = time.time()
            got, to = run_momc(path, 5000, budget)
            el = time.time() - t0
            if to or not got:
                stat[arm]["timeout"].append(1)
                continue
            stat[arm]["timeout"].append(0)
            # MoMC indices are 1-based into `verts`
            cl = []
            for c in got:
                vs = tuple(sorted(verts[i - 1] for i in c if 1 <= i <= len(verts)))
                if len(vs) == mx:
                    cl.append(vs)
            seen, order = set(), []
            for c in cl:                      # preserve EMISSION order, dedup
                if c not in seen:
                    seen.add(c)
                    order.append(c)
            unc = [i for i, c in enumerate(order) if c not in field]
            stat[arm]["found"].append(len(order))
            stat[arm]["unclaimed"].append(len(unc))
            stat[arm]["in_prefix"].append(sum(1 for i in unc if i < args.k))
            stat[arm]["first_rank"].append(unc[0] if unc else -1)
            stat[arm]["secs"].append(el)

    print("hull=%d frac=%.2f K=%d  %d rounds usable of %d\n"
          % (args.hull, args.frac, args.k, used, len(rounds)))
    print("%-7s %8s %9s %11s %14s %9s" % ("arm", "timeout", "maxima", "unclaimed",
                                          "UNCL in K=%d" % args.k, "s/round"))
    for a in ARMS:
        s = stat[a]
        if not s["found"]:
            print("%-7s  all timed out" % a)
            continue
        print("%-7s %7.0f%% %9.1f %11.1f %14.2f %9.2f"
              % (a, 100 * np.mean(s["timeout"]), np.mean(s["found"]),
                 np.mean(s["unclaimed"]), np.mean(s["in_prefix"]), np.mean(s["secs"])))
    print()
    base = np.mean(stat["ident"]["in_prefix"]) if stat["ident"]["in_prefix"] else 0.0
    print("  The control (ident) puts %.2f unclaimed cliques in the submitted prefix."
          % base)
    print("  An arm that beats it means the unclaimed cliques were always reachable and")
    print("  MoMC was simply emitting the contested ones first -- a free fix. An arm")
    print("  that does not means reach is closed: the region contains them and no")
    print("  enumeration order surfaces them inside the deadline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
