#!/usr/bin/env python3
"""Why our fleet is queried less often than the field, per uid.

The validator samples each uid independently at P(difficulty), uniform across uids, so
a fleet of N should collect the same per-uid query rate as anyone else. It does not:
our hotkeys average 48.2 answers per round against the field's 60.7. The warmup accounts
for about 5% of that. This tool locates the rest.

The mechanism it tests: `fleet_sim` seats our fleet on uids chosen by pick_victims, and
`sample="real"` treats a slot as queried this round iff it APPEARS IN THE LOG that round.
A uid that churned -- deregistered, re-registered, sat empty, or was inside immunity --
appears in fewer rounds, so seating ourselves there inherits its absence. That is a
property of which seats we took, not of the sampling, and it is fixable by choosing
different ones.
"""
import argparse
import collections
import json
import os
import sys

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


def like_for_like(path, segments, fleet_sim):
    """Our query rate vs the field's, both measured inside the same simulation."""
    rounds = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rounds.append(json.loads(line))
    T = len(rounds)
    n_our = len({a["who"] for r in rounds for a in r["answers"] if a["ours"]})
    n_fld = len({a["who"] for r in rounds for a in r["answers"] if not a["ours"]})
    if not n_our:
        raise SystemExit("no fleet submissions in the dump")

    def rate(chunk):
        o = np.mean([sum(1 for a in r["answers"] if a["ours"]) for r in chunk]) / n_our
        f = np.mean([sum(1 for a in r["answers"] if not a["ours"]) for r in chunk]) / n_fld
        return o, f

    warm_rounds = fleet_sim.WARMUP_BLOCKS * fleet_sim.BLOCK_S
    print(f"{T} rounds, {n_our} of our identities, {n_fld} field identities")
    print(f"registration warmup {fleet_sim.WARMUP_BLOCKS} blocks x "
          f"{fleet_sim.BLOCK_S:.0f}s = {warm_rounds / 60:.0f} min\n")
    o, f = rate(rounds)
    print("whole window        ours %.4f  field %.4f  ratio %.3f" % (o, f, o / f))
    print()
    print("%-14s %10s %10s %8s" % ("round block", "ours", "field", "ratio"))
    per = max(1, T // segments)
    first_full = None
    for s0 in range(0, T, per):
        ch = rounds[s0:s0 + per]
        if not ch:
            continue
        o, f = rate(ch)
        r = o / f if f else 0.0
        if first_full is None and r >= 0.98:
            first_full = s0
        print("%-14s %10.4f %10.4f %8.3f%s"
              % (f"{s0}-{s0 + len(ch)}", o, f, r,
                 "   <- warmup" if r < 0.9 else ""))
    if first_full:
        o, f = rate(rounds[first_full:])
        print()
        print("past the warmup (from round %d)  ours %.4f  field %.4f  ratio %.3f"
              % (first_full, o, f, o / f))
        print()
        print("  A ratio at or above 1.00 here means there is NO standing query")
        print("  deficit: the whole-window figure is the one-time cost of registering,")
        print("  which every new hotkey pays once and which shrinks as the window")
        print("  lengthens. It is not a property of the fleet to be fixed.")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=os.path.join(DATA_DIR, "sim_rounds.jsonl"))
    ap.add_argument("--metagraph", default=os.path.join(DATA_DIR, "metagraph.json"))
    ap.add_argument("--rounds", type=int, default=100000)
    ap.add_argument("--submissions", default=None,
                    help="a `fleet_sim --dump-submissions` JSONL. With it, the query "
                         "rate is compared LIKE-FOR-LIKE -- our simulated identities "
                         "against the field identities in the SAME simulation -- and "
                         "split by position in the window, which separates the "
                         "registration warmup from any standing deficit. Comparing a "
                         "simulated fleet against the field's LOGGED counts inflates "
                         "the gap; that error produced the 48.2-vs-60.7 figure.")
    ap.add_argument("--segments", type=int, default=8)
    args = ap.parse_args()

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    import fleet_sim

    if args.submissions:
        return like_for_like(args.submissions, args.segments, fleet_sim)

    rounds = []
    with open(args.dataset) as f:
        for line in f:
            r = json.loads(line)
            if r.get("answers"):
                rounds.append(r)
            if len(rounds) >= args.rounds:
                break
    meta = json.load(open(args.metagraph))

    seen = collections.Counter()          # rounds in which this uid answered
    hks = collections.defaultdict(set)    # distinct hotkeys ever seen on this uid
    for r in rounds:
        for a in r["answers"]:
            seen[a["uid"]] += 1
            hks[a["uid"]].add(a["hk"])
    T = len(rounds)
    uids = sorted(seen)
    rate = np.array([seen[u] / T for u in uids])
    churn = np.array([len(hks[u]) for u in uids])

    print(f"{T} rounds, {len(uids)} uids ever answering\n")
    print("per-uid appearance rate (fraction of rounds this uid answered)")
    for q in (5, 25, 50, 75, 95):
        print("  p%-3d %.3f" % (q, np.percentile(rate, q)))
    print("  mean %.3f" % rate.mean())
    print()
    print("appearance rate vs how many DISTINCT hotkeys occupied that uid")
    for c in sorted(set(churn.tolist()))[:6]:
        m = churn == c
        print("  %d hotkey%s  n=%3d   mean rate %.3f" % (c, " " if c == 1 else "s",
                                                         m.sum(), rate[m].mean()))
    print()

    occ = fleet_sim.occupants(rounds, meta)
    for N in (10, 40, 120):
        vic = fleet_sim.pick_victims(meta, N)
        vset = set(vic)
        m = np.array([u in vset for u in uids])
        if not m.any():
            print("  N=%-4d victims never answer in this window" % N)
            continue
        print("  N=%-4d victim uids  mean rate %.3f   field mean %.3f   ratio %.2f"
              % (N, rate[m].mean(), rate[~m].mean(), rate[m].mean() / rate[~m].mean()))
        print("         victims never seen answering: %d of %d"
              % (sum(1 for u in vic if u not in seen), len(vic)))
    print()
    print("  a ratio below 1.00 means pick_victims seats us on uids that were absent")
    print("  more often than average, and we inherit that absence -- the sampling")
    print("  itself is uniform, so nothing else can produce it.")
    print("  occupants() reports %d uids occupied at window start" % len(occ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
