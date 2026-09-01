#!/usr/bin/env python3
"""Emission share, the quantity that actually pays.

    .venv/bin/python research_manual/eda/metric_share.py <sim_out.json> [--json]

`metric_edge.py` reports our_median - field_median. That is a rank statistic and
NOT what a miner is paid: the validator turns per-hotkey mean scores into weights
through simulate.validator_weights (a sigmoid then a power transform tuned so the
top half takes 80% of emission), and pays each hotkey its share of the total.

A strategy can raise the median while lowering the share, or drag the whole field
down and improve the median difference while earning less. This scores the thing
the chain pays out, using the validator's own weighting function so no second
implementation can drift from it.
"""
import argparse
import collections
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
for _p in (PARENT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import simulate

OURS_PREFIX = "our_hotkey_"


def shares(path):
    with open(path) as handle:
        payload = json.load(handle)
    per = collections.defaultdict(list)
    for rec in payload.values():
        for answer, score in zip(rec["answers"], rec["scores"]):
            per[answer[1]].append(float(score))
    means = {h: statistics.mean(v) for h, v in per.items() if v}
    hotkeys = sorted(means)
    weights = simulate.validator_weights([means[h] for h in hotkeys])
    ours = sum(float(w) for h, w in zip(hotkeys, weights)
               if h.startswith(OURS_PREFIX))
    n_ours = sum(1 for h in hotkeys if h.startswith(OURS_PREFIX))
    return ours, n_ours, len(hotkeys), payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    ours, n_ours, n_all, payload = shares(args.dump)
    fair = n_ours / float(n_all)
    if args.json:
        print(json.dumps({"share": ours, "fair_share": fair,
                          "ratio": ours / fair if fair else 0.0,
                          "n_ours": n_ours, "n_all": n_all,
                          "rounds": len(payload)}))
        return
    print("rounds %d | our hotkeys %d of %d" % (len(payload), n_ours, n_all))
    print("  emission share      %.5f" % ours)
    print("  share if average    %.5f  (n_ours / n_all)" % fair)
    print("  ratio to average    %.3f" % (ours / fair if fair else 0.0))


if __name__ == "__main__":
    main()
