#!/usr/bin/env python3
"""The search metric: our_median - field_median, from a simulate.py --out dump.

    .venv/bin/python research_manual/eda/metric_edge.py <sim_out.json> [--json]

A hotkey's score is the mean of the per-round rewards the VALIDATOR assigned it
-- simulate.py stores exactly what CliqueScoreCalculator returned, so this reads
the validator's own numbers and computes no reward of its own. That is the whole
point: a metric script that rescores anything is a second harness, and a second
harness is how a wrong default gets shipped.

our_median  = median over our hotkeys of their mean reward
field_median = median over every other hotkey of their mean reward
edge        = our_median - field_median      <- maximize this

Per-coldkey medians are printed too, because beating the field average while
losing to the largest operator is not winning.
"""
import argparse
import collections
import json
import statistics

OURS_PREFIX = "our_hotkey_"


def load(path):
    with open(path) as handle:
        payload = json.load(handle)
    per = collections.defaultdict(list)
    cold = {}
    for rec in payload.values():
        for (uid, hotkey, coldkey, _clique), score in zip(rec["answers"],
                                                          rec["scores"]):
            per[hotkey].append(float(score))
            cold[hotkey] = coldkey
    return payload, per, cold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--json", action="store_true",
                    help="one machine-readable line, for the experiment ledger")
    args = ap.parse_args()

    payload, per, cold = load(args.dump)
    # A round our fleet answered LATE scores every one of our queried hotkeys at
    # zero, which moves the median hard. It happened once here: a run that built
    # the GPU cache while it scored overran two deadlines and lost 35 answers,
    # which read as a 0.06 strategy regression and got a mechanism invented for
    # it. An arm paying a cost the others do not is not a comparison, so refuse
    # to report the number without flagging it.
    empty = sum(1 for rec in payload.values()
                for a in rec["answers"] if a[1].startswith(OURS_PREFIX) and not a[3])
    total = sum(1 for rec in payload.values()
                for a in rec["answers"] if a[1].startswith(OURS_PREFIX))
    means = {h: statistics.mean(v) for h, v in per.items() if v}
    ours = {h: m for h, m in means.items() if h.startswith(OURS_PREFIX)}
    field = {h: m for h, m in means.items() if not h.startswith(OURS_PREFIX)}
    assert field, "no field hotkeys in the dump"
    if not ours:
        raise SystemExit("no scored hotkeys of ours: the fleet was never queried")

    our_med = statistics.median(ours.values())
    field_med = statistics.median(field.values())
    edge = our_med - field_med

    if args.json:
        print(json.dumps({"edge": edge, "our_median": our_med,
                          "field_median": field_med, "n_ours": len(ours),
                          "n_field": len(field), "rounds": len(payload),
                          "empty_answers": empty,
                          "SUSPECT": empty > 0}))
        return

    print("rounds %d | our hotkeys scored %d | field hotkeys %d"
          % (len(payload), len(ours), len(field)))
    if empty:
        print("  WARNING: %d/%d of our answers are EMPTY (late rounds). Every such"
              % (empty, total))
        print("  answer scores 0 and drags our_median down. This run is NOT")
        print("  comparable with one that had a warm cache.")
    print("  our_median   %.6f" % our_med)
    print("  field_median %.6f" % field_med)
    print("  EDGE         %+.6f" % edge)

    by_cold = collections.defaultdict(list)
    for hotkey, mean in field.items():
        by_cold[cold[hotkey]].append(mean)
    print("\n  per-coldkey median (field), largest first:")
    ranked = sorted(by_cold.items(), key=lambda kv: -len(kv[1]))
    for coldkey, values in ranked[:8]:
        med = statistics.median(values)
        print("    %-14s n=%3d  median %.6f   we lead by %+.6f"
              % (coldkey[:12], len(values), med, our_med - med))


if __name__ == "__main__":
    main()
