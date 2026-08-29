#!/usr/bin/env python3
"""Fetch round TIMINGS only, over a long span, to size concurrency.

    .venv/bin/python research_manual/eda/dump_timing.py --hours 36

Writes research_manual/eda/timing.json: one row per round with uuid, timestamp,
time_limit, number_of_nodes and the validator run that logged it.

Deliberately does NOT pull encoded_matrix -- it is the bulk of a round record and
none of it is needed to answer "how often do two validators have us solving at
the same time".  That makes a 24-hour scan cheap where a full dump is not.
"""

import argparse
import collections
import json
import os
import sys

import wandb

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)

PROJECT = "toptensor-ai/CliqueAI"
VERSION = "0.0.17"
KEYS = ["uuid", "timestamp", "difficulty", "time_limit", "number_of_nodes"]
DEST = os.path.join(HERE, "timing.json")


def load_env():
    if os.environ.get("WANDB_API_KEY"):
        return
    env_path = os.path.join(os.path.dirname(PARENT), ".env")
    if not os.path.exists(env_path):
        return
    for line in open(env_path):
        line = line.strip()
        if line.startswith("WANDB_API_KEY="):
            os.environ["WANDB_API_KEY"] = line.split("=", 1)[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=36.0)
    ap.add_argument("--out", default=DEST)
    args = ap.parse_args()

    load_env()
    assert os.environ.get("WANDB_API_KEY"), "WANDB_API_KEY not set and not in .env"
    major, minor = (int(x) for x in wandb.__version__.split(".")[:2])
    assert (major, minor) < (0, 20), wandb.__version__

    api = wandb.Api(timeout=300)
    runs = list(api.runs(PROJECT, filters={"config.version": {"$in": [VERSION]}},
                         per_page=100))
    assert runs
    rows = {}
    per_run = collections.Counter()
    for run in runs:
        head = int(run.summary["_step"])
        # ~34 s a round per validator, so an hour is ~106 steps; scan generously
        span = int(args.hours * 3600 / 30.0)
        lo = max(0, head - span)
        for row in run.scan_history(keys=KEYS, min_step=lo, max_step=head + 1,
                                    page_size=500):
            rid = row.get("uuid")
            if not rid or rid in rows:
                continue
            rows[rid] = {
                "uuid": rid,
                "run": run.id[:12],
                "timestamp": row["timestamp"],
                "difficulty": row["difficulty"],
                "time_limit": row["time_limit"],
                "number_of_nodes": row["number_of_nodes"],
            }
            per_run[run.id[:12]] += 1
        print("  %s: %d rounds so far" % (run.id[:12], per_run[run.id[:12]]),
              file=sys.stderr, flush=True)

    out = sorted(rows.values(), key=lambda r: r["timestamp"])
    assert out
    span_h = (out[-1]["timestamp"] - out[0]["timestamp"]) / 3600.0
    with open(args.out, "w") as handle:
        json.dump(out, handle)
    print("wrote %s: %d rounds over %.1f h, %d validator runs"
          % (args.out, len(out), span_h, len(per_run)), file=sys.stderr)


if __name__ == "__main__":
    main()
