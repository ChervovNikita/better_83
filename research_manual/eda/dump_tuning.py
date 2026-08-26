#!/usr/bin/env python3
"""Fetch a tuning set of rounds strictly OLDER than rounds.json.

    .venv/bin/python research_manual/eda/dump_tuning.py            # 100 rounds
    .venv/bin/python research_manual/eda/dump_tuning.py --rounds 200

Writes research_manual/eda/tuning_data.json in exactly the schema rounds.json
uses, so every existing tool reads it with --dump:

    .venv/bin/python research_manual/simulate.py -N 40 --rounds 100 \
        --dump research_manual/eda/tuning_data.json \
        --out research_manual/eda/sim_tuning.json
    .venv/bin/python research_manual/distinct.py \
        --dump research_manual/eda/tuning_data.json

WHY OLDER AND NOT A RANDOM SPLIT
--------------------------------
Anything fitted to a round set can be tuned into it: a picker threshold, a
SPARE_CAP, a hold-back rule. rounds.json is the evaluation set, so a threshold
chosen by looking at rounds.json is not measurable on rounds.json afterwards.

The split is by TIME, not at random, and the tuning set is the EARLIER half.
Tune on the past, evaluate on the future -- the direction the validator actually
runs in. A random split would leak: rounds minutes apart share a metagraph, the
same miners with the same solvers, and often similar graph sizes.

The script asserts every fetched round is strictly older than every round in
rounds.json, and that no uuid appears in both. If either fails it writes nothing.

Needs WANDB_API_KEY (it is in the repo .env) and wandb < 0.20, same as
research_manual/dump_wandb.py, whose fetch this mirrors.
"""

import argparse
import json
import os
import sys

import wandb

PROJECT = "toptensor-ai/CliqueAI"
VERSION = "0.0.17"
KEYS = [
    "uuid",
    "timestamp",
    "difficulty",
    "time_limit",
    "number_of_nodes",
    "encoded_matrix",
    "miner_uids",
    "miner_hotkeys",
    "miner_coldkeys",
    "miner_ans",
]

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
EVAL_PATH = os.path.join(PARENT, "rounds.json")
DEST = os.path.join(HERE, "tuning_data.json")

# rounds.json holds the newest 1000, so the tuning window sits below that. The
# offset cannot be a constant: the validator keeps logging, so `head` drifts
# further above rounds.json's newest round every hour this file is not run. The
# scan walks backwards until it has enough rounds older than the cutoff.
EVAL_ROUNDS = 1000
WINDOW = 400
MAX_PASSES = 12


def load_env():
    """Read WANDB_API_KEY out of the repo .env if it is not already exported."""
    if os.environ.get("WANDB_API_KEY"):
        return
    env_path = os.path.join(os.path.dirname(PARENT), ".env")
    if not os.path.exists(env_path):
        return
    for line in open(env_path):
        line = line.strip()
        if line.startswith("WANDB_API_KEY="):
            os.environ["WANDB_API_KEY"] = line.split("=", 1)[1]


def check_wandb_version():
    major, minor = (int(x) for x in wandb.__version__.split(".")[:2])
    assert (major, minor) < (0, 20), wandb.__version__


def eval_bounds():
    """(earliest timestamp, uuid set) of the evaluation rounds."""
    with open(EVAL_PATH) as handle:
        payload = json.load(handle)
    assert payload
    return min(r["timestamp"] for r in payload.values()), set(payload)


def fetch_window(runs, lo_off, want):
    """Rounds from a window `lo_off` steps below each run's eval boundary."""
    rows = {}
    for run in runs:
        hi = int(run.summary["_step"]) - EVAL_ROUNDS - lo_off
        lo = max(0, hi - WINDOW)
        if hi <= 0:
            continue
        for row in run.scan_history(keys=KEYS, min_step=lo, max_step=hi,
                                    page_size=20):
            round_id = row.get("uuid")
            if not round_id or round_id in rows:
                continue
            assert row["encoded_matrix"]
            answers = list(zip(row["miner_uids"], row["miner_hotkeys"],
                               row["miner_coldkeys"], row["miner_ans"],
                               strict=True))
            assert answers
            rows[round_id] = {
                "timestamp": row["timestamp"],
                "difficulty": row["difficulty"],
                "time_limit": row["time_limit"],
                "number_of_nodes": row["number_of_nodes"],
                "encoded_matrix": row["encoded_matrix"],
                "answers": answers,
            }
    return rows


def fetch_older(api, want, cutoff, eval_ids):
    """Walk backwards a window at a time until `want` rounds predate the cutoff."""
    runs = list(api.runs(PROJECT, filters={"config.version": {"$in": [VERSION]}},
                         per_page=100))
    assert runs
    older = {}
    for step in range(MAX_PASSES):
        rows = fetch_window(runs, step * WINDOW, want)
        for round_id, rec in rows.items():
            if rec["timestamp"] < cutoff and round_id not in eval_ids:
                older[round_id] = rec
        print("  pass %d: %d in window, %d older than cutoff so far"
              % (step, len(rows), len(older)), file=sys.stderr)
        if len(older) >= want:
            return older
    raise AssertionError(
        "only %d rounds older than the evaluation set after %d passes"
        % (len(older), MAX_PASSES))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=100)
    ap.add_argument("--out", default=DEST)
    args = ap.parse_args()
    assert args.rounds > 0

    load_env()
    assert os.environ.get("WANDB_API_KEY"), "WANDB_API_KEY not set and not in .env"
    check_wandb_version()

    cutoff, eval_ids = eval_bounds()
    api = wandb.Api(timeout=180)
    older = fetch_older(api, args.rounds, cutoff, eval_ids)

    # the ones closest to, but before, the evaluation set
    picked = sorted(older.items(), key=lambda kv: kv[1]["timestamp"])[-args.rounds:]
    payload = dict(picked)
    assert len(payload) == args.rounds

    # the guarantees this file exists for
    assert not (set(payload) & eval_ids), "uuid overlap with rounds.json"
    newest = max(r["timestamp"] for r in payload.values())
    assert newest < cutoff, (newest, cutoff)

    with open(args.out, "w") as handle:
        json.dump(payload, handle)
    span = (newest - min(r["timestamp"] for r in payload.values())) / 60.0
    print("wrote %s: %d rounds, %.0f min span, ending %.0f min before rounds.json"
          % (args.out, len(payload), span, (cutoff - newest) / 60.0),
          file=sys.stderr)


if __name__ == "__main__":
    main()
