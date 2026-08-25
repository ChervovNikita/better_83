#!/usr/bin/env python3

import json
import os
import sys

import wandb

PROJECT = "toptensor-ai/CliqueAI"
VERSION = "0.0.17"
N_ROUNDS = 1000
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
DEST = os.path.join(HERE, "rounds.json")


def check_wandb_version():
    major, minor = (int(x) for x in wandb.__version__.split(".")[:2])
    assert (major, minor) < (0, 20), wandb.__version__


def fetch_rows(api):
    runs = list(
        api.runs(
            PROJECT,
            filters={"config.version": {"$in": [VERSION]}},
            per_page=100,
        )
    )
    assert runs
    rows = []
    seen = set()
    for run in runs:
        head = int(run.summary["_step"])
        lo = max(0, head - N_ROUNDS)
        history = run.scan_history(
            keys=KEYS,
            min_step=lo,
            max_step=head + 1,
            page_size=20,
        )
        for row in history:
            if "uuid" not in row:
                continue
            round_id = row["uuid"]
            if not round_id or round_id in seen:
                continue
            seen.add(round_id)
            encoded = row["encoded_matrix"]
            assert encoded
            answers = list(
                zip(
                    row["miner_uids"],
                    row["miner_hotkeys"],
                    row["miner_coldkeys"],
                    row["miner_ans"],
                    strict=True,
                )
            )
            assert answers
            rec = {
                "timestamp": row["timestamp"],
                "difficulty": row["difficulty"],
                "time_limit": row["time_limit"],
                "number_of_nodes": row["number_of_nodes"],
                "encoded_matrix": encoded,
                "answers": answers,
            }
            rows.append((rec["timestamp"], round_id, rec))
    return rows


def main():
    check_wandb_version()
    api = wandb.Api(timeout=180)
    rows = fetch_rows(api)
    rows.sort(key=lambda item: item[0])
    assert len(rows) >= N_ROUNDS, len(rows)
    latest = rows[-N_ROUNDS:]
    payload = {round_id: rec for _, round_id, rec in latest}
    assert len(payload) == N_ROUNDS
    with open(DEST, "w") as handle:
        json.dump(payload, handle)
    print(DEST, file=sys.stderr)


if __name__ == "__main__":
    main()
