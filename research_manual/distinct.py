#!/usr/bin/env python3

import argparse
import collections
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DUMP = os.path.join(HERE, "rounds.json")
PREFIX_LEN = 8


def prefix(coldkey):
    if coldkey.startswith("our_"):
        return "our"
    assert len(coldkey) >= PREFIX_LEN, coldkey
    return coldkey[:PREFIX_LEN]


def load(path):
    with open(path) as handle:
        payload = json.load(handle)
    assert payload
    rows = []
    for round_id, rec in payload.items():
        answers = []
        for _uid, _hotkey, coldkey, clique in rec["answers"]:
            key = tuple(sorted(clique))
            if not key:
                continue
            answers.append((coldkey, key))
        if not answers:
            continue
        rows.append((rec["timestamp"], round_id, answers))
    rows.sort(key=lambda item: item[0])
    assert rows
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", default=DEFAULT_DUMP)
    args = parser.parse_args()
    rows = load(args.dump)
    labels = sorted(
        {prefix(coldkey) for _, _, answers in rows for coldkey, _ in answers},
        key=lambda name: (name != "our", name),
    )
    header = ["#", "we", "field"] + labels
    table = [header]
    for index, (_ts, _round_id, answers) in enumerate(rows, 1):
        omega = max(len(clique) for _, clique in answers)
        all_d = {clique for _, clique in answers if len(clique) == omega}
        by = collections.defaultdict(list)
        for coldkey, clique in answers:
            by[prefix(coldkey)].append(clique)
        our_d = {c for c in by.get("our", []) if len(c) == omega}
        cells = [str(index), str(len(our_d)), str(len(all_d))]
        for label in labels:
            cliques = by.get(label, [])
            n = len(cliques)
            if n == 0:
                cells.append("-")
                continue
            distinct = len({c for c in cliques if len(c) == omega})
            cells.append(f"{distinct}/{n}")
        table.append(cells)
    widths = [
        max(len(row[col]) for row in table) for col in range(len(header))
    ]
    for row in table:
        print("  ".join(cell.rjust(widths[i]) for i, cell in enumerate(row)))


if __name__ == "__main__":
    main()
