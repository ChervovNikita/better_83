#!/usr/bin/env python3

import argparse
import collections
import json
import os
import statistics

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
        scored = "scores" in rec
        if scored:
            assert len(rec["scores"]) == len(rec["answers"])
        answers = []
        for i, (_uid, _hotkey, coldkey, clique) in enumerate(rec["answers"]):
            score = float(rec["scores"][i]) if scored else None
            answers.append((coldkey, tuple(sorted(clique)), score))
        if not answers:
            continue
        rows.append((rec["timestamp"], round_id, answers))
    rows.sort(key=lambda item: item[0])
    assert rows
    return rows


def cell(distinct, n, scores):
    if not scores or scores[0] is None:
        return f"{distinct}/{n}"
    return f"{distinct}/{n} ({statistics.mean(scores):.4f})"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", default=DEFAULT_DUMP)
    parser.add_argument("--gaps", action="store_true")
    args = parser.parse_args()
    rows = load(args.dump)
    labels = sorted(
        {prefix(coldkey) for _, _, answers in rows for coldkey, _c, _s in answers},
        key=lambda name: (name != "our", name),
    )
    header = ["#", "we", "field"] + labels
    table = [header]
    for index, (_ts, _round_id, answers) in enumerate(rows, 1):
        nonempty = [clique for _, clique, _score in answers if clique]
        omega = max(len(clique) for clique in nonempty) if nonempty else 0
        all_d = {c for c in nonempty if omega and len(c) == omega}
        by = collections.defaultdict(list)
        for coldkey, clique, score in answers:
            by[prefix(coldkey)].append((clique, score))
        our_all = by.get("our", [])
        our_d = {c for c, _s in our_all if c and omega and len(c) == omega}
        if args.gaps:
            short_field = len(all_d) < 5
            short_ours = bool(our_all) and len(our_d) < len(our_all)
            if not short_field and not short_ours:
                continue
        if our_all:
            we_cell = cell(
                len(our_d),
                len(our_all),
                [score for _c, score in our_all],
            )
        else:
            we_cell = "-"
        cells = [str(index), we_cell, str(len(all_d))]
        for label in labels:
            rows_ck = by.get(label, [])
            n = len(rows_ck)
            if n == 0:
                cells.append("-")
                continue
            distinct = len({c for c, _s in rows_ck if c and omega and len(c) == omega})
            cells.append(cell(distinct, n, [score for _c, score in rows_ck]))
        table.append(cells)
    widths = [
        max(len(row[col]) for row in table) for col in range(len(header))
    ]
    for row in table:
        print("  ".join(cell.rjust(widths[i]) for i, cell in enumerate(row)))


if __name__ == "__main__":
    main()
