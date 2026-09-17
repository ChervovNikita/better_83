#!/usr/bin/env python3

import argparse
import collections
import json
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DUMP = os.path.join(HERE, "rounds.json")
DEFAULT_META = os.path.join(HERE, "metagraph.json")
MERGED = "merged"
PREFIX_LEN = 8


def load_rounds(path):
    with open(path) as handle:
        payload = json.load(handle)
    assert payload
    rows = []
    for rec in payload.values():
        scored = "scores" in rec
        if scored:
            assert len(rec["scores"]) == len(rec["answers"])
        answers = []
        for i, (_uid, hotkey, coldkey, clique) in enumerate(rec["answers"]):
            key = tuple(sorted(clique))
            if not key:
                continue
            score = float(rec["scores"][i]) if scored else None
            answers.append((hotkey, coldkey, key, score))
        if len(answers) >= 2:
            rows.append(answers)
    assert rows
    return rows


def chain_sizes(meta_path):
    with open(meta_path) as handle:
        meta = json.load(handle)
    sizes = collections.Counter()
    for miner in meta["miners"]:
        sizes[miner["coldkey"]] += 1
    assert sizes
    return sizes


def parse_merges(specs):
    """--merge LABEL=prefix,prefix,... : coldkeys of one operator, one row."""
    out = {}
    for spec in specs or ():
        label, _, keys = spec.partition("=")
        assert keys, spec
        for key in keys.split(","):
            assert key
            out[key] = label
    return out


def build_entities(rows, min_hotkeys, meta_path, merges=None):
    merges = merges or {}
    seen = collections.defaultdict(set)
    for answers in rows:
        for hotkey, coldkey, _clique, _score in answers:
            assert coldkey
            seen[coldkey].add(hotkey)
    sizes = dict(chain_sizes(meta_path))
    for coldkey, keys in seen.items():
        if coldkey not in sizes:
            sizes[coldkey] = len(keys)

    def group(coldkey):
        return merges.get(coldkey[:PREFIX_LEN], coldkey)

    # an operator's hotkeys count together, so a merged entity is sized and
    # thresholded as the one miner it actually is
    gsize = collections.Counter()
    for coldkey, count in sizes.items():
        gsize[group(coldkey)] += count
    gseen = {group(coldkey) for coldkey in seen}
    named = sorted(
        (name for name, count in gsize.items()
         if count >= min_hotkeys and name in gseen),
        key=lambda name: (-gsize[name], name),
    )
    named_set = set(named)
    of = {
        coldkey: group(coldkey) if group(coldkey) in named_set else MERGED
        for coldkey in seen
    }
    sizes = dict(sizes)
    sizes.update(gsize)
    names = list(named)
    if any(label == MERGED for label in of.values()):
        names.append(MERGED)
    members = collections.defaultdict(list)
    for coldkey, label in of.items():
        members[label].append(coldkey)
    for label in list(members):
        if label in merges.values():
            sizes[label] = gsize[label]
    for label in members:
        members[label].sort()
    return of, names, sizes, members


def entity_scores(rows, of):
    by_hotkey = collections.defaultdict(list)
    cold_of = {}
    for answers in rows:
        for hotkey, coldkey, _clique, score in answers:
            if score is None:
                return None
            by_hotkey[hotkey].append(score)
            cold_of[hotkey] = coldkey
    means = {hotkey: statistics.mean(xs) for hotkey, xs in by_hotkey.items()}
    by_entity = collections.defaultdict(list)
    for hotkey, mean in means.items():
        by_entity[of[cold_of[hotkey]]].append(mean)
    return {name: statistics.median(xs) for name, xs in by_entity.items() if xs}


def tally(rows, of, names):
    index = {name: i for i, name in enumerate(names)}
    n = len(names)
    obs = [[0] * n for _ in range(n)]
    tot = [[0] * n for _ in range(n)]
    for answers in rows:
        labeled = [
            (index[of[coldkey]], clique)
            for _, coldkey, clique, _score in answers
        ]
        for i, (left, clique_i) in enumerate(labeled):
            for right, clique_j in labeled[i + 1:]:
                a, b = (left, right) if left <= right else (right, left)
                tot[a][b] += 1
                if clique_i == clique_j:
                    obs[a][b] += 1
    for i in range(n):
        for j in range(i + 1, n):
            obs[j][i] = obs[i][j]
            tot[j][i] = tot[i][j]
    return obs, tot


def rate(hit, count):
    return (100.0 * hit / count) if count else 0.0


def prefix(coldkey):
    if coldkey == MERGED or coldkey.startswith("our_") or len(coldkey) < PREFIX_LEN:
        return coldkey
    assert len(coldkey) >= PREFIX_LEN, coldkey
    return coldkey[:PREFIX_LEN]


def axis_label(name, n_hotkeys, score):
    if score is None:
        return f"{prefix(name)}\n{n_hotkeys}hk"
    return f"{prefix(name)}\n{n_hotkeys}hk {score:.2f}"


def draw_matrix(labels, rates, dest, vmax):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(labels)
    assert n > 0
    fig, ax = plt.subplots(figsize=(max(7, 1.1 * n + 2), max(6, 1.1 * n + 2)))
    image = ax.imshow(rates, cmap="YlOrRd", vmin=0.0, vmax=vmax)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="collision %")
    ax.set_xticks(range(n), labels, rotation=45, ha="right")
    ax.set_yticks(range(n), labels)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{rates[i][j]:.1f}", ha="center", va="center", fontsize=8)
    ax.set_title("entity collision %")
    fig.tight_layout()
    fig.savefig(dest, dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", default=DEFAULT_DUMP)
    parser.add_argument("--metagraph", default=DEFAULT_META)
    parser.add_argument("--out", default=os.path.join(HERE, "collisions.png"))
    parser.add_argument("--min-hotkeys", type=int, default=10)
    parser.add_argument("--vmax", type=float, default=10.0)
    parser.add_argument("--merge", action="append",
                        help="LABEL=prefix,prefix,... one row per operator")
    args = parser.parse_args()
    assert args.min_hotkeys > 0
    assert args.vmax > 0
    rows = load_rounds(args.dump)
    of, names, sizes, members = build_entities(
        rows, args.min_hotkeys, args.metagraph, parse_merges(args.merge)
    )
    scores = entity_scores(rows, of)
    obs, tot = tally(rows, of, names)
    rates = [
        [rate(obs[i][j], tot[i][j]) for j in range(len(names))]
        for i in range(len(names))
    ]
    labels = []
    print("entity\tn_hotkeys\tscore\tcoldkeys")
    for name in names:
        n_hotkeys = (sizes[name] if name in members and name not in members[name]
                     and len(members[name]) > 1 and name != MERGED
                     else sum(sizes[ck] for ck in members[name]))
        score = None if scores is None else scores[name]
        labels.append(axis_label(name, n_hotkeys, score))
        score_s = "" if score is None else f"{score:.4f}"
        print(
            f"{prefix(name)}\t{n_hotkeys}\t{score_s}\t"
            f"{','.join(prefix(ck) for ck in members[name])}"
        )
    draw_matrix(labels, rates, args.out, args.vmax)
    print(args.out)


if __name__ == "__main__":
    main()
