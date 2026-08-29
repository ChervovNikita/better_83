#!/usr/bin/env python3

import argparse
import collections
import json
import os
import statistics
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from tqdm import tqdm

from CliqueAI.graph.codec import GraphCodec
from CliqueAI.scoring.clique_scoring import CliqueScoreCalculator
import solver

IMMUNITY_BLOCKS = 6000
NETWORK_S = 2.0
OUR_COLDKEY = "our_coldkey"
MINER_ALPHA_DAY = 2951.6
POWER_TARGET_TOP_HALF_SHARE = 0.80
POWER_MAX_GAMMA = 32.0
ROUNDS_PATH = os.path.join(HERE, "rounds.json")
METAGRAPH_PATH = os.path.join(HERE, "metagraph.json")
OUT_PATH = os.path.join(HERE, "sim_out.json")


class Graph:
    def __init__(self, uuid, number_of_nodes, adjacency_list):
        self.uuid = uuid
        self.label = "general"
        self.number_of_nodes = number_of_nodes
        self.adjacency_list = adjacency_list


def load_rounds(path, n_rounds, only=None):
    """The first n_rounds by timestamp, or those named in `only`.

    `only` is a file of round ids, one per line: it selects a stratum decided
    outside this script rather than a deadline slice, since the deadline
    proxies graph size and slicing on it stratifies by the variable under test.
    """
    with open(path) as handle:
        payload = json.load(handle)
    rows = []
    for round_id, rec in payload.items():
        assert rec["encoded_matrix"]
        assert rec["answers"]
        rows.append((rec["timestamp"], round_id, rec))
    rows.sort(key=lambda item: item[0])
    assert rows
    if only:
        with open(only) as handle:
            wanted = {line.strip() for line in handle if line.strip()}
        rows = [row for row in rows if row[1] in wanted]
        assert len(rows) == len(wanted), (len(rows), len(wanted))
        return rows
    assert n_rounds <= len(rows), (n_rounds, len(rows))
    return rows[:n_rounds]


def pick_victims(meta, n):
    block = meta["block"]
    candidates = [
        miner for miner in meta["miners"]
        if block - miner["block_at_registration"] >= IMMUNITY_BLOCKS
    ]
    assert candidates == sorted(
        candidates, key=lambda miner: (miner["incentive"], miner["uid"])
    )
    assert len(candidates) >= n, (len(candidates), n)
    return candidates[:n]


def our_name(index):
    return f"our_hotkey_{index:03d}"


def percentile(value, values):
    return 100.0 * sum(1 for item in values if item < value) / len(values)


def top_half_share(weights, rank_scores):
    total = float(np.sum(weights))
    if total <= 0:
        return 0.0
    order = np.argsort(-rank_scores)
    top = order[: len(order) // 2]
    return float(np.sum(weights[top]) / total)


def power_weight(sigmoid, rank_scores):
    active = sigmoid > 0
    if not np.any(active):
        return np.zeros_like(sigmoid, dtype=np.float64)

    def transform(gamma):
        weights = np.zeros_like(sigmoid, dtype=np.float64)
        weights[active] = np.power(sigmoid[active], gamma)
        return weights

    lo = 0.0
    hi = min(1.0, POWER_MAX_GAMMA)
    while (
        top_half_share(transform(hi), rank_scores) < POWER_TARGET_TOP_HALF_SHARE
        and hi < POWER_MAX_GAMMA
    ):
        hi = min(hi * 2.0, POWER_MAX_GAMMA)
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if top_half_share(transform(mid), rank_scores) < POWER_TARGET_TOP_HALF_SHARE:
            lo = mid
        else:
            hi = mid
    return transform(hi)


def validator_weights(scores):
    scores = np.asarray(scores, dtype=np.float64)
    assert scores.size
    span = float(np.max(scores) - np.min(scores))
    if span == 0:
        return np.zeros_like(scores)
    normalized = (scores - np.min(scores)) / span
    zero_mask = normalized == 0
    live = normalized[~zero_mask]
    if live.size == 0:
        return np.zeros_like(scores)
    midpoint = np.median(live)
    steepness = max(np.percentile(live, 75) - np.percentile(live, 25), 0.1)
    sigmoid = 1.0 / (1.0 + np.exp(-(normalized - midpoint) / steepness))
    sigmoid[zero_mask] = 0.0
    weights = power_weight(sigmoid, normalized)
    total = float(np.sum(weights))
    assert total > 0
    return weights / total


def our_coldkey_median(scores_by_hotkey, our_hotkeys):
    means = [
        statistics.mean(scores_by_hotkey[hotkey])
        for hotkey in our_hotkeys
        if scores_by_hotkey[hotkey]
    ]
    if not means:
        return None
    return statistics.median(means)


def print_percentile_hist(percentiles, n_unqueried):
    buckets = [0] * 10
    for pct in percentiles:
        buckets[min(int(pct // 10), 9)] += 1
    print("percentile\tn")
    for index, count in enumerate(buckets):
        low = index * 10
        high = 100 if index == 9 else (index + 1) * 10
        print(f"{low}-{high}\t{count}\t{'#' * count}")
    print(f"unqueried\t{n_unqueried}")


def report(scores_by_hotkey, coldkey_of, our_hotkeys, verbose):
    ours = set(our_hotkeys)
    means = {
        hotkey: statistics.mean(scores)
        for hotkey, scores in scores_by_hotkey.items()
        if scores
    }
    field = {
        hotkey: mean
        for hotkey, mean in means.items()
        if hotkey not in ours
    }
    assert field
    if verbose:
        ranked = sorted(field.items(), key=lambda item: item[1], reverse=True)
        print("hotkey\tmean")
        for hotkey, mean in ranked:
            print(f"{hotkey}\t{mean:.4f}")

    by_cold = collections.defaultdict(list)
    for hotkey, mean in field.items():
        by_cold[coldkey_of[hotkey]].append(mean)
    our_means = [means[hotkey] for hotkey in our_hotkeys if hotkey in means]
    if our_means:
        by_cold[OUR_COLDKEY].extend(our_means)
    cold_medians = sorted(
        (
            (statistics.median(values), coldkey, len(values))
            for coldkey, values in by_cold.items()
        ),
        reverse=True,
    )
    ranked_hotkeys = list(means)
    weights = validator_weights([means[hotkey] for hotkey in ranked_hotkeys])
    share_of = {
        hotkey: float(weight)
        for hotkey, weight in zip(ranked_hotkeys, weights, strict=True)
    }
    by_cold_share = collections.defaultdict(float)
    for hotkey, share in share_of.items():
        if hotkey in ours:
            by_cold_share[OUR_COLDKEY] += share
        else:
            by_cold_share[coldkey_of[hotkey]] += share
    print("coldkey\tn_hotkeys\tmedian\talpha/day")
    for median, coldkey, count in cold_medians:
        alpha = by_cold_share[coldkey] * MINER_ALPHA_DAY
        print(f"{coldkey}\t{count}\t{median:.4f}\t{alpha:.1f}")

    field_means = list(field.values())
    print(f"median_all_miners\t{statistics.median(field_means):.4f}")
    our_share = by_cold_share.get(OUR_COLDKEY, 0.0)
    print(f"expected_share\t{our_share:.4%}")
    print(f"expected_alpha/day\t{our_share * MINER_ALPHA_DAY:.1f}")

    our_pct = []
    n_unqueried = 0
    for hotkey in our_hotkeys:
        if hotkey not in means:
            n_unqueried += 1
            continue
        our_pct.append(percentile(means[hotkey], field_means))
    assert our_pct
    print_percentile_hist(our_pct, n_unqueried)
    print(
        "our_percentile\t"
        f"min={min(our_pct):.2f}\t"
        f"median={statistics.median(our_pct):.2f}\t"
        f"max={max(our_pct):.2f}\t"
        f"scored={len(our_pct)}/{len(our_hotkeys)}"
    )


def run(rows, victims):
    taken = {miner["uid"] for miner in victims}
    ours = {
        miner["uid"]: our_name(index)
        for index, miner in enumerate(victims)
    }
    codec = GraphCodec()
    out = {}
    scores_by_hotkey = collections.defaultdict(list)
    coldkey_of = {}
    n_late = 0
    n_queried = 0
    our_hotkey_names = [our_name(index) for index in range(len(victims))]
    bar = tqdm(rows, total=len(rows), file=sys.stderr)
    for _, round_id, rec in bar:
        historical = []
        queried = set()
        for uid, hotkey, coldkey, clique in rec["answers"]:
            if uid in taken:
                queried.add(uid)
                continue
            historical.append((uid, hotkey, coldkey, list(clique)))
            coldkey_of[hotkey] = coldkey
        our_uids = [miner["uid"] for miner in victims if miner["uid"] in queried]
        our_hotkeys = [ours[uid] for uid in our_uids]
        matrix = codec.decode_matrix(rec["encoded_matrix"])
        assert rec["number_of_nodes"] == len(matrix)
        if our_hotkeys:
            n_queried += 1
            started = time.monotonic()
            answers = solver.solve(
                our_hotkeys, matrix, rec["time_limit"], round_id
            )
            elapsed = time.monotonic() - started + NETWORK_S
            if elapsed > rec["time_limit"]:
                answers = [[] for _ in our_hotkeys]
                n_late += 1
            our_rows = [
                (uid, ours[uid], OUR_COLDKEY, list(answer))
                for uid, answer in zip(our_uids, answers, strict=True)
            ]
        else:
            our_rows = []
        combined = historical + our_rows
        assert combined
        adj = codec.matrix_to_list(matrix)
        graph = Graph(round_id, rec["number_of_nodes"], adj)
        responses = [clique for _, _, _, clique in combined]
        calc = CliqueScoreCalculator(
            graph=graph,
            difficulty=rec["difficulty"],
            responses=responses,
        )
        *_, rewards = calc.get_scores()
        assert len(rewards) == len(combined)
        answers = []
        scores = []
        for (uid, hotkey, coldkey, clique), score in zip(
            combined, rewards, strict=True
        ):
            answers.append((uid, hotkey, coldkey, clique))
            scores.append(float(score))
            scores_by_hotkey[hotkey].append(float(score))
            coldkey_of[hotkey] = coldkey
        out[round_id] = {
            "timestamp": rec["timestamp"],
            "difficulty": rec["difficulty"],
            "time_limit": rec["time_limit"],
            "number_of_nodes": rec["number_of_nodes"],
            "encoded_matrix": rec["encoded_matrix"],
            "answers": answers,
            "scores": scores,
        }
        median = our_coldkey_median(scores_by_hotkey, our_hotkey_names)
        if median is None:
            bar.set_postfix_str("median=n/a")
        else:
            bar.set_postfix_str(f"median={median:.4f}")
    return out, scores_by_hotkey, coldkey_of, n_queried, n_late


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-N", type=int, required=True)
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--only", default="",
                        help="file of round ids to run instead of the first --rounds")
    parser.add_argument("--dump", default=ROUNDS_PATH)
    parser.add_argument("--metagraph", default=METAGRAPH_PATH)
    parser.add_argument("--out", default=OUT_PATH)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    assert args.N > 0
    assert args.rounds > 0
    with open(args.metagraph) as handle:
        meta = json.load(handle)
    assert meta["miners"] == sorted(
        meta["miners"], key=lambda miner: (miner["incentive"], miner["uid"])
    )
    victims = pick_victims(meta, args.N)
    rows = load_rounds(args.dump, args.rounds, args.only)
    out, scores_by_hotkey, coldkey_of, n_queried, n_late = run(
        rows, victims
    )
    with open(args.out, "w") as handle:
        json.dump(out, handle)
    our_hotkeys = [our_name(index) for index in range(args.N)]
    print(f"victims {[miner['uid'] for miner in victims]}")
    print(f"wrote {args.out} {len(out)} rounds")
    print(f"queried {n_queried} late {n_late} network_s {NETWORK_S}")
    report(scores_by_hotkey, coldkey_of, our_hotkeys, args.verbose)


if __name__ == "__main__":
    main()
