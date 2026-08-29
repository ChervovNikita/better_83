"""Sweeps the fleet split over the solved rounds."""

import json
import random

import responders
import rounds as rounds_module
import strategies
from scoring import score


def run_split(g, o, strategy, responder, seed, all_rounds):
    """Plays every round at one fleet split."""
    reply_rng = random.Random(seed)
    commit = strategies.REGISTRY[strategy]
    reply = responders.REGISTRY[responder]
    means_a, means_b = [], []
    for rnd in all_rounds:
        q_a, q_b = rounds_module.split_queries(rnd, g, o)
        rnd.q_a, rnd.q_b = q_a, q_b
        rnd.fleet_a, rnd.fleet_b = g, o
        board = commit(rnd, q_a, score)
        assert board
        _trial, mean_a, mean_b = reply(board, rnd, q_b, reply_rng)
        means_a.append(mean_a)
        means_b.append(mean_b)
    return means_a, means_b


def sweep(n_total, strategy, responder, all_rounds, seed=0, step=1,
          progress=None):
    """Runs every fleet split from a bare majority to all but one hotkey."""
    out = []
    for index, g in enumerate(range(n_total // 2 + 1, n_total, step)):
        o = n_total - g
        means_a, means_b = run_split(g, o, strategy, responder, seed + index,
                                     all_rounds)
        out.append({
            "g": g,
            "o": o,
            "a": means_a,
            "b": means_b,
            "a_mean": sum(means_a) / len(means_a),
            "b_mean": sum(means_b) / len(means_b),
            "rounds": len(means_a),
        })
        if progress:
            progress(out[-1])
    return out


def save(result, path):
    """Writes a sweep result to `path` as JSON."""
    with open(path, "w") as handle:
        json.dump(result, handle)
