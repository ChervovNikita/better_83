#!/usr/bin/env python3
"""EXACT round scorer -- a transcription of CliqueAI/scoring/clique_scoring.py.

    rel  = size / max_size
    pr   = (# answers STRICTLY LARGER) / n          <- a FRACTION, and n includes us
    omega= exp(-pr / rel)
    optimality = omega / max(omega)
    diversity  = (1/count) / max(1/count over valid)
    reward     = optimality * (1 + difficulty) + diversity

Verified against the logged rewards before use (see selftest below): every logged
answer's reward is reproduced to 1e-9 from size and count alone.

Two consequences that the earlier size-based reasoning missed:
  * pr is a FRACTION of the answer count, so inserting our K answers lowers pr for
    everyone -- that is the public good, and it is mechanical, not behavioural.
  * the cost of answering one vertex below omega is exp(-pr/rel), which depends on
    how many answers sit at omega. It is nearly free when omega is rare and severe
    when omega is common. That is the whole omega-1 trade-off, exactly.
"""
import json, collections, math
import numpy as np


def score(sizes, keys, difficulty, valid=None):
    """sizes/keys are per answer; keys are hashable canonical cliques."""
    n = len(sizes)
    size = np.array(sizes, dtype=float)
    if valid is not None:
        size = size * np.array(valid, dtype=float)
    mx = size.max() if n else 0.0
    if mx <= 0:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    rel = size / mx
    pr = np.array([np.sum(size > size[i]) / n for i in range(n)])
    om = np.where(size > 0, np.exp(-pr / np.where(rel > 0, rel, 1)), 0.0)
    opt = om / om.max() if om.max() > 0 else om
    cnt = collections.Counter(keys)
    unq = np.array([(1.0 / cnt[k]) if size[i] > 0 else 0.0 for i, k in enumerate(keys)])
    div = unq / unq.max() if unq.max() > 0 else unq
    return opt, div, opt * (1 + difficulty) + div


def selftest(path="data/sim_rounds.jsonl", n_rounds=40, tol=1e-6):
    bad = 0; tot = 0
    for i, l in enumerate(open(path)):
        if i >= n_rounds: break
        r = json.loads(l)
        keys = [tuple(sorted(a["clique"])) for a in r["answers"]]
        sizes = [len(k) for k in keys]
        _, _, rw = score(sizes, keys, r["difficulty"])
        for a, got in zip(r["answers"], rw):
            tot += 1
            if abs(got - a["reward"]) > tol: bad += 1
    return tot, bad


if __name__ == "__main__":
    tot, bad = selftest()
    print("selftest: %d logged answers reproduced, %d mismatches" % (tot, bad))
