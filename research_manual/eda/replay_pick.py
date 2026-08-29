#!/usr/bin/env python3
"""Replay pickers against a saved solver pool, with no GPU and no re-solve.

    .venv/bin/python research_manual/eda/replay_pick.py \
        --pool research_manual/pool_n20.jsonl \
        --dump research_manual/sim_out_n20_value2.json \
        --pickers value,static

The pool file comes from a simulator run with SN83_POOL_DUMP set; it carries
every clique the solver found, its basin hit count, and the true supply counts.
Rounds are joined to the dump by uuid, our answers are replaced by the picker's,
and the round is rescored.

Scoring is fleet_sim.score_round -- the validator's formula, with `valid`
REQUIRED -- and every answer a picker emits is put through
fleet_sim.validate_cliques first.  A clique the validator would reject scores
size * 0, so a harness that skips the test measures a reward nobody is paid.
"""

import argparse
import collections
import importlib
import inspect
import json
import os
import statistics
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for _p in (ROOT, os.path.join(ROOT, "research"), os.path.dirname(HERE), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fleet_sim import score_round

OUR_COLDKEY = "our_coldkey"
PICKERS = {"value": "pick_value", "static": "pick_static", "legacy": "fleet_pick"}


def load_pools(path):
    pools = {}
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                rec = json.loads(line)
                pools[rec["uuid"]] = rec
    assert pools, path
    return pools


def load_picker(name):
    mod = importlib.import_module(PICKERS[name])
    picker = mod.picker
    params = inspect.signature(picker).parameters
    return picker, {
        "n_nodes": "n_nodes" in params,
        "n_top_true": "n_top_true" in params,
        "hits": "hits" in params,
    }


def validate(rec, cliques):
    """The validator's clique+maximality test, from the round's own matrix."""
    A = np.array(rec["_matrix"], dtype=np.uint8)
    n = A.shape[0]
    out = []
    for c in cliques:
        S = list(c)
        ok = bool(S) and len(set(S)) == len(S) and min(S) >= 0 and max(S) < n
        if ok:
            idx = np.array(S)
            ok = A[np.ix_(idx, idx)].sum() == len(S) * (len(S) - 1)
            if ok:
                inC = np.zeros(n, dtype=bool)
                inC[idx] = True
                ok = not np.any((A[idx].sum(axis=0) == len(S)) & (~inC))
        out.append(bool(ok))
    return out


def replay(round_rec, pool_rec, picker, wants, check):
    """Our per-hotkey rewards when `picker` answers this round."""
    answers = round_rec["answers"]
    ours = [i for i, a in enumerate(answers) if a[2] == OUR_COLDKEY]
    assert ours
    hotkeys = [answers[i][1] for i in ours]
    kwargs = {}
    if wants["n_nodes"]:
        kwargs["n_nodes"] = pool_rec["n_nodes"]
    if wants["n_top_true"]:
        kwargs["n_top_true"] = pool_rec["n_top_true"]
        kwargs["n_spare_true"] = pool_rec["n_spare_true"]
    if wants["hits"]:
        kwargs["hits"] = pool_rec["hits"]
    picked = picker(pool_rec["pool"], pool_rec["uuid"], list(hotkeys), **kwargs)
    assert len(picked) == len(hotkeys), (len(picked), len(hotkeys))

    keys = [tuple(sorted(a[3])) for a in answers]
    sizes = [len(a[3]) for a in answers]
    valid = [1 if a[3] else 0 for a in answers]
    ok = validate(round_rec, picked) if check else [True] * len(picked)
    for j, i in enumerate(ours):
        keys[i] = tuple(sorted(picked[j]))
        sizes[i] = len(picked[j])
        valid[i] = 1 if (picked[j] and ok[j]) else 0
    scores = score_round(sizes, valid, keys, round_rec["difficulty"])
    return [float(scores[i]) for i in ours], sum(1 for v in ok if not v), keys, ours


def sign_test(a, b):
    """Paired sign test over CHANGED rounds only."""
    up = sum(1 for x, y in zip(a, b) if y > x + 1e-12)
    dn = sum(1 for x, y in zip(a, b) if y < x - 1e-12)
    n = up + dn
    if n == 0:
        return up, dn, 1.0
    from math import comb
    k = min(up, dn)
    p = min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / (2.0 ** n))
    return up, dn, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--pickers", default="value")
    ap.add_argument("--baseline", default="", help="score the dump's own answers")
    ap.add_argument("--no-validate", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    pools = load_pools(args.pool)
    with open(args.dump) as handle:
        rounds = json.load(handle)
    from CliqueAI.graph.codec import GraphCodec
    codec = GraphCodec()

    names = [n.strip() for n in args.pickers.split(",") if n.strip()]
    for n in names:
        assert n in PICKERS, n
    loaded = {n: load_picker(n) for n in names}

    per = collections.defaultdict(list)
    shipped = []
    invalid = collections.Counter()
    dup = collections.Counter()
    used = 0
    for uuid, rec in rounds.items():
        pool_rec = pools.get(uuid)
        if pool_rec is None:
            continue
        ours = [i for i, a in enumerate(rec["answers"]) if a[2] == OUR_COLDKEY]
        if not ours:
            continue
        if not args.no_validate:
            rec["_matrix"] = codec.decode_matrix(rec["encoded_matrix"])
        used += 1
        shipped.append(statistics.mean(rec["scores"][i] for i in ours))
        for name in names:
            picker, wants = loaded[name]
            got, bad, keys, idx = replay(rec, pool_rec, picker, wants,
                                         not args.no_validate)
            per[name].append(statistics.mean(got))
            invalid[name] += bad
            sub = [keys[i] for i in idx]
            dup[name] += len(sub) - len(set(sub))
        rec.pop("_matrix", None)
        if args.limit and used >= args.limit:
            break

    assert used, "no uuid in the pool file matched the dump"
    print("rounds replayed: %d" % used)
    print("%-10s %10s %10s %8s %8s" % ("picker", "mean", "vs shipped",
                                       "dup", "invalid"))
    print("%-10s %10.4f %10s %8s %8s"
          % ("shipped", statistics.mean(shipped), "-", "-", "-"))
    for name in names:
        xs = per[name]
        print("%-10s %10.4f %+10.4f %8d %8d"
              % (name, statistics.mean(xs),
                 statistics.mean(xs) - statistics.mean(shipped),
                 dup[name], invalid[name]))
    for name in names:
        up, dn, p = sign_test(shipped, per[name])
        print("  %s vs shipped: better %d, worse %d, sign test p=%.4f"
              % (name, up, dn, p))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
