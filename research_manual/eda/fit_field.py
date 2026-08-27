#!/usr/bin/env python3
"""Predict how many OTHER answers reach omega, from what our own solver sees.

    .venv/bin/python research_manual/eda/fit_field.py --build --rounds 100
    .venv/bin/python research_manual/eda/fit_field.py --fit

Step 2 of the two-step scheme.  strategy.plan() needs `a_hat`: the number of
other answers at omega.  Nothing about the field is observable at solve time, so
it is predicted from features our own solver produces:

    omega     the max clique size we found
    n_top     distinct omega-cliques we hold      <- the dominant feature
    n_spare   distinct (omega-1)-cliques we hold
    n, time_limit, difficulty, n_others

--build runs the solver over the TUNING rounds and writes field_features.json.
It removes the N lowest-incentive non-immune miners first, exactly as
simulate.py's pick_victims does, because entering with N hotkeys deregisters
them -- the field we predict is the field that would remain, not the historical
one.

--fit reads that file and fits the model.  Fitting happens ONLY on tuning_data;
rounds.json stays untouched so the result is measurable there afterwards.
"""

import argparse
import collections
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
ROOT = os.path.dirname(PARENT)
for _p in (ROOT, PARENT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

TUNING = os.path.join(HERE, "tuning_data.json")
FEATURES = os.path.join(HERE, "field_features.json")
POOLS = os.path.join(HERE, "pools.json")

# Our pool is cached alongside the features so picker variants can be scored
# without re-solving: 100 rounds costs 21 min of GPU because every round now
# burns its full deadline. Only the top slice is kept -- an allocation never
# uses more than q (<= 13) cliques, and the exact n_top/n_spare counts are
# recorded separately.
POOL_CAP = 64
METAGRAPH = os.path.join(PARENT, "metagraph.json")
IMMUNITY_BLOCKS = 6000


def victims(meta, n):
    """The N lowest-incentive non-immune miners, same rule as simulate.py."""
    block = meta["block"]
    cand = [m for m in meta["miners"]
            if block - m["block_at_registration"] >= IMMUNITY_BLOCKS]
    assert len(cand) >= n, (len(cand), n)
    return {m["uid"] for m in cand[:n]}


def build(args):
    from CliqueAI.graph.codec import GraphCodec
    import fleet_solver_gpu as fg
    import gpu_lib

    with open(METAGRAPH) as handle:
        meta = json.load(handle)
    dropped = victims(meta, args.fleet)
    with open(args.dump) as handle:
        payload = json.load(handle)
    rows = sorted(payload.items(), key=lambda kv: kv[1]["timestamp"])[:args.rounds]

    codec = GraphCodec()
    out = []
    pools = {}
    for i, (rid, rec) in enumerate(rows, 1):
        # the field as it would be AFTER we displace the N worst
        field = [sorted(cl) for uid, _h, _c, cl in rec["answers"]
                 if cl and uid not in dropped]
        if not field:
            continue
        matrix = codec.decode_matrix(rec["encoded_matrix"])
        A = np.array(matrix, dtype=np.uint8)

        budget = rec["time_limit"] - 2.0
        champ = sorted(fg.fleet_solver._solve_one(A, budget * fg.CHAMPION_SHARE,
                                                  seed=1))
        with gpu_lib.GpuClique(A) as gpu:
            raw, _, _hits = gpu.harvest(budget * (1 - fg.CHAMPION_SHARE) - fg.RESERVE_S,
                                 seed=1, max_steps=fg.STEPS,
                                 boot_steps=fg.BOOT_STEPS, init_clique=champ,
                                 max_out=4096)
        pool = sorted({tuple(c) for c in raw if all(gpu_lib.verify(A, c))},
                      key=len, reverse=True)
        if not pool:
            continue
        omega_ours = len(pool[0])
        omega = max(omega_ours, max(len(c) for c in field))

        rec_out = {
            "uuid": rid,
            "n": rec["number_of_nodes"],
            "time_limit": rec["time_limit"],
            "difficulty": rec["difficulty"],
            "n_others": len(field),
            "omega_ours": omega_ours,
            "omega": omega,
            "n_top": sum(1 for c in pool if len(c) == omega_ours),
            "n_spare": sum(1 for c in pool if len(c) == omega_ours - 1),
            # targets
            "a": sum(1 for c in field if len(c) == omega),
            "b": sum(1 for c in field if len(c) == omega - 1),
            "field_distinct_at_omega": len({tuple(c) for c in field
                                            if len(c) == omega}),
        }
        top = [list(c) for c in pool if len(c) == omega_ours][:POOL_CAP]
        spare = [list(c) for c in pool if len(c) == omega_ours - 1][:POOL_CAP]
        pools[rid] = {"top": top, "spare": spare,
                      "field": [list(c) for c in field]}
        out.append(rec_out)
        print("  %3d/%d n=%3d omega=%3d n_top=%4d -> a=%3d b=%3d"
              % (i, len(rows), rec_out["n"], omega, rec_out["n_top"],
                 rec_out["a"], rec_out["b"]), flush=True)

    with open(args.out, "w") as handle:
        json.dump(out, handle, indent=1)
    with open(POOLS, "w") as handle:
        json.dump(pools, handle)
    print("wrote %s: %d rounds" % (args.out, len(out)), file=sys.stderr)
    print("wrote %s: %d rounds" % (POOLS, len(pools)), file=sys.stderr)


# --------------------------------------------------------------------- model

def features(row):
    """The vector the predictor sees. Everything here is known at solve time."""
    n_top = max(1, row["n_top"])
    return np.array([
        1.0,
        np.log(n_top),
        np.log(max(1, row["n_spare"] + 1)),
        row["omega_ours"] / 100.0,
        row["n"] / 1000.0,
        row["difficulty"],
        row["n_others"] / 100.0,
    ])


def fit(args):
    with open(args.out) as handle:
        rows = json.load(handle)
    assert rows, args.out
    X = np.stack([features(r) for r in rows])
    # fraction of other answers at omega, logit-ish target: bounded and
    # scale-free, so one big round cannot dominate the fit
    y = np.array([r["a"] / max(1.0, r["n_others"]) for r in rows])

    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = np.clip(X @ coef, 0.0, 1.0)
    a_hat = pred * np.array([r["n_others"] for r in rows])
    a_true = np.array([r["a"] for r in rows])

    base = np.full_like(a_true, np.mean(a_true), dtype=float)
    def mae(p):
        return float(np.mean(np.abs(p - a_true)))
    print("rounds            %d" % len(rows))
    print("mean a            %.1f  (of %.1f other answers)"
          % (np.mean(a_true), np.mean([r["n_others"] for r in rows])))
    print("MAE, predict-mean %.2f" % mae(base))
    print("MAE, model        %.2f" % mae(a_hat))
    print()
    print("coefficients (target = a / n_others):")
    for name, c in zip(("const", "log n_top", "log n_spare", "omega/100",
                        "n/1000", "difficulty", "n_others/100"), coef):
        print("  %-14s %+.4f" % (name, c))

    # how much of it is n_top alone
    Xt = np.stack([[1.0, np.log(max(1, r["n_top"]))] for r in rows])
    ct, *_ = np.linalg.lstsq(Xt, y, rcond=None)
    at = np.clip(Xt @ ct, 0, 1) * np.array([r["n_others"] for r in rows])
    print()
    print("MAE, n_top only   %.2f" % mae(at))
    np.save(os.path.join(HERE, "field_coef.npy"), coef)
    print("saved field_coef.npy")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--dump", default=TUNING)
    ap.add_argument("--out", default=FEATURES)
    ap.add_argument("--rounds", type=int, default=100)
    ap.add_argument("--fleet", type=int, default=40)
    args = ap.parse_args()
    assert args.build or args.fit, "pass --build or --fit"
    if args.build:
        build(args)
    if args.fit:
        fit(args)


if __name__ == "__main__":
    main()
