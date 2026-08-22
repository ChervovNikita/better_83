#!/usr/bin/env python3
"""Paired yield A/B: distinct optima per round, same rounds, same total budget.

Yield is far lower-variance than reward, so a few dozen rounds settle which solver
feeds an N=40 fleet. Reward needs hundreds of rounds; yield does not.

Guard: a config that wins yield but loses BEST SIZE is disqualified. A size miss
costs ~0.9 reward, which no amount of extra optima repays.
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from CliqueAI.graph.codec import GraphCodec
from _common import DATA_DIR

N_ROUNDS = int(os.environ.get("AB_N", "40"))
K = int(os.environ.get("AB_K", "40"))
THREADS = int(os.environ.get("AB_T", "14"))
NFLEET = int(os.environ.get("AB_FLEET", "40"))

rs = [json.loads(l) for l in open(os.path.join(DATA_DIR, "sim_rounds.jsonl")) if l.strip()]
rs = [r for r in rs if r.get("answers") and r.get("timestamp")]
rs.sort(key=lambda r: r["timestamp"])
recs = rs[-216:][:N_ROUNDS]

import fleetsolver, relabel_solver

ARMS = [("baseline", lambda A, tl: fleetsolver.solve_many(A, tl, K, threads=THREADS), {}),
        ("relabel R=3", lambda A, tl: relabel_solver.solve_many(A, tl, K), {"SN83_RELABELS": "3"}),
        ("relabel R=6", lambda A, tl: relabel_solver.solve_many(A, tl, K), {"SN83_RELABELS": "6"}),
        ("relabel R=12", lambda A, tl: relabel_solver.solve_many(A, tl, K), {"SN83_RELABELS": "12"})]

os.environ["SN83_THREADS"] = str(THREADS)
res = {name: {"y": [], "hit": 0, "t": 0.0} for name, _, _ in ARMS}
for i, r in enumerate(recs, 1):
    A = np.ascontiguousarray(np.array(GraphCodec().decode_matrix(r["matrix_b92"]), dtype=np.uint8))
    tl = max(0.5, r["time_limit"] - 2.0)
    for name, fn, env in ARMS:
        os.environ.update(env); os.environ["SN83_THREADS"] = str(THREADS)
        t0 = time.time(); cl = fn(A, tl); el = time.time() - t0
        res[name]["t"] += el
        if not cl:
            res[name]["y"].append(0); continue
        mx = max(len(c) for c in cl)
        res[name]["y"].append(len({tuple(sorted(c)) for c in cl if len(c) == mx}))
        if mx >= r["best_size"]: res[name]["hit"] += 1
    if i % 10 == 0: print(f"  {i}/{len(recs)}", flush=True)

n = len(recs)
print(f"\n=== paired yield A/B, {n} rounds, k={K}, {THREADS} threads, fleet N={NFLEET} ===")
print(f"{'arm':>14} {'mean':>7} {'median':>7} {'served/40':>10} {'all-40':>8} {'best size':>10} {'s/round':>8}")
base = None
for name, _, _ in ARMS:
    y = np.array(res[name]["y"]); srv = np.minimum(y, NFLEET)
    if base is None: base = y
    print(f"{name:>14} {y.mean():>7.1f} {np.median(y):>7.0f} "
          f"{srv.mean():>10.1f} {100*(y>=NFLEET).mean():>7.1f}% "
          f"{res[name]['hit']:>6}/{n:<3} {res[name]['t']/n:>8.1f}")
print(f"\npaired vs baseline (same rounds):")
for name, _, _ in ARMS[1:]:
    d = np.array(res[name]["y"]) - base
    se = d.std(ddof=1)/np.sqrt(len(d))
    print(f"  {name:>14}  {d.mean():+7.1f} optima/round  t={d.mean()/se if se>0 else 0:+.1f}  "
          f"wins {int((d>0).sum())}/{len(d)}")
json.dump({k: {"y": v["y"], "hit": v["hit"]} for k, v in res.items()},
          open(os.path.join(DATA_DIR, "yield_ab.json"), "w"), indent=1)
