"""Is our search basin-locked -- does it land in the same region from ANY start?

The reach diagnosis says the field cliques we miss sit a median of 24 vertex-swaps
from our entire pool, and 81% are >6 swaps away. That implies our pool is clustered
in one region while the field spans many. If so, the mechanism is not "search
deeper" but "start somewhere else" -- and cold restarts and seed variation were both
already refuted, which would mean our solver converges to the same region regardless
of where it starts.

Test it directly: run the solver from many different seeds and measure the pairwise
swap distance between the answers. Compare that to the distance to the field's
missed cliques (median 24).

  small internal spread + large distance to missed  => basin-locked, wall is HARD
  large internal spread                             => we do explore; something else
                                                        explains the miss
"""
import json, os, sys
sys.path.insert(0,'/workspace/better_83/research'); sys.path.insert(0,'/workspace/better_83')
import numpy as np
from CliqueAI.graph.codec import GraphCodec
os.chdir('/workspace/better_83/research')
import fastsolver, fleetsolver as fs

N=int(os.environ.get("BL_N","25")); S=int(os.environ.get("BL_S","12"))
os.environ["SN83_THREADS"]="3"
recs=[json.loads(l) for l in open('data/sets/recent_val.jsonl')][:N]

seed_spread, pool_spread, to_missed = [], [], []
for i,r in enumerate(recs):
    fld={tuple(sorted(c)) for c in (r.get("best_cliques") or [])}
    if not fld: continue
    A=np.ascontiguousarray(np.array(GraphCodec().decode_matrix(r["b92"]),dtype=np.uint8))
    tl=r["tl"]*0.88
    # (a) many independent seeds, full budget each
    outs=[]
    for sd in range(S):
        c=fastsolver.solve(A, tl/S*2.0, seed=1000+sd*7919)
        if len(c)>=r["best"]: outs.append(set(int(v) for v in c))
    if len(outs)<2: continue
    ds=[len(a-b) for j,a in enumerate(outs) for b in outs[j+1:]]
    seed_spread.append(float(np.mean(ds)))
    # (b) the harvested pool's internal spread
    cl=fs.solve_many(A, tl, 60, threads=3)
    mx=max((len(x) for x in cl), default=0)
    pool=[set(x) for x in cl if len(x)==mx]
    if len(pool)>=2:
        dp=[len(a-b) for j,a in enumerate(pool[:25]) for b in pool[j+1:25]]
        if dp: pool_spread.append(float(np.mean(dp)))
    # (c) distance from our pool to the field cliques we missed
    ps={tuple(sorted(x)) for x in pool}
    miss=[set(c) for c in fld if tuple(sorted(c)) not in ps and len(c)==mx]
    if miss and pool:
        to_missed.append(float(np.mean([min(len(m-p) for p in pool) for m in miss])))
    if (i+1)%5==0: print(f"  {i+1}/{len(recs)}", flush=True)

print(f"\n=== basin lock, {len(seed_spread)} rounds, {S} seeds ===")
for tag,v in (("across SEEDS (independent runs)",seed_spread),
              ("within our harvested POOL",pool_spread),
              ("our pool -> field cliques we MISS",to_missed)):
    if v: print(f"  {tag:34s} mean swap distance {np.mean(v):6.2f}  median {np.median(v):6.2f}")
