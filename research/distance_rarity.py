"""Does distance from our NATURAL ENDPOINT predict whether a clique is un-taken?

Shard 1 showed seeds (spread 20.33) collide MORE than harvested cliques (spread
12.01). My hypothesis for that: independent runs converge to the algorithm's
canonical attractors, which the field's similar solvers also find, while
plateau-harvesting wanders into off-attractor cliques that are rare BECAUSE they are
unnatural.

That hypothesis makes a sharp prediction, testable right here: within a single
harvested pool, cliques FURTHER from the solver's natural endpoint should be un-taken
more often. If true it is also a selection rule -- "take the furthest" -- and unlike
v19 rarest-pick it needs no prediction of field behaviour, only our own geometry.

Wall 5 says uniform random picking gives 34.3% un-taken against the champion's 36.2%.
So the bar is 36.2%: a furthest-pick rule has to beat that to be worth anything.
"""
import json, os, sys
from collections import Counter
sys.path.insert(0,'/workspace/better_83/research'); sys.path.insert(0,'/workspace/better_83')
import numpy as np
from CliqueAI.graph.codec import GraphCodec
os.chdir('/workspace/better_83/research')
import fleetsolver as fs

N=int(os.environ.get("DR_N","60")); K=int(os.environ.get("DR_K","60"))
T=int(os.environ.get("DR_T","3")); os.environ["SN83_THREADS"]=str(T)
recs=[json.loads(l) for l in open('data/sets/recent_val.jsonl')][:N]

rows=[]           # (distance_from_endpoint, is_untaken)
pick_near, pick_far, pick_unif, pick_nat = [], [], [], []
for i,r in enumerate(recs):
    fld={tuple(sorted(c)) for c in (r.get("best_cliques") or [])}
    if not fld: continue
    A=np.ascontiguousarray(np.array(GraphCodec().decode_matrix(r["b92"]),dtype=np.uint8))
    cl=fs.solve_many(A, r["tl"]*0.88, K, threads=T)
    if not cl: continue
    mx=max(len(x) for x in cl)
    if mx < r["best"]: continue
    pool=[tuple(sorted(x)) for x in cl if len(x)==mx]
    pool=list(dict.fromkeys(pool))
    if len(pool)<2: continue
    nat=set(pool[0])                      # solve_many returns the natural endpoint first
    d=[len(nat-set(p)) for p in pool]
    unt=[0 if p in fld else 1 for p in pool]
    rows += list(zip(d,unt))
    order=np.argsort(d)
    pick_nat.append(unt[0])
    pick_near.append(unt[order[0]])
    pick_far.append(unt[order[-1]])
    pick_unif.append(float(np.mean(unt)))
    if (i+1)%15==0: print(f"  {i+1}/{len(recs)}", flush=True)

d=np.array([x[0] for x in rows]); u=np.array([x[1] for x in rows])
print(f"\n=== distance vs rarity, {len(pick_nat)} rounds, {len(rows)} cliques ===")
print(f"  correlation(distance, un-taken) = {np.corrcoef(d,u)[0,1]:+.4f}")
print()
print(f"  un-taken rate by distance from the natural endpoint:")
for lo,hi in ((0,0),(1,3),(4,7),(8,15),(16,99)):
    m=(d>=lo)&(d<=hi)
    if m.sum(): print(f"    {lo:>2}-{hi:<3} swaps: {u[m].mean():>6.1%}  (n={int(m.sum())})")
print()
print(f"  SELECTION RULES (bar to beat: champion 36.2% un-taken)")
print(f"    natural endpoint  : {np.mean(pick_nat):.1%}")
print(f"    uniform random    : {np.mean(pick_unif):.1%}")
print(f"    nearest to natural: {np.mean(pick_near):.1%}")
print(f"    FURTHEST          : {np.mean(pick_far):.1%}")
