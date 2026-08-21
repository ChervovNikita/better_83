"""Fleet answer generation: harvest one plateau vs N independent seeded runs.

Basin-lock test: independent seeds spread 20.33 vertex-swaps apart, harvesting off a
single plateau only 12.01. fleetsolver.py (and therefore the fleet simulation) uses
HARVESTING. If independent seeds cover more of the space, the fleet's per-hotkey
diversity is being understated.

This is also the REALISTIC setup: N rented pods each have their own CPU, so each
hotkey can run the full-budget champion with its own seed. Harvesting is what you do
when one box must serve N hotkeys. So the comparison is not just fairer, it is the
actual deployment.

Per-hotkey diversity = 1 / (our own duplicates + field miners holding the same set),
which is what the validator's delta term reduces to before normalisation.
"""
import json, os, sys
from collections import Counter
sys.path.insert(0,'/workspace/better_83/research'); sys.path.insert(0,'/workspace/better_83')
import numpy as np
from CliqueAI.graph.codec import GraphCodec
os.chdir('/workspace/better_83/research')
import fastsolver, fleetsolver as fs

N=int(os.environ.get("FG_N","30")); T=int(os.environ.get("FG_T","3"))
OFF=int(os.environ.get("FG_OFF","0"))
FLEETS=[1,5,10,20]
os.environ["SN83_THREADS"]=str(T)
recs=[json.loads(l) for l in open("data/sets/recent_val.jsonl")][OFF:OFF+N]

res={m:{k:[] for k in FLEETS} for m in ("harvest","seeds")}
parity={m:[] for m in ("harvest","seeds")}
for i,r in enumerate(recs):
    field={tuple(sorted(c)):int(k) for c,k in
           zip(r.get("best_cliques") or [], r.get("best_clique_counts") or [])}
    if not field: continue
    A=np.ascontiguousarray(np.array(GraphCodec().decode_matrix(r["b92"]),dtype=np.uint8))
    tl=r["tl"]*0.88
    mx=max(FLEETS)
    # (a) harvest: one solve_many call producing up to mx cliques
    h=fs.solve_many(A, tl, mx, threads=T)
    hbest=max((len(x) for x in h), default=0)
    hp=[tuple(sorted(x)) for x in h if len(x)==hbest]
    parity["harvest"].append(1 if hbest>=r["best"] else 0)
    # (b) independent seeds: each hotkey runs the FULL-budget champion, own seed
    #     (realistic: one pod per hotkey). Same wall-clock per hotkey as harvest.
    sp=[]
    for sd in range(mx):
        c=fastsolver.solve(A, tl, seed=7919*(sd+1))
        sp.append(tuple(sorted(int(v) for v in c)))
    sbest=max(len(x) for x in sp)
    parity["seeds"].append(1 if sbest>=r["best"] else 0)
    sp=[x for x in sp if len(x)==sbest]
    for tag,pool in (("harvest",hp),("seeds",sp)):
        if not pool: continue
        for n in FLEETS:
            assign=[pool[j % len(pool)] for j in range(n)]
            mine=Counter(assign)
            d=[1.0/(mine[c]+field.get(c,0)) for c in assign]
            res[tag][n].append(float(np.mean(d)))
    if (i+1)%5==0: print(f"  {i+1}/{len(recs)}", flush=True)

print(f"\n=== fleet generation: harvest vs independent seeds ({len(parity['harvest'])} rounds) ===")
print(f"{'N hotkeys':>10} " + "".join(f"{t:>12}" for t in ("harvest","seeds","delta")))
for n in FLEETS:
    h=np.mean(res["harvest"][n]) if res["harvest"][n] else float('nan')
    s=np.mean(res["seeds"][n]) if res["seeds"][n] else float('nan')
    print(f"{n:>10} {h:>12.4f}{s:>12.4f}{s-h:>+12.4f}")
print(f"\n  best-size parity: harvest {np.mean(parity['harvest']):.1%}   "
      f"seeds {np.mean(parity['seeds']):.1%}")
print(f"  distinct maxima  : harvest {np.mean([len(set()) or 0 for _ in [0]]):.0f}", end="")
print()
