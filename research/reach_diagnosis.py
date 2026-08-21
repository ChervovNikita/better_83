"""What do the field's cliques that we CANNOT reach look like?

Wall 5 says selection over our pool is useless -- on the median round only 5% of our
pool is un-taken, and we reach just 53.3% of the field's own cliques. So the gap is
reach. Rather than trying yet another blind variant, characterise WHAT WE MISS.

For each round, harvest K=200 maxima, then split the field's best-size cliques into
REACHED (in our pool) and MISSED (not). Compare them on:

  * distance to our pool -- the minimum number of vertices you would have to swap to
    get from one of our cliques to theirs. If MISSED cliques sit 2-3 swaps away, a
    deeper or wider local search reaches them and the wall is soft. If they sit far
    away, they are in a different region entirely and no local mechanism will do it.
  * how many miners hold them (are the ones we miss the popular ones or the rare?)
  * vertex overlap with our best clique.

This is diagnosis, not another variant: it says which mechanism, if any, could work.
"""
import json, os, sys
from collections import Counter
sys.path.insert(0,'/workspace/better_83/research'); sys.path.insert(0,'/workspace/better_83')
import numpy as np
from CliqueAI.graph.codec import GraphCodec
os.chdir('/workspace/better_83/research')
import fleetsolver as fs

N=int(os.environ.get("UR_N","40")); K=int(os.environ.get("UR_K","200"))
T=int(os.environ.get("UR_T","3")); os.environ["SN83_THREADS"]=str(T)
recs=[json.loads(l) for l in open('data/sets/recent_val.jsonl')][:N]

reach_d, miss_d, reach_h, miss_h = [], [], [], []
nreach = nmiss = 0
for i,r in enumerate(recs):
    fld={tuple(sorted(c)):int(k) for c,k in
         zip(r.get("best_cliques") or [], r.get("best_clique_counts") or [])}
    if not fld: continue
    A=np.ascontiguousarray(np.array(GraphCodec().decode_matrix(r["b92"]),dtype=np.uint8))
    cl=fs.solve_many(A, r["tl"]*0.88, K, threads=T)
    if not cl: continue
    mx=max(len(x) for x in cl)
    if mx < r["best"]: continue
    pool=[set(x) for x in cl if len(x)==mx]
    pset={tuple(sorted(x)) for x in cl if len(x)==mx}
    for c,k in fld.items():
        if len(c) != mx: continue
        s=set(c)
        # minimum swap distance from ANY clique in our pool
        d=min(len(s-p) for p in pool)
        if tuple(sorted(c)) in pset:
            nreach+=1; reach_d.append(d); reach_h.append(k)
        else:
            nmiss+=1;  miss_d.append(d);  miss_h.append(k)
    if (i+1)%10==0: print(f"  {i+1}/{len(recs)}", flush=True)

print(f"\n=== field cliques: reached vs missed ({nreach} reached, {nmiss} missed) ===")
def show(tag,d,h):
    if not d: return
    d=np.array(d); h=np.array(h)
    print(f"  {tag:8s} n={len(d):>5}  swap-distance from our pool: "
          f"mean {d.mean():.2f} median {np.median(d):.0f} min {d.min()} max {d.max()}")
    print(f"           holders: mean {h.mean():.2f} median {np.median(h):.0f}  "
          f"sole-holder {100*(h==1).mean():.1f}%")
show("REACHED",reach_d,reach_h); show("MISSED",miss_d,miss_h)
if miss_d:
    m=np.array(miss_d)
    print(f"\n  MISSED cliques by swap distance from our pool:")
    for k in range(1,7):
        print(f"    {k} swap(s): {int((m==k).sum()):>5}  {100*(m==k).mean():>5.1f}%")
    print(f"    >6 swaps : {int((m>6).sum()):>5}  {100*(m>6).mean():>5.1f}%")
