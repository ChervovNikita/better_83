"""Is un-takenness a ROUND property or a CLIQUE property? Variance decomposition.

The diagnostic showed rounds where 0% of our pool is un-taken and rounds where 100%
is. If un-takenness is mostly determined by the round, then no rule that chooses
among cliques WITHIN a round can help -- which is exactly what v19 rarest-pick,
v26 random-pick, and nearest/furthest all found, each returning ~36%.

Decompose the variance of the per-clique un-taken indicator:

    total = between-round + within-round

A large between-round share means the +0.2383 correlation between distance and
un-takenness is a round-level artifact (big pools and long distances co-occur with
thin field coverage), not a usable within-round signal.
"""
import json, os, sys
sys.path.insert(0,'/workspace/better_83/research'); sys.path.insert(0,'/workspace/better_83')
import numpy as np
from CliqueAI.graph.codec import GraphCodec
os.chdir('/workspace/better_83/research')
import fleetsolver as fs
os.environ["SN83_THREADS"]="3"

N=int(os.environ.get("VR_N","70"))
recs=[json.loads(l) for l in open('data/sets/recent_val.jsonl')][:N]
fracs=[]; allu=[]; within=[]; wcorr=[]
for i,r in enumerate(recs):
    fld={tuple(sorted(c)) for c in (r.get("best_cliques") or [])}
    if not fld: continue
    A=np.ascontiguousarray(np.array(GraphCodec().decode_matrix(r["b92"]),dtype=np.uint8))
    cl=fs.solve_many(A, r["tl"]*0.88, 60, threads=3)
    if not cl: continue
    mx=max(len(x) for x in cl)
    if mx < r["best"]: continue
    pool=list(dict.fromkeys(tuple(sorted(x)) for x in cl if len(x)==mx))
    if len(pool)<3: continue
    nat=set(pool[0])
    d=np.array([len(nat-set(p)) for p in pool],dtype=float)
    u=np.array([0.0 if p in fld else 1.0 for p in pool])
    fracs.append(u.mean()); allu.extend(u.tolist())
    within.append(u.var())
    if d.std()>0 and u.std()>0: wcorr.append(float(np.corrcoef(d,u)[0,1]))
    if (i+1)%20==0: print(f"  {i+1}/{len(recs)}", flush=True)

f=np.array(fracs); au=np.array(allu)
tot=au.var(); wth=float(np.mean(within)); btw=tot-wth
print(f"\n=== un-takenness: round property or clique property? ({len(f)} rounds, {len(au)} cliques) ===")
print(f"  total variance of the per-clique indicator : {tot:.4f}")
print(f"    between-round : {btw:.4f}  ({100*btw/tot:.1f}%)")
print(f"    within-round  : {wth:.4f}  ({100*wth/tot:.1f}%)")
print()
print(f"  per-round un-taken fraction: mean {f.mean():.3f} median {np.median(f):.3f}")
print(f"    rounds with NOTHING un-taken (0%)      : {int((f==0).sum())}/{len(f)}  ({(f==0).mean():.1%})")
print(f"    rounds with EVERYTHING un-taken (100%) : {int((f==1).sum())}/{len(f)}  ({(f==1).mean():.1%})")
print(f"    rounds in between                      : {int(((f>0)&(f<1)).sum())}/{len(f)}")
print()
print(f"  WITHIN-round correlation(distance, un-taken): mean {np.mean(wcorr):+.4f} "
      f"(n={len(wcorr)} rounds with variation)")
print(f"  compare the POOLED correlation: +0.2383")
