"""Does vertex degree predict FRESHNESS within a round? The conversion test.

We now know the field cliques we miss sit on slightly lower-degree vertices
(within-instance t=-2.70, effect -0.045 sd). That is a property of our SAMPLER's
reach. It only becomes useful if degree also predicts whether a clique is UN-TAKEN,
within a round — otherwise there is nothing to steer toward.

Wall 5 killed selection using distance-from-endpoint as the feature (within-round
correlation -0.0132 against a pooled +0.2383). Degree was never tested. If the
within-round correlation between a clique's mean degree and its un-takenness is also
~0, then the degree finding does not convert and the low-degree sampler is not worth
building.

Measured WITHIN round throughout, since the pooled version of this exact comparison
had the sign backwards.
"""
import json, os, sys
sys.path.insert(0,'/workspace/better_83/research'); sys.path.insert(0,'/workspace/better_83')
import numpy as np
from CliqueAI.graph.codec import GraphCodec
os.chdir('/workspace/better_83/research')
import fleetsolver as fs

N=int(os.environ.get("DF_N","60")); T=int(os.environ.get("DF_T","7"))
os.environ["SN83_THREADS"]=str(T)
recs=[json.loads(l) for l in open('data/sets/recent_val.jsonl')][:N]

wcorr=[]; lo_fresh=[]; hi_fresh=[]; pooled_d=[]; pooled_u=[]
for i,r in enumerate(recs):
    fld={tuple(sorted(c)) for c in (r.get("best_cliques") or [])}
    if not fld: continue
    A=np.ascontiguousarray(np.array(GraphCodec().decode_matrix(r["b92"]),dtype=np.uint8))
    deg=A.sum(axis=1).astype(float)
    cl=fs.solve_many(A, r["tl"]*0.88, 200, threads=T)
    if not cl: continue
    mx=max(len(x) for x in cl)
    pool=list(dict.fromkeys(tuple(sorted(x)) for x in cl if len(x)==mx))
    if len(pool)<4: continue
    d=np.array([deg[list(c)].mean() for c in pool])
    u=np.array([0.0 if c in fld else 1.0 for c in pool])
    pooled_d.extend(((d-d.mean())/(d.std() or 1)).tolist()); pooled_u.extend(u.tolist())
    if d.std()>0 and u.std()>0: wcorr.append(float(np.corrcoef(d,u)[0,1]))
    med=np.median(d)
    if (d<=med).sum() and (d>med).sum():
        lo_fresh.append(float(u[d<=med].mean())); hi_fresh.append(float(u[d>med].mean()))
    if (i+1)%15==0: print(f"  {i+1}/{len(recs)}", flush=True)

w=np.array(wcorr)
print(f"\n=== does degree predict un-takenness WITHIN a round? ({len(w)} rounds) ===")
print(f"  within-round corr(mean degree, un-taken): mean {w.mean():+.4f}  median {np.median(w):+.4f}")
se=w.std(ddof=1)/np.sqrt(len(w)) if len(w)>1 else float('nan')
print(f"    t vs 0: {w.mean()/se:+.2f}   (SE {se:.4f})")
print(f"    rounds with negative corr (low degree = fresher): {int((w<0).sum())}/{len(w)}")
lo,hi=np.array(lo_fresh),np.array(hi_fresh)
print(f"\n  un-taken rate, LOW-degree half of our pool : {lo.mean():.1%}")
print(f"  un-taken rate, HIGH-degree half            : {hi.mean():.1%}")
print(f"  difference                                 : {lo.mean()-hi.mean():+.1%}")
dd=lo-hi; s2=dd.std(ddof=1)/np.sqrt(len(dd))
print(f"    paired t: {dd.mean()/s2:+.2f}")
print(f"\n  for scale: the spec needs the un-taken rate 34.4% -> ~50%")
