"""Same question, WITHIN instance — the pooled version is confounded.

Pooled across instances, missed cliques showed higher mean vertex degree (t=+5.29).
But degree scales with |V| and density, which range over 290-900 and 0.70-0.95 here,
so if missed cliques concentrate in bigger or denser instances the pooled test shows
a difference that says nothing about clique structure.

This is the same Simpson's paradox that produced a spurious +0.2383 distance-vs-
rarity correlation earlier (within-round: -0.0132). So: compare reached vs missed
WITHIN each instance, then aggregate the per-instance differences.
"""
import json, os, sys
sys.path.insert(0,'/workspace/better_83/research'); sys.path.insert(0,'/workspace/better_83')
import numpy as np
from CliqueAI.graph.codec import GraphCodec
os.chdir('/workspace/better_83/research')
import fleetsolver as fs

N=int(os.environ.get("WM_N","90")); K=int(os.environ.get("WM_K","6"))
T=int(os.environ.get("WM_T","6")); os.environ["SN83_THREADS"]=str(T)
recs=[json.loads(l) for l in open('data/sets/recent_val.jsonl')][:N]
rng=np.random.default_rng(7)

per=[]        # per-instance (mean_missed - mean_reached), standardised by instance sd
counts=[]
for i,r in enumerate(recs):
    fld={tuple(sorted(c)) for c in (r.get("best_cliques") or [])}
    if not fld: continue
    A=np.ascontiguousarray(np.array(GraphCodec().decode_matrix(r["b92"]),dtype=np.uint8))
    n=A.shape[0]; deg=A.sum(axis=1).astype(float); tl=r["tl"]*0.88/K
    pool=set()
    for j in range(K):
        if j==0: cl=fs.solve_many(A, tl, 200, threads=T); p=np.arange(n)
        else:
            p=rng.permutation(n)
            cl=fs.solve_many(np.ascontiguousarray(A[np.ix_(p,p)]), tl, 200, threads=T)
        if not cl: continue
        m2=max(len(x) for x in cl)
        for c in cl:
            if len(c)==m2: pool.add(tuple(sorted(int(p[v]) for v in c)))
    if not pool: continue
    mx=max(len(c) for c in pool)
    R=[float(deg[list(c)].mean()) for c in fld if len(c)==mx and c in pool]
    M=[float(deg[list(c)].mean()) for c in fld if len(c)==mx and c not in pool]
    if len(R)<2 or len(M)<2: continue
    sd=deg.std()
    per.append((np.mean(M)-np.mean(R))/(sd if sd>0 else 1.0))
    counts.append((len(R),len(M)))
    if (i+1)%10==0: print(f"  {i+1}/{len(recs)}", flush=True)

d=np.array(per)
print(f"\n=== WITHIN-instance: mean vertex degree, missed minus reached ===")
print(f"  instances with both classes present: {len(d)}")
print(f"  mean standardised difference : {d.mean():+.4f} sd of the instance's degrees")
print(f"  median                       : {np.median(d):+.4f}")
se=d.std(ddof=1)/np.sqrt(len(d)) if len(d)>1 else float('nan')
print(f"  t-stat vs 0                  : {d.mean()/se:+.2f}   (SE {se:.4f})")
print(f"  instances where missed > reached: {int((d>0).sum())}/{len(d)}")
print(f"\n  pooled (confounded) version reported t=+5.29 on the same data.")
