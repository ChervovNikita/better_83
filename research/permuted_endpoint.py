"""Is the natural endpoint under a PERMUTED labelling fresher than under identity?

The asymmetry that motivates this: our natural endpoint is un-taken ~34.4% of the
time, but a random member of our own reachable pool is fresher. Our endpoint is
systematically MORE collided than an arbitrary clique we can reach — which is what
you would see if most of the field runs solvers on the graph AS GIVEN and lands on
the same identity-natural cliques.

Relabelling was tested for pool SIZE (3.34x more maxima) but never for ENDPOINT
freshness. This is "coordination without prediction": we do not guess what others
pick, we just stop standing where their tie-breaking puts them. It costs one array
shuffle and is fully deterministic if seeded from the graph.

Paired: same instances, same budget, identity endpoint vs permuted endpoint.
"""
import json, os, sys
from math import comb
sys.path.insert(0,'/workspace/better_83/research'); sys.path.insert(0,'/workspace/better_83')
import numpy as np
from CliqueAI.graph.codec import GraphCodec
os.chdir('/workspace/better_83/research')
import fastsolver

N=int(os.environ.get("PE_N","250")); T=int(os.environ.get("PE_T","7"))
os.environ["SN83_THREADS"]=str(T)
recs=[json.loads(l) for l in open('data/sets/recent_val.jsonl')][:N]
rng=np.random.default_rng(83)

ident_fresh=[]; perm_fresh=[]; ident_par=[]; perm_par=[]; both=0
for i,r in enumerate(recs):
    fld={tuple(sorted(c)): int(k) for c,k in
         zip(r.get("best_cliques") or [], r.get("best_clique_counts") or [])}
    if not fld: continue
    A=np.ascontiguousarray(np.array(GraphCodec().decode_matrix(r["b92"]),dtype=np.uint8))
    n=A.shape[0]; tl=r["tl"]*0.88
    c1=fastsolver.solve(A, tl, seed=1)
    t1=tuple(sorted(int(v) for v in c1))
    p=rng.permutation(n)
    c2=fastsolver.solve(np.ascontiguousarray(A[np.ix_(p,p)]), tl, seed=1)
    t2=tuple(sorted(int(p[v]) for v in c2))
    ident_par.append(1 if len(t1)>=r["best"] else 0)
    perm_par.append(1 if len(t2)>=r["best"] else 0)
    if len(t1)>=r["best"]: ident_fresh.append(1 if t1 not in fld else 0)
    if len(t2)>=r["best"]: perm_fresh.append(1 if t2 not in fld else 0)
    if t1!=t2: both+=1
    if (i+1)%50==0: print(f"  {i+1}/{len(recs)}", flush=True)

a=np.array(ident_fresh); b=np.array(perm_fresh)
print(f"\n=== identity endpoint vs permuted endpoint ({len(a)} / {len(b)} at best size) ===")
print(f"  parity  identity {np.mean(ident_par):.3%}   permuted {np.mean(perm_par):.3%}")
print(f"  un-taken identity {a.mean():.1%}   permuted {b.mean():.1%}   diff {b.mean()-a.mean():+.1%}")
print(f"  answers that differ: {both}/{len(ident_par)} ({both/len(ident_par):.1%})")
se=np.sqrt(a.var(ddof=1)/len(a)+b.var(ddof=1)/len(b))
print(f"  t-stat: {(b.mean()-a.mean())/se:+.2f}")
print(f"\n  for scale: the spec needs un-taken 34.4% -> ~50%")
