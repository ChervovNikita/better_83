"""Is our reachable pool bounded by TIME or by the ALGORITHM?

The freshness spec needs un-taken 34.4% -> 50%. Only 34.3% of our pool is un-taken,
and the field cliques we miss sit a median 24 vertex-swaps away. The open question is
whether those are unreachable in principle or merely unreached in 10 seconds.

Give the harvester 1x, 4x and 16x the normal budget on the same instances and track:
  * how many DISTINCT maximum cliques it finds (does the count saturate?)
  * what fraction of them the field has not taken (does freshness improve?)

If the count keeps climbing and the un-taken fraction rises, time is the constraint
and a bigger machine closes part of the gap. If both plateau, the algorithm's
reachable set is closed and no amount of compute helps -- which would mean the spec
requires a fundamentally different sampler, not a faster one.

This is the last question that changes what a successor should try.
"""
import json, os, sys, time
sys.path.insert(0,'/workspace/better_83/research'); sys.path.insert(0,'/workspace/better_83')
import numpy as np
from CliqueAI.graph.codec import GraphCodec
os.chdir('/workspace/better_83/research')
import fleetsolver as fs

N=int(os.environ.get("SA_N","20")); T=int(os.environ.get("SA_T","14"))
MULTS=[1,4,16]
os.environ["SN83_THREADS"]=str(T)
recs=[json.loads(l) for l in open('data/sets/recent_val.jsonl')][:N]

res={m:{"n":[], "unt":[], "best":[]} for m in MULTS}
for i,r in enumerate(recs):
    fld={tuple(sorted(c)) for c in (r.get("best_cliques") or [])}
    if not fld: continue
    A=np.ascontiguousarray(np.array(GraphCodec().decode_matrix(r["b92"]),dtype=np.uint8))
    for m in MULTS:
        cl=fs.solve_many(A, r["tl"]*0.88*m, 500, threads=T)
        if not cl: continue
        mx=max(len(x) for x in cl)
        pool=list(dict.fromkeys(tuple(sorted(x)) for x in cl if len(x)==mx))
        res[m]["n"].append(len(pool))
        res[m]["best"].append(1 if mx>=r["best"] else 0)
        res[m]["unt"].append(sum(1 for p in pool if p not in fld)/len(pool))
    print(f"  {i+1}/{len(recs)}", flush=True)

print(f"\n=== does the reachable pool saturate? ({len(res[1]['n'])} instances, {T} threads) ===")
print(f"{'budget':>8} {'distinct maxima':>17} {'median':>8} {'un-taken frac':>15} {'best size':>10}")
for m in MULTS:
    d=res[m]
    if not d["n"]: continue
    print(f"{m:>7}x {np.mean(d['n']):>17.1f} {np.median(d['n']):>8.0f} "
          f"{np.mean(d['unt']):>15.1%} {np.mean(d['best']):>10.0%}")
print()
if res[1]['n'] and res[16]['n']:
    g=np.mean(res[16]['n'])/max(np.mean(res[1]['n']),1e-9)
    du=np.mean(res[16]['unt'])-np.mean(res[1]['unt'])
    print(f"  16x the budget gives {g:.2f}x the distinct maxima and "
          f"{du:+.1%} un-taken fraction")
    print(f"  (the spec needs the un-taken rate to go from 34.4% to ~50%)")
