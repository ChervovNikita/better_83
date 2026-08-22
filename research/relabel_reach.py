"""Is the reachable set closed under RELABELLING the graph?

Wall 6: our pool is fixed at 96 maxima and 16x compute adds nothing. But every probe
so far perturbed RANDOM choices -- seeds, weights, restarts, chain depth. The
solver's DETERMINISTIC tie-breaking (scan order, first-index-wins, which vertex a
bitset scan reaches first) has never been varied, and it is baked into the vertex
numbering.

Permuting the vertex labels changes every tie-break at once while leaving the
algorithm bit-identical. If the union of maxima over k permutations exceeds the
single-labelling pool, the reachable set is NOT closed and relabelling is a genuine
new sampler -- deployable, since it costs one array shuffle.

We know maxima outside our pool exist: the field holds cliques we never reach (we
cover 31.1% of theirs). The question is only whether this reaches them.
"""
import json, os, sys, time
sys.path.insert(0,'/workspace/better_83/research'); sys.path.insert(0,'/workspace/better_83')
import numpy as np
from CliqueAI.graph.codec import GraphCodec
os.chdir('/workspace/better_83/research')
import fleetsolver as fs

N=int(os.environ.get("PM_N","25")); K=int(os.environ.get("PM_K","8"))
T=int(os.environ.get("PM_T","6"))
os.environ["SN83_THREADS"]=str(T)
recs=[json.loads(l) for l in open('data/sets/recent_val.jsonl')][:N]
rng=np.random.default_rng(20260822)

base_n, uni_n, gain, fresh_base, fresh_uni = [], [], [], [], []
for i,r in enumerate(recs):
    fld={tuple(sorted(c)) for c in (r.get("best_cliques") or [])}
    if not fld: continue
    A=np.ascontiguousarray(np.array(GraphCodec().decode_matrix(r["b92"]),dtype=np.uint8))
    n=A.shape[0]; tl=r["tl"]*0.88/K          # SAME total budget as one full solve
    # identity labelling
    cl=fs.solve_many(A, tl, 200, threads=T)
    if not cl: continue
    mx=max(len(x) for x in cl)
    base={tuple(sorted(x)) for x in cl if len(x)==mx}
    union=set(base)
    for _ in range(K-1):
        p=rng.permutation(n)
        B=np.ascontiguousarray(A[np.ix_(p,p)])
        cl2=fs.solve_many(B, tl, 200, threads=T)
        if not cl2: continue
        m2=max(len(x) for x in cl2)
        if m2 < mx: continue
        # map back to original labels
        for c in cl2:
            if len(c)==m2: union.add(tuple(sorted(int(p[v]) for v in c)))
    if not base: continue
    base_n.append(len(base)); uni_n.append(len(union))
    gain.append(len(union)/len(base))
    fresh_base.append(sum(1 for c in base if c not in fld)/len(base))
    fresh_uni.append(sum(1 for c in union if c not in fld)/len(union))
    if (i+1)%5==0: print(f"  {i+1}/{len(recs)}", flush=True)

print(f"\n=== is the reachable set closed under relabelling? "
      f"({len(base_n)} instances, {K} permutations, SAME total budget) ===")
print(f"  maxima, identity labelling : mean {np.mean(base_n):6.1f}")
print(f"  maxima, union over {K} perms: mean {np.mean(uni_n):6.1f}")
print(f"  ratio                      : {np.mean(gain):.2f}x  (1.00 = closed)")
print()
print(f"  un-taken fraction, identity: {np.mean(fresh_base):.1%}")
print(f"  un-taken fraction, union   : {np.mean(fresh_uni):.1%}")
print(f"  change                     : {np.mean(fresh_uni)-np.mean(fresh_base):+.1%}")
