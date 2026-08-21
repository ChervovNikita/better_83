"""How much of OUR reachable pool is un-taken by the field? The decisive number.

The field covers 25.0 distinct best-size cliques per round. Our single answer lands
outside that coverage 34.3% of the time. Two very different worlds produce that:

  A) our pool is mostly INSIDE the field's coverage -> 34.3% is near our ceiling and
     no selection rule helps, because there is little outside to select.
  B) our pool is mostly OUTSIDE it -> selection is the whole problem, and even
     UNIFORM RANDOM picking would land un-taken far more than 34.3% of the time.

Harvest as many distinct maxima as the budget allows and measure the un-taken
fraction directly. This decides whether the +0.1488 diversity gap is reachable.
"""
import json, os, sys, time
from collections import Counter
sys.path.insert(0,'/workspace/better_83/research'); sys.path.insert(0,'/workspace/better_83')
import numpy as np
from CliqueAI.graph.codec import GraphCodec
os.chdir('/workspace/better_83/research')
import fleetsolver as fs

N = int(os.environ.get("PC_N", "60")); K = int(os.environ.get("PC_K", "200"))
T = int(os.environ.get("PC_T", "3"))
os.environ["SN83_THREADS"] = str(T)
champ = {r["uuid"]: r for r in json.load(
    open('/home/dev/autoresearch-runs/sn83-clique/runs/final_shipped_rv500.json'))["rows"]}
recs = [json.loads(l) for l in open('data/sets/recent_val.jsonl')][:N]

pool_sz, untaken_frac, field_sz, champ_untaken, overlap = [], [], [], [], []
for i, r in enumerate(recs):
    c = champ.get(r["uuid"])
    fld = {tuple(sorted(x)) for x in (r.get("best_cliques") or [])}
    if not fld or not c:
        continue
    A = np.ascontiguousarray(np.array(GraphCodec().decode_matrix(r["b92"]), dtype=np.uint8))
    cl = fs.solve_many(A, r["tl"] * 0.88, K, threads=T)
    if not cl:
        continue
    mx = max(len(x) for x in cl)
    if mx < r["best"]:
        continue                      # only compare at the field's best size
    pool = {tuple(sorted(x)) for x in cl if len(x) == mx}
    out = sum(1 for p in pool if p not in fld)
    pool_sz.append(len(pool)); untaken_frac.append(out / len(pool))
    field_sz.append(len(fld)); champ_untaken.append(1 if c.get("collide") == 0 else 0)
    overlap.append(sum(1 for p in pool if p in fld) / max(len(fld), 1))
    if (i + 1) % 15 == 0:
        print(f"  {i+1}/{len(recs)}", flush=True)

print(f"\n=== pool coverage, {len(pool_sz)} rounds, K={K} ===")
print(f"  our distinct maxima harvested : mean {np.mean(pool_sz):6.1f}  median {np.median(pool_sz):.0f}")
print(f"  field distinct best cliques   : mean {np.mean(field_sz):6.1f}")
print(f"  fraction of OUR pool un-taken : mean {np.mean(untaken_frac):.1%}  median {np.median(untaken_frac):.1%}")
print(f"  fraction of FIELD's cliques we reach: mean {np.mean(overlap):.1%}")
print()
print(f"  champion's actual un-taken rate on these rounds: {np.mean(champ_untaken):.1%}")
print(f"  UNIFORM RANDOM pick from our pool would be un-taken: {np.mean(untaken_frac):.1%}")
d = np.mean(untaken_frac) - np.mean(champ_untaken)
print(f"  => random-pick changes un-taken rate by {d:+.1%}")
