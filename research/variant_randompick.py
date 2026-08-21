"""v26 random-pick: harvest K distinct maximum cliques, return a uniformly random one.

Motivation, quantified this session: on the same 483 rounds, field best-size answers
earn 0.7481 diversity and ours earn 0.5993. The field spreads across DISTINCT maximum
cliques (56% sole holders); our solver concentrates on whichever basin it falls into,
which is plausibly the same basin the other popular solvers fall into.

This is NOT v19 rarest-pick (refuted, p=0.31). That tried to PREDICT which clique the
field would avoid, and we showed field popularity is unpredictable -- every structural
feature |corr| <= 0.029. Precisely because it is unpredictable, the right response is
not a better predictor but an UNPREDICTABLE choice: sample uniformly from the maxima
we can reach, so our output decorrelates from any popular basin.

Paired against the champion's stored answers on the same tasks. Reports parity
(McNemar) and diversity over ONLY the tasks whose answer changed.
"""
import ctypes, json, os, sys, time
from collections import Counter
sys.path.insert(0,'/workspace/better_83/research'); sys.path.insert(0,'/workspace/better_83')
import numpy as np
from CliqueAI.graph.codec import GraphCodec
os.chdir('/workspace/better_83/research')
import bench, fleetsolver as fs

K       = int(os.environ.get("RP_K", "24"))
NTASK   = int(os.environ.get("RP_N", "250"))
THREADS = int(os.environ.get("RP_THREADS", "4"))
SEED    = int(os.environ.get("RP_SEED", "12345"))
os.environ["SN83_THREADS"] = str(THREADS)

champ = {r["uuid"]: r for r in
         json.load(open('/home/dev/autoresearch-runs/sn83-clique/runs/final_shipped_rv500.json'))["rows"]}
recs = [json.loads(l) for l in open('data/sets/recent_val.jsonl')][:NTASK]
rng = np.random.default_rng(SEED)

rows = []
t0 = time.time()
for i, r in enumerate(recs):
    c = champ.get(r["uuid"])
    if not c or c.get("reward") is None:
        continue
    A = np.ascontiguousarray(np.array(GraphCodec().decode_matrix(r["b92"]), dtype=np.uint8))
    tl = r["tl"] * 0.88
    t = time.time()
    cl = fs.solve_many(A, tl, K, threads=THREADS)
    el = time.time() - t
    if not cl:
        continue
    ok, why = fs.check(A, cl)
    if not ok:
        print(f"  INVALID {r['uuid'][:8]}: {why}", flush=True); continue
    mx = max(len(x) for x in cl)
    pool = sorted({tuple(sorted(x)) for x in cl if len(x) == mx})
    pick = pool[int(rng.integers(len(pool)))]
    # how many field miners hold our exact set
    collide = 0
    for cc, kk in zip(r.get("best_cliques") or [], r.get("best_clique_counts") or []):
        if tuple(sorted(cc)) == pick:
            collide = int(kk); break
    rw, op, dv = bench.replay_reward(len(pick), collide, r.get("size_hist"),
                                     r.get("best_clique_counts"), r.get("difficulty", 0.85),
                                     r.get("any_unique"), r.get("n_responders"),
                                     count_hist=r.get("count_hist"))
    rows.append(dict(uuid=r["uuid"], n=r["n"], tl=r["tl"], best=r["best"],
                     ours=len(pick), pool=len(pool), collide=collide, elapsed=el,
                     over=el > r["tl"]*1.02+0.25, reward=rw, optimality=op, diversity=dv,
                     c_ours=c["ours"], c_collide=c.get("collide"), c_reward=c["reward"],
                     c_div=c.get("diversity")))
    if (i+1) % 25 == 0:
        print(f"  {i+1}/{len(recs)}  {(time.time()-t0)/(i+1):.1f}s/task", flush=True)

json.dump(rows, open('/home/dev/autoresearch-runs/sn83-clique/runs/randompick.json','w'))
R = rows
par  = sum(1 for x in R if x["ours"] >= x["best"]) / len(R)
cpar = sum(1 for x in R if x["c_ours"] >= x["best"]) / len(R)
print(f"\n=== v26 random-pick (K={K}, {len(R)} tasks, {THREADS} threads) ===")
print(f"  parity  ours {par:.3%}   champion {cpar:.3%}")
print(f"  reward  ours {np.mean([x['reward'] for x in R]):.4f}   "
      f"champion {np.mean([x['c_reward'] for x in R]):.4f}")
print(f"  divers  ours {np.mean([x['diversity'] for x in R]):.4f}   "
      f"champion {np.mean([x['c_div'] for x in R]):.4f}")
print(f"  mean distinct maxima in pool: {np.mean([x['pool'] for x in R]):.1f}")
print(f"  over budget: {sum(1 for x in R if x['over'])}")
# paired tests
b_only = sum(1 for x in R if x["ours"] >= x["best"] and x["c_ours"] < x["best"])
a_only = sum(1 for x in R if x["c_ours"] >= x["best"] and x["ours"] < x["best"])
print(f"\n  McNemar parity: ours-only {b_only}, champion-only {a_only}")
ch = [x for x in R if x["collide"] != x["c_collide"]]
w = sum(1 for x in ch if x["diversity"] > x["c_div"])
l = sum(1 for x in ch if x["diversity"] < x["c_div"])
print(f"  changed answers: {len(ch)}/{len(R)}   ours better {w}, champion better {l}")
if w + l:
    from math import comb
    n_, k_ = w + l, min(w, l)
    p = 2 * sum(comb(n_, j) for j in range(k_ + 1)) / (2 ** n_)
    print(f"  two-sided sign test p = {min(p,1.0):.4f}")
d = np.array([x["reward"] for x in R]) - np.array([x["c_reward"] for x in R])
print(f"  paired reward delta: mean {d.mean():+.4f}  SE {d.std(ddof=1)/np.sqrt(len(d)):.4f}  "
      f"t {d.mean()/(d.std(ddof=1)/np.sqrt(len(d))):+.2f}")
