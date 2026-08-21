"""Operator clustering on the FULL 33k corpus, not the 5k sample.

Methodology is the one that survived review: per-PAIR collision tests have no power
(base rate ~0.45%, a pair shares ~30 rounds, p=0.87), so aggregate to the WALLET
level and compare observed co-answer collisions against what independence predicts.

  expected(A,B) = sum over shared rounds of  P(two random answers collide in it)
  ratio         = observed / expected

ratio ~0 => coordinated (they deliberately never collide)
ratio ~1 => independent
ratio >>1 => same naive solver, colliding more than chance
"""
import json, collections, itertools, os, sys

from _common import DATA_DIR
import numpy as np

rounds = []
for line in open(os.path.join(DATA_DIR,'sim_rounds.jsonl')):
    if not line.strip(): continue
    r = json.loads(line)
    ans = [(a["ck"], tuple(sorted(a["clique"]))) for a in r.get("answers", [])
           if a.get("clique") and a.get("opt", 0) > 0]
    if len(ans) >= 2:
        rounds.append(ans)
print(f"{len(rounds)} rounds with >=2 valid answers", file=sys.stderr)

ck_rounds = collections.Counter()
for ans in rounds:
    for ck in {c for c, _ in ans}:
        ck_rounds[ck] += 1
big = [ck for ck, n in ck_rounds.items() if n >= 200]
print(f"{len(ck_rounds)} coldkeys, {len(big)} with >=200 rounds", file=sys.stderr)

obs = collections.Counter(); exp = collections.defaultdict(float)
shared = collections.Counter()
for ans in rounds:
    # P(two distinct random answers in this round share a vertex set)
    cnt = collections.Counter(c for _, c in ans)
    n = len(ans)
    if n < 2: continue
    p = sum(v * (v - 1) for v in cnt.values()) / (n * (n - 1))
    bck = collections.defaultdict(list)
    for ck, c in ans:
        bck[ck].append(c)
    ks = [k for k in bck if k in set(big)]
    for a, b in itertools.combinations(sorted(ks), 2):
        na, nb = len(bck[a]), len(bck[b])
        shared[(a, b)] += 1
        exp[(a, b)] += p * na * nb
        sb = collections.Counter(bck[b])
        obs[(a, b)] += sum(sb[c] for c in bck[a])

rows = []
for pair, e in exp.items():
    if shared[pair] >= 100 and e >= 5:
        rows.append((obs[pair] / e, obs[pair], e, shared[pair], pair))
rows.sort()
print(f"\n{len(rows)} wallet pairs with >=100 shared rounds and expected>=5\n")
print(f"{'ratio':>8} {'obs':>7} {'exp':>9} {'rounds':>7}  pair")
for ratio, o, e, s, (a, b) in rows[:14]:
    print(f"{ratio:>8.3f} {o:>7} {e:>9.1f} {s:>7}  {a[:10]}.. {b[:10]}..")
print("  ...")
for ratio, o, e, s, (a, b) in rows[-6:]:
    print(f"{ratio:>8.3f} {o:>7} {e:>9.1f} {s:>7}  {a[:10]}.. {b[:10]}..")

r = np.array([x[0] for x in rows])
print(f"\nratio distribution: min {r.min():.3f}  p25 {np.percentile(r,25):.3f}  "
      f"median {np.median(r):.3f}  p75 {np.percentile(r,75):.3f}  max {r.max():.3f}")
print(f"pairs with ratio < 0.05 (never collide => coordinated): {int((r<0.05).sum())}")
print(f"pairs with ratio > 2.0  (collide far more than chance): {int((r>2.0).sum())}")
