"""Group wallets into operators using only HIGH-POWER pairs.

A ratio of 0.000 over 330 rounds with 7 expected collisions is weak: absence of 7
expected events is unsurprising. A ratio of 0.003 over 31,000 rounds with 23,000
expected is overwhelming. So gate on expected>=200 before calling anything
coordinated, and report the weak pairs separately rather than folding them in.
"""
import json, collections, itertools, os, sys

from _common import DATA_DIR
import numpy as np

rounds=[]
for line in open(os.path.join(DATA_DIR,'sim_rounds.jsonl')):
    if not line.strip(): continue
    r=json.loads(line)
    ans=[(a["ck"],tuple(sorted(a["clique"]))) for a in r.get("answers",[])
         if a.get("clique") and a.get("opt",0)>0]
    if len(ans)>=2: rounds.append(ans)

ck_rounds=collections.Counter()
uids=collections.defaultdict(set)
for line in open(os.path.join(DATA_DIR,'sim_rounds.jsonl')):
    if not line.strip(): continue
    r=json.loads(line)
    for a in r.get("answers",[]):
        if a.get("ck"): uids[a["ck"]].add(a["uid"])
for ans in rounds:
    for ck in {c for c,_ in ans}: ck_rounds[ck]+=1
big=set(ck for ck,n in ck_rounds.items() if n>=200)

obs=collections.Counter(); exp=collections.defaultdict(float)
for ans in rounds:
    cnt=collections.Counter(c for _,c in ans); n=len(ans)
    if n<2: continue
    p=sum(v*(v-1) for v in cnt.values())/(n*(n-1))
    bck=collections.defaultdict(list)
    for ck,c in ans:
        if ck in big: bck[ck].append(c)
    for a,b in itertools.combinations(sorted(bck),2):
        exp[(a,b)]+=p*len(bck[a])*len(bck[b])
        sb=collections.Counter(bck[b]); obs[(a,b)]+=sum(sb[c] for c in bck[a])

STRONG=200.0
coord=[(a,b) for (a,b),e in exp.items() if e>=STRONG and obs[(a,b)]/e<0.05]
weak=[(a,b) for (a,b),e in exp.items() if e<STRONG and obs[(a,b)]/e<0.05]
print(f"high-power pairs (expected>=200): {sum(1 for e in exp.values() if e>=STRONG)}")
print(f"  of those, NEVER-COLLIDE (ratio<0.05): {len(coord)}  <- coordinated")
print(f"low-power pairs called never-collide but unconvincing: {len(weak)}\n")

par={}
def find(x):
    while par.setdefault(x,x)!=x: par[x]=par[par[x]]; x=par[x]
    return x
def union(a,b):
    ra,rb=find(a),find(b)
    if ra!=rb: par[ra]=rb
for a,b in coord: union(a,b)
groups=collections.defaultdict(list)
for ck in big: groups[find(ck)].append(ck)
groups={k:v for k,v in groups.items() if len(v)>1}
print(f"{len(groups)} coordinated operator groups among {len(big)} active wallets:\n")
tot=0
for i,(_,members) in enumerate(sorted(groups.items(),key=lambda kv:-len(kv[1])),1):
    nu=len(set().union(*[uids[m] for m in members]))
    tot+=nu
    print(f"  group {i}: {len(members):>2} wallets, {nu:>3} UIDs")
    for m in members: print(f"      {m[:16]}..  {len(uids[m]):>3} UIDs, {ck_rounds[m]:>6} rounds")
solo=[c for c in big if find(c) not in groups]
print(f"\n  {len(solo)} wallets in no coordinated group, "
      f"{len(set().union(*[uids[c] for c in solo])) if solo else 0} UIDs")
print(f"\ncoordinated UIDs: {tot} of {len(set().union(*uids.values()))} seen")
