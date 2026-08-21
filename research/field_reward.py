"""Field per-answer AND per-UID reward on the same 483 rounds we were scored on.

Two comparators, because they answer different questions:
  per-ANSWER  what a typical submitted answer earns   (the earlier bogus 2.6242)
  per-UID     what a typical MINER earns on average   (the right comparator, ~2.50)
Ours: 2.4389 on these rounds.
"""
import json, collections, os, sys

from _common import DATA_DIR
import numpy as np

want=set()
for line in open(os.path.join(DATA_DIR,'sets/recent_val.jsonl')):
    want.add(json.loads(line)["uuid"])

per_ans=[]; per_uid=collections.defaultdict(list); n=0
for line in open(os.path.join(DATA_DIR,'sim_rounds.jsonl')):
    if not line.strip(): continue
    r=json.loads(line)
    if r["uuid"] not in want: continue
    A=[a for a in r.get("answers",[]) if a.get("clique") and a.get("opt",0)>0]
    if len(A)<5: continue
    n+=1
    D=r["difficulty"]; nresp=len(r.get("answers",[]))
    sizes=np.array([len(a["clique"]) for a in A],dtype=float)
    keys=[tuple(sorted(a["clique"])) for a in A]
    mx=sizes.max(); rel=sizes/mx
    pr=np.array([(sizes>s).sum()/nresp for s in sizes])
    om=np.exp(-pr/np.maximum(rel,1e-9)); opt=om/om.max()
    c=collections.Counter(keys); md=max(1.0/v for v in c.values())
    div=np.array([(1.0/c[k])/md for k in keys])
    rew=opt*(1+D)+div
    for a,x in zip(A,rew):
        per_ans.append(float(x)); per_uid[a["uid"]].append(float(x))

um=np.array([np.mean(v) for v in per_uid.values() if len(v)>=30])
print(f"{n} rounds\n")
print(f"field per-ANSWER reward : mean {np.mean(per_ans):.4f}  median {np.median(per_ans):.4f}  (n={len(per_ans)})")
print(f"field per-UID   reward : mean {um.mean():.4f}  median {np.median(um):.4f}  (n={len(um)} UIDs, >=30 answers)")
print(f"our shipped solver      : 2.4389")
print()
print(f"  UID reward distribution: p10 {np.percentile(um,10):.4f}  p25 {np.percentile(um,25):.4f} "
      f"p50 {np.median(um):.4f}  p75 {np.percentile(um,75):.4f}  p90 {np.percentile(um,90):.4f}")
print(f"  UIDs we would beat at 2.4389: {(um<2.4389).sum()}/{len(um)} ({(um<2.4389).mean():.1%})")
