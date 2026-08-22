"""Is the field getting stronger over time? A deployment question, not a solver one.

We are at 2.4389 against a field per-UID median of 2.5304. If that bar has been
rising, the deficit will widen after deployment and any decision based on today's gap
is optimistic. If it is flat, the gap is a stable target.

Pure data -- no solving. The corpus spans 33k rounds in logging order, so bucket by
position and recompute the validator's reward for every field answer in each bucket.
"""
import json, collections, os, sys
sys.path.insert(0,'/workspace/better_83/research')
import numpy as np
from _common import DATA_DIR

rounds=[]
for line in open(os.path.join(DATA_DIR,'sim_rounds.jsonl')):
    if not line.strip(): continue
    r=json.loads(line)
    if r.get("answers"): rounds.append(r)
rounds.sort(key=lambda r:(r.get("_run",""), r.get("_step",0)))
print(f"{len(rounds)} rounds in logging order\n")

NB=8
size=len(rounds)//NB
print(f"{'bucket':>7} {'rounds':>7} {'per-ANSWER':>11} {'per-UID mean':>13} "
      f"{'best-size div':>14} {'uniq@best':>10} {'n_valid':>8}")
for b in range(NB):
    chunk=rounds[b*size:(b+1)*size]
    per_ans=[]; per_uid=collections.defaultdict(list); bdiv=[]; uq=[]; nv=[]
    for r in chunk:
        A=[a for a in r.get("answers",[]) if a.get("clique") and a.get("opt",0)>0]
        if len(A)<5: continue
        D=r["difficulty"]; nresp=len(r.get("answers",[]))
        sz=np.array([len(a["clique"]) for a in A],dtype=float)
        keys=[tuple(sorted(a["clique"])) for a in A]
        mx=sz.max(); rel=sz/mx
        pr=np.array([(sz>s).sum()/nresp for s in sz])
        om=np.exp(-pr/np.maximum(rel,1e-9)); opt=om/om.max()
        c=collections.Counter(keys); md=max(1.0/v for v in c.values())
        div=np.array([(1.0/c[k])/md for k in keys])
        rew=opt*(1+D)+div
        per_ans.extend(rew.tolist())
        for a,x in zip(A,rew): per_uid[a["uid"]].append(float(x))
        bd=[d for d,s_ in zip(div,sz) if s_==mx]
        bdiv.extend(bd); uq.append(float(np.mean([1.0 if c[k]==1 else 0.0
                                                  for k,s_ in zip(keys,sz) if s_==mx])))
        nv.append(len(A))
    um=np.array([np.mean(v) for v in per_uid.values() if len(v)>=20])
    print(f"{b+1:>7} {len(chunk):>7} {np.mean(per_ans):>11.4f} {um.mean():>13.4f} "
          f"{np.mean(bdiv):>14.4f} {np.mean(uq):>10.1%} {np.mean(nv):>8.1f}")
print("\nour shipped solver: 2.4389 (recent_val, corrected scorer)")
