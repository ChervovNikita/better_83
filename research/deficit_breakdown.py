"""Where does our 0.09 reward deficit live? Broken down by instance type.

We sit at 2.4389 against a field per-UID median of 2.5304 on the same 483 rounds.
That is an aggregate. If the deficit is concentrated -- say only on large graphs, or
only at short deadlines -- there may be a regime where we are already competitive,
and the fleet/deployment decision could differ by regime. If it is uniform, there is
no such refuge and the aggregate is the whole story.

Field reward is recomputed per round with the validator's formula so both sides come
from the same rounds and the same code path.
"""
import json, collections, os, sys
sys.path.insert(0,'/workspace/better_83/research')
import numpy as np
from _common import DATA_DIR

ours={r["uuid"]: r for r in json.load(
    open('/home/dev/autoresearch-runs/sn83-clique/runs/final_shipped_rv500.json'))["rows"]}
meta={}
for line in open(os.path.join(DATA_DIR,'sets/recent_val.jsonl')):
    r=json.loads(line); meta[r["uuid"]]=r

rows=[]
for line in open(os.path.join(DATA_DIR,'sim_rounds.jsonl')):
    if not line.strip(): continue
    r=json.loads(line)
    o=ours.get(r["uuid"]); m=meta.get(r["uuid"])
    if not o or not m or o.get("reward") is None: continue
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
    fr=float(np.mean(opt*(1+D)+div))
    rows.append(dict(n=r["n"], tl=r["time_limit"], d=D, ours=o["reward"], field=fr,
                     gap=o["reward"]-fr, collide=o.get("collide")))

print(f"{len(rows)} rounds matched\n")
g=np.array([x["gap"] for x in rows])
print(f"overall: ours {np.mean([x['ours'] for x in rows]):.4f}  "
      f"field {np.mean([x['field'] for x in rows]):.4f}  gap {g.mean():+.4f}  "
      f"(median {np.median(g):+.4f})")
print(f"  rounds where we BEAT the field average: {int((g>0).sum())}/{len(g)} ({(g>0).mean():.1%})")
def bucket(key, label, fn=lambda v: v):
    print(f"\n  by {label}:")
    b=collections.defaultdict(list)
    for x in rows: b[fn(x[key])].append(x["gap"])
    for k in sorted(b):
        v=np.array(b[k])
        print(f"    {str(k):>8}: gap {v.mean():+.4f}  median {np.median(v):+.4f}  "
              f"beat {100*(v>0).mean():>5.1f}%  (n={len(v)})")
bucket("tl","deadline")
bucket("d","difficulty")
bucket("n","|V|", lambda v: (v//200)*200)
print("\n  by collision count (how many field miners held our clique):")
b=collections.defaultdict(list)
for x in rows: b[min(int(x["collide"] or 0),4)].append(x["gap"])
for k in sorted(b):
    v=np.array(b[k]); lab=f"{k}" if k<4 else "4+"
    print(f"    {lab:>8}: gap {v.mean():+.4f}  beat {100*(v>0).mean():>5.1f}%  (n={len(v)})")
