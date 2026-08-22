"""What un-taken rate would we need to reach the field median? Turn the gap into a spec.

We are at 2.4389 against a field per-UID median of 2.5304 on the same 483 rounds.
Saying "close a 0.09 gap" is not actionable. What IS actionable: our diversity is a
function of how often our answer is un-taken, so invert it -- hold everything else
fixed at what we actually achieve and ask what freshness rate lands us at the median.

Method: take our real per-round outcomes and resample the collision counts so that a
target fraction of rounds are un-taken, drawing the non-fresh cases from our own
observed collision distribution. Everything else (sizes, field histograms, difficulty)
stays exactly as measured.
"""
import json, os, sys
sys.path.insert(0,'/workspace/better_83/research')
import numpy as np
import bench
from _common import DATA_DIR

meta={}
for line in open(os.path.join(DATA_DIR,'sets/recent_val.jsonl')):
    r=json.loads(line); meta[r["uuid"]]=r
rows=[r for r in json.load(open(
    '/home/dev/autoresearch-runs/sn83-clique/runs/final_shipped_rv500.json'))["rows"]
      if r.get("reward") is not None and r["uuid"] in meta]

obs=[int(r.get("collide") or 0) for r in rows]
nonzero=[c for c in obs if c>0]
base=float(np.mean([1 if c==0 else 0 for c in obs]))
print(f"{len(rows)} rounds | measured un-taken rate {base:.1%}\n")
print(f"{'un-taken':>9} {'reward':>9} {'diversity':>10}  {'vs field median 2.5304':>24}")
rng=np.random.default_rng(7)
for target in (0.343,0.40,0.45,0.50,0.55,0.60,0.70,0.85,1.00):
    R=[];D=[]
    for _ in range(6):
        rw=[];dv=[]
        for r in rows:
            m=meta[r["uuid"]]
            c=0 if rng.random()<target else int(rng.choice(nonzero))
            a=bench.replay_reward(r["ours"], c, m.get("size_hist"),
                                  m.get("best_clique_counts"), m.get("difficulty",0.85),
                                  m.get("any_unique"), m.get("n_responders"),
                                  count_hist=m.get("count_hist"))
            rw.append(a[0]); dv.append(a[2])
        R.append(np.mean(rw)); D.append(np.mean(dv))
    r_=float(np.mean(R)); d_=float(np.mean(D))
    flag = "<- reaches median" if r_>=2.5304 else ""
    print(f"{target:>8.1%} {r_:>9.4f} {d_:>10.4f}  {r_-2.5304:>+24.4f} {flag}")
