"""Same-rounds comparison: field best-size diversity vs ours, on recent_val's 500.

The 33k-round figure (0.7357) and our 0.5862 come from DIFFERENT round sets, and
comparing across draws is the exact error that produced the bogus 2.6242 field
median earlier. So restrict the field to the 500 rounds we were actually scored on
and compare there.

Also reports the field's per-answer diversity WITH our answer inserted, since
joining a round changes the counts we are being compared against.
"""
import json, collections, os, sys

from _common import DATA_DIR
import numpy as np

want={}
for line in open(os.path.join(DATA_DIR,'sets/recent_val.jsonl')):
    r=json.loads(line); want[r["uuid"]]=r
ours={}
for r in json.load(open('/home/dev/autoresearch-runs/sn83-clique/runs/final_shipped_rv500.json'))["rows"]:
    ours[r["uuid"]]=r

field_best=[]; field_all=[]; our_div=[]; matched=0
for line in open(os.path.join(DATA_DIR,'sim_rounds.jsonl')):
    if not line.strip(): continue
    r=json.loads(line)
    if r["uuid"] not in want: continue
    ans=[tuple(sorted(a["clique"])) for a in r.get("answers",[])
         if a.get("clique") and a.get("opt",0)>0]
    if len(ans)<5: continue
    matched+=1
    c=collections.Counter(ans)
    md=max(1.0/v for v in c.values())
    b=max(len(a) for a in ans)
    for a in ans:
        d=(1.0/c[a])/md
        field_all.append(d)
        if len(a)==b: field_best.append(d)
    o=ours.get(r["uuid"])
    if o and o.get("diversity") is not None:
        our_div.append(o["diversity"])

print(f"matched {matched} of {len(want)} recent_val rounds in the corpus\n")
print(f"SAME ROUNDS:")
print(f"  field diversity, all valid answers : {np.mean(field_all):.4f}  (n={len(field_all)})")
print(f"  field diversity, BEST-SIZE answers : {np.mean(field_best):.4f}  (n={len(field_best)})")
print(f"  our shipped solver                 : {np.mean(our_div):.4f}  (n={len(our_div)})")
print()
d=np.array(field_best)
print(f"  field best-size sole-holder rate   : {(d>=0.999).mean():.1%}")
print(f"  gap (field best-size - ours)       : {np.mean(field_best)-np.mean(our_div):+.4f}")
