"""End-to-end: what does the hull pool do to our FLEET MEDIAN vs the field median?

Converts a hullpool_*.json into the sim_ts cache format fleet_sim expects, then replays
the real round stream. This is the user's actual objective -- "beat the field median by
our median" -- rather than a per-clique proxy.
"""
import json, sys, collections, statistics as st
sys.path.insert(0, '/workspace/better_83/research')
import numpy as np
import fleet_sim as F, fleet_pick as fp

pool_path = sys.argv[1]
K = int(sys.argv[2]) if len(sys.argv) > 2 else 8

base = {}
for line in open('data/sim_ts.jsonl'):
    r = json.loads(line)
    base[r['uuid']] = r

hull = json.load(open(pool_path))
merged = {}
for u, rec in base.items():
    if u in hull and hull[u]:
        merged[u] = {'uuid': u, 'cliques': hull[u], 'elapsed': rec.get('elapsed', 0)}
    else:
        merged[u] = rec
tmp = '/home/dev/.claude/jobs/5443dea7/tmp/hull_cache.jsonl'
with open(tmp, 'w') as f:
    for u in merged:
        f.write(json.dumps(merged[u], separators=(',', ':')) + '\n')

rounds = [json.loads(l) for l in open('data/sim_rounds.jsonl')]
rounds = [r for r in rounds if r.get('answers') and r.get('timestamp')]
rounds.sort(key=lambda r: r['timestamp'])
cache = {}
for line in open(tmp):
    r = json.loads(line); cache[r['uuid']] = r
solved = [r for r in rounds if r['uuid'] in cache]
meta = json.load(open('data/metagraph.json'))
validity = {r['uuid']: F.validate_cliques(r, cache[r['uuid']]['cliques']) for r in solved}

# field per-hotkey median on the SAME rounds
fs = collections.defaultdict(list)
for rec in solved:
    for a in rec['answers']:
        if 'reward' in a:
            fs[a['hk']].append(a['reward'])
fmed = st.median([st.mean(v) for v in fs.values() if len(v) >= 50])

covered = sum(1 for u in hull if u in base and hull[u])
print("pool source %s  (%d of %d cached rounds replaced)"
      % (pool_path.split('/')[-1], covered, len(base)))
print("%6s %13s %14s %9s" % ("N", "our median", "field median", "gap"))
for N in (1, 5, 10, 20, 40):
    stx, vu, oid, al, ev, hk = F.replay(solved, cache, meta, N, 8383, validity,
                                        sample='real', picker=fp.picker)
    per = [np.mean(v) for v in (stx[o] for o in oid) if len(v)]
    m = float(np.median(per))
    print("%6d %13.4f %14.4f %+9.4f" % (N, m, fmed, m - fmed))
