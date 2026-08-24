#!/usr/bin/env python3
"""Score a solver arm's real pool with the EXACT validator reward.

Inserts Q answers, picked from the arm's pool the way the picker does (distinct
maxima first, then omega-1 spares, then repeats), into the LOGGED round and scores
every response with tools/round_score.py. Reports OUR mean reward, the FIELD's mean
reward in that same world, and the gap.

Arms are compared only on the INTERSECTION of the rounds they all cover -- comparing
means over different round sets has produced two false results in this project.
"""
import argparse, json, collections, statistics as st, os, sys
sys.path.insert(0, "/workspace/better_83/research")
from tools.round_score import score

def plan(cl, Q):
    mx = max(len(c) for c in cl)
    top = sorted({c for c in cl if len(c) == mx})
    spare = sorted({c for c in cl if len(c) < mx}, key=len, reverse=True)
    out = [(mx, c) for c in top[:Q]]
    i = 0
    while len(out) < Q and i < len(spare):
        out.append((len(spare[i]), spare[i])); i += 1
    while len(out) < Q:
        out.append((mx, top[len(out) % len(top)]))
    return out[:Q]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+")
    ap.add_argument("--rounds", default="data/sim_rounds.jsonl")
    ap.add_argument("-Q", type=int, default=7)
    a = ap.parse_args()
    rounds = {}
    for l in open(a.rounds):
        r = json.loads(l); rounds[r["uuid"]] = r
    pools = {}
    for p in a.arms:
        d = {}
        for l in open(p):
            r = json.loads(l)
            cl = [tuple(sorted(c)) for c in r.get("cliques", [])]
            if cl and r["uuid"] in rounds: d[r["uuid"]] = cl
        pools[os.path.basename(p)] = d
    common = set.intersection(*[set(d) for d in pools.values()]) if pools else set()
    print("arms %d   common rounds %d   Q=%d\n" % (len(pools), len(common), a.Q))
    print("%-18s %9s %9s %9s %8s %8s" % ("arm", "our rw", "field rw", "GAP", "distinct", "novel%"))
    base = None
    for name, d in pools.items():
        ours = []; fld = []; nd = []; nov = []
        for u in sorted(common):
            rec = rounds[u]
            fk = [tuple(sorted(x["clique"])) for x in rec["answers"]]
            mine = plan(d[u], a.Q)
            keys = fk + [k for _, k in mine]
            sizes = [len(k) for k in fk] + [s for s, _ in mine]
            _, _, rw = score(sizes, keys, rec["difficulty"])
            ours.append(st.mean(rw[len(fk):])); fld.append(st.mean(rw[:len(fk)]))
            mx = max(len(c) for c in d[u])
            top = {c for c in d[u] if len(c) == mx}
            nd.append(len(top)); nov.append(sum(1 for c in top if c not in set(fk)) / len(top))
        o = st.mean(ours); f = st.mean(fld)
        if base is None: base = o
        print("%-18s %9.4f %9.4f %+9.4f %8.2f %7.1f%%"
              % (name, o, f, o - f, st.mean(nd), 100 * st.mean(nov)))

if __name__ == "__main__":
    main()
