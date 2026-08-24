#!/usr/bin/env python3
"""SEARCH x SPREAD-POLICY, scored with the exact validator reward.

The two findings of this generation compose, so measure them together:
  search  supplies the distinct maxima AND the sensor (its own distinct count)
  policy  decides when to answer below omega instead

Spares come from the shipped bf cache, not from the arm, because the arms were run
with SN83_BACKFILL=0 and hold only max-size cliques. That keeps the spare supply
IDENTICAL across arms, so any difference is the search and the policy, not the spares.

  P0_short   pad with spares whenever the pool is shorter than Q   (what ships)
  P1_sensor  spread only when the arm's own distinct count <= 3
  P2_oracle  spread only when nOm <= 10                            (upper bound)
"""
import argparse, json, collections, statistics as st, os, sys, math
sys.path.insert(0, "/workspace/better_83/research")
from tools.round_score import score

def build(top, spare, Q, spread):
    """spread=True -> one omega plus distinct spares; False -> round-robin the maxima."""
    if spread and spare:
        wm1 = len(top[0]) - 1
        out = [(len(top[0]), top[0])]
        i = 0
        while len(out) < Q and i < len(spare):
            k = spare[i]
            out.append((wm1 if isinstance(k[0], str) else len(k), k)); i += 1
    else:
        out = [(len(c), c) for c in top[:Q]]
    while len(out) < Q:
        out.append((len(top[0]), top[len(out) % len(top)]))
    return out[:Q]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+")
    ap.add_argument("--spares", default="data/sim_cliques_bf.jsonl")
    ap.add_argument("--rounds", default="data/sim_rounds.jsonl")
    ap.add_argument("-Q", type=int, default=7)
    a = ap.parse_args()
    rounds = {}
    for l in open(a.rounds):
        r = json.loads(l); rounds[r["uuid"]] = r
    # Spare supply. The arms were run with SN83_BACKFILL=0, so spares must come from
    # somewhere else and MUST be identical across arms.
    #   novel  synthetic unique omega-1 keys -- an upper bound, but our real spares
    #          are 89.4% novel so the truth sits near it
    #   field  the round's own logged omega-1 answers -- a lower bound, since those
    #          are by construction cliques the field already holds
    #   <path> a solve cache, when one covers the same rounds
    spares = {}
    if a.spares in ("novel", "field"):
        for u, r in rounds.items():
            b = r["best_size"]
            if a.spares == "novel":
                spares[u] = [("SP", u, i) for i in range(32)]
            else:
                spares[u] = sorted({tuple(sorted(x["clique"])) for x in r["answers"]
                                    if len(x["clique"]) == b - 1})
    else:
        for l in open(a.spares):
            r = json.loads(l)
            cl = [tuple(sorted(c)) for c in r.get("cliques", [])]
            if not cl: continue
            mx = max(len(c) for c in cl)
            spares[r["uuid"]] = sorted({c for c in cl if len(c) < mx}, key=len, reverse=True)
    pools = {}
    for p in a.arms:
        d = {}
        for l in open(p):
            r = json.loads(l)
            cl = [tuple(sorted(c)) for c in r.get("cliques", [])]
            if cl and r["uuid"] in rounds: d[r["uuid"]] = cl
        pools[os.path.basename(p).replace("arm_", "").replace(".jsonl", "")] = d
    common = sorted(set.intersection(*[set(d) for d in pools.values()]) & set(spares))
    common = [u for u in common if spares[u]]
    print("rounds %d   Q=%d   spares from %s\n" % (len(common), a.Q, os.path.basename(a.spares)))
    print("%-8s %12s %10s %10s %10s" % ("arm", "policy", "our rw", "field rw", "GAP"))
    res = {}
    for arm, d in pools.items():
        for pol in ("P0_short", "P1_sensor", "P2_oracle"):
            ours = []; fld = []; fires = 0
            for u in common:
                rec = rounds[u]; b = rec["best_size"]
                fk = [tuple(sorted(x["clique"])) for x in rec["answers"]]
                sizes = [len(k) for k in fk]
                nOm = sum(1 for s in sizes if s == b)
                cl = d[u]; mx = max(len(c) for c in cl)
                top = sorted({c for c in cl if len(c) == mx})
                sp = spares[u]
                if pol == "P0_short":   spread = len(top) < a.Q
                elif pol == "P1_sensor": spread = len(top) <= 3
                else:                    spread = nOm <= 10
                fires += spread
                mine = build(top, sp, a.Q, spread)
                keys = fk + [k for _, k in mine]
                szs = sizes + [s for s, _ in mine]
                _, _, rw = score(szs, keys, rec["difficulty"])
                ours.append(st.mean(rw[len(fk):])); fld.append(st.mean(rw[:len(fk)]))
            o, f = st.mean(ours), st.mean(fld)
            res[(arm, pol)] = (o, f, o - f, fires / len(common))
            print("%-8s %12s %10.4f %10.4f %+10.4f   spreads %.0f%%"
                  % (arm, pol, o, f, o - f, 100 * fires / len(common)))
        print()

if __name__ == "__main__":
    main()
