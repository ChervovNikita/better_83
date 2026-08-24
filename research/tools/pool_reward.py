#!/usr/bin/env python3
"""Score a POOL against the logged round it came from.

Places Q of our hotkeys on the pool's distinct max-size cliques (round-robin when
the pool is short, which is the picker's real behaviour) and computes the diversity
each answer would earn INSIDE the logged round:

    diversity = (1 / holders) / max_delta,  holders = field holders + our siblings

max_delta is recovered per round from the logged answers (div * holders = 1/max_delta);
it is 1.0 on 95% of rounds. The field's own mean diversity on the same rounds is
printed alongside as the target.

This is a counterfactual, not a simulation: it inserts our answers into the real
round and re-derives holders. It does NOT model the field reacting, and it does not
model `pr` (the strictly-larger count), so it is a diversity comparison only. Use it
to rank POOLS, not to predict rank on chain.

ROUND-SET WARNING. Rounds where the pool fails to reach omega are skipped, because
otherwise the distinct-max count is taken at two different sizes. That makes the
per-file numbers NOT comparable across caches with different reach: a cache that
reaches omega on 43% of rounds is scored on its easy 43%. Compare caches only on the
INTERSECTION of their rounds. Doing this caught a false +0.13 on 2026-08-24 -- paired
on common rounds the same two caches differ by -0.0019.
"""
import argparse, json, collections, statistics as st, sys, os

def load_field(path):
    fld = {}
    for l in open(path):
        r = json.loads(l)
        b = r["best_size"]
        cnt = collections.Counter(tuple(sorted(a["clique"])) for a in r["answers"])
        v = [a["div"] * cnt[tuple(sorted(a["clique"]))] for a in r["answers"] if a["div"]]
        fld[r["uuid"]] = {"best": b, "cnt": cnt,
                          "md": (1.0 / st.median(v)) if v else 1.0,
                          "fdiv": st.mean([a["div"] for a in r["answers"]])}
    return fld

def score(pool, meta, Q):
    top = sorted({tuple(sorted(c)) for c in pool if len(c) == max(len(x) for x in pool)})
    if not top:
        return None
    use = [top[i % len(top)] for i in range(Q)]
    mine = collections.Counter(use)
    out = []
    for c, k in mine.items():
        hold = meta["cnt"].get(c, 0) + k
        out.append(((1.0 / hold) / meta["md"]) * k)
    return sum(out) / Q

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cache", nargs="+", help="solve-cache JSONL with uuid+cliques")
    ap.add_argument("--rounds", default="data/sim_rounds.jsonl")
    ap.add_argument("-Q", type=int, default=7, help="our hotkeys queried per round")
    a = ap.parse_args()
    fld = load_field(a.rounds)
    print("%-26s %7s %10s %10s %10s" % ("pool", "rounds", "our div", "field div", "gap"))
    for path in a.cache:
        ours = []; fdv = []
        for l in open(path):
            r = json.loads(l); u = r["uuid"]
            if u not in fld: continue
            cl = [tuple(sorted(c)) for c in r.get("cliques", [])]
            if not cl: continue
            if max(len(c) for c in cl) != fld[u]["best"]: continue
            s = score(cl, fld[u], a.Q)
            if s is None: continue
            ours.append(s); fdv.append(fld[u]["fdiv"])
        if not ours:
            print("%-26s %7d" % (os.path.basename(path), 0)); continue
        print("%-26s %7d %10.4f %10.4f %+10.4f"
              % (os.path.basename(path), len(ours), st.mean(ours), st.mean(fdv),
                 st.mean(ours) - st.mean(fdv)))

if __name__ == "__main__":
    main()
