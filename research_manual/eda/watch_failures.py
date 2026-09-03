#!/usr/bin/env python3
"""Follow a pool dump and report coverage failures as they are written.

A failure is a round where some miner submitted an omega-clique our harvest does
not hold. The solver cannot detect this itself -- the dump's `answers` are our
own picks, and the field's answers live in rounds.json -- so the check runs here,
against the file the run is still appending to.

    python research_manual/eda/watch_failures.py pool_le127.jsonl [--once] [--closure]

--once     scan what exists and exit, instead of following
--closure  also report what an unbounded offline fixpoint would recover, which
           separates "the search never found it" from "the budget cut it short"
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
REPO = os.path.dirname(PARENT)
for _p in (REPO, PARENT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

ROUNDS = os.path.join(PARENT, "rounds.json")
HDR = ("  %-8s %4s %5s %6s %6s %7s %8s %6s   %s"
       % ("round", "D", "omega", "S_true", "n_top", "missing", "swapdist",
          "iters", "note"))


def field_cliques(rec):
    a = [x for x in rec["answers"] if x[3]]
    if not a:
        return None, 0
    keys = [tuple(sorted(x[3])) for x in a]
    M = max(len(k) for k in keys)
    return {k for k in keys if len(k) == M}, M


def check(rec, p, want_closure):
    field, M = field_cliques(rec)
    if not field or p["omega"] != M:
        return None
    held = {tuple(sorted(c)) for c in p["full_pool_unverified"] if len(c) == M}
    miss = field - held
    if not miss:
        return None
    dist = min(M - max(len(set(m) & set(h)) for h in held) for m in miss) if held else M
    note = ""
    if want_closure:
        import harvest_probe as hp
        import fleet_solver_gpu as F
        A = hp.adjacency(rec)
        grown, iters = F.closure_fixpoint(A, held, M, time.monotonic() + 30.0)
        got = len(miss & grown)
        note = ("unbounded fixpoint recovers %d/%d in %d iters"
                % (got, len(miss), iters))
    return dict(uuid=p["uuid"], d=rec["difficulty"], omega=M,
                s_true=len(held | field), n_top=len(held), missing=len(miss),
                dist=dist, iters=p.get("closure_iters", 0), note=note)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--closure", action="store_true")
    args = ap.parse_args()
    path = args.dump if os.path.isabs(args.dump) else os.path.join(PARENT, args.dump)
    rounds = json.load(open(ROUNDS))

    seen = 0
    fails = 0
    printed_header = False
    while True:
        if not os.path.exists(path):
            if args.once:
                return
            time.sleep(5)
            continue
        with open(path) as handle:
            for _ in range(seen):
                handle.readline()
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                seen += 1
                p = json.loads(line)
                rec = rounds.get(p["uuid"])
                if not rec:
                    continue
                bad = check(rec, p, args.closure)
                if not bad:
                    continue
                fails += 1
                if not printed_header:
                    print(HDR, flush=True)
                    printed_header = True
                print("  %-8s %4.1f %5d %6d %6d %7d %8d %6d   %s"
                      % (bad["uuid"][:8], bad["d"], bad["omega"], bad["s_true"],
                         bad["n_top"], bad["missing"], bad["dist"], bad["iters"],
                         bad["note"]), flush=True)
        if args.once:
            break
        time.sleep(10)
    print("scanned %d rounds, %d failures" % (seen, fails), flush=True)


if __name__ == "__main__":
    main()
