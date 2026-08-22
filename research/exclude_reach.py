#!/usr/bin/env python3
"""Force the search out of its basin by EXCLUDING vertices of the clique it found.

Plateau harvesting and relabelling both stay inside one basin -- measured, they return
the same maxima. Excluding a vertex v of the found clique makes that clique
unreachable, so the solver must land on a maximum clique that avoids v. If omega is
still attainable without v, the result is a maximum clique in a DIFFERENT region by
construction, not by luck.

This differs from the refuted "vertex banning": that banned vertices permanently and
lost clique size (p=0.0028 worse on reward). Here each exclusion is a separate solve
and any result below omega is discarded, so size can never be traded away.
"""
import os

import numpy as np


def reach(A, time_limit, k, omega=None, base=None, threads=None):
    """Distinct maximum cliques found by excluding one vertex at a time."""
    import fleetsolver
    A = np.ascontiguousarray(np.asarray(A, dtype=np.uint8))
    n = A.shape[0]
    threads = threads or int(os.environ.get("SN83_THREADS", "13"))
    share = float(os.environ.get("SN83_EXCL_SHARE", "0.35"))   # budget for the seed
    if base is None:
        cl = fleetsolver.solve_many(A, time_limit * share, k, threads=threads)
        if not cl:
            return []
        omega = max(len(c) for c in cl)
        base = [tuple(sorted(c)) for c in cl if len(c) == omega]
    pool = dict.fromkeys(base)
    seed = list(base[0])
    left = time_limit * (1.0 - share)
    per = max(0.05, left / max(len(seed), 1))
    for v in seed:
        keep = np.array([i for i in range(n) if i != v])
        B = np.ascontiguousarray(A[np.ix_(keep, keep)])
        try:
            cl2 = fleetsolver.solve_many(B, per, k, threads=threads)
        except Exception:
            continue
        if not cl2:
            continue
        m = max(len(c) for c in cl2)
        if m < omega:
            continue                     # never trade size for reach
        for c in cl2:
            if len(c) == m:
                pool.setdefault(tuple(sorted(int(keep[u]) for u in c)), None)
        if len(pool) >= k:
            break
    return [list(t) for t in pool]


if __name__ == "__main__":
    import json
    import sys
    import time
    HERE = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
    from CliqueAI.graph.codec import GraphCodec
    from _common import DATA_DIR
    import fleetsolver
    os.environ.setdefault("SN83_THREADS", "13")
    rounds = {}
    for line in open(os.path.join(DATA_DIR, "sim_rounds.jsonl")):
        if line.strip():
            r = json.loads(line); rounds[r["uuid"]] = r
    rs = [r for r in rounds.values() if r.get("answers") and r.get("timestamp")]
    rs.sort(key=lambda r: r["timestamp"])
    N = int(os.environ.get("ER_N", "12"))
    tot_l, tot_e, tot_new, tot_free_l, tot_free_e = [], [], [], [], []
    for r in rs[-216:][:N]:
        A = np.ascontiguousarray(np.array(GraphCodec().decode_matrix(r["matrix_b92"]),
                                          dtype=np.uint8))
        tl = max(0.5, r["time_limit"] - 2.0)
        cl = fleetsolver.solve_many(A, tl, 40, threads=13)
        if not cl:
            continue
        om = max(len(c) for c in cl)
        lscc = {tuple(sorted(c)) for c in cl if len(c) == om}
        ex = {tuple(sorted(c)) for c in reach(A, tl, 40, threads=13)}
        if not ex:
            continue
        Aw = [a for a in r["answers"] if a.get("clique") and a.get("opt", 0) > 0]
        sz = [len(a["clique"]) for a in Aw]
        fld = {tuple(sorted(a["clique"])) for a, s in zip(Aw, sz) if s == max(sz)} \
            if Aw and max(sz) == om else set()
        tot_l.append(len(lscc)); tot_e.append(len(ex)); tot_new.append(len(ex - lscc))
        if fld:
            tot_free_l.append(len([c for c in lscc if c not in fld]))
            tot_free_e.append(len([c for c in ex if c not in fld]))
    print(f"\n=== exclusion reach, {len(tot_l)} rounds, same total budget ===")
    print(f"  LSCC harvest pool      : mean {np.mean(tot_l):6.1f}")
    print(f"  EXCLUSION pool         : mean {np.mean(tot_e):6.1f}")
    print(f"  cliques NEW vs LSCC    : mean {np.mean(tot_new):6.1f}  "
          f"({100*np.mean([a/max(b,1) for a,b in zip(tot_new,tot_e)]):.0f}% of its pool)")
    if tot_free_l:
        print(f"  field-free in LSCC pool: mean {np.mean(tot_free_l):6.1f}")
        print(f"  field-free in EXCL pool: mean {np.mean(tot_free_e):6.1f}")
