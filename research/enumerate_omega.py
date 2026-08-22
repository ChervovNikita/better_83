#!/usr/bin/env python3
"""Enumerate ALL cliques of a known size omega -- a different reachable set entirely.

Our LSCC harvester returns the maxima one basin reaches: 18.7 per round, of which the
field already holds 9.6. On 54.8% of rounds every clique we find is taken. Selection
cannot fix that (within-round predictors are all ~0); only a bigger pool can.

Knowing omega changes the problem. A branch whose |R| + |P| < omega can never reach
omega and is cut immediately, so this is not "enumerate all maximal cliques" (hopeless
on dense graphs) but a targeted search with a very strong bound. Combined with greedy
colouring -- if the colour count of P plus |R| is below omega, prune -- dense
instances become tractable in seconds.

If the complete omega-clique set is much larger than 18.7, the field cannot hold all
of it, and picking any clique outside their holdings becomes possible.
"""
import sys
import time


def enumerate_omega(adj, n, omega, deadline, cap=2000):
    """All cliques of size exactly omega, as frozensets of vertex indices.

    adj is a list of int bitmasks. Returns (cliques, complete) where complete is
    False if the deadline or cap stopped the search early.
    """
    found = []
    full = (1 << n) - 1
    state = {"complete": True}

    def expand(R, P, size):
        if state["complete"] is False and not found:
            return
        if time.time() > deadline or len(found) >= cap:
            state["complete"] = False
            return
        if size == omega:
            found.append(R)
            return
        # bound: not enough candidates left to reach omega
        need = omega - size
        if bin(P).count("1") < need:
            return
        # colouring bound: greedy colour classes of P; a clique uses at most one
        # vertex per class, so #classes < need means this branch cannot reach omega
        classes = 0
        rest = P
        while rest:
            classes += 1
            if classes >= need:
                break
            avail = rest
            while avail:
                v = (avail & -avail).bit_length() - 1
                avail &= ~(1 << v)
                avail &= ~adj[v]
                rest &= ~(1 << v)
        if classes < need:
            return
        while P:
            if bin(P).count("1") < need:
                return
            v = (P & -P).bit_length() - 1
            P &= ~(1 << v)
            expand(R | (1 << v), P & adj[v], size + 1)
            if time.time() > deadline or len(found) >= cap:
                state["complete"] = False
                return

    expand(0, full, 0)
    out = []
    for mask in found:
        vs = []
        m = mask
        while m:
            v = (m & -m).bit_length() - 1
            vs.append(v)
            m &= ~(1 << v)
        out.append(tuple(vs))
    return out, state["complete"]


if __name__ == "__main__":
    import json
    import os
    HERE = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, HERE)
    sys.path.insert(0, os.path.dirname(HERE))
    import numpy as np
    from CliqueAI.graph.codec import GraphCodec
    from _common import DATA_DIR
    import fleetsolver
    os.environ.setdefault("SN83_THREADS", "13")
    rs = [json.loads(l) for l in open(os.path.join(DATA_DIR, "sim_rounds.jsonl"))
          if l.strip()]
    rs = [r for r in rs if r.get("answers") and r.get("timestamp")]
    rs.sort(key=lambda r: r["timestamp"])
    budget = float(os.environ.get("EO_BUDGET", "8"))
    for r in rs[-216:][:int(os.environ.get("EO_N", "4"))]:
        A = np.array(GraphCodec().decode_matrix(r["matrix_b92"]), dtype=np.uint8)
        n = A.shape[0]
        adj = [int("".join("1" if A[i][j] else "0" for j in range(n - 1, -1, -1)), 2)
               for i in range(n)]
        t0 = time.time()
        cl = fleetsolver.solve_many(np.ascontiguousarray(A), 2.0, 40, threads=13)
        omega = max(len(c) for c in cl)
        lscc = {tuple(sorted(c)) for c in cl if len(c) == omega}
        t1 = time.time()
        got, complete = enumerate_omega(adj, n, omega, time.time() + budget)
        allc = {tuple(sorted(c)) for c in got}
        print(f"n={n:>4} tl={r['time_limit']:>4} omega={omega:>3} | "
              f"LSCC {len(lscc):>3} in {t1-t0:.1f}s | "
              f"ENUM {len(allc):>5} in {time.time()-t1:.1f}s "
              f"{'(complete)' if complete else '(TRUNCATED)'} | "
              f"new {len(allc - lscc):>5}")
