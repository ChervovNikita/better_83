#!/usr/bin/env python3
"""Run a candidate solver over the validation split and score it automatically.

The agent never touches the labels. This script loads them, runs the solver on
`val_problems.jsonl` under each instance's own time limit, and prints one
report: how we did against the best rival answer on every task, and the full
distribution of size deltas.

Two ways to submit:

  # 1. Python solver — we import it and time it ourselves (preferred)
  python score_submission.py --solver mysolver:solve

  # 2. Precomputed answers — for compiled or GPU solvers
  python score_submission.py --submission answers.jsonl

Solver format (mode 1):

    def solve(adjacency: numpy.ndarray, time_limit: float) -> list[int]

  `adjacency` is an n x n uint8 symmetric matrix with a zero diagonal.
  `time_limit` is the seconds you may spend. Return the vertex indices of a
  MAXIMAL clique — one that can still be extended scores zero, as does an empty
  answer, a repeated vertex, or an out-of-range vertex.

Submission format (mode 2), one JSON object per line:

    {"uuid": "...", "clique": [3, 21, 26, ...], "elapsed": 6.9}

  `elapsed` is your own measured solve time in seconds; it is checked against
  the instance's limit and over-budget answers are reported (and, with
  --strict, scored as failures).
"""
import argparse
import collections
import importlib
import json
import os
import sys
import time

import numpy as np

from _common import DATA_DIR

SPLITS = os.path.join(DATA_DIR, "splits")


def decode(b92):
    from CliqueAI.graph.codec import GraphCodec
    return np.array(GraphCodec().decode_matrix(b92), dtype=np.uint8)


def check(A, clique):
    """CliqueScoreCalculator.is_valid_maximum_clique, verbatim in intent."""
    S = list(clique)
    if not S or len(set(S)) != len(S):
        return False, "empty or repeated vertex"
    n = A.shape[0]
    if any(not isinstance(v, (int, np.integer)) or v < 0 or v >= n for v in S):
        return False, "vertex out of range"
    idx = np.array(S, dtype=int)
    if A[np.ix_(idx, idx)].sum() != len(S) * (len(S) - 1):
        return False, "not a clique"
    cnt = A[idx].sum(axis=0)
    inC = np.zeros(n, dtype=bool)
    inC[idx] = True
    if np.any((cnt == len(S)) & (~inC)):
        return False, "not maximal (a vertex can still be added)"
    return True, "ok"


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solver", help="module:function, imported and timed here")
    ap.add_argument("--submission", help="precomputed answers as JSONL")
    ap.add_argument("--split", default="val", choices=["val", "bigger_val"],
                    help="val is the ~10 min steering set; bigger_val is the "
                         "multi-hour audit set — run it rarely")
    ap.add_argument("--problems", help="override the split's problem file")
    ap.add_argument("--labels", help="override the split's label file")
    ap.add_argument("--time-scale", type=float, default=0.88,
                    help="fraction of each deadline the solver may use; the rest is "
                         "network headroom a live miner needs")
    ap.add_argument("--limit", type=int, default=0, help="score only the first N tasks")
    ap.add_argument("--strict", action="store_true",
                    help="score over-budget answers as failures")
    ap.add_argument("--json", help="also write the report here")
    args = ap.parse_args()

    if bool(args.solver) == bool(args.submission):
        print("give exactly one of --solver or --submission", file=sys.stderr)
        return 2

    problems_path = args.problems or os.path.join(SPLITS, f"{args.split}_problems.jsonl")
    labels_path = args.labels or os.path.join(SPLITS, f"{args.split}_labels.jsonl")
    problems = load_jsonl(problems_path)
    labels = {r["uuid"]: r for r in load_jsonl(labels_path)}
    if args.limit:
        problems = problems[:args.limit]
    missing = [p["uuid"] for p in problems if p["uuid"] not in labels]
    if missing:
        print(f"{len(missing)} problems have no label; cannot score", file=sys.stderr)
        return 2

    answers = {}
    if args.submission:
        for r in load_jsonl(args.submission):
            answers[r["uuid"]] = (r.get("clique") or [], float(r.get("elapsed", 0.0)))
    else:
        mod, fn = args.solver.split(":")
        solve = getattr(importlib.import_module(mod), fn)

    budget = sum(p["time_limit"] for p in problems)
    print(f"scoring {len(problems)} tasks from '{args.split}' "
          f"({budget:.0f}s of deadline, ~{budget/60:.1f} min)\n", file=sys.stderr)

    rows, t_start = [], time.time()
    for i, p in enumerate(problems, 1):
        lab = labels[p["uuid"]]
        A = decode(p["matrix_b92"])
        if args.submission:
            clique, elapsed = answers.get(p["uuid"], ([], 0.0))
        else:
            t0 = time.time()
            out = solve(A, p["time_limit"] * args.time_scale)
            elapsed = time.time() - t0
            clique = list(out[2]) if isinstance(out, tuple) else list(out)
        ok, why = check(A, clique)
        over = elapsed > p["time_limit"] * 1.02 + 0.25
        if over and args.strict:
            ok, why = False, f"over budget ({elapsed:.1f}s > {p['time_limit']}s)"
        size = len(clique) if ok else 0
        rows.append(dict(uuid=p["uuid"], n=p["n"], tl=p["time_limit"],
                         density=p["density"], best=lab["best_size"], ours=size,
                         delta=(size - lab["best_size"]) if ok else None,
                         ok=ok, why=why, elapsed=elapsed, over=over))
        if i % 10 == 0:
            print(f"  {i}/{len(problems)}  ({time.time()-t_start:.0f}s elapsed)",
                  file=sys.stderr, flush=True)

    solved = [r for r in rows if r["ok"]]
    failed = [r for r in rows if not r["ok"]]
    deltas = [r["delta"] for r in solved]
    n = len(rows)
    matched = sum(1 for d in deltas if d == 0)
    beat = sum(1 for d in deltas if d > 0)

    print("\n" + "=" * 62)
    print("SCORE vs BEST RIVAL   (our clique size - best rival's)")
    print("=" * 62)
    print(f"  tasks {n}\n")
    hist = collections.Counter(deltas)
    if hist:
        hi = max(max(hist), 1)          # always show at least +1, so "never ahead" is visible
        lo = min(hist)
        for d in range(hi, lo - 1, -1):
            cnt = hist.get(d, 0)
            bar = "#" * round(46 * cnt / n)
            tag = "  <-- same as best rival" if d == 0 else ""
            label = "0" if d == 0 else f"{d:+d}"
            print(f"  {label:>4}  {cnt:>5}  {cnt/n:6.1%}  {bar}{tag}")
    if failed:
        bar = "#" * round(46 * len(failed) / n)
        print(f"  {'inv':>4}  {len(failed):>5}  {len(failed)/n:6.1%}  {bar}  <-- invalid, scores zero")
        for why, c in collections.Counter(r["why"] for r in failed).most_common(4):
            print(f"                        {c:>4}  {why}")
    print(f"\n  total solve time  {sum(r['elapsed'] for r in rows):.0f}s of {budget:.0f}s allowed"
          + (f"   ({sum(1 for r in rows if r['over'])} OVER BUDGET)"
             if any(r["over"] for r in rows) else ""))

    print("\n" + "-" * 62)
    print("BY DEADLINE   (where the time constraint bites)")
    print("-" * 62)
    print(f"  {'limit':>6} {'tasks':>6} {'match+':>7} {'-1':>4} {'<=-2':>5} {'inv':>4} {'mean d':>8}")
    by = collections.defaultdict(list)
    for r in rows:
        by[r["tl"]].append(r)
    for tl in sorted(by):
        g = by[tl]
        gd = [r["delta"] for r in g if r["ok"]]
        print(f"  {tl:6.1f} {len(g):6d} {sum(1 for d in gd if d >= 0)/len(g):6.0%} "
              f"{sum(1 for d in gd if d == -1):4d} {sum(1 for d in gd if d <= -2):5d} "
              f"{sum(1 for r in g if not r['ok']):4d} "
              f"{(np.mean(gd) if gd else float('nan')):+8.2f}")

    print("\n" + "-" * 62)
    print("BY GRAPH SIZE")
    print("-" * 62)
    print(f"  {'|V|':>6} {'tasks':>6} {'match+':>7} {'mean d':>8}")
    by_n = collections.defaultdict(list)
    for r in rows:
        by_n[r["n"] // 100 * 100].append(r)
    for nb in sorted(by_n):
        g = by_n[nb]
        gd = [r["delta"] for r in g if r["ok"]]
        print(f"  {nb:6d} {len(g):6d} {sum(1 for d in gd if d >= 0)/len(g):6.0%} "
              f"{(np.mean(gd) if gd else float('nan')):+8.2f}")

    rate = (matched + beat) / n
    print("\n" + "=" * 62)
    if failed:
        print("VERDICT: fix validity first — invalid answers score zero on chain,")
        print("         which is worse than any size deficit.")
    elif rate >= 0.99:
        print("VERDICT: at parity with the field. Collisions are the next lever.")
    elif rate >= 0.90:
        print(f"VERDICT: {rate:.0%} parity. Close. Look at the deadline breakdown above.")
    elif rate >= 0.60:
        print(f"VERDICT: {rate:.0%} parity. Real progress, not yet competitive.")
    else:
        print(f"VERDICT: {rate:.0%} parity. Not competitive — a miner this far back")
        print("         earns nothing, because weights amplify rank, not reward.")
    print("=" * 62)

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"tasks": n, "parity_rate": rate,
                       "matched": matched, "beat": beat, "invalid": len(failed),
                       "mean_delta": float(np.mean(deltas)) if deltas else None,
                       "delta_histogram": {str(k): v for k, v in sorted(hist.items())},
                       "rows": rows}, f, indent=2)
        print(f"\nreport written to {args.json}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.exit(main())
