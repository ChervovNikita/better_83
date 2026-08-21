#!/usr/bin/env python3
"""Fast, parallel scoring against the TRAIN labels — the inner tuning loop.

`score_submission.py` is the honest signal but it is sequential: one 42-task val
pass costs ten wall-clock minutes, and the brief warns that a two-task move on
42 tasks means nothing. Train is unrestricted and has the same labels, so tune
here and spend val only on decisions.

Two things make this fast without making it dishonest:

  * tasks run in parallel across worker processes, but each worker is pinned to
    its own block of `--threads` cores with sched_setaffinity, so a solve sees
    exactly the CPU a single miner would and its wall clock stays meaningful;
  * a byte-offset index over train.jsonl (built once, cached) lets a stratified
    sample be drawn without reading the 1 GB corpus.

    python3 score_train.py --solver fastsolver:solve --n 200
    python3 score_train.py --solver fastsolver:solve --n 400 --time-limit 6
"""
import argparse
import collections
import importlib
import json
import multiprocessing as mp
import os
import random
import sys
import time

import numpy as np

from _common import DATA_DIR

SPLITS = os.path.join(DATA_DIR, "splits")
TRAIN = os.path.join(SPLITS, "train.jsonl")
INDEX = os.path.join(SPLITS, "train_index.jsonl")

_W = {}                                    # per-worker state, set by _init


def build_index(train=TRAIN, index=INDEX):
    """One pass over the corpus recording where each record starts."""
    print(f"indexing {train} (one-off, ~30s)...", file=sys.stderr)
    n = 0
    with open(train, "rb") as f, open(index + ".tmp", "w") as out:
        while True:
            off = f.tell()
            line = f.readline()
            if not line:
                break
            if not line.strip():
                continue
            r = json.loads(line)
            out.write(json.dumps({"o": off, "l": len(line), "uuid": r["uuid"],
                                  "n": r["n"], "tl": r["time_limit"],
                                  "d": r["difficulty"], "best": r["best_size"],
                                  "n_at_best": r.get("n_at_best", 0),
                                  "n_valid": r.get("n_valid", 0)}) + "\n")
            n += 1
    os.replace(index + ".tmp", index)
    print(f"indexed {n} records -> {index}", file=sys.stderr)


def load_index():
    if not os.path.exists(INDEX) or os.path.getmtime(INDEX) < os.path.getmtime(TRAIN):
        build_index()
    with open(INDEX) as f:
        return [json.loads(line) for line in f if line.strip()]


def sample(idx, n_take, seed, time_limit=None, max_n=None):
    """Stratified by (time_limit, difficulty), the same strata make_splits uses."""
    if time_limit is not None:
        idx = [r for r in idx if r["tl"] == time_limit]
    if max_n is not None:
        idx = [r for r in idx if r["n"] <= max_n]
    if n_take >= len(idx):
        return list(idx)
    rng = random.Random(seed)
    by = collections.defaultdict(list)
    for r in idx:
        by[(r["tl"], r["d"])].append(r)
    keys = sorted(by)
    quotas = {k: len(by[k]) * n_take / len(idx) for k in keys}
    take = {k: int(q) for k, q in quotas.items()}
    for k in sorted(keys, key=lambda k: -(quotas[k] - take[k]))[:n_take - sum(take.values())]:
        take[k] += 1
    out = []
    for k in keys:
        out += rng.sample(by[k], min(take[k], len(by[k])))
    rng.shuffle(out)
    return out


def read_record(off, length):
    with open(TRAIN, "rb") as f:
        f.seek(off)
        return json.loads(f.read(length))


def _init(solver, threads, cores):
    """Pin this worker to its own cores so a solve gets a miner-sized machine."""
    slot = mp.current_process()._identity[0] - 1 if mp.current_process()._identity else 0
    if cores:
        block = cores[(slot * threads) % len(cores):][:threads]
        if len(block) == threads:
            os.sched_setaffinity(0, block)
    os.environ["SN83_THREADS"] = str(threads)
    mod, fn = solver.split(":")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    _W["solve"] = getattr(importlib.import_module(mod), fn)
    _W["threads"] = threads


def _run(job):
    entry, time_scale = job
    from CliqueAI.graph.codec import GraphCodec
    rec = read_record(entry["o"], entry["l"])
    A = np.array(GraphCodec().decode_matrix(rec["matrix_b92"]), dtype=np.uint8)
    t0 = time.time()
    out = _W["solve"](A, rec["time_limit"] * time_scale)
    elapsed = time.time() - t0
    clique = list(out[2]) if isinstance(out, tuple) else list(out)

    ok, why = True, "ok"
    S = list(clique)
    if not S or len(set(S)) != len(S):
        ok, why = False, "empty or repeated vertex"
    else:
        i = np.array(S, dtype=int)
        if i.min() < 0 or i.max() >= A.shape[0]:
            ok, why = False, "vertex out of range"
        elif A[np.ix_(i, i)].sum() != len(S) * (len(S) - 1):
            ok, why = False, "not a clique"
        else:
            cnt = A[i].sum(axis=0)
            inC = np.zeros(A.shape[0], dtype=bool)
            inC[i] = True
            if np.any((cnt == len(S)) & (~inC)):
                ok, why = False, "not maximal (a vertex can still be added)"
    size = len(clique) if ok else 0
    return dict(uuid=rec["uuid"], n=rec["n"], tl=rec["time_limit"],
                density=rec["density"], best=rec["best_size"], ours=size,
                delta=(size - rec["best_size"]) if ok else None, ok=ok, why=why,
                elapsed=elapsed, over=elapsed > rec["time_limit"] * 1.02 + 0.25,
                n_at_best=rec.get("n_at_best", 0), n_valid=rec.get("n_valid", 0))


def report(rows, label):
    n = len(rows)
    solved = [r for r in rows if r["ok"]]
    failed = [r for r in rows if not r["ok"]]
    deltas = [r["delta"] for r in solved]
    hist = collections.Counter(deltas)
    print("\n" + "=" * 62)
    print(f"TRAIN SCORE vs BEST RIVAL   ({label})")
    print("=" * 62)
    print(f"  tasks {n}\n")
    if hist:
        for d in range(max(max(hist), 1), min(hist) - 1, -1):
            c = hist.get(d, 0)
            tag = "  <-- same as best rival" if d == 0 else ""
            print(f"  {('0' if d == 0 else f'{d:+d}'):>4}  {c:>5}  {c/n:6.1%}  "
                  f"{'#' * round(46 * c / n)}{tag}")
    if failed:
        print(f"  {'inv':>4}  {len(failed):>5}  {len(failed)/n:6.1%}  "
              f"{'#' * round(46 * len(failed) / n)}  <-- invalid, scores zero")
        for why, c in collections.Counter(r["why"] for r in failed).most_common(4):
            print(f"                        {c:>4}  {why}")
    over = sum(1 for r in rows if r["over"])
    print(f"\n  parity rate  {sum(1 for d in deltas if d >= 0)/n:.1%}"
          f"   mean delta {np.mean(deltas):+.2f}" if deltas else "")
    if over:
        print(f"  {over} OVER BUDGET")

    print("\n" + "-" * 62)
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
    print(f"  {'|V|':>6} {'tasks':>6} {'match+':>7} {'mean d':>8}")
    by_n = collections.defaultdict(list)
    for r in rows:
        by_n[r["n"] // 100 * 100].append(r)
    for nb in sorted(by_n):
        g = by_n[nb]
        gd = [r["delta"] for r in g if r["ok"]]
        print(f"  {nb:6d} {len(g):6d} {sum(1 for d in gd if d >= 0)/len(g):6.0%} "
              f"{(np.mean(gd) if gd else float('nan')):+8.2f}")
    print("=" * 62)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solver", default="fastsolver:solve")
    ap.add_argument("--n", type=int, default=200, help="tasks to sample")
    ap.add_argument("--seed", type=int, default=1, help="sample seed; keep fixed to compare runs")
    ap.add_argument("--time-limit", type=float, help="only this deadline")
    ap.add_argument("--max-n", type=int, help="only graphs up to this |V|")
    ap.add_argument("--time-scale", type=float, default=0.88)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SN83_THREADS", "8")),
                    help="solver threads per task; also the size of each worker's core block")
    ap.add_argument("--workers", type=int, default=0, help="0 = fill the machine")
    ap.add_argument("--json", help="write per-task rows here")
    ap.add_argument("--index", action="store_true", help="(re)build the index and exit")
    args = ap.parse_args()

    if args.index:
        build_index()
        return 0

    mod, fn = args.solver.split(":")
    getattr(importlib.import_module(mod), fn)   # build/import once, not in 15 workers at once

    idx = load_index()
    tasks = sample(idx, args.n, args.seed, args.time_limit, args.max_n)
    cores = sorted(os.sched_getaffinity(0))
    workers = args.workers or max(1, min(len(tasks), (len(cores) - 4) // args.threads))
    budget = sum(t["tl"] for t in tasks)
    print(f"{len(tasks)} train tasks, {budget:.0f}s of deadline across {workers} workers "
          f"x {args.threads} threads (~{budget/60/workers:.1f} min)", file=sys.stderr)

    t0 = time.time()
    with mp.Pool(workers, initializer=_init,
                 initargs=(args.solver, args.threads, cores)) as pool:
        rows = []
        for i, r in enumerate(pool.imap_unordered(
                _run, [(t, args.time_scale) for t in tasks]), 1):
            rows.append(r)
            if i % 25 == 0:
                print(f"  {i}/{len(tasks)}  ({time.time()-t0:.0f}s)", file=sys.stderr, flush=True)

    report(rows, f"n={len(rows)} seed={args.seed} threads={args.threads}")
    print(f"wall clock {time.time()-t0:.0f}s", file=sys.stderr)
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"tasks": len(rows), "seed": args.seed, "threads": args.threads,
                       "rows": rows}, f, indent=2)
        print(f"rows written to {args.json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.exit(main())
