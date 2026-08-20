#!/usr/bin/env python3
"""Split the collected SN83 instances into train and a held-out validation set.

The validation set is sized by *solver budget*, not by row count: instances are
drawn until their time limits sum to roughly --budget seconds (default 600, so
one full validation pass costs about ten minutes of wall clock).

Sampling is stratified by (time_limit, |V| bucket) with proportional allocation
and largest-remainder rounding, so the validation mix reproduces the source
distribution rather than merely matching it in expectation.

Three files come out of it:

  train.jsonl         full records, labels included — the agent may use all of it
  val_problems.jsonl  graphs + time limits only, NO labels — the agent may solve
                      these but cannot see the answers
  val_labels.jsonl    the withheld labels, read only by score_submission.py

  python make_splits.py --budget 600
"""
import argparse
import collections
import glob
import json
import os
import random
import sys

from _common import DATA_DIR, save_json

SPLITS = os.path.join(DATA_DIR, "splits")

PROBLEM_FIELDS = ["uuid", "n", "edges", "density", "time_limit", "difficulty",
                  "matrix_b92"]
LABEL_FIELDS = ["uuid", "best_size", "best_cliques", "best_clique_counts",
                "size_hist", "any_unique", "n_responders", "n_valid", "n_at_best",
                "difficulty", "time_limit"]


def load_pool(patterns):
    """Every instance we have, deduplicated by uuid."""
    pool, dupes = {}, 0
    for pat in patterns:
        for path in sorted(glob.glob(pat)):
            if os.path.realpath(path).startswith(os.path.realpath(SPLITS)):
                continue                       # never re-ingest our own output
            with open(path) as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not r.get("uuid") or not r.get("matrix_b92"):
                        continue
                    if r["uuid"] in pool:
                        dupes += 1
                        continue
                    pool[r["uuid"]] = r
    return list(pool.values()), dupes


def stratum(rec):
    return (rec["time_limit"], rec["n"] // 100 * 100)


def allocate(pool, n_val):
    """Largest-remainder proportional allocation across strata."""
    counts = collections.Counter(stratum(r) for r in pool)
    total = len(pool)
    exact = {k: n_val * v / total for k, v in counts.items()}
    alloc = {k: int(v) for k, v in exact.items()}
    short = n_val - sum(alloc.values())
    for k, _ in sorted(exact.items(), key=lambda kv: kv[1] - int(kv[1]), reverse=True):
        if short <= 0:
            break
        if alloc[k] < counts[k]:
            alloc[k] += 1
            short -= 1
    return alloc, counts


def dist(records, key):
    total = len(records)
    c = collections.Counter(key(r) for r in records)
    return {str(k): round(100 * v / total, 2) for k, v in sorted(c.items(), key=lambda kv: str(kv[0]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=600.0,
                    help="target sum of time limits in the validation set, seconds")
    ap.add_argument("--seed", type=int, default=8383)
    ap.add_argument("--glob", nargs="*",
                    default=[os.path.join(DATA_DIR, "v*", "*.jsonl"),
                             os.path.join(DATA_DIR, "*.jsonl")])
    args = ap.parse_args()

    pool, dupes = load_pool(args.glob)
    if not pool:
        print(f"no instances found under {DATA_DIR}", file=sys.stderr)
        return 1
    mean_tl = sum(r["time_limit"] for r in pool) / len(pool)
    n_val = max(1, min(len(pool), round(args.budget / mean_tl)))
    print(f"pool: {len(pool):,} instances ({dupes} duplicate uuids dropped), "
          f"mean time limit {mean_tl:.2f}s")
    print(f"validation target: {args.budget:.0f}s -> {n_val} instances")

    rng = random.Random(args.seed)
    alloc, counts = allocate(pool, n_val)
    by_stratum = collections.defaultdict(list)
    for r in pool:
        by_stratum[stratum(r)].append(r)

    val_uuids = set()
    for k, take in alloc.items():
        group = by_stratum[k]
        rng.shuffle(group)
        for r in group[:take]:
            val_uuids.add(r["uuid"])

    val = [r for r in pool if r["uuid"] in val_uuids]
    train = [r for r in pool if r["uuid"] not in val_uuids]
    val_seconds = sum(r["time_limit"] for r in val)

    os.makedirs(SPLITS, exist_ok=True)
    with open(os.path.join(SPLITS, "train.jsonl"), "w") as f:
        for r in sorted(train, key=lambda r: r["uuid"]):
            f.write(json.dumps(r, separators=(",", ":")) + "\n")
    with open(os.path.join(SPLITS, "val_problems.jsonl"), "w") as f:
        for r in sorted(val, key=lambda r: r["uuid"]):
            f.write(json.dumps({k: r[k] for k in PROBLEM_FIELDS if k in r},
                               separators=(",", ":")) + "\n")
    with open(os.path.join(SPLITS, "val_labels.jsonl"), "w") as f:
        for r in sorted(val, key=lambda r: r["uuid"]):
            f.write(json.dumps({k: r[k] for k in LABEL_FIELDS if k in r},
                               separators=(",", ":")) + "\n")

    manifest = {
        "seed": args.seed,
        "budget_s": args.budget,
        "pool": len(pool),
        "train": len(train),
        "val": len(val),
        "val_total_time_limit_s": round(val_seconds, 1),
        "val_wall_clock_estimate_min": round(val_seconds / 60, 2),
        "distribution_time_limit_pct": {
            "pool": dist(pool, lambda r: r["time_limit"]),
            "val": dist(val, lambda r: r["time_limit"]),
        },
        "distribution_vertex_bucket_pct": {
            "pool": dist(pool, lambda r: r["n"] // 100 * 100),
            "val": dist(val, lambda r: r["n"] // 100 * 100),
        },
        "distribution_difficulty_pct": {
            "pool": dist(pool, lambda r: r["difficulty"]),
            "val": dist(val, lambda r: r["difficulty"]),
        },
    }
    save_json(os.path.join(SPLITS, "manifest.json"), manifest)

    print(f"\ntrain {len(train):,}   val {len(val)}   "
          f"val budget {val_seconds:.0f}s ({val_seconds/60:.1f} min)")
    print("\ndistribution check (pool % vs val %):")
    for name, block in (("time limit", manifest["distribution_time_limit_pct"]),
                        ("|V| bucket", manifest["distribution_vertex_bucket_pct"]),
                        ("difficulty", manifest["distribution_difficulty_pct"])):
        print(f"  {name}:")
        for k in sorted(block["pool"], key=lambda x: float(x)):
            p, v = block["pool"][k], block["val"].get(k, 0.0)
            flag = "" if abs(p - v) <= 3.0 else "   <-- drift"
            print(f"    {k:>6}  pool {p:5.1f}%   val {v:5.1f}%{flag}")
    print(f"\nwrote {SPLITS}/{{train,val_problems,val_labels}}.jsonl + manifest.json")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.exit(main())
