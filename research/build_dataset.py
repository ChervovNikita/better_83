#!/usr/bin/env python3
"""Bulk backfill of the SN83 W&B stream into a JSONL dataset.

Use this once to build a benchmark or training corpus; use fetch_new.py to keep
it current. Available history at time of writing: 1,689,167 rounds across 68
runs back to 2025-08-26, of which 233,548 are v0.0.16+ (today's problem mix).
Measured throughput is ~15 rows/s and ~43 KB/row with the graph included.

  python build_dataset.py --versions 0.0.17 --limit 5000 --out bench.jsonl
  python build_dataset.py --versions 0.0.16 0.0.17 --limit 0 --workers 6
"""
import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from _common import KEYS, PROJECT, discover_runs, row_to_record

_lock = threading.Lock()


def pull_run(api, run_id, limit, keep_answers, out, seen, stats):
    run = api.run(f"{PROJECT}/{run_id}")
    head = int(run.summary.get("_step", -1))
    if head < 0:
        return
    lo = max(0, head - limit) if limit else 0
    got = 0
    for row in run.scan_history(keys=KEYS, min_step=lo, max_step=head + 1, page_size=20):
        uid = row.get("uuid")
        if not uid or not row.get("encoded_matrix"):
            continue
        with _lock:
            if uid in seen:            # page-boundary repeats, ~1 per page
                stats["dupes"] += 1
                continue
            seen.add(uid)
        try:
            rec = row_to_record(row, keep_answers)
        except Exception:
            with _lock:
                stats["errors"] += 1
            continue
        rec["_run"] = run_id
        rec["_step"] = row.get("_step")
        line = json.dumps(rec, separators=(",", ":"))
        with _lock:
            out.write(line + "\n")
            stats["rows"] += 1
            stats["bytes"] += len(line) + 1
            if stats["rows"] % 500 == 0:
                el = time.time() - stats["t0"]
                print(f"  {stats['rows']:>8,} rows  {stats['bytes']/1e6:7.1f} MB  "
                      f"{stats['rows']/el:5.1f} rows/s", file=sys.stderr, flush=True)
        got += 1
        if limit and got >= limit:
            return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--versions", nargs="*", default=["0.0.17"])
    ap.add_argument("--limit", type=int, default=2000, help="rows per run, 0 = all")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--keep-answers", action="store_true")
    ap.add_argument("--out", default="sn83.jsonl")
    args = ap.parse_args()

    import wandb
    api = wandb.Api(timeout=180)
    runs = discover_runs(api, args.versions)
    runs.reverse()                      # newest first
    avail = sum(max(int(r.summary.get("_step", -1)) + 1, 0) for r in runs)
    print(f"{len(runs)} runs match {args.versions}: {avail:,} rounds logged",
          file=sys.stderr)

    seen = set()
    stats = {"rows": 0, "bytes": 0, "dupes": 0, "errors": 0, "t0": time.time()}
    with open(args.out, "w") as out, ThreadPoolExecutor(max_workers=args.workers) as pool:
        for f in [pool.submit(pull_run, api, r.id, args.limit, args.keep_answers,
                              out, seen, stats) for r in runs]:
            f.result()

    el = time.time() - stats["t0"]
    print(f"\nwrote {stats['rows']:,} instances to {args.out} "
          f"({stats['bytes']/1e6:.1f} MB) in {el/60:.1f} min "
          f"[{stats['dupes']} dupes, {stats['errors']} errors]", file=sys.stderr)


if __name__ == "__main__":
    main()
