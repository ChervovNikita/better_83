#!/usr/bin/env python3
"""Health report for the SN83 data pipeline. Exit code 0 = healthy, 1 = stale/broken.

  python status.py            # human readable
  python status.py --json     # machine readable
"""
import argparse
import collections
import datetime
import glob
import json
import os
import sys

from _common import DATA_DIR, load_json

STALE_AFTER_MIN = 20      # cron runs every 5 min; 4 misses is a real problem


def scan_shards():
    rows, bytes_, by_tl, by_n, days = 0, 0, collections.Counter(), collections.Counter(), set()
    newest = None
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "v*", "*.jsonl"))):
        bytes_ += os.path.getsize(path)
        days.add(os.path.basename(path)[:-6])
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rows += 1
                by_tl[r.get("time_limit")] += 1
                by_n[(r.get("n", 0) // 100) * 100] += 1
                newest = r
    return rows, bytes_, by_tl, by_n, sorted(days), newest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    status = load_json(os.path.join(DATA_DIR, "status.json"), {})
    state = load_json(os.path.join(DATA_DIR, "state.json"), {})
    rows, bytes_, by_tl, by_n, days, newest = scan_shards()

    age_min = None
    if status.get("last_run"):
        last = datetime.datetime.fromisoformat(status["last_run"])
        age_min = (datetime.datetime.now(datetime.timezone.utc) - last).total_seconds() / 60

    healthy = bool(status.get("ok")) and age_min is not None and age_min < STALE_AFTER_MIN

    if args.json:
        print(json.dumps({"healthy": healthy, "age_min": age_min, "instances": rows,
                          "bytes": bytes_, "runs_tracked": len(state),
                          "status": status}, indent=2, default=str))
        return 0 if healthy else 1

    mark = "OK " if healthy else "BAD"
    print(f"[{mark}] SN83 data pipeline   {DATA_DIR}")
    print(f"  last fetch      {status.get('last_run', 'never')}"
          + (f"  ({age_min:.1f} min ago)" if age_min is not None else ""))
    print(f"  last result     {'ok' if status.get('ok') else status.get('error', 'unknown')}")
    print(f"  duration        {status.get('duration_s', '?')}s")
    print(f"  new last run    {status.get('last_rows', 0)} instances")
    print(f"  runs tracked    {len(state)} ({', '.join(sorted({v.get('version','?') for v in state.values()})) or '-'})")
    print(f"  total on disk   {rows:,} instances, {bytes_/1e6:.1f} MB, {len(days)} day shards")
    if days:
        print(f"  shard range     {days[0]} .. {days[-1]}")
    if by_tl:
        print("  by time limit   " + "  ".join(f"{k}s:{v}" for k, v in sorted(by_tl.items(), key=lambda x: x[0] or 0)))
    if by_n:
        print("  by |V| bucket   " + "  ".join(f"{k}:{v}" for k, v in sorted(by_n.items())))
    if newest:
        print(f"  newest instance n={newest['n']} density={newest['density']} "
              f"limit={newest['time_limit']}s best={newest['best_size']} "
              f"distinct_optima={len(newest.get('best_cliques', []))}")
    if not healthy:
        print(f"\n  UNHEALTHY: no successful fetch in the last {STALE_AFTER_MIN} min, "
              f"or the last fetch errored. Check research/data/fetch.log")
    return 0 if healthy else 1


if __name__ == "__main__":
    sys.exit(main())
