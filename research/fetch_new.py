#!/usr/bin/env python3
"""Incrementally pull new SN83 rounds from the validator W&B stream.

Designed to run from cron every few minutes. Each invocation:
  - discovers every run matching --versions (default 0.0.17)
  - pulls only steps newer than the last one recorded in state.json
  - appends instances to data/<version>/YYYY-MM-DD.jsonl
  - updates status.json so `status.py` can tell you the pipeline is alive

Safe to run concurrently with itself: a lockfile makes overlapping invocations
a no-op rather than a double-write.

  python fetch_new.py                     # incremental, 0.0.17 only
  python fetch_new.py --versions 0.0.16 0.0.17
  python fetch_new.py --backfill 5000     # first run: reach this far back per run
"""
import argparse
import datetime
import errno
import json
import os
import socket
import sys
import time
import traceback

from _common import (DATA_DIR, DEFAULT_VERSIONS, KEYS, PROJECT, check_wandb_version,
                     discover_runs, load_json, row_to_record, save_json)

STATE = os.path.join(DATA_DIR, "state.json")
STATUS = os.path.join(DATA_DIR, "status.json")
LOCK = os.path.join(DATA_DIR, "fetch.lock")


def acquire_lock(stale_after=3600):
    os.makedirs(DATA_DIR, exist_ok=True)
    try:
        fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        age = time.time() - os.path.getmtime(LOCK)
        if age < stale_after:
            return None
        print(f"removing stale lock ({age:.0f}s old)", file=sys.stderr)
        os.unlink(LOCK)
        fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, f"{os.getpid()}@{socket.gethostname()}\n".encode())
    os.close(fd)
    return LOCK


def shard_path(version):
    day = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
    return os.path.join(DATA_DIR, f"v{version}", f"{day}.jsonl")


def fetch_run(api, run, state, backfill, keep_answers, counters):
    last_seen = state.get(run.id, {}).get("last_step", -1)
    head = int(run.summary.get("_step", -1))
    if head < 0:
        return
    lo = last_seen + 1 if last_seen >= 0 else max(0, head - backfill)
    if lo > head:
        counters["up_to_date"] += 1
        return

    version = run.config.get("version", "unknown")
    path = shard_path(version)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    written, dupes, high = 0, 0, last_seen
    seen_uuids = set()
    with open(path, "a") as out:
        for row in run.scan_history(keys=KEYS, min_step=lo, max_step=head + 1,
                                    page_size=20):
            step = row.get("_step")
            uid = row.get("uuid")
            # scan_history repeats ~1 row per page at page boundaries
            if step is None or uid is None or not row.get("encoded_matrix"):
                continue
            if step <= last_seen or uid in seen_uuids:
                dupes += 1
                continue
            seen_uuids.add(uid)
            try:
                rec = row_to_record(row, keep_answers)
            except Exception:
                counters["errors"] += 1
                continue
            rec["_run"] = run.id
            rec["_step"] = step
            out.write(json.dumps(rec, separators=(",", ":")) + "\n")
            written += 1
            high = max(high, step)

    state[run.id] = {"last_step": high, "version": version,
                     "updated": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    counters["rows"] += written
    counters["dupes"] += dupes
    print(f"  {run.id[:24]}… v{version}: steps {lo}..{head} -> {written} new "
          f"({dupes} dupes skipped)", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--versions", nargs="*", default=DEFAULT_VERSIONS)
    ap.add_argument("--backfill", type=int, default=500,
                    help="on a run's first sighting, reach this many steps back")
    ap.add_argument("--keep-answers", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    if acquire_lock() is None:
        print("another fetch is running; nothing to do", file=sys.stderr)
        return 0

    counters = {"rows": 0, "dupes": 0, "errors": 0, "up_to_date": 0, "runs": 0}
    status = load_json(STATUS, {})
    try:
        import wandb
        check_wandb_version()
        api = wandb.Api(timeout=180)
        state = load_json(STATE, {})
        runs = discover_runs(api, args.versions)
        counters["runs"] = len(runs)
        print(f"{len(runs)} runs match {args.versions}", file=sys.stderr)
        for run in runs:
            fetch_run(api, run, state, args.backfill, args.keep_answers, counters)
        save_json(STATE, state)
        status.update({"ok": True, "error": None})
    except Exception as exc:
        status.update({"ok": False,
                       "error": f"{type(exc).__name__}: {exc}",
                       "traceback": traceback.format_exc()[-2000:]})
        print(traceback.format_exc(), file=sys.stderr)
    finally:
        try:
            os.unlink(LOCK)
        except FileNotFoundError:
            pass

    now = datetime.datetime.now(datetime.timezone.utc)
    status["last_run"] = now.isoformat()
    status["duration_s"] = round(time.time() - t0, 2)
    status["versions"] = args.versions
    status.update({f"last_{k}": v for k, v in counters.items()})
    status["total_rows"] = status.get("total_rows", 0) + counters["rows"]
    if counters["rows"]:
        status["last_row_at"] = now.isoformat()
    save_json(STATUS, status)

    print(f"{counters['rows']} new instances in {status['duration_s']}s "
          f"({counters['errors']} errors)", file=sys.stderr)
    return 0 if status.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
