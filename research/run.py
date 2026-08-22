#!/usr/bin/env python3
"""Entry point. One command from nothing to a fleet answer.

    python3 run.py --hours 24 --sizes 1 5 10 20 40 --solver fleet_solver:solve_many

Stages, each skipped when its output is already present and complete:

    1 metagraph   who is displaceable, and when immunity lapses          (seconds)
    2 dataset     rounds + every miner's answer, from the W&B stream     (minutes)
    3 selftest    the invariants in test_fleet_sim.py                    (seconds)
    4 solve       one call per round for max(--sizes) cliques            (HOURS)
    5 simulate    sweep the fleet sizes off that cache                   (seconds)

Stage 4 dominates and is resumable, so re-running after an interrupt costs only
what is left. --hours is the honest unit: rounds arrive at ~105/h across the two
validators, and nothing about deregistration is testable below 20 h because that
is the immunity window.

    python3 run.py --stage selftest        # just check the code is sound
    python3 run.py --hours 24 --dry-run    # print the plan and the ETA
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("SN83_DATA_DIR", os.path.join(HERE, "data"))
ROUNDS_PER_HOUR = 105.0          # measured across both v0.0.17 validators
SECONDS_PER_ROUND = 12.0         # solve cost on ~12 cores at the real deadlines
IMMUNITY_H = 20.0

STAGES = ["metagraph", "dataset", "selftest", "solve", "simulate"]


def sh(cmd, **kw):
    print(f"\n$ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, cwd=HERE, **kw).returncode


def _utc(ts):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))


def _digest(rows):
    """Fingerprint of exactly which rounds are in the test set."""
    import hashlib
    h = hashlib.sha1()
    for u in sorted(r["uuid"] for r in rows):
        h.update(u.encode())
    return h.hexdigest()[:16]


def count_lines(path):
    if not os.path.exists(path):
        return 0
    with open(path) as f:
        return sum(1 for _ in f)


def main():
    ap = argparse.ArgumentParser(
        description="build, verify and run the SN83 fleet simulation")
    ap.add_argument("--hours", type=float, default=24.0,
                    help="hours of subnet time to simulate. Deregistration is not "
                         f"testable below {IMMUNITY_H:.0f} h (the immunity window).")
    ap.add_argument("--sizes", type=int, nargs="+", default=[1, 5, 10, 20, 40])
    ap.add_argument("--solver", default="fleet_solver:solve_many",
                    help="module:func with signature (A, time_limit, k)")
    ap.add_argument("--picker", default="fleet_pick:picker",
                    help="module:func deciding what each QUERIED hotkey submits. "
                         "The solver owns this, not the harness: it may repeat a "
                         "clique rather than leave a hotkey silent, and it may "
                         "return nothing, which scores 0. fleet_pick:picker_silent "
                         "is the distinct-or-nothing control.")
    ap.add_argument("--latency-s", type=float, default=2.0,
                    help="constant reserved for the request/response round trip")
    ap.add_argument("--stage", choices=STAGES,
                    help="run only this stage")
    ap.add_argument("--from-stage", choices=STAGES, default="metagraph")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="redo stages even when their output looks complete")
    args = ap.parse_args()

    rounds = int(args.hours * ROUNDS_PER_HOUR)
    kmax = max(args.sizes)
    dataset = os.path.join(DATA, "sim_rounds.jsonl")
    cache = os.path.join(DATA, "sim_ts.jsonl")
    meta = os.path.join(DATA, "metagraph.json")
    solve_h = rounds * SECONDS_PER_ROUND / 3600.0

    print(f"target        {args.hours:.0f} h of subnet time = {rounds} rounds")
    print(f"fleet sizes   {args.sizes}  (solve pool k={kmax})")
    print(f"solve budget  ~{solve_h:.1f} h at {SECONDS_PER_ROUND:.0f}s/round, resumable")
    print(f"data dir      {DATA}")
    if args.hours < IMMUNITY_H:
        print(f"\n  NOTE: {args.hours:.0f} h is inside the {IMMUNITY_H:.0f} h immunity "
              f"window, so the fleet can never be evicted and 'survived' will be\n"
              f"        vacuous. Use --hours {IMMUNITY_H:.0f} or more to test "
              f"deregistration.")
    elif args.hours < 3 * IMMUNITY_H:
        exposed = args.hours - IMMUNITY_H
        print(f"\n  NOTE: immunity runs {IMMUNITY_H:.0f} h from join, so "
              f"--hours {args.hours:.0f} leaves only {exposed:.0f} h of exposure. At "
              f"~20 registrations/day\n        that is ~{exposed/24*20*10/249:.2f} "
              f"expected evictions of a 10-hotkey fleet — 'survived' will read N/N\n"
              f"        regardless of skill. ~72 h is the first window where it "
              f"discriminates.")

    want = STAGES[STAGES.index(args.stage or args.from_stage):] if not args.stage \
        else [args.stage]
    print(f"\nstages        {' -> '.join(want)}")
    if args.dry_run:
        return 0

    t0 = time.time()

    if "metagraph" in want:
        if args.force or not os.path.exists(meta):
            if sh([sys.executable, "snapshot_metagraph.py", "--out", meta]):
                return 1
        else:
            print(f"\n[skip] metagraph — {meta} exists")

    if "dataset" in want:
        have = count_lines(dataset)
        manifest = os.path.join(DATA, "testset.json")
        frozen = json.load(open(manifest)) if os.path.exists(manifest) else None

        if have >= rounds and not args.force:
            if frozen is None:
                # a dataset that predates the manifest: adopt it and pin it now,
                # otherwise it has no provenance and no frozen head
                rows = [json.loads(l) for l in open(dataset)]
                stamps = [r["timestamp"] for r in rows if r.get("timestamp")]
                frozen = {"rounds": len(rows), "first_round_utc": _utc(min(stamps)),
                          "last_round_utc": _utc(max(stamps)),
                          "span_hours": round((max(stamps)-min(stamps))/3600, 2),
                          "frozen_head_ts": max(stamps),
                          "frozen_head_utc": _utc(max(stamps)),
                          "built_utc": _utc(os.path.getmtime(dataset)),
                          "adopted_utc": _utc(time.time()),
                          "uuids_sha1": _digest(rows)}
                json.dump(frozen, open(manifest, "w"), indent=1)
                print(f"\n[pin]  dataset — adopted {have} pre-existing rounds")
            print(f"\n[skip] dataset — frozen at {have} rounds in {dataset}")
            print(f"       {frozen['first_round_utc']} .. "
                  f"{frozen['last_round_utc']}  sha1 {frozen['uuids_sha1']}")
            print("       --force draws a NEW test set; every earlier result "
                  "becomes incomparable")
        else:
            # Growing must only ever extend BACKWARDS. build_dataset truncates its
            # output and pulls from the CURRENT head, which advances ~105 rounds/h,
            # so a naive re-pull silently swaps in newer rounds at the top and can
            # drop rounds already scored. Keep everything we have, and admit only
            # rounds older than the boundary fixed at first build.
            existing = {}
            if have and not args.force:
                with open(dataset) as f:
                    for line in f:
                        r = json.loads(line)
                        existing[r["uuid"]] = r
                print(f"\n[grow] dataset — {have} rounds present, need {rounds}; "
                      f"extending backwards only")

            boundary = frozen.get("frozen_head_ts") if (frozen and existing) else None
            tmp = dataset + ".new"
            # build_dataset pulls backwards from the CURRENT head, so reaching a
            # frozen boundary means first pulling past everything logged since the
            # freeze. Without this an old test set can never grow: every fetched
            # round is newer than the boundary and gets skipped.
            behind = 0
            if boundary is not None:
                behind = int(max(0.0, time.time() - boundary) / 3600.0
                             * ROUNDS_PER_HOUR)
                print(f"       frozen head is {(time.time()-boundary)/3600:.1f} h old, "
                      f"so ~{behind} rounds must be skipped to reach it")
            per_run = int((rounds + behind) / 2) + 400   # two validators, plus slack
            if sh([sys.executable, "build_dataset.py", "--versions", "0.0.17",
                   "--limit", str(per_run), "--workers", "2", "--keep-answers",
                   "--out", tmp]):
                return 1

            merged = dict(existing)
            added = skipped = 0
            with open(tmp) as f:
                for line in f:
                    r = json.loads(line)
                    if r["uuid"] in merged:
                        continue
                    if boundary is not None and (r.get("timestamp") or 0) > boundary:
                        skipped += 1          # newer than the frozen head: not ours
                        continue
                    merged[r["uuid"]] = r
                    added += 1
            os.unlink(tmp)

            rows = sorted(merged.values(), key=lambda r: r.get("timestamp") or 0)
            with open(dataset, "w") as f:
                for r in rows:
                    f.write(json.dumps(r, separators=(",", ":")) + "\n")

            stamps = [r["timestamp"] for r in rows if r.get("timestamp")]
            head_ts = boundary if boundary is not None else max(stamps)
            json.dump({"rounds": len(rows),
                       "first_round_utc": _utc(min(stamps)),
                       "last_round_utc": _utc(max(stamps)),
                       "span_hours": round((max(stamps) - min(stamps)) / 3600, 2),
                       "frozen_head_ts": head_ts,
                       "frozen_head_utc": _utc(head_ts),
                       "built_utc": (frozen or {}).get("built_utc") or _utc(time.time()),
                       "last_grown_utc": _utc(time.time()),
                       "uuids_sha1": _digest(rows)},
                      open(manifest, "w"), indent=1)
            if existing:
                print(f"  kept {len(existing)} existing rounds, added {added} older "
                      f"ones, ignored {skipped} newer than the frozen head")
            print(f"  test set: {len(rows)} rounds, head frozen at {_utc(head_ts)} "
                  f"-> {manifest}")

    if "selftest" in want:
        if os.path.exists(dataset):
            # Cap the regression. It is O(rounds) and this stage is advertised as a
            # ~30 s check, but on the real 2778-round test set it runs past 10 min
            # and blocks the solve on EVERY resume. 300 rounds already exercises
            # 1199 (round, answer) pairs at 0.000e+00, ample to catch a scoring
            # regression; SN83_REF_ROUNDS=0 restores the full sweep.
            ref_n = os.environ.get("SN83_REF_ROUNDS", "300")
            if sh([sys.executable, "test_reward_reference.py", dataset, ref_n]):
                print("\nreward_reference regression FAILED — stopping")
                return 1
        else:
            print(f"\n[skip] reward_reference — no dataset yet at {dataset}; "
                  f"it is checked once stage 2 has run")
        rc = sh([sys.executable, "test_fleet_sim.py"])
        if rc == 2:
            print("  (selftest needs a solve cache; it will be checked after stage 4)")
        elif rc:
            print("\nfleet_sim invariants FAILED — stopping. Do not trust any output "
                  "until these pass.")
            return 1

    if "solve" in want:
        done = count_lines(cache)
        print(f"\n[solve] {done} rounds cached, need {rounds} at k={kmax} "
              f"(~{max(0, rounds-done)*SECONDS_PER_ROUND/3600:.1f} h left)")
        if sh([sys.executable, "fleet_sim.py", "--solve",
               "--rounds", str(rounds), "--sizes", *[str(x) for x in args.sizes],
               "--solver", args.solver, "--latency-s", str(args.latency_s),
               "--dataset", dataset, "--metagraph", meta, "--cache", cache]):
            return 1
        if sh([sys.executable, "test_fleet_sim.py"]):
            print("\ninvariants FAILED against the real cache — stopping")
            return 1

    if "simulate" in want:
        if sh([sys.executable, "fleet_sim.py",
               "--rounds", str(rounds), "--sizes", *[str(x) for x in args.sizes],
               "--picker", args.picker,
               "--dataset", dataset, "--metagraph", meta, "--cache", cache]):
            return 1

    print(f"\ndone in {(time.time()-t0)/60:.1f} min")
    print(f"per-task rows: {os.path.join(DATA, 'fleet_sim_results.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
