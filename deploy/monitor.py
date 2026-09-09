#!/usr/bin/env python3
"""One row per validator request, from the validators' own W&B logs.

    deploy/monitor.py                      last 50 rounds that queried us
    deploy/monitor.py --limit 200
    deploy/monitor.py --since 6h
    deploy/monitor.py --archive rounds.jsonl     append raw rows while printing
    deploy/monitor.py --replay  rounds.jsonl     re-read an archive, no network
    deploy/monitor.py --csv out.csv
    deploy/monitor.py --watch 300                poll every 5 minutes

Every number here is the VALIDATOR's, read back from what it logged after
scoring. Nothing is recomputed locally, so a row is what we were actually paid,
not what we hoped for.

Columns
-------
size    our clique size (0 = no answer, or an answer the validator rejected)
best    the largest valid clique anyone submitted that round
rel     size / best
opt     optimality, ALREADY normalised so the round's best miner scores 1.000
div     diversity, likewise normalised; 1/(miners submitting our exact vertex set)
        before normalisation, so a clique we share with others is worth less
reward  opt * (1 + difficulty) + div  -- the final score, max 2 + difficulty
place   rank by reward among the miners queried that round (ties share a place)
dup     how many miners submitted our exact vertex set, us included
"""

import argparse
import collections
import csv
import datetime as dt
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

PROJECT = os.environ.get("SN83_WANDB_PROJECT", "toptensor-ai/CliqueAI")

# The fields the validator logs. Asking scan_history for a narrow key set is
# what keeps this fast -- adjacency_list and encoded_matrix are megabytes a row
# and nothing here needs them.
KEYS = [
    "uuid", "timestamp", "difficulty", "time_limit", "number_of_nodes",
    "miner_uids", "miner_hotkeys", "miner_ans",
    "miner_rel", "miner_pr", "miner_omega",
    "miner_optimality", "miner_diversity", "miner_rewards",
]


def load_api_key():
    """WANDB_API_KEY from the environment, else from the repo's .env."""
    key = os.environ.get("WANDB_API_KEY")
    if key:
        return key
    path = os.path.join(REPO, ".env")
    if os.path.exists(path):
        with open(path) as handle:
            for line in handle:
                line = line.strip()
                if line.startswith("WANDB_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def our_hotkey(explicit=None):
    """The hotkey to report on: --hotkey, else WALLET_* from deploy/sn83.env."""
    if explicit:
        return explicit
    env = os.path.join(HERE, "sn83.env")
    name = hotkey = None
    if os.path.exists(env):
        with open(env) as handle:
            for line in handle:
                line = line.strip()
                if line.startswith("export WALLET_NAME="):
                    name = line.split("=", 1)[1].strip()
                elif line.startswith("export WALLET_HOTKEY="):
                    hotkey = line.split("=", 1)[1].strip()
    if not (name and hotkey):
        raise SystemExit("cannot infer the hotkey; pass --hotkey <ss58>")
    path = os.path.expanduser(
        "~/.bittensor/wallets/%s/hotkeys/%s" % (name, hotkey))
    with open(path) as handle:
        return json.load(handle)["ss58Address"]


def parse_since(text):
    """'6h', '30m', '2d' -> a unix timestamp, or None."""
    if not text:
        return None
    unit = text[-1].lower()
    mult = {"s": 1, "m": 60, "h": 3600, "d": 86400}.get(unit)
    if mult is None:
        raise SystemExit("--since wants a suffix of s/m/h/d, e.g. 6h")
    return time.time() - float(text[:-1]) * mult


# W&B history is paged by step, and one step is one round. A miner is queried
# on a fraction of rounds, so a fixed window finds nothing for a fleet of one:
# walk backwards in chunks until enough of OUR rounds have turned up.
# Bigger chunks and pages mean fewer HTTP round trips; the scan is entirely
# latency-bound, not bandwidth-bound, because KEYS excludes the big fields.
CHUNK_STEPS = 2000
PAGE_SIZE = 500
# ~1 round/minute, so 5000 steps is roughly the last three days. A miner that
# has not been queried in three days has a problem the scan cannot diagnose.
DEFAULT_MAX_STEPS = 5000


def _open_api():
    import wandb

    major, minor = (int(x) for x in wandb.__version__.split(".")[:2])
    if (major, minor) >= (0, 20):
        print("warning: wandb %s; scan_history was verified on 0.19.x"
              % wandb.__version__, file=sys.stderr)
    key = load_api_key()
    if not key:
        raise SystemExit("no WANDB_API_KEY in the environment or .env")
    return wandb.Api(api_key=key)


def fetch_rows(hotkey, limit, version, since, verbose=False,
               max_steps=DEFAULT_MAX_STEPS):
    """History rows that queried `hotkey`, newest run first.

    Returns (rows, scanned, oldest_ts) so the caller can say how far it looked
    -- an empty result means "not queried in this window", not "not mining",
    and the two are worth telling apart.
    """
    print("scanning %s for hotkey %s..." % (PROJECT, hotkey[:12] + "..."),
          file=sys.stderr, flush=True)
    api = _open_api()
    filters = {"config.version": {"$in": [version]}} if version else None
    runs = list(api.runs(PROJECT, filters=filters, per_page=100))
    if not runs:
        raise SystemExit("no runs in %s%s" % (
            PROJECT, " for version %s" % version if version else ""))
    # A running validator first: it holds the rounds that just happened.
    runs.sort(key=lambda r: (r.state != "running", -int(r.summary.get("_step") or 0)))

    hits, raw_all, scanned, oldest = [], [], 0, None
    for run in runs:
        head = run.summary.get("_step")
        if head is None:
            continue
        head = int(head)
        hi = head + 1
        while hi > 0 and len(hits) < limit and (head + 1 - hi) < max_steps:
            lo = max(0, hi - CHUNK_STEPS)
            chunk = [r for r in run.scan_history(keys=KEYS, min_step=lo,
                                                 max_step=hi, page_size=PAGE_SIZE)
                     if r.get("uuid")]
            scanned += len(chunk)
            for r in chunk:
                ts = float(r.get("timestamp") or 0)
                oldest = ts if oldest is None else min(oldest, ts)
                if since and ts < since:
                    continue
                raw_all.append(r)
                if hotkey in (r.get("miner_hotkeys") or []):
                    hits.append(r)
            # progress goes to stderr unconditionally: a silent multi-minute
            # scan is indistinguishable from a hang, which is how this first got
            # reported as "it returned nothing".
            print("  %s steps %d-%d: %d rounds, %d ours%s"
                  % (run.name.split("-")[0][:12], lo, hi, len(chunk), len(hits),
                     " (%s)" % run.config.get("version") if verbose else ""),
                  file=sys.stderr, flush=True)
            # Walked past the requested window; older chunks cannot help.
            if since and oldest is not None and oldest < since:
                break
            hi = lo
        if len(hits) >= limit:
            break
    return raw_all, scanned, oldest


def build_row(raw, hotkey):
    """One report row, or None if this round did not query us."""
    hotkeys = raw.get("miner_hotkeys") or []
    if hotkey not in hotkeys:
        return None
    i = hotkeys.index(hotkey)

    def at(field, default=0.0):
        seq = raw.get(field) or []
        return seq[i] if i < len(seq) else default

    answers = raw.get("miner_ans") or []
    rewards = [float(x) for x in (raw.get("miner_rewards") or [])]
    ours = list(answers[i]) if i < len(answers) and answers[i] else []

    # The validator zeroes an invalid or non-maximal clique, so size alone does
    # not say whether it counted. rel > 0 does.
    rel = float(at("miner_rel"))
    size = len(ours)
    sizes_valid = [len(a or []) * (1 if float(r) > 0 else 0)
                   for a, r in zip(answers, rewards)]
    best = max(sizes_valid) if sizes_valid else 0

    canon = collections.Counter(tuple(sorted(a or [])) for a in answers)
    dup = canon[tuple(sorted(ours))] if ours else 0

    reward = float(at("miner_rewards"))
    place = 1 + sum(1 for r in rewards if r > reward + 1e-12)

    return {
        "timestamp": float(raw.get("timestamp") or 0),
        "uuid": raw.get("uuid", ""),
        "n": int(raw.get("number_of_nodes") or 0),
        "time_limit": float(raw.get("time_limit") or 0),
        "difficulty": float(raw.get("difficulty") or 0),
        "uid": int(at("miner_uids", 0)),
        "size": size,
        "best": best,
        "rel": rel,
        "optimality": float(at("miner_optimality")),
        "diversity": float(at("miner_diversity")),
        "reward": reward,
        "place": place,
        "n_miners": len(hotkeys),
        "dup": dup,
        "max_reward": 2.0 + float(raw.get("difficulty") or 0),
    }


HEAD = ("time      uuid      n    tl    d    size best  rel    opt    div    "
        "reward place    dup")


def format_row(r):
    when = dt.datetime.fromtimestamp(r["timestamp"], dt.timezone.utc).strftime("%H:%M:%S")
    flag = ""
    if r["reward"] <= 0:
        flag = "  <- ZERO"
    elif r["size"] and r["size"] == r["best"]:
        flag = "  *"          # we matched the best clique anyone found
    return ("%s  %-8s %-4d %-5.1f %-4.1f %-4d %-4d %-6.3f %-6.3f %-6.3f %-6.3f "
            "%-8s %-3d%s" % (
                when, str(r["uuid"])[:8], r["n"], r["time_limit"], r["difficulty"],
                r["size"], r["best"], r["rel"], r["optimality"], r["diversity"],
                r["reward"], "%d/%d" % (r["place"], r["n_miners"]), r["dup"], flag))


def summarise(rows):
    if not rows:
        return "no rounds"
    n = len(rows)
    rewards = [r["reward"] for r in rows]
    places = [r["place"] for r in rows]
    at_best = sum(1 for r in rows if r["size"] and r["size"] == r["best"])
    zeros = sum(1 for r in rows if r["reward"] <= 0)
    shared = sum(1 for r in rows if r["dup"] > 1)
    span = rows[-1]["timestamp"] - rows[0]["timestamp"]
    out = [
        "",
        "%d rounds over %.1f h" % (n, span / 3600.0) if span > 0 else "%d rounds" % n,
        "  reward     mean %.4f   min %.4f   max %.4f" % (
            sum(rewards) / n, min(rewards), max(rewards)),
        "  place      mean %.2f   best %d   worst %d   (of %d miners typical)" % (
            sum(places) / n, min(places), max(places),
            collections.Counter(r["n_miners"] for r in rows).most_common(1)[0][0]),
        "  matched best clique   %d/%d  (%.1f%%)" % (at_best, n, 100.0 * at_best / n),
        "  scored zero           %d/%d  (%.1f%%)" % (zeros, n, 100.0 * zeros / n),
        "  answer shared with a rival  %d/%d  (%.1f%%)" % (
            shared, n, 100.0 * shared / n),
    ]
    return "\n".join(out)


def emit(rows, args):
    print(HEAD)
    print("-" * len(HEAD))
    for r in rows:
        print(format_row(r))
    print(summarise(rows))
    if args.csv:
        with open(args.csv, "w", newline="") as handle:
            w = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
            w.writeheader()
            w.writerows(rows)
        print("\ncsv -> %s" % args.csv, file=sys.stderr)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hotkey", default=None, help="ss58; default from sn83.env")
    p.add_argument("--limit", type=int, default=50, help="rounds to show (default 50)")
    p.add_argument("--since", default=None, help="e.g. 6h, 30m, 2d")
    p.add_argument("--version", default="",
                   help="validator version filter (e.g. 0.0.17); default: all")
    p.add_argument("--csv", default=None)
    p.add_argument("--archive", default=None,
                   help="append the raw W&B rows to this jsonl as they are read")
    p.add_argument("--replay", default=None,
                   help="read raw rows from a jsonl instead of W&B")
    p.add_argument("--watch", type=int, default=0, help="poll every N seconds")
    p.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS,
                   dest="max_steps",
                   help="how far back to scan, in rounds (default %d)" % DEFAULT_MAX_STEPS)
    p.add_argument("-v", "--verbose", action="store_true",
                   help="report the step windows being scanned")
    args = p.parse_args()

    hotkey = our_hotkey(args.hotkey)
    since = parse_since(args.since)

    def once():
        if args.replay:
            with open(args.replay) as handle:
                raw = [json.loads(l) for l in handle if l.strip()]
        else:
            raw, scanned, oldest = fetch_rows(
                hotkey, args.limit, args.version or None, since, args.verbose,
                args.max_steps)
            if args.archive:
                seen = set()
                if os.path.exists(args.archive):
                    with open(args.archive) as handle:
                        seen = {json.loads(l).get("uuid")
                                for l in handle if l.strip()}
                with open(args.archive, "a") as handle:
                    for row in raw:
                        if row.get("uuid") not in seen:
                            handle.write(json.dumps(row) + "\n")
        rows = [x for x in (build_row(r, hotkey) for r in raw) if x]
        rows.sort(key=lambda r: r["timestamp"])
        print("hotkey %s   project %s" % (hotkey, PROJECT))
        if not rows and not args.replay:
            since_txt = ("since %s UTC" % dt.datetime.fromtimestamp(
                oldest, dt.timezone.utc).strftime("%Y-%m-%d %H:%M")) if oldest else ""
            print("\nNot queried in the %d rounds scanned %s.\n"
                  "That is not the same as not mining -- a validator only sees a new\n"
                  "hotkey after it resyncs its metagraph, and it samples a fraction of\n"
                  "miners per round. Check the axon is served with:\n"
                  "    btcli s metagraph --netuid 83 | grep <your uid>"
                  % (scanned, since_txt))
            return
        emit(rows[-args.limit:], args)

    once()
    while args.watch:
        time.sleep(args.watch)
        print("\n" + "=" * 100)
        once()


if __name__ == "__main__":
    main()
