#!/usr/bin/env python3
"""Validator results as they land, from the validators' own W&B logs.

    deploy/live.sh 5HgXPXsB...          every round that queried this hotkey
    deploy/live_all.sh                  every miner of every round

    --backfill N    print the last N matching rounds before following (default 3)
    --poll S        seconds between polls (default 0.5)
    --validator P   only validators whose hotkey starts with P
    --jsonl FILE    also append each printed round, raw, to FILE
    --once          print the backfill and exit

Every number is the VALIDATOR's, read back from what it logged after scoring --
nothing is recomputed, so a row is what the miner was actually paid.

Where the delay comes from (measured 2026-09-18)
------------------------------------------------
A validator scores a round, stamps `timestamp`, and POSTs the result to the
subnet's relay (lambda.toptensor.ai), which writes it to W&B:

    validator -> relay -> W&B recorded      1-5 s   not ours to change
    W&B recorded -> readable by any client  2-12 s  W&B's own propagation
    this tool                               < 1 s   one 0.15 s query / 0.5 s

Two sources are polled and the first to show a round wins: the run SUMMARY
(always the latest round, a 0.2 KB query when narrowed to the needed keys) and
the run HISTORY (every round, used to fill any round the summary skipped when
two land between polls). `delay` on each row is measured from the validator's
own timestamp, so it shows the whole chain, not just this tool's share.

Only validators that log to W&B are visible. On 2026-09-18 that was one
(5EHGayLm..., version 0.0.17); other validators score rounds this cannot see.
"""
import argparse
import collections
import datetime as dt
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from monitor import PROJECT, load_api_key  # noqa: E402

ENTITY, PROJ = PROJECT.split("/", 1)

# What a row needs. Narrowing the summary to these is what makes a poll 0.2 KB
# instead of 48 KB: the summary also holds the round's graph.
KEYS = [
    "_step", "_timestamp", "timestamp", "uuid", "difficulty", "time_limit",
    "number_of_nodes", "miner_uids", "miner_hotkeys", "miner_ans", "miner_rel",
    "miner_optimality", "miner_diversity", "miner_rewards",
]

ACTIVE_S = 2 * 3600        # a run silent this long is not polled
REDISCOVER_S = 60          # how often to look for new validator runs
MISSING_GIVE_UP_S = 180    # stop waiting for a skipped round after this
# A validator posts a round every 1-2 minutes. Silence longer than this is the
# validator (stalled, restarting, or no longer logging), and the status line
# says so rather than leaving a quiet screen that looks like a hung tool.
QUIET_MIN = 10


# ------------------------------------------------------------------ rounds

def miner_rows(raw):
    """One dict per miner the round scored, in the validator's order."""
    hotkeys = raw.get("miner_hotkeys") or []
    answers = raw.get("miner_ans") or []

    def col(name, default=0.0):
        seq = raw.get(name) or []
        return [seq[i] if i < len(seq) else default for i in range(len(hotkeys))]

    uids = col("miner_uids", -1)
    rel = [float(x) for x in col("miner_rel")]
    opt = [float(x) for x in col("miner_optimality")]
    div = [float(x) for x in col("miner_diversity")]
    rew = [float(x) for x in col("miner_rewards")]
    ans = [list(answers[i]) if i < len(answers) and answers[i] else []
           for i in range(len(hotkeys))]
    # The validator zeroes an invalid or non-maximal clique before rel, so
    # rel > 0 is "counted"; the largest counted clique is the round's best.
    best = max((len(a) for a, r in zip(ans, rel) if a and r > 0), default=0)
    same = collections.Counter(tuple(sorted(a)) for a in ans if a)
    out = []
    for i, hk in enumerate(hotkeys):
        if not ans[i]:
            status = "NO ANSWER"
        elif rel[i] <= 0:
            status = "INVALID"
        else:
            status = "ok"
        out.append({
            "uid": int(uids[i]), "hotkey": hk, "status": status,
            "size": len(ans[i]), "best": best,
            "diversity": div[i], "optimality": opt[i], "reward": rew[i],
            "place": 1 + sum(1 for r in rew if r > rew[i] + 1e-12),
            "dup": same[tuple(sorted(ans[i]))] if ans[i] else 0,
        })
    return out


def round_head(raw, validator):
    ts = float(raw.get("timestamp") or 0)
    return {
        "time": dt.datetime.fromtimestamp(ts, dt.timezone.utc).strftime("%H:%M:%S"),
        "validator": validator,
        "delay": time.time() - ts if ts else float("nan"),
        "uuid": str(raw.get("uuid") or "")[:8],
        "n": int(raw.get("number_of_nodes") or 0),
        "tl": float(raw.get("time_limit") or 0),
        "d": float(raw.get("difficulty") or 0),
    }


HOTKEY_HEAD = ("time(UTC) validator  delay  round     n    tl   D    "
               "answered   size/best  div    opt    reward place   dup")


def fmt_hotkey(h, m, n_miners):
    return ("%s  %-9s %6.1fs  %-8s  %-4d %-4g %-4.1f %-10s %3d/%-3d    "
            "%-6.3f %-6.3f %-6.3f %-7s %d" % (
                h["time"], h["validator"], h["delay"], h["uuid"], h["n"], h["tl"],
                h["d"], m["status"], m["size"], m["best"], m["diversity"],
                m["optimality"], m["reward"], "%d/%d" % (m["place"], n_miners),
                m["dup"]))


def fmt_all(h, rows):
    answered = sum(1 for m in rows if m["status"] != "NO ANSWER")
    valid = sum(1 for m in rows if m["status"] == "ok")
    best = rows[0]["best"] if rows else 0
    lines = ["", "=== %s UTC  validator %s  delay %.1fs  round %s  n=%d tl=%g D=%.1f  "
             "%d miners: %d answered, %d valid, best clique %d" % (
                 h["time"], h["validator"], h["delay"], h["uuid"], h["n"], h["tl"],
                 h["d"], len(rows), answered, valid, best),
             "  place  uid  answered   size/best  div    opt    reward  dup  hotkey"]
    for m in sorted(rows, key=lambda m: (-m["reward"], m["uid"])):
        lines.append("  %-6s %-4d %-10s %3d/%-3d    %-6.3f %-6.3f %-7.3f %-4d %s" % (
            m["place"], m["uid"], m["status"], m["size"], m["best"],
            m["diversity"], m["optimality"], m["reward"], m["dup"], m["hotkey"]))
    return "\n".join(lines)


# --------------------------------------------------------------- W&B access

class Source(object):
    """The active validator runs, and a batched summary poll across them."""

    def __init__(self, api, validator_prefix=""):
        self.api = api
        self.prefix = validator_prefix
        self.runs = {}          # name -> {"validator", "run"}
        self._query = None
        self._query_names = None
        self.discovered = 0.0

    def discover(self):
        runs = self.api.runs(PROJECT, filters={"state": "running"}, per_page=100)
        now = time.time()
        found = {}
        for r in runs:
            hotkey = str(r.config.get("hotkey") or r.name.split("-")[0])
            if self.prefix and not hotkey.startswith(self.prefix):
                continue
            last = float(r.summary.get("_timestamp") or 0)
            # keep a run we were already following even if it went quiet:
            # dropping it would hide the moment it comes back
            if now - last > ACTIVE_S and r.name not in self.runs:
                continue
            found[r.name] = {
                "validator": hotkey[:8],
                "version": r.config.get("version", "?"),
                "run": r,
            }
        self.runs = found
        self.discovered = now
        return found

    def poll(self):
        """{run name: summary dict} for every active run, in ONE request."""
        from wandb_gql import gql
        names = sorted(self.runs)
        if not names:
            return {}
        if names != self._query_names:
            parts = " ".join(
                "r%d: run(name: %s) { summaryMetrics(keys: %s) }"
                % (i, json.dumps(n), json.dumps(KEYS)) for i, n in enumerate(names))
            self._query = gql("query L($e: String!, $p: String!) { project("
                              "name: $p, entityName: $e) { %s } }" % parts)
            self._query_names = names
        res = self.api.client.execute(self._query, variable_values={"e": ENTITY, "p": PROJ})
        out = {}
        for i, n in enumerate(names):
            node = (res.get("project") or {}).get("r%d" % i) or {}
            raw = node.get("summaryMetrics")
            if raw:
                out[n] = json.loads(raw)
        return out

    def history(self, name, lo, hi):
        """Rounds with lo <= step < hi, oldest first."""
        run = self.runs[name]["run"]
        rows = [r for r in run.scan_history(keys=KEYS, min_step=lo, max_step=hi,
                                            page_size=500) if r.get("uuid")]
        return sorted(rows, key=lambda r: r["_step"])


# ------------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("hotkey", nargs="?", help="ss58 of the miner to follow")
    p.add_argument("--all", action="store_true", help="every miner of every round")
    p.add_argument("--backfill", type=int, default=3)
    p.add_argument("--poll", type=float, default=0.5)
    p.add_argument("--validator", default="")
    p.add_argument("--jsonl", default=None)
    p.add_argument("--once", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="also report rounds that did not query the hotkey")
    args = p.parse_args()
    if not args.all and not args.hotkey:
        p.error("give a hotkey, or --all")

    key = load_api_key()
    if not key:
        raise SystemExit("no WANDB_API_KEY in the environment or .env")
    os.environ["WANDB_API_KEY"] = key
    os.environ.setdefault("WANDB_SILENT", "true")
    import wandb
    api = wandb.Api(api_key=key, timeout=30)

    src = Source(api, args.validator)
    if not src.discover():
        # wait rather than exit: the loop rediscovers every REDISCOVER_S, so a
        # validator that comes back is picked up without restarting this
        print("  [no validator run has logged in the last %d h -- waiting for one]"
              % (ACTIVE_S // 3600), file=sys.stderr, flush=True)
    print("following %s  |  %d validator(s): %s  |  poll %.1fs" % (
        "ALL miners" if args.all else "hotkey " + args.hotkey,
        len(src.runs), ", ".join("%s (v%s)" % (v["validator"], v["version"]) for v in src.runs.values()),
        args.poll), file=sys.stderr, flush=True)

    out_jsonl = open(args.jsonl, "a") if args.jsonl else None
    seen = collections.defaultdict(set)   # run -> steps already handled
    last = {}                             # run -> newest step known
    newest_ts = {n: float(v["run"].summary.get("timestamp") or 0)
                 for n, v in src.runs.items()}   # run -> its newest round's time
    missing = {}                          # (run, step) -> when first missed
    stats = {"rounds": 0, "ours": 0, "delays": []}
    if not args.all:
        print(HOTKEY_HEAD)
        print("-" * len(HOTKEY_HEAD))

    def emit(name, raw, live=True, late=False):
        step = int(raw["_step"])
        if step in seen[name]:
            return
        seen[name].add(step)
        head = round_head(raw, src.runs[name]["validator"])
        rows = miner_rows(raw)
        tag = "  (late: summary skipped it)" if late else ""
        if live:
            stats["rounds"] += 1
        if args.all:
            print(fmt_all(head, rows) + tag, flush=True)
            if live:
                stats["delays"].append(head["delay"])
        else:
            mine = [m for m in rows if m["hotkey"] == args.hotkey]
            if mine:
                if live:
                    stats["ours"] += 1
                    stats["delays"].append(head["delay"])
                print(fmt_hotkey(head, mine[0], len(rows)) + tag, flush=True)
            elif args.verbose:
                print("%s  %-9s %6.1fs  %-8s  not queried (%d miners)%s" % (
                    head["time"], head["validator"], head["delay"], head["uuid"],
                    len(rows), tag), flush=True)
        if out_jsonl:
            out_jsonl.write(json.dumps(dict(raw, validator=head["validator"])) + "\n")
            out_jsonl.flush()

    # backfill: the last rounds that match, so the screen is never empty. The
    # `delay` on these is their age, and they are kept out of the delay stats.
    for name in src.runs:
        head_step = int(src.runs[name]["run"].summary.get("_step") or 0)
        want = max(0, args.backfill)
        got = []
        # A hotkey is queried on a fraction of rounds, so look back further
        # than `want` -- widening only if the first window is not enough.
        for span in (want if args.all else 300, 2000):
            if not want:
                break
            rows = src.history(name, max(0, head_step + 1 - span), head_step + 1)
            got = [r for r in rows
                   if args.all or args.hotkey in (r.get("miner_hotkeys") or [])]
            if len(got) >= want:
                break
        for raw in got[-want:] if want else []:
            emit(name, raw, live=False)
        last[name] = head_step
        seen[name].add(head_step)
    if args.backfill and not args.all:
        print("-" * len(HOTKEY_HEAD) + "  live from here")

    def freshness():
        now = time.time()
        parts = []
        for n in sorted(src.runs):
            ts = newest_ts.get(n, 0.0)
            if not ts:
                continue
            age = (now - ts) / 60.0
            parts.append("%s last posted %s UTC (%s)" % (
                src.runs[n]["validator"],
                dt.datetime.fromtimestamp(ts, dt.timezone.utc).strftime("%H:%M:%S"),
                "%.0f min ago -- validator silent, not this tool" % age
                if age >= QUIET_MIN else "%.0f min ago" % age))
        return "; ".join(parts) or "no validator has posted recently"

    print("  [%s]" % freshness(), file=sys.stderr, flush=True)
    if args.once:
        return

    last_status = time.time()
    last_fill = 0.0
    while True:
        t0 = time.time()
        try:
            if t0 - src.discovered > REDISCOVER_S:
                src.discover()
            for name, s in src.poll().items():
                if s.get("_step") is None:
                    continue
                step = int(s["_step"])
                newest_ts[name] = max(newest_ts.get(name, 0.0),
                                      float(s.get("timestamp") or 0))
                top = last.get(name)
                if top is None:                  # a validator run that just appeared
                    top = step - 1
                # the summary only ever holds the NEWEST round: anything it
                # jumped over has to come from history, which lags behind it
                for gap in range(top + 1, step):
                    if gap not in seen[name]:
                        missing.setdefault((name, gap), t0)
                last[name] = max(top, step)
                emit(name, s)                    # newest round, straight away
            if missing and t0 - last_fill > 2.0:
                last_fill = t0
                for name in sorted({k[0] for k in missing}):
                    steps = sorted(st for (n, st) in missing if n == name)
                    for raw in src.history(name, steps[0], steps[-1] + 1):
                        st = int(raw["_step"])
                        if (name, st) in missing:
                            emit(name, raw, late=True)
                            del missing[(name, st)]
                    for st in steps:
                        if (name, st) in missing and t0 - missing[(name, st)] > MISSING_GIVE_UP_S:
                            print("  [round at step %d of %s never became readable]"
                                  % (st, src.runs[name]["validator"]),
                                  file=sys.stderr, flush=True)
                            del missing[(name, st)]
        except Exception as exc:        # a flaky poll must not end a long watch
            print("  [poll failed: %s]" % str(exc)[:120], file=sys.stderr, flush=True)
            time.sleep(2.0)
        quiet = all(time.time() - newest_ts.get(n, 0.0) > QUIET_MIN * 60
                    for n in src.runs) if src.runs else True
        if time.time() - last_status > (60 if quiet else 300):
            d = sorted(stats["delays"][-50:])
            print("  [%s UTC: %d rounds seen%s%s | %s]" % (
                dt.datetime.now(dt.timezone.utc).strftime("%H:%M:%S"),
                stats["rounds"],
                "" if args.all else ", %d queried this hotkey" % stats["ours"],
                ", delay p50 %.1fs max %.1fs" % (d[len(d) // 2], d[-1]) if d else "",
                freshness()),
                file=sys.stderr, flush=True)
            last_status = time.time()
        time.sleep(max(0.0, args.poll - (time.time() - t0)))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
