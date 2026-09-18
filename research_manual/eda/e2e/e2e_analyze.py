#!/usr/bin/env python3
"""Reduce an E2E run: validity (validator's own check), timing budget, fallbacks,
sibling duplicates, dispatcher counters, and the rare-query alert contract."""
import collections
import glob
import json
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))   # <repo>
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research_manual"))
from CliqueAI.graph.codec import GraphCodec  # noqa: E402
from CliqueAI.scoring.clique_scoring import CliqueScoreCalculator  # noqa: E402
import pick_derived  # noqa: E402

R = sys.argv[1]
rounds = json.load(open(os.path.join(ROOT, "research_manual/artifacts/data/rounds.json")))
rows = [json.loads(l) for l in open(os.path.join(R, "validator.jsonl"))]
meta = [r for r in rows if r.get("tag") == "meta"]
resp = [r for r in rows if "clique" in r]
K = meta[0]["fleet"]
ev = []
for f in glob.glob(os.path.join(R, "events", "*.jsonl")):
    ev.extend(json.loads(l) for l in open(f))
by_kind = collections.defaultdict(list)
for e in ev:
    by_kind[e["kind"]].append(e)

codec = GraphCodec()
_calc = {}


class G:
    def __init__(self, n, adj):
        self.number_of_nodes = n
        self.adjacency_list = adj
        self.uuid = ""
        self.label = "general"


def calc(round_id):
    if round_id not in _calc:
        rec = rounds[round_id]
        m = codec.decode_matrix(rec["encoded_matrix"])
        adj = codec.matrix_to_list(m)
        c = CliqueScoreCalculator(graph=G(rec["number_of_nodes"], adj),
                                  difficulty=rec["difficulty"], responses=[])
        valid_sizes = [len(a[3]) for a in rec["answers"] if a[3] and c.is_valid_maximum_clique(list(a[3]))]
        _calc[round_id] = (c, max(valid_sizes) if valid_sizes else 0)
    return _calc[round_id]


def pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    return xs[min(len(xs) - 1, int(round(p / 100.0 * (len(xs) - 1))))]


def summ(xs):
    return "n=%d p50=%.3f p90=%.3f p99=%.3f max=%.3f" % (
        len(xs), pct(xs, 50), pct(xs, 90), pct(xs, 99), max(xs) if xs else float("nan"))


disp = {(e["uuid"], e["index"]): e for e in by_kind["dispatch"]}
recv = {(e["uuid"], e["index"]): e for e in by_kind["recv"]}
fdone = {(e["uuid"], e["index"]): e for e in by_kind["forward_done"]}
local = collections.defaultdict(list)
for e in by_kind["local_fallback"]:
    local[e["index"]].append(e)

out = {}
print("fleet K =", K, " responses =", len(resp), " rounds =", len({r["uuid"] for r in resp}))
for phase in sorted({r["tag"] for r in resp}):
    rs = [r for r in resp if r["tag"] == phase]
    uu = {r["uuid"] for r in rs}
    lat, mfwd, dhttp, over = [], [], [], []
    bad_status = invalid = fallback = 0
    below_field = 0
    roles = collections.Counter()
    dup_rounds = 0
    dup_answers = 0
    by_round = collections.defaultdict(list)
    for r in rs:
        by_round[r["uuid"]].append(r)
        key = (r["uuid"], r["index"])
        if r["status"] != 200:
            bad_status += 1
        lat.append(r["t_headers"] - r["t_send"])
        if key in recv and key in fdone:
            mfwd.append(fdone[key]["t1"] - recv[key]["mono"])
        d = disp.get(key)
        if d:
            dhttp.append(d["t1"] - d["t0"])
            roles[d["role"] or "NONE(timeout/err)"] += 1
            if not d["size"]:
                fallback += 1
            if key in recv and key in fdone:
                over.append((fdone[key]["t1"] - recv[key]["mono"]) - (d["t1"] - d["t0"]))
        c, best = calc(r["round_id"])
        ok = bool(r["clique"]) and c.is_valid_maximum_clique(list(r["clique"]))
        if not ok:
            invalid += 1
        elif len(r["clique"]) < best:
            below_field += 1
    for u, group in by_round.items():
        keys = [tuple(sorted(g["clique"])) for g in group if g["clique"]]
        dups = len(keys) - len(set(keys))
        if dups:
            dup_rounds += 1
            dup_answers += dups
    depth = collections.Counter(min(by_round[u][0]["depth_at_send"], 5) for u in uu)
    print("\n=== phase %s: rounds=%d answers=%d" % (phase, len(uu), len(rs)))
    print("  concurrency at send (rounds already in flight):", dict(sorted(depth.items())))
    print("  validator-observed latency  ", summ(lat))
    print("  miner forward (recv->done)  ", summ(mfwd))
    print("  miner->dispatcher HTTP      ", summ(dhttp))
    print("  miner overhead (fwd - http) ", summ(over))
    print("  latency > 2.0s: %d   > 3.0s: %d" % (sum(x > 2.0 for x in lat), sum(x > 3.0 for x in lat)))
    print("  non-200: %d   invalid (validator check): %d   below field-best size: %d" % (bad_status, invalid, below_field))
    print("  dispatcher role:", dict(roles), "  empty dispatcher answer -> local fallback:", fallback)
    print("  rounds with duplicate cliques among our siblings: %d (extra dup answers %d)" % (dup_rounds, dup_answers))
    by_tl = collections.defaultdict(list)
    for r in rs:
        by_tl[r["tl"]].append(r["t_headers"] - r["t_send"])
    for tl in sorted(by_tl):
        print("    tl=%-4s latency %s" % (tl, summ(by_tl[tl])))

# ---- alert contract ----
print("\n=== rare-query alert")
q_of = {}
tl_of = {}
D_of = {}
tag_of = {}
for r in resp:
    q_of[r["uuid"]] = r["q"]
    tl_of[r["uuid"]] = r["tl"]
    D_of[r["uuid"]] = r["D"]
    tag_of[r["uuid"]] = r["tag"]
alerts = collections.defaultdict(list)
for e in by_kind["ALERT"]:
    alerts[e["uuid"]].append(e)
watches = collections.Counter(e["uuid"] for e in by_kind["watch_start"])
snap = {e["uuid"]: e for e in by_kind["claims_snapshot"]}
waitr = {e["uuid"]: e for e in by_kind["wait_result"]}
print("  watch threads started: %d on %d uuids; max per uuid %d" % (
    sum(watches.values()), len(watches), max(watches.values()) if watches else 0))
print("  watch started on tl<10 rounds:", sum(1 for u in watches if tl_of.get(u, 99) < 10))
print("  alerts fired: %d on %d uuids; max per uuid %d" % (
    sum(len(v) for v in alerts.values()), len(alerts), max((len(v) for v in alerts.values()), default=0)))
expected = set()
for u in q_of:
    if tl_of[u] >= 10 and pick_derived.p_at_most_queried(q_of[u], K, D_of[u]) < 0.05:
        expected.add(u)
fired = set(alerts)
print("  expected (tl>=10 and P(X<=q_final)<0.05): %d   fired: %d   missed: %d   unexpected: %d" % (
    len(expected), len(fired), len(expected - fired), len(fired - expected)))
for u in sorted(expected - fired):
    print("    MISSED", tag_of[u], u[:8], "q", q_of[u], "tl", tl_of[u], "snapshot", snap.get(u, {}).get("claims"),
          "wait", waitr.get(u, {}).get("ok"), "watch", watches.get(u, 0))
for u in sorted(fired - expected):
    print("    UNEXPECTED", tag_of[u], u[:8], "q_final", q_of[u], "claims_used", alerts[u][0]["claims"], "tl", tl_of[u])
early = 0
for u, al in alerts.items():
    a = al[0]
    sib_fd = [e for e in by_kind["forward_done"] if e["uuid"] == u]
    sib = [e for e in by_kind["dispatch"] if e["uuid"] == u] or sib_fd
    last_disp = max(e["t1"] for e in sib)
    last_fwd = max(e["t1"] for e in sib_fd)
    ok = a["mono"] >= last_disp and a["mono"] >= last_fwd and len(sib) == q_of[u]
    if not ok:
        early += 1
    print("    %-26s %s q=%d claims_used=%d task_at_fire=%s  fire-after-last-sibling-dispatch=%+.3fs  -forward=%+.3fs  siblings_seen=%d %s" % (
        tag_of[u], u[:8], q_of[u], a["claims"], a["task"], a["mono"] - last_disp, a["mono"] - last_fwd,
        len(sib), "OK" if ok else "EARLY/INCOMPLETE"))
print("  alerts fired before every sibling finished:", early)
for u in sorted(u for u in snap if tag_of.get(u, "").startswith("S")):
    print("    scenario %-26s %s q=%d snapshot_claims=%s wait_ok=%s alert=%d" % (
        tag_of[u], u[:8], q_of[u], snap[u]["claims"], waitr.get(u, {}).get("ok"), len(alerts.get(u, []))))

for f in ("health_start.json", "health_end.json"):
    p = os.path.join(R, f)
    if os.path.exists(p) and os.path.getsize(p):
        h = json.load(open(p))
        print("\n%s counters=%s gpu_threads=%s core_plan=%s" % (f, h["counters"], h["gpu_threads"], h["core_plan"]))
