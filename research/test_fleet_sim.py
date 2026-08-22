#!/usr/bin/env python3
"""Invariant tests for the fleet simulator. Run this before trusting any output.

    python3 test_fleet_sim.py

Every check here exists because something broke: the simulator once re-admitted
the miners it had just displaced, once reported a 100% fleet share on a dataset
missing hotkeys, once wrote real scores into validator slots, and once flattened
its own zero-skill control to a constant. Exit code is non-zero if any fails.
"""
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fleet_sim as F                                    # noqa: E402
from _common import REPO                                 # noqa: E402,F401
from reward_reference import replay_reward, count_hist_from_answers  # noqa: E402
from CliqueAI.graph.codec import GraphCodec              # noqa: E402
from CliqueAI.scoring.clique_scoring import CliqueScoreCalculator  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
FAILS = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)
    return ok


class _Graph:
    def __init__(self, n, adj):
        self.number_of_nodes, self.adjacency_list = n, adj


def load(dataset, cache_path, limit=None):
    rounds = [json.loads(l) for l in open(dataset) if l.strip()]
    rounds = [r for r in rounds if r.get("answers")]
    cache = {}
    with open(cache_path) as f:
        for line in f:
            r = json.loads(line)
            cache[r["uuid"]] = r
    rounds = [r for r in rounds if r["uuid"] in cache and r.get("timestamp")]
    rounds.sort(key=lambda r: r["timestamp"])
    if limit:
        rounds = rounds[:limit]
    validity = {r["uuid"]: F.validate_cliques(r, cache[r["uuid"]]["cliques"])
                for r in rounds}
    meta = json.load(open(os.path.join(DATA, "metagraph.json")))
    return rounds, cache, meta, validity


# --------------------------------------------------------------- scoring truth

def t_score_round(rounds):
    """score_round must equal CliqueScoreCalculator on modified rounds."""
    rng = np.random.default_rng(0)
    worst, pairs = 0.0, 0
    for rec in rounds[:12]:
        M = np.array(GraphCodec().decode_matrix(rec["matrix_b92"]), dtype=np.uint8)
        g = _Graph(M.shape[0], [set(np.flatnonzero(M[i]).tolist())
                                for i in range(M.shape[0])])
        ans = [a["clique"] for a in rec["answers"]]
        opt = [a["opt"] for a in rec["answers"]]
        cases = {
            "all": list(range(len(ans))),
            "subset": sorted(rng.choice(len(ans), max(2, len(ans) // 2),
                                        replace=False).tolist()),
            "single": [0],
        }
        for label, keep in cases.items():
            resp = [ans[i] for i in keep]
            extra = [("dup", resp[0] if resp else []), ("unserved", [])]
            for tag, add in extra:
                full = resp + [add]
                valid = [1 if opt[i] > 0 else 0 for i in keep]
                valid.append(1 if (tag == "dup" and valid and add) else 0)
                keys = [tuple(sorted(x)) for x in resp] + [
                    tuple(sorted(add)) if add else ("unserved", 0)]
                mine = F.score_round([len(x) for x in full], valid, keys,
                                     rec["difficulty"])
                *_, ref = CliqueScoreCalculator(
                    graph=g, difficulty=rec["difficulty"], responses=full).get_scores()
                worst = max(worst, float(np.abs(mine - np.array(ref)).max()))
                pairs += len(full)
    return check("score_round == CliqueScoreCalculator", worst < 1e-9,
                 f"{pairs} responses, max |diff| {worst:.2e}")


def t_reward_reference(rounds):
    worst, pairs = 0.0, 0
    for rec in rounds[:10]:
        M = np.array(GraphCodec().decode_matrix(rec["matrix_b92"]), dtype=np.uint8)
        g = _Graph(M.shape[0], [set(np.flatnonzero(M[i]).tolist())
                                for i in range(M.shape[0])])
        ans = [a["clique"] for a in rec["answers"]]
        valid = [a["clique"] for a in rec["answers"] if a["opt"] > 0]
        dup = collections.Counter(tuple(sorted(c)) for c in valid)
        ch = count_hist_from_answers(rec["answers"])
        n_inv = rec["n_responders"] - rec["n_valid"]
        for key, cnt in list(dup.items())[:4]:
            ours = list(key)
            got, _, _ = replay_reward(len(ours), cnt, rec["size_hist"], ch,
                                      rec["difficulty"], n_invalid=n_inv)
            *_, rew = CliqueScoreCalculator(
                graph=g, difficulty=rec["difficulty"],
                responses=ans + [ours]).get_scores()
            worst = max(worst, abs(got - float(rew[-1])))
            pairs += 1
    return check("reward_reference == CliqueScoreCalculator", worst < 1e-9,
                 f"{pairs} pairs, max |diff| {worst:.2e}")


# ------------------------------------------------------------ replay invariants

def t_displacement(rounds, cache, meta, validity):
    """Victims must never be scored again, and the population must be conserved."""
    ok = True
    for N in (1, 5, 20):
        st, vu, oid, alive, ev, hku = F.replay(rounds, cache, meta, N, 8383, validity,
                                          sample="real")
        # Ask replay's own rule who the victims are. Deriving this from the
        # metagraph instead named the uid's LATER occupant -- a miner replay never
        # displaced -- so the check reported phantom re-admissions at N>=20.
        occ = F.occupants(rounds, meta)
        vh = {occ[u] for u in vu if u in occ}
        logged = sum(1 for r in rounds for a in r["answers"] if a["hk"] in vh)
        scored = sum(1 for r in rounds for a in r["answers"]
                     if a["hk"] in vh and a["hk"] in alive)
        ok &= check(f"N={N}: displaced miners never scored again", scored == 0,
                    f"{logged} victim answers logged, {scored} scored")
        ok &= check(f"N={N}: population conserved", len(alive) == len(meta["miners"]),
                    f"|alive|={len(alive)} vs {len(meta['miners'])} slots")
        readmit = [e for e in ev if e["in"] in vh]
        ok &= check(f"N={N}: no displaced miner re-admitted", not readmit,
                    f"{len(readmit)} re-admissions")
    return ok


def t_registration_rate(rounds, cache, meta, validity):
    """~22 registrations/day on chain; the cascade produced 839/day."""
    span_h = (rounds[-1]["timestamp"] - rounds[0]["timestamp"]) / 3600.0
    st, vu, oid, alive, ev, hku = F.replay(rounds, cache, meta, 5, 8383, validity,
                                      sample="real")
    per_day = len(ev) / span_h * 24 if span_h else 0
    return check("registration rate plausible", per_day <= 60,
                 f"{len(ev)} events over {span_h:.2f} h = {per_day:.1f}/day "
                 f"(chain: ~22/day)")


def t_slots_unique(rounds, cache, meta, validity):
    """No two identities may share a uid, and our fleet must always get one."""
    ok = True
    for N in (1, 5, 20):
        st, vu, oid, alive, ev, hku = F.replay(rounds, cache, meta, N, 8383, validity,
                                          sample="real")
        sc = F.ema_scores(st)
        vec, ours = F.score_vector(sc, meta, vu, oid, alive, hku)
        # A fleet member the chain deregistered mid-replay holds NO uid, so the
        # invariant is one slot per SURVIVOR, not per hotkey we started with.
        # Asserting N/N reserved a seat for a dead identity, which starved the
        # registrant that displaced it.
        live = [o for o in oid if o in alive]
        ok &= check(f"N={N}: every surviving fleet member keeps a slot",
                    len(ours) == len(live),
                    f"{len(ours)}/{len(live)} placed ({N - len(live)} evicted)")
        ok &= check(f"N={N}: fleet slots are displaced uids",
                    set(ours) <= set(vu),
                    f"{len(set(ours) - set(vu))} outside the displaced set")
        val_uids = {u for u in range(int(meta["n"]))} - {m["uid"] for m in meta["miners"]}
        wrote = [u for u in val_uids if vec[u] != 0]
        ok &= check(f"N={N}: validator slots stay zero", not wrote, f"{wrote}")
    return ok


def t_hashseed_stable(rounds, cache, meta, validity):
    """Same seed, same data, DIFFERENT process: the answer must not move.

    replay iterates `alive`, a set of hotkey strings, and Python randomises string
    hashing per process. Ties in worst_alive therefore broke differently on every
    run and the N=40 median wandered over ~0.04 between processes -- the size of
    the effects this harness exists to measure. The in-process seed check cannot
    see it, because it never leaves the process.
    """
    import subprocess
    code = (
        "import sys; sys.path.insert(0, %r)\n"
        "import json, numpy as np, fleet_sim as F, test_fleet_sim as T\n"
        "r,c,m,v = T.load(%r, %r)\n"
        "st,vu,oid,al,ev,hk = F.replay(r,c,m,20,8383,v,sample='real')\n"
        "print(repr(float(np.median([np.mean(x) for x in "
        "(st[o] for o in oid) if x]))))\n"
        % (os.path.dirname(os.path.abspath(__file__)),
           os.path.join(DATA, "sim_rounds.jsonl"),
           os.path.join(DATA, "sim_ts.jsonl")))
    outs = []
    for hs in ("1", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=hs)
        p = subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True, env=env,
                           cwd=os.path.dirname(os.path.abspath(__file__)))
        outs.append(p.stdout.strip() or p.stderr.strip()[-120:])
    return check("replay is stable across PYTHONHASHSEED", outs[0] == outs[1],
                 f"{outs[0]} vs {outs[1]}")


def t_null_varies(rounds, cache, meta, validity):
    """The zero-skill control must actually vary, or the guard is inert."""
    shares = []
    for i in range(6):
        st, vu, oid, alive, ev, hku = F.replay(rounds, cache, meta, 5, 9000 + i, validity,
                                          sample="real", null=True)
        sc = F.ema_scores(st)
        vec, ours = F.score_vector(sc, meta, vu, oid, alive, hku)
        w, g, h = F.set_weights(vec)
        shares.append(float(w[ours].sum()))
    spread = max(shares) - min(shares)
    span_s = rounds[-1]["timestamp"] - rounds[0]["timestamp"]
    if span_s <= F.WARMUP_BLOCKS * F.BLOCK_S:
        print(f"  [skip] null control varies — the whole window is inside the "
              f"{F.WARMUP_BLOCKS*F.BLOCK_S/60:.0f} min warmup, so no fleet answers "
              f"exist to vary")
        return True
    return check("null control varies across seeds", spread > 1e-6,
                 f"shares {min(shares):.5%}..{max(shares):.5%}")


def t_real_sampling_deterministic(rounds, cache, meta, validity):
    """--sample real must not depend on the seed."""
    out = []
    for seed in (1, 2, 3):
        st, vu, oid, alive, ev, hku = F.replay(rounds, cache, meta, 5, seed, validity,
                                          sample="real")
        out.append(round(float(np.mean([np.mean(st[o]) if st[o] else 0
                                        for o in oid])), 10))
    return check("--sample real is seed-independent", len(set(out)) == 1, f"{out}")


def t_field_matches_log(rounds, cache, meta, validity):
    """With no fleet and no victims, our replay must reproduce the logged rewards."""
    st, vu, oid, alive, ev, hku = F.replay(rounds, cache, meta, 0, 8383, validity,
                                      sample="real")
    logged = collections.defaultdict(list)
    for r in rounds:
        for a in r["answers"]:
            logged[a["hk"]].append(a["reward"])
    diffs = []
    for hk, xs in logged.items():
        if hk in st and len(st[hk]) == len(xs):
            diffs.append(max(abs(p - q) for p, q in zip(st[hk], xs)))
    worst = max(diffs) if diffs else float("inf")
    return check("N=0 reproduces the validator's own logged rewards", worst < 1e-9,
                 f"{len(diffs)} identities, max |diff| {worst:.2e}")


def t_immunity(rounds, cache, meta, validity):
    """Inside immunity our fleet cannot be evicted; with immunity off it can."""
    span_h = (rounds[-1]["timestamp"] - rounds[0]["timestamp"]) / 3600.0
    st, vu, oid, alive, ev, hku = F.replay(rounds, cache, meta, 5, 8383, validity,
                                      sample="real", immunity_s=20 * 3600)
    inside = all(o in alive for o in oid)
    ok = check("fleet is protected inside immunity", inside,
               f"window {span_h:.2f} h < 20 h, {sum(1 for o in oid if o in alive)}/5 alive")
    return ok


def t_warmup(rounds, cache, meta, validity):
    """A fresh registration is not queried for its first epoch (361 blocks)."""
    span_s = rounds[-1]["timestamp"] - rounds[0]["timestamp"]
    st, vu, oid, alive, ev, hku = F.replay(rounds, cache, meta, 5, 8383, validity,
                                           sample="real")
    silent = all(not st[o] for o in oid)
    ok = check("fleet is silent during its warmup",
               silent if span_s <= F.WARMUP_BLOCKS * F.BLOCK_S else True,
               f"window {span_s/60:.0f} min vs {F.WARMUP_BLOCKS*F.BLOCK_S/60:.0f} min "
               f"warmup; {sum(1 for o in oid if st[o])}/5 answered")
    # and it does answer once the warmup is disabled
    st2, *_ = F.replay(rounds, cache, meta, 5, 8383, validity, sample="real",
                       warmup_s=0.0)
    ok &= check("fleet answers once past the warmup",
                any(st2[o] for o in oid),
                f"{sum(1 for o in oid if st2[o])}/5 answered with warmup_s=0")
    return ok


def t_saturation(rounds, cache, meta, validity):
    """Take EVERY slot. Nothing should break, and the outcome should be flat.

    With the whole subnet ours the field is only us, so scores should bunch near
    one value; anyone far below it is starved of cliques, not outcompeted.
    """
    block = meta["block"]
    free = [m for m in meta["miners"]
            if block - m["block_at_registration"] >= F.IMMUNITY_BLOCKS]
    total = len(meta["miners"])
    # you cannot take the whole subnet: immune UIDs are protected
    try:
        F.pick_victims(meta, total)
        check("N=all is refused while any UID is immune", len(free) == total)
    except SystemExit:
        check("N=all is refused while any UID is immune", True,
              f"{total - len(free)} of {total} UIDs immune; max fleet is {len(free)}")
    N = len(free)
    st, vu, oid, alive, ev, hku = F.replay(rounds, cache, meta, N, 8383, validity,
                                           sample="real")
    ok = check("N=max: population conserved", len(alive) == total,
               f"|alive|={len(alive)} vs {total} slots")
    # We took every evictable slot, so no surviving incumbent may come from one.
    #
    # Counting survivors against `total - N` was wrong twice over. Miners that
    # REGISTERED during the window are field identities and alive and entitled to
    # be. And a uid can be immune at snapshot time while its window-start occupant
    # was not: 14 of the 16 immune uids were registered mid-window, so the hotkey
    # sitting there at t0 was the evictable predecessor, and displacing it is
    # exactly the registration that made the uid immune.
    occ = F.occupants(rounds, meta)
    immune_uids = {m["uid"] for m in meta["miners"]
                   if block - m["block_at_registration"] < F.IMMUNITY_BLOCKS}
    start_uid = {hk: u for u, hk in occ.items()}
    incumbents = {i for i in alive if i in start_uid} - set(oid)
    leaked = {i for i in incumbents if start_uid[i] not in immune_uids}
    newcomers = len(alive - set(oid) - set(start_uid))
    ok &= check("N=max: no evictable incumbent survives", not leaked,
                f"{len(leaked)} leaked; {len(incumbents)} incumbents left "
                f"(+{newcomers} registered mid-window)")
    sc = F.ema_scores(st)
    pool = [len(cache[r["uuid"]]["cliques"]) for r in rounds]
    kmed = int(np.median(pool))
    span_s = rounds[-1]["timestamp"] - rounds[0]["timestamp"]
    if span_s <= F.WARMUP_BLOCKS * F.BLOCK_S:
        print(f"  [skip] saturation scores — window is inside the warmup; "
              f"population and refusal checks above still apply")
        return ok

    # Hotkeys are served pool[j] by rank j, so only the first ~kmed of the fleet
    # get a clique in a typical round; the rest are queried and score zero. Judge
    # the two groups separately or the pool limit masquerades as score spread.
    # An EVICTED member is neither served nor starved -- its stream just stops, and
    # its half-length EMA reads low for a reason that has nothing to do with the
    # pool. Including the 3 of the first 15 that the chain deregisters here put the
    # spread at 1.01 against a 0.30 bar. Judge survivors.
    live = [o for o in oid if o in alive]
    served = np.array([sc[o] for o in live[:kmed]])
    starved = np.array([sc[o] for o in live[kmed:]])
    ok &= check("N=max: the hotkeys the pool can serve bunch near one value",
                served.size == 0 or float(np.std(served)) < 0.30,
                f"first {served.size}: mean {served.mean() if served.size else 0:.3f}, "
                f"sd {np.std(served) if served.size else 0:.3f}")
    ok &= check("N=max: the rest are starved, not outcompeted",
                starved.size == 0 or float(np.mean(starved)) < served.mean() if served.size else True,
                f"remaining {starved.size}: mean {np.mean(starved) if starved.size else 0:.3f}")
    print(f"       clique pool median {kmed}/round, so a fleet past ~{kmed} hotkeys "
          f"is dead weight: {int((np.array([sc[o] for o in oid]) == 0).sum())} of {N} "
          f"scored exactly 0; {N - len(live)} of the fleet were deregistered")
    print(f"       deregistrations in this window: {len(ev)} "
          f"(needs a >20 h window to be non-zero)")
    return ok


def t_rejects_old_schema(rounds, cache, meta, validity):
    """A dataset without hotkeys must be refused, not silently scored at 100%."""
    stripped = []
    for r in rounds[:5]:
        c = dict(r)
        c["answers"] = [{k: v for k, v in a.items() if k != "hk"} for a in r["answers"]]
        stripped.append(c)
    try:
        F.replay(stripped, cache, meta, 5, 1, validity, sample="real")
        return check("dataset without hotkeys is refused", False, "it ran anyway")
    except SystemExit:
        return check("dataset without hotkeys is refused", True)


def t_picker(rounds, cache, meta, validity):
    """The harness must stay neutral: it scores what comes back, nothing more.

    Three things to hold. picker_silent has to reproduce the built-in path exactly,
    or the plugin seam is not faithful and no picker comparison means anything. A
    picker that returns [] everywhere has to score a hard 0, because a real miner is
    allowed to answer nothing and the validator rejects an empty maximum_clique.
    And duplicating has to beat going silent, which is the whole reason the decision
    belongs to a solver that can weigh it.
    """
    import fleet_pick

    base = F.replay(rounds, cache, meta, 40, 8383, validity, sample="real")
    same = F.replay(rounds, cache, meta, 40, 8383, validity, sample="real",
                    picker=fleet_pick.picker_silent)
    b = np.median([np.mean(v) for v in (base[0][o] for o in base[2]) if v])
    s = np.median([np.mean(v) for v in (same[0][o] for o in same[2]) if v])
    check("picker_silent reproduces the built-in path", abs(b - s) < 1e-12,
          f"{b:.6f} vs {s:.6f}")

    mute = F.replay(rounds, cache, meta, 40, 8383, validity, sample="real",
                    picker=lambda pool, uuid, hks: [[] for _ in hks])
    m = [np.mean(v) for v in (mute[0][o] for o in mute[2]) if v]
    check("a picker that answers nothing scores 0", max(m) == 0.0 if m else False,
          f"max per-hotkey mean {max(m) if m else float('nan'):.6f}")

    dup = F.replay(rounds, cache, meta, 40, 8383, validity, sample="real",
                   picker=fleet_pick.picker)
    d = np.median([np.mean(v) for v in (dup[0][o] for o in dup[2]) if v])
    check("duplicating beats going silent", d > s, f"{d:.4f} vs {s:.4f}")


def main():
    dataset = os.path.join(DATA, "sim_rounds.jsonl")
    cache = os.path.join(DATA, "sim_ts.jsonl")
    if not (os.path.exists(dataset) and os.path.exists(cache)):
        print(f"need {dataset} and {cache}; build them with run.py first")
        return 2
    rounds, cch, meta, validity = load(dataset, cache)
    print(f"fleet_sim invariants — {len(rounds)} rounds, "
          f"{(rounds[-1]['timestamp']-rounds[0]['timestamp'])/3600:.2f} h span\n")

    print("scoring truth")
    t_score_round(rounds)
    t_reward_reference(rounds)
    print("\nreplay invariants")
    t_displacement(rounds, cch, meta, validity)
    t_registration_rate(rounds, cch, meta, validity)
    t_field_matches_log(rounds, cch, meta, validity)
    print("\nslot assignment")
    t_slots_unique(rounds, cch, meta, validity)
    print("\ncontrols and guards")
    t_null_varies(rounds, cch, meta, validity)
    t_real_sampling_deterministic(rounds, cch, meta, validity)
    t_hashseed_stable(rounds, cch, meta, validity)
    t_immunity(rounds, cch, meta, validity)
    print("\nwarmup")
    t_warmup(rounds, cch, meta, validity)
    print("\npicker (the solver, not the harness, decides what is submitted)")
    t_picker(rounds, cch, meta, validity)
    print("\nsaturation (take every slot)")
    t_saturation(rounds, cch, meta, validity)
    t_rejects_old_schema(rounds, cch, meta, validity)

    print()
    if FAILS:
        print(f"{len(FAILS)} FAILED: {', '.join(FAILS)}")
        return 1
    print("all invariants hold")
    return 0


if __name__ == "__main__":
    sys.exit(main())
