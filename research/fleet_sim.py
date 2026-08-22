#!/usr/bin/env python3
"""Simulate entering SN83 with a fleet of N hotkeys, and score it fairly.

Registering N hotkeys does not just add N answers to a round — it *removes* N
miners, and every clique we return that collides with a survivor drags that
survivor's diversity down too. Both effects change the field we are being ranked
against, so the round has to be rebuilt and rescored from scratch:

  1. pick the N victims: lowest-incentive miners that are not immune, in order
  2. for each logged round, drop the victims' answers entirely
  3. sample which of our N hotkeys the validator would have queried
     (independent Bernoulli at P(difficulty), same as MinerSelector)
  4. ask the solver for that many distinct cliques, best first, and insert them
  5. rescore the whole round with the validator's formula
  6. run the resulting per-UID reward streams through update_scores/set_weights
     and report where our hotkeys land

Two phases, because the solving is the only expensive part:

  solve   one call per round for max(--sizes) distinct cliques, cached to disk.
          ~12 s a round, resumable, and shared across every N.
  replay  every N swept off that cache, seconds.

The rounds are replayed in their real order — this is not a sampled split. The
solver is an algorithm, not something fitted to the data, so the honest test is
the sequence the validator actually ran. make_splits.py exists only for the parts
that ARE fitted, such as a collision picker, which can overfit a held-out set.

    python3 fleet_sim.py --sizes 1 5 10 20 40 --rounds 1000 --solve
    python3 fleet_sim.py --sizes 1 5 10 20 40 --rounds 1000     # replay only
"""
import argparse
import collections
import json
import os
import sys
import time

import numpy as np

from _common import DATA_DIR

IMMUNITY_BLOCKS = 6000
REF_R = 1.5
# MinerSelector._filter_validators skips a uid while
# current_block - last_update <= epoch_length, so a freshly registered neuron is
# not queried at all for its first epoch: 361 blocks x 12s = 72.2 min.
WARMUP_BLOCKS = 361
BLOCK_S = 12.0


def selection_p(difficulty):
    """MinerSelector.miner_selection_probabilities, per hotkey per round."""
    return 1.0 - np.exp(-max(0.0, np.sqrt(1.0 + REF_R) - difficulty - 0.5))


def score_round(sizes, valid, keys, difficulty):
    """CliqueScoreCalculator.get_scores(), vectorised, without needing the graph.

    sizes  clique size per response (any value; masked by `valid`)
    valid  1 if the validator would accept it, else 0
    keys   hashable canonical vertex set per response, used for duplicate counts
    Returns the reward for every response, index-aligned.
    """
    size = np.asarray(sizes, dtype=float) * np.asarray(valid, dtype=float)
    n = len(size)
    if n == 0 or size.max() <= 0:
        return np.zeros(n)
    rel = size / size.max()
    pr = np.array([(size > s).sum() / n for s in size])
    omega = np.where(size > 0, np.exp(-pr / np.maximum(rel, 1e-12)), 0.0)
    optimality = omega / omega.max() if omega.max() > 0 else omega

    counts = collections.Counter(k for k, v in zip(keys, valid) if v)
    delta = np.array([(1.0 / counts[k]) if v else 0.0 for k, v in zip(keys, valid)])
    diversity = delta / delta.max() if delta.max() > 0 else delta
    return optimality * (1 + difficulty) + diversity


def pick_victims(meta, n):
    """The n UIDs a fresh registration would displace, worst first.

    Subtensor replaces the lowest pruning score among non-immune neurons; for
    miners that tracks incentive. Immune UIDs (registered within IMMUNITY_BLOCKS)
    are protected and skipped.
    """
    block = meta["block"]
    cands = [m for m in meta["miners"]
             if block - m["block_at_registration"] >= IMMUNITY_BLOCKS]
    cands.sort(key=lambda m: m["incentive"])
    if len(cands) < n:
        raise SystemExit(f"only {len(cands)} non-immune miners; cannot displace {n}")
    return [m["uid"] for m in cands[:n]]


# ---------------------------------------------------------------- solve phase

def solve_phase(rounds, k, solver, latency_s, cache_path):
    """One solver call per round for up to k distinct cliques. Resumable."""
    import importlib
    mod, fn = solver.split(":")
    solve_many = getattr(importlib.import_module(mod), fn)

    done = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            for line in f:
                r = json.loads(line)
                done[r["uuid"]] = r
    todo = [r for r in rounds if r["uuid"] not in done
            or len(done[r["uuid"]]["cliques"]) < k]
    print(f"solve: {len(todo)} rounds to do, {len(done)} cached", file=sys.stderr)

    from CliqueAI.graph.codec import GraphCodec
    t0 = time.time()
    with open(cache_path, "a") as out:
        for i, rec in enumerate(todo, 1):
            A = np.array(GraphCodec().decode_matrix(rec["matrix_b92"]), dtype=np.uint8)
            t = time.time()
            # deadline minus a CONSTANT round-trip reserve. A fraction of the
            # deadline is wrong in both directions: it hands a 6s task 5.28s (32%
            # more than deployment allows, and 6s/7.5s are 35% of rounds) while
            # starving a 30s one of 1.6s it would really have.
            cl = solve_many(A, max(0.5, rec["time_limit"] - latency_s), k)
            row = {"uuid": rec["uuid"],
                   "cliques": [sorted(int(v) for v in c) for c in cl],
                   "elapsed": time.time() - t}
            out.write(json.dumps(row) + "\n")
            out.flush()
            done[rec["uuid"]] = row
            if i % 10 == 0:
                rate = (time.time() - t0) / i
                print(f"  {i}/{len(todo)}  {rate:.1f}s/round, "
                      f"{(len(todo)-i)*rate/60:.0f} min left", file=sys.stderr, flush=True)
    return done


def validate_cliques(rec, cliques):
    """Apply the validator's own validity+maximality test to our answers."""
    from CliqueAI.graph.codec import GraphCodec
    A = np.array(GraphCodec().decode_matrix(rec["matrix_b92"]), dtype=np.uint8)
    n = A.shape[0]
    out = []
    for c in cliques:
        S = list(c)
        ok = bool(S) and len(set(S)) == len(S) and min(S) >= 0 and max(S) < n
        if ok:
            idx = np.array(S)
            ok = A[np.ix_(idx, idx)].sum() == len(S) * (len(S) - 1)
            if ok:
                inC = np.zeros(n, dtype=bool)
                inC[idx] = True
                ok = not np.any((A[idx].sum(axis=0) == len(S)) & (~inC))
        out.append(ok)
    return out


# --------------------------------------------------------------- replay phase

def worst_alive(streams, alive, joined, now, immunity_s, alpha=0.01):
    """The identity a fresh registration would displace: lowest current score
    among those out of immunity. Ours are eligible too — that is the point.

    Immunity is 6000 blocks of WALL CLOCK (~20 h), not a round count: rounds from
    two validators interleave at ~105/h, so counting rounds would misdate it.
    """
    best_id, best_score = None, None
    for ident in alive:
        if now - joined.get(ident, -1e18) < immunity_s:
            continue
        xs = streams.get(ident, [])
        e = 0.0
        for x in xs:
            e = alpha * x + (1 - alpha) * e
        corr = 1 - (1 - alpha) ** len(xs) if xs else 0.0
        sc = e / corr if corr > 0 else 0.0
        if best_score is None or sc < best_score:
            best_id, best_score = ident, sc
    return best_id


def replay(rounds, cache, meta, N, seed, validity, immunity_s=20 * 3600,
           apply_dereg=True, null=False, sample="real",
           warmup_s=WARMUP_BLOCKS * BLOCK_S):
    """Replay the real rounds in order, with our fleet in place of N miners.

    Identity is the HOTKEY, not the uid slot, because slots get recycled. Three
    things happen as the replay walks forward:

      * a freshly registered neuron is not queried for its first epoch (361
        blocks, ~72 min), because MinerSelector filters on
        current_block - last_update <= epoch_length. That applies to our fleet
        at t0 and to every mid-replay registrant, and skipping it biases the
        early EMA upward.
      * our fleet answers the rounds it is sampled into. With sample="real" we do
        not re-roll that: we take over specific uid SLOTS, and the log already
        records whether each slot was queried in each round. Selection is
        `np.full(n, P)` — uniform across uids, independent of incentive or stake —
        so the slot's own history is an exact draw from the right distribution,
        with none of the variance a fresh Bernoulli adds. A queried hotkey we
        cannot serve takes a zero rather than vanishing
      * every round is rescored in full, so our collisions lower the colliding
        miner's diversity, and removing a miner can RAISE a survivor's diversity
        if they had been colliding with the one we removed
      * with apply_dereg, each re-registration visible in the data evicts the
        worst non-immune identity in OUR world and admits the new hotkey. That
        may be one of ours — which is the question the simulation exists to
        answer.
    """
    if not any(a.get("hk") for r in rounds for a in r["answers"]):
        raise SystemExit(
            "this dataset has no per-answer hotkeys, so field identities cannot be "
            "tracked and score_vector would keep only our fleet - which reports a "
            "100% share. Rebuild with build_dataset.py --keep-answers.")
    rng = np.random.default_rng(seed)
    victim_uids = pick_victims(meta, N)

    # Who occupies each uid AT THE START OF THE WINDOW, taken from the data, not
    # from the metagraph snapshot. The snapshot is captured later, so a miner that
    # answered during the window and was deregistered before the snapshot is
    # missing from it entirely — seeding `alive` from the snapshot silently
    # dropped those answers and broke the N=0 identity with the log.
    uid_hotkey_now = {}
    for rec in rounds:
        for a in rec["answers"]:
            uid_hotkey_now.setdefault(a["uid"], a["hk"])
    for m in meta["miners"]:                       # uids that never answered
        uid_hotkey_now.setdefault(m["uid"], m["hotkey"])
    victim_hotkeys = {uid_hotkey_now[u] for u in victim_uids if u in uid_hotkey_now}

    our_ids = [f"OURS-{i}" for i in range(N)]
    our_slot = dict(zip(our_ids, victim_uids))       # which uid each of ours holds
    alive = (set(uid_hotkey_now.values()) - victim_hotkeys) | set(our_ids)
    t0 = rounds[0].get("timestamp") or 0.0
    joined = {ident: -1e18 for ident in alive}    # incumbents are long out of immunity
    for o in our_ids:
        joined[o] = t0                            # we register at the first round

    streams = collections.defaultdict(list)
    for o in our_ids:
        streams[o] = []
    uid_seen = {}
    events = []
    # Displaced identities are gone for good. Without this, a victim's very next
    # answer looked like a fresh registration ("hk not in alive"), it was
    # re-admitted, and it evicted somebody else - a cascade that re-admitted 249
    # of 255 identities and left 100% of victim answers still being scored. The
    # simulator displaced nobody.
    retired = set(victim_hotkeys)
    n_slots = len(meta["miners"])
    rejected = 0        # registrations the chain could not place: all slots immune

    for t, rec in enumerate(rounds):
        now = rec.get("timestamp") or (t0 + t * 34.0)      # ~34s median round gap
        if apply_dereg:
            for a in rec["answers"]:
                hk, uid = a.get("hk"), a["uid"]
                if hk is None:
                    continue
                prev = uid_seen.get(uid)
                # a registration is an OCCUPANT CHANGE at a uid, not merely an
                # unfamiliar hotkey
                if (prev is not None and prev != hk and hk not in retired
                        and hk not in alive):
                    # The chain deregisters the lowest pruning score. In our world
                    # the named predecessor is the faithful choice when it is still
                    # here; when it is not, it is because WE took that slot, and
                    # then the eviction falls to the worst - which can be us.
                    evicted = prev if prev in alive else worst_alive(
                        streams, alive, joined, now, immunity_s)
                    if evicted is None:
                        # Every neuron is inside immunity, so the chain has no slot
                        # to recycle and rejects the registration. Admitting anyway
                        # grew the population past the uid count.
                        rejected += 1
                        uid_seen[uid] = hk
                        continue
                    alive.discard(evicted)
                    retired.add(evicted)
                    alive.add(hk)
                    joined[hk] = now
                    events.append({"round": t, "t": now, "uid": uid,
                                   "in": hk, "out": evicted})
                uid_seen[uid] = hk
            if len(alive) > n_slots:
                raise AssertionError(
                    f"population {len(alive)} exceeds {n_slots} miner slots at round {t}")

        d = rec["difficulty"]
        # The log is ground truth: a hotkey we first observe at time T was
        # registered at least a warmup earlier, because it could not have been
        # queried before that. Applying warmup to the field would double-count
        # the delay and delete answers that really happened.
        survivors = [a for a in rec["answers"] if a["hk"] in alive]
        sizes = [len(a["clique"]) for a in survivors]
        valid = [1 if a["opt"] > 0 else 0 for a in survivors]
        keys = [tuple(sorted(a["clique"])) for a in survivors]
        idents = [a["hk"] for a in survivors]

        def warm(ident):
            """Past the first-epoch filter? Only meaningful for our own fleet."""
            return now - joined.get(ident, -1e18) > warmup_s

        if sample == "real":
            # the slot was queried this round iff it appears in the log
            queried = {a["uid"] for a in rec["answers"]}
            picked = [o for o in our_ids
                      if o in alive and warm(o) and our_slot[o] in queried]
        else:
            p = selection_p(d)
            picked = [o for o in our_ids
                      if o in alive and warm(o) and rng.random() < p]
        pool = cache.get(rec["uuid"], {}).get("cliques", [])
        ok = validity.get(rec["uuid"], [])
        for j, o in enumerate(picked):
            if j >= len(pool):
                # queried but unserved: the validator scores [] as invalid and
                # still takes an EMA step
                sizes.append(0); valid.append(0); keys.append(("unserved", o))
            else:
                c = pool[j]
                sizes.append(len(c))
                valid.append(1 if (j < len(ok) and ok[j]) else 0)
                keys.append(tuple(c))
            idents.append(o)

        rewards = score_round(sizes, valid, keys, d)
        if null:
            surv = [r for r, i in zip(rewards, idents) if not str(i).startswith("OURS-")]
            if surv:
                rewards = [float(rng.choice(surv)) if str(i).startswith("OURS-") else r
                           for r, i in zip(rewards, idents)]
        for i, r in zip(idents, rewards):
            streams[i].append(float(r))

    if rejected:
        print(f"  note: {rejected} registrations rejected - every neuron was inside "
              f"immunity, which is what the chain does when no slot can be recycled",
              file=sys.stderr)
    return (streams, victim_uids, our_ids, alive, events,
            {h: u for u, h in uid_hotkey_now.items()})


def ema_scores(streams, alpha=0.01):
    """update_scores: per-UID debiased EMA -> the score set_weights sees."""
    out = {}
    for u, xs in streams.items():
        e = 0.0
        for x in xs:
            e = alpha * x + (1 - alpha) * e
        corr = 1 - (1 - alpha) ** len(xs)
        out[u] = e / corr if corr > 0 else 0.0
    return out


def score_vector(sc, meta, victim_uids, our_ids, alive, hk_uid=None):
    """The vector set_weights actually sees: np.zeros(metagraph.n), uid-indexed.

    Validators are never selected by MinerSelector, so their slots hold 0 forever
    and min_val is always 0 — the min-max step is a stretch of the score axis, not
    an affine no-op, so handing it a compressed vector changes the answer.

    Slots are assigned EXCLUSIVELY. Our fleet claims the uids it displaced first,
    because the metagraph snapshot is taken later than the replayed rounds and can
    already map a mid-replay registrant onto one of those uids; letting both write
    the slot silently overwrote our score with a rival's, which flattened the
    zero-skill null into a constant.
    """
    n = int(meta["n"])
    vec = np.zeros(n)
    taken = {}
    for our, victim in zip(our_ids, victim_uids):
        taken[our] = victim
    reserved = set(taken.values())

    if hk_uid is None:
        hk_uid = {m["hotkey"]: m["uid"] for m in meta["miners"]}
    # ALIVE identities claim their slot first. `sc` also holds identities that were
    # DEREGISTERED mid-replay, and they still map to the uid they used to occupy --
    # so iterating in stream order let a retired hotkey reserve the very slot its
    # replacement needed, and the live registrant was then homeless and tripped the
    # assertion below. On chain the newcomer owns that uid and the evicted hotkey
    # owns nothing, so the live claim must win.
    for ident in sorted(sc, key=lambda i: i not in alive):
        if ident in taken:
            continue
        if ident not in alive:
            continue          # deregistered: on chain it owns no uid at all
        u = hk_uid.get(ident)
        if u is not None and u not in reserved:
            taken[ident] = u
            reserved.add(u)

    # Slots freed by identities that have LEFT (retired incumbents) are the only
    # ones a mid-replay registrant may take. The 7 validator uids are not free:
    # MinerSelector never queries them, so on chain they hold 0 forever, and
    # writing a real score there both distorts the vector and is then destroyed by
    # weights[0] = 0.
    validator_uids = {u for u in range(n)} - {m["uid"] for m in meta["miners"]}
    homeless = [i for i in sc if i not in taken and i in alive]
    free = [u for u in range(n)
            if u not in reserved and u not in validator_uids and vec_owner_free(u, sc, hk_uid, alive)]
    for ident in homeless:
        if not free:
            raise AssertionError(
                f"{len(homeless)} live identities have no uid slot (n={n}, "
                f"{len(reserved)} reserved). Dropping them would delete real "
                f"competitors from the denominator.")
        taken[ident] = free.pop()

    for ident, v in sc.items():
        if ident in alive and ident in taken:
            vec[taken[ident]] = v
    return vec, [taken[o] for o in our_ids if o in taken]


def vec_owner_free(u, sc, hk_uid, alive):
    """True when no live identity already claims uid u."""
    for hk, uid in hk_uid.items():
        if uid == u and hk in alive and hk in sc:
            return False
    return True


def set_weights(scores, target=0.80, max_gamma=32.0, force_gamma=None):
    """BaseValidatorNeuron.set_weights, returning normalised weights."""
    s = np.asarray(scores, dtype=float)
    lo, hi = s.min(), s.max()
    nz = (s - lo) / (hi - lo) if hi > lo else np.zeros_like(s)
    zero = nz == 0
    live = nz[~zero]
    if live.size == 0:
        return nz
    mid = np.median(live)
    steep = max(np.percentile(live, 75) - np.percentile(live, 25), 0.1)
    sig = 1.0 / (1.0 + np.exp(-(nz - mid) / steep))
    sig[zero] = 0.0

    order = np.argsort(-nz)
    top = order[: len(order) // 2]

    def transform(g):
        w = np.zeros_like(sig)
        a = sig > 0
        w[a] = sig[a] ** g
        return w

    def share(w):
        t = w.sum()
        return 0.0 if t <= 0 else w[top].sum() / t

    if force_gamma is not None:
        g_hi = float(force_gamma)
    else:
        g_lo, g_hi = 0.0, 1.0
        while share(transform(g_hi)) < target and g_hi < max_gamma:
            g_hi = min(g_hi * 2, max_gamma)
        for _ in range(80):
            m = (g_lo + g_hi) / 2
            if share(transform(m)) < target:
                g_lo = m
            else:
                g_hi = m
    w = transform(g_hi)
    w[0] = 0.0                      # set_weights forces uid 0 to zero weight
    return (w / w.sum() if w.sum() > 0 else w), g_hi, share(w)


MINER_ALPHA_DAY = 2951.6
ALPHA_TAO = 0.0106


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[1, 5, 10, 20, 40],
                    help="fleet sizes to simulate")
    ap.add_argument("--rounds", type=int, default=1000,
                    help="how many of the most recent rounds to use")
    ap.add_argument("--dataset", default=os.path.join(DATA_DIR, "sim_rounds.jsonl"),
                    help="rounds WITH per-miner answers (build with --keep-answers)")
    ap.add_argument("--metagraph", default=os.path.join(DATA_DIR, "metagraph.json"))
    ap.add_argument("--cache", default=os.path.join(DATA_DIR, "sim_cliques.jsonl"))
    ap.add_argument("--solver", default="fleet_solver:solve_many",
                    help="module:func with signature (A, time_limit, k) -> list of cliques")
    ap.add_argument("--latency-s", type=float, default=2.0,
                    help="seconds reserved for the request/response round trip. A "
                         "constant, not a fraction: the validator's timeout covers "
                         "one HTTP exchange whose cost does not scale with the "
                         "deadline.")
    ap.add_argument("--solve", action="store_true",
                    help="run the (slow) solve phase; otherwise replay the cache")
    ap.add_argument("--seed", type=int, default=8383)
    ap.add_argument("--gamma", type=float, default=None,
                    help="DIAGNOSTIC ONLY. set_weights derives gamma by binary search "
                         "until the top half takes exactly 80%%; forcing a value the "
                         "search would not have produced models a validator that does "
                         "not exist. Forcing 16.4 onto a short, noise-widened vector "
                         "gives the top half 99.9%%, not 80%%. Converge the field with "
                         "--history instead.")
    ap.add_argument("--apply-dereg", action="store_true", default=True,
                    help="track re-registrations in the data: each one evicts the worst "
                         "non-immune identity in our world and admits the new hotkey. "
                         "Ours are eligible, so this is what tells us whether we survive. "
                         "Not optional in practice - without it our simulated field "
                         "drifts away from the data, which only ever contains the "
                         "NEW hotkey's answers, never the one it replaced.")
    ap.add_argument("--no-apply-dereg", dest="apply_dereg", action="store_false")
    ap.add_argument("--sample", choices=["real", "bernoulli"], default="real",
                    help="how to decide which of our hotkeys a round queries. 'real' "
                         "reuses the query pattern the log already records for the uid "
                         "slots we take over - exact, and free of resampling variance. "
                         "'bernoulli' re-rolls it at P(difficulty); use it only to test "
                         "sensitivity to the slots chosen.")
    ap.add_argument("--immunity-hours", type=float, default=20.0,
                    help="6000 blocks x 12s. Measured in wall clock from the round "
                         "timestamps, because two validators interleave at ~105 rounds "
                         "an hour and a round count would misdate it.")
    ap.add_argument("--null-seeds", type=int, default=30,
                    help="zero-skill control runs: our hotkeys draw a reward from the "
                         "round's own survivors. A share inside the null is noise.")
    args = ap.parse_args()

    rounds = [json.loads(l) for l in open(args.dataset) if l.strip()]
    rounds = [r for r in rounds if r.get("answers")]
    # chronological, NOT (_run, _step): _step is per-validator, so sorting by it
    # concatenates one validator's whole stream after the other's. Churn and
    # immunity are wall-clock, and the two streams interleave at ~105 rounds/h.
    if all(r.get("timestamp") for r in rounds):
        rounds.sort(key=lambda r: r["timestamp"])
    else:
        raise SystemExit("dataset lacks timestamps; rebuild with build_dataset.py "
                         "so rounds can be replayed in chronological order")
    rounds = rounds[-args.rounds:]      # the real sequence, in order, not a sample
    meta = json.load(open(args.metagraph))
    kmax = max(args.sizes)
    print(f"{len(rounds)} rounds | fleet sizes {args.sizes} | metagraph block {meta['block']}",
          file=sys.stderr)
    stamps = [r["timestamp"] for r in rounds if r.get("timestamp")]
    span_h = (max(stamps) - min(stamps)) / 3600.0 if len(stamps) > 1 else 0.0
    p_mean = float(np.mean([selection_p(r["difficulty"]) for r in rounds]))
    per_uid = len(rounds) * p_mean
    print(f"  simulated span {span_h:.2f} h ({span_h/24:.2f} days) at "
          f"{len(rounds)/span_h if span_h else 0:.0f} rounds/h", file=sys.stderr)
    if span_h < args.immunity_hours:
        need = (args.immunity_hours * len(rounds) / span_h) if span_h else float("nan")
        print(f"  WARNING: the window ({span_h:.2f} h) is shorter than the "
              f"{args.immunity_hours:.0f} h immunity, so our fleet can never be "
              f"evicted and 'survived' is meaningless. Deregistration only becomes "
              f"testable past ~{need:.0f} rounds."
              + ("  (no timestamps in this dataset)" if not span_h else ""),
              file=sys.stderr)
    print(f"  ~{per_uid:.0f} samples per simulated hotkey "
          f"(the live field rests on ~595)", file=sys.stderr)
    warm_h = WARMUP_BLOCKS * BLOCK_S / 3600.0
    if span_h and span_h <= warm_h:
        print(f"  WARNING: the whole {span_h:.2f} h window is inside the "
              f"{warm_h*60:.0f} min registration warmup (MinerSelector skips a uid "
              f"while current_block - last_update <= 361), so our fleet is never "
              f"queried and every score below is 0.", file=sys.stderr)
    if per_uid < 150:
        print(f"  WARNING: {per_uid:.0f} samples per hotkey is far too few. Our fleet's "
              f"scores are then mostly noise, and because the weight curve is a rank "
              f"amplifier a lucky draw can dominate the sweep. Every figure below is "
              f"unreliable until this is in the hundreds; the null columns are there to "
              f"prove it.", file=sys.stderr)
    if args.gamma is not None:
        print(f"  WARNING: gamma forced to {args.gamma}. set_weights would have derived "
              f"it so the top half takes exactly 80%; a forced value models a validator "
              f"that cannot exist. Diagnostic only.", file=sys.stderr)

    if args.solve:
        solve_phase(rounds, kmax, args.solver, args.latency_s, args.cache)

    cache = {}
    with open(args.cache) as f:
        for line in f:
            r = json.loads(line)
            cache[r["uuid"]] = r
    missing = [r["uuid"] for r in rounds if r["uuid"] not in cache]
    if missing:
        raise SystemExit(f"{len(missing)} rounds have no cached cliques; rerun with --solve")

    pool_sizes = [len(cache[r["uuid"]]["cliques"]) for r in rounds]
    if min(pool_sizes) < kmax:
        starved = [N for N in args.sizes if N > min(pool_sizes)]
        print(f"  WARNING: the cache holds {min(pool_sizes)}-{max(pool_sizes)} cliques "
              f"per round but sizes {starved} need up to {kmax}. Hotkeys past the pool "
              f"are queried and score ZERO, which is correct behaviour but means those "
              f"rows measure a starved solver, not a fleet. Re-run --solve after "
              f"widening --sizes.", file=sys.stderr)

    print("validating our cliques against the graphs...", file=sys.stderr)
    validity = {r["uuid"]: validate_cliques(r, cache[r["uuid"]]["cliques"]) for r in rounds}
    bad = sum(1 for v in validity.values() for x in v if not x)
    print(f"  {bad} invalid cliques out of {sum(len(v) for v in validity.values())}",
          file=sys.stderr)

    print()
    print(f"{'N':>4} {'displaced':>10} {'gamma':>6} {'our best':>9} {'our worst':>10} "
          f"{'median rank':>12} {'fleet share':>12} {'null p5':>9} {'null p95':>9} "
          f"{'a/day':>8} {'survived':>9} {'dereg':>6}")
    results = []
    for N in args.sizes:
        def one(seed, null):
            streams, victim_uids, our_ids, alive, events, hk_uid = replay(
                rounds, cache, meta, N, seed, validity,
                immunity_s=args.immunity_hours * 3600.0,
                apply_dereg=args.apply_dereg, null=null, sample=args.sample)
            sc = ema_scores(streams)
            vec, ours = score_vector(sc, meta, victim_uids, our_ids, alive, hk_uid)
            survived = sum(1 for o in our_ids if o in alive)
            evicted_ours = [e for e in events if str(e["out"]).startswith("OURS-")]
            w, gamma, half = set_weights(vec, force_gamma=args.gamma)
            live = vec > 0
            order = np.argsort(-vec)
            rank = {int(u): r + 1 for r, u in enumerate(order)}
            # every identity's final score, ours and the field's alike: this is
            # what set_weights consumed, so it is the leaderboard the round
            # produced, not just our slice of it
            uid_hk = {u: h for h, u in hk_uid.items()}
            board = [{"uid": int(u), "rank": rank[int(u)], "score": float(vec[u]),
                      "weight": float(w[u]),
                      "who": ("OURS" if int(u) in ours else uid_hk.get(int(u), "?")),
                      "ours": int(u) in ours}
                     for u in np.flatnonzero(live)]
            board.sort(key=lambda r: r["rank"])
            return dict(share=float(w[ours].sum()) if ours else 0.0,
                        gamma=gamma, half=half,
                        ranks=sorted(rank[u] for u in ours) or [0],
                        scores=[float(vec[u]) for u in ours] or [0.0],
                        n_live=int(live.sum()), victims=victim_uids,
                        survived=survived, of=len(our_ids),
                        dereg_events=len(events), we_were_evicted=len(evicted_ours),
                        leaderboard=board, events=events)

        real = one(args.seed, False)
        nulls = [one(args.seed + 1000 + i, True)["share"] for i in range(args.null_seeds)]
        lo, hi = np.percentile(nulls, [5, 95]) if nulls else (0.0, 0.0)
        informative = real["share"] > hi
        results.append(dict(N=N, null_p05=float(lo), null_p95=float(hi),
                            informative=bool(informative), **real))
        ad = real["share"] * MINER_ALPHA_DAY
        print(f"{N:>4} {len(real['victims']):>10} {real['gamma']:>6.2f} "
              f"{min(real['ranks']):>9} {max(real['ranks']):>10} "
              f"{int(np.median(real['ranks'])):>12} {real['share']:>11.3%} "
              f"{lo:>9.3%} {hi:>9.3%} "
              + (f"{ad:>8.1f}" if informative else f"{'--':>8}")
              + f" {real['survived']}/{real['of']:<7} {real['dereg_events']:>6}"
              + ("" if informative else "   <- inside the null"))

    print()
    for r in results:
        s = r["scores"]
        flag = "" if r["informative"] else "   [NOT A RESULT: inside the null]"
        print(f"  N={r['N']:<3} our scores {np.mean(s):.4f} "
              f"[{min(s):.4f}, {max(s):.4f}]   top-half share {r['half']:.3f}   "
              f"ranks {r['ranks'][:6]}{'...' if len(r['ranks']) > 6 else ''}"
              f" of {r['n_live']}{flag}")
    if not any(r["informative"] for r in results):
        print("\nNo fleet size beat its zero-skill control. This run measures nothing "
              "about the solver — raise --rounds.", file=sys.stderr)
    out = os.path.join(DATA_DIR, "fleet_sim_results.json")
    json.dump(results, open(out, "w"), indent=1)

    # the field leaderboard for the largest fleet, which is the case that most
    # changes the field
    big = max(results, key=lambda r: r["N"])
    board = big["leaderboard"]
    ours = [b for b in board if b["ours"]]
    field = [b["score"] for b in board if not b["ours"]]
    print(f"\nFINAL SCORES at N={big['N']} — {len(board)} scoring identities")
    print(f"  field   min {min(field):.4f}  p25 {np.percentile(field, 25):.4f}  "
          f"median {np.median(field):.4f}  p75 {np.percentile(field, 75):.4f}  "
          f"max {max(field):.4f}")
    if ours:
        os_ = [b["score"] for b in ours]
        print(f"  ours    min {min(os_):.4f}  median {np.median(os_):.4f}  "
              f"max {max(os_):.4f}   ranks {[b['rank'] for b in ours][:8]}"
              f"{'...' if len(ours) > 8 else ''}")
    print(f"\n  {'rank':>5} {'uid':>4} {'score':>8} {'weight':>9}  who")
    for b in board[:10]:
        print(f"  {b['rank']:>5} {b['uid']:>4} {b['score']:>8.4f} {b['weight']:>9.5f}  "
              f"{'>>> OURS' if b['ours'] else b['who'][:12]}")
    print("    ...")
    for b in board[-5:]:
        print(f"  {b['rank']:>5} {b['uid']:>4} {b['score']:>8.4f} {b['weight']:>9.5f}  "
              f"{'>>> OURS' if b['ours'] else b['who'][:12]}")
    print(f"\n  full per-identity scores and every deregistration event: {out}")
    print("\n  caveats: a/day is GROSS — no registration burn, no rate limit, so it "
          "is not a\n           break-even figure. weights[0]=0 holds only while "
          "LAMBDA_WEIGHTS==1.0\n           (it is a burn target and can change). "
          "Emission is a weight-share proxy:\n           Yuma consensus across the "
          "seven validators is not modelled.")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.exit(main())
