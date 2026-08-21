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

def solve_phase(rounds, k, solver, time_scale, cache_path):
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
            cl = solve_many(A, rec["time_limit"] * time_scale, k)
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

def replay(rounds, cache, meta, N, seed, validity):
    """Rebuild every round with N of our hotkeys present and the victims gone."""
    rng = np.random.default_rng(seed)
    victims = set(pick_victims(meta, N))
    our_uids = [-(i + 1) for i in range(N)]            # negative ids = ours
    streams = collections.defaultdict(list)

    for rec in rounds:
        d = rec["difficulty"]
        survivors = [a for a in rec["answers"] if a["uid"] not in victims]
        sizes = [len(a["clique"]) for a in survivors]
        valid = [1 if a["opt"] > 0 else 0 for a in survivors]
        keys = [tuple(sorted(a["clique"])) for a in survivors]
        uids = [a["uid"] for a in survivors]

        # which of our hotkeys the validator would have queried this round
        p = selection_p(d)
        picked = [u for u in our_uids if rng.random() < p]
        pool = cache.get(rec["uuid"], {}).get("cliques", [])
        ok = validity.get(rec["uuid"], [])
        for j, u in enumerate(picked):
            if j >= len(pool):
                break                                   # solver had fewer answers
            c = pool[j]
            sizes.append(len(c))
            valid.append(1 if (j < len(ok) and ok[j]) else 0)
            keys.append(tuple(c))
            uids.append(u)

        rewards = score_round(sizes, valid, keys, d)
        for u, r in zip(uids, rewards):
            streams[u].append(float(r))
    return streams, victims


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
    return (w / w.sum() if w.sum() > 0 else w), g_hi


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
    ap.add_argument("--time-scale", type=float, default=0.88)
    ap.add_argument("--solve", action="store_true",
                    help="run the (slow) solve phase; otherwise replay the cache")
    ap.add_argument("--seed", type=int, default=8383)
    ap.add_argument("--gamma", type=float, default=None,
                    help="force the weight exponent instead of deriving it from this "
                         "sample. The live validators sit at ~16.4 after thousands of "
                         "rounds; a short simulation derives a much smaller gamma and "
                         "will overstate what a below-median fleet earns.")
    args = ap.parse_args()

    rounds = [json.loads(l) for l in open(args.dataset) if l.strip()]
    rounds = [r for r in rounds if r.get("answers")]
    rounds.sort(key=lambda r: (r.get("_run", ""), r.get("_step", 0)))
    rounds = rounds[-args.rounds:]                      # latest consecutive block
    meta = json.load(open(args.metagraph))
    kmax = max(args.sizes)
    print(f"{len(rounds)} rounds | fleet sizes {args.sizes} | metagraph block {meta['block']}",
          file=sys.stderr)
    per_uid = len(rounds) * float(np.mean([selection_p(r["difficulty"]) for r in rounds]))
    print(f"  ~{per_uid:.0f} samples per simulated hotkey "
          f"(the live field's scores rest on ~595)", file=sys.stderr)
    if len(rounds) < 500 and args.gamma is None:
        print("  WARNING: under 500 rounds the field's score vector is noise-widened, "
              "so the derived gamma comes out far below the live ~16.4 and every "
              "alpha/day figure below is optimistic. Pass --gamma 16.4, or use more "
              "rounds.", file=sys.stderr)

    if args.solve:
        solve_phase(rounds, kmax, args.solver, args.time_scale, args.cache)

    cache = {}
    with open(args.cache) as f:
        for line in f:
            r = json.loads(line)
            cache[r["uuid"]] = r
    missing = [r["uuid"] for r in rounds if r["uuid"] not in cache]
    if missing:
        raise SystemExit(f"{len(missing)} rounds have no cached cliques; rerun with --solve")

    print("validating our cliques against the graphs...", file=sys.stderr)
    validity = {r["uuid"]: validate_cliques(r, cache[r["uuid"]]["cliques"]) for r in rounds}
    bad = sum(1 for v in validity.values() for x in v if not x)
    print(f"  {bad} invalid cliques out of {sum(len(v) for v in validity.values())}",
          file=sys.stderr)

    print()
    print(f"{'N':>4} {'displaced':>10} {'gamma':>6} {'our best':>9} {'our worst':>10} "
          f"{'median rank':>12} {'fleet share':>12} {'a/day':>8} {'USD/day':>9}")
    results = []
    for N in args.sizes:
        streams, victims = replay(rounds, cache, meta, N, args.seed, validity)
        sc = ema_scores(streams)
        uids = sorted(sc)
        vec = np.array([sc[u] for u in uids])
        w, gamma = set_weights(vec, force_gamma=args.gamma)
        ours = [i for i, u in enumerate(uids) if u < 0]
        rank = {i: r + 1 for r, i in enumerate(np.argsort(-vec))}
        our_ranks = sorted(rank[i] for i in ours)
        share = float(w[ours].sum())
        results.append(dict(N=N, gamma=gamma, share=share, ranks=our_ranks,
                            scores=[float(vec[i]) for i in ours],
                            n_field=len(uids) - N))
        print(f"{N:>4} {len(victims):>10} {gamma:>6.2f} {min(our_ranks):>9} "
              f"{max(our_ranks):>10} {int(np.median(our_ranks)):>12} "
              f"{share:>11.3%} {share*MINER_ALPHA_DAY:>8.1f} "
              f"{share*MINER_ALPHA_DAY*ALPHA_TAO*190:>9.0f}")

    print()
    for r in results:
        s = r["scores"]
        print(f"  N={r['N']:<3} our scores {np.mean(s):.4f} "
              f"[{min(s):.4f}, {max(s):.4f}]   ranks {r['ranks'][:6]}"
              f"{'...' if len(r['ranks']) > 6 else ''} of {r['n_field']+r['N']}")
    json.dump(results, open(os.path.join(DATA_DIR, "fleet_sim_results.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.exit(main())
