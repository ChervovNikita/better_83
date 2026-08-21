#!/usr/bin/env python3
"""Break-even miner count, derived — not regressed off what other operators earn.

Everything here is either a live chain value or the repo's own code:

  * scoring        CliqueAI/scoring/clique_scoring.py, replayed per round
  * weights        common/base/validator.py set_weights, reimplemented exactly:
                   min-max normalise -> sigmoid(midpoint=median, steepness=max(IQR,0.1))
                   -> ^gamma, gamma bisected so the top half takes 80%, capped at 32
  * chain          recycle cost, tempo, immunity period, emission per UID (live)
  * collisions     measured from 5000 real rounds with per-UID answers

Our N miners are modelled at the UPPER BOUND the user asked for: optimality 1.0 on
every round (we always tie the best size — our shipped solver does this 99.8% of the
time), coordinated so they never self-collide, and taking distinct cliques from the
pool our solver actually reaches. Cross-collision with the field is then whatever the
real data says it is, not an assumption.
"""
import collections
import itertools
import json
import os
import sys

import numpy as np

from _common import DATA_DIR

SRC = os.path.join(DATA_DIR, "identities.jsonl")

# ---- live chain values (subnet 83, finney) --------------------------------
RECYCLE_TAO = 0.142934258        # cost to register one UID
ALPHA_PRICE = 0.010578190        # TAO per alpha
TEMPO_BLOCKS = 360
BLOCK_SEC = 12
TEMPOS_PER_DAY = 86400 / (TEMPO_BLOCKS * BLOCK_SEC)     # 20
IMMUNITY_BLOCKS = 6000
REGS_PER_INTERVAL = 1            # target_regs_per_interval, interval = 360 blocks
MINER_EMISSION_PER_TEMPO = 147.4713      # alpha, summed over incentive>0 UIDs
N_UIDS = 256
DIFF = 0.85


def validator_weights(scores):
    """set_weights from common/base/validator.py, verbatim in behaviour."""
    scores = np.asarray(scores, dtype=float)
    mn, mx = scores.min(), scores.max()
    rng = mx - mn
    normalized = np.zeros_like(scores) if rng == 0 else (scores - mn) / rng
    zero = normalized == 0
    nz = normalized[~zero]
    if nz.size == 0:
        return normalized
    midpoint = np.median(nz)
    steepness = max(np.percentile(nz, 75) - np.percentile(nz, 25), 0.1)
    sig = 1.0 / (1.0 + np.exp(-(normalized - midpoint) / steepness))
    sig[zero] = 0.0

    def top_half(w):
        order = np.argsort(-normalized)
        return w[order[: len(order) // 2]].sum() / max(w.sum(), 1e-12)

    lo, hi = 0.0, 1.0
    active = sig > 0
    def tr(g):
        w = np.zeros_like(sig)
        w[active] = np.power(sig[active], g)
        return w
    while top_half(tr(hi)) < 0.80 and hi < 32.0:
        hi = min(hi * 2, 32.0)
    for _ in range(80):
        mid = (lo + hi) / 2
        if top_half(tr(mid)) < 0.80:
            lo = mid
        else:
            hi = mid
    w = tr(hi)
    return w / max(w.sum(), 1e-12)


def score_round(answers, n_resp, extra_cliques):
    """Exact CliqueScoreCalculator replay, with our extra miners inserted."""
    all_ans = [a for a in answers] + list(extra_cliques)
    sizes = np.array([len(a) for a in all_ans], dtype=float)
    if sizes.size == 0 or sizes.max() <= 0:
        return None
    mx = sizes.max()
    rel = sizes / mx
    denom = n_resp + len(extra_cliques)
    pr = np.array([(sizes > s).sum() / denom for s in sizes])
    omega = np.exp(-pr / np.maximum(rel, 1e-9))
    optim = omega / omega.max()
    cnt = collections.Counter(all_ans)
    delta = np.array([1.0 / cnt[a] for a in all_ans])
    div = delta / delta.max()
    return optim * (1 + DIFF) + div


def main():
    rounds = [json.loads(l) for l in open(SRC) if l.strip()]
    print(f"derivation over {len(rounds)} real rounds with per-UID answers\n")

    Ns = [1, 2, 3, 5, 10, 20, 30, 50, 75, 100]
    out = {}
    for N in Ns:
        our_scores = [[] for _ in range(N)]
        field_scores = collections.defaultdict(list)
        for r in rounds:
            ans = [tuple(a) for a in r["answers"] if a]
            if not ans:
                continue
            best = max(len(a) for a in ans)
            atbest = [a for a in ans if len(a) == best]
            pool = list(dict.fromkeys(atbest))       # distinct optima the field reached
            if not pool:
                continue
            # our N miners: coordinated, never self-collide, optimality 1.0 (max size).
            # They take distinct cliques from the reachable pool, least-crowded first —
            # if N exceeds the pool, the surplus must reuse cliques, which is the
            # self-collision the coordinator cannot avoid.
            # ORACLE vs ACHIEVABLE. Ordering by true crowding uses information no
            # coordinator has at inference time — we proved field popularity is
            # unpredictable (every structural feature ~0.00 correlation). The
            # achievable rule is: take DISTINCT cliques from the pool, chosen
            # arbitrarily. Set SN83_ORACLE=1 to see the upper bound instead.
            if os.environ.get("SN83_ORACLE") == "1":
                crowd = collections.Counter(atbest)
                ordered = sorted(pool, key=lambda c: crowd[c])
            else:
                ordered = list(pool)
                rs = np.random.default_rng(hash(r["uuid"]) & 0xFFFFFFFF)
                rs.shuffle(ordered)
            mine = [ordered[i % len(ordered)] for i in range(N)]
            sc = score_round(ans, len(r["uids"]), mine)
            if sc is None:
                continue
            for i in range(N):
                our_scores[i].append(sc[len(ans) + i])
            for u, s in zip([u for u, a in zip(r["uids"], r["answers"]) if a],
                            sc[: len(ans)]):
                field_scores[u].append(s)

        our_mean = np.array([np.mean(s) for s in our_scores if s])
        field_mean = np.array([np.mean(v) for v in field_scores.values() if len(v) >= 50])

        # THE SUBNET IS FULL AT 256 UIDS. Registering N does not add N to the field —
        # it DISPLACES the N lowest-incentive incumbents. What survives is the
        # strongest part of the field, so our miners land at the bottom of a tougher
        # distribution than the one they replaced.
        keep = np.sort(field_mean)[::-1][: max(len(field_mean) - N, 0)]
        allsc = np.concatenate([keep, our_mean])
        w = validator_weights(allsc)
        our_w = w[len(keep):].sum()
        per_uid_alpha = our_w * MINER_EMISSION_PER_TEMPO * TEMPOS_PER_DAY / max(N, 1)
        total_alpha = our_w * MINER_EMISSION_PER_TEMPO * TEMPOS_PER_DAY
        total_tao = total_alpha * ALPHA_PRICE
        cost = N * RECYCLE_TAO
        # our miners occupy a BLOCK of ranks, not one position
        order = np.argsort(-allsc)
        ours_idx = set(range(len(keep), len(allsc)))
        our_ranks = [i + 1 for i, j in enumerate(order) if j in ours_idx]
        best_rank, worst_rank = min(our_ranks), max(our_ranks)
        n_total = len(allsc)
        prune_zone = n_total - 20            # bottom 20 are displaced each day
        in_danger = sum(1 for r in our_ranks if r > prune_zone)
        out[N] = (our_mean.mean(), our_w, total_alpha, total_tao, cost,
                  cost / total_tao if total_tao > 0 else float("inf"),
                  best_rank, worst_rank, in_danger, n_total)

    print(f"{'N':>4} {'reward/UID':>11} {'weight':>9} {'TAO/day':>9} {'reg cost':>9} "
          f"{'payback':>9} {'our ranks':>13} {'in prune zone':>14}")
    for N in Ns:
        s, w, a, t, c, days, br, wr, dang, ntot = out[N]
        pay = f"{days:>8.2f}d" if days < 1e4 else "    never"
        print(f"{N:>4} {s:>11.4f} {w:>9.5f} {t:>9.4f} τ{c:>8.4f} {pay} "
              f"{br:>5}-{wr:<7} {dang:>6}/{N:<7}")

    print(f"\nchain: recycle τ{RECYCLE_TAO}, alpha τ{ALPHA_PRICE}, tempo {TEMPO_BLOCKS} "
          f"blocks, immunity {IMMUNITY_BLOCKS} blocks "
          f"({IMMUNITY_BLOCKS*BLOCK_SEC/3600:.1f} h)")
    print(f"miner emission {MINER_EMISSION_PER_TEMPO} alpha/tempo = "
          f"{MINER_EMISSION_PER_TEMPO*TEMPOS_PER_DAY:.0f} alpha/day = "
          f"τ{MINER_EMISSION_PER_TEMPO*TEMPOS_PER_DAY*ALPHA_PRICE:.2f}/day over 244 miners")
    regs_per_day = REGS_PER_INTERVAL * TEMPOS_PER_DAY
    print(f"\nPRUNING: {regs_per_day:.0f} registrations/day displace the "
          f"{regs_per_day:.0f} lowest-incentive UIDs.")
    print(f"  immunity protects a new UID for {IMMUNITY_BLOCKS*BLOCK_SEC/3600:.1f} h; "
          f"after that it must stay out of the bottom {regs_per_day:.0f} of {N_UIDS}.")


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
