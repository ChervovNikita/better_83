"""Validator-faithful reward replay: what one extra answer would have scored.

This is the single source of truth for scoring a candidate answer offline. It
reproduces CliqueScoreCalculator.get_scores() exactly, without needing the graph
or the other miners' vertex sets — only summary statistics carried in the dataset.

    reward = optimality * (1 + difficulty) + diversity

Both terms have a normaliser, and the diversity one is where offline replays
usually go wrong:

  optimality  omega_i = exp(-pr_i / rel_i), rel = size / max_size,
              pr = (count of VALID answers strictly larger) / (all responders).
              Normalised by the round max, which is always 1.0 in practice, so
              any answer tying for max size scores exactly 1.0.

  diversity   1 / (miners returning your exact vertex set), normalised by the
              best such value over ALL valid answers — with NO size term. A
              smaller unique clique therefore sets the normaliser to 1.0 and
              frequently out-earns every max-size answer in the round.

The normaliser must be computed over the *augmented* round — after our answer is
added, since joining a group changes that group's count. Normalising over
best-size cliques only inflates reward whenever we collide; normalising by
`1.0 if any_unique` is right 149 times in 150 but breaks when we duplicate the
field's unique minimum-count clique.
"""
import collections

import numpy as np


def replay_reward(our_size, our_count_before, size_hist, count_hist, difficulty,
                  n_invalid=0):
    """Reward for one extra answer inserted into a logged round.

    our_size          vertices in our clique; 0 if invalid (scores 0)
    our_count_before  how many field answers already have our exact vertex set
                      (0 = we are a new distinct clique)
    size_hist         {size: n_valid_answers} for the round
    count_hist        {duplicate_count: n_distinct_valid_cliques} for the round
    difficulty        0.7 / 0.8 / 0.9 / 1.0
    n_invalid         answers the validator scored zero (n_responders - n_valid).
                      These carry masked size 0, so they never count as "strictly
                      larger" — but they DO sit in pr's denominator, which is
                      len(responses), not the number of valid answers. Omitting
                      them shrinks the denominator and understates optimality.

    Returns (reward, optimality, diversity).
    """
    if our_size <= 0:
        return 0.0, 0.0, 0.0

    sizes = []
    for s, c in (size_hist or {}).items():
        sizes.extend([int(s)] * int(c))
    if not sizes:
        return 0.0, 0.0, 0.0
    sizes.extend([0] * int(n_invalid or 0))
    sizes.append(int(our_size))

    a = np.array(sizes, dtype=float)
    if a.max() <= 0:
        return 0.0, 0.0, 0.0
    rel = a / a.max()
    pr = np.array([(a > x).sum() / len(a) for x in a])
    omega = np.where(a > 0, np.exp(-pr / np.maximum(rel, 1e-12)), 0.0)
    optimality = float((omega / omega.max())[-1])

    # augment the duplicate-count multiset with our answer, then take the best
    # delta over the whole round
    h = collections.Counter({int(k): int(v) for k, v in (count_hist or {}).items()})
    c = int(our_count_before or 0)
    if c > 0:
        h[c] -= 1
        h[c + 1] += 1
    else:
        h[1] += 1
    live = [k for k, v in h.items() if v > 0]
    if not live:
        return 0.0, optimality, 0.0
    max_delta = 1.0 / min(live)

    our_delta = 1.0 / (c + 1)
    diversity = our_delta / max_delta
    return float(optimality * (1 + difficulty) + diversity), optimality, float(diversity)


def count_hist_from_answers(answers):
    """{duplicate_count: n_distinct_cliques} over the VALID answers of a round.

    `answers` is the dataset's per-miner detail (needs --keep-answers); an answer
    counts as valid when the validator gave it optimality > 0.
    """
    valid = [a["clique"] for a in answers if a.get("opt", 0) > 0]
    dup = collections.Counter(tuple(sorted(c)) for c in valid)
    return dict(collections.Counter(dup.values()))


def our_count_before(our_clique, best_cliques, best_clique_counts, our_size, best_size):
    """How many field answers already hold our exact vertex set.

    Only resolvable when we tie for best size, because the dataset records the
    field's vertex sets only for best-size cliques. Below best size we cannot
    detect a collision and assume none — optimistic, but such answers are already
    losing on optimality by far more than the diversity term is worth.
    """
    if our_size != best_size or not best_cliques:
        return 0
    key = tuple(sorted(our_clique))
    for cl, cnt in zip(best_cliques, best_clique_counts or []):
        if tuple(sorted(cl)) == key:
            return int(cnt)
    return 0
