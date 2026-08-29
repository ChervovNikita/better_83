"""Cross-checks the solvers against slower references."""

import itertools
import random

import optimal
import partial
import responders
import rounds as rounds_module
import strategies
from scoring import score


def check_scoring(trials=400, seed=0):
    """Returns the largest disagreement with fleet_sim.score_round."""
    import os
    import sys
    import numpy as np
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(os.path.dirname(here)))
    research = os.path.join(root, "research")
    if research not in sys.path:
        sys.path.insert(0, research)
    from fleet_sim import score_round

    rng = random.Random(seed)
    worst = 0.0
    for _ in range(trials):
        difficulty = rng.choice([0.7, 0.8, 0.9, 1.0])
        omega = rng.randint(3, 40)
        board = [(omega - rng.randint(0, 2), rng.randint(0, 4),
                  rng.randint(0, 4)) for _ in range(rng.randint(1, 8))]
        if not any(a + b for _s, a, b in board):
            continue
        mean_a, mean_b = score(board, difficulty)
        sizes, keys, who = [], [], []
        for i, (size, n_a, n_b) in enumerate(board):
            sizes += [size] * (n_a + n_b)
            keys += [i] * (n_a + n_b)
            who += [0] * n_a + [1] * n_b
        got = np.asarray(score_round(sizes, [1] * len(sizes), keys, difficulty))
        mask = np.array(who)
        want_a = float(got[mask == 0].mean()) if (mask == 0).any() else 0.0
        want_b = float(got[mask == 1].mean()) if (mask == 1).any() else 0.0
        worst = max(worst, abs(mean_a - want_a), abs(mean_b - want_b))
    return worst


def check_optimal(trials=200, seed=11, wide=False):
    """Returns how often the full responder falls below brute force."""
    rng = random.Random(seed)
    bad = total = 0
    worst = 0.0
    for _ in range(trials):
        rnd = _fake_round(rng, wide)
        board = strategies.REGISTRY["greedy"](rnd, rnd.q_a, score)
        _trial, mean_a, mean_b = optimal.best_response(board, rnd, rnd.q_b)
        reference = _brute_full(board, rnd, rnd.q_b)
        total += 1
        if reference - (mean_b - mean_a) > 1e-9:
            bad += 1
            worst = max(worst, reference - (mean_b - mean_a))
    return bad, total, worst


def check_partial(trials=300, seed=555):
    """Returns how often the fast partial search falls below the exhaustive one."""
    rng = random.Random(seed)
    bad = total = 0
    worst = 0.0
    for _ in range(trials):
        omega = rng.choice([20, 30, 38, 41])
        view = {omega: {"counts": sorted(rng.randint(1, 5)
                                         for _ in range(rng.randint(1, 7))),
                        "free": rng.randint(0, 6)}}
        if rng.random() < 0.7:
            view[omega - 1] = {"counts": sorted(rng.randint(1, 4)
                                                for _ in range(rng.randint(1, 5))),
                               "free": rng.randint(0, 6)}
        difficulty = rng.choice([0.7, 0.8, 0.9, 1.0])
        q = rng.randint(1, 6)
        fast = partial.best_response_fast(view, difficulty, q, omega)[0]
        exact = partial.best_response(view, difficulty, q, omega)[0]
        total += 1
        if exact - fast > 1e-9:
            bad += 1
            worst = max(worst, exact - fast)
    return bad, total, worst


def _fake_round(rng, wide=False):
    rnd = rounds_module.Round()
    rnd.uuid = "test"
    rnd.difficulty = rng.choice([0.7, 0.8, 1.0])
    rnd.omega = 30
    rnd.n_top = rng.randint(1, 6 if wide else 5)
    rnd.n_spare = rng.randint(0, 4)
    rnd.n_answers = 0
    rnd.q_a = rng.randint(2, 12 if wide else 7)
    rnd.q_b = rng.randint(4, 6) if wide else rng.randint(1, 3)
    return rnd


def _brute_full(board, rnd, q):
    slots = [("occupied", i) for i in range(len(board))]
    used_top = sum(1 for s, a, b in board if s == rnd.omega and a + b > 0)
    slots += [("fresh_top", j)
              for j in range(min(max(0, rnd.n_top - used_top), q))]
    used_spare = sum(1 for s, a, b in board if s == rnd.omega - 1 and a + b > 0)
    slots += [("fresh_spare", j)
              for j in range(min(max(0, rnd.n_spare - used_spare), q))]
    best = None
    for combo in itertools.combinations_with_replacement(range(len(slots)), q):
        trial = [list(entry) for entry in board]
        extra = {}
        for i in combo:
            kind, j = slots[i]
            if kind == "occupied":
                trial[j][2] += 1
            else:
                extra[(kind, j)] = extra.get((kind, j), 0) + 1
        trial = [tuple(entry) for entry in trial]
        for (kind, _j), count in extra.items():
            size = rnd.omega if kind == "fresh_top" else rnd.omega - 1
            trial.append((size, 0, count))
        mean_a, mean_b = score(trial, rnd.difficulty)
        if best is None or mean_b - mean_a > best + 1e-12:
            best = mean_b - mean_a
    return best


def main():
    """Runs every check and prints the results."""
    print("scoring vs fleet_sim        : max disagreement %.3g"
          % check_scoring())
    bad, total, worst = check_optimal()
    print("full responder vs brute     : %d/%d below, worst %.9f"
          % (bad, total, worst))
    bad, total, worst = check_optimal(60, 23, True)
    print("full responder, wide boards : %d/%d below, worst %.9f"
          % (bad, total, worst))
    bad, total, worst = check_partial()
    print("fast partial vs exhaustive  : %d/%d below, worst %.9f"
          % (bad, total, worst))
    import native
    bad, total, worst, _f = check_optimal_grid(solver=native.best_response)
    print("C++ vs exhaustive grid      : %d/%d below, worst %.9f"
          % (bad, total, worst))
    bad, total, worst = check_prior_occupancy(solver=native.best_response)
    print("C++ on boards with prior B  : %d/%d below, worst %.9f"
          % (bad, total, worst))
    return 0




def check_optimal_grid(n_top_max=4, n_spare_max=3, q_a_max=8, q_b_max=5,
                       difficulties=(0.7, 1.0), solver=None):
    """Compares the full responder to exhaustive search over a covered grid."""
    bad = total = 0
    worst = 0.0
    failures = []
    for difficulty in difficulties:
        for n_top in range(1, n_top_max + 1):
            for n_spare in range(0, n_spare_max + 1):
                for q_a in range(2, q_a_max + 1):
                    for q_b in range(1, q_b_max + 1):
                        rnd = rounds_module.Round()
                        rnd.uuid = "grid"
                        rnd.difficulty = difficulty
                        rnd.omega = 30
                        rnd.n_top = n_top
                        rnd.n_spare = n_spare
                        rnd.n_answers = 0
                        rnd.q_a = q_a
                        rnd.q_b = q_b
                        board = strategies.REGISTRY["greedy"](rnd, q_a, score)
                        fn = solver or optimal.best_response
                        _t, mean_a, mean_b = fn(board, rnd, q_b)
                        reference = _brute_full(board, rnd, q_b)
                        total += 1
                        if reference - (mean_b - mean_a) > 1e-9:
                            bad += 1
                            worst = max(worst, reference - (mean_b - mean_a))
                            if len(failures) < 5:
                                failures.append(
                                    (difficulty, n_top, n_spare, q_a, q_b,
                                     mean_b - mean_a, reference))
    return bad, total, worst, failures


def check_prior_occupancy(trials=400, seed=77, solver=None):
    """Compares against exhaustive search on boards that already carry B hotkeys."""
    rng = random.Random(seed)
    fn = solver or optimal.best_response
    bad = total = 0
    worst = 0.0
    for _ in range(trials):
        rnd = rounds_module.Round()
        rnd.uuid = "prior"
        rnd.difficulty = rng.choice([0.7, 0.8, 1.0])
        rnd.omega = 30
        rnd.n_top = rng.randint(1, 5)
        rnd.n_spare = rng.randint(0, 4)
        rnd.n_answers = 0
        rnd.q_a = 0
        rnd.q_b = rng.randint(1, 3)
        board = [(30, rng.randint(0, 3), rng.randint(0, 3))
                 for _ in range(rng.randint(1, 4))]
        if rng.random() < 0.4:
            board.append((29, rng.randint(0, 2), rng.randint(0, 2)))
        board = [b for b in board if b[1] + b[2] > 0]
        if not board or not any(b[1] for b in board):
            continue
        _t, mean_a, mean_b = fn(board, rnd, rnd.q_b)
        reference = _brute_full(board, rnd, rnd.q_b)
        total += 1
        if reference - (mean_b - mean_a) > 1e-9:
            bad += 1
            worst = max(worst, reference - (mean_b - mean_a))
    return bad, total, worst


if __name__ == "__main__":
    raise SystemExit(main())
