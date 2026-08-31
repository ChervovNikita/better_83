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
        # randint(1, 7) cliques means the omega class is NEVER empty, so the
        # region where the fast family is known to fall short -- no occupied
        # maximum clique but free capacity to open one -- was unsampled.
        n_top_occ = rng.randint(0, 7)
        view = {omega: {"counts": sorted(rng.randint(1, 5)
                                         for _ in range(n_top_occ)),
                        "free": rng.randint(0, 6)}}
        if rng.random() < 0.7:
            view[omega - 1] = {"counts": sorted(rng.randint(1, 4)
                                                for _ in range(rng.randint(1, 5))),
                               "free": rng.randint(0, 6)}
        difficulty = rng.choice([0.7, 0.8, 0.9, 1.0])
        q = rng.randint(1, 6)
        if not any(view[s_]["counts"] for s_ in view):
            continue
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


def check_weighted(trials=200, seed=5):
    """Compares the weighted best response to exhaustive search on w_b*B - w_a*A."""
    rng = random.Random(seed)
    import native
    bad = total = 0
    worst = 0.0
    for _ in range(trials):
        rnd = _fake_round(rng)
        w_a = rng.choice([0.25, 0.5, 1.0, 2.0])
        w_b = rng.choice([0.25, 0.5, 1.0, 2.0])
        board = strategies.REGISTRY["greedy"](rnd, rnd.q_a, score)
        _t, mean_a, mean_b = native.best_response_weighted(board, rnd, rnd.q_b,
                                                           w_a, w_b)
        got = w_b * mean_b - w_a * mean_a
        best = None
        for trial in _all_replies(board, rnd, rnd.q_b):
            x, y = score(trial, rnd.difficulty)
            v = w_b * y - w_a * x
            if best is None or v > best:
                best = v
        total += 1
        if best - got > 1e-9:
            bad += 1
            worst = max(worst, best - got)
    return bad, total, worst


def _all_replies(board, rnd, q):
    """Every placement of q hotkeys, as complete boards."""
    slots = [("occupied", i) for i in range(len(board))]
    used_top = sum(1 for s, a, b in board if s == rnd.omega and a + b > 0)
    slots += [("fresh_top", j)
              for j in range(min(max(0, rnd.n_top - used_top), q))]
    used_spare = sum(1 for s, a, b in board if s == rnd.omega - 1 and a + b > 0)
    slots += [("fresh_spare", j)
              for j in range(min(max(0, rnd.n_spare - used_spare), q))]
    for combo in itertools.combinations_with_replacement(range(len(slots)), q):
        trial = [list(e) for e in board]
        extra = {}
        for i in combo:
            kind, j = slots[i]
            if kind == "occupied":
                trial[j][2] += 1
            else:
                extra[(kind, j)] = extra.get((kind, j), 0) + 1
        trial = [tuple(e) for e in trial]
        for (kind, _j), count in extra.items():
            size = rnd.omega if kind == "fresh_top" else rnd.omega - 1
            trial.append((size, 0, count))
        yield trial


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
    bad, total, worst = check_weighted()
    print("weighted best response      : %d/%d below, worst %.9f"
          % (bad, total, worst))
    check_maximin()
    check_bayes()
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




def _partitions(total, slots):
    """Every multiset of positive counts summing to total, at most slots parts."""
    if total == 0:
        yield ()
        return
    if slots == 0:
        return
    for head in range(1, total + 1):
        for rest in _partitions(total - head, slots - 1):
            if not rest or head >= rest[0]:
                yield (head,) + rest


def _all_a_boards(rnd, q):
    for at_omega in range(q + 1):
        rest = q - at_omega
        for top in _partitions(at_omega, rnd.n_top):
            for spare in _partitions(rest, rnd.n_spare):
                board = ([(rnd.omega, c, 0) for c in top]
                         + [(rnd.omega - 1, c, 0) for c in spare])
                if board:
                    yield board


def check_maximin(trials=400, seed=17):
    """Compares the maximin solver against every A board on small instances."""
    import native
    import strategies
    rng = random.Random(seed)
    worst = worst_start = 0.0
    below = below_start = 0
    for _ in range(trials):
        rnd = _fake_round(rng)
        rnd.n_top = rng.randint(1, 4)
        rnd.n_spare = rng.randint(0, 4)
        q_a = rng.randint(1, 7)
        q_b = rng.randint(1, 6)
        rnd.fleet_a = rng.randint(q_a, 30)
        rnd.fleet_b = rng.randint(q_b, 30)
        w_a, w_b = q_a / float(rnd.fleet_a), q_b / float(rnd.fleet_b)

        def value(board):
            # B optimises the pooled objective, matching what maximin assumes.
            _t, ma, mb = native.best_response_weighted(board, rnd, q_b, w_a, w_b)
            return w_a * ma - w_b * mb

        best = max(value(b) for b in _all_a_boards(rnd, q_a))
        gap = best - value(native.maximin(rnd, q_a, q_b))
        if gap > 1e-9:
            below += 1
            worst = max(worst, gap)
        start = best - max(value(b) for b in strategies.a_candidates(rnd, q_a))
        if start > 1e-9:
            below_start += 1
            worst_start = max(worst_start, start)
    print("maximin vs every A board    : %d/%d below, worst %.9f"
          % (below, trials, worst))
    print("  (family start, no climb)  : %d/%d below, worst %.9f"
          % (below_start, trials, worst_start))


def check_bayes(trials=150, seed=29):
    """Compares the bayes solver against every A board under its own objective."""
    import native
    import strategies
    rng = random.Random(seed)
    worst = 0.0
    below = 0
    for _ in range(trials):
        rnd = _fake_round(rng)
        rnd.n_top = rng.randint(1, 4)
        rnd.n_spare = rng.randint(0, 4)
        q_a = rng.randint(1, 6)
        rnd.fleet_a = rng.randint(q_a, 30)
        rnd.fleet_b = rng.randint(4, 30)
        posterior = strategies._posterior(rnd)

        def value(board):
            total = 0.0
            for k, w in posterior:
                wa, wb = q_a / rnd.fleet_a, k / rnd.fleet_b
                _t, ma, mb = native.best_response_weighted(board, rnd, k, wa, wb)
                total += w * (wa * ma - wb * mb)
            return total

        best = max(value(b) for b in _all_a_boards(rnd, q_a))
        gap = best - value(native.bayes(rnd, q_a, posterior))
        if gap > 1e-9:
            below += 1
            worst = max(worst, gap)
    print("bayes vs every A board      : %d/%d below, worst %.9f"
          % (below, trials, worst))


if __name__ == "__main__":
    raise SystemExit(main())
