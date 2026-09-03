"""Closed-form scoring for a board described by clique occupancies."""

import math


def score(board, difficulty):
    """Returns the mean reward of each player."""
    live = []
    total = 0
    omega = 0
    x_min = 1 << 30
    bigger = {}
    for entry in board:
        size, n_a, n_b = entry[0], entry[1], entry[2]
        mult = entry[3] if len(entry) > 3 else 1
        held = n_a + n_b
        if mult <= 0 or held <= 0:
            continue
        live.append((size, n_a, n_b, mult, held))
        total += held * mult
        omega = max(omega, size)
        x_min = min(x_min, held)
        bigger[size] = bigger.get(size, 0) + held * mult
    assert live

    run = 0
    optimality = {}
    inv_total = 1.0 / total
    for size in sorted(bigger, reverse=True):
        optimality[size] = math.exp(-(run * inv_total) / (size / float(omega)))
        run += bigger[size]
    scale = (1.0 + difficulty) / max(optimality.values())

    sum_a = sum_b = 0.0
    count_a = count_b = 0
    for size, n_a, n_b, mult, held in live:
        reward = optimality[size] * scale + x_min / float(held)
        sum_a += n_a * mult * reward
        count_a += n_a * mult
        sum_b += n_b * mult * reward
        count_b += n_b * mult
    return (sum_a / count_a if count_a else 0.0,
            sum_b / count_b if count_b else 0.0)
