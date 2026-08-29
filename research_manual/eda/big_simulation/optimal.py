"""Exact best response for a second player who sees the whole board."""

from scoring import score

NEG = -1e18


def best_response(board, rnd, q, max_xmin=0, anchors=-1):
    """Returns the board and both means after the second player answers."""
    assert q >= 0
    if q == 0:
        mean_a, mean_b = score(board, rnd.difficulty)
        return list(board), mean_a, mean_b

    sizes = (rnd.omega, rnd.omega - 1)
    supply = {rnd.omega: rnd.n_top, rnd.omega - 1: rnd.n_spare}
    occupied = {s: [i for i, e in enumerate(board) if e[0] == s] for s in sizes}
    held_a = sum(e[1] for e in board)
    held_b = sum(e[2] for e in board)
    n_cliques = sum(1 for e in board if e[1] + e[2] > 0)
    if not max_xmin:
        max_xmin = (held_a + held_b + q) // max(1, n_cliques) + 1
    free = {s: max(0, supply[s] - len(occupied[s])) for s in sizes}

    best = None
    for k_top in range(0, q + 1):
        budget = {sizes[0]: k_top, sizes[1]: q - k_top}
        for xmin in range(1, max_xmin + 1):
            for anchor in _anchors(board, anchors):
                plan = {}
                for size in sizes:
                    plan[size] = _allocate(board, occupied[size], free[size],
                                           budget[size], xmin, held_b + q,
                                           held_a, anchor)
                    if plan[size] is None:
                        break
                if (any(plan.get(s) is None for s in sizes)
                        or len(plan) < len(sizes)):
                    continue
                trial = _build(board, occupied, plan, sizes)
                mean_a, mean_b = score(trial, rnd.difficulty)
                if best is None or (mean_b - mean_a) > best[0] + 1e-12:
                    best = (mean_b - mean_a, trial, mean_a, mean_b)
    assert best is not None
    return best[1], best[2], best[3]


def _anchors(board, cap):
    """Which clique is pinned to exactly xmin, shallowest first."""
    if cap == 0:
        return (None,)
    order = sorted(range(len(board)), key=lambda i: board[i][1] + board[i][2])
    if cap < 0:
        return [None] + order
    return [None] + order[:cap]


def _allocate(board, occupied, free, budget, xmin, q, held_a, anchor=None):
    """Distributes `budget` over one size class, every clique reaching xmin."""
    items = []
    for i in occupied:
        low = max(0, xmin - board[i][1] - board[i][2])
        items.append(("occupied", i, low, low if i == anchor else budget))
    items += [("fresh", None, xmin, budget)] * min(free, budget // xmin)
    if not items:
        return ([], []) if budget == 0 else None
    if sum(low for kind, _i, low, _hi in items if kind == "occupied") > budget:
        return None

    table = [[NEG] * (budget + 1) for _ in range(len(items) + 1)]
    table[0][0] = 0.0
    back = [[0] * (budget + 1) for _ in range(len(items) + 1)]
    for step, (kind, i, low, high) in enumerate(items):
        allowed = [0] if kind == "fresh" else []
        allowed += list(range(max(low, 1 if kind == "fresh" else low),
                              budget + 1))
        for spent in range(budget + 1):
            best_value, best_take = NEG, 0
            for take in allowed:
                if take > high:
                    continue
                if take > spent or (kind == "fresh" and 0 < take < xmin):
                    continue
                prev = table[step][spent - take]
                if prev <= NEG / 2:
                    continue
                value = prev + _gain(board, kind, i, take, xmin, q, held_a)
                if value > best_value:
                    best_value, best_take = value, take
            table[step + 1][spent] = best_value
            back[step + 1][spent] = best_take
    if table[len(items)][budget] <= NEG / 2:
        return None

    takes = [0] * len(items)
    spent = budget
    for step in range(len(items), 0, -1):
        takes[step - 1] = back[step][spent]
        spent -= back[step][spent]
    return takes[:len(occupied)], [t for t in takes[len(occupied):] if t > 0]


def _gain(board, kind, i, take, xmin, q, held_a):
    """Objective contribution of putting `take` hotkeys on one clique."""
    if kind == "fresh":
        return xmin / float(q) if take else 0.0
    held = board[i][1] + board[i][2]
    total = held + take
    return (xmin * take / float(total) / float(q)
            - xmin * board[i][1] / float(total) / float(held_a))


def _build(board, occupied, plan, sizes):
    """Applies an allocation to the board."""
    add = {}
    extra = []
    for size in sizes:
        allocation, fresh = plan[size]
        for i, take in zip(occupied[size], allocation):
            add[i] = take
        extra += [(size, 0, depth) for depth in fresh]
    out = [(s, a, b + add.get(i, 0)) for i, (s, a, b) in enumerate(board)]
    return out + extra
