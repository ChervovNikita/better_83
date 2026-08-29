"""Best response for a second player who sees occupancies but not which clique."""

import collections
import functools
import itertools
import math

import native
from scoring import score


def _compositions(total, parts):
    if parts == 1:
        yield (total,)
        return
    for head in range(total + 1):
        for rest in _compositions(total - head, parts - 1):
            yield (head,) + rest


def _tables(rows, cols):
    """Returns every contingency table with the given margins and its weight."""
    n = sum(rows)
    assert n == sum(cols), (rows, cols)
    out = []
    log_den = math.lgamma(n + 1)
    log_num = (sum(math.lgamma(r + 1) for r in rows)
               + sum(math.lgamma(c + 1) for c in cols))

    def walk(i, remaining, acc):
        if i == len(rows) - 1:
            if all(c >= 0 for c in remaining) and sum(remaining) == rows[i]:
                table = acc + [list(remaining)]
                weight = log_num - log_den - sum(
                    math.lgamma(x + 1) for row in table for x in row)
                out.append((table, math.exp(weight)))
            return
        for split in _compositions(rows[i], len(cols)):
            if any(s > c for s, c in zip(split, remaining)):
                continue
            walk(i + 1, [c - s for c, s in zip(remaining, split)],
                 acc + [list(split)])

    walk(0, list(cols), [])
    return out


def _classes(counts, assign):
    left = collections.Counter(counts)
    right = collections.Counter(assign)
    values_a, mult_a = zip(*sorted(left.items()))
    values_b, mult_b = zip(*sorted(right.items()))
    return values_a, list(mult_a), values_b, list(mult_b)


@functools.lru_cache(maxsize=1 << 17)
def _prep(counts, assign):
    values_a, mult_a, values_b, mult_b = _classes(counts, assign)
    return values_a, values_b, tuple(_tables(mult_a, mult_b))


def expected_scores(plan, difficulty):
    """Returns the expected mean of each player over the random matching."""
    fresh = collections.Counter(plan.get("fresh", []))
    sizes = [s for s in plan if s != "fresh"]
    per_class = []
    for size in sizes:
        counts, assign = plan[size]
        assert len(counts) == len(assign), (counts, assign)
        values_a, values_b, tables = _prep(tuple(counts), tuple(assign))
        per_class.append((size, values_a, values_b, tables))

    mean_a = mean_b = 0.0
    for combo in itertools.product(*[t for _s, _a, _b, t in per_class]):
        weight = 1.0
        board = []
        for (size, values_a, values_b, _t), (table, prob) in zip(per_class,
                                                                 combo):
            weight *= prob
            for i, row in enumerate(table):
                for j, cell in enumerate(row):
                    if cell:
                        board.append((size, values_a[i], values_b[j], cell))
        board += [(size, 0, count, mult)
                  for (size, count), mult in fresh.items()]
        got_a, got_b = score(board, difficulty)
        mean_a += weight * got_a
        mean_b += weight * got_b
    return mean_a, mean_b


def _multisets(total, slots):
    if slots == 0:
        if total == 0:
            yield ()
        return
    if slots == 1:
        yield (total,)
        return
    for head in range(total // slots, total + 1):
        for rest in _multisets(total - head, slots - 1):
            if not rest or head >= rest[0]:
                yield (head,) + rest


def _j_grid(slots, budget, cap):
    high = min(slots, budget)
    if cap <= 0 or high <= cap:
        return range(1, high + 1)
    picks = {1, high}
    for i in range(1, cap - 1):
        picks.add(1 + (high - 1) * i // (cap - 1))
    return sorted(picks)


def _fresh_plans(view, sizes, n_fresh, stack, cap):
    caps = [view[s]["free"] for s in sizes]
    if cap <= 0:
        for combo in _compositions(n_fresh, len(sizes)):
            if all(c <= cp for c, cp in zip(combo, caps)):
                yield [(s, stack) for s, c in zip(sizes, combo)
                       for _ in range(c)]
        return
    candidates = []
    for i in range(len(sizes)):
        combo = [0] * len(sizes)
        combo[i] = n_fresh
        candidates.append(tuple(combo))
    if len(sizes) > 1 and n_fresh:
        combo, left = [], n_fresh
        for capacity in caps:
            take = min(capacity, left)
            combo.append(take)
            left -= take
        if not left:
            candidates.append(tuple(combo))
        base, extra = divmod(n_fresh, len(sizes))
        candidates.append(tuple(base + (1 if i < extra else 0)
                                for i in range(len(sizes))))
    seen = set()
    for combo in candidates[:max(cap, 1) + len(sizes)]:
        if combo in seen:
            continue
        seen.add(combo)
        if all(c <= cp for c, cp in zip(combo, caps)):
            yield [(s, stack) for s, c in zip(sizes, combo) for _ in range(c)]


def best_response(view, difficulty, q, omega, max_fresh_stack=None):
    """Returns the expected gap and plan, searching every multiset.

    Exhaustive: no stack cap, no width grid and no shape family. This is the
    reference the fast responder is checked against, so it must not prune.
    """
    return _search(view, difficulty, q, omega, max_fresh_stack or q, None, 0, 0,
                   prune=False)


def best_response_fast(view, difficulty, q, omega, max_fresh_stack=3,
                       j_cap=0, fresh_cap=3, head_cap=8):
    """Returns the same as best_response, restricted to a two-parameter family.

    Even spreads alone cannot express a skewed assignment such as [4,1,1,1,1,1],
    which is the exhaustive optimum on real views often enough to matter, so the
    family is even-over-j plus one deepened head.
    """
    return _search(view, difficulty, q, omega, max_fresh_stack, "head", j_cap,
                   fresh_cap, head_cap=head_cap)


def _search(view, difficulty, q, omega, max_fresh_stack, shape, j_cap,
            fresh_cap, weighted=True, prune=True, head_cap=0):
    assert q > 0
    sizes = sorted(view, reverse=True)
    occupied = [c for s in sizes for c in view[s]["counts"]]
    held_a = sum(occupied)
    if prune and occupied and sum(1 for c in occupied if c <= 1) > q:
        max_fresh_stack = 1
    active = [s for s in sizes if view[s]["counts"]]

    best = None
    for active_split in _compositions(q, len(active) + 1):
        fresh_budget = active_split[-1]
        budgets = dict(zip(active, active_split[:-1]))
        split = [budgets.get(s, 0) for s in sizes]
        for stack in range(1, max_fresh_stack + 1):
            if fresh_budget % stack:
                continue
            n_fresh = fresh_budget // stack
            if n_fresh > sum(view[s]["free"] for s in sizes):
                continue
            for fresh_plan in _fresh_plans(view, sizes, n_fresh, stack,
                                           fresh_cap):
                for combo in _assignments(view, sizes, split, shape, j_cap,
                                          head_cap):
                    plan = {"fresh": fresh_plan}
                    for size, assign in zip(sizes, combo):
                        if assign is not None:
                            plan[size] = (view[size]["counts"], list(assign))
                    mean_a, mean_b = native.expected_scores(plan, difficulty,
                                                            omega)
                    if weighted:
                        objective = q * mean_b - held_a * mean_a
                    else:
                        objective = mean_b - mean_a
                    if best is None or objective > best[0] + 1e-12:
                        best = (objective, plan, mean_a, mean_b)
    assert best is not None
    return best


def _heads(budget, j, cap):
    """Extra depth to pile on one clique, on top of an even spread over j."""
    high = budget - j
    if high <= 0:
        return (0,)
    if cap <= 0 or high + 1 <= cap:
        return range(high + 1)
    return sorted({high * i // (cap - 1) for i in range(cap)})


def _assignments(view, sizes, split, shape, j_cap, head_cap=0):
    per_class = []
    for size, budget in zip(sizes, split):
        slots = len(view[size]["counts"])
        if slots == 0:
            assert budget == 0
            per_class.append([None])
            continue
        if shape is None:
            per_class.append(list(_multisets(budget, slots)))
            continue
        seen = set()
        options = []
        for j in _j_grid(slots, budget, j_cap) if budget else (0,):
            if j == 0:
                options.append([0] * slots)
                continue
            heads = _heads(budget, j, head_cap) if shape == "head" else (None,)
            for head in heads:
                if head is None:
                    base, extra = divmod(budget, j)
                    counts = [base + 1] * extra + [base] * (j - extra)
                else:
                    rest = budget - 1 - head
                    if rest < j - 1:
                        continue
                    base, extra = divmod(rest, j - 1) if j > 1 else (0, 0)
                    counts = ([1 + head] + [base + 1] * extra
                              + [base] * (j - 1 - extra))
                counts = sorted(counts + [0] * (slots - j), reverse=True)
                key = tuple(counts)
                if key in seen:
                    continue
                seen.add(key)
                options.append(counts)
        per_class.append(options)
    return itertools.product(*per_class)


def realise(plan, rng):
    """Returns a concrete board, shuffling the assignment onto the cliques."""
    board = []
    for size, value in plan.items():
        if size == "fresh":
            board += [(s, 0, count) for s, count in value]
            continue
        counts, assign = value
        shuffled = list(assign)
        rng.shuffle(shuffled)
        board += [(size, c, m) for c, m in zip(counts, shuffled)]
    return board
