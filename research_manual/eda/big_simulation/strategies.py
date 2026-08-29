"""Strategies for the first player, who commits before seeing the reply."""

import collections
import math

REF_R = 1.5
METAGRAPH = 249

REGISTRY = {}


def selection_p(difficulty):
    """The validator's own per-hotkey query probability."""
    return 1.0 - math.exp(-max(0.0, math.sqrt(1.0 + REF_R) - difficulty - 0.5))


def estimate_q_b(rnd):
    """A's estimate of how many answers the opponent submits.

    Uses only what A can see at submit time: its own queried count, and the
    difficulty, which the vertex count determines exactly. The validator queries
    p(difficulty) * M hotkeys in total and q_a of them are A's.
    """
    total = int(round(selection_p(rnd.difficulty) * METAGRAPH))
    return max(1, total - rnd.q_a)


def strategy(name):
    """Registers a first-player strategy under `name`."""
    def wrap(fn):
        REGISTRY[name] = fn
        return fn
    return wrap


def _board(omega, top_counts, spare_counts):
    board = [(omega, c, 0) for c in top_counts if c > 0]
    board += [(omega - 1, c, 0) for c in spare_counts if c > 0]
    return board


def _spread(total, slots):
    base, extra = divmod(total, slots)
    return [base + 1] * extra + [base] * (slots - extra)


@strategy("greedy")
def greedy(rnd, q, score):
    """Places hotkeys one at a time on the position with the best mean."""
    top = collections.Counter()
    spare = collections.Counter()
    used_top = used_spare = 0
    for _ in range(q):
        best = None
        for depth in [0] + sorted(top):
            if depth == 0 and used_top >= rnd.n_top:
                continue
            if depth and not top[depth]:
                continue
            trial = top.copy()
            if depth:
                trial[depth] -= 1
            trial[depth + 1] += 1
            mean_a, _ = score(_board(rnd.omega, list(trial.elements()),
                                     list(spare.elements())), rnd.difficulty)
            if best is None or mean_a > best[0] + 1e-12:
                best = (mean_a, "top", depth)
        for depth in [0] + sorted(spare):
            if depth == 0 and used_spare >= rnd.n_spare:
                continue
            if depth and not spare[depth]:
                continue
            trial = spare.copy()
            if depth:
                trial[depth] -= 1
            trial[depth + 1] += 1
            mean_a, _ = score(_board(rnd.omega, list(top.elements()),
                                     list(trial.elements())), rnd.difficulty)
            if best is None or mean_a > best[0] + 1e-12:
                best = (mean_a, "spare", depth)
        assert best is not None
        _value, kind, depth = best
        target = top if kind == "top" else spare
        if depth:
            target[depth] -= 1
            if not target[depth]:
                del target[depth]
        elif kind == "top":
            used_top += 1
        else:
            used_spare += 1
        target[depth + 1] += 1
    return _board(rnd.omega, list(top.elements()), list(spare.elements()))


@strategy("even")
def even(rnd, q, score):
    """Spreads every hotkey over the maximum cliques as evenly as possible."""
    slots = max(1, min(rnd.n_top, q))
    return _board(rnd.omega, _spread(q, slots), [])


@strategy("solo")
def solo(rnd, q, score):
    """Takes one maximum clique per hotkey, stacking evenly once they run out."""
    slots = max(1, rnd.n_top)
    if q <= slots:
        return _board(rnd.omega, [1] * q, [])
    return _board(rnd.omega, _spread(q, slots), [])


@strategy("maximin_oracle")
def maximin_oracle(rnd, q, score):
    """Upper bound only: picks the split knowing the opponent's true count."""
    return _maximin(rnd, q, rnd.q_b)


@strategy("maximin")
def maximin(rnd, q, score):
    """Picks the omega/omega-1 split against an ESTIMATED opponent count."""
    return _maximin(rnd, q, estimate_q_b(rnd))


def _maximin(rnd, q, q_b):
    import native
    if rnd.n_top >= q:
        return _board(rnd.omega, [1] * q, [])
    best = None
    for at_omega in range(min(rnd.n_top, q), q + 1):
        rest = q - at_omega
        if at_omega and not rnd.n_top:
            continue
        if rest and not rnd.n_spare:
            continue
        board = []
        if at_omega:
            board += [(rnd.omega, c, 0)
                      for c in _spread(at_omega, min(rnd.n_top, at_omega))]
        if rest:
            board += [(rnd.omega - 1, c, 0)
                      for c in _spread(rest, min(rnd.n_spare, rest))]
        board = [b for b in board if b[1] > 0]
        if not board:
            continue
        _t, mean_a, mean_b = native.best_response(board, rnd, q_b)
        if best is None or (mean_a - mean_b) > best[0] + 1e-12:
            best = (mean_a - mean_b, board)
    assert best is not None
    return best[1]
