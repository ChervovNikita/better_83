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

    The validator queries every eligible miner independently with the same
    probability, so the opponent's count is Binomial(fleet_b, p) and A's own
    count carries no information about it. The mean is the estimate; using
    p * METAGRAPH - q_a instead injects q_a's noise for nothing and measured
    23% relative error against 12% for this. Zero is a legal estimate: at the
    lopsided splits p * fleet_b rounds to 0, and flooring at 1 made A defend
    against an opponent it does not expect to face.
    """
    return max(0, int(round(selection_p(rnd.difficulty) * rnd.fleet_b)))


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
    """Upper bound only: reads q_b_oracle, which a real miner cannot observe."""
    return _maximin(rnd, q, rnd.q_b_oracle)


def _posterior(rnd, tail=1e-4, buckets=9):
    """Binomial(fleet_b, p) support and weights, trimmed to the useful mass."""
    n = rnd.fleet_b
    p = selection_p(rnd.difficulty)
    mean = n * p
    sd = math.sqrt(max(1e-9, n * p * (1.0 - p)))
    lo = max(1, int(mean - 4 * sd))
    hi = min(n, int(mean + 4 * sd) + 1)
    out = []
    total = 0.0
    for k in range(lo, hi + 1):
        w = math.comb(n, k) * (p ** k) * ((1.0 - p) ** (n - k))
        if w > tail:
            out.append((k, w))
            total += w
    assert out
    out = [(k, w / total) for k, w in out]
    if buckets <= 0 or len(out) <= buckets:
        return out
    step = len(out) / float(buckets)
    merged = []
    for i in range(buckets):
        chunk = out[int(i * step):int((i + 1) * step)] or [out[-1]]
        mass = sum(w for _k, w in chunk)
        centre = int(round(sum(k * w for k, w in chunk) / mass))
        merged.append((max(1, centre), mass))
    total = sum(w for _k, w in merged)
    return [(k, w / total) for k, w in merged]


@strategy("bayes")
def bayes(rnd, q, score):
    """Maximises the expected q-weighted margin over A's posterior for q_b.

    A dominates on emissions when sum(q_a*mean_A) >= sum(q_b*mean_B), so at the
    crossover the per-round objective is q_a*mean_A - q_b*mean_B, and q_b is a
    known Binomial rather than a point estimate.
    """
    import native
    return native.bayes(rnd, q, _posterior(rnd))


@strategy("maximin")
def maximin(rnd, q, score):
    """Picks the omega/omega-1 split against an ESTIMATED opponent count."""
    return _maximin(rnd, q, estimate_q_b(rnd))


def _maximin(rnd, q, q_b):
    import native
    return native.maximin(rnd, q, q_b)


def a_candidates(rnd, q):
    """A's board family: every (split, width) pair on the width/minimum frontier.

    A's diversity term depends on its own board only through the number of
    distinct cliques occupied and the smallest count on any of them, so for a
    target minimum m the best board is the widest spread whose counts all reach
    m. Sweeping m therefore covers the frontier.
    """
    seen = set()
    out = []
    for at_omega in range(q + 1):
        rest = q - at_omega
        if at_omega and not rnd.n_top:
            continue
        if rest and not rnd.n_spare:
            continue
        for m in range(1, q + 1):
            j_top = max(1, min(rnd.n_top, at_omega // m)) if at_omega else 0
            j_spare = max(1, min(rnd.n_spare, rest // m)) if rest else 0
            if not j_top and not j_spare:
                continue
            if (at_omega, j_top, j_spare) in seen:
                continue
            seen.add((at_omega, j_top, j_spare))
            board = []
            if j_top:
                board += [(rnd.omega, c, 0) for c in _spread(at_omega, j_top)]
            if j_spare:
                board += [(rnd.omega - 1, c, 0) for c in _spread(rest, j_spare)]
            board = [b for b in board if b[1] > 0]
            if board:
                out.append(board)
    return out


def _maximin_python(rnd, q, q_b):
    import native
    w_a, w_b = q / float(rnd.fleet_a), q_b / float(rnd.fleet_b)
    best = None
    for board in a_candidates(rnd, q):
        _t, mean_a, mean_b = native.best_response(board, rnd, q_b)
        obj = w_a * mean_a - w_b * mean_b
        if best is None or obj > best[0] + 1e-12:
            best = (obj, board)
    assert best is not None
    return best[1]
