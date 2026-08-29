"""Strategies for the second player, who answers knowing part of the board."""

import collections

import native
import partial as partial_module
from scoring import score

REGISTRY = {}


def responder(name):
    """Registers a second-player responder under `name`."""
    def wrap(fn):
        REGISTRY[name] = fn
        return fn
    return wrap


@responder("full")
def full(board, rnd, q, rng):
    """Answers knowing which clique carries which occupancy."""
    return native.best_response(board, rnd, q)


@responder("full_pooled")
def full_pooled(board, rnd, q, rng):
    """Answers maximising the POOLED objective the sweep actually scores.

    pooled_B - pooled_A weights each round by its answer count, so the
    per-round objective is q_b*mean_B - q_a*mean_A. Answering with the
    unweighted margin instead optimises a different quantity than the one
    reported, which made a first player appear to win from the minority.
    """
    # pooled_B - pooled_A weights by q/sum(q), and sum(q) is proportional to the
    # FLEET size, so the per-round weights are q_a/fleet_a and q_b/fleet_b.
    # Using q_a and q_b raw is only correct when the fleets are equal.
    return native.best_response_weighted(
        board, rnd, q, rnd.q_a / float(rnd.fleet_a), q / float(rnd.fleet_b))


@responder("partial")
def partial(board, rnd, q, rng):
    """Answers knowing only the occupancy multiset, then places at random."""
    view = build_view(board, rnd)
    _gap, plan, _mean_a, _mean_b = partial_module.best_response_fast(
        view, rnd.difficulty, q, rnd.omega)
    trial = partial_module.realise(plan, rng)
    mean_a, mean_b = score(trial, rnd.difficulty)
    return trial, mean_a, mean_b


def build_view(board, rnd):
    """Returns the occupancy multiset and free capacity per size class."""
    occupied = collections.defaultdict(list)
    for size, n_a, n_b in board:
        if n_a + n_b > 0:
            occupied[size].append(n_a + n_b)
    view = {}
    for size in (rnd.omega, rnd.omega - 1):
        supply = rnd.n_top if size == rnd.omega else rnd.n_spare
        counts = occupied.get(size, [])
        view[size] = {"counts": counts,
                      "free": max(0, supply - len(counts))}
    return view
