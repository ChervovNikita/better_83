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
