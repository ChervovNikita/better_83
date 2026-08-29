"""Rounds loaded from a solved pool dump, with their cliques already known."""

import json
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
DEFAULT_POOL = os.path.join(_ROOT, "pool_n20.jsonl")
DEFAULT_ROUNDS = os.path.join(_ROOT, "rounds.json")

DIFFICULTY_BY_NODES = ((290, 300, 0.7), (490, 500, 0.8),
                       (690, 700, 0.9), (890, 900, 1.0))


def difficulty_from_nodes(n_nodes):
    """Returns the difficulty the validator assigned to this vertex count."""
    for low, high, value in DIFFICULTY_BY_NODES:
        if low <= n_nodes <= high:
            return value
    assert False, n_nodes


class Round(object):
    """One solved round: its clique supply and how many miners answered."""

    __slots__ = ("uuid", "difficulty", "omega", "n_top", "n_spare",
                 "n_answers", "q_a", "q_b", "fleet_a", "fleet_b")

    def __repr__(self):
        return ("Round(%s D=%.1f omega=%d P=%d spare=%d answers=%d)"
                % (self.uuid[:8], self.difficulty, self.omega, self.n_top,
                   self.n_spare, self.n_answers))


def load(pool_path=None, rounds_path=None):
    """Returns every round that the solver enumerated cliques for."""
    pool_path = pool_path or DEFAULT_POOL
    rounds_path = rounds_path or DEFAULT_ROUNDS
    assert os.path.exists(pool_path), pool_path
    assert os.path.exists(rounds_path), rounds_path

    with open(rounds_path) as handle:
        field = json.load(handle)

    out = []
    with open(pool_path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            uuid = record["uuid"]
            assert uuid in field, uuid
            rnd = Round()
            rnd.uuid = uuid
            rnd.difficulty = difficulty_from_nodes(record["n_nodes"])
            rnd.omega = record["omega"]
            rnd.n_top = record["n_top_true"]
            rnd.n_spare = record["n_spare_true"]
            rnd.n_answers = sum(1 for a in field[uuid]["answers"] if a[3])
            rnd.q_a = 0
            rnd.q_b = 0
            rnd.fleet_a = 0
            rnd.fleet_b = 0
            out.append(rnd)
    assert out
    return out


def split_queries(rnd, g, o):
    """Divides the round's real answer count between the two fleets."""
    assert g > 0 and o > 0
    assert rnd.n_answers >= 2
    q_a = int(round(rnd.n_answers * g / float(g + o)))
    q_a = max(1, min(rnd.n_answers - 1, q_a))
    return q_a, rnd.n_answers - q_a
