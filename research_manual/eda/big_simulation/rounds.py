"""Rounds loaded from a solved pool dump, with their cliques already known."""

import json
import math
import os
import random
import zlib

REF_R = 1.5

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
def _load_paths():
    """Imports research_manual/paths.py, the project's own path registry."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "rm_paths", os.path.join(_ROOT, "paths.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# research_manual moved its data under artifacts/ and now resolves it through
# paths.py; follow that rather than keep a second copy of the layout here.
_PATHS = _load_paths()
DEFAULT_POOL = os.path.join(_PATHS.POOLS, "pool_n20.jsonl")
DEFAULT_ROUNDS = _PATHS.ROUNDS_JSON

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
                 "n_answers", "q_a", "q_b", "q_b_oracle", "fleet_a",
                 "fleet_b")

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
            rnd.q_b_oracle = 0
            rnd.fleet_a = 0
            rnd.fleet_b = 0
            out.append(rnd)
    assert out
    return out


def selection_p(difficulty):
    """The validator's per-hotkey query probability, from MinerSelector."""
    return 1.0 - math.exp(-max(0.0, math.sqrt(1.0 + REF_R) - difficulty - 0.5))


def split_queries(rnd, g, o, rng=None, seed=0):
    """Draws how many hotkeys of each fleet the validator queries.

    MinerSelector.sample_miner_uids gives every eligible miner the SAME
    probability P and draws each independently, so the two counts are
    independent Binomials rather than a split of one total. Deriving them from a
    single observed answer count instead ties them together at correlation
    0.999, which hands the first player an estimate of its opponent that the
    real mechanism does not provide.
    """
    assert g > 0 and o > 0
    p = selection_p(rnd.difficulty)
    if rng is None:
        # zlib.crc32 is stable across processes; hash() on a str is salted by
        # PYTHONHASHSEED, so seeding from it made every run draw different
        # queries and no two runs comparable.
        rng = random.Random(zlib.crc32(rnd.uuid.encode()) ^ (seed * 0x9E3779B1))
    q_a = sum(1 for _ in range(g) if rng.random() < p)
    q_b = sum(1 for _ in range(o) if rng.random() < p)
    return max(1, q_a), max(1, q_b)
