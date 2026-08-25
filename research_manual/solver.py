#!/usr/bin/env python3

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "eda"))

from fleet_pick import picker

LATENCY_S = 2.0

SOLVER = os.environ.get("SN83_SOLVER", "gpu").lower()
assert SOLVER in ("gpu", "cpu"), SOLVER

if SOLVER == "cpu":
    from fleet_solver import solve_many
else:
    from fleet_solver_gpu import solve_many


def solve(hotkeys, adjacency_matrix, time_limit, uuid):
    assert hotkeys
    budget = time_limit - LATENCY_S
    assert budget > 0, time_limit
    matrix = np.asarray(adjacency_matrix, dtype=np.uint8)
    pool = solve_many(matrix, budget, len(hotkeys))
    answers = picker(pool, uuid, list(hotkeys))
    assert len(answers) == len(hotkeys)
    return answers
