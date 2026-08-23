"""The researched champion (research/native/clique.cpp), wired for the live miner.

Everything in research/ was measured against a solver that the deployed miner never
called: CliqueAI/miner.py runs `nx.approximation.max_clique`, a greedy approximation
that does not reach omega. "Promoted to native/clique.cpp" meant promoted within the
research harness. This module is the missing link.

Three properties matter more than speed here, because a bad answer scores a hard zero
while the upstream answer at least scores something:

  1. FALLBACK. Any failure -- missing .so, build error, ctypes fault, timeout overrun --
     returns the upstream result instead of raising. The miner must always answer.
  2. VALIDATION. The returned vertex set is checked to be a clique AND maximal before it
     is used. The validator tests maximality, not maximumness, so a non-maximal answer
     is worth zero even when it is large.
  3. DEADLINE. The solver is given the synapse timeout minus a round-trip reserve. A
     late answer is a zero on chain. The 2 s reserve was measured: parity 99.800% ->
     99.600%, McNemar p=1.0000, reward -0.0001, 0 invalid, 0 over budget.
"""
import os
import sys

import numpy as np

_RESEARCH = os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "research")

# Seconds held back from the deadline for the trip home through the dendrite.
ROUND_TRIP_S = float(os.environ.get("SN83_ROUND_TRIP_S", "2.0"))
# Used only when the synapse carries no timeout. The observed range is 6-30 s, so
# this is deliberately at the bottom of it: overrunning scores zero, finishing early
# only costs unused search.
DEFAULT_TIMEOUT_S = float(os.environ.get("SN83_DEFAULT_TIMEOUT_S", "6.0"))


def _is_valid_maximal_clique(A, verts):
    """The validator's own test: a clique, and extendable by no vertex."""
    if not verts:
        return False
    v = np.asarray(sorted(set(int(x) for x in verts)), dtype=np.int64)
    if len(v) != len(verts):
        return False                       # duplicate vertices
    if v.min() < 0 or v.max() >= A.shape[0]:
        return False
    sub = A[np.ix_(v, v)]
    if not np.array_equal(sub, sub.T):
        return False
    k = len(v)
    if sub.sum() != k * (k - 1):           # every pair adjacent, zero diagonal
        return False
    inC = np.zeros(A.shape[0], dtype=bool)
    inC[v] = True
    # maximal: no outside vertex is adjacent to all k members
    if (A[v].sum(axis=0)[~inC] == k).any():
        return False
    return True


def native_algorithm(number_of_nodes, adjacency_list, adjacency_matrix=None,
                     timeout=None, fallback=None):
    """Return a maximal clique, falling back to `fallback` on any problem.

    `fallback` is called with no arguments and must return a vertex list; pass the
    upstream algorithm. If it is None and the native path fails, an empty list is
    returned, which the validator scores as zero -- so always pass one.
    """
    def _fb():
        try:
            return list(fallback()) if fallback is not None else []
        except Exception:
            return []

    try:
        if adjacency_matrix is None:
            A = np.zeros((number_of_nodes, number_of_nodes), dtype=np.uint8)
            for i, nbrs in enumerate(adjacency_list):
                for j in nbrs:
                    A[i, int(j)] = 1
        else:
            A = np.ascontiguousarray(adjacency_matrix, dtype=np.uint8)
        np.fill_diagonal(A, 0)

        budget = float(timeout) if timeout else DEFAULT_TIMEOUT_S
        budget = max(0.5, budget - ROUND_TRIP_S)

        if _RESEARCH not in sys.path:
            sys.path.insert(0, _RESEARCH)
        from fastsolver import solve as solve_one       # builds the .so if stale

        clique = solve_one(A, budget)
        clique = [int(x) for x in clique]
        if _is_valid_maximal_clique(A, clique):
            return sorted(clique)
        return _fb()
    except Exception:
        return _fb()
