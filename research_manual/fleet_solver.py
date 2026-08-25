#!/usr/bin/env python3
"""One solve, many DISTINCT maximal cliques.

    solve_many(adjacency_matrix, time_limit, k) -> list[list[int]]

The validator sends the SAME graph to every hotkey it queries, so a fleet does not
get k solve budgets -- one search has to produce every answer the fleet returns.

Two stages, because they are two different jobs:

  1. Find omega. The native core (clique.cpp, variant v7_fastscan: bitset
     adjacency, strong configuration checking, stagnation-driven perturbation)
     does this. It matched the field's best size on 10 of 10 rounds measured.
  2. Find OTHER cliques of that size. This is what the fleet is actually paid for:
     the reward is optimality*(1+difficulty) + diversity, and diversity is
     1/(number of miners submitting the identical vertex set). Optimality is
     already pinned at 1.0, so every extra distinct clique is the only headroom
     left.

A group of c identical answers is worth c*(1/c) = 1 unit of diversity in total,
whatever c is -- so a repeat inside the fleet is worth NOTHING, and widening the
pool is the entire job of stage 2.

Stage 2 is delete-and-resolve: take a clique we already hold, DELETE one of its
vertices from the graph, and re-run the full solver on what is left. The clique we
held is now impossible, so the solver has to land somewhere else.

The alternative -- keep 7/8 of a clique and greedily re-extend -- was measured at
26,000-32,000 attempts on one round producing ZERO new maximum cliques, because a
greedy extension of 25 of a 36-clique's vertices returns that same clique
essentially always. Distinct maxima harvested on three rounds:

    round (field distinct)   keep-and-extend      delete-and-resolve
    4  (22)                  1  jac  n/a          10  jac 0.291
    5  (23)                  2  jac 0.968          8  jac 0.237
    8  (37)                  2  jac 0.929         16  jac 0.190

The field's own distinct maxima sit at Jaccard 0.4669, so 0.19-0.29 is not merely
better than the walk, it is more spread out than the field itself.

BUILDING
--------
The search core is `clique.cpp`, compiled to `libclique.so` next to it. Both live
in this directory, so nothing outside it is needed.

    g++ -O3 -march=native -funroll-loops -fno-plt -std=c++17 -pthread \
        -shared -fPIC clique.cpp -o libclique.so

You do not normally have to run that. Importing this module builds the .so if it
is missing, if clique.cpp is newer than it, or if loading it fails -- so
`rm libclique.so` is a complete "rebuild it". The build needs g++ and libstdc++
and nothing else, and takes about 2 seconds (measured on this box).

The build happens at IMPORT, never inside solve_many: a compile inside a solve
would eat the round's deadline, and a late answer scores zero.

The shipped .so is compiled with -march=native, i.e. for the CPU that built it.
On an older machine loading it can fail; that is caught and triggers one local
rebuild. For a binary meant to travel, replace -march=native with
`-mavx2 -mbmi2 -mpopcnt`.

No environment variables. Every knob below is a module constant holding the value
the sweeps settled on; the native core's own getenv() hooks are overrides that are
simply never set, so its compiled-in defaults apply.
"""
import ctypes
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "clique.cpp")
LIB = os.path.join(HERE, "libclique.so")

# How libclique.so is built. Run by hand as:
#
#     g++ -O3 -march=native -funroll-loops -fno-plt -std=c++17 -pthread \
#         -shared -fPIC clique.cpp -o libclique.so
#
# or just delete libclique.so and import this module -- _load() runs exactly this
# command whenever the .so is missing, older than clique.cpp, or refuses to load.
# Nothing beyond g++ and libstdc++ is needed; the build takes about 2s.
#
# -march=native is why the checked-in .so may not be portable: it is compiled for
# the CPU that built it, and moving it to an older machine can fault on an
# unsupported instruction. That case is handled -- a CDLL failure triggers one
# rebuild -- but if you want a binary that travels, swap -march=native for
# `-mavx2 -mbmi2 -mpopcnt` and rebuild.
BUILD = ["g++", "-O3", "-march=native", "-funroll-loops", "-fno-plt",
         "-std=c++17", "-pthread", "-shared", "-fPIC", SRC, "-o", LIB]

# min_compute.yml puts a real miner at 4 recommended / 8 cores. This box reports
# 128 but is CFS-capped near 15, so 8 is both the honest deployment number and
# inside the quota. Thread count changes the ANSWER here, not just the speed.
THREADS = 8

# Handed to the C++ side so it can finish its final maximality pass and get back
# through ctypes before the caller's clock runs out. A late answer scores zero.
GUARD_S = 0.03

# Budget split. Stage 1 gets CHAMPION_SHARE, stage 2 gets the rest. Measured
# time-to-omega on this round set is at or under 10% of the budget on 10 of 10
# rounds (n=290..900, deadlines 6-30s), so spending three quarters of the clock
# on a size we already have is pure waste -- the pool is what the fleet needs.
# 0.15 keeps a 50% margin over the measured figure, and stage 2 raises `target`
# anyway if a re-solve ever comes back larger.
CHAMPION_SHARE = 0.15

# Vertices deleted from the graph before each re-solve. One is enough to make the
# held clique impossible; deleting more aims the re-solve below omega and the
# harvest comes back empty.
BAN_N = 1

# Each re-solve gets this fraction of the total budget, DOUBLED whenever a solve
# comes back below the size we already know is reachable. A fixed 5% slice found
# 9 maxima on n=698 but only 1 on n=894, where 14 of 15 re-solves never reached
# omega at all. The doubling makes the slice safe on big graphs without giving up
# the many-cheap-solves behaviour on small ones. It starts at the measured
# time-to-omega rather than below it, so the first re-solve is not a guaranteed
# miss that only exists to trigger the doubling.
BAN_FRAC = 0.10

# Maximal cliques BELOW omega are still valid answers -- the validator tests
# maximality, not maximumness -- and the fleet needs them: when the max-size pool
# is shorter than the number of hotkeys queried, the picker otherwise hands two
# siblings the same clique. Keep them down to omega-SPARE_MARGIN.
SPARE_MARGIN = 1

# Left for the caller (fleet_pick, and the trip back out through solver.py).
RESERVE_S = 0.15

_lib = None


def _load():
    """ctypes bridge onto libclique.so, rebuilt if the source is newer.

    Done at import time, not inside solve_many: a g++ -O3 build takes tens of
    seconds and would blow the deadline of whichever round happened to be first.
    """
    global _lib
    if _lib is not None:
        return _lib
    if (not os.path.exists(LIB)) or os.path.getmtime(SRC) > os.path.getmtime(LIB):
        subprocess.run(BUILD, check=True, stderr=sys.stderr.fileno())
    try:
        lib = ctypes.CDLL(LIB)
    except OSError:
        # A .so built elsewhere with -march=native for a newer CPU. Rebuild here.
        subprocess.run(BUILD, check=True, stderr=sys.stderr.fileno())
        lib = ctypes.CDLL(LIB)
    lib.sn83_solve.restype = ctypes.c_int
    lib.sn83_solve.argtypes = [
        ctypes.POINTER(ctypes.c_uint8), ctypes.c_int, ctypes.c_double,
        ctypes.c_uint64, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int32)]
    _lib = lib
    return lib


def _extend(A, clique):
    """Grow a clique until nothing can be added to it, in THIS graph.

    Stage 2 solves a graph with vertices deleted, so what comes back is maximal
    in that graph and not necessarily in the original: the deleted vertex may be
    adjacent to every member of the clique the re-solve landed on, which makes
    the answer extendable and therefore INVALID -- `is_valid_maximum_clique`
    rejects any clique a vertex can be added to, and the whole answer scores 0 on
    both terms. Measured before this pass existed: 3 of 435 answers scored a hard
    zero, all of them harvested ones.
    """
    n = A.shape[0]
    members = list(clique)
    inC = np.zeros(n, dtype=bool)
    inC[members] = True
    cnt = A[members].sum(axis=0, dtype=np.int32)
    size = len(members)
    while True:
        cand = np.flatnonzero((cnt == size) & ~inC)
        if cand.size == 0:
            break
        v = int(cand[np.argmax(A[cand].sum(axis=1, dtype=np.int32))])
        inC[v] = True
        cnt = cnt + A[v]
        size += 1
    return tuple(sorted(np.flatnonzero(inC).tolist()))


def _solve_one(A, time_limit, seed):
    """One maximal clique, as large as the core can find inside time_limit."""
    lib = _load()
    A = np.ascontiguousarray(A, dtype=np.uint8)
    n = A.shape[0]
    out = np.zeros(n, dtype=np.int32)
    budget = max(0.01, float(time_limit) - GUARD_S)
    # bms_k=0 and restart_steps=0 mean "use the compiled-in defaults" (64 / 4000).
    size = lib.sn83_solve(
        A.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), n, budget,
        ctypes.c_uint64(seed), THREADS, 0, 0,
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
    return out[:size].tolist()


def solve_many(adjacency_matrix, time_limit, k):
    """Up to k distinct maximal cliques for this graph, largest first.

    Cliques of the maximum size found come first; any maximal cliques one vertex
    short follow, and are used by the picker only when the fleet would otherwise
    have to repeat an answer.
    """
    assert k > 0
    assert time_limit > 0
    A = np.ascontiguousarray(adjacency_matrix, dtype=np.uint8)
    n = A.shape[0]
    assert A.shape == (n, n)
    assert n > 0

    deadline = time.monotonic() + time_limit - RESERVE_S
    best = _solve_one(A, time_limit * CHAMPION_SHARE, seed=1)
    assert best
    out = [sorted(int(v) for v in best)]
    if k <= 1:
        return out

    rng = np.random.default_rng(1)
    seen = {tuple(out[0])}
    spare = []
    target = len(out[0])
    frac = BAN_FRAC

    # `len(out) < k`, NOT `len(out) + len(spare) < k`. Spares are a by-product of
    # re-solves that came back short; counting them against the budget lets a
    # couple of early short answers end the search for the max-size cliques the
    # fleet is actually paid for. Measured with spares counted: a median of 4.5
    # distinct maxima against 7.5 answers to fill, so 23.9% of what the fleet
    # submitted was a repeat of a sibling -- worth nothing at all in diversity.
    while len(out) < k and time.monotonic() < deadline:
        left = deadline - time.monotonic()
        if left < 0.2:
            break
        src = out[int(rng.integers(len(out)))]
        drop = rng.choice(list(src), size=min(BAN_N, len(src)), replace=False)
        B = A.copy()
        for v in drop:
            B[int(v), :] = 0
            B[:, int(v)] = 0
        cand = _solve_one(B, min(left, time_limit * frac),
                          seed=int(rng.integers(1 << 30)))
        if len(cand) == 0:
            continue
        # Maximal in B is not maximal in A -- the deleted vertices are back.
        cand = _extend(A, cand)
        if len(cand) > target:
            # A graph with vertices removed cannot beat the full one, but size the
            # pool off what we actually hold rather than trusting that.
            target = len(cand)
            out = [list(cand)]
            seen = {cand}
            spare = []
            continue
        if len(cand) < target:
            # Keep it as a spare BEFORE deciding what to do about the slice. An
            # earlier revision doubled and continued here, which consumed every
            # sub-omega result and left `spare` empty.
            if cand not in seen and len(cand) >= target - SPARE_MARGIN:
                seen.add(cand)
                spare.append(list(cand))
            if frac < 0.5:
                frac *= 2.0          # the slice was too short to reach omega
            continue
        if cand in seen:
            continue
        seen.add(cand)
        out.append(list(cand))

    out.sort(key=len, reverse=True)
    spare.sort(key=len, reverse=True)
    out.extend(spare)
    return out[:k]


_load()
