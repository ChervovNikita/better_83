"""Baseline SN83 solver: greedy multi-start + (1,2)-swap plateau local search.

This is the reference point to beat, not a competitive miner. Measured at mean
reward 1.755 against the live field, which the score-to-income curve puts in
the dead zone. Kept so every change has something to regress against.
"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from CliqueAI.graph.codec import GraphCodec

def decode(enc):
    c = GraphCodec()
    m = c.decode_matrix(enc)
    A = np.array(m, dtype=np.uint8)
    return A

def greedy_from(A, seed_v, rng, order_noise=0.0):
    n = A.shape[0]
    cnt = A[seed_v].astype(np.int32).copy()
    C = [seed_v]
    k = 1
    while True:
        cand = np.flatnonzero(cnt == k)
        cand = cand[cand != seed_v]
        cand = np.setdiff1d(cand, np.array(C), assume_unique=False)
        if cand.size == 0:
            break
        # pick candidate with max degree within candidate set (+ noise)
        sub = A[np.ix_(cand, cand)].sum(axis=1).astype(np.float64)
        if order_noise:
            sub = sub + rng.random(cand.size) * order_noise
        v = int(cand[int(np.argmax(sub))])
        C.append(v); k += 1
        cnt += A[v]
    return C

def local_search(A, C, rng, deadline, best_global):
    """(1,2)-swap plateau local search with penalties (DLS-MC flavour)."""
    n = A.shape[0]
    inC = np.zeros(n, dtype=bool)
    cnt = np.zeros(n, dtype=np.int32)
    for v in C:
        inC[v] = True
        cnt += A[v]
    k = len(C)
    penalty = np.zeros(n, dtype=np.int32)
    best = list(C); bestk = k
    it = 0
    while time.time() < deadline:
        it += 1
        # expand
        addable = np.flatnonzero((cnt == k) & (~inC))
        if addable.size:
            p = penalty[addable]
            pick = addable[p == p.min()]
            v = int(pick[rng.integers(pick.size)])
            inC[v] = True; cnt += A[v]; k += 1
            if k > bestk:
                bestk = k; best = np.flatnonzero(inC).tolist()
            continue
        # plateau swap: v adjacent to all but one member
        swappable = np.flatnonzero((cnt == k - 1) & (~inC))
        if swappable.size:
            p = penalty[swappable]
            pick = swappable[p == p.min()]
            v = int(pick[rng.integers(pick.size)])
            members = np.flatnonzero(inC)
            u = int(members[np.flatnonzero(A[v][members] == 0)[0]])
            inC[u] = False; cnt -= A[u]
            inC[v] = True;  cnt += A[v]
            penalty[u] += 1
            continue
        # stuck: perturb — drop a random subset, bump penalties
        members = np.flatnonzero(inC)
        penalty[members] += 1
        drop = rng.choice(members, size=max(1, len(members) // 4), replace=False)
        for u in drop:
            inC[u] = False; cnt -= A[u]
        k = int(inC.sum())
        if it % 50 == 0:
            penalty //= 2
    return bestk, best, it

def solve(A, time_limit, seed=0):
    rng = np.random.default_rng(seed)
    n = A.shape[0]
    deadline = time.time() + time_limit
    deg = A.sum(axis=1)
    order = np.argsort(-deg)
    bestk, best = 0, []
    # a few greedy restarts to seed
    for i in range(min(5, n)):
        v = int(order[int(rng.integers(min(50, n)))])
        C = greedy_from(A, v, rng, order_noise=2.0)
        if len(C) > bestk:
            bestk, best = len(C), C
        if time.time() > deadline: break
    greedy_best = bestk
    k2, b2, iters = local_search(A, best, rng, deadline, bestk)
    if k2 > bestk:
        bestk, best = k2, b2
    return greedy_best, bestk, best, iters

def verify(A, C):
    S = list(C)
    for i in range(len(S)):
        for j in range(i+1, len(S)):
            if not A[S[i], S[j]]:
                return False, "not a clique"
    inC = np.zeros(A.shape[0], dtype=bool); inC[S] = True
    cnt = A[S].sum(axis=0)
    if np.any((cnt == len(S)) & (~inC)):
        return False, "not maximal"
    return True, "ok"

# eval_harness calls solve(A, time_limit) and takes the third tuple element.
# Replace this module with the real solver; the interface is the contract.
