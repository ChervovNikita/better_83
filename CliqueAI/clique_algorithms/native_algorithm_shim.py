import os
import sys
import threading
import time

import numpy as np

_RESEARCH = os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "research")
if _RESEARCH not in sys.path:
    sys.path.insert(0, _RESEARCH)

from fastsolver import solve as solve_one

def available_cores():
    n = os.cpu_count() or 1
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:
            quota, period = f.read().split()
        if quota != "max":
            n = min(n, max(1, int(int(quota) / int(period))))
    except Exception:
        pass
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as f:
            q = int(f.read())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f:
            per = int(f.read())
        if q > 0 and per > 0:
            n = min(n, max(1, q // per))
    except Exception:
        pass
    try:
        n = min(n, len(os.sched_getaffinity(0)))
    except Exception:
        pass
    return max(1, n)


TOTAL_THREADS = int(os.environ.get("SN83_THREADS", "0")) or min(8, available_cores())
_active = 0
_active_lock = threading.Lock()
_SIBLINGS = {}


def _load_sibling(name):
    import importlib.util
    if name in _SIBLINGS:
        return _SIBLINGS[name]
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), name + ".py")
    spec = importlib.util.spec_from_file_location("_sn83_" + name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _SIBLINGS[name] = mod
    return mod
MIN_BUDGET_S = float(os.environ.get("SN83_MIN_BUDGET_S", "0.75"))

ROUND_TRIP_S = float(os.environ.get("SN83_ROUND_TRIP_S", "2.0"))
DEFAULT_TIMEOUT_S = float(os.environ.get("SN83_DEFAULT_TIMEOUT_S", "6.0"))


def _is_valid_maximal_clique(A, verts):
    if not verts:
        return False
    v = np.asarray(sorted(set(int(x) for x in verts)), dtype=np.int64)
    if len(v) != len(verts):
        return False
    if v.min() < 0 or v.max() >= A.shape[0]:
        return False
    sub = A[np.ix_(v, v)]
    if not np.array_equal(sub, sub.T):
        return False
    k = len(v)
    if sub.sum() != k * (k - 1):
        return False
    inC = np.zeros(A.shape[0], dtype=bool)
    inC[v] = True
    if (A[v].sum(axis=0)[~inC] == k).any():
        return False
    return True


def solver_seed(hotkey, uuid):
    import hashlib
    h = hashlib.sha1(("%s|%s" % (hotkey, uuid)).encode()).hexdigest()
    return int(h[:16], 16) & 0x7FFFFFFFFFFFFFFF


SPREAD = int(os.environ.get("SN83_SPREAD", "0"))
COORD = int(os.environ.get("SN83_COORD", "1"))


def fleet_size():
    v = os.environ.get("SN83_FLEET_SIZE")
    if v:
        try:
            return max(1, int(v))
        except ValueError:
            pass
    try:
        return max(1, _load_sibling("pool_coordinator").observed_fleet())
    except Exception:
        return 1


def _warn_thread_budget():
    return




def difficulty_from_n(number_of_nodes):
    n = int(number_of_nodes)
    if 290 <= n <= 300:
        return 0.7
    if 490 <= n <= 500:
        return 0.8
    if 690 <= n <= 700:
        return 0.9
    if 890 <= n <= 900:
        return 1.0
    return 0.8


def _spread_pick(pool, hotkey, uuid, difficulty, fleet_size):
    if not pool:
        return None
    mx = max(len(c) for c in pool)
    top = [c for c in pool if len(c) == mx]
    spare = [c for c in pool if len(c) == mx - 1]
    import hashlib
    import math
    x_m = math.sqrt(1.0 + 1.5)
    p = 1.0 - math.exp(-max(0.0, x_m - float(difficulty) - 0.5))
    q = max(1, int(round(fleet_size * p)))
    max_top = int(os.environ.get("SN83_SPREAD_MAX_TOP", "3"))
    if len(top) > max_top or len(top) >= q or not spare:
        return None
    else:
        slots = top + spare[:q - len(top)]
    slots = top + spare[:q - len(top)]
    h = int(hashlib.sha1(("%s|%s" % (hotkey, uuid)).encode()).hexdigest()[:16], 16)
    return slots[h % len(slots)]


def native_algorithm(number_of_nodes, adjacency_list, adjacency_matrix=None,
                     timeout=None, seed=0, hotkey=None, uuid=None,
                     difficulty=None):
    if adjacency_matrix is None:
        A = np.zeros((number_of_nodes, number_of_nodes), dtype=np.uint8)
        for i, nbrs in enumerate(adjacency_list):
            for j in nbrs:
                A[i, int(j)] = 1
    else:
        A = np.ascontiguousarray(adjacency_matrix, dtype=np.uint8)
    np.fill_diagonal(A, 0)

    deadline = time.monotonic() + (float(timeout) if timeout
                                   else DEFAULT_TIMEOUT_S) - ROUND_TRIP_S

    budget = deadline - time.monotonic()
    if budget < MIN_BUDGET_S:
        return []
    global _active
    with _active_lock:
        _active += 1
        in_process = _active
    _concurrent = max(in_process, fleet_size())
    share = max(1, min(TOTAL_THREADS, available_cores() // _concurrent))
    try:
        if COORD and hotkey is not None and uuid is not None:
            pcoord = _load_sibling("pool_coordinator")
            from fleet_solver import solve_many
            clique = None
            if int(os.environ.get("SN83_LAZY_HARVEST", "1")):
                _res = float(os.environ.get("SN83_HARVEST_RESERVE", "0.20"))
                _t = max(0.5, (deadline - time.monotonic()) * (1.0 - _res))
                _first = tuple(sorted(int(v) for v in
                                      solve_one(A, _t, seed=seed, threads=share)))
                if _first and pcoord.claim_clique(uuid, hotkey, _first):
                    clique = list(_first)
            _scarce = int(os.environ.get("SN83_SCARCE_SPREAD", "1"))
            mine = [] if clique is not None else [
                tuple(sorted(int(v) for v in c))
                for c in solve_many(A, max(0.5, deadline - time.monotonic()),
                                    int(os.environ.get("SN83_HARVEST_N", "10")),
                                    seed=seed, threads=share,
                                    pool_mode="ban" if _scarce else None,
                                    ban_n=1 if _scarce else None,
                                    champion_share=0.35 if _scarce else None)]
            if mine:
                mmax = max(len(c) for c in mine)
                ordered = [c for c in mine if len(c) == mmax]
                chosen = None
                if _scarce and len(ordered) <= int(
                        os.environ.get("SN83_SCARCE_MAX_ND", "2")):
                    for cand in sorted((c for c in mine if len(c) == mmax - 1
                                        and _is_valid_maximal_clique(A, list(c))),
                                       key=len, reverse=True):
                        if pcoord.claim_clique(uuid, hotkey, cand):
                            chosen = cand
                            break
                if chosen is None:
                    for cand in ordered:
                        if pcoord.claim_clique(uuid, hotkey, cand):
                            chosen = cand
                            break
                if chosen is not None:
                    clique = list(chosen)
                else:
                    agree, claimants = pcoord.distinct_claimed(uuid)
                    spares = sorted((c for c in mine if len(c) < mmax),
                                    key=len, reverse=True)
                    ratio = float(os.environ.get("SN83_AGREE_RATIO", "0.50"))
                    picked = None
                    min_claim = int(os.environ.get("SN83_AGREE_MIN_CLAIMANTS", "3"))
                    if (claimants >= min_claim and agree > 0
                            and agree / float(claimants) <= ratio and spares):
                        for cand in spares:
                            if len(cand) == mmax - 1 and \
                                    pcoord.claim_clique(uuid, hotkey, cand):
                                picked = cand
                                break
                    if picked is None:
                        for cand in ordered:
                            if pcoord.claim_clique(uuid, hotkey, cand):
                                picked = cand
                                break
                        else:
                            pthr = int(os.environ.get("SN83_PARTIAL_THR", "99"))
                            if agree <= pthr:
                                for cand in spares:
                                    if len(cand) == mmax - 1 and \
                                            pcoord.claim_clique(uuid, hotkey, cand):
                                        picked = cand
                                        break
                    clique = list(picked if picked is not None else ordered[0])
            elif clique is None:
                clique = solve_one(A, max(0.5, deadline - time.monotonic()),
                                   seed=seed, threads=share)
        elif SPREAD and hotkey is not None and uuid is not None:
            from fleet_solver import solve_many
            want = max(2, fleet_size())
            pool = solve_many(A, budget, want, seed=seed, threads=share)
            pick = _spread_pick([tuple(c) for c in pool], hotkey, uuid,
                                difficulty if difficulty is not None else 0.8,
                                fleet_size())
            clique = list(pick) if pick else solve_one(A, budget, seed=seed,
                                                       threads=share)
        else:
            clique = solve_one(A, budget, seed=seed, threads=share)
    finally:
        with _active_lock:
            _active -= 1

    clique = [int(x) for x in clique]
    if _is_valid_maximal_clique(A, clique):
        return sorted(clique)
    return []
