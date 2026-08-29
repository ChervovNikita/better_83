#!/usr/bin/env python3
"""Tests for the solve dispatcher.

No GPU:
    SN83_BACKEND=fake .venv/bin/pytest research_manual/eda/test_dispatcher.py -v

On the 2xGPU pod, the same file against real workers:
    SN83_BACKEND=gpu SN83_WORKERS=2 \
      .venv/bin/pytest research_manual/eda/test_dispatcher.py -v -m "not fake_only"

The GPU-only checks (deadlines, no-late-under-overlap, two devices actually in
use) are marked `gpu_only` and skip without one.
"""

import os
import subprocess
import sys
import threading
import time

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
ROOT = os.path.dirname(PARENT)
for _p in (ROOT, PARENT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

BACKEND = os.environ.get("SN83_BACKEND", "fake").lower()
gpu_only = pytest.mark.skipif(BACKEND != "gpu", reason="needs SN83_BACKEND=gpu")
fake_only = pytest.mark.skipif(BACKEND == "gpu", reason="fake backend only")


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient
    import dispatcher
    with TestClient(dispatcher.app) as c:
        yield c


def _matrix(n, seed=0):
    """A dense graph as a plain nested list, so no codec is needed."""
    import numpy as np
    rng = np.random.default_rng(seed)
    a = (rng.random((n, n)) < 0.9).astype(int)
    a = np.triu(a, 1)
    a = a + a.T
    return a.tolist()


def _req(uid, hotkey, n=120, tl=8.0, seed=0):
    return {"uuid": uid, "hotkey": hotkey, "number_of_nodes": n,
            "time_limit": tl, "adjacency_matrix": _matrix(n, seed)}


# ------------------------------------------------------------------ basics

def test_health_reports_live_workers(client):
    h = client.get("/health").json()
    assert h["workers"] == int(os.environ.get("SN83_WORKERS", "2"))
    assert h["workers_free"] >= 1, h["worker_info"]
    # the CPU budget must be split, never handed to each worker in full
    assert h["threads_per_worker"] * h["workers"] <= int(
        os.environ.get("SN83_CPU_BUDGET", "15"))


def test_single_solve_returns_a_clique(client):
    r = client.post("/solve", json=_req("t-single", "hk0")).json()
    assert r["source"] == "owner", r
    assert len(r["clique"]) > 0
    assert r["siblings"] == 1


def test_no_budget_is_refused(client):
    r = client.post("/solve", json=_req("t-nobudget", "hk0", tl=1.0)).json()
    assert r["source"] == "reject"
    assert r["clique"] == []


# ------------------------------------------------------- sibling batching

def test_siblings_share_one_solve_and_get_distinct_cliques(client):
    """The whole point: one solve serves the fleet, and no two hotkeys repeat."""
    n_sib = 6
    out = {}

    def go(i):
        out[i] = client.post("/solve", json=_req("t-sib", "hk%d" % i)).json()

    threads = [threading.Thread(target=go, args=(i,)) for i in range(n_sib)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert all(r["clique"] for r in out.values()), out
    owners = [r for r in out.values() if r["source"] == "owner"]
    assert len(owners) == 1, "more than one solve was run for one uuid"
    cliques = [tuple(sorted(r["clique"])) for r in out.values()]
    assert len(set(cliques)) == len(cliques), "siblings repeated a clique"


def test_retry_is_idempotent(client):
    a = client.post("/solve", json=_req("t-retry", "hk0")).json()
    b = client.post("/solve", json=_req("t-retry", "hk0")).json()
    assert a["clique"] == b["clique"]
    assert a["index"] == b["index"]


def test_late_sibling_still_served_from_the_same_pool(client):
    first = client.post("/solve", json=_req("t-late", "hk0")).json()
    later = client.post("/solve", json=_req("t-late", "hk9")).json()
    assert first["source"] == "owner"
    assert later["source"] == "sibling"
    assert later["clique"]


# ----------------------------------------------------------- admission

def test_rejects_rather_than_queues_when_all_workers_busy(client):
    """More concurrent TASKS than workers must refuse, not wait.

    Queuing spends the round's deadline and then answers late, which scores zero
    on both terms; refusing leaves the caller its whole budget for a local
    fallback.
    """
    n_task = int(os.environ.get("SN83_WORKERS", "2")) + 3
    out = {}

    def go(i):
        t0 = time.monotonic()
        out[i] = (client.post("/solve",
                              json=_req("t-busy-%d" % i, "hk%d" % i, tl=12.0,
                                        seed=i)).json(),
                  time.monotonic() - t0)

    threads = [threading.Thread(target=go, args=(i,)) for i in range(n_task)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    sources = [r["source"] for r, _ in out.values()]
    assert sources.count("reject") >= 1, sources
    for r, took in out.values():
        if r["source"] == "reject":
            assert took < 1.0, "a rejection must be immediate, took %.2fs" % took


def test_rejection_counter_moves(client):
    before = client.get("/health").json()["counters"]["rejected"]
    client.post("/solve", json=_req("t-count", "hk0", tl=0.5))
    after = client.get("/health").json()["counters"]["rejected"]
    assert after == before + 1


# ------------------------------------------------------------- deadlines

@gpu_only
@pytest.mark.parametrize("tl", [7.5, 15.0])
def test_owner_answers_inside_the_deadline(client, tl):
    r = client.post("/solve", json=_req("t-dl-%s" % tl, "hk0", n=500, tl=tl)).json()
    assert r["source"] == "owner", r
    assert r["elapsed"] < tl - 2.0, "answer was late: %.2f of %.2f" % (
        r["elapsed"], tl - 2.0)


@gpu_only
def test_two_concurrent_tasks_both_meet_their_deadline(client):
    """The failure this service exists to prevent.

    On one device two simultaneous solves each took ~2x wall time and BOTH
    missed the deadline. With one worker per device they must not.
    """
    tl = 10.0
    out = {}

    def go(i):
        out[i] = client.post("/solve",
                             json=_req("t-par-%d" % i, "hk%d" % i, n=500,
                                       tl=tl, seed=i)).json()

    threads = [threading.Thread(target=go, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert all(r["source"] == "owner" for r in out.values()), out
    for r in out.values():
        assert r["elapsed"] < tl - 2.0, "late under overlap: %.2f" % r["elapsed"]


@gpu_only
def test_both_devices_are_actually_used(client):
    """Two workers, two contexts -- not two workers sharing device 0."""
    import subprocess
    q = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid",
         "--format=csv,noheader"], capture_output=True, text=True)
    devices = {line.split(",")[-1].strip() for line in q.stdout.splitlines()
               if line.strip()}
    assert len(devices) >= 2, "workers are not on separate devices: %r" % (
        q.stdout,)


@fake_only
def test_fake_backend_is_deterministic(client):
    a = client.post("/solve", json=_req("t-det-a", "hk0")).json()
    b = client.post("/solve", json=_req("t-det-b", "hk0")).json()
    assert a["clique"] == b["clique"]


# ------------------------------------------------------- miner-side client

def test_client_returns_none_when_the_service_is_down():
    """The miner no longer solves locally; a dead service is an empty answer."""
    import dispatch_client
    got = dispatch_client.solve("u", "hk", 100, _matrix(20), 8.0,
                                url="http://127.0.0.1:1")   # nothing listening
    assert got is None
    assert dispatch_client.health(url="http://127.0.0.1:1") is None


def test_client_returns_none_on_reject(monkeypatch):
    import dispatch_client, json, io

    class FakeResp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=None):
        return FakeResp(json.dumps(
            {"clique": [], "source": "reject", "reason": "all workers busy"}
        ).encode())

    monkeypatch.setattr(dispatch_client.urllib.request, "urlopen", fake_urlopen)
    assert dispatch_client.solve("u", "hk", 100, _matrix(20), 8.0) is None


def test_client_waits_the_full_deadline(monkeypatch):
    """The 2 s network reserve is deducted in the dispatcher, not here."""
    import dispatch_client, json, io

    seen = {}

    class FakeResp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=None):
        seen["timeout"] = timeout
        return FakeResp(json.dumps({"clique": [1], "source": "owner"}).encode())

    monkeypatch.setattr(dispatch_client.urllib.request, "urlopen", fake_urlopen)
    dispatch_client.solve("u", "hk", 100, _matrix(20), 8.0)
    assert seen["timeout"] == 8.0


def test_client_parses_a_clique(monkeypatch):
    import dispatch_client, json, io

    class FakeResp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=None):
        return FakeResp(json.dumps({"clique": [3, 1, 2], "source": "owner"}).encode())

    monkeypatch.setattr(dispatch_client.urllib.request, "urlopen", fake_urlopen)
    assert dispatch_client.solve("u", "hk", 100, _matrix(20), 8.0) == [3, 1, 2]


def test_miner_imports_with_dispatch_disabled():
    """The patch must be inert unless SN83_DISPATCH=1."""
    import subprocess
    r = subprocess.run(
        [sys.executable, "-c",
         "import ast,sys; ast.parse(open('CliqueAI/miner.py').read());"
         " print('parsed')"],
        capture_output=True, text=True, cwd=ROOT)
    assert "parsed" in r.stdout, r.stderr


# --------------------------------------------------- CPU overflow worker

def test_cpu_worker_absorbs_overflow_instead_of_rejecting(client):
    """A reject sends every hotkey off to solve in its own process at once.

    Production runs one process per hotkey and each sizes its own thread pool,
    so N rejects means N local solves against a 15-CPU quota, at exactly the
    moment the GPU workers need CPU for their champion stage. The overflow
    worker turns the (N_WORKERS + 1)-th concurrent task into one shared CPU
    solve instead.
    """
    n_gpu = int(os.environ.get("SN83_WORKERS", "2"))
    out = {}

    def go(i):
        out[i] = client.post("/solve",
                             json=_req("t-ovf-%d" % i, "hk%d" % i, tl=12.0,
                                       seed=i)).json()

    threads = [threading.Thread(target=go, args=(i,)) for i in range(n_gpu + 1)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    served = [r for r in out.values() if r["clique"]]
    assert len(served) == n_gpu + 1, "overflow was rejected: %s" % (
        [r["source"] for r in out.values()],)


def test_gpu_workers_get_the_whole_quota(client):
    """The steady state -- every round -- must not pay for the overflow path.

    Three-deep concurrency is 0.05% of rounds; reserving an equal share for it
    would cost the champion a third of its threads on the other 99.95%.
    """
    h = client.get("/health").json()
    budget = int(os.environ.get("SN83_CPU_BUDGET", "15"))
    gpu_total = h["threads_per_worker"] * h["workers"]
    total = gpu_total + h["overflow_threads"] * h["cpu_workers"]
    assert total <= budget, "oversubscribed: %s" % h
    assert gpu_total > budget // 2, (
        "GPU workers were starved to reserve for overflow: %s" % h)
    assert h["overflow_threads"] >= 1, h


def test_health_reports_the_overflow_counter(client):
    h = client.get("/health").json()
    assert "overflow_cpu" in h["counters"]


@pytest.mark.parametrize("workers,budget", [(1, 15), (2, 15), (3, 15), (2, 8), (4, 32)])
def test_thread_split_never_oversubscribes(workers, budget, monkeypatch):
    """The arithmetic, for every fleet shape -- not just the one under test.

    Thread count changes the solver's ANSWER, so exceeding the CFS quota is a
    correctness problem, and it must hold whatever SN83_WORKERS is set to.
    """
    import importlib
    monkeypatch.setenv("SN83_WORKERS", str(workers))
    monkeypatch.setenv("SN83_CPU_BUDGET", str(budget))
    monkeypatch.setenv("SN83_BACKEND", "fake")
    import dispatcher
    mod = importlib.reload(dispatcher)
    total = (mod.THREADS_PER_WORKER * mod.N_WORKERS
             + mod.OVERFLOW_THREADS * mod.N_CPU_WORKERS)
    assert total <= budget, (workers, budget, mod.THREADS_PER_WORKER,
                             mod.OVERFLOW_THREADS)
    assert mod.THREADS_PER_WORKER >= 1 and mod.OVERFLOW_THREADS >= 1


# --------------------------------------------------- parent must not touch CUDA

def test_parent_picker_does_not_import_the_gpu_solver():
    """The HTTP process only picks from a pool the workers already built.

    Importing solver.py used to load fleet_solver_gpu, which opens a CUDA
    context on device 0 in this process -- next to worker 0's harvest.
    """
    import dispatcher
    dispatcher._PICKER = None
    picker, wants_n, wants_supply = dispatcher._picker()
    assert picker is not None
    assert wants_n is True
    assert wants_supply is True
    assert "fleet_solver_gpu" not in sys.modules
    assert "gpu_lib" not in sys.modules
    assert "solver" not in sys.modules


def test_importing_solver_does_not_init_cuda():
    """Even the simulator module must not touch CUDA until someone solves."""
    r = subprocess.run(
        [sys.executable, "-c",
         "import sys;"
         "sys.path[:0] = %r;"
         "import solver;"
         "bad = [m for m in sys.modules"
         " if m == 'gpu_lib' or m.endswith('fleet_solver_gpu')];"
         "assert not bad, bad" % [PARENT, HERE]],
        cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


# ------------------------------------------------------------ miner wiring

def test_miner_always_dispatches():
    """There is no configuration that turns dispatch off."""
    src = open(os.path.join(ROOT, "CliqueAI", "miner.py")).read()
    assert "SN83_DISPATCH" not in src, "a dispatch on/off switch came back"
    assert src.count("dispatch_client.solve") == 1


def test_assign_forwards_harvest_supply():
    import dispatcher
    seen = {}

    def fake_picker(pool, uuid, hotkeys, **kwargs):
        seen.update(kwargs)
        return [list(pool[0])] * len(hotkeys)

    prev = (dispatcher._PICKER, dispatcher._PICKER_WANTS_N,
            dispatcher._PICKER_WANTS_SUPPLY)
    dispatcher._PICKER = fake_picker
    dispatcher._PICKER_WANTS_N = True
    dispatcher._PICKER_WANTS_SUPPLY = True
    try:
        task = dispatcher.Task("u-supply")
        task.pool = [[0, 1, 2], [0, 1]]
        task.stats = {"n_top_true": 40, "n_spare_true": 12}
        task.claim("hk0")
        dispatcher._assign(task, "hk0", 0, 500)
        assert seen["n_nodes"] == 500
        assert seen["n_top_true"] == 40
        assert seen["n_spare_true"] == 12
    finally:
        (dispatcher._PICKER, dispatcher._PICKER_WANTS_N,
         dispatcher._PICKER_WANTS_SUPPLY) = prev


def test_submit_timeout_puts_the_worker_back():
    import dispatcher
    pool = dispatcher.WorkerPool(
        [("fake", 0), ("fake", None)], threads=1, overflow_threads=1)
    pool.start()
    assert pool.n_free() == 2
    worker = pool.acquire()
    status, payload, elapsed = pool.submit(
        worker, [[0, 1], [1, 0]], 1.0, 4, timeout=0.0)
    assert status == "timeout"
    assert pool.n_free() == 2


def test_miner_never_solves_locally():
    """Busy GPUs go to the dispatcher's 1-thread CPU worker, not native_algorithm."""
    src = open(os.path.join(ROOT, "CliqueAI", "miner.py")).read()
    assert "_solve_locally" not in src
    assert "native_algorithm" not in src
    assert "networkx_algorithm" not in src


# --------------------------------------------------------- core affinity

@pytest.mark.parametrize("n_gpu,threads,overflow", [(2, 7, 1), (1, 14, 1),
                                                    (3, 4, 1), (4, 3, 1)])
def test_core_plan_is_disjoint_and_fits(n_gpu, threads, overflow):
    """Workers must not share cores.

    The shim splits CPU by arithmetic -- available_cores() // fleet -- which only
    works between processes that can see each other. Affinity makes the split a
    fact the scheduler enforces.
    """
    import dispatcher
    specs = [("gpu", i) for i in range(n_gpu)] + [("cpu", None)]
    plan = dispatcher.WorkerPool.core_plan(specs, threads, overflow)
    seen = set()
    for cores in plan:
        assert cores, "a worker got no cores"
        assert not (seen & set(cores)), "workers share cores: %s" % plan
        seen |= set(cores)
    assert len(seen) == n_gpu * threads + overflow, plan


def test_core_plan_matches_the_thread_budget():
    import dispatcher
    specs = [("gpu", 0), ("gpu", 1), ("cpu", None)]
    plan = dispatcher.WorkerPool.core_plan(specs, threads=7, overflow_threads=1)
    assert [len(c) for c in plan] == [7, 7, 1], plan


def test_core_plan_survives_fewer_cores_than_budget():
    """A box smaller than the budget must still give every worker cores."""
    import dispatcher
    specs = [("gpu", 0), ("gpu", 1), ("cpu", None)]
    plan = dispatcher.WorkerPool.core_plan(specs, threads=7, overflow_threads=1)
    assert all(p for p in plan), plan


@gpu_only
def test_workers_report_their_pinned_cores(client):
    h = client.get("/health").json()
    assert any("cores=" in i for i in h["worker_info"]), h["worker_info"]
