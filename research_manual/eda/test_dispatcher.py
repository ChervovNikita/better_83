#!/usr/bin/env python3
"""Tests for the solve dispatcher.

No GPU:
    SN83_BACKEND=fake .venv/bin/pytest research_manual/eda/test_dispatcher.py -v

On the 4xGPU pod, the same file against real workers:
    SN83_BACKEND=gpu SN83_WORKERS=4 \
      .venv/bin/pytest research_manual/eda/test_dispatcher.py -v -m "not fake_only"

The GPU-only checks (deadlines, no-late-under-overlap, four devices actually in
use) are marked `gpu_only` and skip without one.
"""

import inspect
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
    assert h["workers"] == int(os.environ.get("SN83_WORKERS", "4"))
    assert h["workers_free"] >= 1, h["worker_info"]
    # the CPU budget must be split, never handed to each worker in full
    budget = int(os.environ.get("SN83_CPU_BUDGET", "15"))
    assert h["threads_per_worker"] * h["workers"] <= budget
    assert "gpu_threads" in h
    assert sum(h["gpu_threads"]) + h["overflow_threads"] * h["cpu_workers"] <= budget


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


def test_task_reports_who_has_started_waiting(client):
    """claim() is on arrival; finish() is when that /solve returns."""
    client.post("/solve", json=_req("t-started", "hk0"))
    one = client.get("/task/t-started").json()
    assert one["claims"] == 1
    assert one["finished"] == 1
    assert one["done"] is True
    client.post("/solve", json=_req("t-started", "hk1"))
    two = client.get("/task/t-started").json()
    assert two["claims"] == 2
    assert two["finished"] == 2


def test_late_sibling_still_served_from_the_same_pool(client):
    first = client.post("/solve", json=_req("t-late", "hk0")).json()
    later = client.post("/solve", json=_req("t-late", "hk9")).json()
    assert first["source"] == "owner"
    assert later["source"] == "sibling"
    assert later["clique"]


@fake_only
def test_waiting_siblings_do_not_starve_the_next_round(client, monkeypatch):
    """Waiting siblings are coroutines, not threads.

    Threaded, every sibling held one of anyio's 40 threads while the owner
    solved. Measured at fleet 100 (up to 32 siblings per round), two
    overlapping rounds exhausted the pool, the next round's /solve queued for a
    thread, and every hotkey in it missed its 2 s HTTP deadline.
    """
    import dispatcher
    real = dispatcher.POOL.submit_async

    async def slow(worker, matrix, budget, k, timeout):
        if k == dispatcher.SOLVE_K and slow.first:
            slow.first = False
            import asyncio
            await asyncio.sleep(1.5)
        return await real(worker, matrix, budget, k, timeout)
    slow.first = True
    monkeypatch.setattr(dispatcher.POOL, "submit_async", slow)

    n_sib = 120
    out = {}

    def sib(i):
        out[i] = client.post("/solve", json=_req("t-starve-a", "hk%d" % i,
                                                 n=40, tl=8.0)).json()

    threads = [threading.Thread(target=sib, args=(i,)) for i in range(n_sib)]
    for t in threads:
        t.start()
    time.sleep(0.4)            # round A's siblings are all waiting now
    t0 = time.monotonic()
    other = client.post("/solve", json=_req("t-starve-b", "hk0", n=40, tl=8.0)).json()
    took = time.monotonic() - t0
    for t in threads:
        t.join()

    assert other["source"] == "owner", other
    assert took < 0.8, "round B waited behind round A's siblings: %.2fs" % took
    assert all(r["clique"] for r in out.values())
    assert sum(r["source"] == "owner" for r in out.values()) == 1


def test_async_client_returns_none_when_the_service_is_down():
    import asyncio
    import dispatch_client
    got = asyncio.run(dispatch_client.solve_source_async(
        "u", "hk", 100, _matrix(20), 4.0, url="http://127.0.0.1:1", timeout=1.0))
    assert got == (None, None)


# ----------------------------------------------------------- admission

def test_rejects_rather_than_queues_when_all_workers_busy(client):
    """More concurrent TASKS than workers must refuse, not wait.

    Queuing spends the round's deadline and then answers late, which scores zero
    on both terms; refusing leaves the caller its whole budget for a local
    fallback.
    """
    n_task = int(os.environ.get("SN83_WORKERS", "4")) + 3
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
    """One worker per card -- not four workers sharing device 0."""
    import subprocess
    q = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid",
         "--format=csv,noheader"], capture_output=True, text=True)
    devices = {line.split(",")[-1].strip() for line in q.stdout.splitlines()
               if line.strip()}
    want = int(os.environ.get("SN83_WORKERS", "4"))
    assert len(devices) >= want, "workers are not on separate devices: %r" % (
        q.stdout,)


@fake_only
def test_fake_backend_pool_is_deterministic_but_picks_are_not(client):
    """The fake pool is identical for every round; which of its cliques a round
    submits is a keyed shuffle, so two rounds pick differently."""
    picks = {}
    for u in ("t-det-a", "t-det-b"):
        out = {}

        def go(i, u=u):
            out[i] = client.post("/solve", json=_req(u, "hk%d" % i)).json()
        threads = [threading.Thread(target=go, args=(i,)) for i in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        picks[u] = {tuple(sorted(r["clique"])) for r in out.values()}
    assert picks["t-det-a"] != picks["t-det-b"]
    again = client.post("/solve", json=_req("t-det-a", "hk0")).json()
    assert tuple(sorted(again["clique"])) in picks["t-det-a"]


def test_shuffle_levels_keeps_levels_and_hits_aligned():
    import pick_derived as pd
    pool = [[0, 1, 2, i] for i in range(3, 40)] + [[0, 1, i] for i in range(40, 70)]
    hits = list(range(len(pool)))
    hit_of = {tuple(c): h for c, h in zip(pool, hits)}
    a, ha = pd.shuffle_levels(pool, hits, b"k|round-1")
    b, _ = pd.shuffle_levels(pool, hits, b"k|round-1")
    c, _ = pd.shuffle_levels(pool, hits, b"k|round-2")
    assert sorted(map(tuple, a)) == sorted(map(tuple, pool))       # same cliques
    assert [len(x) for x in a] == [len(x) for x in pool]            # omega first
    assert all(hit_of[tuple(x)] == h for x, h in zip(a, ha))        # hits aligned
    assert a == b                                                   # keyed
    assert a[:10] != c[:10]                                         # per round
    assert a[:10] != pool[:10]                                      # not id order


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


# --------------------------------------------------- acquire order

def test_acquire_prefers_gpus_in_device_order_then_cpu():
    """gpu0, then gpu1, then gpu2, then gpu3, and only then the CPU worker.

    The free list is deliberately scrambled so a leftover-from-release order
    cannot masquerade as the policy.
    """
    import dispatcher
    pool = dispatcher.WorkerPool(
        [("gpu", 0), ("gpu", 1), ("gpu", 2), ("gpu", 3), ("cpu", None)],
        threads=1, overflow_threads=1)
    pool.free = [4, 2, 0, 3, 1]
    assert [pool.acquire() for _ in range(5)] == [0, 1, 2, 3, 4]
    assert pool.acquire() is None


# --------------------------------------------------- CPU overflow worker

def test_cpu_worker_absorbs_overflow_instead_of_rejecting(client):
    """A reject sends every hotkey off to solve in its own process at once.

    Production runs one process per hotkey and each sizes its own thread pool,
    so N rejects means N local solves against a 15-CPU quota, at exactly the
    moment the GPU workers need CPU for their champion stage. The overflow
    worker turns the (N_WORKERS + 1)-th concurrent task into one shared CPU
    solve instead.
    """
    n_gpu = int(os.environ.get("SN83_WORKERS", "4"))
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
    gpu_total = sum(h["gpu_threads"])
    total = gpu_total + h["overflow_threads"] * h["cpu_workers"]
    assert total <= budget, "oversubscribed: %s" % h
    assert gpu_total > budget // 2, (
        "GPU workers were starved to reserve for overflow: %s" % h)
    assert h["overflow_threads"] >= 1, h


def test_health_reports_the_overflow_counter(client):
    h = client.get("/health").json()
    assert "overflow_cpu" in h["counters"]


@pytest.mark.parametrize("workers,budget", [(1, 15), (2, 15), (3, 15), (2, 8),
                                            (4, 32), (4, 20)])
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
    total = (sum(mod.GPU_THREADS)
             + mod.OVERFLOW_THREADS * mod.N_CPU_WORKERS)
    assert total <= budget, (workers, budget, mod.GPU_THREADS,
                             mod.OVERFLOW_THREADS)
    assert mod.THREADS_PER_WORKER >= 1 and mod.OVERFLOW_THREADS >= 1
    if workers == 4 and budget == 20:
        # This box: leftover 3 of 19 GPU cores go +2 to gpu0, +1 to gpu1.
        assert mod.GPU_THREADS == [6, 5, 4, 4]
        assert mod.OVERFLOW_THREADS == 1


# --------------------------------------------------- parent must not touch CUDA

def test_parent_picker_does_not_import_the_gpu_solver():
    """The HTTP process only picks from a pool the workers already built.

    Importing solver.py used to load fleet_solver_gpu, which opens a CUDA
    context on device 0 in this process -- next to worker 0's harvest.
    """
    import dispatcher
    import pick_derived
    assert dispatcher.pick_derived is pick_derived
    sig = inspect.signature(pick_derived.picker).parameters
    # The three inputs the dispatcher observes that a lone miner cannot: the
    # graph's difficulty, the pre-truncation supply, and our own fleet size.
    for name in ("n_nodes", "n_top_true", "n_spare_true", "fleet_n"):
        assert name in sig, name
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
    assert "dispatch_client.solve_source" in src


def test_miner_caps_the_solver_at_two_seconds():
    src = open(os.path.join(ROOT, "CliqueAI", "miner.py")).read()
    assert "SOLVE_CAP_S" in src
    assert 'os.environ.get("SN83_SOLVE_CAP_S", "2.0")' in src
    assert "min(timeout - reserve, MAX_HOLD_S, SOLVE_CAP_S)" in src


def test_miner_waits_past_the_solve_budget_by_a_grace():
    """The dispatcher's budget stays the solve budget; only the wait is longer.

    With both at exactly `budget`, an answer finished at 2.01 s was discarded for
    a greedy local clique -- 84 of 3,501 answers at fleet 160.
    """
    src = open(os.path.join(ROOT, "CliqueAI", "miner.py")).read()
    assert 'os.environ.get("SN83_HTTP_GRACE_S", "0.5")' in src
    assert "timeout=budget + DISPATCH_GRACE_S" in src
    assert "budget + DISPATCH_LATENCY_S" in src     # what the dispatcher is told


def test_miner_starts_cpu_followup_after_rare_gather():
    """The follow-up is a CPU thread, gated on the binomial left tail."""
    src = open(os.path.join(ROOT, "CliqueAI", "miner.py")).read()
    assert "_gather_watch" in src
    assert "_cpu_followup" in src
    assert "threading.Thread" in src
    assert "fleet_solver.solve_many" in src
    assert "p_at_most_queried" in src
    assert "RARE_QUERY_MIN_TL_S" in src
    for gpu_module in ("fleet_solver_gpu", "gpu_lib"):
        assert gpu_module not in src, gpu_module
    assert "started_waiting" in src
    assert "_wait_until_finished" in src
    assert "_wait_for_gather" not in src
    assert src.index("dispatch_client.solve") < src.index("self._gather_watch")
    assert src.index("def _cpu_followup") > src.index("def _gather_watch")


def test_p_at_most_queried_is_the_binomial_left_tail():
    import pick_derived
    p = pick_derived.selection_p(0.7)
    # one-hotkey fleet that was queried: never rare
    assert pick_derived.p_at_most_queried(1, 1, 0.7) == 1.0
    # P(X=0) = (1-p)^n
    got = pick_derived.p_at_most_queried(0, 10, 0.7)
    assert abs(got - (1.0 - p) ** 10) < 1e-12
    # 20 hotkeys, D=0.7, only one claimed is a left-tail event
    assert pick_derived.p_at_most_queried(1, 20, 0.7) < 0.05
    assert pick_derived.p_at_most_queried(20, 20, 0.7) == 1.0


def test_picker_rereads_a_refreshed_metagraph(tmp_path, monkeypatch):
    """refresh_metagraph.sh rewrites the file under a running dispatcher.

    The picker used to cache the first snapshot for the life of the process, so
    the hourly refresh never reached it.
    """
    import json
    import pick_derived as pd
    src = json.load(open(pd.METAGRAPH))
    path = tmp_path / "metagraph.json"
    path.write_text(json.dumps(src))
    monkeypatch.setattr(pd, "METAGRAPH", str(path))
    before = dict(pd.fleet_profile(10))
    assert pd._load_metagraph()["block"] == src["block"]

    fresh = dict(src, block=src["block"] + 68000, miners=src["miners"][:50])
    tmp = tmp_path / "metagraph.json.tmp"
    tmp.write_text(json.dumps(fresh))
    os.replace(tmp, path)
    st = os.stat(path)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))

    assert pd._load_metagraph()["block"] == src["block"] + 68000
    after = dict(pd.fleet_profile(10))
    assert sum(after.values()) < sum(before.values())
    monkeypatch.setattr(pd, "METAGRAPH", pd.paths.METAGRAPH_JSON)
    pd._load_metagraph()


def test_fleet_size_is_counted_from_our_coldkeys(tmp_path, monkeypatch):
    """SN83_FLEET_N=auto: hotkeys registered under the listed coldkeys.

    And in a live snapshot our own hotkeys are the ones taken out of the rival
    field -- not the fleet_n weakest, which our registrations already displaced.
    """
    import json
    import pick_derived as pd
    src = json.load(open(pd.METAGRAPH))
    ours = sorted({m["coldkey"] for m in src["miners"]})[:2]
    want = {m["hotkey"] for m in src["miners"] if m["coldkey"] in ours}
    meta_path = tmp_path / "metagraph.json"
    meta_path.write_text(json.dumps(src))
    keys = tmp_path / "coldkeys.txt"
    keys.write_text("# ours\n%s\n\n%s  # second\n" % (ours[0], ours[1]))
    monkeypatch.setattr(pd, "METAGRAPH", str(meta_path))
    monkeypatch.setattr(pd, "COLDKEYS_FILE", str(keys))

    assert pd.our_coldkeys() == frozenset(ours)
    assert pd.fleet_n_auto() == len(want)
    assert pd.resolve_fleet_n("auto") == max(1, len(want))
    assert pd.resolve_fleet_n("7") == 7              # an integer still pins it
    assert pd.victim_hotkeys(len(want)) == want
    rivals = sum(pd.fleet_profile(len(want)).values())
    assert rivals == len(src["miners"]) - len(want)

    monkeypatch.setattr(pd, "COLDKEYS_FILE", "")     # replay: weakest displaced
    assert pd.resolve_fleet_n("auto") == 1
    assert len(pd.victim_hotkeys(5)) == 5
    monkeypatch.setattr(pd, "METAGRAPH", pd.paths.METAGRAPH_JSON)
    pd._load_metagraph()


def test_assign_forwards_harvest_supply():
    """n_top_true is the pre-truncation count, and it has to reach the picker.

    The truncated len(pool) inflates the crowding estimate by the truncation
    factor, so passing the wrong one silently changes every allocation.
    """
    import dispatcher
    seen = {}

    def fake_picker(pool, uuid, hotkeys, **kwargs):
        seen.update(kwargs)
        return [list(pool[0])] * len(hotkeys)

    prev = dispatcher.pick_derived.picker
    dispatcher.pick_derived.picker = fake_picker
    try:
        task = dispatcher.Task("u-supply")
        task.pool = [[0, 1, 2], [0, 1]]
        task.stats = {"n_top_true": 40, "n_spare_true": 12, "hits": [9, 4]}
        task.claim("hk0")
        dispatcher._assign(task, "hk0", 0, 500)
        assert seen["n_nodes"] == 500
        assert seen["n_top_true"] == 40
        assert seen["n_spare_true"] == 12
        assert seen["hits"] == [9, 4]
        assert seen["fleet_n"] == dispatcher.fleet_n()
    finally:
        dispatcher.pick_derived.picker = prev


def test_assign_switches_to_maximin_at_the_fleet_threshold(monkeypatch):
    """Below SN83_MINIMAX_N the derived picker runs unbounded; at or above it
    maximin runs with what the owner's budget has left."""
    import types
    import dispatcher
    calls = []

    def derived(pool, uuid, hotkeys, **kw):
        calls.append(("derived", kw))
        return [list(pool[0])] * len(hotkeys)

    def maximin(pool, uuid, hotkeys, **kw):
        calls.append(("maximin", kw))
        return [list(pool[0])] * len(hotkeys)

    monkeypatch.setattr(dispatcher.pick_derived, "picker", derived)
    monkeypatch.setitem(sys.modules, "minimax",
                        types.SimpleNamespace(picker=maximin, STATS={}))
    monkeypatch.setattr(dispatcher, "MINIMAX_N", 150)
    for fleet, want in ((149, "derived"), (150, "maximin")):
        monkeypatch.setattr(dispatcher, "FLEET_N", fleet)
        task = dispatcher.Task("u-switch-%d" % fleet)
        task.pool = [[0, 1, 2], [0, 1]]
        task.stats = {"n_top_true": 1, "n_spare_true": 1, "hits": [1, 1]}
        task.deadline = time.monotonic() + 0.5
        task.claim("hk0")
        dispatcher._assign(task, "hk0", 0, 500)
        name, kw = calls[-1]
        assert name == want, (fleet, calls)
        if want == "maximin":
            assert 0.0 < kw["deadline_s"] <= 0.5 - dispatcher.PICK_SAFETY_S
        else:
            assert "deadline_s" not in kw


def test_overrunning_picker_is_cut_off_at_the_deadline(monkeypatch):
    """The picker is the one stage without its own clock.

    Under CPU contention at fleet 160 it took 0.65-1.45 s and the round answered
    at 2.4-3.2 s. Past the owner's deadline the round must be answered from the
    cheap spread instead, and the picker's late result must not replace it.
    """
    import asyncio
    import dispatcher

    def slow_picker(pool, uuid, hotkeys, **kw):
        time.sleep(2.0)
        return [list(pool[0])] * len(hotkeys)     # would repeat one clique

    monkeypatch.setattr(dispatcher.pick_derived, "picker", slow_picker)
    monkeypatch.setattr(dispatcher, "FLEET_N", 1)
    # in-process picker, so the stub above is the one that runs
    import concurrent.futures
    monkeypatch.setattr(dispatcher, "PICK_POOL",
                        concurrent.futures.ThreadPoolExecutor(max_workers=2))
    task = dispatcher.Task("u-overrun")
    task.pool = [[0, 1, 2, i] for i in range(3, 9)] + [[0, 1, i] for i in range(10, 14)]
    task.stats = {"n_top_true": 6, "n_spare_true": 4, "hits": []}
    for i in range(5):
        task.claim("hk%d" % i)
    task.deadline = time.monotonic() + 0.3
    before = dispatcher.COUNTERS["pick_timeout"]

    async def run():
        return await asyncio.gather(*[
            dispatcher._assign_async(task, "hk%d" % i, i, 300) for i in range(5)])

    t0 = time.monotonic()
    got = asyncio.run(run())
    took = time.monotonic() - t0
    assert took < 0.6, "siblings waited for the overrunning picker: %.2fs" % took
    assert len({tuple(c) for c in got}) == 5, got
    assert all(len(c) == 4 for c in got), "spread must use omega cliques first"
    assert dispatcher.COUNTERS["pick_timeout"] == before + 1
    assert task.picker_used == "emergency-spread"
    time.sleep(2.2)                                # let the picker thread finish
    assert [task.assigned["hk%d" % i] for i in range(5)] == got


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


def test_miner_falls_back_on_the_cpu_only():
    """A BUSY dispatcher is served by its CPU overflow worker; a DOWN one is not.

    This replaces an assertion that `_solve_locally` must not appear in
    miner.py at all. That assertion and DISPATCHER.md arrived in the same
    commit (838ad8b) contradicting each other -- the doc names _solve_locally
    as the kept fallback and gives the reason ("a dispatcher restart costs a
    worse clique rather than a missed round, which would score zero"), while
    the test forbids it. It was never green: miner.py at that commit called
    native_algorithm, which the test also forbids.

    The doc wins. The test's rationale covers only the reject path, where the
    dispatcher is alive and its 1-thread overflow worker answers; it says
    nothing about the dispatcher being unreachable, where no fallback means an
    empty answer and a hard zero on both reward terms for every round of the
    outage. What actually has to hold is narrower:

      - the miner never opens a CUDA context of its own (the workers own the
        devices; a second context makes both solves miss the deadline),
      - the old unshared local paths stay gone,
      - dispatch is tried first, with no switch to turn it off.
    """
    src = open(os.path.join(ROOT, "CliqueAI", "miner.py")).read()
    assert "native_algorithm" not in src
    assert "networkx_algorithm" not in src
    for gpu_module in ("fleet_solver_gpu", "gpu_lib", "solver_gpu"):
        assert gpu_module not in src, gpu_module
    # the fallback exists, and it is reached only after dispatch returned nothing
    assert "_solve_locally" in src
    assert src.index("dispatch_client.solve") < src.index("self._solve_locally")


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


@pytest.mark.parametrize("n_gpu,threads,overflow", [(2, 7, 1), (4, 4, 1)])
def test_core_plan_matches_the_thread_budget(n_gpu, threads, overflow):
    import dispatcher
    specs = [("gpu", i) for i in range(n_gpu)] + [("cpu", None)]
    plan = dispatcher.WorkerPool.core_plan(specs, threads=threads,
                                           overflow_threads=overflow)
    want = [threads] * n_gpu if isinstance(threads, int) else list(threads)
    assert [len(c) for c in plan] == want + [overflow], plan


def test_core_plan_honours_uneven_gpu_threads():
    """Leftover cores pin to gpu0 and gpu1, not shared across the pool."""
    import dispatcher
    specs = [("gpu", i) for i in range(4)] + [("cpu", None)]
    plan = dispatcher.WorkerPool.core_plan(specs, threads=[6, 5, 4, 4],
                                           overflow_threads=1)
    assert [len(c) for c in plan] == [6, 5, 4, 4, 1], plan
    seen = set()
    for cores in plan:
        assert not (seen & set(cores)), "workers share cores: %s" % plan
        seen |= set(cores)


def test_gpu_thread_plan_gives_leftovers_to_the_busy_cards():
    import dispatcher
    assert dispatcher.gpu_thread_plan(4, 16) == [4, 4, 4, 4]
    assert dispatcher.gpu_thread_plan(4, 19) == [6, 5, 4, 4]
    assert dispatcher.gpu_thread_plan(4, 18) == [6, 4, 4, 4]
    assert dispatcher.gpu_thread_plan(4, 17) == [5, 4, 4, 4]
    assert dispatcher.gpu_thread_plan(2, 15) == [8, 7]


def test_core_plan_survives_fewer_cores_than_budget():
    """A box smaller than the budget must still give every worker cores."""
    import dispatcher
    specs = [("gpu", i) for i in range(4)] + [("cpu", None)]
    plan = dispatcher.WorkerPool.core_plan(specs, threads=4, overflow_threads=1)
    assert all(p for p in plan), plan


@gpu_only
def test_workers_report_their_pinned_cores(client):
    h = client.get("/health").json()
    assert any("cores=" in i for i in h["worker_info"]), h["worker_info"]
