#!/usr/bin/env python3

import asyncio
import concurrent.futures
from concurrent.futures.process import BrokenProcessPool
import logging
import multiprocessing as mp
import os
import sys
import threading
import time
import uuid as uuid_mod

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
ROOT = os.path.dirname(PARENT)
for _p in (ROOT, PARENT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fastapi import FastAPI
from pydantic import BaseModel, Field

import dispatch_worker
import pick_derived
import pick_worker

BACKEND = os.environ.get("SN83_BACKEND", "gpu").lower()
N_WORKERS = int(os.environ.get("SN83_WORKERS", "4"))

N_CPU_WORKERS = int(os.environ.get("SN83_CPU_WORKERS", "1"))

CPU_BUDGET = int(os.environ.get("SN83_CPU_BUDGET", "15"))

# The overflow worker is reserved BEFORE the GPU split, so that the 0.05% of
# rounds it serves do not cost the champion a share of its threads on the other
# 99.95%. But the reservation is clamped: asking for more overflow than the
# budget can spare used to hand the GPU workers max(1, ...) threads each and
# then exceed the quota anyway -- at SN83_CPU_BUDGET=8, SN83_WORKERS=2,
# SN83_OVERFLOW_THREADS=8 the split came out to 2*1 + 8 = 10 threads on an
# 8-thread budget. Oversubscribing is not a slowdown here: thread count changes
# which clique the solver finds, so it changes the ANSWER.
_WANT_OVERFLOW = int(os.environ.get("SN83_OVERFLOW_THREADS", "1"))
# Leave at least one thread for each GPU worker before honouring the request.
_MAX_OVERFLOW = max(0, CPU_BUDGET - N_WORKERS)
_OVERFLOW_TOTAL = min(_WANT_OVERFLOW * N_CPU_WORKERS, _MAX_OVERFLOW)
OVERFLOW_THREADS = (max(1, _OVERFLOW_TOTAL // N_CPU_WORKERS)
                    if N_CPU_WORKERS else 0)
_GPU_BUDGET = max(1, CPU_BUDGET - OVERFLOW_THREADS * N_CPU_WORKERS)
THREADS_PER_WORKER = max(1, _GPU_BUDGET // max(1, N_WORKERS))


def gpu_thread_plan(n_workers, gpu_budget):
    """Even split, then leftover cores onto the GPUs that actually run.

    acquire() walks device 0, then 1, then 2, then 3. One-deep is 81% of
    rounds, two-deep is 19%, three-deep is 0.1%. Spreading a remainder of 3
    as +1/+1/+1 would spend two extra threads on cards that almost never
    fire. +2 on gpu0 and +1 on gpu1 puts them where the work is.
    """
    if n_workers <= 0:
        return []
    base = max(1, gpu_budget // n_workers)
    leftover = max(0, gpu_budget - base * n_workers)
    plan = [base] * n_workers
    if leftover:
        take = min(2, leftover)
        plan[0] += take
        leftover -= take
        if leftover and n_workers > 1:
            plan[1] += leftover
    return plan


GPU_THREADS = gpu_thread_plan(N_WORKERS, _GPU_BUDGET)

LATENCY_S = float(os.environ.get("SN83_LATENCY_S", "2.0"))

SIBLING_WAIT_S = float(os.environ.get("SN83_SIBLING_WAIT_S", "30.0"))

TASK_TTL_S = float(os.environ.get("SN83_TASK_TTL_S", "120.0"))

# Cliques kept per level. Every simulate.py number was measured with 8192 (the
# research solver's POOL_CAP); 64 here meant production chose from a pool the
# simulator never saw, and a rich round's 2,600 omega-cliques were cut to 64.
SOLVE_K = int(os.environ.get("SN83_SOLVE_K", "8192"))

# How many hotkeys OUR fleet holds on the subnet. The picker needs it to know
# which rivals our registrations displaced; it is not len(claims), which is
# only the siblings this round happened to query.
#
# SN83_FLEET_N=auto (the default) counts the hotkeys registered under the
# coldkeys in SN83_COLDKEYS_FILE, in the current metagraph snapshot, every
# round -- so it follows registrations and the hourly refresh with no edit. An
# integer pins it (simulation, tests).
_FLEET_ENV = os.environ.get("SN83_FLEET_N", "auto").strip().lower()
FLEET_N = None if _FLEET_ENV in ("", "auto") else max(1, int(_FLEET_ENV))


def fleet_n():
    if FLEET_N is not None:
        return FLEET_N
    return pick_derived.resolve_fleet_n("auto")

# At or above this fleet size the maximin allocation replaces the derived
# picker, the same switch simulate.py exercises as --minimax-n. It gets
# whatever the owner's budget has left after the harvest, minus PICK_SAFETY_S,
# and its own watchdog falls back to the derived board when that runs out.
MINIMAX_N = int(os.environ.get("SN83_MINIMAX_N", "150"))
PICK_SAFETY_S = float(os.environ.get("SN83_PICK_SAFETY_S", "0.12"))
# OpenMP threads for the maximin search. Every core is already pinned to a
# worker; this bounds how hard the picker competes with a concurrent round's
# champion stage. Boards are identical at any thread count (checked at 1/4/12).
PICK_THREADS = int(os.environ.get("SN83_PICK_THREADS", "4"))
os.environ.setdefault("OMP_NUM_THREADS", str(PICK_THREADS))


# Per-process secret for the pool shuffle (pick_derived.shuffle_levels).
SHUFFLE_SECRET = os.urandom(32)


def use_minimax():
    return fleet_n() >= MINIMAX_N


# uvicorn's own handler, so one line per round lands in the pm2 log with no
# logging setup of our own
LOG = logging.getLogger("uvicorn.error")


class SolveRequest(BaseModel):
    uuid: str
    hotkey: str
    number_of_nodes: int
    time_limit: float
    encoded_matrix: str = ""
    adjacency_matrix: list = Field(default_factory=list)


class Task(object):

    def __init__(self, key):
        self.key = key
        self.created = time.monotonic()
        # asyncio, not threading: a waiting sibling is a suspended coroutine on
        # the event loop, not a thread. Measured with the threaded version at
        # fleet 100: two overlapping rounds hold 57 idle threads against
        # anyio's pool of 40, the next round's /solve queues for a thread, and
        # every hotkey in it misses its 2 s HTTP deadline.
        self.done = asyncio.Event()
        # one picker run per round, shared by every sibling that wakes for it
        self.pick_future = None
        self.pool = None
        self.error = None
        self.claims = []
        self.finished = []
        self.assigned = {}
        self.stats = {}
        self.lock = threading.Lock()
        self.owner = None
        # monotonic time by which the owner's answer is due; the picker's
        # deadline is measured against it
        self.deadline = None
        # per-round timing for the log line: arrival, worker done, pick done
        self.t_arrived = None
        self.t_solved = None
        self.t_picked = None
        self.picker_used = None
        # set when the picker overran and the round was answered without it
        self.emergency = False

    def claim(self, hotkey):
        with self.lock:
            if hotkey not in self.claims:
                self.claims.append(hotkey)
            return self.claims.index(hotkey)

    def finish(self, hotkey):
        with self.lock:
            if hotkey not in self.finished:
                self.finished.append(hotkey)


class WorkerPool(object):

    @staticmethod
    def core_plan(specs, threads, overflow_threads):
        try:
            avail = sorted(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            avail = list(range(os.cpu_count() or 1))
        gpu_threads = None if isinstance(threads, int) else list(threads)
        plan = []
        cursor = 0
        gi = 0
        for _kind, device in specs:
            if device is None:
                want = overflow_threads
            elif gpu_threads is None:
                want = threads
            else:
                want = gpu_threads[gi]
                gi += 1
            take = avail[cursor:cursor + want]
            if len(take) < want:
                take = avail[-want:] if want <= len(avail) else avail
            plan.append(take)
            cursor += want
        return plan

    def __init__(self, specs, threads, overflow_threads=2):
        self.specs = list(specs)
        self.n = len(self.specs)
        self.threads = threads
        self.overflow_threads = overflow_threads
        self.thread_plan = []
        gi = 0
        for _kind, device in self.specs:
            if device is None:
                self.thread_plan.append(overflow_threads)
            elif isinstance(threads, int):
                self.thread_plan.append(threads)
            else:
                self.thread_plan.append(threads[gi])
                gi += 1
        self.cores = self.core_plan(self.specs, threads, overflow_threads)
        self.ctx = mp.get_context("spawn")
        self.req_qs = []
        self.res_q = self.ctx.Queue()
        self.procs = []
        self.info = []
        self.free = list(range(self.n))
        self.lock = threading.Lock()
        self.pending = {}
        self.started = False

    def start(self):
        self.req_qs = [None] * self.n
        self.procs = [None] * self.n
        self.info = [""] * self.n
        alive = []
        for i in range(self.n):
            if self._spawn_one(i):
                alive.append(i)
        alive.sort()
        with self.lock:
            self.free = alive
        threading.Thread(target=self._collect, daemon=True).start()
        self.started = True

    def _spawn_one(self, i):
        kind, device = self.specs[i]
        threads = self.thread_plan[i]
        ready = self.ctx.Queue()
        q = self.ctx.Queue()
        p = self.ctx.Process(
            target=dispatch_worker.worker_main,
            args=(kind, device, threads, q, self.res_q, ready,
                  self.cores[i]),
            daemon=True)
        p.start()
        status, message = ready.get()
        self.req_qs[i] = q
        self.procs[i] = p
        self.info[i] = "%s: %s" % (status, message)
        return status == "ok"

    def _collect(self):
        while True:
            job_id, status, payload, elapsed = self.res_q.get()
            with self.lock:
                entry = self.pending.pop(job_id, None)
                if entry is None:
                    continue
                worker, deliver = entry
                self.free.append(worker)
            deliver((status, payload, elapsed))

    def acquire(self):
        """Lowest free GPU first (device 0, 1, 2, 3), overflow CPU last.

        Five concurrent rounds is the only way to reach the CPU worker. That
        has never been observed -- two-deep is 19%, three-deep is 0.1%, and
        five-deep is not in the 9584-round sample -- so overflow is last
        resort, not a mode.
        """
        with self.lock:
            if not self.free:
                return None

            def _order(i):
                _kind, device = self.specs[i]
                # A device index means a GPU (or fake stand-in). None is the
                # overflow worker and always sorts after every GPU.
                if device is not None:
                    return (0, device, i)
                return (1, 0, i)

            self.free.sort(key=_order)
            return self.free.pop(0)

    def release(self, worker):
        with self.lock:
            if worker not in self.free:
                self.free.append(worker)

    def kind_of(self, worker):
        return self.specs[worker][0]

    def submit(self, worker, matrix, budget, k, timeout):
        """Blocking form, for callers outside the event loop."""
        job_id = str(uuid_mod.uuid4())
        done = threading.Event()
        box = []

        def deliver(result):
            box.append(result)
            done.set()

        with self.lock:
            self.pending[job_id] = (worker, deliver)
        self.req_qs[worker].put((job_id, matrix, budget, k))
        if not done.wait(timeout):
            self._reclaim(worker, job_id)
            return "timeout", None, timeout
        return box[0]

    async def submit_async(self, worker, matrix, budget, k, timeout):
        """Same contract as submit(), without holding a thread while it waits.

        The collector thread hands the result to the loop. mp.Queue.put only
        buffers -- its feeder thread does the pickling -- so nothing here blocks
        the loop. A hung worker is respawned off the loop, because respawning
        re-warms a GPU and takes seconds.
        """
        loop = asyncio.get_running_loop()
        fut = loop.create_future()

        def _resolve(result):
            if not fut.done():
                fut.set_result(result)

        def deliver(result):
            loop.call_soon_threadsafe(_resolve, result)

        job_id = str(uuid_mod.uuid4())
        with self.lock:
            self.pending[job_id] = (worker, deliver)
        self.req_qs[worker].put((job_id, matrix, budget, k))
        try:
            return await asyncio.wait_for(fut, timeout)
        except asyncio.TimeoutError:
            loop.run_in_executor(None, self._reclaim, worker, job_id)
            return "timeout", None, timeout

    def _reclaim(self, worker, job_id):
        with self.lock:
            if self.pending.pop(job_id, None) is None:
                return
        proc = self.procs[worker]
        if proc is not None and proc.is_alive():
            proc.terminate()
            proc.join(1)
            if proc.is_alive():
                proc.kill()
                proc.join(1)
        ok = self._spawn_one(worker)
        with self.lock:
            if ok and worker not in self.free:
                self.free.append(worker)

    def n_free(self):
        with self.lock:
            return len(self.free)


app = FastAPI(title="sn83 solve dispatcher")
# GPU workers first (device 0, 1, 2, 3, ...), overflow CPU last. acquire()
# walks that order, so a free GPU is always taken before the CPU worker.
_SPECS = [(BACKEND, i) for i in range(N_WORKERS)]
_SPECS += [("fake" if BACKEND == "fake" else "cpu", None)
           for _ in range(N_CPU_WORKERS)]
POOL = WorkerPool(_SPECS, GPU_THREADS, OVERFLOW_THREADS)
TASKS = {}
TASKS_LOCK = threading.Lock()
COUNTERS = {"served": 0, "rejected": 0, "sibling": 0, "error": 0, "late": 0,
            "overflow_cpu": 0, "pick_timeout": 0, "pick_error": 0,
            "pick_derived": 0, "pick_maximin": 0, "pick_maximin_fallback": 0}
# The picker runs once per round, off the loop. In production that is a pool
# of PROCESSES (see pick_worker.py for why threads were not enough), one per
# worker so every concurrently finishing round picks at once. A thread pool is
# the in-process form: tests use it, and SN83_PICK_PROCESSES=0 selects it.
PICK_PROCESSES = os.environ.get("SN83_PICK_PROCESSES", "1") != "0"
N_PICKERS = max(1, N_WORKERS + N_CPU_WORKERS)
PICK_POOL = concurrent.futures.ThreadPoolExecutor(
    max_workers=N_PICKERS, thread_name_prefix="pick")


def _process_pick_pool():
    pool = concurrent.futures.ProcessPoolExecutor(
        max_workers=N_PICKERS, mp_context=mp.get_context("spawn"),
        initializer=pick_worker.init, initargs=(use_minimax(),))
    # every process up and its picker loaded before the first round
    list(pool.map(pick_worker.warm, range(N_PICKERS)))
    return pool


def _get_task(key):
    with TASKS_LOCK:
        now = time.monotonic()
        for k, t in list(TASKS.items()):
            if t.done.is_set() and now - t.created > TASK_TTL_S:
                del TASKS[k]
        task = TASKS.get(key)
        if task is None:
            task = TASKS[key] = Task(key)
            fresh = True
        else:
            fresh = False
        return task, fresh


def _picker():
    if use_minimax():
        import minimax
        return minimax.picker, True
    return pick_derived.picker, False


def _assign(task, hotkey, index, n_nodes):
    """This hotkey's answer, allocating the whole batch on first demand.

    The picker is called ONCE per round, over every sibling seen so far, and
    the result cached -- calling it per hotkey would re-run the allocation with
    a different `a` each time and hand two siblings the same clique.

    The picker runs OUTSIDE task.lock, so the event loop can answer the round
    from _emergency_assign() while a picker that has run past the deadline is
    still going. Whoever writes a hotkey's answer first wins; nothing is ever
    overwritten, so a hotkey that has been answered keeps that answer.
    """
    with task.lock:
        if hotkey in task.assigned:
            return task.assigned[hotkey]
        q = max(len(task.claims), index + 1)
        claims = list(task.claims)
        need = len(task.assigned) < q
    answers = None
    if need:
        fn, bounded = _picker()
        kw = {}
        if bounded:
            left = (task.deadline - time.monotonic()
                    if task.deadline is not None else 0.0)
            kw["deadline_s"] = max(0.0, left - PICK_SAFETY_S)
        mm = sys.modules.get("minimax") if bounded else None
        fell_before = mm.STATS.get("fallback", 0) if mm is not None else 0
        answers = fn(
            task.pool, task.key, list(range(q)),
            n_nodes=n_nodes,
            hits=list(task.stats.get("hits", [])),
            n_top_true=task.stats.get("n_top_true", 0),
            n_spare_true=task.stats.get("n_spare_true", 0),
            fleet_n=fleet_n(),
            **kw
        )
        if not bounded:
            used = "derived"
        else:
            mm = sys.modules.get("minimax")
            fell = mm is not None and mm.STATS.get("fallback", 0) > fell_before
            used = "maximin->derived" if fell else "maximin"
    with task.lock:
        if answers is not None and not task.emergency:
            task.t_picked = time.monotonic()
            task.picker_used = used
            for i, hk in enumerate(claims[:len(answers)]):
                task.assigned.setdefault(hk, answers[i])
            if hotkey not in task.assigned:
                task.assigned[hotkey] = answers[index % len(answers)]
        if hotkey not in task.assigned:
            _fill_spread(task)
        return task.assigned[hotkey]


def _fill_spread(task):
    """Cheap allocation: distinct cliques, omega first, then omega-1, cycling.

    Caller holds task.lock. Milliseconds on a full 8192 pool, so it can run on
    the event loop.
    """
    omega, top, spare = pick_derived._levels(task.pool)
    ordered = [list(c) for c in top] + [list(c) for c in spare]
    for i, hk in enumerate(task.claims):
        task.assigned.setdefault(hk, ordered[i % len(ordered)])


def _emergency_assign(task):
    """Answer the round now: the picker has run past the owner's deadline.

    Measured at fleet 160 with 60 siblings: the solve is deadline-bounded, but
    under CPU contention (every core pinned to a worker, 60+ miner processes
    waking at once) the picker took 0.65-1.45 s instead of 0.37 s and the whole
    round answered at 2.4-3.2 s -- past the miners' wait, so every hotkey fell
    back to a greedy clique.
    """
    with task.lock:
        if task.emergency:
            return
        task.emergency = True
        _fill_spread(task)
        task.t_picked = time.monotonic()
        task.picker_used = "emergency-spread"
    COUNTERS["pick_timeout"] += 1


def _fill_unused(task, hotkey):
    """A late sibling's answer: a clique nobody in this round holds yet, omega
    first. Caller holds task.lock."""
    if hotkey in task.assigned:
        return
    omega, top, spare = pick_derived._levels(task.pool)
    ordered = [list(c) for c in top] + [list(c) for c in spare]
    held = {tuple(sorted(c)) for c in task.assigned.values()}
    for c in ordered:
        if tuple(sorted(c)) not in held:
            task.assigned[hotkey] = c
            return
    task.assigned[hotkey] = ordered[len(task.assigned) % len(ordered)]


def _apply_pick(task, claims, fut):
    """Picker-process result -> assignments, unless the round was already
    answered without it. Runs on the event loop."""
    global PICK_POOL
    try:
        answers, used = fut.result()
    except Exception:                               # noqa: BLE001
        COUNTERS["pick_error"] += 1
        if isinstance(fut.exception(), BrokenProcessPool):
            # a picker process died; the next round gets a fresh pool
            PICK_POOL = _process_pick_pool()
        LOG.exception("picker failed for %s", task.key)
        return
    COUNTERS[{"derived": "pick_derived", "maximin": "pick_maximin"}.get(
        used, "pick_maximin_fallback")] += 1
    with task.lock:
        if task.emergency:
            return
        task.t_picked = time.monotonic()
        task.picker_used = used
        for i, hk in enumerate(claims[:len(answers)]):
            task.assigned.setdefault(hk, answers[i])


def _start_pick(task, hotkey, index, n_nodes, loop):
    if isinstance(PICK_POOL, concurrent.futures.ProcessPoolExecutor):
        with task.lock:
            claims = list(task.claims)
        q = max(len(claims), index + 1)
        fut = loop.run_in_executor(
            PICK_POOL, pick_worker.pick, task.pool, task.key, q, n_nodes,
            list(task.stats.get("hits", [])), task.stats.get("n_top_true", 0),
            task.stats.get("n_spare_true", 0), fleet_n(), task.deadline,
            use_minimax(), PICK_SAFETY_S)
        fut.add_done_callback(lambda f: _apply_pick(task, claims, f))
        return fut
    return loop.run_in_executor(PICK_POOL, _assign, task, hotkey, index, n_nodes)


async def _assign_async(task, hotkey, index, n_nodes):
    """One pick per round, shared by every sibling that wakes for it.

    Bounded by the owner's deadline: past it -- or if the picker fails -- the
    round is answered from _emergency_assign() and a late pick is discarded.
    """
    if hotkey in task.assigned:
        return task.assigned[hotkey]
    loop = asyncio.get_running_loop()
    if task.pick_future is None:
        task.pick_future = _start_pick(task, hotkey, index, n_nodes, loop)
    wait = (None if task.deadline is None
            else max(0.0, task.deadline - time.monotonic()))
    try:
        if not task.pick_future.done():
            await asyncio.wait_for(asyncio.shield(task.pick_future), wait)
        else:
            task.pick_future.result()
    except asyncio.TimeoutError:
        _emergency_assign(task)
    except Exception:                               # noqa: BLE001
        _emergency_assign(task)
    await asyncio.sleep(0)          # let _apply_pick's done-callback run
    with task.lock:
        if hotkey not in task.assigned:
            if task.emergency:
                _fill_spread(task)
            _fill_unused(task, hotkey)
        return task.assigned[hotkey]


@app.on_event("startup")
def _startup():
    global PICK_POOL
    if use_minimax():
        # builds libminimax.so if needed (g++, ~2 s) -- never inside a round
        import minimax  # noqa: F401
    if PICK_PROCESSES and not isinstance(
            PICK_POOL, concurrent.futures.ProcessPoolExecutor):
        PICK_POOL = _process_pick_pool()
    if not POOL.started:
        POOL.start()


@app.get("/task/{uuid}")
async def task_started(uuid: str):
    with TASKS_LOCK:
        task = TASKS.get(uuid)
    if task is None:
        return {"uuid": uuid, "claims": 0, "done": False}
    with task.lock:
        return {
            "uuid": uuid,
            "claims": len(task.claims),
            "finished": len(task.finished),
            "done": task.done.is_set(),
        }


@app.get("/health")
async def health():
    with TASKS_LOCK:
        inflight = sum(1 for t in TASKS.values() if not t.done.is_set())
        tracked = len(TASKS)
    return {
        "backend": BACKEND,
        "fleet_n": fleet_n(),
        "fleet_n_source": ("pinned" if FLEET_N is not None else
                           "auto: %d coldkeys in %s" % (
                               len(pick_derived.our_coldkeys()),
                               pick_derived.COLDKEYS_FILE or "(SN83_COLDKEYS_FILE unset)")),
        "workers": N_WORKERS,
        "cpu_workers": N_CPU_WORKERS,
        "threads_per_worker": THREADS_PER_WORKER,
        "gpu_threads": list(GPU_THREADS),
        "overflow_threads": OVERFLOW_THREADS,
        "core_plan": POOL.cores,
        "workers_free": POOL.n_free(),
        "worker_info": POOL.info,
        "tasks_inflight": inflight,
        "tasks_tracked": tracked,
        "counters": dict(COUNTERS),
        "solve_k": SOLVE_K,
        "picker": "minimax" if use_minimax() else "derived",
        "minimax_n": MINIMAX_N,
        "picker_pool": type(PICK_POOL).__name__,
    }


@app.post("/solve")
async def solve(req: SolveRequest):
    arrived = time.monotonic()
    budget = req.time_limit - LATENCY_S
    if budget <= 0:
        COUNTERS["rejected"] += 1
        return {"clique": [], "source": "reject", "reason": "no budget"}

    task, fresh = _get_task(req.uuid)
    index = task.claim(req.hotkey)
    try:
        return await _solve_claimed(req, task, fresh, index, arrived, budget)
    finally:
        task.finish(req.hotkey)


async def _solve_claimed(req, task, fresh, index, arrived, budget):
    if fresh:
        worker = POOL.acquire()
        if worker is None:
            with TASKS_LOCK:
                TASKS.pop(req.uuid, None)
            COUNTERS["rejected"] += 1
            return {"clique": [], "source": "reject", "reason": "all workers busy"}
        handed = False
        try:
            task.owner = worker
            if POOL.kind_of(worker) != BACKEND:
                COUNTERS["overflow_cpu"] += 1
            # The base92 string goes to the worker UNDECODED. Decoding an
            # n=894 matrix is 0.109s of pure Python, and doing it here runs it
            # on the HTTP thread: two concurrent rounds then serialise on the
            # GIL and each pays about double, which is most of the reason a
            # two-deep round used to answer past its budget.
            matrix = req.encoded_matrix or req.adjacency_matrix
            if not matrix:
                raise ValueError("neither encoded_matrix nor adjacency_matrix")
            # Anchor the worker's deadline to when the REQUEST arrived, not to
            # when the worker happens to start. Everything spent getting here --
            # parsing, admission, queueing -- is time already gone from the
            # round, and handing the worker the full budget anyway is what made
            # the answer land after it.
            left = budget - (time.monotonic() - arrived)
            task.deadline = arrived + budget
            if left <= 0:
                COUNTERS["rejected"] += 1
                task.error = "no budget left after admission"
                task.done.set()
                return {"clique": [], "source": "reject", "reason": task.error}
            task.t_arrived = arrived
            status, payload, elapsed = await POOL.submit_async(
                worker, matrix, left, SOLVE_K, left + 5.0)
            task.t_solved = time.monotonic()
            handed = True
        finally:
            if not handed:
                POOL.release(worker)
        if status != "ok":
            task.error = str(payload)
            task.done.set()
            COUNTERS["error"] += 1
            return {"clique": [], "source": "error", "reason": task.error}
        pool, stats = payload
        # Which cliques we submit must not be computable from outside: the
        # shuffle is keyed with a secret only this process knows. Every sibling
        # of the round reads this same order, so picks stay consistent.
        pool, hits = pick_derived.shuffle_levels(
            pool, stats.get("hits", []), SHUFFLE_SECRET + req.uuid.encode())
        task.pool, task.stats = pool, dict(stats, hits=hits)
        task.done.set()
    else:
        COUNTERS["sibling"] += 1
        left = budget - (time.monotonic() - arrived)
        if not task.done.is_set():
            try:
                await asyncio.wait_for(task.done.wait(),
                                       min(SIBLING_WAIT_S, max(0.0, left)))
            except asyncio.TimeoutError:
                pass
        if not task.done.is_set():
            COUNTERS["rejected"] += 1
            return {"clique": [], "source": "reject", "reason": "owner too slow"}
        if task.error:
            COUNTERS["error"] += 1
            return {"clique": [], "source": "error", "reason": task.error}

    clique = await _assign_async(task, req.hotkey, index, req.number_of_nodes)
    elapsed = time.monotonic() - arrived
    if elapsed > budget:
        COUNTERS["late"] += 1
    if fresh and task.t_solved is not None:
        LOG.info(
            "round uuid=%s n=%d worker=%s siblings=%d solve=%.3fs pick=%.3fs "
            "picker=%s pool=%d answered=%.3fs%s", req.uuid, req.number_of_nodes,
            POOL.specs[task.owner][1] if task.owner is not None else None,
            len(task.claims), task.t_solved - arrived,
            (task.t_picked or task.t_solved) - task.t_solved, task.picker_used,
            len(task.pool or []), elapsed, " LATE" if elapsed > budget else "")
    COUNTERS["served"] += 1
    return {
        "clique": list(clique),
        "source": "owner" if fresh else "sibling",
        "worker": POOL.kind_of(task.owner) if task.owner is not None else None,
        "index": index,
        "siblings": len(task.claims),
        "elapsed": round(elapsed, 4),
    }
