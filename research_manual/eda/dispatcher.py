#!/usr/bin/env python3

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

BACKEND = os.environ.get("SN83_BACKEND", "gpu").lower()
N_WORKERS = int(os.environ.get("SN83_WORKERS", "2"))

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

LATENCY_S = float(os.environ.get("SN83_LATENCY_S", "2.0"))

SIBLING_WAIT_S = float(os.environ.get("SN83_SIBLING_WAIT_S", "30.0"))

TASK_TTL_S = float(os.environ.get("SN83_TASK_TTL_S", "120.0"))

SOLVE_K = int(os.environ.get("SN83_SOLVE_K", "64"))

# How many hotkeys OUR fleet holds on the subnet. The picker needs it to know
# which rivals our registrations displaced; it is not len(claims), which is
# only the siblings this round happened to query.
FLEET_N = int(os.environ.get("SN83_FLEET_N", "1"))


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
        self.done = threading.Event()
        self.pool = None
        self.error = None
        self.claims = []
        self.assigned = {}
        self.stats = {}
        self.lock = threading.Lock()
        self.owner = None

    def claim(self, hotkey):
        with self.lock:
            if hotkey not in self.claims:
                self.claims.append(hotkey)
            return self.claims.index(hotkey)


class WorkerPool(object):

    @staticmethod
    def core_plan(specs, threads, overflow_threads):
        try:
            avail = sorted(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            avail = list(range(os.cpu_count() or 1))
        plan = []
        cursor = 0
        for _kind, device in specs:
            want = threads if device is not None else overflow_threads
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
        threads = self.threads if device is not None else self.overflow_threads
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
                worker, done, box = entry
                box.append((status, payload, elapsed))
                self.free.append(worker)
            done.set()

    def acquire(self):
        with self.lock:
            if not self.free:
                return None
            self.free.sort()
            return self.free.pop(0)

    def release(self, worker):
        with self.lock:
            if worker not in self.free:
                self.free.append(worker)

    def kind_of(self, worker):
        return self.specs[worker][0]

    def submit(self, worker, matrix, budget, k, timeout):
        job_id = str(uuid_mod.uuid4())
        done = threading.Event()
        box = []
        with self.lock:
            self.pending[job_id] = (worker, done, box)
        self.req_qs[worker].put((job_id, matrix, budget, k))
        if not done.wait(timeout):
            self._reclaim(worker, job_id)
            return "timeout", None, timeout
        return box[0]

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
_SPECS = [(BACKEND, i) for i in range(N_WORKERS)]
_SPECS += [("fake" if BACKEND == "fake" else "cpu", None)
           for _ in range(N_CPU_WORKERS)]
POOL = WorkerPool(_SPECS, THREADS_PER_WORKER, OVERFLOW_THREADS)
TASKS = {}
TASKS_LOCK = threading.Lock()
COUNTERS = {"served": 0, "rejected": 0, "sibling": 0, "error": 0, "late": 0,
            "overflow_cpu": 0}


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


def _assign(task, hotkey, index, n_nodes):
    """This hotkey's answer, allocating the whole batch on first demand.

    The picker is called ONCE per round, over every sibling seen so far, and
    the result cached -- calling it per hotkey would re-run the allocation with
    a different `a` each time and hand two siblings the same clique.
    """
    with task.lock:
        if hotkey in task.assigned:
            return task.assigned[hotkey]
        q = max(len(task.claims), index + 1)
        if len(task.assigned) < q:
            answers = pick_derived.picker(
                task.pool, task.key, list(range(q)),
                n_nodes=n_nodes,
                hits=list(task.stats.get("hits", [])),
                n_top_true=task.stats.get("n_top_true", 0),
                n_spare_true=task.stats.get("n_spare_true", 0),
                fleet_n=FLEET_N,
            )
            for i, hk in enumerate(task.claims[:len(answers)]):
                task.assigned[hk] = answers[i]
            if hotkey not in task.assigned:
                task.assigned[hotkey] = answers[index % len(answers)]
        return task.assigned[hotkey]


@app.on_event("startup")
def _startup():
    if not POOL.started:
        POOL.start()


@app.get("/health")
def health():
    with TASKS_LOCK:
        inflight = sum(1 for t in TASKS.values() if not t.done.is_set())
        tracked = len(TASKS)
    return {
        "backend": BACKEND,
        "fleet_n": FLEET_N,
        "workers": N_WORKERS,
        "cpu_workers": N_CPU_WORKERS,
        "threads_per_worker": THREADS_PER_WORKER,
        "overflow_threads": OVERFLOW_THREADS,
        "core_plan": POOL.cores,
        "workers_free": POOL.n_free(),
        "worker_info": POOL.info,
        "tasks_inflight": inflight,
        "tasks_tracked": tracked,
        "counters": dict(COUNTERS),
    }


@app.post("/solve")
def solve(req: SolveRequest):
    arrived = time.monotonic()
    budget = req.time_limit - LATENCY_S
    if budget <= 0:
        COUNTERS["rejected"] += 1
        return {"clique": [], "source": "reject", "reason": "no budget"}

    task, fresh = _get_task(req.uuid)
    index = task.claim(req.hotkey)

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
            if left <= 0:
                COUNTERS["rejected"] += 1
                task.error = "no budget left after admission"
                task.done.set()
                return {"clique": [], "source": "reject", "reason": task.error}
            status, payload, elapsed = POOL.submit(
                worker, matrix, left, SOLVE_K, left + 5.0)
            handed = True
        finally:
            if not handed:
                POOL.release(worker)
        if status != "ok":
            task.error = str(payload)
            task.done.set()
            COUNTERS["error"] += 1
            return {"clique": [], "source": "error", "reason": task.error}
        task.pool, task.stats = payload
        task.done.set()
    else:
        COUNTERS["sibling"] += 1
        left = budget - (time.monotonic() - arrived)
        if not task.done.wait(min(SIBLING_WAIT_S, max(0.0, left))):
            COUNTERS["rejected"] += 1
            return {"clique": [], "source": "reject", "reason": "owner too slow"}
        if task.error:
            COUNTERS["error"] += 1
            return {"clique": [], "source": "error", "reason": task.error}

    clique = _assign(task, req.hotkey, index, req.number_of_nodes)
    elapsed = time.monotonic() - arrived
    if elapsed > budget:
        COUNTERS["late"] += 1
    COUNTERS["served"] += 1
    return {
        "clique": list(clique),
        "source": "owner" if fresh else "sibling",
        "worker": POOL.kind_of(task.owner) if task.owner is not None else None,
        "index": index,
        "siblings": len(task.claims),
        "elapsed": round(elapsed, 4),
    }
