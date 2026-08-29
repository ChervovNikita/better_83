# Solve dispatcher — file inventory and run recipe

Everything added or changed to move from "simulator only" to "answers in
production", with one shared GPU service behind a fleet of miner hotkeys.

## Why it exists

Two measured problems, one service.

**Concurrency.** `harvest_kernel` is persistent and sized to fill the card, so a
second launch queues behind the first. Measured on one A4000: two simultaneous
solves each took ~2× wall time and **both missed the deadline**, which scores
zero on both reward terms.

```
round              mode   elapsed   budget
n=892 tl=7.5      alone      5.54     5.50
n=892 tl=7.5    concur0     10.21     5.50   <- late
n=294 tl=30       alone     28.01    28.00
n=294 tl=30     concur0     51.82    28.00   <- late
```

Over **87.5 h / 9584 logged rounds** (`dump_timing.py`):

```
simultaneous solves      rounds    share
1                          7766    81.0%
2                          1813    18.9%
3                             5     0.1%
```

So 2 GPUs cover 99.9%, and the 19% two-deep case is routine, not an edge case.

**Sibling allocation.** In production each hotkey is its own miner process, so it
sees `q = 1` and cannot know how many siblings were queried — yet the picker's
whole omega / omega-1 decision is a function of `q`. Routing every hotkey through
one service turns `q` into an observation: the claims count the siblings.

## New files

| file | role |
|---|---|
| `dispatcher.py` | FastAPI service. `POST /solve`, `GET /health`. Owns the worker pool, batches siblings per uuid, admission control. |
| `dispatch_worker.py` | One process per device, pinned with `CUDA_VISIBLE_DEVICES`. Warms nvcc + CUDA context at startup. Backends: `gpu`, `cpu`, `fake`. |
| `dispatch_client.py` | Miner-side client. Returns `None` on **any** failure so the caller falls back locally; nothing here may raise into the request path. |
| `test_dispatcher.py` | 25 tests. 21 run with no GPU (`SN83_BACKEND=fake`); 4 are `gpu_only`. |
| `dump_timing.py` | Fetches round timings only (no matrices) over a long span, to size concurrency. |

## Changed files

**`CliqueAI/miner.py`** — always dispatches. There is no switch:

```python
clique = await asyncio.to_thread(dispatch_client.solve, ...)
if not clique:
    bt.logging.warning("Dispatcher gave no answer; solving locally")
    clique = await self._solve_locally(synapse, adjacency_matrix, timeout)
```

`_solve_locally` is error handling, not a mode — nothing selects it. It keeps the
previous path (`native_algorithm`, per-(hotkey, uuid) seed, networkx fallback) so
a dispatcher restart costs a worse clique rather than a missed round, which would
score zero.

**`research_manual/solver.py`** — picker selection and the inputs it needs:

```python
+PICKER = os.environ.get("SN83_PICKER", "value").lower()   # was "static"
+PICKER_WANTS_N = "n_nodes" in inspect.signature(picker).parameters
+PICKER_WANTS_SUPPLY = "n_top_true" in inspect.signature(picker).parameters
...
+    answers = picker(pool, uuid, list(hotkeys), **kwargs)
```

`n_nodes` gives difficulty the way production does (the four problems in
`problem_selector.py` have non-overlapping vertex ranges; verified exact on all
1100 logged rounds). `n_top_true` is the distinct-clique count **before** the
pool is truncated to `k` — using the truncated length inflates the crowding
estimate by the truncation factor.

`research_manual/simulate.py` is **not** changed by this work.

## Design decisions, each forced by a measurement

**Reject, never queue.** When every worker is busy the request returns
`source="reject"` in under a second so the caller keeps its budget for a local
fallback. Queuing spends the deadline and then answers late, which scores zero.

**Workers are processes, not threads.** Two CUDA contexts in one process
time-slice the same device, which is the failure being fixed.

**A CPU overflow worker.** A reject sends *every* hotkey off to solve in its own
process at once — production runs one process per hotkey and each sizes its own
thread pool, so a fleet of 20 would ask a 15-CPU box for 160 threads at exactly
the moment the GPU workers need CPU for their champion stage. One shared CPU
solve beats N unshared ones.

**Thread split reserves overflow first.** At `SN83_CPU_BUDGET=15`:

```
workers=1 -> gpu 1x14 + overflow 1x1 = 15
workers=2 -> gpu 2x7  + overflow 1x1 = 15
workers=3 -> gpu 3x4  + overflow 1x1 = 13
```

Giving the overflow worker an equal share would cost the champion a third of its
threads on 99.95% of rounds to serve the 0.05% case. Taking the *remainder*
instead oversubscribes whenever the budget divides evenly.

**The owner asks for `SOLVE_K=64`, not `len(claims)`.** Siblings arrive while the
solve is already running, so at submit time the owner has only seen itself.
Sizing the pool to the claims known then left one clique for the whole fleet — a
bug the sibling test caught.

## Running it

```bash
# 1. the service, one per box, owns both GPUs
SN83_BACKEND=gpu SN83_WORKERS=2 SN83_CPU_BUDGET=15 \
  .venv/bin/uvicorn research_manual.eda.dispatcher:app --host 127.0.0.1 --port 8899

curl -s 127.0.0.1:8899/health | python3 -m json.tool     # wait for both workers

# 2. one miner per hotkey
SN83_THREADS=1 ./start_miner.sh \
    --netuid 83 --subtensor.network finney \
    --wallet.name default --wallet.hotkey miner1 --axon.port 8091
```

`start_miner.sh` uses a fixed pm2 process name, so a fleet needs `PROCESS_NAME`
varied or `pm2 start` called directly.

`SN83_THREADS=1` on the miners bounds the emergency path: if the dispatcher is
unreachable entirely, every hotkey solves locally at once, and the shim's own
`_concurrent` division counts only within one process.

The miner must be able to import `research_manual/eda/dispatch_client.py`, which
it locates relative to its own file. Deploying `CliqueAI/` without
`research_manual/` breaks the miner at import.

## Tests

```bash
SN83_BACKEND=fake .venv/bin/pytest research_manual/eda/test_dispatcher.py -v   # no GPU
SN83_BACKEND=gpu SN83_WORKERS=2 .venv/bin/pytest research_manual/eda/test_dispatcher.py -v
```

The four `gpu_only` tests are the ones that matter:

- `test_two_concurrent_tasks_both_meet_their_deadline` — the failure this exists to prevent
- `test_owner_answers_inside_the_deadline` at 7.5 s and 15 s
- `test_both_devices_are_actually_used` — that the workers are not both on device 0

## Environment

| variable | default | meaning |
|---|---|---|
| `SN83_BACKEND` | `gpu` | `gpu` / `cpu` / `fake` |
| `SN83_WORKERS` | `2` | GPU workers, one per device |
| `SN83_CPU_WORKERS` | `1` | overflow workers, no device |
| `SN83_CPU_BUDGET` | `15` | CFS quota to divide |
| `SN83_OVERFLOW_THREADS` | `1` | reserved before the GPU split |
| `SN83_SOLVE_K` | `64` | cliques the owner requests |
| `SN83_DISPATCH_URL` | `http://127.0.0.1:8899` | service address |
| `SN83_PICKER` | `value` | `value` / `static` / `legacy` |

## Not validated

- **The dispatcher has never run against a real GPU.** Every test so far used the
  `fake` backend. The `CUDA_VISIBLE_DEVICES` pinning in particular is untested.
- **7 threads vs the 8 everything was tuned at.** The shim records that thread
  count changes *which clique* the solver finds, so champion quality at 7 is an
  open question. CPU-only to check.
- **The overflow worker at 1 thread has never been timed.** It must finish inside
  the round's deadline or it is worse than useless.
- **`pick_value` at N=60 measured 2.6205 in the harness and 2.6061 live.** The
  gap is not fully explained, and no paired `static` run exists at N=60.
