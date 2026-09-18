# SN83 miner — runbook for this box

4× RTX A4000 (sm_86), 20 cores, no root, no CFS cap.

Two processes: **one dispatcher** owning all four GPUs, and **one miner per hotkey**
talking to it over localhost. The miner never touches a GPU itself.

    validator --> miner (axon) --> dispatcher --> worker[gpu0]
                    |                          \- worker[gpu1]
                    |                          \- worker[gpu2]
                    |                          \- worker[gpu3]
                    |                          \- worker[cpu]  (overflow; last resort)
                    \- local CPU fallback, only if the dispatcher is unreachable

## Why the dispatcher exists

Two measured problems, one service.

**Concurrency.** The harvest kernel is sized to fill the card, so a second launch
queues behind the first: two simultaneous solves on one device each took ~2×
wall time and *both* missed the deadline, which scores zero on both reward
terms. One worker process per device — processes, not threads, because two CUDA
contexts in one process time-slice the same card.

**Sibling allocation.** Each hotkey is its own miner process and sees `q = 1`,
but the picker's whole omega / omega-1 decision is a function of `q`. Routing
every hotkey through one service turns `q` into something observed rather than
guessed. With `SN83_FLEET_N=1` this does nothing yet; it is what makes growing
the fleet a config change rather than a rewrite.

## First run

    deploy/refresh_metagraph.sh      # picker's field model; must exist
    deploy/start_dispatcher.sh       # builds both native libs, warms all four GPUs
    deploy/start_miner.sh            # wallet must be restored and registered
    deploy/monitor.sh                # what the validators scored us

Currently deployed: coldkey `83_entity`, hotkey `83_miner1`, **uid 216**,
axon published as `122.228.216.178:20064` (internal `8000`).

`start_dispatcher.sh` blocks until every worker reports `ok:` and prints
`/health`. If it times out, `pm2 logs sn83-dispatcher`.

## Before the miner will start

1. **Wallet.** Restore your coldkey and hotkey into `~/.bittensor/wallets/`.

   **Keyfile format trap.** `btcli` (bittensor-cli 9.23.2, bundling
   `bittensor_wallet` 4.1.0) writes `"cryptoType": 1` as an integer.
   `bittensor==10.3.2` hard-pins `bittensor-wallet==4.0.1`, whose deserialiser
   accepts that field only as a *string* — so a wallet written by btcli fails
   in the miner with `KeyFileError: Failed to parse keyfile data`, and the
   supervisor restart-loops. The wallet here was rewritten to
   `"cryptoType": "sr25519"`, which both versions read. If you restore a fresh
   wallet with btcli, apply the same change (key material is untouched — only
   the JSON type of that one field).
2. **Register** the hotkey on netuid 83 (costs TAO).
3. **`AXON_IP` / ports — the Vast.ai NAT trap.** This container is behind NAT
   and only a fixed set of internal ports is forwarded, each under a *different*
   external number:

       env | grep VAST_TCP_PORT_      # internal -> external
       #   22 -> 20020,  80 -> 20040,  8000 -> 20064,  8001 -> 20058, ...

   So the axon must BIND a mapped internal port and PUBLISH the external one:
   `AXON_PORT=8000`, `AXON_EXTERNAL_PORT=20064`.

   And `AXON_IP` must be Vast's **ingress** address from `PUBLIC_IPADDR`
   (`122.228.216.178`) — *not* what `api.ipify.org` or `ifconfig.me` returns
   (`174.136.205.7`), which is the **egress** address and refuses inbound
   connections.

   Getting either wrong is silently fatal: `is_serving` reads `True` on chain,
   the miner logs look healthy, and every round scores zero with an empty
   answer because the request never reaches the process. That cost 28 rounds
   here. Verify reachability from OUTSIDE after any change — a probe from this
   box to its own public IP proves nothing, because NAT hairpinning fails even
   when inbound works.

4. If the container is recreated, the port map and `PUBLIC_IPADDR` change.
   Re-read them and update `sn83.env` before starting.

## Everything is in sn83.env

Both scripts source it. The settings that are decisions rather than defaults:

| setting | value | why |
|---|---|---|
| `SN83_GPU_ARCH` | `86` | A4000 is sm_86, same as gpu_lib's default |
| `SN83_CPU_BUDGET` | `20` | this box's core count (`nproc`); 6+5+4+4 GPU + 1 overflow |
| `SN83_OVERFLOW_THREADS` | `1` | last resort; reserving 8 of 20 would drop every GPU worker to 3 threads |
| `SN83_FLEET_N` | `1` | our registered hotkey count — **raise it when you add hotkeys** |
| `--neuron.autoupdate 0` | (in start_miner.sh) | autoupdate `git pull`s on every 12s tick; against a branch with local changes the pull fails, the miner exits, the supervisor restarts it, and it never serves a request |

**Thread count changes the ANSWER here, not just the speed.** The solver was
tuned at 8 threads and every number in `research_manual/` was measured there.
`SN83_CPU_BUDGET=20` is this box. Four workers at the tuned 8 would need 32
disjoint cores we do not have. The even split is 4 each; leftover cores go
+2 to gpu0 and +1 to gpu1, which take 99.9% of rounds: 6+5+4+4+1.

## Keeping the metagraph fresh

The picker models the field from `research_manual/artifacts/data/metagraph.json`
— who is registered, and which rivals our own registrations displace. Both drift.
Put it on a timer:

    crontab -e
    17 * * * * /home/dev/better_83/deploy/refresh_metagraph.sh >> ~/sn83-metagraph.log 2>&1

Hourly is ample; immunity alone is 6000 blocks (~20 h). **This cron is
installed** (`crontab -l` to confirm). The dumper writes through
a temp file and `os.replace`, so a miner reading it mid-refresh sees one whole
snapshot or the other, never half.

## Day to day

    pm2 status
    pm2 logs sn83-dispatcher
    pm2 logs sn83-miner-<hotkey>
    curl -s 127.0.0.1:8899/health | python3 -m json.tool

`counters` in `/health` is the thing to watch:

- `late` — answers that overran the internal budget. Should stay 0.
- `rejected` — all workers busy. Occasional is by design (rejecting instantly
  leaves the miner its budget); sustained means you need more workers.
- `overflow_cpu` — rounds that fell to the CPU worker. With 4 GPUs this is
  five-deep concurrency and should stay at 0.
- `error` — worker crashes. Investigate any.

`pm2 restart` reuses the environment captured at `pm2 start`, so after editing
`sn83.env` re-run the start script rather than restarting.

pm2 does not survive a reboot here (`pm2 startup` needs root). After a reboot,
re-run both start scripts.

## Monitoring: one row per validator request

    deploy/monitor.sh                     last 50 rounds that queried us
    deploy/monitor.sh --since 6h
    deploy/monitor.sh --limit 200 --csv scores.csv
    deploy/monitor.sh --watch 300         refresh every 5 minutes
    deploy/monitor.sh --archive rounds.jsonl    keep the raw rows
    deploy/monitor.sh --replay rounds.jsonl     re-read them, no network

Reads the validators' own W&B logs (`toptensor-ai/CliqueAI`). Every number is
the validator's, read back after it scored the round — nothing is recomputed
locally, so a row is what we were actually paid.

    time      uuid      n    tl    d    size best  rel    opt    div    reward place    dup
    07:34:20  fixture1 494  10.0  0.8  15   15   1.000  1.000  0.200  2.000  1/12     5    *

| column | meaning |
|---|---|
| `size` | our clique size; 0 means no answer, or one the validator rejected |
| `best` | largest *valid* clique anyone submitted that round |
| `rel` | `size / best` |
| `opt` | optimality, already normalised so the round's best miner scores 1.000 |
| `div` | diversity, normalised; `1/(miners sending our exact vertex set)` before that |
| `reward` | `opt * (1 + difficulty) + div` — the final score, max `2 + difficulty` |
| `place` | rank by reward among miners queried that round; ties share a place |
| `dup` | how many miners sent our exact vertex set, us included |

`*` marks a round where we matched the best clique found; `<- ZERO` marks a
rejected answer. The row above is the case worth understanding: we found the
best clique and still scored 2.000 of a possible 2.800, because four rivals sent
the *same* vertex set and diversity pays `1/5`. That is the whole reason the
solver harvests a pool instead of returning one champion.

It runs from `.venv-monitor`, not the miner venv — `wandb` is deliberately kept
out of the miner's dependency set so nothing can move `bittensor==10.3.2`'s pins
under a running miner.

**Credentials.** `WANDB_API_KEY` is read from the environment, else from
`.env` at the repo root (gitignored). The key there is a read/service token: it
can list runs and scan history, but `api.viewer` returns nothing, so ignore any
"could not verify credentials" noise from `wandb login`.

**"Not queried in the N rounds scanned"** is a real state, not an error. A
validator only sees a newly registered hotkey after it resyncs its metagraph,
and it samples a fraction of miners each round. Check the axon is actually
served before assuming something is broken.

## Adding hotkeys later

1. Register them under a coldkey listed in `deploy/coldkeys.txt`.
2. `deploy/fleet_size.sh` -- confirms the count the picker will use. Nothing to
   edit: `SN83_FLEET_N=auto` counts them from the metagraph snapshot.
3. `deploy/start_fleet.sh` -- one miner per hotkey, each on its own forwarded
   port. `DRY=1 deploy/start_fleet.sh` prints the port plan first.

Hotkeys cannot share a port: bittensor verifies the validator's signature
against the hotkey that axon serves, so a request for one hotkey arriving on
another's port is rejected. The fleet is therefore capped by how many ports the
host forwards. This box forwards internal 10-250; 22 (ssh) and 80 (taken) are
reserved, and the launcher uses 100-249 by default (`FLEET_PORT_LO/HI`,
`FLEET_RESERVED`). Publishing a port the host does not forward is silently
fatal: `is_serving` reads true on chain and every round scores zero.

The dispatcher already batches siblings per round; nothing else changes.

## A fresh pod, before anything else

**Stop the OS from replacing the GPU driver under a running miner.** As root,
on every new pod, before starting anything:

    apt-mark hold 'nvidia-*' 'libnvidia-*'
    systemctl disable --now apt-daily-upgrade.timer
    nvidia-smi -pm 1                  # persistence: GPUs stay initialised

MEASURED 2026-09-18: unattended-upgrades upgraded the NVIDIA userspace to
580.178 under a running 580.95 kernel module (`nvidia-smi` broke immediately,
CUDA survived), then 35 minutes later DELETED the running driver's GSP firmware
(`/lib/firmware/nvidia/580.95.05/`). Every GPU then failed to initialise --
`cuInit` returned CUDA_ERROR_UNKNOWN with no project code loaded -- and stayed
dead until the firmware package was reinstalled. A container cannot reload the
host's kernel module, so this is unrecoverable without root or a new pod.
Persistence mode keeps the GPUs from re-initialising in the first place.

Then bring the repo up:

    git clone <this repo> && cd better_83
    uv venv -p 3.12 .venv && uv pip install -p .venv/bin/python -r requirements.txt
    uv pip install -p .venv/bin/python fastapi uvicorn tqdm pytest 'wandb<0.20'

    # the two data files the picker and the simulator read (both gitignored)
    .venv/bin/python research_manual/eda/dump_metagraph.py --live --netuid 83
    .venv/bin/python research_manual/eda/dump_wandb.py      # needs WANDB_API_KEY

    cp deploy/coldkeys.txt.example deploy/coldkeys.txt   # then edit
    deploy/fleet_size.sh                                 # confirms SN83_FLEET_N
    rm -f deploy/MINER_DISABLED                          # if this box should serve

`start_dispatcher.sh` builds the native libraries on first run (nvcc ~17 s,
g++ ~2 s), never inside a round.

**One reachable port per hotkey.** Each hotkey is its own miner process with
its own axon, and bittensor verifies the validator's signature against *that
axon's* hotkey, so hotkeys cannot share a port. A NAT'd box forwarding 12 ports
hosts 12 miners, not 150 -- check the port map before registering a fleet.

## Toolchain, for rebuilding this box

No root was available, so nothing is installed system-wide:

- `uv` in `~/.local/bin`, CPython 3.12 under `~/.local/share/uv`
- venv at `/home/dev/better_83/.venv`
- CUDA 12.6 in `~/opt/cuda` — NVIDIA's apt debs unpacked with `dpkg-deb -x`
  (`cuda-nvcc-12-6`, `cuda-cudart-12-6`, `cuda-cudart-dev-12-6`, `cuda-crt-12-6`,
  `cuda-nvvm-12-6`). nvcc is needed at *runtime*, not just build time: `gpu_lib`
  rebuilds the `.so` whenever `clique_gpu.cu` is newer.
- node LTS + pm2 via nvm in `~/.nvm`

## Tests

    SN83_BACKEND=fake .venv/bin/python -m pytest research_manual/eda/test_dispatcher.py -q
    set -a; . deploy/sn83.env; set +a
    SN83_BACKEND=gpu  .venv/bin/python -m pytest research_manual/eda/test_dispatcher.py -q

Two known failures, both pre-existing and outside the production path:

- `test_importing_solver_does_not_init_cuda` — `research_manual/solver.py`
  imports `fleet_solver_gpu`, which opens a CUDA context at import. Nothing in
  the miner or the dispatcher parent imports `solver.py`, so no production
  process is affected; `simulate.py` is.
- `test_workers_report_their_pinned_cores` — passes alone. It fails after
  `test_thread_split_never_oversubscribes` calls `importlib.reload(dispatcher)`,
  which swaps the session fixture's *started* worker pool for an unstarted one.
  Test isolation, not the service.

The unit tests do not exercise the axon, the HTTP hop, sibling batching under
overlap, or the alert. `research_manual/eda/e2e/` does, on production code:

    K=100 research_manual/eda/e2e/run_e2e.sh
    .venv/bin/python research_manual/eda/e2e/e2e_analyze.py ../better_83_e2e/run1

Both failures this audit found -- siblings exhausting the dispatcher's thread
pool at fleet 100, and pickers starving the event loop at 150+ -- passed every
unit test and `simulate.py`. Run it after any change to the dispatcher, the
miner, or the picker, before deploying.
