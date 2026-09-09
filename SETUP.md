# SN83 miner — setting up from scratch

Written after doing it once, on a box where almost everything that could be
wrong was wrong. Follow it in order. Every section marked **TRAP** cost real
time or real money; none of them announce themselves.

---

## 0. Choose the host — read this before you rent anything

**Do not host this miner in China.** This is not a preference, it is the single
biggest determinant of income, and it is invisible until you measure it.

Measured on a Vast.ai box in Wenzhou, Zhejiang (`AS134771`, China Telecom):

```
our miner (China)        : 36/95 validator requests never arrived (38%)
every other miner        : 104/5685 (1.83%)
median other miner       : 0.0%
```

The requests never reached the process — no TCP connection, no blacklist, no
exception, nothing in the kernel's drop counters. Meanwhile 330 probe requests
through the same public endpoint from inside the box succeeded 100%, and global
TCP reachability measured 96–100% from 25 vantage points. Cross-border transit
into China is intermittent and asymmetric, and short probes from well-peered
networks do not see it. **You cannot diagnose this from inside the box, and you
cannot fix it in code.**

Requirements:

- **Location:** US or EU. Somewhere validators reach without crossing a national
  firewall. (Validators observed on SN83: Los Angeles, Albuquerque.)
- **GPUs:** 2× is the sweet spot. 18.9% of rounds are two-deep concurrent, so one
  GPU means both solves miss their deadline. A 4090 (sm_89) is plenty.
- **CPU:** the solver was tuned at 8 threads per worker; you want ~24 cores free.
- **Inbound TCP:** you must be able to accept connections on some port. See §5.

---

## 1. Toolchain (assumes no root)

None of this needs sudo. If you have root, apt is fine instead.

```bash
# Python 3.12 via uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"          # uv appends this to ~/.bashrc
uv python install 3.12

cd /path/to/better_83
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
uv pip install --python .venv/bin/python -e . --no-deps
uv pip install --python .venv/bin/python fastapi "uvicorn[standard]" httpx
```

**CUDA toolkit without root.** `nvcc` is needed at *runtime*, not just build
time: `gpu_lib` rebuilds the `.so` whenever `clique_gpu.cu` is newer. The pip
package `nvidia-cuda-nvcc-cu12` ships only `ptxas`, not `nvcc` — it is not
enough. Unpack NVIDIA's debs instead:

```bash
mkdir -p /tmp/cudadebs && cd /tmp/cudadebs
apt-get download cuda-nvcc-12-6 cuda-cudart-dev-12-6 cuda-cudart-12-6 \
                 cuda-crt-12-6 cuda-nvvm-12-6      # no root needed to download
mkdir -p x && for d in *.deb; do dpkg-deb -x "$d" x; done
mkdir -p ~/opt && cp -a x/usr/local/cuda-12.6 ~/opt/ && ln -sfn ~/opt/cuda-12.6 ~/opt/cuda
~/opt/cuda/bin/nvcc --version
```

(If the NVIDIA apt repo is not configured, fetch the same debs directly from
`https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/`.)

**node + pm2:**

```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
export NVM_DIR="$HOME/.nvm"; . "$NVM_DIR/nvm.sh"
nvm install --lts && npm install -g pm2
```

**btcli** — since bittensor 8.x the CLI is a *separate package*. Installing
`bittensor` does not give you `btcli`. Install it isolated so it cannot move the
miner venv's pins (`bittensor==10.3.2` hard-pins `bittensor-wallet==4.0.1`):

```bash
uv tool install bittensor-cli
```

### TRAP: GPU architecture

`gpu_lib.py` defaults to `SN83_GPU_ARCH=86`. A 4090 is **sm_89**. The wrong arch
still builds and runs — through PTX JIT, silently slower. Set it in
`deploy/sn83.env`. Check yours with `nvidia-smi --query-gpu=compute_cap --format=csv`.

---

## 2. Wallet

Restore your coldkey **public** part and your hotkey. The miner signs with the
hotkey only; the coldkey private key does not belong on a mining box.

```bash
btcli w regen-coldkeypub --wallet.name <name>    # watch-only, public part
btcli w regen-hotkey     --wallet.name <name> --wallet.hotkey <hk>
btcli w list
```

Register on netuid 83 from wherever the coldkey lives (costs TAO).

### TRAP: keyfile `cryptoType` version skew

`btcli` (bittensor-cli 9.x → `bittensor_wallet` 4.1.0) writes

```json
"cryptoType": 1        // integer
```

`bittensor==10.3.2` pins `bittensor-wallet==4.0.1`, whose deserialiser accepts
that field **only as a string**. The miner then dies at startup with

```
KeyFileError: Failed to get hotkey: DeserializationError("Failed to parse keyfile data.")
```

and pm2 restart-loops. The key material is fine; only the JSON type is wrong.
Fix (back up first — these are keys):

```bash
tar czf ~/wallet-backup.tgz -C ~ .bittensor/wallets && chmod 600 ~/wallet-backup.tgz
python3 - <<'EOF'
import re, os, json, glob
for f in glob.glob(os.path.expanduser("~/.bittensor/wallets/*/coldkeypub.txt")) + \
         glob.glob(os.path.expanduser("~/.bittensor/wallets/*/hotkeys/*")):
    raw = open(f, "rb").read()
    new, n = re.subn(rb'("cryptoType"\s*:\s*)(\d+)', rb'\1"sr25519"', raw)
    if n:
        before, after = json.loads(raw), json.loads(new)
        assert {k:v for k,v in before.items() if k!="cryptoType"} == \
               {k:v for k,v in after.items()  if k!="cryptoType"}
        mode = os.stat(f).st_mode & 0o777
        fd = os.open(f + ".tmp", os.O_WRONLY|os.O_CREAT|os.O_TRUNC, mode)
        os.fdopen(fd, "wb").write(new); os.replace(f + ".tmp", f)
        print("fixed", f)
EOF
```

Both `bittensor_wallet` 4.0.1 and 4.1.0 read a string, so this is forward-safe.

---

## 3. The metagraph snapshot (required — the picker reads it per request)

```bash
deploy/refresh_metagraph.sh
```

Writes `research_manual/artifacts/data/metagraph.json`. Without it the picker
raises **inside the request path** and every round falls back. Put it on a timer;
the file is written atomically so a mid-refresh read is safe:

```
17 * * * * /path/to/better_83/deploy/refresh_metagraph.sh >> ~/sn83-metagraph.log 2>&1
```

Miners must be sorted weakest-first by `(incentive, uid)` — `dump_metagraph.py`
does this, `research/snapshot_metagraph.py` does **not**. Use the former.

---

## 4. Configure `deploy/sn83.env`

Everything is in one file. The settings that are decisions, not defaults:

| setting | why |
|---|---|
| `SN83_GPU_ARCH` | your GPU's compute capability; 89 for a 4090 |
| `SN83_CPU_BUDGET=24` | 2×8 GPU + 8 overflow on disjoint cores |
| `SN83_OVERFLOW_THREADS=8` | the tuned thread count, not the cramped default of 1 |
| `SN83_FLEET_N` | your registered hotkey count — **raise it when you add hotkeys** |
| `AXON_PORT` / `AXON_EXTERNAL_PORT` / `AXON_IP` | see §5 |

**Thread count changes the ANSWER, not just the speed.** The solver was tuned at
8 threads and every number in `research_manual/` was measured there.

---

## 5. Networking — where the first four hours of zeros came from

The axon has two different notions of address and they are easy to confuse:

- `--axon.port` and `--axon.ip` are what the process **BINDS** (`ip` defaults to
  `[::]`; leave it alone).
- `--axon.external_ip` / `--axon.external_port` are what gets **PUBLISHED on
  chain**. Validators dial these.

### TRAP: behind NAT (Vast.ai and similar)

On Vast only a fixed set of internal ports is forwarded, each under a
**different** external number:

```bash
env | grep VAST_TCP_PORT_
#   22 -> 20020,  80 -> 20040,  8000 -> 20064,  8001 -> 20058, ...
echo $PUBLIC_IPADDR
```

So bind a *mapped* internal port and publish the *external* number:

```bash
export AXON_PORT=8000            # internal, must be in the map
export AXON_EXTERNAL_PORT=20064  # what VAST_TCP_PORT_8000 says
export AXON_IP=$PUBLIC_IPADDR    # ingress address
```

### TRAP: ingress vs egress IP

`api.ipify.org` / `ifconfig.me` return your **egress** address. On Vast that is a
*different host* which refuses inbound connections. Use `PUBLIC_IPADDR`. Getting
this wrong is silently fatal: `is_serving` reads `True` on chain, the miner logs
look perfectly healthy, and every round scores zero with an empty answer because
the request never reaches the process. **28 rounds cost us this.**

### Verify from OUTSIDE, always

A probe from the box to its own public IP proves nothing — NAT hairpinning fails
even when inbound works, and it can also *succeed* when inbound is broken. Use an
external checker:

```bash
curl -s "https://check-host.net/check-tcp?host=<IP>:<PORT>&max_nodes=20" \
     -H "Accept: application/json"
# then GET https://check-host.net/check-result/<request_id>
```

A live axon answers `HTTP 404` on `/` (it only routes
`/MaximumCliqueOfLambdaGraph`). A 404 from three continents is the green light.

**If the container is ever recreated, `PUBLIC_IPADDR` and the whole port map
change.** Re-read them and update `sn83.env` before starting.

---

## 6. Start it

```bash
deploy/start_dispatcher.sh    # builds both native libs, warms both GPUs, waits for ready
deploy/start_miner.sh         # one process per hotkey
deploy/monitor.sh --limit 30  # what the validators scored
```

`start_dispatcher.sh` blocks until every worker reports `ok:` and prints
`/health`. Both libraries are built there, never inside a request — an nvcc build
is ~16 s and would eat a round's deadline.

### TRAP: autoupdate restart loop

`--neuron.autoupdate` defaults to **1** and `git pull`s every 12 s. On a branch
with local changes the pull fails, `update_repo_if_needed()` returns True, the
miner exits, pm2 restarts it, and it never serves a request.
`deploy/start_miner.sh` passes `--neuron.autoupdate 0`. Keep it.

pm2 does not survive reboot here (`pm2 startup` needs root) — re-run both scripts.
`pm2 restart` reuses the environment captured at `pm2 start`, so after editing
`sn83.env` re-run the start script rather than restarting.

---

## 7. Monitoring

```bash
deploy/monitor.sh                  # last 50 rounds that queried us
deploy/monitor.sh --watch 300      # live
deploy/monitor.sh --since 6h --csv scores.csv
deploy/healthcheck.sh              # mechanical checks + safe auto-repair
```

`monitor.sh` reads the validators' own W&B logs (`toptensor-ai/CliqueAI`) and
prints one row per request: size, field best, optimality, diversity, reward,
place, and `dup` (how many miners sent our exact vertex set).

- `WANDB_API_KEY` goes in `.env` at the repo root (gitignored). A read/service
  token works; `api.viewer` returning nothing is normal, ignore it.
- **Pin `wandb<0.20`.** `research_manual/eda/dump_wandb.py` asserts it, and 0.29
  fails auth against this key. `.venv-monitor` is a separate venv on purpose so
  `wandb` cannot move `bittensor`'s pins.

```bash
uv venv --python 3.12 .venv-monitor
uv pip install --python .venv-monitor/bin/python "wandb>=0.19,<0.20" python-dotenv
```

---

## 8. Diagnosing a zero — do this before changing anything

Every zero is one of three things, and they look identical in W&B. The
classification is the whole game:

| symptom | meaning | fix |
|---|---|---|
| not in our miner log at all | request never **arrived** | network / hosting (§0, §5) |
| in our log with `size>0` | answer was **late** | reserve constants (§9) |
| in our log with `size=0` | **solver** failed | dispatcher |

```bash
grep -a "source=" ~/.pm2/logs/sn83-miner-<hotkey>-out.log      # what we received
deploy/monitor.sh --limit 30                                   # what was scored
```

Then compare by uuid. And **compare yourself against the field** —
`miner_ans` in each W&B row contains every miner's answer:

```
if your empty-rate >> the field's empty-rate, it is YOU, not the network
```

That single comparison would have saved me hours. I spent them blaming transit
because five separate negative tests (kernel drop counters, idle correlation,
deadline correlation, global reachability, alternate-port probes) all looked
consistent with it. They were consistent with it, and it was still the wrong
conclusion, because I never checked whether anyone else had the problem.

**Beware self-inflicted log noise.** An HTTP probe against the axon with no
synapse headers makes it log `UnknownSynapseError` or `BlacklistedException:
Missing dendrite or hotkey`. Those are *your* probes, not validator traffic.
`healthcheck.sh` uses a plain TCP connect for exactly this reason.

Blacklist decisions log at **TRACE**, not INFO. If you suspect requests are being
rejected rather than lost, restart with `--logging.trace` — otherwise a silently
blacklisted request is indistinguishable from one that never arrived.

---

## 9. Deadline constants (`CliqueAI/miner.py`)

The miner must answer *and get the response back* inside the validator's
deadline. Two knobs, both measured rather than guessed:

```python
MAX_HOLD_S   = 18.0   # never hold a connection longer than this
LATENCY_FRAC = 0.25   # round-trip reserve as a fraction of the deadline
LATENCY_S    = 2.0    # floor for that reserve
```

Evidence behind them:

- At `tl=30` we answered at ~27.5 s and **3 of 4 rounds scored zero**, including
  cliques of 44 and 76 that matched the field's best. `tl=10`/`tl=15`, holding
  7.5 s and 12.5 s, scored reliably. Capping the hold at 18 s fixed it — 0 late
  since. Something between miner and validator will not keep a request alive for
  ~27 s.
- Later, one `tl=10` and one `tl=15` round lost the same way at margins of 2.37 s
  and 2.75 s. Raising `LATENCY_FRAC` 0.15 → 0.25 widened `tl=10` and `tl=15` while
  deliberately leaving `tl=6`/`tl=7.5` alone (the 2.0 s floor), because neither had
  ever lost a round.

Time-to-omega is under 10% of budget, so shortening the hold costs **pool
breadth** (fewer distinct cliques for the picker), not the clique itself. A
narrower pool still scores; a late answer scores zero.

**Do not change solver constants** (`THREADS`, `CHAMPION_SHARE`, picker
parameters) from live samples. Those need `simulate.py` — see `CLAUDE.md`.

---

## 10. Tests

```bash
SN83_BACKEND=fake .venv/bin/python -m pytest research_manual/eda/test_dispatcher.py -q
set -a; . deploy/sn83.env; set +a
SN83_BACKEND=gpu  .venv/bin/python -m pytest research_manual/eda/test_dispatcher.py -q
```

Two known failures, both pre-existing and outside the production path:

- `test_importing_solver_does_not_init_cuda` — `research_manual/solver.py` imports
  `fleet_solver_gpu`, whose `_init_gpu()` runs at import and opens an 8-vertex
  graph on device 0 purely to read `maxn`, a **compile-time constant** readable
  from `sn83_gpu_config` with no device at all. Costs 386 MiB and a CUDA context
  in any process that imports `solver`. Nothing in the miner or the dispatcher
  parent does, so no production process is affected; `simulate.py` is.
- `test_workers_report_their_pinned_cores` — passes alone. Fails after
  `test_thread_split_never_oversubscribes` calls `importlib.reload(dispatcher)`,
  which swaps the session fixture's *started* worker pool for an unstarted one
  (`worker_info == []`) and orphans its worker processes. Test isolation, not the
  service. Any `/solve` after that reload returns HTTP 500; the suite only
  survives because nothing posts one.

---

## 11. What good looks like

```
dispatcher: 3/3 workers free, served=N late=0 error=0 rejected=0
requests handled in the last 25 min: 8-12
axon on chain: <ingress-ip>:<external-port> (matches sn83.env)
0 problem(s)
```

and in `monitor.sh`, rows like

```
time      uuid      n    tl    d    size best  rel    opt    div    reward place    dup
18:44:10  69a328d3 497  10.0  0.8  53   53   1.000  1.000  1.000  2.800  1/44     1    *
```

`opt 1.000` means you matched the best clique in the field. `div` is where the
competition actually is: everyone reaches omega, so ranking is decided almost
entirely by *which* clique you pick. A single hotkey cannot win on diversity —
`dup > 1` costs you proportionally. That is a picker problem, measured with
`simulate.py`, not something to tune live.

Expected: **~100% of arrived requests scoring**, on a host outside China.
