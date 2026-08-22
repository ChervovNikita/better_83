# SN83 research tooling

Turns the CliqueAI validators' public W&B stream into a labelled max-clique
dataset, and scores a candidate solver against the real field before any TAO is
spent on registration.

## Why this exists

Scoring on SN83 is *relative*: reward is `ω·(1+d) + δ`, where ω compares your
clique size to the other responders in that round and δ is `1/(number of miners
who returned your exact vertex set)`. Because the validators log every round in
full — the graph, every miner's answer, and the resulting scores — you can
replay that scoring for a hypothetical extra miner and learn exactly what a
solver would have earned.

Two facts drive everything here:

- **Weights are a rank amplifier.** `set_weights()` min-max normalises, applies
  a sigmoid whose steepness floors at 0.1 (the field's real IQR is ~0.02), then
  raises to a power γ≈10 chosen so the top half takes 80%. A mean reward of
  1.94 against a field median of 2.47 quantises to *zero* on chain. There is no
  partial credit.
- **Uniqueness, not size, is the prize.** ~72% of answers already tie for best
  size, and 41% are byte-identical to another miner's. Diversity is ~72% of the
  field's total reward loss.

## Start here

```bash
python3 run.py --hours 24 --dry-run     # the plan and the ETA
python3 run.py --stage selftest         # verify the code before trusting it
python3 run.py --hours 24               # metagraph -> data -> selftest -> solve -> simulate
```

`run.py` is the entry point. Each stage is skipped when its output is already
complete, and the solve stage — the only slow one — is resumable, so an
interrupted run costs only what is left. `--hours` is the honest unit: rounds
arrive at ~105/h, and **nothing about deregistration is testable below 20 h**,
which is the immunity window; `run.py` says so rather than letting you read a
vacuous `survived` column as a result.

`test_fleet_sim.py` is the gate. It runs inside `run.py` before the expensive
stage and again against the real cache afterwards, and a failure stops the
pipeline. Every check in it exists because something actually broke — the
simulator once re-admitted the miners it had just displaced, once reported a
100% fleet share on a dataset missing hotkeys, once wrote real scores into
validator slots, and once flattened its own zero-skill control to a constant.
The strongest of them is `N=0 reproduces the validator's own logged rewards`:
with no fleet inserted and nobody displaced, the replay must return the log
byte for byte.

## The saturation result

Taking every slot is the cheapest way to find out what actually limits a fleet.
Two hard limits fall out, neither of them the scoring mechanism:

- **You cannot take the whole subnet.** Immune UIDs are protected, so the largest
  instantaneous fleet is the non-immune count — 230 of 249 today. `pick_victims`
  refuses anything larger rather than pretending.
- **The clique pool is the binding constraint.** With the fleet holding every
  displaceable slot, the hotkeys the pool can serve bunch tightly (first 4: mean
  2.638, sd 0.214 — everyone runs the same solver, so they differ only in which
  clique they were handed), while the remaining 226 average 0.296 and **131 score
  exactly zero**. They are queried and have nothing to answer with.

`solve_many` currently yields a median of **4** distinct cliques per round against
20 requested, so a fleet past ~4 hotkeys is dead weight. Fleet scale and collision
avoidance turn out to be the same capability: both need the solver to emit many
distinct maximum cliques, and neither is reachable without it.

## Layout

| file | what it does |
| --- | --- |
| `fetch_new.py` | incremental pull of new rounds; the cron entry point |
| `build_dataset.py` | bulk backfill into a JSONL corpus |
| `eval_harness.py` | run a solver, replay scoring, report reward + implied emission |
| `status.py` | pipeline health; exit 1 if stale or errored |
| `solver.py` | baseline local search (mean reward 1.755 — the dead zone) |
| `make_splits.py` | stratified train / validation split, sized by solver budget |
| `score_submission.py` | run a candidate solver on validation and score it |
| `reward_reference.py` | exact validator reward replay (pinned by a test) |
| `fleet_sim.py` | simulate entering with N hotkeys, with displacement modelled |
| `fleet_solver.py` | `solve_many(A, time_limit, k)` — one solve, k distinct cliques |
| `run.py` | **entry point** — the whole pipeline, resumable |
| `test_fleet_sim.py` | invariant suite; the gate before any result is trusted |
| `snapshot_metagraph.py` | who is displaceable, and when immunity lapses |
| `AGENT.md` | the brief handed to whoever builds the solver |
| `setup_server.sh` | provision a box (venv; cron optional) |
| `requirements.txt` | **pinned** deps — see the wandb note below |
| `data/` | shards, state and logs (gitignored) |

## Dataset schema

One JSON object per round, one round per line:

| field | meaning |
| --- | --- |
| `uuid`, `_run`, `_step` | provenance |
| `n`, `edges`, `density` | graph shape |
| **`time_limit`** | the deadline that round ran under (6 / 7.5 / 10 / 15 / 30 s) — tune against this |
| `difficulty` | 0.7 / 0.8 / 0.9 / 1.0, the tier multiplier in the reward |
| `matrix_b92` | the graph; decode with `CliqueAI.graph.codec.GraphCodec` |
| `best_size` | label: largest valid clique any miner found |
| `best_cliques` + `best_clique_counts` | every distinct optimum the field found, and how many miners hit each — the anti-collision signal |
| `size_hist`, `any_unique` | sufficient statistics to replay scoring for a new answer |
| `n_responders`, `n_valid`, `n_at_best` | round context |
| `answers` | full per-miner detail, only with `--keep-answers` |

The label is *best known*, not proven optimal: spot-checked against a
first-moment bound on ω it sits at the bound on 6 of 15 instances and within 1–2
on the rest.

## Usage

```bash
# keep the corpus current (this is what cron runs)
python3 fetch_new.py --versions 0.0.17

# one-off benchmark set
python3 build_dataset.py --versions 0.0.17 --limit 5000 --out bench.jsonl

# score a solver — overall, then broken out per deadline and per graph size
python3 eval_harness.py bench.jsonl --solver solver:solve --limit 200

# only the 7.5s rounds, to tune the short-deadline path
python3 eval_harness.py bench.jsonl --solver mysolver:run --time-limit 7.5

# is the pipeline alive?
python3 status.py

# build the splits, then score a solver against the withheld labels
python3 make_splits.py --budget 600 --bigger-n 500
python3 score_submission.py --solver mymodule:solve                     # val, ~10 min
python3 score_submission.py --solver mymodule:solve --split bigger_val  # audit, ~2 h
```

## Splits

`make_splits.py` sizes validation by *solver budget* rather than row count: it
draws instances until their deadlines sum to `--budget` seconds (default 600, so
one validation pass costs about ten minutes). Sampling is stratified by
(time_limit, |V| bucket) with largest-remainder proportional allocation, so the
validation mix reproduces the pool rather than merely matching it in
expectation; `manifest.json` records the side-by-side check.

**The splits are for fitted components only.** A solver is an algorithm, not
something trained on this data, so the honest test is `fleet_sim.py` replaying the
real sequence of rounds in order. Split only what can overfit — a collision
picker that learns which cliques to avoid.

Two held-out sets are drawn, disjoint from train and from each other: `val`
(~10 min a pass, the steering signal) and `bigger_val` (500 instances, ~2 h, the
rarely-run audit). Each is split into `*_problems.jsonl` — graphs and deadlines,
**no** labels — and `*_labels.jsonl`, which only `score_submission.py` reads.
`train.jsonl` keeps full labels and is unrestricted. See `AGENT.md` for the
brief this is built around.

A solver is any `f(A, time_limit) -> list[int]`, with `A` an `n×n` uint8 numpy
adjacency matrix. It must return a **maximal** clique — one that can still be
extended scores zero, as does an empty answer or a repeated vertex.

Targets to beat: **2.35** clears the dead zone, **2.470** is the field median,
**2.591** is today's best UID, **2.845** is best-size-and-always-unique.

## Volume

1,689,167 rounds are logged across 68 runs back to 2025-08-26; 233,548 of those
are v0.0.16+, which matches today's problem mix. Throughput is ~15 rows/s and
~43 KB/row with the graph included, so a 5,000-instance benchmark takes about
six minutes and the full v0.0.17 history about half an hour. Threads help little
— the limit is server-side, so parallelise across runs rather than pages.

Two gotchas worth knowing: `scan_history` repeats roughly one row per page at
page boundaries (deduplicate by `uuid`), and requesting `adjacency_list` makes
the history endpoint return HTTP 500 — `matrix_b92` is the same graph in a tenth
of the bytes.

## The wandb pin is load-bearing

`wandb>=0.20` reroutes `Run.scan_history` through a service API that downloads
the *entire* run history before yielding a row. `use_cache=False` does not avoid
it. SN83 runs hold ~23,000 steps each carrying a full `adjacency_list`, so on
0.28.2 the call simply never returns — a 200-step incremental fetch that takes
21s on the pin timed out at 400s. `requirements.txt` pins `wandb==0.17.0`, and
`_common.check_wandb_version()` fails loudly at startup if something newer gets
installed, so cron logs an error instead of hanging forever.

Server deployment uses a dedicated venv at `<repo>/.venv` so the pin cannot
disturb anything else on the host.

## Deployment

```bash
git clone <repo> && cd <repo>
WANDB_API_KEY=... bash research/setup_server.sh
```

That creates the venv, writes `~/.netrc` (0600), installs and starts cron, adds
a `*/5 * * * *` job, runs a first backfill, and prints the health report. Re-run
it after a `git pull`; it is idempotent.

The cron job only matters while you are accumulating history. Once the corpus is
large enough, drop it — `crontab -l | grep -v sn83-fetch | crontab -` — and
refresh on demand with `fetch_new.py` or a one-shot `build_dataset.py`.

### If the box restarts

The target is a container with no init system, so `cron` is started directly by
`setup_server.sh` rather than by systemd. A pod restart therefore leaves the
crontab intact but the daemon stopped, and fetches silently stop. Recovery is
one idempotent command:

```bash
cd /workspace/better_83 && git pull && WANDB_API_KEY=... bash research/setup_server.sh
```

`status.py` exits non-zero once the last successful fetch is more than 20
minutes old, so it is safe to use as an external liveness check.

## Fleet simulation

`fleet_sim.py` answers "what happens if we register N hotkeys", modelling the two
effects a naive replay misses: registering **removes** N miners, and any clique we
return that collides with a survivor drags that survivor's diversity down as well
as our own. Both change the field we are ranked against, so each round is rebuilt
and rescored from scratch.

Per round it drops the victims' answers, samples which of our hotkeys the
validator would have queried (independent Bernoulli at `P(difficulty)`, exactly
`MinerSelector`), inserts that many distinct cliques from one solve, and rescores.
The resulting per-UID reward streams go through `update_scores` and `set_weights`
to give our hotkeys' ranks and total emission share.

```bash
python3 snapshot_metagraph.py                       # incentive + registration blocks
python3 build_dataset.py --versions 0.0.17 --limit 1000 --keep-answers \
        --out data/sim_rounds.jsonl
python3 fleet_sim.py --sizes 1 5 10 20 40 --rounds 1000 --solve   # slow, cached
python3 fleet_sim.py --sizes 1 5 10 20 40 --rounds 1000 --history 3000
```

The solve phase is one call per round for `max(--sizes)` cliques (~12 s each,
resumable, shared across every N); every N after that replays off the cache in
seconds.

**Guards, all of them added because the first version tripped over them.** A
queried hotkey with no clique scores **zero** rather than vanishing from the round
— the validator scores every selected UID, and dropping it made the debiased EMA a
mean over answered rounds only, which inflated a 40-hotkey fleet by ~1100x. Every
run reports a **zero-skill null** (our hotkeys draw a reward from the round's own
survivors) and refuses to print alpha/day when the measured share falls inside it.
`set_weights` is handed the real 256-wide uid-indexed vector, since validator
slots hold 0 forever and the min-max step is a stretch, not an affine no-op. And
`--gamma` is diagnostic only: the validator derives it by binary search until the
top half takes exactly 80%, so forcing a converged value onto a short vector
models a validator that hands the top half 99.9%. Converge the field with
`--history` instead.

**Two further traps it guards against.** `score_round` is pinned to
`CliqueScoreCalculator` — 1,317 responses across 40 rounds with a third of the
miners removed, max |Δ| = 0.0 — because the whole point is scoring a *modified*
round. And γ derived from a short simulation comes out far below the live ~16.4,
since sampling noise widens the field's score vector; under 500 rounds the script
warns and you should pass `--gamma 16.4`. It also reports samples-per-hotkey,
which at `P≈0.2` is only a fifth of the round count against the live field's ~595.

### Sizing a fleet simulation

Rounds arrive at **~105/h** across the two v0.0.17 validators (median gap 34 s), so
the round count you need is set by what you want to observe, not by taste:

| you want to see | subnet time | rounds | solve time on 12 cores |
| --- | --- | --- | --- |
| scores settle | ~6 h | 630 | 2.1 h |
| our fleet leave immunity at all | 20 h | 2,091 | 7.0 h |
| a few days of churn against us | 72 h | 7,528 | 25.1 h |
| a week | 168 h | 17,565 | 58.6 h |

Immunity is 6000 blocks ≈ 20 h of wall clock, taken from the round timestamps
rather than a round count, because the two validators interleave. **Below ~2,100
rounds the fleet never leaves immunity and the `survived` column is vacuous** —
`fleet_sim.py` says so rather than letting the number be read as a result.
