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
python3 make_splits.py --budget 600
python3 score_submission.py --solver mymodule:solve
```

## Splits

`make_splits.py` sizes validation by *solver budget* rather than row count: it
draws instances until their deadlines sum to `--budget` seconds (default 600, so
one validation pass costs about ten minutes). Sampling is stratified by
(time_limit, |V| bucket) with largest-remainder proportional allocation, so the
validation mix reproduces the pool rather than merely matching it in
expectation; `manifest.json` records the side-by-side check.

The split is three files: `train.jsonl` with labels included, `val_problems.jsonl`
with graphs and deadlines but **no** labels, and `val_labels.jsonl`, which only
`score_submission.py` reads. See `AGENT.md` for the brief this is built around.

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
