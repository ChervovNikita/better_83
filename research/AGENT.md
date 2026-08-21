# Brief: build a maximum-clique solver that matches the SN83 field

## Your target

**Phase 1 (parity) is done — 150/150, optimality pinned at 1.0. Stop optimising it.**
Every remaining point is in the *diversity* term, and today we are the worst in the
field on it.

Measured, 150 rounds, scored with the validator's own `CliqueScoreCalculator`:

| | ours | field | field leaders |
| --- | --- | --- | --- |
| optimality | **1.0000** | 0.946 | ~0.96 |
| diversity | **0.5927** | 0.751 | 0.84–0.86 |
| reward | **2.4453** | 2.5055 (median) | 2.58–2.60 |
| collision rate | **65.3%** | — | — |

Because optimality is pinned, `reward ≈ 1.85 + δ`. δ is the only free variable:

| δ | reward | where that puts us |
| --- | --- | --- |
| **0.593 (today)** | 2.445 | bottom quartile (92% confidence), pruned ~21h after registering |
| 0.62 | 2.47 | ≈ field p10 |
| **0.66** | **2.51** | ≈ median — the first genuinely safe point |
| 0.70 | 2.55 | ≈ p75 |
| 0.75 | 2.60 | field-typical δ, near the top of the leaderboard |

**Target: δ ≥ 0.66, i.e. collision rate below ~40%.** That is merely field-typical,
not exceptional. Aim at 0.75.

### The one constraint that matters

**Never trade a vertex for uniqueness.** Measured over the same 150 rounds:

| | mean reward |
| --- | --- |
| max size, our actual collisions | 2.4453 |
| **one below max, always unique** | **1.8843  (−0.5581)** |
| max size, always unique | 2.8527  (+0.4073) |

The trap is that sandbagging wins on **32 of 150 individual rounds (~21%)**, so a
per-round optimiser will find those and take them, and every one is negative in
expectation. The objective is:

> maximise δ **subject to** size == best achievable.

Note also that "beat the field by +1 vertex" is worth **exactly 0.0000** to us
across all 479 rounds checked — optimality is already 1.0 and its normalisation is
a no-op. There is no upside above parity. Falling one short of a unanimous field
costs **−1.18**. The asymmetry is ≥30×.

### How to get δ up without spending search time

A plateau local search already passes through many distinct maximum cliques. Record
the distinct ones it visits and pick at the end by `hash(hotkey ‖ uuid)`. This is a
logging change, not a search change — important, because the deadline is already
88% consumed.

**Do not tune the picker against `best_cliques` in the dataset.** Avoiding the ~19
cliques the field happened to submit scores δ≈1.0 on validation and transfers
nothing, because live you will not know them. The version that generalises is
"sample from a large pool of distinct optima by hash", and it must be measured
that way.

### Scoring: use `reward_reference.replay_reward`

`bench.py`'s own `replay_reward` normalised the diversity term over **best-size
cliques only**; the validator normalises over **all valid answers**. That inflated
reward on every colliding round (mean +0.10, max +0.50) — i.e. it made collisions
look cheaper than they are, which is exactly the wrong bias for this work.

`research/reward_reference.py` is exact: pinned to `CliqueScoreCalculator` on 160
(round, answer) pairs, max |Δ| = 0.000e+00, by `test_reward_reference.py`. Two
details it gets right that hand-rolled replays miss:

- the normaliser is `1 / min(augmented duplicate counts)` — computed *after*
  adding our answer, since joining a group changes that group's count. The
  shortcut `1.0 if any_unique` is right 149 times in 150 and fails when we
  duplicate the field's unique minimum-count clique.
- `pr`'s denominator is **all responders including invalid ones**, not just valid
  answers. Invalid answers carry masked size 0, so they never count as "strictly
  larger", but they do sit in the denominator.

## The problem

Undirected graphs, 290–900 vertices, **density 0.70–0.95** — extremely dense, so
cliques are large (20 to 105 vertices) and there are many near-optimal ones.
Each instance carries its own deadline: **6, 7.5, 10, 15 or 30 seconds**. You may
use 88% of it; the rest is network headroom a live miner needs.

An answer is scored **zero**, not merely penalised, if it is:

- not a clique (any missing edge between two returned vertices)
- **not maximal** — if even one vertex could still be added, it is worthless
- empty, containing a repeated vertex, or containing an out-of-range vertex

Always run a maximality extension before returning. It is nearly free and a
non-maximal answer throws away the whole task.

## Data

| path | what it is |
| --- | --- |
| `data/splits/train.jsonl` | **fully yours.** Graphs *and* every label: what the field answered, the best size, every distinct optimum. Train, tune, memorise, whatever helps. |
| `data/splits/val_problems.jsonl` | 42 graphs + deadlines, no labels. Your steering signal — run it as often as you like. |
| `data/splits/bigger_val_problems.jsonl` | 500 graphs, same format. The **audit** set. Run it rarely. |
| `data/splits/val_labels.jsonl`, `bigger_val_labels.jsonl` | **do not open.** The withheld answers, read only by the scorer. |
| `data/splits/manifest.json` | split sizes and the distribution check |

Both held-out sets are stratified by (deadline, problem tier) to reproduce the
live problem mix, and both are disjoint from train and from each other.

**`val` — 42 tasks, ~10 minutes a pass.** Your working loop. Cheap enough to run
after every change. But 42 tasks resolves the parity rate only to about ±2.4
points, and you will run it dozens of times, so expect to drift into fitting it:
a change that moves val by one or two tasks has told you nothing.

**`bigger_val` — 500 tasks, ~2 hours a pass.** The honest number. Run it **rarely**
— when you think you have made a real step, and before any go/no-go decision.
Not after every tweak, and never in a tuning loop. Its whole value is that you
have not been optimising against it; spend that value only when the answer
matters.

```bash
python3 score_submission.py --solver mymodule:solve                    # val, ~10 min
python3 score_submission.py --solver mymodule:solve --split bigger_val # ~2 h, rarely
```

If `val` and `bigger_val` disagree, `bigger_val` is right.

Every record:

```json
{"uuid": "...", "n": 496, "edges": 104755, "density": 0.85333,
 "time_limit": 30, "difficulty": 0.8, "matrix_b92": "..."}
```

Decode the graph with the repo's own codec:

```python
from CliqueAI.graph.codec import GraphCodec
import numpy as np
A = np.array(GraphCodec().decode_matrix(rec["matrix_b92"]), dtype=np.uint8)
```

`A` is `n x n`, symmetric, zero diagonal. Train records additionally carry
`best_size` (your target for that graph), `best_cliques` (every distinct optimum
the field found), `size_hist`, and `n_at_best`.

## The environment you have

Already set up on this box, so start from it rather than rebuilding it:

| what | where |
| --- | --- |
| compiled solver | `native/clique.cpp` — bitset adjacency, SCC local search, thread portfolio |
| build | `native/build.sh`, one `g++ -O3 -march=native` call; `fastsolver.py` reruns it when the source is newer |
| python entry point | `fastsolver.py` → `--solver fastsolver:solve` |
| fast tuning loop | `score_train.py` — train labels, parallel, core-pinned |

The box is 128 cores (znver2, AVX2 + BMI2, **no** AVX-512), 440 GB RAM, one RTX
A4000. Installed: `g++ 11.4` and numpy. **Not** installed: numba, Cython, torch,
scipy. The C++/ctypes path needs none of them; if you want something else,
install it into a venv rather than the system python.

`min_compute.yml` puts a real miner at 4 recommended / 8 cores, so the solver
defaults to **8 threads** (`SN83_THREADS`). Do not raise it to use this box's
128 — a val number won by 16x the CPU a miner has does not transfer to chain.

### Tune on train, decide on val

`score_train.py` samples train (same (time_limit, difficulty) strata as the
splits) through a byte-offset index and runs tasks in parallel, each worker
pinned to its own block of `--threads` cores so wall clock stays honest:

```bash
python3 score_train.py --n 200                  # ~3 min for 46 min of deadlines
python3 score_train.py --n 400 --time-limit 6   # the tight deadline alone
```

Fix `--seed` when comparing revisions. Train is unrestricted, so tune there and
spend `val` only on decisions — 42 tasks resolve parity to ±2.4 points, and the
brief's warning about fitting the steering set is the reason this loop exists.

`SN83_DEBUG=1` prints per-thread step/add/swap/perturb counts. Look there first
when a change fails to help: it is what caught the current core's predecessor
running 3.1M plateau swaps against 2 adds.

## Solver format

Expose one function:

```python
def solve(adjacency: np.ndarray, time_limit: float) -> list[int]:
    """adjacency: n x n uint8, symmetric, zero diagonal.
       time_limit: seconds you may spend.
       returns:    vertex indices of a MAXIMAL clique."""
```

Score it:

```bash
python3 score_submission.py --solver mymodule:solve
```

For a compiled or GPU solver, produce answers yourself and submit them, one JSON
object per line — `elapsed` is your own measured solve time and is checked
against the deadline:

```json
{"uuid": "...", "clique": [3, 21, 26], "elapsed": 6.9}
```

```bash
python3 score_submission.py --submission answers.jsonl --strict
```

## What you get back

```
==============================================================
SCORE vs BEST RIVAL   (our clique size - best rival's)
==============================================================
  tasks 500

    +2      1    0.2%
    +1      6    1.2%  #
     0    463   92.6%  ###########################################  <-- same as best rival
    -1     28    5.6%  ###
    -2      2    0.4%

  total solve time  6903s of 7146s allowed

BY DEADLINE   (where the time constraint bites)
BY GRAPH SIZE
```

One row per exact delta, with empty bins shown so "never ahead" is visible
rather than implied. The bin at `0` is the one to grow; everything below it is a
task you lost outright.

Plus per-deadline and per-size breakdowns, so you can see whether you are losing
to the clock or to the graph. Add `--json report.json` for the raw per-task rows.

## What is already known — do not rediscover it

Measured against the live field, so you can skip these dead ends:

- **The field is at or within 1–2 of the statistical optimum.** Against a
  first-moment bound on ω, the best rival answer sat *on* the bound on 6 of 15
  sampled instances. You are not looking for a large gap to exploit; you are
  looking to close a one-vertex gap on the hardest instances.
- **The margin is one vertex.** Across ~15,000 logged answers, 71.8% tie for
  best and 27.2% are exactly one short. Almost nothing is two or more behind.
- **More compute does not help.** A greedy multi-start plus (1,2)-swap plateau
  local search was given 1x, 4x and 10x the deadline on three instances and
  gained *exactly zero* vertices each time — 431k iterations to 4.27M, same
  answer. Plateau search saturates. The gap is algorithmic.
- **Parity is solved; the native solver reaches 100%.** The remaining notes are
  historical context for the size race.
- **The baseline in `solver.py` scores 4.8% parity** on the 42-task validation
  set (2 matched, 5 at -1, 16 at -2, and a tail out to -7; mean delta -2.71).
  The best rival, on the same metric, scores 89.0%.
  It is there as a regression floor, not a starting point — beating it is easy
  and means nothing on its own.
- **`networkx.approximation.max_clique`, which the subnet ships as its example
  miner, is far worse still.**
- **The compiled core in `native/clique.cpp` scores 88.2% parity on the
  `bigger_val` audit** (500 tasks: 441 ties, 1 at +4, 50 at -1, 7 at -2), 97.6%
  on val, 91% on a 200-task train sample. That is the current floor to beat, and
  the spread across those three is itself the lesson — val's 42 tasks flattered
  it by nine points, so believe the audit.
- **The remaining gap is graph size, not the clock.** By deadline the audit runs
  82–95% with no ordering; by size it is 100% up to |V| = 500 and then falls to
  92% / 83% / 62% / 73% at 600 / 700 / 800 / 900. Whatever closes the last ten
  points lives on the big graphs.

Where to look instead: the modern local-search family for maximum clique —
LSCC+BMS, MN/TS (multi-neighbourhood tabu), Breakout Local Search, SCCWalk —
over bitset adjacency, compiled. Configuration checking and strong perturbation
are what separate these from the naive plateau search that saturates. Pure
Python will not hold 900 vertices inside 6 seconds.

## Rules

1. Never read `val_labels.jsonl`, and never fit on validation. It is the only
   honest signal you have about the live field.
2. Return a maximal clique. Validity beats size, always.
3. Respect the deadline. A late answer is a zero on chain.
4. `train.jsonl` is unrestricted — the labels there are the same kind the scorer
   uses, so use them freely for tuning and for regression tests.

## Corrections to earlier guidance

Numbers previously circulated that were wrong, so they are not acted on again:

- **γ (the weight exponent) is 16.4, not ~10.** It converges upward with sample
  size (302 rounds → 10.5, 3,002 → 16.43) because sampling noise inflates the
  apparent spread between miners. γ is monotone, so it does **not** change anyone's
  rank — only the payout attached to a rank. It has been rising as the field
  tightens; the cap is 32.
- **The field median is 2.5055**, converged over 3,002 rounds. It is not
  validator-dependent — an apparent 0.04–0.09 gap between validators was a day
  effect; at convergence they agree to 0.0004.
- **`E[1 + difficulty]` is 1.85 on an offline set** (tiers are drawn uniformly by
  `random.choice(PROBLEMS)`), and ~1.80 live, where miner selection probability
  falls with difficulty. Use 1.85 when reasoning about the dataset.
- **Rank has a wide confidence interval.** Our 150-round CI on reward is ±0.057,
  while the middle 80% of the entire field spans 0.068 — the error bar is wider
  than the leaderboard. Bootstrapped rank is 134–244. Rely on "below median (98%)"
  and "bottom quartile (92%)", not on a specific rank.
- **Nothing here is measured live.** All of it is offline replay at 0.88× the
  deadline. `validator.py` passes `timeout=time_limit` to the miner, so on chain
  that budget must also absorb the network round-trip and deserialization. Yuma
  consensus across the seven validators has never been modelled; every alpha/day
  figure is a weight-share proxy.
