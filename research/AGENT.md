# Brief: build a maximum-clique solver that matches the SN83 field

## Your target

For every problem in the validation set, return a clique **at least as large as
the best answer any rival miner produced** on that same graph.

That is the whole objective right now. Not "find the maximum clique" — find one
as big as the best of ~50 competing solvers found, inside the same deadline they
had. The measure is the **parity rate**: the fraction of tasks where your size
minus the best rival's size is `>= 0`.

```
parity rate = (tasks where delta >= 0) / (all tasks)
```

Calibrate against what the live field actually achieves. Scoring every rival
operator on this exact metric — their answer against the best of all the *other*
responders, over 302 rounds and ~15,000 answers:

| operator | parity | mean optimality | mean reward |
| --- | --- | --- | --- |
| best rival solver | **89.0%** | 0.992 | 2.474 |
| second | 80.2% | 0.974 | 2.483 |
| every miner pooled | 71.8% | 0.946 | 2.451 |
| top *earner* (wins on diversity, not size) | 68.2% | 0.943 | 2.501 |
| weakest of the six fleets | 56.4% | 0.883 | 2.458 |
| the shipped baseline | 4.8% | — | 1.755 |

So the milestones are: **60%** means the approach works and you are already
mid-field. **80%** puts you second. **89% ties the best solver on the subnet,
and anything above it makes you the best.**

100% is not the target and is probably not reachable. The label is the maximum
over ~50 competing solvers, so it is a union that no single competitor matches —
the strongest operator on the network still misses it 11% of the time. Do not
read "we lost 11% of tasks" as failure; read it against the 11% the best rival
loses.

One honest caveat about where this leads: the operator taking the largest share
of emission has only the *seventh* best solver here (68.2% parity) and wins on
answer diversity instead. Size parity is the right thing to fix first because it
is a hard floor, but once you are past ~80% the remaining money is in
uniqueness, not in the last few vertices.

**Out of scope for now:** answer *diversity*. The live subnet also rewards
returning a clique no other miner returned, and that is worth more than size
once you are at parity — but ignore it entirely until the parity rate is high.
Chasing uniqueness before you can match sizes makes both worse.

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
- **The baseline in `solver.py` scores 4.8% parity** on the 42-task validation
  set (2 matched, 5 at -1, 16 at -2, and a tail out to -7; mean delta -2.71).
  The best rival, on the same metric, scores 89.0%.
  It is there as a regression floor, not a starting point — beating it is easy
  and means nothing on its own.
- **`networkx.approximation.max_clique`, which the subnet ships as its example
  miner, is far worse still.**

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
