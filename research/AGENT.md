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

Milestones: **60%** means the approach works. **90%** means you are close.
**99%+** is the real bar, and here is why it has to be that high — on chain,
validator weights amplify *rank*, not reward. A solver that trails the field by
one vertex on most rounds does not earn a bit less; it earns nothing. There is
no partial credit, so treat every task you lose as a full loss.

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
| `data/splits/val_problems.jsonl` | graphs + deadlines only. Solve these. |
| `data/splits/val_labels.jsonl` | **do not open.** The withheld answers, read only by the scorer. |
| `data/splits/manifest.json` | split sizes and the distribution check |

Validation is stratified by (deadline, |V| bucket) to reproduce the live problem
mix, and sized so that one full pass costs about **10 minutes** of solve time.
Run it as often as you like.

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
SCORE vs BEST RIVAL
  tasks                     440
  matched or beat rivals    418   ( 95.0%)   <-- the target
  strictly behind            22   (  5.0%)
  invalid / no answer         0   (  0.0%)
  mean size delta          -0.050 vertices

DELTA DISTRIBUTION   (our clique size − best rival's)
   +1     3    0.7%  ## ahead
   +0   415   94.3%  ######################################## same as best rival
   -1    20    4.5%  ## behind
   -2     2    0.5%  # behind

BY DEADLINE   (where the time constraint bites)
BY GRAPH SIZE
```

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
- **The baseline in `solver.py` is not competitive** and is there only as a
  regression floor: ~17% parity, mean reward 1.755, which earns nothing.
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
