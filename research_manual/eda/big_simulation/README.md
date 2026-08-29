# big_simulation — two-player fleet-split sweep

Player **A** holds the majority (`floor(N/2)+1 … N-1` hotkeys) and commits first.
Player **B** holds the rest, sees A's board, and answers with an exact best
response. Every split plays the same solved rounds; the output is both players'
reward distributions.

## Run

```bash
.venv/bin/python -m research_manual.eda.big_simulation \
    --n 249 --strategy greedy --responder full \
    --out research_manual/eda/big_simulation/out
```

| flag | default | meaning |
|---|---|---|
| `--n` | 249 | hotkeys split between the two players |
| `--strategy` | `greedy` | A's strategy, any name in `strategies.REGISTRY` |
| `--responder` | `full` | B's information level, any name in `responders.REGISTRY` |
| `--step` | 1 | stride over splits |
| `--limit` | 0 | use only the first N rounds |
| `--pool` | `research_manual/pool_n20.jsonl` | solved pool dump |
| `--rounds` | `research_manual/rounds.json` | round dump holding the real answers |

Writes `sweep_<strategy>_<responder>.json` and four images tagged the same way.

`table.py --g <A's size>` runs one split through both strategies and both
responders at once, reusing A's board across responders, and writes
`out/split_<g>.json` with a `(q_a, q_b, mean_a, mean_b)` row per round. The
answer counts are in the rows because the headline figure is **pooled** —
`sum(q * mean) / sum(q)`, the share of total reward per answer — and a mean of
per-round means is a different quantity.

## Where the rounds come from

Nothing is generated. `rounds.py` reads **938 real rounds** whose cliques the GPU
harvester already enumerated (`pool_n20.jsonl`, written by `SN83_POOL_DUMP` with
`SN83_FULL_POOL=1024`), joined by uuid to `rounds.json` for the field's real
answer count.

| per round | source |
|---|---|
| `omega` | solved |
| `n_top` — distinct maximum cliques | solved (`n_top_true`) |
| `n_spare` — distinct omega-1 cliques | solved (`n_spare_true`) |
| `difficulty` | vertex count, exact |
| `n_answers` — miners who answered | `rounds.json` |

`q_a` / `q_b` divide that real answer count by the fleet ratio.

Measured over the 938: omega median 36, P median 32 (range 1–1024), spare
median 456, field answers median 54 (range 10–97).

## Files

| file | role |
|---|---|
| `scoring.py` | closed-form scorer, O(#cliques) |
| `rounds.py` | loads the solved rounds |
| `strategies.py` | A's strategies, `@strategy("name")` |
| `optimal.py` | B's best response with full sight |
| `partial.py` | B's best response knowing only the occupancy multiset |
| `responders.py` | B's information levels, `@responder("name")` |
| `sweep.py` | the split sweep |
| `plots.py` | the four images |
| `verify.py` | cross-checks against slower references |
| `table.py` | one split, both strategies x both responders |
| `native.cpp` | C++ best response, maximin and bayes |
| `native_partial.cpp` | C++ expectation over the random matching |

## What is exact and what is not

`verify.py` runs every check below on small instances against brute force.

| check | scope |
|---|---|
| `check_scoring` | the scorer against `research/fleet_sim.score_round` |
| `check_optimal` | B's full-sight response against every reply |
| `check_partial` | B's fast partial response against every multiset |
| `check_optimal_grid` | a covered grid of round shapes |
| `check_prior_occupancy` | boards that already carry B hotkeys |
| `check_weighted` | the pooled-objective response |
| `check_maximin` | A's board against **every partition** of its budget |
| `check_bayes` | the same under A's posterior objective |

Two searches are families, not enumerations, because A's move space is the set
of partitions of up to 125 hotkeys:

- **A's board** (`native.cpp: a_candidates` + `climb`). The family sweeps the
  split between the two size classes and, for each target minimum, the widest
  spread reaching it; steepest ascent over single-hotkey moves then refines it.
  The family alone is short of the exhaustive optimum on 8/400 small instances
  (worst 0.090); with the climb it is 0/400. That is evidence, not proof.
- **B's partial response** (`partial.py: best_response_fast`). Even spreads
  alone cannot express a skewed assignment like `[4,1,1,1,1,1]`, which is the
  exhaustive optimum often enough to matter, so the family is even-over-`j`
  plus one deepened head, with `j` no longer subsampled. 0/160 against the
  exhaustive reference on medium views, 0/300 on small ones.

`partial.best_response` is the exhaustive reference: no stack cap, no width
grid, no shape family, no pruning. It is not usable at fleet scale.

## Adding a strategy or responder

```python
@strategy("mine")
def mine(rnd, q, score):
    return [(rnd.omega, 1, 0)] * min(q, rnd.n_top)

@responder("mine")
def mine(board, rnd, q, rng):
    return board, mean_a, mean_b
```

Then `--strategy mine` / `--responder mine`.

## Information levels

`full` sees which clique carries which occupancy and answers deterministically.

`partial` is told only the multiset per size class: for cliques
`[w w w-1 w-1 w-1]` holding `[2 3 1 1 1]` it learns `w: {2,3}`, `w-1: {1,1,1}`.
Choosing a clique is then a draw against a uniformly random matching, so it
maximises expected gap and places at random. On that board with `q=2` and
nothing free, `w:[1,1]` scores +0.0857 against `w:[2,0]` at +0.0023 — the gap is
what the missing information costs, and splitting is insurance, not
diversification.

## Correctness

```bash
.venv/bin/python research_manual/eda/big_simulation/verify.py
scoring vs fleet_sim        : max disagreement 1.33e-15
full responder vs brute     : 0/200 below, worst 0.000000000
fast partial vs exhaustive  : 0/300 below, worst 0.000000000
```

`partial.best_response` searches every multiset; `best_response_fast` restricts
to even spreads with bounded grids and matches it. Both were also checked
against a labelled brute force that enumerates placements on identified slots
and averages over every permutation, i.e. one that assumes no multiset
reduction.

## Assumptions

Both players can find any clique the solver enumerated. The sweep models the
allocation game at equal enumeration, not the enumeration race — on the live
subnet those four coldkeys occupy ~21 maximum cliques per round where 454 exist,
so A here is stronger than the real one.
