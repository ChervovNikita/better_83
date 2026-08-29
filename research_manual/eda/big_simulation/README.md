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
