# The brief's STATE block, re-derived on the frozen 279-round set

Generated 2026-08-23. Every heartbeat of generation 2 carried a STATE block inherited
from generation 0/1. Two of its claims aimed the whole generation at *reach*. Re-measured
on the current frozen round set:

| brief claim | brief value | measured now | verdict |
|---|---|---|---|
| "we return an un-taken clique on ~36% of tasks" | ~36% | **24.7%** of rounds | **overstated** |
| "only ~37.5% of instances have ANY un-taken clique in our pool" | ~37.5% | **44.8%** of rounds | **understated** |
| "74.6% of the field's answers are unique" | 74.6% | **54.0%** pooled, **61.4%** median | **overstated by 13-21 pts** |
| "the field covers ~25 optima" | ~25 | **23.6** | accurate |
| "none of the field's cliques shared by >7" | <=7 | max observed **7** | accurate |
| (context) responders per round | — | 53.7 | — |

## The one that changes the framing

The brief pairs *"we return an un-taken clique on 36% of tasks"* with *"only 37.5% of
instances have any un-taken clique"*. Together those say we capture **96%** of what is
available — i.e. nothing left to get, so the constraint must be reach.

Measured, the same ratio is **24.7 / 44.8 = 55%**. Un-taken cliques are available almost
half the time and we submit one about a quarter of the time.

**That is a selection gap, not a reach gap** — and it is consistent with the oracle
measurements this generation produced independently: reach headroom **+0.0016**,
selection headroom **+0.0125 median / +0.0726 mean**.

## Caveats, stated plainly

- I do not have the original gen-0/1 computations. The differences may come from a
  different round set (this one was frozen later), a different denominator, or a
  different definition of "un-taken". Any of those would explain them without anyone
  having been wrong.
- Two of five claims reproduce exactly, so the block is not broadly unreliable.
- The three that differ do so in **different directions**, which argues against a single
  systematic cause and for genuine definitional drift.

## Recommendation

Use this table, not the inherited block, to aim any continuation. The inherited version
supported "reach, not selection, is the constraint"; the re-derived version does not, and
neither does any oracle measurement taken this generation.

Reproduce with the snippet in `~/autoresearch-runs/sn83-clique/FINDINGS.md` under
"auditing the BRIEF's own premises", or re-run against `data/sim_rounds.jsonl` +
`data/sim_ts.jsonl`.
