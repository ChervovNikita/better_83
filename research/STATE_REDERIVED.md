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

---

# 2026-08-23 evening — the STATE block after the deployment audit

The corrections above are about the research solver. This section corrects claims that
were true of the research solver and false of production, plus claims the 1000-round
paired run superseded.

| STATE claim | status | replacement |
|---|---|---|
| "SHIPPED champion v7_fastscan, promoted to native/clique.cpp" | **misleading** | Promoted WITHIN the research harness. `CliqueAI/miner.py` called `nx.approximation.max_clique` until commit c67afbe. Production never ran it. Worth +0.7276/answer |
| "optimality is pinned at 1.0; all headroom is diversity" | true of research, **false of production** | Production's greedy approximation reaches ~76% of omega. Optimality is pinned only once the champion is wired in |
| "SHIPPED: never-go-silent picker, +0.385 at N=40" | **needs a coordinator** | Measured with one shared pool and rank assignment. Deployed, each hotkey is a separate process with no shared state. Hash assignment costs -0.1664/hotkey (1000 rounds) |
| "Reach, not selection, is the constraint" | **retracted** | Nine reach mechanisms null; novel share pinned at 25.0% across 3 hull sizes and 4 padding rules. The constraint is SELF-COLLISION: 16.26% of our answer pairs identical, against 0.00-0.98% for field entities |
| "the field is ~34 coordinated miners covering ~25 optima" | **wrong shape** | FIVE entities by coldkey linkage: 77 / 60 / 44 / 38 / 20 hotkeys. ~24 distinct cliques per round at 1.77 holders each |
| "the gap is -0.14 like-for-like" | **cache-dependent** | -0.2922 on a cache with max-size median 4; -0.14 came from one with median 15. Gap size tracks harvest depth |
| "diversity is SIZE-BLIND, pays for uniqueness at any size" | true but **incomplete** | It pays through `pr`, which counts answers strictly larger. So a unique omega-1 is cheap only when few answers sit above it -- which is why the field spreads exactly when nOm is small |

## What actually constrains us, measured on 1000 paired rounds

    assignment (rank vs hash)   +0.2059 gap closure   PRIVATE
    backfill (omega-1)          +0.0137 gap closure   mostly a public good
    everything else measured this generation:  null

The single largest measured effect in the whole project is not a solver property. It is
that `CliqueAI/miner.py` did not call the solver, and that on any branch but main the
miner restart-loops every 12 seconds without serving a request.
