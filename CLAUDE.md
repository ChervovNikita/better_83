# CLAUDE.md — SN83 maximum-clique mining

## The rule that overrides convenience

**Explore however you like. Final numbers come from the simulator.**

Run whatever scratch harness makes an idea cheap to test — a 20-round probe, a log-only
sweep, a two-round smoke. That is how most of the useful work here gets found, and nothing
below discourages it.

But **no number enters RESULTS.md, a commit message, a shipped default, or a report to the
user unless it came from `research_manual/simulate.py`**. Scratch harnesses produce
direction. The simulator produces numbers.

### Why this rule exists

On 2026-08-24 a generation of coordinator work was measured by sixty-six ad-hoc scripts
calling `tools/round_score.score(sizes, keys, difficulty, valid=None)`. None passed `valid=`.
The validator scores `size * is_valid_maximum_clique(response)`, so every one of those runs
credited cliques the validator would have rejected — 5% of delete-and-resolve pool entries
and 27% of the omega-1 spares the shipped rule actually picks, each costing ~0.73 per answer
through the fallback path.

`simulate.py` makes that mistake impossible by construction: it hands the round's responses
to the validator's own scorer rather than reimplementing it.

    calc = CliqueScoreCalculator(graph=graph, difficulty=..., responses=responses)
    *_, rewards = calc.get_scores()

It also replays the real field — every historical answer from `rounds.json` — so a number is
a reward against the actual opponents, not against an assumed one.

A default was shipped on a number from the wrong harness. The scratch tool was not wrong to
exist; it was wrong to be the last word.

### The check, before any fast harness is trusted

Diff its contract against `simulate.py`'s and write down what is being dropped. **A parameter
that is REQUIRED in the existing tool and OPTIONAL in yours is not a simplification — it is
the check you are about to skip.** In particular: anything that scores a clique without
running the validator's maximality test is measuring a reward nobody will ever be paid.

### Corollary

A smoke test is not a benchmark. Four rounds in forty seconds proves the code path executes;
it measures no reward. Say which one you ran when reporting.

## Other standing constraints

Full operating rules live in `~/autoresearch-runs/sn83-clique/STANDING_ORDERS.md`. The ones
that bite most often:

- CFS quota is ~15 CPUs though `nproc` says 128. `workers * threads <= 15`, always. Thread
  count changes the ANSWER here, not just the speed.
- Chain runs on LOG MARKERS, never `pgrep` — and a `pgrep -f` pattern naming a script matches
  its waiter too, so killing the match set kills the chain.
- Paired comparisons only: same rounds, same K, sign test over CHANGED answers.
- Draw rounds RANDOMLY within a stratum. `sorted(rounds, key=time_limit)[a:b]` is a deadline
  slice, and deadline proxies graph size, which drives how many miners answer.
- `research/` is the measurement apparatus and stays reviewable. Fix production defects in
  `CliqueAI/`, not by editing the thing that measures them.
