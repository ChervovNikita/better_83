# SN83 maximum-clique mining — results

**Status 2026-08-23.** Champion unchanged: `v7_fastscan`, promoted to
`native/clique.cpp`. Fifteen mechanisms were tested against the DIVERSITY objective this
generation; **none beat the baseline**, and the reason is specific rather than
exhaustion (section 8). One change shipped, one deployment blocker closed, one
operational recommendation.

Everything below is measured. Two headline claims made during the run were later
**retracted** and are recorded as such — the retractions are kept deliberately, because
the corrections are as informative as the results.

Full working notes, including every refuted mechanism and the bugs found in the
instruments: `~/autoresearch-runs/sn83-clique/{FINDINGS,EXPERIMENTS,PLAN,IDEA_ANGLES}.md`.
Reproduction scripts: `research/tools/`.

## 1. The champion

`recent_val`, 500 tasks, 8 threads x 1 worker:

| | value |
|---|---|
| parity with the field's best | 99.800% (1 loss / 500) |
| reward | 2.4317 |
| optimality | 0.9997 |
| diversity | 0.5802 |
| invalid answers | 0 |
| over budget | 0 |

**Optimality is saturated and cannot be improved.** A 5x deadline produced 149 ties and
zero extra vertices. All remaining headroom is diversity.

## 2. What the objective actually rewards (verified in source)

`reward = optimality x (1 + difficulty) + diversity`, and in
`CliqueAI/scoring/clique_scoring.py`:

- `diversity()` = `1 / Counter[tuple(sorted(clique))]` — the count of the
  **byte-identical vertex set**. A clique differing from a rival's by one vertex scores
  exactly as well as one differing by thirty. Distance and dispersion are not measured.
- `optimality()` normalises against the best size among **that round's responses**, so
  certifying true omega is never required.
- `is_valid_maximum_clique` tests **maximality**, not maximumness.

Measured value of one submitted clique by how many field miners already hold it:

| holders before us | n | mean reward |
|---|---|---|
| **0 (novel)** | 2535 | **2.8524** |
| 1 | 1146 | 2.3180 |
| 2 | 850 | 2.1596 |
| 4+ | 474 | 1.9696 |

Paired within round, novel beats already-held by **+0.5322 on 116/116 rounds**. The
entire cliff is between 0 and 1 co-holders.

## 2a. Final measured position (deterministic, PYTHONHASHSEED fixed)

Per-hotkey median against the field median on identical rounds, `fleet_pick:picker`:

| N | our median | field median | gap |
|---|---|---|---|
| 1 | 2.1872 | 2.4276 | -0.2403 |
| 5 | 2.2489 | 2.4276 | -0.1786 |
| 10 | 2.2579 | 2.4276 | -0.1697 |
| **20** | **2.2591** | 2.4276 | **-0.1685** |
| 40 | 2.2442 | 2.4276 | -0.1833 |

Best fleet size is N=10-20 and the curve is flat between them; N=1 and N=40 are both
worse. **The gap to the field median is -0.169.** That is the number this run set out to
close, and it is unchanged by thirteen mechanisms.

## 3. The gap, decomposed

Per-hotkey median against the field median, on identical rounds, deterministic:

| picker | N=1 | N=5 | N=10 | N=20 | N=40 |
|---|---|---|---|---|---|
| silent (old behaviour) | -0.267 | -0.237 | -0.271 | -0.334 | -0.568 |
| **duplicate** | -0.249 | -0.179 | **-0.170** | -0.174 | -0.183 |

**Shipped win:** never going silent. A hotkey with no distinct clique left submits a
duplicate rather than nothing; a repeat earns full optimality (~1.76) and splits only
diversity, silence earns zero. This removed the fleet-size penalty entirely and is
worth +0.39 at N=40. Commit `ba8eacb`.

Splitting the remaining -0.170 (per-clique basis, 279 rounds):

| | mean reward |
|---|---|
| first 8 of our pool (today) | 2.3618 |
| best 8 of the SAME pool (oracle selection) | 2.4291 |
| field mean | 2.4885 |

- **selection: +0.067** — real, better on 143/143 rounds, and **CLOSED**. Eight
  predictors refuted (v19 rarest-pick, five pool-centrality rules, pool position, chain
  multiplicity). Every one is a function of our own pool or our own search. Which
  optimum a rival submits is a property of the rival's solver and no statistic
  computable from our side predicts it. The ceiling is visible in hindsight and
  unreachable in advance.
- **reach: +0.059** — the only remaining lever. Requires cliques the field's solvers do
  not reach AT ALL. Target: novel share of the submitted prefix from **30.2% -> 54.1%**.

**The cliques needed to close it EXIST.** Union-hull oracle (exact enumeration inside
the vertices spanned by our maxima and the field's; uses field data, so a measurement
and not a mechanism), 80 rounds, 20% timeouts:

| per round | count |
|---|---|
| maximum cliques that EXIST in the union region | **46.5** |
| union we and the field actually find | 22.5 |
| our pool | 14.0 (30.1% of what exists) |
| the field | 18.5 |
| **reached by NOBODY** | **24.6** |

More maxima go unclaimed than we and the whole field find combined. Against a decision
rule fixed in writing beforehand (E~31-35 = supply exhausted, stop; E>>40 = continue),
E=46.5 selects continue. **The remaining gap is not a supply ceiling — it is a reach
failure, and the targets are demonstrably there.**

## 4. Refuted mechanisms (this generation)

| mechanism | result |
|---|---|
| H-HULL — exact enumeration (MoMC) inside our pool's hull | +0.0002, p=1.00, 120 rounds x 3 hull sizes |
| H-REPLICATOR — batched Motzkin-Straus, randomised c | converges 6.0 vertices short of omega |
| H-CHAINMULT — order pool by our own chain multiplicity | -0.0054, p=1.00 |
| H-HULL at 90/110/130/180/240 | null at every size; budget disproved as the cause (MoMC uses 1.5s of a 10.8s allowance) |
| H-PERMUTE — relabel the hull, 4 permutations | Jaccard **1.000**, union = 1.00x a single run. Predicted DEAD in advance |
| H-REGION — hull padding rule high/low/anti/rand | novel share **25.0% in all four arms**; padding genuinely occurs on 78% of rounds |
| H-FORCEVERTEX — pin a vertex per chain, search G[N(v)] | -0.0113, p=0.64 (fixed binary; implementation verified clean) |
| pool ORDER (last-8, spread-8) | -0.015, -0.013 |
| 6 pool-centrality selection rules | -0.003 to -0.010, p 0.07-0.93 |
| targeting the field's RAREST maxima | **-0.1234** — actively harmful |

Earlier generations additionally refuted: reservoir, cold restarts, vertex banning,
seed variation, parameter diversity, AMTS, (1,2)-swap, cheap scoring, learned
selection, GPU in all forms, QUBO/Ising, GNNs, exact branch-and-bound on the full graph.

**Why they all fail, correctly stated.** Two independent omega-cliques share `omega^2/n`
vertices in expectation — 3-4 on our instances — so moving between optima means
dropping ~27-37 vertices. Every perturbation mechanism searches a radius that provably
cannot reach another optimum. This is combinatorics, not the overlap-gap property; the
earlier OGP framing was wrong and the d<=8 probe was measuring only that 8 < 27.

## 5. Simulator corrections (commit `ba8eacb`)

Four defects, all of which moved numbers:

1. **`replay` was not deterministic across processes** — it iterated a set of hotkey
   strings, so per-process hash randomisation broke eviction ties differently each run.
   The N=40 median wandered ~0.04 on identical seed and data. Now sorted, guarded by a
   subprocess invariant across `PYTHONHASHSEED`. **Fleet figures dated before
   2026-08-22 are good to ~0.05, not to three decimals.**
2. Immunity protected no incumbent — 14 of 16 immune uids were being evicted.
3. `score_vector` reserved slots for already-deregistered fleet members.
4. The victim set was derived from the metagraph in tests and from the data in `replay`.

## 6a. The one live lead: H-HULL's null is a BUDGET artifact

Block J separated two candidate causes of the hull lever's failure, at S=180 over the
rounds that carry unclaimed cliques, with a 20s enumeration budget:

| | value |
|---|---|
| maxima found in OUR OWN hull | **108.5/round** |
| unclaimed cliques that are inside our hull | 62.8% |
| ...and actually RETURNED by MoMC | **100.0%** |

**MoMC is complete** — it returns every unclaimed clique the hull contains — so the five
H-HULL nulls are not an instrument artifact. But the yield is entirely a function of
time: 108.5 maxima at 20s against **30.4** at the 4.78s budget the earlier runs used,
same hull and same rounds. And the ~30 found first are the CONTESTED ones, because
search order starts in the attractor.

Of the 108.5 found per round, 52.9 are unclaimed by anyone — **48.8% of the pool**
against our current submitted-prefix novel share of 30.2%. At the measured +0.5322 per
conversion that is **+0.099, about 58% of the -0.170 gap**, from a pool built with no
field data.

**WITHDRAWN pending Block L** — the 108.5 figure is measured on the 11 rounds that had
unclaimed cliques (selected on an outcome correlated with clique count), while 30.4 is
the mean over 120 unselected rounds. The comparison is confounded and the +0.099
estimate built on it is withdrawn. What survives: MoMC is complete (returns 100% of the
unclaimed cliques its hull contains) and our hull contains 62.8% of them.

The original claim was: **the obstacle is cost, not reachability** — the first time this generation that has
been true. ~20s of exact enumeration against a 6-30s deadline that must also fund the
local search. Block L is testing whether raising enumeration's share of the budget from
40% to 75/90% realises the predicted gain.

I mis-attributed this null twice before getting here: first as "larger hulls are worse"
(from timeouts on a different code path at a flat cap), then as "the deployable version
does not work". Both were wrong, and both were recorded as general conclusions when
they were scoped to one budget.

## 5a. Why everything failed — one sentence

**The quantity we would need to condition on is not observable from our side at solve
time.** Selection failed because which optimum a rival submits is a property of the
rival's solver. Region-targeting failed because the optima we miss are statistically
indistinguishable from the ones we hold — degree rank 0.3355 against our 0.3283. Twelve
mechanisms, two directions, one wall.

And the sharpest single measurement: on the clique-richest rounds our hull yields
**133.9 maxima against LSCC's 37.9** and the reward delta is **negative**. We can
already generate 17x more cliques than we submit. Producing more is not the problem;
knowing which to send is, and that information does not exist on our side.

## 6. Open

- ~~tight-deadline robustness~~ **RESOLVED, safe to deploy.** 2s cut from every
  deadline: parity 99.800% -> 99.600% (McNemar p=1.00), diversity sign test p=1.00,
  reward delta -0.0001 (t=-0.04), **0 invalid, 0 over budget**. One task flipped of 500
  (`eb820983`, n=891, tl=10, 38->37) — the largest graph at the tightest deadline.
- **H-DLSMC** (`v27_penalty`) — **REFUTED at production width**: 7 threads, 200 rounds,
  **-0.0052, p=0.78**, novel share 32.4% -> 31.6%. The pre-registered difficulty test
  failed: screening measured +0.1994 at d>=0.9 on n=7, confirmation measures -0.0207 on
  n=56. The screening result was seven lucky rounds. Original screening read: **+0.0069, p=0.39, novel share 30.2% -> 32.0%**, pool 11.79 ->
  12.62, 0 invalid. The novel-share move (+1.8 pts) independently predicts +0.0096
  against +0.0069 observed, so the mechanism is doing what it claims — but it is ~25x
  too small and not significant. Real effect, wrong order of magnitude. 2 arms left.
- **H-HULL is refuted only for hulls <= 130.** The oracle enumerated a 177-vertex hull
  in 0.96s, a budget H-HULL never tested; its "larger hulls are worse" conclusion came
  from timeouts on a different code path at a flat 15s cap. Block H is measuring whether
  an OUR-POOL-ONLY hull, made larger, contains the unclaimed cliques — if so the lever
  is deployable with no field data.
- Phi1/Phi2-regularised replicator — open, poor prior.
- Whether the optimum supply is genuinely exhausted. A research sweep argued our 18.7
  maxima/round is near the ER-model ceiling; our own union with the field is 31.3 on
  real rounds, so that is not supported, but it is untested at our density.

## 7. Methodological rules earned the hard way

- **Only paired, same-round, same-K comparisons on the submitted prefix are
  admissible.** Six wrong answers in this project came from averaging a ratio across
  rounds of unequal size, including a retracted "62.2% novel" headline.
- **A control that matches the treatment too exactly is evidence of a dead code path,
  not of a null result.** Two of my own variants had bugs found this way.
- Score the mechanism on the quantity the objective pays for. AMTS was closed for
  finding no bigger cliques — the wrong test for a diversity objective.

---

## 8. Recommendation

### Ship as-is
The champion is deployable now. It is robust to a 2s network round trip (parity
99.800% -> 99.600%, McNemar p=1.00, 0 invalid, 0 over budget), and the picker fix
(`ba8eacb`) is worth **+0.39 reward at N=40** — it is what turned the fleet from
degrading with size into flat.

**Run N=10-20 hotkeys, not 40.** Measured gap to the field median: -0.1697 at N=10,
-0.1685 at N=20, **-0.1833 at N=40**. Beyond ~20 the extra hotkeys are queried on rounds
where our pool must duplicate, and duplication splits the diversity term.

### Do not spend more on solver diversity without new information
Fourteen mechanisms were tested against the diversity objective this generation and none
beat the baseline. The reason is specific, not exhaustion:

- **Selection is closed.** Eight predictors refuted. Which optimum a rival submits is a
  property of the rival's solver. The oracle ceiling is +0.075 to +0.10 and is real
  (better on 143/143 rounds) but unreachable in advance.
- **Producing more cliques does not help.** On the clique-richest rounds we already
  generate **133.9 maxima and submit 8**, and the reward delta there is negative.

### The one lead worth resuming
Paired within round, the maxima we miss sit in **sparser pockets** — lower
common-neighbour density (p=0.0117) and lower degree (p=0.0220) than the ones we find.
It fails as a selection rule (significantly: -0.0289, p=0.0161) because within our own
pool the sparser cliques are the ones other solvers also reach. It was tested as a REACH rule
(H-SPARSE, growing the hull toward sparse pockets): **-0.0064, p=0.755, null.** The
signal describes where the missed optima live but does not localise WHICH pocket, and
the effect is far smaller than the within-clique spread. **This line is finished and the
residual -0.169 is structural.**

### What would actually change the answer
Nothing in the solver. The gap is an information problem: we cannot observe, at solve
time, which optimum the ~250 other miners will submit. The levers that remain are
outside this codebase — more hotkeys (measured: does not help past ~20), or information
about the field that a miner does not have.

## 9. Reproducing any of this
`research/tools/` holds the measurement scripts and vendored MoMC, with a README stating
the one rule they encode: **only paired, same-round, same-K comparisons on the submitted
prefix are admissible.** Eight wrong answers in this project came from violating it,
including two of my own headline claims that were retracted.
