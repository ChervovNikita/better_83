# SN83 maximum-clique mining — generation 2 results

**Status: closed, 2026-08-23.** Champion unchanged (`v7_fastscan` in
`native/clique.cpp`). One change shipped, one deployment blocker cleared, seventeen
mechanisms tested against the diversity objective — **none beat the baseline**.

This file states only what survives paired, same-round measurement. Thirteen claims made
during the run were withdrawn; the chronological record with every retraction and its
cause is `~/autoresearch-runs/sn83-clique/FINDINGS.md`.

---

## 1. Shipped

**Never go silent** (`fleet_pick`, commit `ba8eacb`). A hotkey with no distinct clique
left submits a duplicate rather than nothing. A repeat earns full optimality (~1.76) and
splits only the diversity term; silence earns zero.

| N | silent | duplicate | gain | 95% CI |
|---|---|---|---|---|
| 10 | 2.1565 | 2.2581 | **+0.1016** | [+0.0489, +0.1573] |
| 20 | 2.0940 | 2.2538 | +0.1598 | [+0.0564, +0.3106] |
| 40 | 1.8596 | 2.2447 | **+0.3851** | [+0.2964, +0.4557] |

Every interval excludes zero, and the effect grows with fleet size exactly as the
mechanism predicts. **This is worth about 2.3x the entire remaining gap at N=40**, and it
turned the fleet from degrading with size into flat.

Origin worth noting: it came from finding that the simulator scored silence as a design
choice rather than as the miner behaviour it models — a bug in the harness, not an
algorithmic idea.

## 2. Deployment blocker: cleared

Champion on `recent_val`, 500 tasks, **2 seconds cut from every deadline** to model the
network round trip:

| | control | tight (-2s) |
|---|---|---|
| parity | 99.800% | 99.600% |
| invalid / over budget | 0 / 0 | **0 / 0** |

Paired: McNemar exact **p = 1.0000** on parity (1 flip); diversity sign test
**p = 1.0000**; reward **-0.0001** (SE 0.0029). One task of 500 flipped — `eb820983`,
n=891 at tl=10, the largest graph at the tightest deadline.

**Safe to deploy at 2s.** If real latency exceeds that, the 6s tier breaks first and
`eb820983` is the canary.

## 3. Where we stand

Per-hotkey median against the field median, deterministic under fixed `PYTHONHASHSEED`:

| N | 5 | 10 | 20 | 40 |
|---|---|---|---|---|
| gap | -0.179 | **-0.170** | -0.174 | -0.183 |

**Gap: -0.1695, bootstrap 95% CI [-0.1968, -0.1449].** Fleet size barely matters between
5 and 30; only very large fleets are measurably worse, and by ~0.013 — a quarter of the
CI width, so not worth re-provisioning for.

## 4. What the headroom actually is

Measured with an oracle that knows each clique's holder count at submit time (not
available to a real miner; this bounds what information would be worth):

| lever | mean | median |
|---|---|---|
| **selection** — better choice from the pool we already hold | +0.0726 | **+0.0125** |
| **reach** — any optimum, including ones nobody has claimed | **+0.0016** | — |

**Reach holds 1% of the available headroom.** Every one of the seventeen mechanisms this
generation was aimed at reach.

## 5. The seventeen mechanisms

All paired, same-round, scored on the submitted prefix.

| mechanism | result |
|---|---|
| H-HULL — exact enumeration (MoMC) inside our pool's hull, 5 hull sizes | null at 90/110/130/180/240 |
| H-HULL budget — enumeration share 40% -> 75% -> 90% | null; MoMC uses 1.5s of a 10.8s allowance |
| H-PERMUTE — relabel the hull, 4 permutations | Jaccard **1.000**; predicted dead in advance |
| H-REGION — hull padding rule high/low/anti/rand | novel share **25.0% in all four arms** |
| H-SPARSE — grow the hull toward sparse pockets | -0.0064, p=0.755 |
| H-FORCEVERTEX — pin a vertex per chain, search G[N(v)] | -0.0113, p=0.64 |
| H-DLSMC — DLS-MC penalties + emitted-clique spike | screened +0.0149; **confirmed -0.0052** at 7 threads |
| H-CHAINMULT — order pool by our own chain multiplicity | -0.0054, p=1.00 |
| H-REPLICATOR — batched Motzkin-Straus, randomised c | converges 6.0 vertices short of omega |
| **portfolio at 4x compute** (3 families x 7 threads) | union 1.65x the pool, **-0.0001** |
| **portfolio at equal compute** (3 families x 2 threads vs champion at 7) | +0.0015, p=0.65; novel share identical to one decimal |
| v23_weighted | +0.0120 (SE 0.0147); McNemar p=1.00 |
| 9 selection predictors (rarest-pick, 5 pool-centrality rules, pool order, chain multiplicity, cross-family agreement) | all null |
| targeting the field's rarest maxima | **-0.1234 — actively harmful** |
| the shipped reference miner as an avoid-signal | 10 vertices short of omega, submitted by nobody in 40 rounds |

## 6. Why no mechanism worked — stated only as far as the evidence supports

**No supported mechanism.** Three explanations were constructed during the run and all
three were withdrawn under within-round testing. What survives is the measurement, not a
story:

- perfect, field-informed reach is worth **+0.0016**, and is significantly NEGATIVE in
  the high-K regime (p=0.049)
- on the median round the field submits **every optimum that exists** — the unclaimed set
  is empty
- we submit **0.0875 below** the field median on the median round; oracle selection
  recovers **+0.0125** of that, about 14%
- nine selection predictors are refuted, and none of them is a function of anything a
  miner can observe at solve time

Whether the field coordinates is **unresolved**: a pairwise detector found 0 coordinated
pairs among 252 hotkeys, but the aggregate independence test is invalid (it models the
optimum count from the observed distinct-clique count, which collisions bias).

## 7. Recommendation

**Ship as-is.** Champion is deployable and robust to a 2s round trip. Run any fleet size
between 5 and 30; avoid 40+.

**Do not spend more on solver diversity without new information.** Reach is 1% of the
headroom. Selection holds the rest and requires each clique's holder count at submit
time, which does not exist on our side.

**Before aiming any continuation, re-derive the brief's STATE block on the frozen round
set.** Two of its premises directed this generation at reach; one ("81.8% of the maxima
we miss are sole-held") is a pooling artifact, and the other ("74.6% of field answers are
unique") measures 52.7% pooled / 61.4% median on current data.

## 8. Reproducing this

`research/tools/` holds the measurement scripts and vendored MoMC (MIT).

**Read `tools/pooling_guard.py` first.** Eleven of the thirteen retractions in this
generation were one error: a statistic pooled across rounds of unequal size, reported as
a property of the solver. Round size correlates with optimum count, holder count and
collision rate here, so pooling manufactures effects that vanish within round. The guard
computes any contrast both ways and flags the discrepancy.

The paired scoring path (`score_pools.py`, `compare.py`) produced no false positives in
this generation. **Every retraction came from analysis written outside it.**
