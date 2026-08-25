# GPU clique harvester — what is built, what it measures, what it does not claim

Implementation of `gpu_clique_design.md`. The design's status line said *"design
under review, nothing built"*; this directory is the build, plus the gates that
were supposed to decide whether each stage earned the next one.

`clique.cpp`, `fleet_solver.py` and `fleet_pick.py` are untouched, and so is
everything under `research/` and `simulate.py` — the measurement apparatus stays
reviewable. The GPU path is a sibling of `fleet_solver.solve_many` behind the
same signature; the only edit outside this directory is `research_manual/solver.py`
choosing which of the two to import.

```
clique_gpu.cu         device search + scheduler + the C ABI + a host mirror
gpu_lib.py            build (nvcc) and ctypes bridge
fleet_solver_gpu.py   solve_many(adjacency_matrix, time_limit, k)  — stage 5
gates.py              the §8 gates, one subcommand per stage
```

## Quick start

```bash
cd research_manual/eda
python3 gates.py all --quick          # smoke test: proves the path runs, measures nothing
python3 gates.py all --both-arms      # the real gates
python3 gates.py stage3 --rounds 16   # paired GPU-vs-CPU at matched wall time
```

Everything builds on import (`rm libcliquegpu_*.so` is a complete rebuild). nvcc
is at `/usr/local/cuda/bin/nvcc` here and is found automatically; `SN83_NVCC`
overrides it.

## Where the numbers in this file come from, and what they are worth

**Direction, not reward.** CLAUDE.md's rule binds this work as much as any other:
no number enters RESULTS.md, a commit message, a shipped default or a report to
the user unless it came out of `research/fleet_sim.py`, whose
`score_round(sizes, valid, keys, difficulty)` takes a **required** `valid=` and
whose `validate_cliques` runs the validator's own maximality test.

Everything below is from `gates.py`, which counts **distinct cliques** — a proxy
for the diversity term, not the term itself. The reward measurement is

```bash
cd research
python3 fleet_sim.py --solver fleet_solver:solve_many --sizes 1 5 10 20 40 --rounds 1000 --solve

PYTHONPATH=../research_manual/eda \
python3 fleet_sim.py --solver fleet_solver_gpu:solve_many --sizes 1 5 10 20 40 --rounds 1000 --solve
```

(`fleet_solver_gpu` is imported by bare name off PYTHONPATH; it loads
`research_manual/fleet_solver.py` by explicit path, so `research/`'s
same-named placeholder cannot shadow the CPU arm it compares against.)

That has NOT been run. Until it has, the honest claim is "more distinct cliques
per round in a paired same-wall-clock comparison", not a reward figure. What is
verified end to end is that the answers survive the validator's own test:
`fleet_sim.validate_cliques` returns True for every clique this path returns.

## Stage results

Hardware: RTX A4000, sm_86, 48 SMs, 16 GB, 4 MB L2. CUDA 12.4.

### Stage 0 — per-walker step throughput (§1, the load-bearing gate)

`gates.py stage0 --both-arms --steps 20000`, rounds drawn from `rounds.json` at
each of the four sizes §8 names.

| arm | n=292 | n=497 | n=700 | n=892 | walkers |
|---|---|---|---|---|---|
| carry, 32 lanes | 258k | 194k | 186k | 165k | 1152 |
| prefix/suffix, 32 lanes | 89k | 34k | 45k | 41k | 768 |
| carry, 64 lanes | 245k | 202k | 209k | 184k | 576 |

(solo steps/s, one walker alone on the card.)

Three things the design did not know:

1. **Per-walker throughput is ~2x better than the estimate.** §1 guessed ~1e5
   steps/s against ~1e6 for a CPU thread; measured is 1.6–2.6e5 solo and
   6.5–7.5e4 at full occupancy. The 10x per-walker deficit §1 worried about is
   real at full occupancy but not worse than feared.

2. **The prefix/suffix arm is 4–6x slower**, not the ~2x I expected when I chose
   the carry form. Its 41 KB of per-walker state has to live in global memory,
   which both costs bandwidth and drops occupancy from 1152 walkers to 768. It
   is kept, it is correct, and stage 1 proves it computes the same sets — but on
   this card it is not the arm to run.

3. **`LANES_PER_WALKER` does not do what §3 expected.** §3 offers 64/128 lanes as
   the response to a failed depth gate: "~4x fewer concurrent walkers, each
   ~2–4x faster". Measured: 64 lanes is **not faster per walker at all**
   (202k vs 194k at n=497) and halves the walker count. The step is not
   lane-bound — at W=15 words, 32 lanes already cover the candidate recurrence
   with the (i,w) tiling, so the extra lanes have nothing to do. **The dial in §3
   is not available on this shape of problem.** If per-walker depth ever has to
   go up, it has to come from a cheaper step, not from more lanes.

Fraction of walkers reaching the field's omega inside a fixed 20k steps: 100% at
n=292, 88% at n=497, 14% at n=700, 2% at n=892. So on the big graphs a fixed
short slice mostly does *not* reach omega — which is exactly the failure mode
`fleet_solver`'s `BAN_FRAC *= 2.0` exists for, and the device carries the same
fix (short-return doubling, §5).

### Stage 1 — candidate sets vs brute force (§8)

`gates.py stage1 --graphs 100 --trials 1024`

100 random graphs (n from 64 to 900, density 0.5–0.95), states that are half
real cliques and half arbitrary vertex subsets, with 0/1/4 bans:

```
lanes=32 carry           100 graphs, 307200 states checked, 0 mismatched
lanes=32 prefix/suffix   100 graphs, 230400 states checked, 0 mismatched
```

The reference counts `slack(v) = |C| - |N(v) cap C|` one member at a time — the
definition, not a second clever implementation. Both device arms are bit-equal
to it on every state.

### Stage 2 — trajectory diff (§8)

`gates.py stage2 --graphs 10 --trajectories 10 --jobs 512 --steps 20000`

```
200 trajectories, 0 mismatched | 512 answers, 0 not cliques, 0 not maximal in full G
```

The trajectory hash folds in every add/drop with its vertex and its kind, in
order, so a match is move-for-move agreement rather than the same final size.
§4 says three things cannot be bit-identical to `clique.cpp` (RNG, BMS
tie-breaks, clock); the host mirror in `clique_gpu.cu` makes the *same three
choices*, which is what leaves anything for the diff to bind.

### Stage 3 — distinct cliques vs the CPU arm at matched wall time (§8)

`gates.py stage3 --rounds 16 --k 10`. Same 16 rounds, same k, same wall clock on
both arms; rounds drawn randomly from `rounds.json` (not a deadline slice —
deadline proxies graph size). The GPU arm is the hybrid: `clique.cpp` pins omega
for `CHAMPION_SHARE`, the device spends the rest widening the pool.

| n | tl | CPU omega / distinct | GPU omega / distinct | field omega / distinct |
|---:|---:|---:|---:|---:|
| 297 | 6.0 | 23 / 7 | 23 / **11** | 23 / 11 |
| 894 | 10.0 | 86 / 7 | 86 / **83** | 86 / 21 |
| 697 | 7.5 | 32 / 7 | 32 / **10** | 32 / 10 |
| 698 | 15.0 | 87 / 4 | 87 / **32** | 87 / 32 |
| 694 | 6.0 | 63 / 10 | 63 / **609** | 63 / 54 |
| 894 | 7.5 | 36 / 10 | 36 / **244** | 36 / 17 |
| 893 | 7.5 | 57 / 8 | 57 / **281** | 57 / 18 |
| 696 | 15.0 | 35 / 1 | 35 / 1 | 35 / 1 |
| 892 | 7.5 | 42 / 5 | 42 / **22** | 42 / 18 |
| 697 | 6.0 | 34 / 9 | 34 / **54** | 34 / 37 |
| 692 | 15.0 | 52 / 1 | 52 / **24** | 52 / 24 |
| 898 | 30.0 | 44 / 4 | 44 / **26** | 44 / 16 |
| 700 | 7.5 | 37 / 2 | 37 / **6** | 37 / 6 |
| 900 | 7.5 | 28 / 10 | 28 / **81** | 28 / 14 |
| 499 | 7.5 | 50 / 1 | 50 / **3** | 50 / 3 |
| 697 | 15.0 | 36 / 5 | 36 / **6** | 36 / 6 |

**GPU better on 15, tied on 1, worse on 0 of 16.** Omega matched the field on
every round on both arms, so nothing was traded for the diversity.

Read that with three caveats:

* **`distinct` is a proxy for the diversity term, not the term itself.** Reward
  is `optimality*(1+difficulty) + diversity`, and diversity is `1/(miners
  submitting the identical set)`. A fleet only submits `k` answers, so the 609
  distinct cliques on the n=694 round are worth exactly as much as the first
  `k` of them. What the surplus buys is that the picker never has to repeat, and
  the number that settles whether it is worth anything is fleet_sim's.
* On 8 of these 16 rounds the GPU's distinct count *equals the field's total*,
  which is the signature of having enumerated every omega-clique that exists
  rather than of being better than the field.
* The tied round (n=696) has exactly one omega-clique. There was nothing to win.
* **These counts are not reproducible run to run.** A re-run of the same six
  rounds gave 84 instead of 83 on n=894 and 595 instead of 609 on n=694. §7's
  determinism claim — "the same set of completed jobs always produces the same
  output" — holds, and `solve_many` does sort the pool canonically; but *which*
  jobs finish is decided by a wall-clock deadline against a device whose job
  order is not fixed, so the set itself moves. See "Reproducibility" below,
  because it changes how a paired comparison has to be run.

### Stage 4 — queue health and the §6 stall detector (§8)

`gates.py stage4 --rounds 8 --budget 5.0`. Across every round run:
`drop=0 overflow=0 kmaxhit=0 invalid=0`, no deadlock, and every answer a clique
maximal in the full graph. Jobs enqueued exceeds jobs finished because the
deadline cuts the queue, which is the intended behaviour and is reported rather
than hidden.

The stall statistic behaves exactly as §6 predicted, and it is the most useful
number the harvester produces:

```
n=297 omega=23 distinct=11  ( 2.7/s)  stall=1.00   <- ban-based generation exhausted
n=697 omega=32 distinct=10  ( 2.0/s)  stall=0.99   <- exhausted
n=698 omega=87 distinct=32  ( 6.4/s)  stall=0.98   <- exhausted
n=894 omega=86 distinct=73  (14.5/s)  stall=0.40   <- still climbing
```

§6 says a ban excludes cliques *containing* the banned vertex, so it excludes the
parent but never the siblings, and past some pool size every job returns
something already held. A `stall` near 1 while the card is busy is that ceiling
arriving, not a defect.

One correction to the design's own formulation: §6 proposes tracking "the
fraction of finished jobs whose result was already in the pool". Measured over
the whole recorded band that reads far too low, because the omega-1 spares keep
arriving new long after the omega pool is exhausted — the same n=297 round reads
0.98 by that definition and 1.00 when restricted to omega-sized results. The
ceiling §6 describes is about omega-cliques, so `gpu_lib.stall()` counts
duplicates among results at the current target only (`dupmax / (dupmax + newmax)`).

When that number is near 1, §6's successors are the next thing to build —
hitting-set bans first (cheapest, inherits the ceiling), clique-level tabu second
(most promising: it fixes the two-disjoint-cliques stall *without* aiming below
omega). Neither is built here and neither should be assumed better than v1. §6's
own cautionary case is `plateau_walk`, which looked like a dose-response peak on
13 changed answers and came back 23 better / 30 worse when re-measured with a
corrected scorer.

## Design decisions that differ from the document

Stated in full at the top of `clique_gpu.cu`. In brief:

| § | design says | built | why |
|---|---|---|---|
| §2 | prefix/suffix ANDs | saturating carry by default, prefix/suffix behind `-DSN83_CANDS_PREFIX=1` | same sets (stage 1 proves it), ~3kW word-ops instead of ~6kW, O(W) state instead of O(kW). Measured 4–6x faster. The prefix arm keeps §2's incremental-update path open; the carry arm cannot have it. |
| §2 | `blocker(v)` falls out of which `A_j` the vertex came from | warp-parallel scan of `C` | keeping k separate masks costs more than an O(k/32) bit-test scan, and the scan is what `clique.cpp` does |
| §4 | make `clique.cpp`'s RNG pluggable for the differential test | host mirror inside `clique_gpu.cu` | keeps `clique.cpp` untouched and makes the stage-2 gate self-contained; the mirror is an independent serial implementation, which is the part that makes a diff worth running |
| §5 | host refills the queues | a starved warp synthesizes its own level-1 job | §5's own "escalate to more seeds at level 1 before level 2", done at the point of need rather than through a host round-trip. Off in batch mode. |

Everything in §4's parity checklist is implemented as written, including the row
that is not a detail: **perturbation is checked every step, not gated behind "no
legal move"**. The gated variant is the one that measured 3.1M swaps against 2
adds.

`SN83_KMAX` is 160. Measured omega over the 1000-round dump tops out at 110; a
walker that would exceed 160 stops adding and increments `kmaxhit`, which the
gates print. It is never a silent truncation.

## Occupancy, and §9 risk 5

`-Xptxas -v` output is written to `libcliquegpu_<arm>.so.ptxas` on every build
rather than discarded, because §9 risk 5 is register pressure silently halving
occupancy. It did:

```
harvest_kernel: 80 registers, 9168 bytes smem, 0 bytes spill
```

80 registers gives 6 blocks/SM, not the 12 §3 assumed — **1152 resident walkers,
half the design's 2304**. There are no spills, so this is a real occupancy
ceiling and not a fixable stack problem. `sn83_gpu_open` sizes the grid from
`cudaOccupancyMaxActiveBlocksPerMultiprocessor` rather than from the assumption,
so the number in `GpuClique.info()` is what the card actually runs.

## Running it in the simulator

`research_manual/solver.py` imports `solve_many` from `fleet_solver_gpu`, so the
usual command runs the GPU arm with no other change:

```bash
.venv/bin/python research_manual/simulate.py -N 40 --rounds 100 \
    --out research_manual/sim_out_gpu.json

SN83_SOLVER=cpu \
.venv/bin/python research_manual/simulate.py -N 40 --rounds 100 \
    --out research_manual/sim_out_cpu.json
```

That is a proper paired comparison: same rounds in the same order, same N, same
victims, same scorer, one environment variable different. Pass distinct `--out`
paths or the second run overwrites the first.

`simulate.py` itself is untouched. It scores through
`CliqueAI.scoring.clique_scoring.CliqueScoreCalculator` — the validator's own
calculator — so unlike everything in `gates.py`, its output *is* reward.

Watch the `late` count in its summary. A late answer is scored empty and takes a
hard zero, and the GPU arm pays a per-round device-setup cost the CPU arm does
not. `late 0` is the only acceptable reading.

### NO FALLBACK

No CUDA device, no nvcc, a graph past `SN83_MAXN`, a device-side error, or a
harvest that returns nothing the validator would accept is an **assertion
failure**, and the run stops.

This is deliberate, and it is the opposite of what a deployed miner should do. A
solver that quietly drops to the CPU arm mid-measurement produces a number that
is a blend of two solvers and reads as one — the same class of mistake as the
sixty-six scripts that called `round_score` without `valid=`, and just as
invisible after the fact. Run the CPU arm by asking for it (`SN83_SOLVER=cpu`),
never by letting the GPU arm fail into it. Whoever ships this to a live miner has
to add a fallback back, deliberately, at that boundary.

### Knobs

```bash
SN83_SOLVER=cpu     # solver.py takes fleet_solver.solve_many instead (the baseline arm)
SN83_GPU_ONLY=1     # drop the CPU champion, bootstrap omega on device (§5)
SN83_GPU_LANES=64   # rebuild with 64 lanes per walker (§3) — measured not to help
SN83_GPU_PREFIX=1   # rebuild with the design's prefix/suffix candidate sets (§2)
SN83_GPU_STEPS      # steps a level-1 job gets before it is judged (default 20000)
SN83_GPU_DEBUG=1    # per-solve omega / distinct / stall / banback on stderr
```

`fleet_solver_gpu.solve_many` is a **hybrid** by default: `clique.cpp` still pins
omega for `CHAMPION_SHARE` of the budget, and the GPU spends the rest widening
the pool. That split is deliberate — stage 0 shows a single GPU walker often does
not reach omega on a 900-vertex graph inside a short slice, and the CPU stage is
the measured one.

Every clique the device returns is extended in the **full** graph with the bans
lifted (§7 step 2) before it is fingerprinted, and re-verified exactly on the
host before it is returned. A clique maximal in `G - bans` can be extendable in
`G`, and the validator rejects anything extendable — *both* reward terms go to
zero. `fleet_solver._extend` exists because 3 of 435 answers took a hard zero
that way, all of them harvested.

## Reproducibility, and what it means for a paired comparison

The device search itself is deterministic: same job, same seed, same
`max_steps`, same answer — that is what the stage-2 trajectory diff asserts, and
it is why the gate is worth anything. `harvest()` is **not**, because the
deadline decides which jobs finish and thousands of warps do not finish in a
fixed order.

CLAUDE.md requires paired comparisons — same rounds, same k, sign test over
CHANGED answers. Against a nondeterministic arm that needs one extra care:

* `fleet_sim.py --solve` caches per-round answers to disk, so a single solve pass
  fixes the GPU arm's answers for every subsequent `--sizes` sweep. Sweeping off
  one cache is paired correctly.
* Comparing two GPU *configurations* against each other is not, because the
  difference between them is confounded with run-to-run noise. That needs
  replications, and the replication count has to come from how big the noise is
  — which is not measured yet.

The cautionary case is in `clique.cpp`'s own header: `plateau_walk` shipped on a
sign test over thirteen changed answers and came back 23 better / 30 worse on
re-measurement. Thirteen outcomes was too few to see anything. A nondeterministic
arm makes that trap easier to fall into, not harder.

## What is not built

* §5's optional multi-graph batching (`graph_id` in the job record). One graph
  per launch. The device is not short of work on a live round; it would matter
  for offline harvesting over many graphs, which is open decision 1 in §10.
* §2's incremental prefix/suffix update. Only reachable from the prefix arm, and
  the prefix arm is the slow one.
* §6's successors — hitting-set bans, clique-level tabu, soft frequency penalty.
  The stall detector that tells you when they are needed *is* built.
* Level >= 2 job generation. `q_push` takes a level and the rings exist for
  0..4, but nothing enqueues above level 1: `BAN_N = 1` is what `fleet_solver`
  settled on, and `tests/test_deployed_path.py` records `ban_n=3` harvesting
  nothing. Open decision 2 in §10 ("seeds before level 2, or level 2 before more
  seeds") is answered here in favour of seeds, on that existing evidence, and
  the opposite arm is a one-line change to the synthesis path.
