# GPU clique harvester — design

Status: **design under review, nothing built.** Target file `research_manual/clique_gpu.cu`.
Device on this box: NVIDIA RTX A4000, sm_86, 48 SMs, 16 GB, 4 MB L2 (from `nvidia-smi`).

Every throughput figure in this document is an **estimate derived from hardware specs and
op counts**, not a measurement. None of it is reward. See "Measurement rules" at the end.

---

## 0. What this is and is not

The job: given one graph and a budget, return **many distinct maximal cliques of size ω**,
not one clique faster. Optimality is already pinned by `clique.cpp`; the headroom is
uniqueness. That is the same problem stage 2 of `research_manual/fleet_solver.py` attacks
with delete-and-resolve, and this is that idea run wide.

What a GPU does **not** do here: speed up a single search. One walker is a sequential chain
of moves — `C`, then `slack`, then the candidate sets, then the next move. The parallelism
is **across independent searches** that share one read-only graph and differ in
`(ban set, seed)`.

Non-goal for v1: many *different* graphs per launch. Kept as an option in §5 because it is
the clean way to fill the card, but the first target is one graph, many bans.

---

## 1. The gate: per-walker step throughput

**This measurement decides whether the rest of the document is worth building.**

Aggregate throughput is not the question. A single job has to reach ω inside its own slice,
or every result comes back short and the harvest is empty regardless of how many walkers ran.

Estimate: a warp-per-walker step reads ~10 KB from L2 (§3), so on the order of 10^5 steps/s
per walker, against something like 10^6/s for a CPU thread. Aggregate is far ahead
(~2,300 resident walkers); **per-walker is roughly 10x behind.**

Whether 10x slower per walker still reaches ω: `fleet_solver.py` records time-to-ω at or
under 10% of the budget on 10 of 10 rounds measured (n=290..900, deadlines 6-30s), which
leaves room for a 10x slower walker. "Leaves room" is not "does". Stage 0 exists to
replace this paragraph with a number.

If per-walker depth is too low, the dial is `LANES_PER_WALKER` (§3), not more walkers.

---

## 2. Representation: drop `slack`, use prefix/suffix ANDs

The one intentional deviation from `clique.cpp`. It is a change of **data structure only** —
the candidate sets are provably the same sets, and §4 is the contract that keeps them so.

### Why the CPU structure does not port

v7's win is incremental `slack(v) = |C| - |N(v) cap C|`, updated only over the complement
row (~n/10 vertices at density 0.9). On one core that beats an n-wide scan. On a GPU those
~90 scattered writes into a walker-private array are ~90 uncoalesced transactions per move
and they serialize the lanes. The complement representation is a **CPU** optimization.

### What replaces it

With `C = [c_0 .. c_{k-1}]`, all bitsets of `W = ceil(n/64)` words:

```
P_i = N(c_0) & ... & N(c_{i-1})        prefix ANDs, P_0 = ALL
S_i = N(c_i) & ... & N(c_{k-1})        suffix ANDs, S_k = ALL

cand0                  = P_k & ~Cbits
cand1 with blocker c_j = (P_j & S_{j+1}) & ~N(c_j) & ~Cbits
```

Correctness: `P_j & S_{j+1}` is exactly `intersect_{i != j} N(c_i)`, so a vertex in it is
adjacent to every clique member except possibly `c_j`. ANDing with `~N(c_j)` keeps exactly
those non-adjacent to `c_j`, i.e. `slack == 1` with blocker `c_j`. The sets over `j` are
disjoint and their union is the full slack-1 set. `P_k & ~Cbits` is `slack == 0`.

### What this buys

- `blocker(v)` falls out of *which* `A_j` the vertex came from. The CPU does a linear scan
  of `C` for it (`blocker()` in `clique.cpp`).
- the SCC `conf` filter becomes one AND against a `confbits` bitset, replacing a per-element
  branch over the candidate list.
- **the complement matrix `co` is not needed at all.** `~N(v)` is a NOT of the word with a
  tail mask and the self bit cleared. Halves the shared read-only footprint.
- branch-free, coalesced, warp-parallel: the shape a GPU wants, replacing the shape it hates.

Cost per step: `3*k*W` word-ops, about 1,900 at k=40, W=16, split across 32 lanes.

### Later, not v1

A swap changes one element of `C`, so `P` stays valid below index `j` and `S` above it —
only `O((k-j)*W)` needs rebuilding. An add is just `P_k &= N(u)`. Skip both until the
correctness gates in §8 pass.

---

## 3. Warp mapping, occupancy, memory

| Level | Choice |
|---|---|
| walker | one per **warp**, 32 lanes cooperating inside one step |
| block | 128 threads = 4 warps = 4 walkers |
| resident | 12 blocks/SM x 48 SMs = **576 blocks, 2,304 walkers** |
| launched | ~4,096 blocks, each looping over jobs; driver backfills as blocks exit |

One walker per **thread** is the trap: 32 walkers in a warp diverge on the first
add-vs-swap decision and touch 32 unrelated vertex sets.

Blocks never wait on each other — a block pops a job or exits — so oversubscribing the
grid is safe and there is no deadlock risk from non-resident blocks.

`LANES_PER_WALKER` stays a compile-time knob (32 / 64 / 128). It is the response to a
failed stage-0 gate: 128 lanes per walker is ~4x fewer concurrent walkers, each ~2-4x
faster. That dial trades harvest breadth for per-job depth.

Registers: 48 warps/SM needs <= 42 regs/thread on sm_86. Use `__launch_bounds__(128)` and
watch `-Xptxas -v` from stage 0. Accepting 32 warps/SM is fine if it buys a faster loop.

### Read-only, one copy, shared by all walkers

`bits`: `n * W` words. At n=900 that is ~108 KB, so the **whole graph sits in the 4 MB L2**.
Mark `const __restrict__`.

### Per-walker state, global memory, coalesced

| field | size at n=1024, K_max=64 |
|---|---|
| `prefix`, `suffix` | 2 * 65 * 128 B ~ 16.6 KB |
| `age` (int16) | 2 KB |
| `Cbits`, `cand0bits`, `confbits`, `banbits` | 128 B each |
| `C`, `best` (int16) | 256 B |
| RNG state (Philox / splitmix64) | 16 B |
| **total** | **~22 KB** |

8,192 walkers x 22 KB ~ 180 MB. On 16 GB, **VRAM is not the limit; occupancy is.** The card
can store hundreds of thousands of walker states and execute ~2,300.

Shared memory per walker: candidate list staging, warp reduction scratch, `C` copy. Keep
under **1.5 KB** — 4 walkers/block x 12 blocks/SM against the 100 KB/SM cap leaves nothing
more. This is why walker state cannot live in shared memory at all.

---

## 4. Parity checklist

The device must implement the same search, or it finds nothing. Line by line:

| `clique.cpp` rule | device implementation |
|---|---|
| `cand0` = slack 0 | `P_k & ~Cbits` |
| `cand1` = slack 1 | `(P_j & S_{j+1}) & ~N(c_j) & ~Cbits`, union over j |
| adds tried before swaps, `conf` filter | AND with `confbits`; `cand0` first |
| add frees neighbours' `conf` | `confbits |= N(u)` |
| swap does **not** free (strong CC) | `add(v, free=false)` — unchanged, this is the mechanism |
| drop sets `conf[v]=0` | clear bit |
| BMS: sample <=64, score = overlap with the candidate set, oldest breaks ties | one candidate per lane, `popcount(row(v) & candmask)`, warp argmax over a packed `(score, -age, v)` 64-bit key |
| perturbation after 4000 non-improving steps, **checked every step** | same counter, same placement |
| keep 4 of the incumbent, 1/10 cold restart | same |
| `construct()` before returning | same |
| final maximality pass | §7 — runs in **full G**, bans ignored |

The "checked every step" row is not a detail. `clique.cpp`'s header records an earlier
revision that gated perturbation behind "no legal move" and measured 3.1M swaps against
2 adds, because on a dense graph a plateau move is essentially always available. Gating it
is a silent way to build a search that never restarts.

### Three things that cannot be bit-identical

1. **RNG.** `mt19937_64` carries 2.5 KB of state; unusable per walker. Device uses
   Philox/splitmix64. *Mitigation:* make the RNG pluggable in `clique.cpp` too, so a
   1-walker GPU run and a 1-thread CPU run on the same stream can be diffed move for move.
   That is the difference between "looks equivalent" and a differential test.
2. **BMS tie-breaks.** The CPU's `age[v] < age[bestv]` depends on sequential scan order.
   The packed-key warp argmax is deterministic but not the *same* deterministic. Equivalence
   is asserted over the candidate **sets**, not over which vertex gets picked.
3. **Clock.** No `steady_clock` per walker. Host writes a `volatile int stop` in mapped
   pinned memory, checked every 64 steps, plus a per-job `max_steps`. **Measurement runs
   drive off `max_steps` only** — wall-clock budgets make results irreproducible.

Independent check to build before trusting any of it: a host reference that recomputes the
slack-0 and slack-1 sets by brute force, asserted bit-equal against the device sets.

---

## 5. Scheduler

### Job record — 32 B, so a warp loads one in a single transaction

```
graph_id  : u16     level     : u8     // = |ban set|, doubles as priority
n_bans    : u8      epoch     : u16
bans[4]   : u16     seed      : u64
parent_fp : u64     max_steps : u32
```

Ban sets deeper than 4 spill to a side array.

### Queues: one bounded ring per level, ascending-preference pop

`pop` scans levels `L1..L4` in order and `atomicAdd`s the head of the first non-empty one.

**No deletion is needed to make "smaller priorities drop larger" work.** A queued level-2
job is never touched while any level-1 job exists, and new cliques constantly refill L1, so
L2+ starves on its own. Queue surgery would only add races.

Two mechanisms, for two different events:

- **level preference (soft):** a new distinct ω-clique enqueues its level-1 children, which
  starve L2+. The common case.
- **epoch bump (hard):** a job returns *larger* than ω. Every pool clique is now stale and
  its children pointless. Bump `epoch`; jobs with `job.epoch < current` are discarded at pop
  for one compare. This mirrors `target = len(cand); out = [cand]; seen = {cand}; spare = []`
  in `fleet_solver.py`.

Rings are bounded; overflow drops at high levels only.

### Bootstrap

Do **not** take ω from one walker. Launch a burst of ~512 **unbanned** walkers on distinct
seeds with a short `max_steps`. That yields ω, *several* distinct ω-cliques immediately (so
L1 starts wide instead of with one parent's ~40 children), and the §1 calibration for free.

### Generation, for a new pool clique K

- **level 1:** `{ban v : v in K} x S` seeds, `S` starting at 1.
- **refill trigger:** when `queued + running < capacity * F`, escalate — but **to more seeds
  at level 1 before level 2.** A new seed at level 1 still aims at ω; level 2 aims below it.
  `fleet_solver.py` settled on `BAN_N = 1` for exactly this reason ("deleting more aims the
  re-solve below omega and the harvest comes back empty"), and `tests/test_deployed_path.py`
  records `ban_n=3` harvesting nothing. The opposite arm is the first sweep to run.
- **level >= 2:** never enumerate. C(40,2) is 780 and C(40,3) is ~10^4; sample subsets.
- **short-return doubling:** a job returning below `target` is requeued at the same level
  with `2 * max_steps`, capped. Port of `BAN_FRAC *= 2.0` — same failure mode (slice too
  short to reach ω), same fix.
- **spares (ω-1):** kept as results per `SPARE_MARGIN = 1`, but **no children generated from
  them** in v1; a spare's children aim at ω-2.

### Ban choice policy

Uniform over K in v1. Two alternatives to sweep, neither asserted: weight by how many pool
cliques contain the vertex, or ban from the intersection of all known ω-cliques. See §6 —
this knob turns out to be the whole ballgame.

**Instrument the waste directly.** `native_algorithm_shim.py` notes that a re-solve "tends
to add back the vertex that was deleted, reproducing the same clique". So record whether the
maximality pass re-added a banned vertex, and count it. That is the per-job waste signal and
the feedback for whether to escalate a parent.

### Optional: many graphs per launch

`graph_id` in the job record, graphs packed with offsets. This is the clean way to fill the
card when one graph yields only a few hundred level-1 jobs. Build at stage 3 if the use case
is offline harvesting.

---

## 6. Known limitation: sibling repeats, and why level = |bans| is the wrong priority

**This is the reason v1 will plateau, and it is accepted for v1.** Recording it here so the
plateau is recognised as predicted behaviour rather than a bug.

### The failure

Take a graph with two disjoint maximum cliques, `A = {1..N}` and `B = {N+1..2N}`. Pool
starts as `{A}`.

1. Job bans `v in A`. `A` is now unreachable, so the search returns `B`. New clique, good.
2. Pool is `{A, B}`. Next job: parent `A`, ban `v in A` — the search can return `B`, which
   is already held. Parent `B`, ban `u in B` — it can return `A`.

From here **every job returns something already in the pool.** Dedup rejects all of it. The
card stays 100% busy and the harvest stays at 2 forever.

The root cause: a ban excludes only cliques *containing the banned vertex*. It excludes the
parent. It does not exclude the siblings.

### The fix, and why it is not free

To guarantee a job cannot return a known clique, the ban set must be a **hitting set of the
whole pool** — at least one vertex from every clique in it. Then no pool member is reachable.

That reframes the priority. `level = |bans|` is arbitrary; the meaningful quantity is

```
level = |bans| - |H(P)|        H(P) = a (greedy-)minimal hitting set of the current pool
```

the **excess over the floor forced by what we already hold**. A job at level 0 is aiming at
ω with the minimum perturbation that can possibly yield something new.

Cost of the hitting set depends on pool geometry, and the two extremes matter:

- **overlapping cliques:** one vertex shared by many pool members hits them all. Hitting
  sets stay tiny, and the scheme keeps working for a long time.
- **disjoint cliques:** `m` known cliques need `m` bans. So the required ban count grows
  with the harvest, and it is aiming further below ω on every step.

That second case is a **structural ceiling on ban-based harvesting**, not an implementation
bug: past some pool size, the bans needed to exclude the pool also destroy ω. The
`ban_n=3`-harvests-nothing result is this same effect seen early.

### Stall detector

Cheap and worth having in v1, because it is the trigger for everything in the next section:
track the fraction of finished jobs whose result was already in the pool. When that fraction
approaches 1 while the card is busy, ban-based generation is exhausted.

### Successors, in the order I would try them

1. **Hitting-set bans.** Direct fix for the failure above; inherits the ceiling. Cheap to
   add on top of v1 — it only changes which ban sets get enqueued.
2. **Clique-level tabu instead of vertex bans.** Forbid the known *vertex sets* rather than
   banning vertices: when a search lands on a clique already in the pool, force a lateral
   move and keep going instead of returning. This fixes the two-disjoint-cliques stall
   *without* aiming below ω, because nothing is removed from the graph. Most promising of
   the three.
3. **Soft frequency penalty.** Bias the BMS score against vertices that are frequent in the
   pool, instead of hard-banning anything. Keeps ω reachable everywhere; weaker forcing.

Note that (2) and (3) both change move selection, so both need the §4 parity argument redone
and both must be measured against v1 rather than assumed better. The `plateau_walk` history
in `clique.cpp` is the cautionary case: an unguided-diversification idea that looked like a
dose-response peak on 13 changed answers and came back 23 better / 30 worse on re-measurement
with a corrected scorer.

---

## 7. Result path — where validity is won or lost

Per finished job, on device:

1. `construct()` to maximality **under the ban mask** (the search invariant).
2. **Extend in full G, bans ignored.** A clique maximal in `G - bans` can be extendable in
   `G`, and the validator rejects anything extendable — *both* reward terms go to zero.
   `_extend` exists in `fleet_solver.py` because 3 of 435 answers took a hard zero this way,
   all of them harvested. The ban is a **search** constraint; the **answer** must be maximal
   in the real graph. Flag it when this step re-adds a banned vertex (§5).
3. Fingerprint: XOR of per-vertex random 64-bit hashes — order-independent, warp-parallel —
   plus the size.
4. Dedup: device open-addressed hash table, `atomicCAS` insert. On win, `atomicAdd` a
   result slot and write the vertex list. 64-bit fingerprints over ~10^6 cliques collide at
   ~10^-7; the host re-verifies the final pool exactly, so a collision costs one lost clique
   and never a bad answer.
5. If the size exceeds ω, bump the epoch (§5).

Host drains the result buffer, maintains the pool, enqueues children, and sorts the final
pool canonically (size desc, then lexicographic) so the same set of completed jobs always
produces the same output.

---

## 8. Build order, with a gate at each stage

| stage | deliverable | gate |
|---|---|---|
| 0 | one kernel, one warp, one job, no queue, fixed `max_steps`; instrument steps/s | **per-walker steps/s, and does one walker reach ω at n=290/490/690/890?** If not, raise `LANES_PER_WALKER` before building anything else |
| 1 | prefix/suffix candidate sets + brute-force host reference | device `cand0`/`cand1` bit-equal to the reference over >=10^4 steps on >=100 random graphs |
| 2 | full move loop; Philox in both CPU and device | 1-walker trajectory diff against CPU; every returned clique is a clique and maximal in full G |
| 3 | persistent kernel, static job list, no dynamic enqueue | aggregate throughput; **distinct** cliques vs CPU `solve_many` at matched wall time |
| 4 | multi-level queues, epoch, device-side child enqueue | distinct ω-cliques per second; no queue deadlock, no overflow loss; stall-detector wired (§6) |
| 5 | `fleet_solver_gpu.py` exposing the **same `solve_many` signature** | end-to-end through `research/fleet_sim.py` |

The matching signature at stage 5 is the point: it makes the GPU path measurable by the
apparatus that already exists instead of by a new harness.

Files: `research_manual/clique_gpu.cu`, nvcc `-arch=sm_86`, flat C ABI mirroring
`sn83_solve` so ctypes needs nothing new, plus `fleet_solver_gpu.py`. The only change to
`clique.cpp` is the pluggable RNG for stage 2.

---

## 9. Risks

1. **Per-walker depth** (§1). The load-bearing assumption of the whole design. Stage 0
   confirms or kills it.
2. **Sibling repeats** (§6). v1 will plateau. Known, accepted, instrumented.
3. **Level-1 exhaustion.** ~40 children per clique, so a handful of pool cliques gives
   hundreds of jobs against 2,304 resident walkers. Filling the card means seeds (safe,
   stays at ω), or level 2 (aims below ω), or batching several graphs (§5).
4. **Diversity may not follow throughput.** 2,300 walkers under the same guided rules fall
   into the same large basins — the exact collision problem `clique.cpp`'s header describes.
   Bans force divergence; extra seeds may not. Stage 3's metric is *distinct* cliques per
   second, never cliques per second.
5. **Register pressure** silently halving occupancy. Watch `-Xptxas -v` from stage 0.
6. **Live rounds vs offline.** A 6-30 s deadline minus 2 s network, plus launch and H2D, on
   a per-walker path ~10x slower than CPU. Build for offline harvesting first; ask about
   live rounds only after stage 3 has numbers.

---

## 10. Open decisions

1. **Offline harvesting over many graphs, or live SN83 rounds?** Decides whether multi-graph
   batching is stage 3 (offline: the natural way to fill the card) or a non-goal (live: one
   graph, and per-walker depth is everything).
2. **Seeds before level 2, or level 2 before more seeds?** Default is seeds first, on the
   `ban_n=1` evidence. Either way it is the first sweep to run.

---

## Measurement rules

From `CLAUDE.md`, and they bind this work specifically:

- Stages 0-4 produce **direction**. No number from them is reward.
- Reward numbers come only from `research/fleet_sim.py`, whose `score_round(sizes, valid,
  keys, difficulty)` requires `valid=` and whose `validate_cliques` runs the validator's own
  maximality test. A harness that makes `valid=` optional is not a simplification, it is the
  check being skipped.
- Paired comparisons only: same rounds, same K, sign test over **changed** answers.
- Draw rounds randomly within a stratum. `sorted(rounds, key=time_limit)[a:b]` is a deadline
  slice, and deadline proxies graph size.
- A smoke test is not a benchmark. Say which one was run.
- `workers * threads <= 15` for any CPU-side comparison arm — the box is CFS-capped near 15
  though `nproc` reports 128, and thread count changes the answer, not just the speed.
