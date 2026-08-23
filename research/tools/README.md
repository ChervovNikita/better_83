# Measurement tooling for the gen-2 diversity work

These are the scripts behind the findings in `~/autoresearch-runs/sn83-clique/`. They
are preserved here because the results are only as trustworthy as the code that produced
them, and several of the findings are corrections to earlier findings.

## The one rule these encode

**Only paired, same-round, same-K comparisons on the SUBMITTED PREFIX are admissible.**
Seven wrong answers in this project came from averaging a ratio across rounds of unequal
size, or from comparing a conditioned subset against an unconditioned one. `score_pools.py`
is the sanctioned scorer and does the paired sign test over changed prefixes only.

| script | what it measures |
|---|---|
| `score_pools.py` | **the scorer.** Paired prefix reward + novel share, sign test over changed prefixes |
| `probe.py` | run a `solve_many` variant, validate every clique, dump the pool |
| `hull.py` | hull construction + MoMC invocation (shared by the rest) |
| `hull2.py` | H-HULL with real deadlines, timeouts counted as fallbacks |
| `hull3.py` | H-REGION: hull padding rule high/low/anti/rand |
| `supply.py` | union-hull ORACLE: how many maxima exist. Uses field data, NOT deployable |
| `cover.py` | are unclaimed maxima inside an our-pool-only hull, by hull size |
| `diag.py` | separates "hull lacks them" from "MoMC does not return them" |
| `permute.py` | H-PERMUTE Jaccard gate |
| `replicator.py` | batched Motzkin-Straus with randomised regulariser |
| `fleet_hull.py` | end-to-end fleet median vs field median from a pool file |
| `MoMC2016.c` | Li & Jiang's exact enumerator, MIT. Build: `gcc -O3 -w -DMOMC -o momc MoMC2016.c -lm` |
| **`pooling_guard.py`** | **run this before reporting any descriptive statistic.** See below. |

## pooling_guard.py — the single most useful file here

Eleven claims in `FINDINGS.md` were withdrawn. **Nine were the same error:** a statistic
pooled across rounds of unequal size, reported as a property of the solver. Round size
correlates with almost everything in this problem — optimum count, holder count,
collision rate — so pooling manufactures effects that vanish within round.

    from pooling_guard import check
    check({"in_pool": [(round_id, value), ...], "out": [...]})

It reports the contrast pooled AND within-round. If the pooled effect is large and the
within-round effect is ~0, the pooled number is measuring round size.

Cases it would have caught before they were published:

| claim | pooled | within round | verdict |
|---|---|---|---|
| "reach bias 1.21-1.74" | +0.897 | **-0.006** | artifact |
| "agreement predicts popularity" | +1.577 | **-0.019** | artifact |
| same, on an independent 200-round rerun | +0.795 | **+0.039** | artifact |
| "field sole-held 46.8% vs our 30%" | +0.076 | +0.064 | real (but median 0) |

The paired scoring path (`score_pools.py`, `compare.py`) never produced a false positive
in this project. Every retraction came from analysis written outside it.

## Scripts that use field data

`supply.py`, `cover.py` and `diag.py` read the field's submitted answers. They are
**measurements, not mechanisms** — nothing they compute is available at solve time. They
exist to bound what is reachable in principle. Do not build a solver around them.

## Known correction

`hull.py`'s padding score was computed on a uint8 sum and then negated, which is an
unsigned negate. Measured as harmless on these instances (0 of 4372 vertices have zero
connection to the hull, and pad order is identical either way) and fixed. It would NOT
be harmless on sparse graphs.
