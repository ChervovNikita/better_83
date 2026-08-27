# Fleet allocation: exact derivation

What `strategy.py` implements, and what it is allowed to claim.

## The scorer

From `CliqueAI/scoring/clique_scoring.py`, for a round with answers `R`
(a multiset of cliques) and difficulty `D`:

```
val_i   = 1 if answer i is a valid maximal clique else 0
size_i  = |answer_i| * val_i
M       = max_j size_j
rel_i   = size_i / M
pr_i    = |{j : size_j > size_i}| / |R|
w_i     = exp(-pr_i / rel_i) * val_i
opt_i   = w_i / max_j w_j
d_i     = val_i / count(answer_i)          count over the whole multiset R
div_i   = d_i / max_j d_j
reward_i = opt_i * (1 + D) + div_i
```

## Two normalisation lemmas

**Lemma 1. `max_j w_j = 1`, so `opt_i = exp(-pr_i / rel_i)`.**

Let `i*` be any valid answer with `size_{i*} = M` (one exists whenever any answer
is valid, since `M` is the max over valid sizes). Nothing is strictly larger, so
`pr_{i*} = 0` and `rel_{i*} = 1`, giving `w_{i*} = exp(0) = 1`. For every `i`,
`pr_i >= 0` and `rel_i > 0`, so `w_i <= 1`. Hence the max is exactly 1. ∎

**Lemma 2. If some valid answer is submitted by exactly one miner, `max_j d_j = 1`
and `div_i = 1 / count_i`.**

That answer has `d = 1/1 = 1`; every `d_j <= 1`; so the max is 1. ∎

Lemma 2 is an *assumption about the round*, not a theorem — see A2 below.

Under both: **`reward_i = exp(-pr_i/rel_i) * (1+D) + 1/count_i`.**

## Assumptions

- **A1** At least one submitted answer is valid. (Lemma 1.)
- **A2** At least one valid answer is unique in `R`. (Lemma 2.) With ~50 field
  answers over hundreds of distinct cliques this holds on every tuning round
  checked; if it fails, every `div` is scaled by the same `1/max_j d_j > 1`, which
  changes the objective's scale but not its argmax over our allocation — except
  where our own choice is what breaks uniqueness.
- **A3** Every answer we submit is valid. Guaranteed: the device extends each
  answer to maximality in the full graph and the host re-verifies exactly.
- **A4** `F_c`, the number of field miners on our clique `c`, is
  `Binomial(a_s, 1/N_s)` where `a_s` is the number of field answers at size `s`
  and `N_s` the number of distinct cliques of that size. This treats the `F_c` as
  independent when they are really multinomial (they sum to `a_s`), and it
  assumes the field picks uniformly when it in fact concentrates on large basins.
  **This is the weakest assumption.** Measured: it is right on average for the
  omega-1 pool, because the chance one of our `N` spares is among the field's `n`
  is itself `n/N`, and the two cancel.
- **A5** `pr` depends on the sizes *we* submit, so the objective is not separable
  until the size split is fixed. Handled by enumerating the split (below), not by
  assuming it away.

## The objective

We place `m_c >= 0` hotkeys on each distinct clique `c` of our pool, with
`sum_c m_c = q`. Write `f_c` for the field count on `c`, `s_c` for its size, and
`A_s = opt(s) * (1 + D)`.

Grouping the `q` per-answer rewards by clique:

```
V(m) = sum_c [ m_c * A_{s_c}  +  m_c / (f_c + m_c) ]
```

## Marginals

Let `h(m) = m / (f + m)`, `h(0) = 0`. For `m >= 1`:

```
h(m) - h(m-1) = m/(f+m) - (m-1)/(f+m-1)
              = [ m(f+m-1) - (m-1)(f+m) ] / [ (f+m)(f+m-1) ]
              = [ mf + m^2 - m - mf - m^2 + f + m ] / [ (f+m)(f+m-1) ]
              = f / [ (f+m)(f+m-1) ]
```

so the gain from the `m`-th hotkey on clique `c` is

```
Delta_c(m) = A_{s_c} + f_c / [ (f_c+m)(f_c+m-1) ]     f_c >= 1
Delta_c(1) = A_{s_c} + 1,   Delta_c(m) = A_{s_c}       f_c = 0, m >= 2
```

The `f_c = 0` case is the `0/0` limit taken directly from `h`: with nobody else
on `c`, `h(m) = 1` for all `m >= 1`, so the first hotkey gains a full unit of
diversity and every later one gains none.

**`Delta_c` is non-increasing in `m`.** For `f >= 1`, `(f+m)(f+m-1)` is increasing
in `m`, so the second term decreases. For `f = 0` it steps from `A+1` to `A` and
stays. Hence each `g_c(m) = m*A + h(m)` is discretely concave. ∎

## Theorem: greedy on marginals is exactly optimal

*Given fixed `A_s` (i.e. a fixed size split, per A5), the allocation that assigns
each of the `q` hotkeys in turn to the clique with the largest current marginal
maximises `V`.*

**Proof.** By telescoping, `g_c(m_c) = sum_{j=1}^{m_c} Delta_c(j)`, so

```
V(m) = sum_c sum_{j=1}^{m_c} Delta_c(j)
```

Every feasible `m` therefore corresponds to choosing `q` marginals from the
infinite table `{Delta_c(j)}` subject to one restriction: the chosen set must be
*prefix-closed* per clique — if `Delta_c(j)` is chosen then so is `Delta_c(j')`
for all `j' < j`.

Now take the `q` largest entries of the table. Because `Delta_c` is
non-increasing in `j`, that selection is automatically prefix-closed: if
`Delta_c(j)` is among the top `q` then `Delta_c(j') >= Delta_c(j)` for `j' < j`,
so those are too (ties broken consistently by `j`). Hence the top-`q` selection is
feasible, and since every feasible allocation is *some* selection of `q` table
entries, no feasible allocation can beat it.

Greedy picks exactly the top `q` entries, because after taking `Delta_c(j)` the
next available entry for `c` is `Delta_c(j+1)`, which is the largest remaining
one for that clique. ∎

**Corollary (even spread within a size class).** Under A4 all cliques of the same
size `s` have identically distributed `F_c`, so identical expected marginals.
Greedy therefore round-robins among them: `t` hotkeys over `N_s` cliques go
`floor(t/N_s)` or `ceil(t/N_s)` each. Duplication inside a size class begins only
once `t > N_s`.

**Corollary (when to duplicate rather than spread to a smaller clique).** Placing
an `(m+1)`-th hotkey on a size-`s` clique beats taking a fresh size-`s'` clique iff

```
A_s + E[ F/((F+m+1)(F+m)) ]   >   A_{s'} + E[ 1/(F'+1) ]
```

With `f = 0` and `m >= 1` the left side is just `A_s`: a second hotkey on an
uncrowded clique earns no diversity at all. It still wins when the only
alternatives are crowded — `E[1/(F'+1)]` small — or much smaller, `A_{s'} << A_s`.

## Field-blind form

We do not know `f_c`. Replacing `f_c` by the model of A4 and taking expectations
— legitimate because `V` is linear in the per-clique terms and `Delta` is defined
by differences of them — gives

```
Delta_s(m) = A_s + E[m/(F+m)] - E[(m-1)/(F+m-1)],   F ~ Bin(a_s, 1/N_s)
```

computed by direct summation over `k = 0..a_s` (exact, no approximation beyond
A4). `expected_share(a, N, m)` in `strategy.py` is `E[m/(F+m)]`, checked against
Monte Carlo to < 5e-3.

## Handling A5

`opt(s) = exp(-pr(s)/rel(s))` and `pr(s)` counts answers strictly larger than `s`
— including *our own*. So `A_s` is not fixed until the split is. Since our pool
has only two sizes in play (omega and omega-1), the split is one integer `t` =
how many of our hotkeys sit at omega, and we enumerate `t = 0..q`, solving the
greedy exactly for each and keeping the best by exact evaluation.

`t = 0` is a genuine discontinuity, not an edge case: if no answer in the round
reaches omega then `M` drops to omega-1 and every omega-1 answer gets `pr = 0`,
`opt = 1`. That is why holding back can cost nothing at all when we are the only
finder.

## What this does not cover

- The `F_c` independence and uniformity of A4. Real solvers concentrate; the
  measured collision matrix shows the field's own cross-rate at 1.3-1.8% rather
  than the ~1/N uniform prediction.
- Our submissions perturb the field's rewards too (they share the `count` and
  `pr` denominators). We optimise our own total, not the differential.
- `a_s` is predicted, not known. Errors there propagate into every `Delta`.
