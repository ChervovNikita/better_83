"""Guard against the error that produced nine of eleven retractions in this project.

THE ERROR
    A statistic is computed by pooling items across rounds of unequal size, and the
    result is reported as a property of the solver. Round size correlates with almost
    everything here — optimum count, holder count, collision rate — so pooling
    manufactures effects that vanish within round.

THE CHECK
    Compute the same contrast two ways: pooled over all items, and within each round
    then averaged. If the pooled effect is large and the within-round effect is ~0, the
    pooled number is measuring round size, not the thing you named it after.

USAGE
    from pooling_guard import check
    check(groups)          # groups: {label: [(round_id, value), ...]}

Real cases this would have caught, from FINDINGS.md:

    "reach bias 1.21-1.74"          pooled +0.897   within -0.006   ARTIFACT
    "agreement predicts popularity" pooled +1.577   within -0.019   ARTIFACT
    "agreement, 200-round rerun"    pooled +0.795   within +0.039   ARTIFACT
    "field sole 46.8% vs our 30%"   pooled +0.076   within +0.064   real, but median 0
"""
import collections
import statistics as st


def check(groups, label_a=None, label_b=None, quiet=False):
    """groups: {label: [(round_id, value), ...]}. Compares the first two labels.

    Returns (pooled_diff, within_mean, within_median, frac_in_direction, verdict).
    """
    labels = list(groups)
    a = label_a or labels[0]
    b = label_b or labels[1]
    pa = [v for _, v in groups[a]]
    pb = [v for _, v in groups[b]]
    pooled = st.mean(pb) - st.mean(pa)

    by_round = collections.defaultdict(lambda: ([], []))
    for r, v in groups[a]:
        by_round[r][0].append(v)
    for r, v in groups[b]:
        by_round[r][1].append(v)
    within = [st.mean(y) - st.mean(x) for x, y in by_round.values() if x and y]
    if not within:
        return pooled, 0.0, 0.0, 0.0, "NO PAIRED ROUNDS — cannot check"

    wm, wmed = st.mean(within), st.median(within)
    frac = sum(1 for d in within if d > 0) / len(within)
    # an artifact is: large pooled effect, negligible within-round effect
    artifact = abs(pooled) > 0.1 and abs(wm) < abs(pooled) * 0.25
    verdict = ("LIKELY ARTIFACT — the pooled effect is %.2fx the within-round effect"
               % (abs(pooled) / max(abs(wm), 1e-9)) if artifact else
               "consistent — pooled and within-round agree")
    if not quiet:
        print("  pooled  %-22s %+.4f" % ("(%s vs %s)" % (b, a), pooled))
        print("  within  mean %+.4f  median %+.4f  in-direction %.0f%% of %d rounds"
              % (wm, wmed, 100 * frac, len(within)))
        print("  -> %s" % verdict)
    return pooled, wm, wmed, frac, verdict


if __name__ == "__main__":
    # self-test on a synthetic case with the exact structure of the real failures:
    # big rounds have many items and low values; small rounds have few and high.
    g = {"one": [], "many": []}
    for rid in range(60):
        big = rid % 2 == 0
        n = 40 if big else 4
        base = 0.4 if big else 2.0
        for _ in range(n):
            g["one"].append((rid, base))
        for _ in range(max(1, n // 8)):
            g["many"].append((rid, base))       # SAME value within round
    print("self-test: identical values within every round, only sizes differ")
    check(g, "one", "many")
    print("\n  (a correct check reports ~0 within-round; any pooled effect is size)")
