"""Distribution images for a sweep."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_all(result, outdir, tag, n_total, n_rounds):
    """Writes every image for one sweep and returns their paths."""
    os.makedirs(outdir, exist_ok=True)
    return [_means(result, outdir, tag, n_total, n_rounds),
            _bands(result, outdir, tag),
            _histograms(result, outdir, tag),
            _margin(result, outdir, tag)]


def _means(result, outdir, tag, n_total, n_rounds):
    hotkeys = [r["g"] for r in result]
    figure, axes = plt.subplots(figsize=(9, 5))
    axes.plot(hotkeys, [r["a_mean"] for r in result], lw=2, label="A (first)")
    axes.plot(hotkeys, [r["b_mean"] for r in result], lw=2, label="B (second)")
    axes.set_xlabel("A's hotkeys (of %d)" % n_total)
    axes.set_ylabel("mean reward per answer")
    axes.set_title("mean reward vs fleet split -- %s, %d solved rounds"
                   % (tag, n_rounds))
    axes.grid(alpha=.3)
    axes.legend()
    return _save(figure, outdir, "means_%s.png" % tag)


def _bands(result, outdir, tag):
    hotkeys = [r["g"] for r in result]
    figure, axes = plt.subplots(figsize=(9, 5))
    for key, name, colour in (("a", "A (first)", "C0"),
                              ("b", "B (second)", "C1")):
        low = [np.percentile(r[key], 10) for r in result]
        high = [np.percentile(r[key], 90) for r in result]
        mid = [np.percentile(r[key], 50) for r in result]
        axes.fill_between(hotkeys, low, high, alpha=.20, color=colour)
        axes.plot(hotkeys, mid, lw=2, color=colour,
                  label="%s median, p10-p90" % name)
    axes.set_xlabel("A's hotkeys")
    axes.set_ylabel("reward per answer")
    axes.set_title("reward distribution vs fleet split -- %s" % tag)
    axes.grid(alpha=.3)
    axes.legend()
    return _save(figure, outdir, "bands_%s.png" % tag)


def _histograms(result, outdir, tag):
    picks = [result[0], result[len(result) // 2], result[-1]]
    figure, axes = plt.subplots(1, len(picks), figsize=(14, 4), sharey=True)
    axes = np.atleast_1d(axes)
    for axis, row in zip(axes, picks):
        edges = np.linspace(min(min(row["a"]), min(row["b"])),
                            max(max(row["a"]), max(row["b"])), 40)
        axis.hist(row["a"], bins=edges, alpha=.6, label="A (first)")
        axis.hist(row["b"], bins=edges, alpha=.6, label="B (second)")
        axis.set_title("A=%d  B=%d" % (row["g"], row["o"]))
        axis.set_xlabel("reward per answer")
        axis.grid(alpha=.3)
    axes[0].set_ylabel("rounds")
    axes[0].legend()
    return _save(figure, outdir, "histograms_%s.png" % tag)


def _margin(result, outdir, tag):
    hotkeys = [r["g"] for r in result]
    margin = [r["b_mean"] - r["a_mean"] for r in result]
    figure, axes = plt.subplots(figsize=(9, 5))
    axes.axhline(0, color="k", lw=1)
    axes.plot(hotkeys, margin, lw=2, color="C3")
    axes.fill_between(hotkeys, 0, margin, where=[m > 0 for m in margin],
                      alpha=.25, color="C3", label="second player ahead")
    axes.fill_between(hotkeys, 0, margin, where=[m <= 0 for m in margin],
                      alpha=.25, color="C0", label="first player ahead")
    axes.set_xlabel("A's hotkeys")
    axes.set_ylabel("mean_B - mean_A")
    axes.set_title("margin of the second player -- %s" % tag)
    axes.grid(alpha=.3)
    axes.legend()
    return _save(figure, outdir, "margin_%s.png" % tag)


def _save(figure, outdir, name):
    path = os.path.join(outdir, name)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path
