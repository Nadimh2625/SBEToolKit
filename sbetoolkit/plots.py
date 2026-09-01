"""Figures for the interview README: naive A/B vs switchback vs truth."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_naive_vs_switchback(
    comparison: pd.DataFrame,
    *,
    path: str | Path | None = None,
    title: str | None = None,
) -> plt.Figure:
    """One chart: Monte Carlo estimates against the known global ATE."""
    truth = float(comparison["truth_ate"].iloc[0])
    naive = comparison["naive_ate"].to_numpy()
    sw = comparison["switchback_ate"].to_numpy()

    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    rng = np.random.default_rng(0)
    jitter = 0.08

    ax.scatter(
        rng.normal(0, jitter, size=len(naive)),
        naive,
        s=22,
        alpha=0.45,
        c="#c0392b",
        edgecolors="none",
        label="Naive rider A/B (one run)",
        zorder=2,
    )
    ax.scatter(
        rng.normal(1, jitter, size=len(sw)),
        sw,
        s=22,
        alpha=0.45,
        c="#2471a3",
        edgecolors="none",
        label="Switchback (one run)",
        zorder=2,
    )

    ax.errorbar(
        [0],
        [naive.mean()],
        yerr=[1.96 * naive.std(ddof=1) / np.sqrt(len(naive))],
        fmt="D",
        color="#7b241c",
        ms=9,
        capsize=6,
        zorder=4,
        label="Naive mean ± 1.96 SE",
    )
    ax.errorbar(
        [1],
        [sw.mean()],
        yerr=[1.96 * sw.std(ddof=1) / np.sqrt(len(sw))],
        fmt="D",
        color="#1a5276",
        ms=9,
        capsize=6,
        zorder=4,
        label="Switchback mean ± 1.96 SE",
    )

    ax.axhline(truth, color="#1e8449", lw=2.2, ls="--", zorder=3, label=f"Ground truth ({truth:.3f})")
    ax.axhline(0.0, color="#888888", lw=0.8, ls=":", zorder=1)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Naive rider A/B", "Time-region\nswitchback"])
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylabel("Estimated effect on match rate")
    ax.set_title(
        title
        or "Supply-constrained marketplace: naive A/B overstates the launch effect"
    )
    ax.legend(frameon=False, loc="best")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160)
    return fig


def plot_type_i(
    result,
    *,
    path: str | Path | None = None,
) -> plt.Figure:
    """Bar chart: null reject rates vs nominal α."""
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    labels = [
        "Iid SE\nat ride level",
        "Cluster-robust SE\nat block",
        "Difference in means\non blocks",
    ]
    rates = [result.iid_rate, result.cluster_rate, result.block_rate]
    colors = ["#c0392b", "#1a5276", "#2471a3"]
    bars = ax.bar(np.arange(3), rates, color=colors, width=0.62, zorder=2)
    ax.axhline(result.alpha, color="#1e8449", lw=2.2, ls="--", zorder=3, label=f"Nominal α = {result.alpha:.2f}")
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(0.22, max(rates) * 1.25))
    ax.set_ylabel("False-positive rate (true effect = 0)")
    ax.set_title(
        f"Null check: {result.n_reps} switchbacks, "
        f"{result.n_blocks} blocks, ~{result.n_rides_per_rep:,.0f} rides"
    )
    for bar, rate in zip(bars, rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.006,
            f"{100 * rate:.1f}%",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )
    ax.legend(frameon=False, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160)
    return fig
