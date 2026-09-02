"""Estimators that respect the randomization unit.

The failure mode this module exists to catch: treatment is assigned to
time-region *blocks*, outcomes are recorded on *rides*. An iid standard
error pretends every ride is an independent experiment. Cluster-robust
(and block-aggregated) standard errors count the 500 blocks you actually
randomized, not the 40,000 rides you observed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from sbetoolkit.randomization import SwitchbackAssignment


@dataclass(frozen=True)
class Estimate:
    ate: float
    se: float
    z: float
    pvalue: float
    n_obs: int
    n_clusters: int
    method: str

    def reject(self, alpha: float = 0.05) -> bool:
        return self.pvalue < alpha

    def interval(self, level: float = 0.95) -> tuple[float, float]:
        """Wald interval ``ate ± z_{1-α/2} SE``."""
        if not 0 < level < 1:
            raise ValueError("level must be in (0, 1)")
        z = float(norm.ppf(1 - (1 - level) / 2))
        return (self.ate - z * self.se, self.ate + z * self.se)

    def covers(self, truth: float, level: float = 0.95) -> bool:
        lo, hi = self.interval(level)
        return bool(lo <= truth <= hi)


def _welch(y: np.ndarray, treatment: np.ndarray, method: str, n_clusters: int) -> Estimate:
    y = np.asarray(y, dtype=float)
    t = np.asarray(treatment, dtype=int)
    yt = y[t == 1]
    yc = y[t == 0]
    if yt.size == 0 or yc.size == 0:
        raise ValueError("both treatment arms must be non-empty")
    ate = float(yt.mean() - yc.mean())
    se = float(np.sqrt(yt.var(ddof=1) / yt.size + yc.var(ddof=1) / yc.size))
    z = ate / se if se > 0 else np.inf
    p = float(2 * norm.sf(abs(z)))
    return Estimate(
        ate=ate,
        se=se,
        z=float(z),
        pvalue=p,
        n_obs=int(y.size),
        n_clusters=n_clusters,
        method=method,
    )


def iid_ate(y: np.ndarray, treatment: np.ndarray) -> Estimate:
    """Difference in means with an iid (Welch) standard error.

    Wrong when ``treatment`` is constant inside larger clusters.
    """
    return _welch(y, treatment, method="iid", n_clusters=int(np.asarray(y).size))


def block_ate(block_y: np.ndarray, block_treatment: np.ndarray) -> Estimate:
    """Difference in means on *one row per randomized block*.

    This is the design-based estimator when you assigned treatment to
    (region, period) cells: the sample size is the number of cells.
    """
    y = np.asarray(block_y, dtype=float)
    return _welch(y, block_treatment, method="block", n_clusters=int(y.size))


def clustered_ate(
    y: np.ndarray,
    treatment: np.ndarray,
    cluster: np.ndarray,
) -> Estimate:
    """OLS treatment effect with Liang–Zeger cluster-robust SE (CR1).

    Model: ``Y = α + τ T + e``. The meat of the sandwich is summed at
    ``cluster`` (the randomization unit). Degrees-of-freedom correction
    is ``G / (G - 1) × (n - 1) / (n - 2)``.
    """
    y = np.asarray(y, dtype=float)
    t = np.asarray(treatment, dtype=float)
    g = np.asarray(cluster)
    n = y.size
    if n < 4:
        raise ValueError("need at least 4 observations")
    x = np.column_stack([np.ones(n), t])
    xtx = x.T @ x
    try:
        xtx_inv = np.linalg.inv(xtx)
    except np.linalg.LinAlgError as exc:
        raise ValueError("design matrix is singular") from exc
    beta = xtx_inv @ (x.T @ y)
    resid = y - x @ beta

    meat = np.zeros((2, 2))
    # Factorize cluster ids without sorting the whole frame twice.
    _, inverse, counts = np.unique(g, return_inverse=True, return_counts=True)
    G = int(counts.size)
    order = np.argsort(inverse, kind="mergesort")
    sorted_inv = inverse[order]
    x_s = x[order]
    e_s = resid[order]
    start = 0
    for c in counts:
        sl = slice(start, start + c)
        xg = x_s[sl]
        eg = e_s[sl]
        score = xg.T @ eg
        meat += np.outer(score, score)
        start += c

    scale = (G / max(G - 1, 1)) * ((n - 1) / max(n - 2, 1))
    vcov = xtx_inv @ meat @ xtx_inv * scale
    se = float(np.sqrt(max(vcov[1, 1], 0.0)))
    ate = float(beta[1])
    z = ate / se if se > 0 else np.inf
    return Estimate(
        ate=ate,
        se=se,
        z=float(z),
        pvalue=float(2 * norm.sf(abs(z))),
        n_obs=n,
        n_clusters=G,
        method="cluster-robust",
    )


def estimate_switchback(
    assignment: SwitchbackAssignment,
    outcomes: pd.DataFrame,
    *,
    outcome: str = "match_rate",
    treatment: str = "treatment",
) -> Estimate:
    """Block-level ATE using only ``assignment.analysis_table`` rows.

    ``outcomes`` may contain washout periods. They are dropped by an
    inner join onto the analysis grid, not by filtering ``is_washout``
    on the outcome frame (which is easy to forget).
    """
    sample = assignment.for_analysis(outcomes)
    if outcome not in sample.columns or treatment not in sample.columns:
        raise ValueError(f"outcomes must contain {outcome!r} and {treatment!r}")
    return block_ate(sample[outcome].to_numpy(), sample[treatment].to_numpy())
