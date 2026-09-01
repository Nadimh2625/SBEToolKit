"""CUPED (Controlled-experiment Using Pre-Experiment Data).

Deng, Xu, Kohavi, Walker (2013): replace the outcome ``Y`` with

    Y_cv = Y − θ (X − μ_X),   θ = Cov(Y, X) / Var(X)

The ATE is unchanged (X is pre-period) and Var(Y_cv) = Var(Y)(1 − ρ²).
In a switchback, ``X`` is typically the same region's lagged outcome
from a pre-experiment window, *not* the previous experimental period
(that would be post-treatment).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CUPEDFit:
    theta: float
    mu_x: float
    rho: float
    var_y: float
    var_ycv: float
    variance_reduction: float


def cuped_theta(y: np.ndarray, x: np.ndarray) -> CUPEDFit:
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if y.shape != x.shape:
        raise ValueError("y and x must have the same shape")
    if y.size < 3:
        raise ValueError("need at least 3 observations")
    var_x = float(np.var(x, ddof=1))
    if var_x <= 0:
        raise ValueError("x has zero variance; CUPED is unidentified")
    cov = float(np.cov(y, x, ddof=1)[0, 1])
    theta = cov / var_x
    var_y = float(np.var(y, ddof=1))
    rho = cov / np.sqrt(var_y * var_x) if var_y > 0 else 0.0
    ycv = y - theta * (x - np.mean(x))
    var_ycv = float(np.var(ycv, ddof=1))
    reduction = 1.0 - var_ycv / var_y if var_y > 0 else 0.0
    return CUPEDFit(
        theta=theta,
        mu_x=float(np.mean(x)),
        rho=float(rho),
        var_y=var_y,
        var_ycv=var_ycv,
        variance_reduction=float(reduction),
    )


def cuped_adjust(y: np.ndarray, x: np.ndarray, *, theta: float | None = None) -> np.ndarray:
    """Return CUPED-adjusted outcomes.

    If ``theta`` is omitted it is estimated on the same sample. For
    honest sequential tests, fit θ on a locked pre-period and pass it in.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if theta is None:
        theta = cuped_theta(y, x).theta
    return y - theta * (x - np.mean(x))


def difference_in_means(
    y: np.ndarray,
    treatment: np.ndarray,
    *,
    x: np.ndarray | None = None,
) -> dict[str, float]:
    """ATE via difference in means, optionally CUPED-adjusted.

    Standard errors are heteroskedasticity-robust (Welch).
    """
    y = np.asarray(y, dtype=float)
    t = np.asarray(treatment, dtype=int)
    if x is not None:
        y = cuped_adjust(y, np.asarray(x, dtype=float))
    yt = y[t == 1]
    yc = y[t == 0]
    if yt.size == 0 or yc.size == 0:
        raise ValueError("both treatment arms must be non-empty")
    ate = float(yt.mean() - yc.mean())
    se = float(np.sqrt(yt.var(ddof=1) / yt.size + yc.var(ddof=1) / yc.size))
    return {
        "ate": ate,
        "se": se,
        "n_treat": float(yt.size),
        "n_control": float(yc.size),
    }


def cuped_from_frame(
    frame: pd.DataFrame,
    outcome: str,
    treatment: str,
    covariate: str,
) -> dict[str, float]:
    fit = cuped_theta(frame[outcome].to_numpy(), frame[covariate].to_numpy())
    est = difference_in_means(
        frame[outcome].to_numpy(),
        frame[treatment].to_numpy(),
        x=frame[covariate].to_numpy(),
    )
    return {**est, **fit.__dict__}
