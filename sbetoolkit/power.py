"""Power and sample-size calculation for switchback experiments.

The estimand is the difference in block-level means between treated and
control time-region cells. Outcomes may be serially correlated (AR(1)),
which inflates variance relative to an iid calculation.

Given a treatment sequence ``A_t ∈ {0,1}`` and weights
``w_t = 1/n_t`` on treated periods and ``-1/n_c`` on control periods,
the contrast variance is ``w' Γ w`` where ``Γ_ij = σ² ρ^{|i-j|}``.
Independent regions divide that variance by ``n_regions``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from sbetoolkit.randomization import assign_switchback


@dataclass(frozen=True)
class PowerResult:
    n_regions: int
    n_periods: int
    n_analysis_blocks: int
    delta: float
    sigma: float
    rho_ar1: float
    alpha: float
    power: float
    variance: float
    se: float
    n_switches_per_region: float
    washout: int


def _ar1_cov(n: int, sigma2: float, rho: float) -> np.ndarray:
    lags = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
    return sigma2 * (rho**lags)


def contrast_variance(
    treatment: np.ndarray,
    sigma: float,
    rho_ar1: float = 0.0,
    analysis_mask: np.ndarray | None = None,
) -> float:
    """Variance of Ȳ_T − Ȳ_C for one region with AR(1) outcomes.

    Washout periods in ``analysis_mask`` (True = keep) are dropped from
    the contrast but still occupy time, so nearby kept periods remain
    correlated through the original lag.
    """
    treatment = np.asarray(treatment, dtype=int)
    n = len(treatment)
    if analysis_mask is None:
        keep = np.ones(n, dtype=bool)
    else:
        keep = np.asarray(analysis_mask, dtype=bool)
        if keep.shape != treatment.shape:
            raise ValueError("analysis_mask must match treatment length")

    treated = keep & (treatment == 1)
    control = keep & (treatment == 0)
    n_t = int(treated.sum())
    n_c = int(control.sum())
    if n_t == 0 or n_c == 0:
        return np.inf

    w = np.zeros(n)
    w[treated] = 1.0 / n_t
    w[control] = -1.0 / n_c
    gamma = _ar1_cov(n, sigma**2, rho_ar1)
    return float(w @ gamma @ w)


def _z_values(alpha: float, power: float) -> tuple[float, float]:
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if not 0 < power < 1:
        raise ValueError("power must be in (0, 1)")
    z_a = norm.ppf(1 - alpha / 2)
    z_b = norm.ppf(power)
    return z_a, z_b


def switchback_power(
    delta: float,
    sigma: float,
    n_regions: int,
    n_periods: int,
    *,
    rho_ar1: float = 0.0,
    alpha: float = 0.05,
    p_treat: float = 0.5,
    design: str = "balanced",
    switch_every: int = 1,
    washout: int = 0,
    seed: int = 0,
    n_design_draws: int = 8,
) -> PowerResult:
    """Power of a two-sided test of the switchback contrast.

    Averaged over a few design draws so blocked/independent assignments
    do not depend on one unlucky sequence.
    """
    if n_regions < 1 or n_periods < 2:
        raise ValueError("need at least 1 region and 2 periods")
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    vars_ = []
    n_keep = []
    n_sw = []
    for i in range(n_design_draws):
        assignment = assign_switchback(
            regions=["_power"],
            periods=list(range(n_periods)),
            p_treat=p_treat,
            design=design,  # type: ignore[arg-type]
            switch_every=switch_every,
            washout=washout,
            seed=None if seed is None else seed + i,
        )
        treat = assignment.table["treatment"].to_numpy()
        keep = ~assignment.table["is_washout"].to_numpy()
        vars_.append(contrast_variance(treat, sigma, rho_ar1, keep))
        n_keep.append(int(keep.sum()))
        n_sw.append(float(assignment.n_switches().iloc[0]))

    var_one = float(np.mean(vars_))
    var_total = var_one / n_regions
    se = np.sqrt(var_total)
    z_a, _ = _z_values(alpha, 0.8)
    # two-sided power against alternative delta
    power = float(
        norm.sf(z_a - delta / se) + norm.cdf(-z_a - delta / se)
    )
    return PowerResult(
        n_regions=n_regions,
        n_periods=n_periods,
        n_analysis_blocks=int(np.mean(n_keep)) * n_regions,
        delta=delta,
        sigma=sigma,
        rho_ar1=rho_ar1,
        alpha=alpha,
        power=power,
        variance=var_total,
        se=se,
        n_switches_per_region=float(np.mean(n_sw)),
        washout=washout,
    )


def switchback_sample_size(
    delta: float,
    sigma: float,
    *,
    n_regions: int,
    rho_ar1: float = 0.0,
    alpha: float = 0.05,
    power: float = 0.8,
    p_treat: float = 0.5,
    design: str = "balanced",
    switch_every: int = 1,
    washout: int = 0,
    min_periods: int = 4,
    max_periods: int = 10_000,
    seed: int = 0,
) -> PowerResult:
    """Smallest ``n_periods`` per region achieving the target power.

    Binary search over period count. Washout periods consume horizon
    without contributing to the contrast, so the search automatically
    inflates the calendar length.
    """
    if delta == 0:
        raise ValueError("delta must be non-zero")
    lo, hi = min_periods, min_periods
    fitted = switchback_power(
        delta,
        sigma,
        n_regions,
        hi,
        rho_ar1=rho_ar1,
        alpha=alpha,
        p_treat=p_treat,
        design=design,
        switch_every=switch_every,
        washout=washout,
        seed=seed,
    )
    while fitted.power < power and hi < max_periods:
        lo = hi
        hi = min(hi * 2, max_periods)
        fitted = switchback_power(
            delta,
            sigma,
            n_regions,
            hi,
            rho_ar1=rho_ar1,
            alpha=alpha,
            p_treat=p_treat,
            design=design,
            switch_every=switch_every,
            washout=washout,
            seed=seed,
        )
    if fitted.power < power:
        return fitted

    left, right = lo, hi
    best = fitted
    while left < right:
        mid = (left + right) // 2
        cand = switchback_power(
            delta,
            sigma,
            n_regions,
            mid,
            rho_ar1=rho_ar1,
            alpha=alpha,
            p_treat=p_treat,
            design=design,
            switch_every=switch_every,
            washout=washout,
            seed=seed,
        )
        if cand.power >= power:
            best = cand
            right = mid
        else:
            left = mid + 1
    return best
