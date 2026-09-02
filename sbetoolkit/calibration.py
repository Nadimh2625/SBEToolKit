"""Null (true effect = 0) Type I calibration for switchback inference.

Run the marketplace with ``p_request_treat == p_request_control`` a
thousand times. Under the null a 5% test should reject about 5% of the
time. Analyzing at the ride level after randomizing blocks typically
rejects 10–20%: the iid SE thinks you have tens of thousands of
independent observations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sbetoolkit.inference import block_ate, clustered_ate, estimate_switchback, iid_ate
from sbetoolkit.marketplace import MarketplaceConfig, MarketplaceSimulator
from sbetoolkit.power import (
    simulate_ar1_switchback,
    switchback_formula_se,
    switchback_power,
    switchback_sample_size,
)
from sbetoolkit.randomization import assign_switchback


def null_config(
    *,
    n_regions: int = 10,
    n_periods: int = 50,
    riders_per_cell: int = 80,
    drivers_per_cell: int = 40,
    p_request: float = 0.65,
    cell_intercept_sd: float = 0.15,
    seed: int = 11,
) -> MarketplaceConfig:
    """Demand treatment turned off: treated and control request at ``p_request``."""
    return MarketplaceConfig(
        n_regions=n_regions,
        n_periods=n_periods,
        riders_per_cell=riders_per_cell,
        drivers_per_cell=drivers_per_cell,
        p_request_control=p_request,
        p_request_treat=p_request,
        extra_drivers_if_treated=0,
        cell_intercept_sd=cell_intercept_sd,
        seed=seed,
    )


@dataclass(frozen=True)
class TypeIResult:
    n_reps: int
    alpha: float
    n_blocks: int
    n_rides_per_rep: float
    iid_rate: float
    cluster_rate: float
    block_rate: float
    iid_mean_n: float
    cluster_mean_n_clusters: float
    table: pd.DataFrame

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "estimator": [
                    "iid at ride level (wrong)",
                    "cluster-robust at block (right)",
                    "difference in means on blocks (right)",
                ],
                "randomization_unit": ["ride", "block", "block"],
                "mean_n_used_in_se": [
                    self.iid_mean_n,
                    self.cluster_mean_n_clusters,
                    self.cluster_mean_n_clusters,
                ],
                "reject_rate": [self.iid_rate, self.cluster_rate, self.block_rate],
                "nominal_alpha": [self.alpha, self.alpha, self.alpha],
            }
        )


def type_i_null_check(
    n_reps: int = 1000,
    *,
    alpha: float = 0.05,
    seed: int = 11,
    config: MarketplaceConfig | None = None,
) -> TypeIResult:
    """Monte Carlo Type I rates for iid vs block-aware switchback tests."""
    if n_reps < 1:
        raise ValueError("n_reps must be positive")
    cfg = config or null_config(seed=seed)
    if abs(cfg.p_request_treat - cfg.p_request_control) > 1e-15:
        raise ValueError("null check requires p_request_treat == p_request_control")
    if cfg.extra_drivers_if_treated != 0:
        raise ValueError("null check requires extra_drivers_if_treated == 0")

    sim = MarketplaceSimulator(cfg)
    rng = np.random.default_rng(seed)
    rows = []
    n_rides = []
    for i in range(n_reps):
        s = int(rng.integers(0, 1_000_000_000))
        run = sim.run_switchback(seed=s, return_riders=True)
        y = run["rider_y"]
        t = run["rider_treatment"]
        block = run["rider_block"]
        analysis = run["analysis"]
        iid = iid_ate(y, t)
        cl = clustered_ate(y, t, block)
        blk = block_ate(
            analysis["match_rate"].to_numpy(),
            analysis["treatment"].to_numpy(),
        )
        n_rides.append(y.size)
        rows.append(
            {
                "rep": i,
                "iid_ate": iid.ate,
                "iid_se": iid.se,
                "iid_reject": iid.reject(alpha),
                "iid_n": iid.n_obs,
                "cluster_ate": cl.ate,
                "cluster_se": cl.se,
                "cluster_reject": cl.reject(alpha),
                "cluster_g": cl.n_clusters,
                "block_ate": blk.ate,
                "block_se": blk.se,
                "block_reject": blk.reject(alpha),
            }
        )
    table = pd.DataFrame(rows)
    return TypeIResult(
        n_reps=n_reps,
        alpha=alpha,
        n_blocks=cfg.n_regions * cfg.n_periods,
        n_rides_per_rep=float(np.mean(n_rides)),
        iid_rate=float(table["iid_reject"].mean()),
        cluster_rate=float(table["cluster_reject"].mean()),
        block_rate=float(table["block_reject"].mean()),
        iid_mean_n=float(table["iid_n"].mean()),
        cluster_mean_n_clusters=float(table["cluster_g"].mean()),
        table=table,
    )


@dataclass(frozen=True)
class PowerCheckResult:
    n_reps: int
    alpha: float
    target_power: float
    predicted_power: float
    empirical_power: float
    welch_power: float
    n_regions: int
    n_periods: int
    n_blocks: int
    delta: float
    sigma: float
    rho_ar1: float
    washout: int
    formula_se: float
    monte_carlo_se: float
    se_ratio: float
    table: pd.DataFrame

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "quantity": [
                    "predicted power (w′Γw)",
                    "empirical detection rate (formula SE)",
                    "empirical detection rate (Welch SE)",
                    "n_regions",
                    "n_periods per region",
                    "blocks (region × period)",
                    "δ",
                    "σ",
                    "ρ",
                    "formula SE",
                    "Monte Carlo SD of ATE",
                    "MC SD / formula SE",
                ],
                "value": [
                    self.predicted_power,
                    self.empirical_power,
                    self.welch_power,
                    self.n_regions,
                    self.n_periods,
                    self.n_blocks,
                    self.delta,
                    self.sigma,
                    self.rho_ar1,
                    self.formula_se,
                    self.monte_carlo_se,
                    self.se_ratio,
                ],
            }
        )


def empirical_power_check(
    delta: float = 0.05,
    sigma: float = 0.12,
    *,
    n_regions: int = 8,
    rho_ar1: float = 0.4,
    target_power: float = 0.8,
    alpha: float = 0.05,
    n_reps: int = 500,
    washout: int = 0,
    design: str = "balanced",
    seed: int = 13,
) -> PowerCheckResult:
    """Size an experiment for ``target_power``, then simulate it.

    Outcomes are ``Y = δ T + AR(1)`` on the switchback grid — the model
    behind ``w'Γw``. Each replication draws a new design, simulates
    outcomes, and tests H₀: ATE = 0 two ways:

    * formula SE (known σ, the test :func:`switchback_sample_size` powers)
    * Welch SE on the analysis table (what :func:`estimate_switchback` uses)

    If the formula is right, the first detection rate is near ``target_power``
    and the Monte Carlo SD of the ATE matches the formula SE.
    """
    from scipy.stats import norm

    if n_reps < 1:
        raise ValueError("n_reps must be positive")
    sized = switchback_sample_size(
        delta,
        sigma,
        n_regions=n_regions,
        rho_ar1=rho_ar1,
        alpha=alpha,
        power=target_power,
        design=design,
        washout=washout,
        seed=seed,
    )
    predicted = switchback_power(
        delta,
        sigma,
        n_regions,
        sized.n_periods,
        rho_ar1=rho_ar1,
        alpha=alpha,
        design=design,
        washout=washout,
        seed=seed,
    )
    z_crit = float(norm.ppf(1 - alpha / 2))
    rng = np.random.default_rng(seed)
    regions = [f"r{i}" for i in range(n_regions)]
    periods = list(range(sized.n_periods))
    rows = []
    for i in range(n_reps):
        assignment = assign_switchback(
            regions,
            periods,
            design=design,  # type: ignore[arg-type]
            washout=washout,
            seed=int(rng.integers(0, 1_000_000_000)),
        )
        outcomes = simulate_ar1_switchback(
            assignment, delta, sigma, rho_ar1, rng
        )
        welch = estimate_switchback(assignment, outcomes)
        se_f = switchback_formula_se(assignment, sigma, rho_ar1)
        z = welch.ate / se_f if se_f > 0 else np.inf
        rows.append(
            {
                "rep": i,
                "ate": welch.ate,
                "formula_se": se_f,
                "welch_se": welch.se,
                "formula_reject": bool(abs(z) >= z_crit),
                "welch_reject": welch.reject(alpha),
            }
        )
    table = pd.DataFrame(rows)
    mc_se = float(table["ate"].std(ddof=1))
    formula_se_mean = float(table["formula_se"].mean())
    return PowerCheckResult(
        n_reps=n_reps,
        alpha=alpha,
        target_power=target_power,
        predicted_power=predicted.power,
        empirical_power=float(table["formula_reject"].mean()),
        welch_power=float(table["welch_reject"].mean()),
        n_regions=n_regions,
        n_periods=sized.n_periods,
        n_blocks=n_regions * sized.n_periods,
        delta=delta,
        sigma=sigma,
        rho_ar1=rho_ar1,
        washout=washout,
        formula_se=formula_se_mean,
        monte_carlo_se=mc_se,
        se_ratio=mc_se / formula_se_mean if formula_se_mean > 0 else np.nan,
        table=table,
    )
