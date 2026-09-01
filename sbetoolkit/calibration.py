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

from sbetoolkit.inference import block_ate, clustered_ate, iid_ate
from sbetoolkit.marketplace import MarketplaceConfig, MarketplaceSimulator


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
