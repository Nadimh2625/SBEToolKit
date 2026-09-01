"""Diagnostics for interference / SUTVA violations.

The naive rider-level contrast and the global policy effect need not
agree. Following the usual marketplace decomposition:

    naive  = E[Y | treat, mixed] − E[Y | control, mixed]
    global = E[Y | all treat] − E[Y | all control]

    crowding-out of control = E[Y | all control] − E[Y | control, mixed]
        (positive ⇒ control is hurt by sharing the market with treated)

    dilution of treated     = E[Y | treat, mixed] − E[Y | all treat]
        (positive ⇒ treated look better in a mixed market than they
        would under a full rollout — they are harvesting scarce supply)

If ``naive > global`` the A/B test **overstates** a launch. If
``naive < global`` it **understates** (typical of positive spillovers,
e.g. extra drivers that also serve the control group).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sbetoolkit.marketplace import MarketplaceSimulator


@dataclass(frozen=True)
class InterferenceReport:
    naive_ate: float
    global_ate: float
    mixed_treat_rate: float
    mixed_control_rate: float
    all_treat_rate: float
    all_control_rate: float
    control_crowding_out: float
    treated_dilution: float
    bias: float
    relative_bias: float
    direction: str

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "quantity": [
                    "naive A/B ATE",
                    "global ATE (truth)",
                    "mixed treated match rate",
                    "mixed control match rate",
                    "all-treat match rate",
                    "all-control match rate",
                    "control crowding-out",
                    "treated dilution",
                    "naive − global (bias)",
                    "relative bias (naive/global − 1)",
                ],
                "value": [
                    self.naive_ate,
                    self.global_ate,
                    self.mixed_treat_rate,
                    self.mixed_control_rate,
                    self.all_treat_rate,
                    self.all_control_rate,
                    self.control_crowding_out,
                    self.treated_dilution,
                    self.bias,
                    self.relative_bias,
                ],
            }
        )


def _cell_world_rate(
    sim: MarketplaceSimulator,
    *,
    all_treated: bool,
    n_mc: int,
    rng: np.random.Generator,
) -> float:
    cfg = sim.config
    rates = []
    n_cells = cfg.n_regions * cfg.n_periods
    for _ in range(n_mc):
        riders, drivers = sim._cell_sizes(rng, n_cells)
        cell_rates = []
        for n_r, n_d in zip(riders, drivers):
            treated = np.full(n_r, all_treated)
            extra = cfg.extra_drivers_if_treated if all_treated else 0
            from sbetoolkit.marketplace import _match

            y = _match(
                treated,
                n_d + extra,
                cfg.p_request_control,
                cfg.p_request_treat,
                rng,
            )
            cell_rates.append(y.mean())
        rates.append(float(np.mean(cell_rates)))
    return float(np.mean(rates))


def diagnose_interference(
    sim: MarketplaceSimulator,
    *,
    n_mc: int = 40,
    seed: int | None = None,
) -> InterferenceReport:
    """Compare mixed-market A/B moments to both global counterfactuals."""
    cfg = sim.config
    rng = np.random.default_rng(cfg.seed if seed is None else seed)

    naive = sim.run_naive_ab(seed=int(rng.integers(0, 1_000_000_000)))
    mixed_t = float(naive["cells"]["match_treat"].mean())
    mixed_c = float(naive["cells"]["match_control"].mean())
    naive_ate = mixed_t - mixed_c

    all_t = _cell_world_rate(sim, all_treated=True, n_mc=n_mc, rng=rng)
    all_c = _cell_world_rate(sim, all_treated=False, n_mc=n_mc, rng=rng)
    global_ate = all_t - all_c

    crowding = all_c - mixed_c
    dilution = mixed_t - all_t
    bias = naive_ate - global_ate
    if abs(global_ate) < 1e-12:
        rel = np.inf if abs(naive_ate) > 1e-12 else 0.0
    else:
        rel = naive_ate / global_ate - 1.0

    if bias > 1e-4:
        direction = "naive A/B overstates the global effect"
    elif bias < -1e-4:
        direction = "naive A/B understates the global effect"
    else:
        direction = "naive A/B matches the global effect"

    return InterferenceReport(
        naive_ate=float(naive_ate),
        global_ate=float(global_ate),
        mixed_treat_rate=mixed_t,
        mixed_control_rate=mixed_c,
        all_treat_rate=all_t,
        all_control_rate=all_c,
        control_crowding_out=float(crowding),
        treated_dilution=float(dilution),
        bias=float(bias),
        relative_bias=float(rel),
        direction=direction,
    )
