"""Two-sided marketplace with a known global treatment effect.

Each (region, period) cell has a rider pool and a driver pool. Drivers
serve at most one trip per period. Riders request a trip with a
probability that depends on treatment. Matching is a random permutation
of requesters into the available driver slots.

Two assignment mechanisms share the same primitives:

* **Naive A/B** — 50% of *riders* in every cell are treated. Treated and
  control riders compete for the same drivers. Under binding supply the
  treated steal matches, so the rider-level contrast overstates the
  global policy effect (often near zero when the market is saturated).
* **Switchback** — the whole cell is treated or control. The block-level
  contrast estimates the global ATE (all-treat world minus all-control
  world).

Ground truth is computed by simulating both counterfactual worlds on the
same rider/driver counts (plus independent matching noise).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sbetoolkit.randomization import assign_switchback


@dataclass(frozen=True)
class MarketplaceConfig:
    n_regions: int = 8
    n_periods: int = 24
    riders_per_cell: int = 40
    drivers_per_cell: int = 18
    p_request_control: float = 0.70
    p_request_treat: float = 0.90
    # Extra drivers under cell-level (switchback) treatment. Zero in the
    # default crowding-out story; set >0 to study supply-side policies.
    extra_drivers_if_treated: int = 0
    # Pre-period covariate noise used by CUPED (cell intercept persistence).
    cell_intercept_sd: float = 0.04
    # Serial correlation of the cell intercept within a region (AR(1)).
    ar1_rho: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.riders_per_cell < 2:
            raise ValueError("need at least 2 riders to split a naive A/B")
        if not 0 < self.p_request_control < 1 or not 0 < self.p_request_treat < 1:
            raise ValueError("request probabilities must be in (0, 1)")


def _match(
    treated: np.ndarray,
    n_drivers: int,
    p_ctrl: float,
    p_treat: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return a 0/1 match indicator per rider."""
    p = np.where(treated, p_treat, p_ctrl)
    requests = rng.random(len(treated)) < p
    idx = np.flatnonzero(requests)
    rng.shuffle(idx)
    n_match = min(len(idx), int(n_drivers))
    y = np.zeros(len(treated), dtype=float)
    y[idx[:n_match]] = 1.0
    return y


class MarketplaceSimulator:
    def __init__(self, config: MarketplaceConfig | None = None) -> None:
        self.config = config or MarketplaceConfig()

    def _cell_sizes(self, rng: np.random.Generator, n_cells: int) -> tuple[np.ndarray, np.ndarray]:
        cfg = self.config
        # Mild Poisson jitter so cells are not identical.
        riders = rng.poisson(cfg.riders_per_cell, size=n_cells) + 2
        drivers = rng.poisson(cfg.drivers_per_cell, size=n_cells) + 1
        return riders, drivers

    def _region_intercepts(self, rng: np.random.Generator, table: pd.DataFrame) -> np.ndarray:
        from sbetoolkit.power import stationary_ar1

        cfg = self.config
        intercepts = np.zeros(len(table))
        for _, sub in table.groupby("region", sort=False):
            sub = sub.sort_values("period")
            intercepts[sub.index.to_numpy()] = stationary_ar1(
                len(sub), cfg.cell_intercept_sd, cfg.ar1_rho, rng
            )
        return intercepts

    def ground_truth(self, n_mc: int = 80, seed: int | None = None) -> dict[str, float]:
        """Monte Carlo global ATE: all-treat match rate − all-control match rate."""
        cfg = self.config
        rng = np.random.default_rng(cfg.seed if seed is None else seed)
        diffs = []
        n_cells = cfg.n_regions * cfg.n_periods
        for _ in range(n_mc):
            riders, drivers = self._cell_sizes(rng, n_cells)
            y_t = []
            y_c = []
            for n_r, n_d in zip(riders, drivers):
                treated = np.ones(n_r, dtype=bool)
                control = np.zeros(n_r, dtype=bool)
                extra = cfg.extra_drivers_if_treated
                y_t.append(
                    _match(treated, n_d + extra, cfg.p_request_control, cfg.p_request_treat, rng).mean()
                )
                y_c.append(
                    _match(control, n_d, cfg.p_request_control, cfg.p_request_treat, rng).mean()
                )
            diffs.append(float(np.mean(y_t) - np.mean(y_c)))
        return {
            "ate": float(np.mean(diffs)),
            "se": float(np.std(diffs, ddof=1) / np.sqrt(len(diffs))),
            "n_mc": float(n_mc),
        }

    def run_naive_ab(self, seed: int | None = None) -> dict:
        """Rider-level 50/50 randomization inside every cell."""
        cfg = self.config
        rng = np.random.default_rng(cfg.seed if seed is None else seed)
        n_cells = cfg.n_regions * cfg.n_periods
        riders, drivers = self._cell_sizes(rng, n_cells)
        intercepts = rng.normal(0.0, cfg.cell_intercept_sd, size=n_cells)

        y_t, y_c = [], []
        cell_rows = []
        rider_treat, rider_y, rider_x, rider_cell = [], [], [], []

        for cell, (n_r, n_d, b) in enumerate(zip(riders, drivers, intercepts)):
            treated = np.zeros(n_r, dtype=bool)
            treated[: n_r // 2] = True
            rng.shuffle(treated)
            y = _match(treated, n_d, cfg.p_request_control, cfg.p_request_treat, rng)
            # Covariate: pre-period match rate ≈ intercept + noise (untreated world).
            x = np.clip(cfg.p_request_control + b + rng.normal(0, 0.05, n_r), 0, 1)
            y_obs = np.clip(y + b, 0, 1)

            y_t.append(y_obs[treated].mean() if treated.any() else np.nan)
            y_c.append(y_obs[~treated].mean() if (~treated).any() else np.nan)
            cell_rows.append(
                {
                    "cell": cell,
                    "n_riders": n_r,
                    "n_drivers": n_d,
                    "match_treat": float(y_obs[treated].mean()),
                    "match_control": float(y_obs[~treated].mean()),
                    "match_all": float(y_obs.mean()),
                }
            )
            rider_treat.append(treated.astype(int))
            rider_y.append(y_obs)
            rider_x.append(x)
            rider_cell.append(np.full(n_r, cell))

        yt = np.concatenate(rider_y)
        tt = np.concatenate(rider_treat)
        xt = np.concatenate(rider_x)
        naive_ate = float(yt[tt == 1].mean() - yt[tt == 0].mean())
        se = float(
            np.sqrt(
                yt[tt == 1].var(ddof=1) / (tt == 1).sum()
                + yt[tt == 0].var(ddof=1) / (tt == 0).sum()
            )
        )
        return {
            "ate": naive_ate,
            "se": se,
            "rider_y": yt,
            "rider_treatment": tt,
            "rider_x": xt,
            "rider_cell": np.concatenate(rider_cell),
            "cells": pd.DataFrame(cell_rows),
        }

    def run_switchback(
        self,
        seed: int | None = None,
        *,
        design: str = "balanced",
        washout: int = 0,
        switch_every: int = 1,
        return_riders: bool = False,
    ) -> dict:
        """Cell-level switchback: one policy per (region, period).

        ``ate`` / ``se`` are the *block-level* difference in means (one
        row per randomized cell). Set ``return_riders=True`` to also get
        ride-level arrays for the iid-vs-cluster Type I check.
        """
        from sbetoolkit.inference import estimate_switchback

        cfg = self.config
        rng = np.random.default_rng(cfg.seed if seed is None else seed)
        regions = [f"r{i}" for i in range(cfg.n_regions)]
        periods = list(range(cfg.n_periods))
        assignment = assign_switchback(
            regions,
            periods,
            design=design,  # type: ignore[arg-type]
            washout=washout,
            switch_every=switch_every,
            seed=int(rng.integers(0, 1_000_000_000)),
        )
        table = assignment.table
        n_cells = len(table)
        riders, drivers = self._cell_sizes(rng, n_cells)
        intercepts = self._region_intercepts(rng, table)
        treat_col = table["treatment"].to_numpy()

        outcomes = []
        rider_y: list[np.ndarray] = []
        rider_t: list[np.ndarray] = []
        rider_block: list[np.ndarray] = []

        for i in range(n_cells):
            n_r = int(riders[i])
            n_d = int(drivers[i])
            treat = int(treat_col[i])
            treated = np.full(n_r, bool(treat))
            extra = cfg.extra_drivers_if_treated if treat else 0
            y = _match(treated, n_d + extra, cfg.p_request_control, cfg.p_request_treat, rng)
            b = intercepts[i]
            y_obs = np.clip(y + b, 0, 1)
            x = float(
                np.clip(
                    cfg.p_request_control * min(1.0, n_d / max(n_r * cfg.p_request_control, 1e-9))
                    + b
                    + rng.normal(0, 0.03),
                    0,
                    1,
                )
            )
            outcomes.append(
                {
                    "region": table["region"].iloc[i],
                    "period": table["period"].iloc[i],
                    "treatment": treat,
                    "n_riders": n_r,
                    "n_drivers": n_d + extra,
                    "match_rate": float(y_obs.mean()),
                    "pre_match_rate": x,
                    "block_id": i,
                }
            )
            if return_riders:
                rider_y.append(y_obs)
                rider_t.append(np.full(n_r, treat, dtype=int))
                rider_block.append(np.full(n_r, i, dtype=int))

        frame = pd.DataFrame(outcomes)
        analysis = assignment.for_analysis(frame)
        est = estimate_switchback(assignment, frame)
        out: dict = {
            "ate": est.ate,
            "se": est.se,
            "pvalue": est.pvalue,
            "estimate": est,
            "assignment": assignment,
            "blocks": frame,
            "analysis": analysis,
        }
        if return_riders:
            keep_ids = set(analysis["block_id"].to_numpy())
            block_ids = np.concatenate(rider_block)
            keep = np.isin(block_ids, list(keep_ids))
            out["rider_y"] = np.concatenate(rider_y)[keep]
            out["rider_treatment"] = np.concatenate(rider_t)[keep]
            out["rider_block"] = block_ids[keep]
        return out

    def compare_estimators(self, n_reps: int = 64, seed: int = 1) -> pd.DataFrame:
        """Paired Monte Carlo of naive A/B vs switchback vs ground truth.

        Ground truth is shared (computed once). Each replication redraws
        cell sizes, matching, and the switchback assignment.
        """
        truth = self.ground_truth(n_mc=60, seed=seed)
        rows = []
        rng = np.random.default_rng(seed)
        for i in range(n_reps):
            s = int(rng.integers(0, 1_000_000_000))
            naive = self.run_naive_ab(seed=s)
            sw = self.run_switchback(seed=s + 1)
            rows.append(
                {
                    "rep": i,
                    "naive_ate": naive["ate"],
                    "switchback_ate": sw["ate"],
                    "truth_ate": truth["ate"],
                    "naive_bias": naive["ate"] - truth["ate"],
                    "switchback_bias": sw["ate"] - truth["ate"],
                }
            )
        return pd.DataFrame(rows)
