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
    # Share of leftover drivers who will serve unmatched demand in a
    # neighboring region (0 = sealed zones, 1 = leftover supply is shared).
    # Has bite when control cells have idle drivers and treated cells have
    # leftover demand — not when every cell is already supply-constrained.
    driver_leakage: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.riders_per_cell < 2:
            raise ValueError("need at least 2 riders to split a naive A/B")
        if not 0 < self.p_request_control < 1 or not 0 < self.p_request_treat < 1:
            raise ValueError("request probabilities must be in (0, 1)")
        if not 0 <= self.driver_leakage <= 1:
            raise ValueError("driver_leakage must be in [0, 1]")


def _match(
    treated: np.ndarray,
    n_drivers: int,
    p_ctrl: float,
    p_treat: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int, int]:
    """Local matching. Returns (y, leftover_drivers, leftover_demand)."""
    p = np.where(treated, p_treat, p_ctrl)
    requests = rng.random(len(treated)) < p
    idx = np.flatnonzero(requests)
    rng.shuffle(idx)
    n_match = min(len(idx), int(n_drivers))
    y = np.zeros(len(treated), dtype=float)
    y[idx[:n_match]] = 1.0
    leftover_drivers = max(0, int(n_drivers) - n_match)
    leftover_demand = max(0, int(len(idx) - n_match))
    return y, leftover_drivers, leftover_demand


def _leak_extra_matches(
    leftover_drivers: np.ndarray,
    leftover_demand: np.ndarray,
    leakage: float,
) -> np.ndarray:
    """Inbound extra matches from neighboring leftover supply (ring)."""
    leftover_drivers = np.asarray(leftover_drivers, dtype=int)
    leftover_demand = np.asarray(leftover_demand, dtype=int)
    n = leftover_drivers.size
    extra = np.zeros(n, dtype=int)
    if leakage <= 0 or n < 2:
        return extra
    supply = leftover_drivers.copy()
    need = leftover_demand.copy()
    for i in range(n):
        budget = min(int(round(leakage * int(leftover_drivers[i]))), int(supply[i]))
        if budget <= 0:
            continue
        if n == 2:
            neighbors = [1 - i]
        else:
            neighbors = [(i - 1) % n, (i + 1) % n]
        needy = [j for j in neighbors if need[j] > 0]
        for j in needy:
            if budget <= 0:
                break
            give = min(need[j], budget)
            extra[j] += give
            need[j] -= give
            budget -= give
            supply[i] -= give
    return extra


def _attach_inbound(y: np.ndarray, n_extra: int, rng: np.random.Generator) -> np.ndarray:
    """Assign leftover inbound matches to currently unmatched riders."""
    if n_extra <= 0:
        return y
    y = y.copy()
    open_slots = np.flatnonzero(y == 0)
    rng.shuffle(open_slots)
    y[open_slots[: min(n_extra, open_slots.size)]] = 1.0
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

    def _match_regions(
        self,
        n_riders: np.ndarray,
        n_drivers: np.ndarray,
        treatment: np.ndarray,
        rng: np.random.Generator,
    ) -> list[np.ndarray]:
        """Match one period across regions, then leak leftover drivers to neighbors."""
        cfg = self.config
        ys: list[np.ndarray] = []
        leftover_d: list[int] = []
        leftover_q: list[int] = []
        for n_r, n_d, t in zip(n_riders, n_drivers, treatment):
            extra = cfg.extra_drivers_if_treated if int(t) else 0
            treated = np.full(int(n_r), bool(t))
            y, ld, lq = _match(
                treated,
                int(n_d) + extra,
                cfg.p_request_control,
                cfg.p_request_treat,
                rng,
            )
            ys.append(y)
            leftover_d.append(ld)
            leftover_q.append(lq)
        inbound = _leak_extra_matches(
            np.asarray(leftover_d), np.asarray(leftover_q), cfg.driver_leakage
        )
        return [_attach_inbound(y, int(k), rng) for y, k in zip(ys, inbound)]

    def _world_mean_rate(self, rng: np.random.Generator, all_treated: bool) -> float:
        cfg = self.config
        n_cells = cfg.n_regions * cfg.n_periods
        riders, drivers = self._cell_sizes(rng, n_cells)
        riders = riders.reshape(cfg.n_regions, cfg.n_periods)
        drivers = drivers.reshape(cfg.n_regions, cfg.n_periods)
        treat = np.full(cfg.n_regions, 1 if all_treated else 0)
        rates = []
        for p in range(cfg.n_periods):
            ys = self._match_regions(riders[:, p], drivers[:, p], treat, rng)
            rates.extend(float(y.mean()) for y in ys)
        return float(np.mean(rates))

    def ground_truth(self, n_mc: int = 80, seed: int | None = None) -> dict[str, float]:
        """Monte Carlo global ATE: all-treat match rate − all-control match rate."""
        cfg = self.config
        rng = np.random.default_rng(cfg.seed if seed is None else seed)
        diffs = []
        for _ in range(n_mc):
            y_t = self._world_mean_rate(rng, True)
            y_c = self._world_mean_rate(rng, False)
            diffs.append(y_t - y_c)
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
            y, _, _ = _match(treated, n_d, cfg.p_request_control, cfg.p_request_treat, rng)
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
        cluster_size: int = 1,
        spatial_buffer: int = 0,
        return_riders: bool = False,
    ) -> dict:
        """Cell-level switchback: one policy per (region, period).

        ``cluster_size`` groups adjacent regions onto one sequence so a
        spatial buffer has an interior. ``spatial_buffer`` drops cells
        that border the opposite arm (same period). Leftover drivers
        leak to neighbors when ``config.driver_leakage > 0``.
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
            cluster_size=cluster_size,
            spatial_buffer=spatial_buffer,
            seed=int(rng.integers(0, 1_000_000_000)),
        )
        table = assignment.table
        n_cells = len(table)
        riders, drivers = self._cell_sizes(rng, n_cells)
        intercepts = self._region_intercepts(rng, table)
        work = table.copy()
        work["n_riders"] = riders
        work["n_drivers"] = drivers
        work["intercept"] = intercepts
        region_order = {r: i for i, r in enumerate(regions)}

        outcomes: list[dict] = []
        rider_y: list[np.ndarray] = []
        rider_t: list[np.ndarray] = []
        rider_block: list[np.ndarray] = []

        for period, g in work.groupby("period"):
            g = g.assign(_ord=g["region"].map(region_order)).sort_values("_ord")
            treat = g["treatment"].to_numpy()
            ys = self._match_regions(
                g["n_riders"].to_numpy(),
                g["n_drivers"].to_numpy(),
                treat,
                rng,
            )
            for row, y, t in zip(g.itertuples(), ys, treat):
                b = float(row.intercept)
                y_obs = np.clip(y + b, 0, 1)
                extra = cfg.extra_drivers_if_treated if int(t) else 0
                n_r = int(row.n_riders)
                n_d = int(row.n_drivers) + extra
                x = float(
                    np.clip(
                        cfg.p_request_control * min(1.0, n_d / max(n_r * cfg.p_request_control, 1e-9))
                        + b
                        + rng.normal(0, 0.03),
                        0,
                        1,
                    )
                )
                idx = int(row.Index)
                outcomes.append(
                    {
                        "region": row.region,
                        "period": row.period,
                        "treatment": int(t),
                        "n_riders": n_r,
                        "n_drivers": n_d,
                        "match_rate": float(y_obs.mean()),
                        "pre_match_rate": x,
                        "block_id": idx,
                    }
                )
                if return_riders:
                    rider_y.append(y_obs)
                    rider_t.append(np.full(n_r, int(t), dtype=int))
                    rider_block.append(np.full(n_r, idx, dtype=int))

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

        Each row is one experiment: point estimates, 95% Wald intervals,
        and whether those intervals contain the known global ATE.
        """
        from sbetoolkit.inference import iid_ate

        truth = self.ground_truth(n_mc=200, seed=seed)
        truth_ate = truth["ate"]
        rows = []
        rng = np.random.default_rng(seed)
        for i in range(n_reps):
            s = int(rng.integers(0, 1_000_000_000))
            naive = self.run_naive_ab(seed=s)
            sw = self.run_switchback(seed=s + 1)
            naive_est = iid_ate(naive["rider_y"], naive["rider_treatment"])
            sw_est = sw["estimate"]
            n_lo, n_hi = naive_est.interval()
            s_lo, s_hi = sw_est.interval()
            rows.append(
                {
                    "rep": i,
                    "naive_ate": naive_est.ate,
                    "naive_se": naive_est.se,
                    "naive_lo": n_lo,
                    "naive_hi": n_hi,
                    "naive_covers": naive_est.covers(truth_ate),
                    "switchback_ate": sw_est.ate,
                    "switchback_se": sw_est.se,
                    "switchback_lo": s_lo,
                    "switchback_hi": s_hi,
                    "switchback_covers": sw_est.covers(truth_ate),
                    "truth_ate": truth_ate,
                    "naive_bias": naive_est.ate - truth_ate,
                    "switchback_bias": sw_est.ate - truth_ate,
                }
            )
        return pd.DataFrame(rows)


def summarize_estimators(comparison: pd.DataFrame) -> pd.DataFrame:
    """Coverage, bias, and RMSE of naive A/B vs switchback vs truth."""
    truth = float(comparison["truth_ate"].iloc[0])

    def _row(name: str, ate: np.ndarray, covers: np.ndarray | None) -> dict:
        err = ate - truth
        return {
            "estimator": name,
            "mean": float(ate.mean()),
            "bias": float(err.mean()),
            "rmse": float(np.sqrt(np.mean(err**2))),
            "coverage": float(covers.mean()) if covers is not None else float("nan"),
        }

    return pd.DataFrame(
        [
            _row(
                "Naive rider A/B",
                comparison["naive_ate"].to_numpy(),
                comparison["naive_covers"].to_numpy(),
            ),
            _row(
                "Time-region switchback",
                comparison["switchback_ate"].to_numpy(),
                comparison["switchback_covers"].to_numpy(),
            ),
            _row(
                "Ground truth",
                np.full(len(comparison), truth),
                None,
            ),
        ]
    )
