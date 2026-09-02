"""Switchback randomization over time × region blocks.

A *block* is a (region, period) cell. Every unit in a treated block sees
the same policy, which is what makes the design identify a global ATE
instead of a within-market cannibalization contrast.

Designs
-------
``independent``
    Each block is treated independently with probability ``p_treat``.
``balanced``
    Each region gets a permutation of a sequence with a fixed number of
    treated periods (default: half). This is the usual production design.
``blocked_random``
    Periods are partitioned into windows of length ``switch_every``. Each
    window is assigned as a unit (reduces switches / carryover).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

DesignName = Literal["independent", "balanced", "blocked_random"]


@dataclass(frozen=True)
class SwitchbackAssignment:
    """Realized switchback design."""

    table: pd.DataFrame
    design: DesignName
    p_treat: float
    washout: int
    switch_every: int
    spatial_buffer: int = 0
    cluster_size: int = 1

    @property
    def analysis_table(self) -> pd.DataFrame:
        """Blocks that are not in a temporal washout or spatial buffer."""
        drop = self.table["is_washout"].to_numpy()
        if "is_spatial_buffer" in self.table.columns:
            drop = drop | self.table["is_spatial_buffer"].to_numpy()
        return self.table.loc[~drop].copy()

    def analysis_mask(self) -> np.ndarray:
        """Boolean mask on ``table``, True iff the row is in ``analysis_table``.

        Power calculations keep the full calendar (for AR(1) lags) and
        use this mask so washout periods occupy time without entering
        the contrast.
        """
        kept = pd.MultiIndex.from_frame(self.analysis_table[["region", "period"]])
        idx = pd.MultiIndex.from_frame(self.table[["region", "period"]])
        return np.asarray(idx.isin(kept))

    def for_analysis(self, outcomes: pd.DataFrame) -> pd.DataFrame:
        """Inner-join ``outcomes`` onto :attr:`analysis_table`.

        Rows whose ``(region, period)`` is a washout (or burn-in) never
        appear in the result, even if they are present in ``outcomes``
        with garbage values.
        """
        keys = ["region", "period"]
        missing = [c for c in keys if c not in outcomes.columns]
        if missing:
            raise ValueError(f"outcomes missing columns {missing}")
        sample = outcomes.merge(self.analysis_table[keys], on=keys, how="inner")
        if sample.empty:
            raise ValueError("no (region, period) overlap with analysis_table")
        return sample

    def n_switches(self) -> pd.Series:
        """Number of treatment flips per region."""
        def _flips(s: pd.Series) -> int:
            t = s.to_numpy()
            return int(np.sum(t[1:] != t[:-1]))

        return self.table.groupby("region")["treatment"].apply(_flips)


def _balanced_sequence(
    n_periods: int,
    n_treated: int,
    rng: np.random.Generator,
) -> np.ndarray:
    seq = np.zeros(n_periods, dtype=int)
    seq[:n_treated] = 1
    rng.shuffle(seq)
    return seq


def _blocked_sequence(
    n_periods: int,
    switch_every: int,
    p_treat: float,
    rng: np.random.Generator,
) -> np.ndarray:
    n_windows = int(np.ceil(n_periods / switch_every))
    window_treat = (rng.random(n_windows) < p_treat).astype(int)
    seq = np.repeat(window_treat, switch_every)[:n_periods]
    return seq


def _mark_washout(treatment: np.ndarray, washout: int) -> np.ndarray:
    """Flag the first ``washout`` periods after each switch (inclusive of the new state)."""
    if washout <= 0:
        return np.zeros_like(treatment, dtype=bool)
    flags = np.zeros(len(treatment), dtype=bool)
    flags[0] = True  # burn-in at the start of the series
    switches = np.where(treatment[1:] != treatment[:-1])[0] + 1
    for s in switches:
        flags[s : s + washout] = True
    return flags


def ring_adjacency(regions: list[str]) -> dict[str, list[str]]:
    """Each region neighbors the previous and next on a circle."""
    n = len(regions)
    if n <= 1:
        return {regions[0]: []} if n == 1 else {}
    if n == 2:
        a, b = regions
        return {a: [b], b: [a]}
    adj: dict[str, list[str]] = {}
    for i, r in enumerate(regions):
        adj[r] = [regions[(i - 1) % n], regions[(i + 1) % n]]
    return adj


def _k_hop_neighbors(region: str, adjacency: dict[str, list[str]], hops: int) -> set[str]:
    seen = {region}
    frontier = {region}
    for _ in range(hops):
        nxt: set[str] = set()
        for node in frontier:
            nxt.update(adjacency.get(node, []))
        nxt -= seen
        seen |= nxt
        frontier = nxt
    seen.discard(region)
    return seen


def _mark_spatial_buffer(
    table: pd.DataFrame,
    adjacency: dict[str, list[str]],
    hops: int,
) -> np.ndarray:
    """Flag cells that border opposite treatment within ``hops`` hops."""
    flags = np.zeros(len(table), dtype=bool)
    if hops <= 0 or not adjacency:
        return flags
    pos = {idx: i for i, idx in enumerate(table.index)}
    for _, g in table.groupby("period"):
        treat = dict(zip(g["region"].to_numpy(), g["treatment"].to_numpy()))
        for idx, region, t in zip(g.index, g["region"].to_numpy(), g["treatment"].to_numpy()):
            for nb in _k_hop_neighbors(str(region), adjacency, hops):
                if treat.get(nb, t) != t:
                    flags[pos[idx]] = True
                    break
    return flags


def assign_switchback(
    regions: list[str] | np.ndarray | pd.Index,
    periods: list[int] | np.ndarray | range,
    *,
    p_treat: float = 0.5,
    design: DesignName = "balanced",
    switch_every: int = 1,
    washout: int = 0,
    n_treated: int | None = None,
    cluster_size: int = 1,
    spatial_buffer: int = 0,
    seed: int | None = None,
) -> SwitchbackAssignment:
    """Assign treatment to time-region blocks.

    Parameters
    ----------
    regions, periods
        Cartesian product of these forms the experimental grid.
        Region *order* is the geography: neighbors are adjacent on a ring.
    p_treat
        Target treatment probability (used by independent / blocked designs
        and to set the treated count when ``n_treated`` is omitted).
    design
        See module docstring.
    switch_every
        Window length for ``blocked_random``. Ignored otherwise.
    washout
        Number of periods after each switch to drop from analysis. Models
        carryover / marketplace re-equilibration.
    n_treated
        Exact treated-period count per region for ``balanced``. Defaults to
        ``round(p_treat * n_periods)``.
    cluster_size
        Adjacent regions that share one treatment sequence. Size 1 is
        independent regions; size 3 plus ``spatial_buffer=1`` leaves an
        interior cell that does not border the opposite arm.
    spatial_buffer
        Drop cells whose treatment differs from a region within this many
        hops (same period). Models a border where drivers can leak.
    """
    if not 0 < p_treat < 1:
        raise ValueError("p_treat must be in (0, 1)")
    if washout < 0:
        raise ValueError("washout must be >= 0")
    if switch_every < 1:
        raise ValueError("switch_every must be >= 1")
    if cluster_size < 1:
        raise ValueError("cluster_size must be >= 1")
    if spatial_buffer < 0:
        raise ValueError("spatial_buffer must be >= 0")

    regions = list(regions)
    periods = list(periods)
    n_periods = len(periods)
    if n_periods == 0 or not regions:
        raise ValueError("regions and periods must be non-empty")

    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    n_groups = int(np.ceil(len(regions) / cluster_size))
    for g in range(n_groups):
        members = regions[g * cluster_size : (g + 1) * cluster_size]
        if design == "independent":
            seq = (rng.random(n_periods) < p_treat).astype(int)
        elif design == "balanced":
            k = n_treated if n_treated is not None else int(round(p_treat * n_periods))
            k = min(max(k, 0), n_periods)
            seq = _balanced_sequence(n_periods, k, rng)
        elif design == "blocked_random":
            seq = _blocked_sequence(n_periods, switch_every, p_treat, rng)
        else:
            raise ValueError(f"unknown design {design!r}")

        wash = _mark_washout(seq, washout)
        for region in members:
            for t, treat, is_wo in zip(periods, seq, wash):
                rows.append(
                    {
                        "region": region,
                        "period": t,
                        "treatment": int(treat),
                        "is_washout": bool(is_wo),
                    }
                )

    table = pd.DataFrame(rows)
    table = table.sort_values(["region", "period"]).reset_index(drop=True)
    adj = ring_adjacency(regions)
    table["is_spatial_buffer"] = _mark_spatial_buffer(table, adj, spatial_buffer)
    return SwitchbackAssignment(
        table=table,
        design=design,
        p_treat=p_treat,
        washout=washout,
        switch_every=switch_every,
        spatial_buffer=spatial_buffer,
        cluster_size=cluster_size,
    )


def spatial_buffer_cost(
    n_regions: int = 6,
    n_periods: int = 16,
    *,
    cluster_sizes: tuple[int, ...] = (1, 2, 3),
    spatial_buffer: int = 1,
    washout: int = 0,
    n_draws: int = 200,
    seed: int = 0,
    design: DesignName = "balanced",
) -> pd.DataFrame:
    """Share of cells that survive a spatial buffer, by cluster size.

    Bigger clusters mean fewer T/C borders and more interior, but fewer
    independent sequences to randomize. This is that tradeoff as a keep
    rate: ``keep_frac`` is the mean of ``len(analysis_table) / len(table)``.
    """
    regions = [f"r{i}" for i in range(n_regions)]
    periods = list(range(n_periods))
    n_cells = n_regions * n_periods
    rows = []
    rng = np.random.default_rng(seed)
    for k in cluster_sizes:
        kept = []
        for _ in range(n_draws):
            a = assign_switchback(
                regions,
                periods,
                design=design,
                washout=washout,
                cluster_size=int(k),
                spatial_buffer=spatial_buffer,
                seed=int(rng.integers(0, 1_000_000_000)),
            )
            kept.append(len(a.analysis_table) / len(a.table))
        arr = np.asarray(kept)
        rows.append(
            {
                "cluster_size": int(k),
                "n_clusters": int(np.ceil(n_regions / k)),
                "spatial_buffer": spatial_buffer,
                "washout": washout,
                "n_cells": n_cells,
                "keep_frac": float(arr.mean()),
                "keep_frac_sd": float(arr.std(ddof=1)) if n_draws > 1 else 0.0,
                "n_draws": n_draws,
            }
        )
    return pd.DataFrame(rows)
