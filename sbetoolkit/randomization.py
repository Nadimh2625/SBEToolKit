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

    @property
    def analysis_table(self) -> pd.DataFrame:
        """Blocks that are not in a washout window.

        This is the only definition of the estimation sample. Downstream
        code should join outcomes onto this frame (see :meth:`for_analysis`)
        instead of filtering ``table`` themselves.
        """
        return self.table.loc[~self.table["is_washout"]].copy()

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


def assign_switchback(
    regions: list[str] | np.ndarray | pd.Index,
    periods: list[int] | np.ndarray | range,
    *,
    p_treat: float = 0.5,
    design: DesignName = "balanced",
    switch_every: int = 1,
    washout: int = 0,
    n_treated: int | None = None,
    seed: int | None = None,
) -> SwitchbackAssignment:
    """Assign treatment to time-region blocks.

    Parameters
    ----------
    regions, periods
        Cartesian product of these forms the experimental grid.
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
    """
    if not 0 < p_treat < 1:
        raise ValueError("p_treat must be in (0, 1)")
    if washout < 0:
        raise ValueError("washout must be >= 0")
    if switch_every < 1:
        raise ValueError("switch_every must be >= 1")

    regions = list(regions)
    periods = list(periods)
    n_periods = len(periods)
    if n_periods == 0 or not regions:
        raise ValueError("regions and periods must be non-empty")

    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    for region in regions:
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
    return SwitchbackAssignment(
        table=table,
        design=design,
        p_treat=p_treat,
        washout=washout,
        switch_every=switch_every,
    )
