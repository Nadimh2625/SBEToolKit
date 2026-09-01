"""Sequential testing with Lan–DeMets alpha spending.

Information fractions ``t_k ∈ (0, 1]`` index looks. Spending functions
map ``t`` to cumulative Type I error used by look ``k``. Boundaries are
the two-sided critical values ``c_k`` satisfying

    P(|Z_1| ≥ c_1) = α(t_1)
    P(|Z_1| < c_1, …, |Z_{k-1}| < c_{k-1}, |Z_k| ≥ c_k) = α(t_k) − α(t_{k-1})

with ``Cov(Z_i, Z_j) = √(t_i / t_j)`` (Brownian-motion increments), the
standard group-sequential model (Lan & DeMets 1983, Jennison & Turnbull).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy.stats import multivariate_normal, norm

SpendingName = Literal["obrien_fleming", "pocock", "kim_deMets"]


def spend(t: float, alpha: float, family: SpendingName, *, rho: float = 3.0) -> float:
    """Cumulative alpha spent by information fraction ``t``."""
    t = float(np.clip(t, 0.0, 1.0))
    if t == 0:
        return 0.0
    if family == "obrien_fleming":
        # Approximate OBF: 2(1 − Φ(z_{α/2} / √t))
        return float(2 * (1 - norm.cdf(norm.ppf(1 - alpha / 2) / np.sqrt(t))))
    if family == "pocock":
        return float(alpha * np.log(1 + (np.e - 1) * t))
    if family == "kim_deMets":
        # Power family α t^ρ. ρ=3 ≈ OBF, ρ=1 ≈ Pocock.
        return float(alpha * (t**rho))
    raise ValueError(f"unknown spending family {family!r}")


def _bm_cov(info: np.ndarray) -> np.ndarray:
    t = np.asarray(info, dtype=float)
    return np.sqrt(np.minimum.outer(t, t) / np.maximum.outer(t, t))


def _crossing_prob(bounds: np.ndarray, cov: np.ndarray) -> float:
    """P(some |Z_k| ≥ c_k) under N(0, cov) via inclusion of the staying set."""
    k = len(bounds)
    stay_lo = -bounds
    stay_hi = bounds
    stay = float(
        multivariate_normal.cdf(
            stay_hi,
            mean=np.zeros(k),
            cov=cov,
            lower_limit=stay_lo,
        )
    )
    return 1.0 - stay


def _increment_prob(prev_c: np.ndarray, c_k: float, cov: np.ndarray) -> float:
    """P(stay through k-1, cross at k)."""
    k = len(prev_c) + 1
    stay_prev = 1.0
    if len(prev_c):
        stay_prev = float(
            multivariate_normal.cdf(
                prev_c,
                mean=np.zeros(k - 1),
                cov=cov[: k - 1, : k - 1],
                lower_limit=-prev_c,
            )
        )
    all_c = np.append(prev_c, c_k)
    stay_all = float(
        multivariate_normal.cdf(
            all_c,
            mean=np.zeros(k),
            cov=cov[:k, :k],
            lower_limit=-all_c,
        )
    )
    return max(0.0, stay_prev - stay_all)


def spending_boundaries(
    info_fractions: list[float] | np.ndarray,
    *,
    alpha: float = 0.05,
    family: SpendingName = "obrien_fleming",
    rho: float = 3.0,
    two_sided: bool = True,
) -> np.ndarray:
    """Critical values ``c_k`` for two-sided (or one-sided) looks.

    ``info_fractions`` must be strictly increasing and end at 1.0
    (the final look spends any remainder so total Type I error is α).
    """
    if not two_sided:
        raise NotImplementedError("one-sided boundaries are not implemented")
    t = np.asarray(info_fractions, dtype=float)
    if t.ndim != 1 or len(t) == 0:
        raise ValueError("info_fractions must be a 1-d non-empty array")
    if np.any(np.diff(t) <= 0) or t[0] <= 0:
        raise ValueError("info_fractions must be strictly increasing and positive")
    if t[-1] > 1 + 1e-12:
        raise ValueError("information fractions cannot exceed 1")
    t = np.clip(t, None, 1.0)
    # Force the last look to spend remaining alpha exactly.
    spent = np.array([spend(x, alpha, family, rho=rho) for x in t])
    spent = np.clip(spent, 0.0, alpha)
    spent[-1] = alpha
    spent = np.maximum.accumulate(spent)
    increments = np.diff(spent, prepend=0.0)
    cov = _bm_cov(t)

    bounds = np.zeros(len(t))
    for k, da in enumerate(increments):
        if da <= 1e-15:
            bounds[k] = np.inf
            continue
        if k == 0:
            bounds[k] = float(norm.ppf(1 - da / 2))
            continue

        prev = bounds[:k]
        lo, hi = 0.0, 12.0
        # If even c=0 overspends, clamp to 0 (pathological tiny increment).
        p_lo = _increment_prob(prev, hi, cov)
        if p_lo > da:
            # hi is too large (small crossing). Search is: larger c → smaller p.
            pass
        for _ in range(48):
            mid = 0.5 * (lo + hi)
            p = _increment_prob(prev, mid, cov)
            if p > da:
                lo = mid
            else:
                hi = mid
        bounds[k] = 0.5 * (lo + hi)
    return bounds


@dataclass
class SequentialLook:
    look: int
    info_fraction: float
    z: float
    bound: float
    spent_alpha: float
    reject: bool


@dataclass
class SequentialMonitor:
    """Streaming group-sequential test.

    Call :meth:`update` with the current z-statistic and information
    fraction at each look. ``reject`` becomes True at the first crossing
    and stays True (alpha is not reused).
    """

    alpha: float = 0.05
    family: SpendingName = "obrien_fleming"
    planned_info: list[float] | None = None
    rho: float = 3.0
    looks: list[SequentialLook] = field(default_factory=list)
    reject: bool = False
    _planned_bounds: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.planned_info is not None:
            self._planned_bounds = spending_boundaries(
                self.planned_info, alpha=self.alpha, family=self.family, rho=self.rho
            )

    def update(self, z: float, info_fraction: float) -> SequentialLook:
        if self.planned_info is not None:
            k = len(self.looks)
            if k >= len(self.planned_info):
                raise RuntimeError("no remaining planned looks")
            expected = self.planned_info[k]
            if abs(expected - info_fraction) > 1e-6:
                raise ValueError(
                    f"look {k} expected information {expected}, got {info_fraction}"
                )
            bound = float(self._planned_bounds[k])
            spent = spend(info_fraction, self.alpha, self.family, rho=self.rho)
            if k == len(self.planned_info) - 1:
                spent = self.alpha
        else:
            # Unplanned looks: recompute bounds on the observed grid,
            # treating the current look as interim (not forced to spend α
            # unless t=1). This is the Lan–DeMets flexible-calendar method.
            grid = [lk.info_fraction for lk in self.looks] + [info_fraction]
            if info_fraction >= 1:
                grid[-1] = 1.0
            bounds = spending_boundaries(
                grid, alpha=self.alpha, family=self.family, rho=self.rho
            )
            bound = float(bounds[-1])
            spent = spend(min(info_fraction, 1.0), self.alpha, self.family, rho=self.rho)
            if info_fraction >= 1:
                spent = self.alpha

        crossed = abs(z) >= bound
        if crossed:
            self.reject = True
        look = SequentialLook(
            look=len(self.looks) + 1,
            info_fraction=float(info_fraction),
            z=float(z),
            bound=bound,
            spent_alpha=float(spent),
            reject=crossed or self.reject,
        )
        self.looks.append(look)
        return look
