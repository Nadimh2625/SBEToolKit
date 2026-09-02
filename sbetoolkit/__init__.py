"""Switchback experiment toolkit for marketplace A/B tests.

The library is built around one fact: rider- or listing-level A/B tests
violate SUTVA in two-sided markets. Switchback randomization over
time-region blocks estimates the global policy effect instead of a
contaminated contrast.

Public surface
--------------
- :mod:`sbetoolkit.power` — sample size and power for switchbacks
- :mod:`sbetoolkit.randomization` — time-region switchback designs
- :mod:`sbetoolkit.cuped` — CUPED variance reduction
- :mod:`sbetoolkit.sequential` — Lan–DeMets alpha spending
- :mod:`sbetoolkit.interference` — naive A/B vs global ATE diagnostics
- :mod:`sbetoolkit.marketplace` — simulated two-sided market with known truth
- :mod:`sbetoolkit.inference` — iid vs cluster-robust vs block-level SEs
- :mod:`sbetoolkit.calibration` — Type I null check and empirical power check
"""

from sbetoolkit.calibration import (
    PowerCheckResult,
    TypeIResult,
    empirical_power_check,
    null_config,
    type_i_null_check,
)
from sbetoolkit.cuped import cuped_adjust, cuped_theta
from sbetoolkit.inference import Estimate, block_ate, clustered_ate, estimate_switchback, iid_ate
from sbetoolkit.interference import InterferenceReport, diagnose_interference, diagnose_spatial_spillover
from sbetoolkit.marketplace import MarketplaceConfig, MarketplaceSimulator, summarize_estimators
from sbetoolkit.power import PowerResult, switchback_power, switchback_sample_size
from sbetoolkit.randomization import SwitchbackAssignment, assign_switchback
from sbetoolkit.sequential import SequentialMonitor, spending_boundaries

__all__ = [
    "Estimate",
    "MarketplaceConfig",
    "MarketplaceSimulator",
    "PowerCheckResult",
    "PowerResult",
    "SwitchbackAssignment",
    "InterferenceReport",
    "SequentialMonitor",
    "TypeIResult",
    "assign_switchback",
    "block_ate",
    "clustered_ate",
    "estimate_switchback",
    "cuped_adjust",
    "cuped_theta",
    "diagnose_interference",
    "diagnose_spatial_spillover",
    "empirical_power_check",
    "iid_ate",
    "null_config",
    "spending_boundaries",
    "summarize_estimators",
    "switchback_power",
    "switchback_sample_size",
    "type_i_null_check",
]

__version__ = "0.1.0"
