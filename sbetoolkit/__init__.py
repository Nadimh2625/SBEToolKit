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
"""

from sbetoolkit.cuped import cuped_adjust, cuped_theta
from sbetoolkit.interference import InterferenceReport, diagnose_interference
from sbetoolkit.marketplace import MarketplaceConfig, MarketplaceSimulator
from sbetoolkit.power import PowerResult, switchback_power, switchback_sample_size
from sbetoolkit.randomization import SwitchbackAssignment, assign_switchback
from sbetoolkit.sequential import SequentialMonitor, spending_boundaries

__all__ = [
    "MarketplaceConfig",
    "MarketplaceSimulator",
    "PowerResult",
    "SwitchbackAssignment",
    "InterferenceReport",
    "SequentialMonitor",
    "assign_switchback",
    "cuped_adjust",
    "cuped_theta",
    "diagnose_interference",
    "spending_boundaries",
    "switchback_power",
    "switchback_sample_size",
]

__version__ = "0.1.0"
