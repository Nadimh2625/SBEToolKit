import numpy as np
from scipy.stats import norm

from sbetoolkit.sequential import (
    SequentialMonitor,
    _bm_cov,
    _crossing_prob,
    spend,
    spending_boundaries,
)


def test_final_look_spends_full_alpha():
    t = [0.25, 0.5, 0.75, 1.0]
    for family in ("obrien_fleming", "pocock", "kim_deMets"):
        b = spending_boundaries(t, alpha=0.05, family=family)
        assert np.all(np.isfinite(b))
        assert b[-1] > 0


def test_obf_is_conservative_early():
    t = [0.2, 0.5, 1.0]
    obf = spending_boundaries(t, alpha=0.05, family="obrien_fleming")
    poc = spending_boundaries(t, alpha=0.05, family="pocock")
    assert obf[0] > poc[0]
    assert obf[0] > norm.ppf(1 - 0.025)  # first look harder than a fixed test


def test_boundaries_spend_nominal_alpha():
    t = [0.25, 0.5, 0.75, 1.0]
    for family in ("obrien_fleming", "pocock"):
        b = spending_boundaries(t, alpha=0.05, family=family)
        cov = _bm_cov(np.asarray(t))
        assert abs(_crossing_prob(b, cov) - 0.05) < 5e-4


def test_first_boundary_matches_spent_alpha():
    t = [0.3, 1.0]
    alpha = 0.05
    b = spending_boundaries(t, alpha=alpha, family="pocock")
    spent0 = spend(0.3, alpha, "pocock")
    expected = norm.ppf(1 - spent0 / 2)
    assert abs(b[0] - expected) < 1e-8


def test_monitor_rejects_on_large_z():
    mon = SequentialMonitor(alpha=0.05, family="obrien_fleming", planned_info=[0.5, 1.0])
    look = mon.update(8.0, 0.5)
    assert look.reject
    assert mon.reject


def test_monitor_does_not_reject_on_nullish_z():
    mon = SequentialMonitor(alpha=0.05, family="obrien_fleming", planned_info=[0.5, 1.0])
    look = mon.update(0.1, 0.5)
    assert not look.reject
    look2 = mon.update(0.2, 1.0)
    assert not look2.reject


def test_flexible_calendar_last_look():
    mon = SequentialMonitor(alpha=0.05, family="pocock")
    mon.update(0.0, 0.4)
    last = mon.update(0.1, 1.0)
    assert np.isfinite(last.bound)
    assert abs(last.spent_alpha - 0.05) < 1e-12
