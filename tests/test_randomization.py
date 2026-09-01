from sbetoolkit.power import contrast_variance, switchback_power, switchback_sample_size
from sbetoolkit.randomization import assign_switchback

import numpy as np
import pytest


def test_balanced_assignment_counts():
    a = assign_switchback(["a", "b"], range(10), design="balanced", seed=1)
    for _, g in a.table.groupby("region"):
        assert g["treatment"].sum() == 5


def test_independent_assignment_shape():
    a = assign_switchback(["r"], range(20), design="independent", p_treat=0.4, seed=2)
    assert len(a.table) == 20
    assert set(a.table["treatment"].unique()) <= {0, 1}


def test_blocked_random_window_constant():
    a = assign_switchback(
        ["r"], range(12), design="blocked_random", switch_every=3, seed=3
    )
    t = a.table["treatment"].to_numpy()
    for i in range(0, 12, 3):
        assert len(set(t[i : i + 3])) == 1


def test_washout_flags_after_switch():
    a = assign_switchback(["r"], range(8), design="balanced", washout=1, seed=4)
    t = a.table["treatment"].to_numpy()
    wo = a.table["is_washout"].to_numpy()
    assert wo[0]  # burn-in
    for i in range(1, 8):
        if t[i] != t[i - 1]:
            assert wo[i]


def test_analysis_table_drops_washout():
    a = assign_switchback(["r"], range(10), washout=2, seed=0)
    assert len(a.analysis_table) < len(a.table)
    assert not a.analysis_table["is_washout"].any()


def test_invalid_p_treat():
    with pytest.raises(ValueError):
        assign_switchback(["r"], range(4), p_treat=1.0)
