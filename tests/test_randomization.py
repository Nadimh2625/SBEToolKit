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


def test_analysis_table_drops_every_post_switch_row():
    washout = 2
    a = assign_switchback(["east", "west"], range(16), washout=washout, seed=5)
    kept = set(zip(a.analysis_table["region"], a.analysis_table["period"]))
    for region, g in a.table.groupby("region"):
        g = g.sort_values("period")
        treat = g["treatment"].to_numpy()
        periods = g["period"].to_numpy()
        assert (region, periods[0]) not in kept  # burn-in
        for i in range(1, len(treat)):
            if treat[i] != treat[i - 1]:
                for k in range(washout):
                    if i + k < len(treat):
                        assert (region, periods[i + k]) not in kept
        for period, is_wo in zip(periods, g["is_washout"]):
            if is_wo:
                assert (region, period) not in kept
            else:
                assert (region, period) in kept


def test_analysis_mask_matches_analysis_table():
    a = assign_switchback(["r"], range(12), washout=2, seed=6)
    mask = a.analysis_mask()
    assert mask.sum() == len(a.analysis_table)
    kept_periods = set(a.analysis_table["period"])
    for period, keep in zip(a.table["period"], mask):
        assert keep == (period in kept_periods)


def test_for_analysis_ignores_rows_missing_from_analysis_table():
    a = assign_switchback(["r"], range(8), washout=1, seed=7)
    outcomes = a.table[["region", "period", "treatment"]].copy()
    outcomes["match_rate"] = 0.1
    sample = a.for_analysis(outcomes)
    sample_keys = set(zip(sample["region"], sample["period"]))
    assert sample_keys == set(zip(a.analysis_table["region"], a.analysis_table["period"]))


def test_poisoned_washout_does_not_move_estimate():
    """Garbage in washout periods must not change the ATE.

    If an estimator reads ``table`` instead of ``analysis_table``, this
    fails: the poisoned rows pull the contrast.
    """
    from sbetoolkit.inference import block_ate, estimate_switchback

    a = assign_switchback(["r0", "r1"], range(20), washout=2, seed=8)
    rng = np.random.default_rng(8)
    outcomes = a.table[["region", "period", "treatment", "is_washout"]].copy()
    outcomes["match_rate"] = np.where(
        outcomes["treatment"] == 1,
        0.6 + rng.normal(0, 0.02, len(outcomes)),
        0.5 + rng.normal(0, 0.02, len(outcomes)),
    )
    clean = estimate_switchback(a, outcomes)
    poisoned = outcomes.copy()
    poisoned.loc[poisoned["is_washout"], "match_rate"] = 1e6
    after = estimate_switchback(a, poisoned)
    assert abs(clean.ate - after.ate) < 1e-12
    assert abs(clean.se - after.se) < 1e-12
    naive = block_ate(poisoned["match_rate"].to_numpy(), poisoned["treatment"].to_numpy())
    assert abs(naive.ate - clean.ate) > 100


def test_invalid_p_treat():
    with pytest.raises(ValueError):
        assign_switchback(["r"], range(4), p_treat=1.0)
