import numpy as np

from sbetoolkit.interference import diagnose_interference
from sbetoolkit.marketplace import MarketplaceConfig, MarketplaceSimulator


def test_ground_truth_near_zero_when_supply_binds():
    sim = MarketplaceSimulator(
        MarketplaceConfig(
            n_regions=4,
            n_periods=6,
            riders_per_cell=40,
            drivers_per_cell=12,
            p_request_control=0.8,
            p_request_treat=0.95,
            seed=0,
        )
    )
    truth = sim.ground_truth(n_mc=20, seed=0)
    # Saturated market: extra demand does not create extra matches.
    assert abs(truth["ate"]) < 0.03


def test_naive_overstates_when_supply_binds():
    sim = MarketplaceSimulator(
        MarketplaceConfig(
            n_regions=6,
            n_periods=10,
            riders_per_cell=36,
            drivers_per_cell=14,
            p_request_control=0.7,
            p_request_treat=0.95,
            seed=1,
        )
    )
    report = diagnose_interference(sim, n_mc=12, seed=1)
    assert report.naive_ate > report.global_ate + 0.05
    assert report.control_crowding_out > 0
    assert "overstates" in report.direction


def test_switchback_closer_to_truth_than_naive():
    sim = MarketplaceSimulator(
        MarketplaceConfig(
            n_regions=6,
            n_periods=16,
            riders_per_cell=32,
            drivers_per_cell=14,
            p_request_control=0.7,
            p_request_treat=0.9,
            seed=2,
        )
    )
    df = sim.compare_estimators(n_reps=18, seed=2)
    naive_err = np.abs(df["naive_bias"]).mean()
    sw_err = np.abs(df["switchback_bias"]).mean()
    assert sw_err < naive_err / 2


def test_naive_understates_with_supply_spillover():
    """If treatment adds drivers to the whole cell, a rider split still
    shares those drivers with control → naive ATE shrinks toward zero
    relative to the global (cell-level) policy effect.

    We approximate that by giving extra drivers only in the all-treat
    world (``extra_drivers_if_treated``) while naive A/B never flags a
    cell as treated. Then global ATE is large and naive stays near the
    crowding-out contrast.
    """
    crowded = MarketplaceSimulator(
        MarketplaceConfig(
            n_regions=4,
            n_periods=8,
            riders_per_cell=40,
            drivers_per_cell=12,
            extra_drivers_if_treated=0,
            p_request_control=0.80,
            p_request_treat=0.80,
            seed=3,
        )
    )
    supply = MarketplaceSimulator(
        MarketplaceConfig(
            n_regions=4,
            n_periods=8,
            riders_per_cell=40,
            drivers_per_cell=12,
            extra_drivers_if_treated=16,
            p_request_control=0.80,
            p_request_treat=0.80,
            seed=3,
        )
    )
    r0 = diagnose_interference(crowded, n_mc=10, seed=3)
    r1 = diagnose_interference(supply, n_mc=10, seed=3)
    assert r1.global_ate > r0.global_ate + 0.05
    # Rider-level A/B cannot apply a cell-level driver incentive.
    assert r1.naive_ate < r1.global_ate
    assert "understates" in r1.direction


def test_marketplace_poisoned_washout_does_not_move_ate():
    from sbetoolkit.inference import block_ate, estimate_switchback

    sim = MarketplaceSimulator(
        MarketplaceConfig(
            n_regions=4,
            n_periods=20,
            riders_per_cell=16,
            drivers_per_cell=8,
            p_request_control=0.6,
            p_request_treat=0.75,
            seed=9,
        )
    )
    run = sim.run_switchback(seed=9, washout=2, return_riders=True)
    assignment = run["assignment"]
    assert len(run["analysis"]) == len(assignment.analysis_table)
    assert len(run["blocks"]) == len(assignment.table)

    poisoned = run["blocks"].copy()
    wash_keys = assignment.table.loc[
        assignment.table["is_washout"], ["region", "period"]
    ]
    hit = (
        poisoned.merge(wash_keys.assign(_w=True), on=["region", "period"], how="left")["_w"]
        .fillna(False)
        .to_numpy()
    )
    assert hit.any()
    poisoned.loc[hit, "match_rate"] = 1e6

    after = estimate_switchback(assignment, poisoned)
    assert abs(after.ate - run["ate"]) < 1e-12
    assert abs(after.se - run["se"]) < 1e-12
    naive = block_ate(
        poisoned["match_rate"].to_numpy(),
        poisoned["treatment"].to_numpy(),
    )
    assert abs(naive.ate - run["ate"]) > 100

    # Ride-level arrays are already restricted to analysis_table blocks.
    analysis_ids = set(run["analysis"]["block_id"])
    assert set(run["rider_block"]).issubset(analysis_ids)
