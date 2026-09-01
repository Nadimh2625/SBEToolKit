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
