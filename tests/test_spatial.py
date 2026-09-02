import numpy as np
import pytest

from sbetoolkit.interference import diagnose_spatial_spillover
from sbetoolkit.marketplace import MarketplaceConfig, MarketplaceSimulator, _leak_extra_matches
from sbetoolkit.randomization import assign_switchback, ring_adjacency


def test_cluster_size_copies_treatment_across_neighbors():
    a = assign_switchback(
        [f"r{i}" for i in range(6)],
        range(8),
        cluster_size=3,
        seed=0,
    )
    for period, g in a.table.groupby("period"):
        g = g.set_index("region")
        assert g.loc["r0", "treatment"] == g.loc["r1", "treatment"] == g.loc["r2", "treatment"]
        assert g.loc["r3", "treatment"] == g.loc["r4", "treatment"] == g.loc["r5", "treatment"]


def test_spatial_buffer_drops_border_keeps_interior():
    regions = [f"r{i}" for i in range(6)]
    a = assign_switchback(regions, range(10), cluster_size=3, spatial_buffer=1, seed=1)
    assert a.table["is_spatial_buffer"].any()
    assert not a.analysis_table["is_spatial_buffer"].any()
    # interiors of the two triples: r1 and r4 never border the other arm
    # (ring: groups 0-1-2 and 3-4-5; borders are 2-3 and 5-0).
    interiors = a.table.loc[a.table["region"].isin(["r1", "r4"])]
    assert not interiors["is_spatial_buffer"].any()


def test_ring_adjacency():
    adj = ring_adjacency(["a", "b", "c"])
    assert set(adj["a"]) == {"c", "b"}


def test_leftover_drivers_fill_neighbor_demand():
    extra = _leak_extra_matches([5, 0, 0], [0, 4, 0], 1.0)
    assert extra[1] == 4
    assert extra[0] == 0


def _cross_border_config(**kwargs) -> MarketplaceConfig:
    # Control has idle drivers; treated has unmatched demand. Leakage only
    # moves mass when those two sit on a border.
    params = dict(
        n_regions=6,
        n_periods=16,
        riders_per_cell=40,
        drivers_per_cell=30,
        p_request_control=0.50,
        p_request_treat=0.95,
        seed=2,
    )
    params.update(kwargs)
    return MarketplaceConfig(**params)


def test_leakage_biases_unbuffered_switchback():
    """Treated demand pulls leftover drivers from neighboring control."""
    sealed = MarketplaceSimulator(_cross_border_config(driver_leakage=0.0))
    leaky = MarketplaceSimulator(_cross_border_config(driver_leakage=1.0))
    truth = leaky.ground_truth(n_mc=24, seed=2)["ate"]
    sealed_ates = []
    leaky_ates = []
    for i in range(24):
        sealed_ates.append(sealed.run_switchback(seed=10 + i, cluster_size=1)["ate"])
        leaky_ates.append(leaky.run_switchback(seed=10 + i, cluster_size=1)["ate"])
    sealed_bias = abs(np.mean(sealed_ates) - truth)
    leaky_bias = abs(np.mean(leaky_ates) - truth)
    assert leaky_bias > sealed_bias + 0.01
    assert np.mean(leaky_ates) > np.mean(sealed_ates)


def test_spatial_buffer_reduces_leakage_bias():
    sim = MarketplaceSimulator(_cross_border_config(driver_leakage=1.0, seed=3))
    report = diagnose_spatial_spillover(sim, n_reps=20, seed=3, cluster_size=3)
    unbuf = abs(report.set_index("spatial_buffer").loc[0, "bias"])
    buf = abs(report.set_index("spatial_buffer").loc[1, "bias"])
    assert buf < unbuf


def test_diagnose_spatial_requires_leakage():
    sim = MarketplaceSimulator(MarketplaceConfig(driver_leakage=0.0, seed=0))
    with pytest.raises(ValueError, match="driver_leakage"):
        diagnose_spatial_spillover(sim)
