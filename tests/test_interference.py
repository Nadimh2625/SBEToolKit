from sbetoolkit.interference import diagnose_interference
from sbetoolkit.marketplace import MarketplaceConfig, MarketplaceSimulator


def test_report_frame_rows():
    sim = MarketplaceSimulator(
        MarketplaceConfig(n_regions=3, n_periods=4, riders_per_cell=20, drivers_per_cell=8, seed=0)
    )
    report = diagnose_interference(sim, n_mc=5, seed=0)
    frame = report.to_frame()
    assert len(frame) == 10
    assert report.bias == report.naive_ate - report.global_ate
