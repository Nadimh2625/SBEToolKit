import numpy as np

from sbetoolkit.power import contrast_variance, switchback_power, switchback_sample_size


def test_iid_variance_matches_two_sample_formula():
    t = np.array([1, 1, 1, 0, 0, 0])
    sigma = 2.0
    var = contrast_variance(t, sigma, rho_ar1=0.0)
    expected = sigma**2 * (1 / 3 + 1 / 3)
    assert abs(var - expected) < 1e-12


def test_ar1_inflates_variance_for_blocked_switches():
    # Long treated/control runs: positive serial correlation inflates the
    # contrast. (Alternating switches are the opposite — they pair neighbors.)
    t = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    v0 = contrast_variance(t, 1.0, 0.0)
    v1 = contrast_variance(t, 1.0, 0.6)
    assert v1 > v0


def test_power_increases_with_more_periods():
    p_small = switchback_power(0.1, 0.5, n_regions=4, n_periods=8, seed=0)
    p_big = switchback_power(0.1, 0.5, n_regions=4, n_periods=40, seed=0)
    assert p_big.power > p_small.power


def test_sample_size_meets_target():
    res = switchback_sample_size(
        0.08, 0.4, n_regions=6, power=0.8, alpha=0.05, seed=1, min_periods=6
    )
    assert res.power >= 0.8
    one_less = switchback_power(
        0.08, 0.4, n_regions=6, n_periods=max(res.n_periods - 1, 4), seed=1
    )
    # The returned horizon is a minimizer, so dropping a period should
    # usually fall under target — allow equality at the discrete boundary.
    assert one_less.power <= res.power + 1e-9


def test_washout_reduces_power():
    p0 = switchback_power(0.1, 0.4, 8, 24, washout=0, seed=2)
    p1 = switchback_power(0.1, 0.4, 8, 24, washout=2, seed=2)
    assert p1.power < p0.power
    assert p1.n_analysis_blocks < p0.n_analysis_blocks
