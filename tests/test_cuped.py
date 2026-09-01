import numpy as np

from sbetoolkit.cuped import cuped_adjust, cuped_theta, difference_in_means


def test_theta_recovers_linear_coefficient():
    rng = np.random.default_rng(0)
    x = rng.normal(size=2000)
    y = 2.5 * x + rng.normal(scale=0.1, size=2000)
    fit = cuped_theta(y, x)
    assert abs(fit.theta - 2.5) < 0.05
    assert fit.variance_reduction > 0.95


def test_cuped_preserves_ate():
    rng = np.random.default_rng(1)
    n = 1500
    t = rng.integers(0, 2, size=n)
    x = rng.normal(size=n)
    y = 0.3 * t + 1.4 * x + rng.normal(scale=0.5, size=n)
    raw = difference_in_means(y, t)
    cv = difference_in_means(y, t, x=x)
    assert abs(raw["ate"] - cv["ate"]) < 0.08
    assert cv["se"] < raw["se"]


def test_adjust_is_mean_preserving_in_x():
    rng = np.random.default_rng(2)
    x = rng.normal(size=400)
    y = x + rng.normal(size=400)
    ycv = cuped_adjust(y, x)
    # CUPED subtracts a mean-zero term, so E[Ycv] ≈ E[Y]
    assert abs(ycv.mean() - y.mean()) < 1e-10
