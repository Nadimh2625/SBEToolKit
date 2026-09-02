import numpy as np

from sbetoolkit.inference import Estimate, block_ate, clustered_ate, iid_ate


def _clustered_dgp(rng, n_blocks=80, m=40, icc_sd=0.2):
    t = rng.integers(0, 2, size=n_blocks)
    u = rng.normal(0, icc_sd, size=n_blocks)
    y = (u[:, None] + rng.normal(0, 0.5, size=(n_blocks, m))).ravel()
    treat = np.repeat(t, m)
    cluster = np.repeat(np.arange(n_blocks), m)
    block_y = y.reshape(n_blocks, m).mean(axis=1)
    return y, treat, cluster, block_y, t


def test_cluster_se_larger_than_iid_when_icc_is_high():
    rng = np.random.default_rng(0)
    y, t, g, _, _ = _clustered_dgp(rng)
    iid = iid_ate(y, t)
    cl = clustered_ate(y, t, g)
    assert cl.n_clusters == 80
    assert iid.n_obs == 80 * 40
    assert cl.se > 1.5 * iid.se


def test_block_and_cluster_agree_on_constant_within_block_treatment():
    rng = np.random.default_rng(1)
    y, t, g, block_y, block_t = _clustered_dgp(rng, n_blocks=60, m=25)
    cl = clustered_ate(y, t, g)
    blk = block_ate(block_y, block_t)
    # Cluster OLS is size-weighted; equal block sizes ⇒ same ATE.
    assert abs(cl.ate - blk.ate) < 1e-10
    assert abs(cl.se / blk.se - 1) < 0.15


def test_iid_rejects_null_too_often_cluster_does_not():
    rng = np.random.default_rng(2)
    iid_r = []
    cl_r = []
    blk_r = []
    for _ in range(250):
        y, t, g, block_y, block_t = _clustered_dgp(rng, n_blocks=100, m=50, icc_sd=0.25)
        iid_r.append(iid_ate(y, t).reject())
        cl_r.append(clustered_ate(y, t, g).reject())
        blk_r.append(block_ate(block_y, block_t).reject())
    assert np.mean(iid_r) > 0.12
    assert 0.02 <= np.mean(cl_r) <= 0.10
    assert 0.02 <= np.mean(blk_r) <= 0.10


def test_interval_covers_its_own_point_estimate():
    est = Estimate(ate=0.2, se=0.05, z=4.0, pvalue=0.0, n_obs=100, n_clusters=100, method="iid")
    lo, hi = est.interval()
    assert lo < 0.2 < hi
    assert abs((hi - lo) / 2 - 1.95996398454 * 0.05) < 1e-9
    assert est.covers(0.2)
    assert not est.covers(0.4)
