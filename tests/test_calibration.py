from sbetoolkit.calibration import empirical_power_check, null_config, type_i_null_check
from sbetoolkit.marketplace import MarketplaceConfig
import pytest


def test_null_config_is_actually_null():
    cfg = null_config()
    assert cfg.p_request_treat == cfg.p_request_control
    assert cfg.extra_drivers_if_treated == 0


def test_type_i_rejects_nonzero_effect_config():
    with pytest.raises(ValueError, match="p_request"):
        type_i_null_check(
            n_reps=1,
            config=MarketplaceConfig(p_request_control=0.5, p_request_treat=0.8),
        )


def test_marketplace_null_block_rate_near_nominal():
    """Short marketplace null: block-level test should not explode."""
    cfg = null_config(
        n_regions=6,
        n_periods=16,
        riders_per_cell=24,
        drivers_per_cell=12,
        cell_intercept_sd=0.15,
        seed=4,
    )
    result = type_i_null_check(n_reps=80, seed=4, config=cfg)
    assert result.iid_rate > result.block_rate
    assert result.iid_rate > result.cluster_rate
    assert result.block_rate < 0.15
    assert result.cluster_rate < 0.15


def test_empirical_power_near_target_on_ar1_dgp():
    """Sized for 80%; formula-SE test should not collapse to ~50%."""
    result = empirical_power_check(
        0.08,
        0.15,
        n_regions=6,
        rho_ar1=0.45,
        n_reps=120,
        seed=13,
        target_power=0.8,
    )
    assert result.predicted_power >= 0.8
    assert 0.65 <= result.empirical_power <= 0.95
    assert 0.7 <= result.se_ratio <= 1.3
