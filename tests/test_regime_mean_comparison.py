import numpy as np
import pandas as pd

from mc_quadrants.calibration import estimate_weighted_regime_moments
from scripts.compare_regime_means import (
    mean_variants,
    paired_stats,
    score_variant,
    shift_paths,
    training_slice,
)


def test_mean_variants_match_production_soft_membership_estimation():
    rng = np.random.default_rng(12)
    dates = pd.date_range('2000-01-31', periods=70, freq='ME')
    returns = pd.DataFrame(rng.normal(0, .04, (70, 4)), index=dates)
    probabilities = pd.DataFrame(rng.dirichlet(np.ones(4), size=70), index=dates, columns=list('abcd'))
    variants = mean_variants(returns.to_numpy(), probabilities.to_numpy())
    for name, strength in [('current24', 24), ('strong96', 96)]:
        fitted = estimate_weighted_regime_moments(returns, probabilities, list('abcd'), mean_prior_strength=strength)
        np.testing.assert_allclose(variants[name], [fitted[s].mean for s in 'abcd'], atol=1e-15)
    assert np.all(np.abs(variants['strong96'] - variants['unconditional']) <=
                  np.abs(variants['current24'] - variants['unconditional']) + 1e-15)


def test_future_data_cannot_enter_origin_training_slice():
    dates = pd.date_range('2000-01-31', periods=80, freq='ME')
    data = pd.DataFrame({'a': np.arange(80.)}, index=dates)
    macro, returns = training_slice(data, data, 60)
    changed = data.copy()
    changed.iloc[60:] = -1e9
    new_macro, new_returns = training_slice(changed, changed, 60)
    pd.testing.assert_frame_equal(macro, new_macro)
    pd.testing.assert_frame_equal(returns, new_returns)
    assert returns.index[-1] < dates[60]


def test_forecast_integrates_first_regime_exactly_and_compounds_correct_targets():
    means = np.array([[.01] * 4, [-.02] * 4])
    # Deliberately unrepresentative draws must not contaminate the exact 1m mean.
    codes = np.zeros((12, 8), dtype=int)
    samples = np.full((12, 8, 4), .7)
    actual = np.full((3, 4), np.log1p(.01))
    result = score_variant(samples, codes, means, np.array([.25, .75]), actual,
                           list('ABCD'), np.ones(4))
    assert np.isclose(result['forecast_A_1m'], -.0125)
    assert np.isclose(result['forecast_A_3m'], -.0125 + .02)
    assert np.isclose(result['actual_A_3m'], 3 * np.log1p(.01))
    assert np.isclose(result['portfolio_actual_3m'], 1.01**3 - 1)
    assert 'mse_12m' not in result
    changed = actual.copy()
    changed[1:] = -.9
    new = score_variant(samples, codes, means, np.array([.25, .75]), changed,
                        list('ABCD'), np.ones(4))
    assert new['mse_1m'] == result['mse_1m']
    assert new['mse_3m'] != result['mse_3m']


def test_mean_shift_preserves_residuals_along_the_same_state_path():
    rng = np.random.default_rng(13)
    codes = rng.integers(4, size=(12, 200))
    old, new = rng.normal(size=(4, 4)), rng.normal(size=(4, 4))
    innovations = rng.normal(size=(12, 200, 4))
    shifted = shift_paths(old[codes] + innovations, codes, old, new)
    np.testing.assert_allclose(shifted - new[codes], innovations, atol=1e-14)


def test_paired_bootstrap_uses_matching_dates_and_constant_difference():
    baseline = pd.Series(np.linspace(1, 2, 50), index=np.arange(50))
    candidate = (baseline + .25).iloc[::-1]
    result = paired_stats(candidate, baseline, family=2)
    assert result['n'] == 50
    np.testing.assert_allclose(result['adjusted_ci95'], [.25, .25])
