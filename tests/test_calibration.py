import numpy as np
import pandas as pd

from mc_quadrants.calibration import (
    _ledoit_wolf_alpha,
    calibrate_quadrant_model,
    estimate_regime_moments,
)
from mc_quadrants.regimes import (
    Regime,
    classify_quadrants,
)


def _sample_data():
    rng = np.random.default_rng(5)
    dates = pd.date_range("2010-01-31", periods=120, freq="ME")
    macro = pd.DataFrame(
        {
            "growth": rng.normal(2.0, 1.5, len(dates)),
            "inflation": rng.normal(3.0, 1.0, len(dates)),
        },
        index=dates,
    )
    returns = pd.DataFrame(
        {
            "Stocks": rng.normal(0.01, 0.04, len(dates)),
            "Bonds": rng.normal(0.002, 0.02, len(dates)),
        },
        index=dates,
    )
    return returns, macro


def test_ledoit_wolf_alpha_is_between_zero_and_one():
    rng = np.random.default_rng(1)
    observations = rng.normal(0.0, 1.0, (200, 3))
    sample = np.cov(observations, rowvar=False)
    target = np.eye(3)

    alpha = _ledoit_wolf_alpha(observations, sample, target)

    assert 0.0 <= alpha <= 1.0


def test_ledoit_wolf_alpha_grows_with_sample_noise():
    rng = np.random.default_rng(1)
    target = np.eye(2)
    small_sample = rng.normal(0.0, 4.0, (10, 2))
    large_sample = rng.normal(0.0, 4.0, (1000, 2))

    alpha_small = _ledoit_wolf_alpha(small_sample, np.cov(small_sample, rowvar=False), target)
    alpha_large = _ledoit_wolf_alpha(large_sample, np.cov(large_sample, rowvar=False), target)

    assert alpha_small > alpha_large


def test_ledoit_wolf_alpha_shrinks_toward_matching_target():
    rng = np.random.default_rng(1)
    observations = rng.normal(0.0, 1.0, (500, 2))
    sample = np.cov(observations, rowvar=False)

    alpha_exact = _ledoit_wolf_alpha(observations, sample, sample)
    alpha_far = _ledoit_wolf_alpha(observations, sample, 3.0 * sample)

    assert alpha_exact > alpha_far


def test_auto_shrinkage_produces_psd_moments():
    returns, macro = _sample_data()
    regimes = classify_quadrants(macro)

    moments = estimate_regime_moments(returns, regimes, shrinkage=None)

    for state, state_moments in moments.items():
        values = state_moments.covariance.to_numpy(dtype=float)
        assert np.allclose(values, values.T)
        assert (np.linalg.eigvalsh(values) >= 0).all()
        assert state_moments.correlation.shape == state_moments.covariance.shape


def test_auto_shrinkage_blends_sparse_regimes_toward_global():
    returns, macro = _sample_data()
    regimes = classify_quadrants(macro)
    global_covariance = returns.cov()
    moments = estimate_regime_moments(
        returns,
        regimes,
        min_observations=200,
        shrinkage=None,
    )

    for state in regimes.unique():
        local_covariance = returns.loc[regimes == state].cov()
        blended = moments[state].covariance
        distance_to_global = np.linalg.norm(blended.to_numpy() - global_covariance.to_numpy())
        distance_to_local = np.linalg.norm(blended.to_numpy() - local_covariance.to_numpy())
        assert distance_to_global <= distance_to_local


def test_calibrate_model_records_threshold_window_in_metadata():
    returns, macro = _sample_data()

    model = calibrate_quadrant_model(
        returns,
        macro,
        growth_threshold="median",
        inflation_threshold="median",
        threshold_window=24,
    )

    assert model.metadata["threshold_window"] == 24
    assert "sojourn_durations" in model.metadata
    assert set(model.metadata["sojourn_durations"]) == set(Regime)


def test_probabilistic_calibration_uses_explicit_duration_hsmm():
    returns, macro = _sample_data()
    model = calibrate_quadrant_model(
        returns,
        macro,
        probabilistic_regimes=True,
        regime_smoothing_window=3,
        regime_hysteresis=0.15,
        regime_confirmation_periods=2,
    )
    assert np.allclose(model.transition_matrix.sum(axis=1), 1.0)
    assert model.metadata["transition_estimator"] == "hsmm_forward_backward_joint_posteriors"
    assert model.metadata["duration_model_kind"] == "hidden_semi_markov_explicit_duration"
    assert set(model.metadata["duration_hazards"]) == set(Regime)
    assert all(
        np.allclose(hazards[:4], 0.0)
        for hazards in model.metadata["duration_hazards"].values()
    )
    exit_matrix = model.metadata["hsmm_exit_transition_matrix"]
    assert np.allclose(np.diag(exit_matrix), 0.0)
    assert np.allclose(exit_matrix.sum(axis=1), 1.0)
    assert np.isfinite(model.metadata["hsmm_log_likelihood"])
    probabilities = model.metadata["historical_regime_probabilities"].dropna()
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_joint_macro_uses_stabilized_bvar_and_structural_asset_profiles():
    returns, macro = _sample_data()
    macro = macro.copy()
    macro["interest_rate"] = np.linspace(1.0, 5.0, len(macro))
    model = calibrate_quadrant_model(
        returns.rename(columns={"Stocks": "SPY", "Bonds": "IEF"}),
        macro,
        joint_macro=True,
        structural_returns=True,
        macro_model="bvar_ensemble",
    )

    dynamics = model.metadata["macro_dynamics"]
    assert dynamics["macro_model"] == "bvar_ensemble"
    assert dynamics["structural_returns"] is True
    assert np.asarray(dynamics["var_coefficient_std"]).shape == (3, 3)
    assert dynamics["asset_profiles"]["SPY"]["asset_class"] == "equity"
    assert dynamics["asset_profiles"]["IEF"]["asset_class"] == "government_bond"
    rate_index = dynamics["columns"].index("interest_rate")
    assert dynamics["return_beta_priors"][rate_index][1] < 0
