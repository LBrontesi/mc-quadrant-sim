from copy import deepcopy
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from mc_quadrants.calibration import calibrate_quadrant_model
from mc_quadrants.dependence import DEFAULT_DEPENDENCE, fit_dependence
from mc_quadrants.mnts import fit_mnts_parameters
from mc_quadrants.model_selection import select_calibration_hyperparameters
from mc_quadrants.pipeline import run_scenario
from mc_quadrants.types import RegimeMoments
from mc_quadrants.uncertainty import anchor_parameter_models
from mc_quadrants.validation import WalkForwardResult


def panel(n=120):
    rng = np.random.default_rng(33)
    dates = pd.date_range("2000-01-31", periods=n, freq="ME")
    returns = pd.DataFrame(rng.normal(.003, .03, (n, 2)), index=dates, columns=["A", "B"])
    macro = pd.DataFrame(rng.normal(size=(n, 3)), index=dates,
                         columns=["growth", "inflation", "interest_rate"])
    return returns, macro


def moments_for(returns):
    return RegimeMoments(returns.mean(), returns.cov(), returns.corr(), len(returns))


def test_dependence_fit_is_constrained_and_repeatable():
    returns, _ = panel(200)
    regimes = pd.Series("state", index=returns.index)
    moments = {"state": moments_for(returns)}
    fitted = fit_dependence(returns, regimes, moments)
    assert fitted == fit_dependence(returns, regimes, moments)
    assert fitted["status"] == "fitted"
    assert all(fitted[key] >= 0 for key in DEFAULT_DEPENDENCE)
    assert fitted["garch_alpha"] + fitted["garch_beta"] < .995
    assert sum(fitted[key] for key in ("dcc_alpha", "dcc_beta", "dcc_asymmetry")) < .995
    assert np.isfinite([fitted["garch_objective"], fitted["dcc_objective"]]).all()
    assert fitted["garch_objective"] <= fitted["garch_default_objective"]
    assert fitted["dcc_objective"] <= fitted["dcc_default_objective"]
    assert np.min(fitted["next_variance"]) > 0
    assert np.linalg.eigvalsh(fitted["next_gaussian_correlation"]).min() > 0


def test_short_history_dependence_fallback_is_explicit():
    returns, _ = panel(24)
    fitted = fit_dependence(returns, pd.Series("state", index=returns.index),
                            {"state": moments_for(returns)})
    assert fitted["status"] == "insufficient_history"
    assert {key: fitted[key] for key in DEFAULT_DEPENDENCE} == DEFAULT_DEPENDENCE


def test_dependence_recovers_synthetic_garch_clustering():
    rng = np.random.default_rng(19)
    variance = np.full(2, .0009)
    observations = []
    for _ in range(1800):
        shock = rng.normal(size=2)*np.sqrt(variance)
        observations.append(shock)
        variance = .1*.0009 + .2*shock**2 + .7*variance
    returns = pd.DataFrame(observations[300:], columns=["A", "B"])
    fitted = fit_dependence(returns, pd.Series("state", index=returns.index),
                            {"state": moments_for(returns)})
    assert abs(fitted["garch_alpha"]-.2) < .08
    assert abs(fitted["garch_beta"]-.7) < .12
    assert fitted["garch_objective"] < fitted["garch_default_objective"]


@pytest.mark.parametrize("duplicate", [False, True])
def test_ecf_refinement_preserves_covariance_and_improves_objective(duplicate):
    returns, _ = panel()
    if duplicate:
        returns["B"] = returns["A"]
    moments = moments_for(returns)
    fitted = fit_mnts_parameters(moments, returns.iloc[:60], returns)
    fitted.validate(list(returns))
    assert fitted.estimation["method"] == "regularized_multivariate_ecf"
    assert np.isfinite(fitted.estimation["objective"])
    assert fitted.estimation["objective"] <= fitted.estimation["initial_objective"]
    variance = (1-fitted.tail_index/2)/fitted.tempering
    beta = fitted.skewness.to_numpy()*np.sqrt(variance)
    gamma = np.sqrt(1-beta**2)
    reconstructed = np.outer(beta, beta) + np.outer(gamma, gamma)*fitted.gaussian_correlation
    np.testing.assert_allclose(reconstructed, moments.correlation, atol=1e-7)


def test_selection_never_sees_final_holdout():
    returns, macro = panel(96)
    calls = []

    def runner(training, training_macro, **kwargs):
        calls.append((training.copy(), training_macro.copy()))
        assert training.index.max() < returns.index[-12]
        assert training_macro.index.max() < returns.index[-12]
        score = abs(kwargs["mean_prior_strength"]-12)
        return WalkForwardResult(pd.DataFrame(), pd.Series({"regime_energy_score_mean": score, "splits": 4}))

    report = select_calibration_hyperparameters(returns, macro, validation_kwargs={}, validation_runner=runner)
    changed_returns, changed_macro = returns.copy(), macro.copy()
    changed_returns.iloc[-12:] = 1000
    changed_macro.iloc[-12:] = -1000
    repeated = select_calibration_hyperparameters(changed_returns, changed_macro,
                                                  validation_kwargs={}, validation_runner=runner)
    assert repeated == report
    assert report["selected_parameters"] == {"mean_prior_strength": 12.0}
    assert report["selection_scores_are_test_scores"] is False
    assert len(calls) == 10


def test_selection_rejects_insufficient_history():
    returns, macro = panel(72)
    with pytest.raises(ValueError, match="84"):
        select_calibration_hyperparameters(returns, macro, validation_kwargs={})


def test_pipeline_selection_reports_only_final_holdout_scores():
    returns, macro = panel(96)
    scenario = run_scenario(
        returns=returns, macro=macro, selected_tickers=list(returns),
        growth_col="growth", inflation_col="inflation", growth_threshold="median",
        inflation_threshold="median", periods=3, paths=8, random_seed=5,
        start_state=None, weights={"A": .5, "B": .5},
        walk_forward=True, select_hyperparameters=True, parameter_draws=2,
    )
    report = scenario.model.metadata["hyperparameter_selection"]
    assert scenario.model.metadata["mean_prior_strength"] == report["selected_parameters"]["mean_prior_strength"]
    assert scenario.walk_forward is not None
    assert len(scenario.walk_forward.splits) == 12
    assert pd.to_datetime(scenario.walk_forward.splits["date"]).min() >= pd.Timestamp(report["holdout_start"])
    assert len(scenario.model.metadata["simulation_dependence_draws"]) == 2


def test_macro_residual_fit_and_anchoring_do_not_mutate_cached_model():
    returns, macro = panel()
    reference = calibrate_quadrant_model(returns, macro, joint_macro=True, probabilistic_regimes=True)
    dynamics = reference.metadata["macro_dynamics"]
    assert dynamics["return_exposure_basis"] == "state_centered_forecast_innovations"
    assert "_residual_returns" not in dynamics
    assert np.isfinite(dynamics["forecast_rmse"]).all()
    assert set(dynamics["residual_mnts"]) == set(reference.states)
    assert reference.metadata["dependence_fit"]["input_basis"] == "macro_residual_returns"
    metadata = deepcopy(reference.metadata)
    metadata["macro_dynamics"]["latest"] = [999.0]*3
    metadata["duration_hazards"] = {state: [0.2] for state in reference.states}
    sampled = replace(reference, metadata=metadata)
    anchored = anchor_parameter_models([sampled], reference)[0]
    assert sampled.metadata["macro_dynamics"]["latest"] == [999.0]*3
    assert anchored.metadata["macro_dynamics"]["latest"] == dynamics["latest"]
    assert anchored.metadata["forecast_origin"] == "observed_reference_state"
    for state, posterior in reference.metadata["hsmm_latest_state_age_probabilities"].items():
        assert len(anchored.metadata["duration_hazards"][state]) >= len(posterior)
        assert len(sampled.metadata["duration_hazards"][state]) == 1
