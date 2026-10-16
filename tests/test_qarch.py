"""Independent accounting of QARCH feedback and production routing checks."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import mc_quadrants.simulation as simulation
from mc_quadrants.qarch import (
    DEFAULT_WEIGHTS,
    LAGS,
    filtered_ratios,
    fit_qarch,
    kernel,
    variance_ratio,
)
from mc_quadrants.types import MNTSParameters, RegimeMoments, ScenarioModel


def test_qarch_is_positive_and_distinguishes_equal_size_return_patterns():
    matrix = kernel(DEFAULT_WEIGHTS)
    assert np.linalg.eigvalsh(matrix).min() >= -1e-12
    assert np.trace(matrix) < 1
    trending = np.ones(LAGS)
    alternating = np.tile([1., -1.], LAGS // 2)
    assert variance_ratio(trending, matrix) > variance_ratio(alternating, matrix)
    # A constant path has explicit quadratic block contributions of 3 and 12.
    expected = 1 - DEFAULT_WEIGHTS.sum() + DEFAULT_WEIGHTS @ [1, 3, 12]
    assert variance_ratio(trending, matrix) == pytest.approx(expected)
    draws = np.random.default_rng(2).normal(size=(500, LAGS)) * 100
    assert np.min(variance_ratio(draws, matrix)) >= 1 - np.trace(matrix)
    assert variance_ratio(np.zeros(LAGS), matrix, observed=0) == pytest.approx(1)


@pytest.mark.parametrize("weights", [[-0.1, .1, .1], [.8, .1, .1], [np.nan, .1, .1], [.1]])
def test_qarch_rejects_invalid_kernels(weights):
    with pytest.raises(ValueError, match="QARCH"):
        kernel(weights)


def test_qarch_fit_and_forecast_history_are_scale_invariant():
    rng = np.random.default_rng(13)
    residuals = rng.normal(size=(180, 2))
    levels = np.ones_like(residuals)
    fitted = fit_qarch(residuals, levels)
    scaled = fit_qarch(residuals * [2, 5], levels * [4, 25])
    assert fitted["status"] == "fitted"
    assert fitted["objective"] <= fitted["default_objective"] + 1e-12
    np.testing.assert_allclose(fitted["weights"], scaled["weights"])
    np.testing.assert_allclose(fitted["history"], residuals[-LAGS:][::-1].T)
    fallback = fit_qarch(residuals[:20], levels[:20])
    assert fallback["status"] == "insufficient_history"
    np.testing.assert_allclose(fallback["weights"], DEFAULT_WEIGHTS)


def test_qarch_filter_is_causal():
    returns = np.random.default_rng(4).normal(size=(30, 2))
    altered = returns.copy()
    altered[15:] *= 100
    matrix = kernel(DEFAULT_WEIGHTS)
    original = filtered_ratios(returns, matrix)
    changed = filtered_ratios(altered, matrix)
    # The month-15 forecast is already fixed before its return is observed.
    np.testing.assert_array_equal(original[:16], changed[:16])
    assert not np.allclose(original[16:], changed[16:])


def test_qarch_simulation_keeps_feedback_across_regimes_and_bypasses_garch_native(monkeypatch):
    assets, states = ["A"], ["calm", "stress"]
    correlation = pd.DataFrame([[1.]], index=assets, columns=assets)
    moments = {
        state: RegimeMoments(
            mean=pd.Series([0.], index=assets), covariance=correlation * sigma**2,
            correlation=correlation, observations=100,
            mnts=MNTSParameters(1.2, 1., pd.Series([0.], index=assets), correlation),
        ) for state, sigma in zip(states, [.1, .2])
    }
    history = np.zeros((1, LAGS))
    history[0, 0] = 4.
    model = ScenarioModel(states, pd.DataFrame(.5, index=states, columns=states), moments,
                          metadata={"dependence_fit": {"qarch": {
                              "history": history.tolist(), "history_observations": LAGS,
                              "weights": DEFAULT_WEIGHTS.tolist(),
                          }}})
    monkeypatch.setattr(simulation, "simulate_regime_paths", lambda *a, **kw: np.array([[0], [1], [0]]))
    monkeypatch.setattr(simulation, "_rng", lambda *a: SimpleNamespace(standard_normal=lambda shape: np.ones(shape)))
    monkeypatch.setattr(simulation, "sample_mnts_subordinators", lambda rng, n, *a: np.ones(n))

    def forbidden(**kwargs):
        raise AssertionError("QARCH entered a GARCH-only native return kernel")

    monkeypatch.setattr(simulation, "simulate_parametric_native", forbidden)
    result = simulation.simulate_returns(model, periods=3, paths=1, volatility_model="qarch", random_seed=1)
    matrix = kernel(DEFAULT_WEIGHTS)
    expected = []
    for sigma in [.1, .2, .1]:
        normalized = np.sqrt(1 - np.trace(matrix) + history[0] @ matrix @ history[0])
        expected.append(sigma * normalized)
        history[:, 1:] = history[:, :-1].copy()
        history[:, 0] = normalized
    np.testing.assert_allclose(result.returns[:, 0, 0], expected, rtol=1e-8)


@pytest.mark.parametrize("joint_macro,draws,workers", [(False, 0, 1), (True, 2, 1), (False, 0, 2)])
def test_qarch_pipeline_runs_chunked_with_macro_bootstrap_and_workers(joint_macro, draws, workers):
    from test_pipeline import _scenario_kwargs

    from mc_quadrants.pipeline import run_scenario

    run = run_scenario(**_scenario_kwargs({
        "volatility_model": "qarch", "joint_macro": joint_macro, "parameter_draws": draws,
        "paths": 8, "periods": 6, "chunk_size": 2, "workers": workers, "walk_forward": False,
        "dynamic_correlation": True,
    }))
    assert run.wealth.shape == (6, 8)
    assert np.isfinite(run.wealth.to_numpy()).all()
    assert run.model.metadata["volatility_model"] == "qarch"
    assert "qarch_weights" in run.model.metadata["simulation_dependence"]
    assert not run.wealth.attrs.get("compact_reporting", False)


def test_qarch_api_selection_and_native_planning():
    from mc_quadrants.api import _native_fused_payload, scenario_kwargs

    kwargs = scenario_kwargs({"volatility_model": "qarch", "weights": {"SPY": 1.0}})
    assert kwargs["volatility_model"] == "qarch"
    assert not _native_fused_payload({"volatility_model": "qarch", "paths": 100000})
    with pytest.raises(ValueError, match="volatility_model"):
        scenario_kwargs({"volatility_model": "unknown", "weights": {"SPY": 1.0}})


def test_qarch_api_returns_serializable_methodology_and_wealth():
    import json

    from test_api import _csv_payload

    from mc_quadrants.api import build_simulate_response

    response = build_simulate_response(_csv_payload(volatility_model="qarch", walk_forward=False))
    assert response["ok"] is True
    assert response["methodology"]["volatility_model"] == "qarch"
    assert len(response["methodology"]["simulation_dependence"]["qarch_weights"]) == 3
    json.dumps(response, allow_nan=False)


def test_qarch_hmm_manual_coefficients_and_bootstrap_origin():
    from test_pipeline import _scenario_kwargs

    from mc_quadrants.pipeline import run_scenario
    from mc_quadrants.uncertainty import anchor_parameter_models

    run = run_scenario(**_scenario_kwargs({
        "volatility_model": "qarch", "model_kind": "hmm", "hmm_states": 2,
        "calibrate_dependence": False, "walk_forward": False,
    }))
    np.testing.assert_array_equal(run.model.metadata["simulation_dependence"]["qarch_weights"], DEFAULT_WEIGHTS)
    assert run.model.metadata["dependence_fit"]["qarch"]["history_observations"] == LAGS
    from copy import deepcopy
    from dataclasses import replace

    alternate = replace(run.model, metadata=deepcopy(run.model.metadata))
    alternate.metadata["dependence_fit"]["qarch"]["history"] = np.zeros((2, LAGS)).tolist()
    anchored = anchor_parameter_models([alternate], run.model)[0]
    np.testing.assert_array_equal(anchored.metadata["dependence_fit"]["qarch"]["history"],
                                  run.model.metadata["dependence_fit"]["qarch"]["history"])
    assert np.count_nonzero(alternate.metadata["dependence_fit"]["qarch"]["history"]) == 0


def test_qarch_walk_forward_scores_selected_model():
    from test_validation import _sample_history

    from mc_quadrants.validation import walk_forward_validation

    returns, macro = _sample_history()
    kwargs = dict(growth_col="growth", inflation_col="inflation", growth_threshold="median",
                  inflation_threshold="median", min_train_periods=60, max_splits=2)
    qarch = walk_forward_validation(returns.iloc[:85], macro.iloc[:85], volatility_model="qarch", **kwargs)
    garch = walk_forward_validation(returns.iloc[:85], macro.iloc[:85], volatility_model="garch", **kwargs)
    assert np.isfinite(qarch.summary["regime_energy_score_mean"])
    assert not np.allclose(qarch.splits["regime_energy_score"], garch.splits["regime_energy_score"])
