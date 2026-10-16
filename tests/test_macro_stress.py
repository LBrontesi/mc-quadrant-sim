from copy import deepcopy

import numpy as np
import pytest

from mc_quadrants.api import scenario_kwargs
from mc_quadrants.calibration import calibrate_quadrant_model
from mc_quadrants.economic_returns import economic_assumptions
from mc_quadrants.macro_stress import (
    filter_news,
    fit_macro_stress,
    news_correction,
    update_stress,
    variance_filter,
)
from mc_quadrants.simulation import simulate_joint_regime_macro_paths
from mc_quadrants.uncertainty import anchor_parameter_models, bootstrap_quadrant_models
from tests.test_economic_returns import history


def test_news_and_variance_filters_use_only_prior_observations_and_reset_at_joins():
    errors = np.arange(24, dtype=float).reshape(12, 2)
    valid = np.ones(12, dtype=bool)
    valid[5] = False
    features, final = filter_news(errors, valid)
    changed = errors.copy()
    changed[8:] += 1000
    alternative, _ = filter_news(changed, valid)
    np.testing.assert_array_equal(features[:9], alternative[:9])
    np.testing.assert_array_equal(features[6], [0, 0])
    assert (final > 0).all()
    energy = np.ones(12)
    energy[4] = 20
    predicted, _ = variance_filter(energy, valid, 0.1, 0.8)
    assert predicted[4] == pytest.approx(1)
    assert predicted[5] > 1
    assert predicted[6] == pytest.approx(1)


def test_persistent_news_improves_recoverable_synthetic_forecast_mean():
    rng = np.random.default_rng(42)
    values = np.empty((3000, 2))
    state = np.zeros(2)
    for t in range(len(values)):
        values[t] = state * 1.5 + rng.normal(size=2)
        state = 0.8 * state + 0.2 * values[t]
    valid = np.ones(len(values), dtype=bool)
    specification, residuals, _ = fit_macro_stress(values[:2000], valid[:2000], "persistent")
    features, _ = filter_news(values, valid)
    predictions = news_correction(features, specification)
    assert np.mean((values[2000:] - predictions[2000:]) ** 2) < 0.7 * np.mean(values[2000:] ** 2)
    transition = 0.8 * np.eye(2) + 0.2 * np.asarray(specification["response"])
    assert np.max(np.abs(np.linalg.eigvals(transition))) <= 0.97
    assert np.isfinite(residuals).all()


def test_macro_variance_fit_recovers_clustering_without_inflating_unconditional_risk():
    rng = np.random.default_rng(123)
    errors = np.empty((2400, 4))
    variance = 1.0
    for t in range(len(errors)):
        errors[t] = rng.normal(size=4) * np.sqrt(variance)
        variance = 0.05 + 0.15 * np.mean(errors[t] ** 2) + 0.8 * variance
    spec, _, _ = fit_macro_stress(errors, np.ones(len(errors), dtype=bool), "volatility")
    assert spec["alpha"] > 0
    assert spec["alpha"] + spec["beta"] < 1
    variance = np.ones(3000)
    news = np.zeros((3000, 4))
    means = []
    for t in range(240):
        shocks = rng.normal(size=news.shape) * np.sqrt(variance[:, None])
        news, variance = update_stress(news, variance, np.zeros_like(news), shocks, spec)
        if t > 120:
            means.append(variance.mean())
    assert 0.85 < np.mean(means) < 1.15


def test_stress_fit_is_unit_invariant_and_join_outliers_cannot_train_it():
    rng = np.random.default_rng(92)
    errors = rng.normal(size=(400, 3))
    valid = np.ones(400, dtype=bool)
    valid[::12] = False
    first, residuals, _ = fit_macro_stress(errors, valid, "combined")
    altered = errors.copy()
    altered[~valid] = 1e10
    second, altered_residuals, _ = fit_macro_stress(altered, valid, "combined")
    assert first == second
    np.testing.assert_array_equal(residuals, altered_residuals)
    units = np.array([100.0, 0.01, 1.0])
    converted, converted_residuals, _ = fit_macro_stress(errors * units, valid, "combined")
    np.testing.assert_allclose(np.array(first["response"]), converted["response"], atol=1e-12)
    np.testing.assert_allclose(residuals * units, converted_residuals, atol=1e-12)
    assert first["latest_variance"] == pytest.approx(converted["latest_variance"])


def test_simulated_yield_response_to_news_is_persistent_without_level_attraction():
    returns, macro = history()
    model = calibrate_quadrant_model(
        returns,
        macro,
        joint_macro=True,
        economic_returns={"macro_stress": "persistent", "yield_model": "random_walk"},
    )
    dynamics = model.metadata["macro_dynamics"]
    spec = dynamics["stress_dynamics"]
    dimension = len(dynamics["columns"])
    index = dynamics["columns"].index("bond_yield")
    response = np.zeros((dimension, dimension))
    response[index, index] = 0.5
    spec.update(
        response=response.tolist(),
        scale=np.ones(dimension).tolist(),
        latest_news=np.eye(dimension)[index].tolist(),
    )
    for state in model.states:
        dynamics["state_innovation_covariances"][state] = (np.eye(dimension) * 1e-14).tolist()
    _, levels, _ = simulate_joint_regime_macro_paths(
        model, 6, 20, random_seed=42, macro_parameter_uncertainty=False
    )
    changes = np.diff(np.vstack([np.full((1, 20), dynamics["latest"][index]), levels[..., index]]), axis=0)
    expected = np.broadcast_to((0.5 * 0.9 ** np.arange(6))[:, None], changes.shape)
    np.testing.assert_allclose(changes, expected, atol=1e-4)
    clean = deepcopy(model)
    clean.metadata["macro_dynamics"]["stress_dynamics"]["latest_news"] = np.zeros(dimension).tolist()
    _, levels, _ = simulate_joint_regime_macro_paths(
        clean, 6, 20, random_seed=42, macro_parameter_uncertainty=False
    )
    np.testing.assert_allclose(levels[..., index], dynamics["latest"][index], atol=1e-4)


def test_bootstrap_restores_observed_stress_state_without_replacing_fitted_parameters():
    returns, macro = history()
    settings = dict(joint_macro=True, economic_returns={"macro_stress": "combined"})
    reference = calibrate_quadrant_model(returns, macro, **settings)
    draws = bootstrap_quadrant_models(returns, macro, draws=2, random_seed=34, **settings)
    anchored = anchor_parameter_models(draws, reference)
    for before, after in zip(draws, anchored):
        old = before.metadata["macro_dynamics"]["stress_dynamics"]
        new = after.metadata["macro_dynamics"]["stress_dynamics"]
        observed = reference.metadata["macro_dynamics"]["stress_dynamics"]
        assert new["latest_news"] == observed["latest_news"]
        assert new["latest_variance"] == observed["latest_variance"]
        assert new["response"] == old["response"]
        assert new["scale"] == old["scale"]
    assert (
        draws[0].metadata["macro_dynamics"]["stress_dynamics"]["response"]
        != draws[1].metadata["macro_dynamics"]["stress_dynamics"]["response"]
    )


def test_api_preserves_stress_configuration_and_rejects_incompatible_options():
    parsed = scenario_kwargs(
        {"weights": {"SPY": 1}, "joint_macro": True, "economic_returns": {"macro_stress": "combined"}}
    )
    assert parsed["economic_returns"]["macro_stress"] == "combined"
    with pytest.raises(ValueError, match="macro_stress"):
        economic_assumptions({"macro_stress": "wrong"})
    with pytest.raises(ValueError, match="rate_stress"):
        economic_assumptions({"macro_stress": "combined", "rate_stress": False})


def test_production_accounting_with_stress_and_monthly_cpi():
    from mc_quadrants.pipeline import run_scenario

    returns, macro = history()
    macro["cpi_monthly_log"] = 0.002
    macro["inflation"] = 100 * np.expm1(0.024)
    scenario = run_scenario(
        returns,
        macro,
        list(returns),
        "growth",
        "inflation",
        "median",
        "median",
        periods=24,
        paths=100,
        weights={"SPY": 0.6, "IEF": 0.4},
        random_seed=7,
        start_state=None,
        joint_macro=True,
        economic_returns={"macro_stress": "combined"},
        parameter_draws=2,
        parameter_seed=8,
        walk_forward=False,
    )
    assert scenario.model.metadata["macro_dynamics"]["stress_dynamics"]["mode"] == "combined"
    assert np.isfinite(scenario.wealth).all().all()
    assert np.isfinite(scenario.reporting_wealth).all().all()


def test_macro_volatility_accounts_for_risk_already_explained_by_regimes():
    rng = np.random.default_rng(953)
    states = (np.arange(2400) // 60) % 2
    membership = np.eye(2)[states]
    errors = rng.normal(size=(2400, 4)) * np.where(states == 0, 1, 5)[:, None]
    valid = np.ones(len(errors), dtype=bool)
    conditional, _, variance = fit_macro_stress(errors, valid, "volatility", membership)
    pooled, _, pooled_variance = fit_macro_stress(errors, valid, "volatility")
    assert conditional["alpha"] < pooled["alpha"]
    assert np.std(variance) < 0.25 * np.std(pooled_variance)
