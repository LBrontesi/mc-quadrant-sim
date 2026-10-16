from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from test_validation import _sample_history

from mc_quadrants import validation
from mc_quadrants.api import _validation_response
from mc_quadrants.path_validation import crps, score_horizon, variogram_score
from mc_quadrants.types import RegimeMoments, ScenarioModel, SimulationResult

OPTIONS = dict(
    growth_col="growth",
    inflation_col="inflation",
    growth_threshold="median",
    inflation_threshold="median",
    min_train_periods=60,
    step=200,
    validation_paths=32,
)


def test_crps_matches_pairwise_definition():
    draws = np.array([-4.0, -0.2, -0.2, 1.3, 8.0])
    expected = np.abs(draws - 0.7).mean() - 0.5 * np.abs(draws[:, None] - draws).mean()
    assert crps(0.7, draws) == pytest.approx(expected)
    assert crps(3, np.array([3, 3])) == 0
    assert crps(4, np.array([3, 3])) == 1


def test_variogram_detects_dependence_with_identical_marginals():
    positive = np.array([[-1, -1], [1, 1]])
    negative = np.array([[-1, 1], [1, -1]])
    actual = np.array([1, 1])
    assert variogram_score(actual, positive) == 0
    assert variogram_score(actual, negative) == pytest.approx(2)
    assert variogram_score(np.array([1]), positive[:, :1]) == 0


def test_horizon_replays_exact_simple_returns_cpi_and_buy_hold():
    assets = ["A", "B"]
    moments = RegimeMoments(
        pd.Series([0, 0], index=assets),
        pd.DataFrame(np.eye(2), index=assets, columns=assets),
        pd.DataFrame(np.eye(2), index=assets, columns=assets),
        30,
    )
    model = ScenarioModel(["s"], pd.DataFrame([[1.0]], index=["s"], columns=["s"]), {"s": moments})
    monthly = np.log(np.array([[2.0, 0.5], [0.5, 2.0]]))
    forecast = SimulationResult(
        np.repeat(monthly[:, None, :], 4, axis=1), np.full((2, 4), "s"), assets, ["s"], "M"
    )
    observed = replace(forecast, returns=forecast.returns[:, :1], regimes=forecast.regimes[:, :1])
    dates = pd.date_range("2020-01-31", periods=3, freq="ME")
    row = score_horizon(
        model=model,
        forecast=forecast,
        benchmark=forecast,
        observed=observed,
        weights={"A": 0.5, "B": 0.5},
        portfolio_options={"rebalance_frequency": 0},
        inflation_col="inflation",
        rate_col=None,
        observed_macro=pd.DataFrame(index=dates),
        price_index=pd.Series([100, 110, 125], index=dates),
        origin=dates[0],
        horizon=2,
        outcome_dates=dates[1:],
        forecast_components=[(model, forecast)],
        macro_cutoff=dates[0],
        energy_score=validation._energy_score,
    )
    # Monthly rebalancing: (0.5*2 + 0.5*0.5)^2 = 1.5625. Buy & hold: 1.0.
    assert row["portfolio_return_actual"] == pytest.approx(0.5625)
    assert row["portfolio_return_crps"] == 0
    assert row["wealth_actual"] == pytest.approx(100)
    assert row["wealth_crps"] == 0
    assert row["drawdown_actual"] == pytest.approx(0.2)
    assert row["real_wealth_actual"] == pytest.approx(80)
    assert row["real_wealth_mean"] == pytest.approx(100)  # zero predicted inflation
    assert row["real_wealth_crps"] == pytest.approx(20)
    assert row["real_wealth_coverage_90"] == 0
    assert row["cumulative_inflation_actual"] == pytest.approx(0.25)
    assert row["cumulative_inflation_crps"] == pytest.approx(0.25)


def test_full_paths_are_causal_and_use_production_settings(monkeypatch):
    returns, macro = _sample_history()
    returns, macro = returns.iloc[:100], macro.iloc[:100].copy()
    macro["interest_rate"] = 2.0
    macro.attrs["observed_price_index"] = {str(d): float(100 + i) for i, d in enumerate(macro.index)}
    fits, forecasts = [], []
    original_fit, original_simulate = validation.calibrate_quadrant_model, validation.simulate_returns

    def fit(training, training_macro, **kwargs):
        fits.append((training.copy(), training_macro.copy(), kwargs))
        assert "observed_price_index" not in training_macro.attrs
        return original_fit(training, training_macro, **kwargs)

    def simulate(model, **kwargs):
        forecasts.append((model, kwargs.copy()))
        return original_simulate(model, **kwargs)

    monkeypatch.setattr(validation, "calibrate_quadrant_model", fit)
    monkeypatch.setattr(validation, "simulate_returns", simulate)
    kwargs = dict(
        OPTIONS,
        calibration_options={"joint_macro": True},
        macro_lag_periods=1,
        horizons=(1, 3, 12, 60),
        simulation_options={"duration_model": "semi_markov"},
        portfolio_options={
            "rebalance_frequency": 3,
            "transaction_cost_bps": 20,
            "asset_expense_ratios": {"Stocks": 0.01},
        },
    )
    baseline = validation.walk_forward_validation(returns, macro, **kwargs)
    origin = baseline.horizon_scores.iloc[0]["origin"]
    assert fits[0][0].index.max() == origin
    assert fits[0][1].index.max() == origin
    assert fits[0][1].iloc[-1]["growth"] == macro.loc[origin - pd.offsets.MonthEnd(1), "growth"]
    assert fits[0][2]["macro_lag_periods"] == 0
    assert forecasts[0][1]["joint_macro"] is True
    assert forecasts[0][1]["periods"] == 12
    assert forecasts[0][1]["garch_alpha"] == forecasts[0][0].metadata["dependence_fit"]["garch_alpha"]
    assert "start_state" not in forecasts[0][1]
    assert set(baseline.horizon_scores.horizon_months) == {1, 3, 12}
    assert baseline.horizon_summary.set_index("horizon_months").loc[60, "origins"] == 0
    assert (
        np.isfinite(baseline.horizon_scores[["energy_score", "real_wealth_crps", "inflation_crps"]])
        .all()
        .all()
    )

    changed_returns = returns.copy()
    changed_returns.loc[changed_returns.index > origin] += 0.03
    changed_macro = macro.copy()
    changed_macro.loc[changed_macro.index >= origin, "growth"] += 10
    changed_macro.loc[changed_macro.index >= origin, "inflation"] += 5
    changed_macro.attrs["observed_price_index"] = {
        date: value * (2 if pd.Timestamp(date) > origin else 1)
        for date, value in macro.attrs["observed_price_index"].items()
    }
    changed = validation.walk_forward_validation(changed_returns, changed_macro, **kwargs)
    forecast_columns = [
        column for column in baseline.horizon_scores if column.endswith(("_mean", "_p05", "_p50", "_p95"))
    ]
    pd.testing.assert_frame_equal(
        baseline.horizon_scores[forecast_columns], changed.horizon_scores[forecast_columns]
    )
    assert not np.allclose(baseline.horizon_scores.wealth_actual, changed.horizon_scores.wealth_actual)
    assert not np.allclose(
        baseline.horizon_scores.real_wealth_actual, changed.horizon_scores.real_wealth_actual
    )
    response = _validation_response(baseline)
    import json

    json.dumps(response, allow_nan=False)
    assert response["metadata"]["forecast_engine"] == "production_simulate_returns"
    assert "real_wealth_crps" in response["horizon_scores"]["columns"]
    assert len(response["horizon_summary"]["rows"]) == 4


def test_manual_coefficients_and_bootstrap_forecast_origin(monkeypatch):
    returns, macro = _sample_history()
    calls = []
    original = validation.simulate_returns

    def simulate(model, **kwargs):
        calls.append((model, kwargs.copy()))
        assert model.metadata["forecast_origin"] == "observed_reference_state"
        assert model.metadata["calibrate_dependence"] is False
        assert kwargs["garch_alpha"] == 0.07
        assert kwargs["dcc_alpha"] == 0.02
        assert kwargs["macro_parameter_uncertainty"] is False
        assert kwargs["transition_concentration"] is None
        return original(model, **kwargs)

    monkeypatch.setattr(validation, "simulate_returns", simulate)
    result = validation.walk_forward_validation(
        returns.iloc[:90],
        macro.iloc[:90],
        **OPTIONS,
        horizons=(1, 3),
        parameter_draws=2,
        use_fitted_dependence=False,
        calibration_options={"joint_macro": True},
        simulation_options={"garch_alpha": 0.07, "dcc_alpha": 0.02, "transition_concentration": 10},
    )
    assert len(calls) == 2
    assert sum(kwargs["paths"] for _, kwargs in calls) == 32
    assert result.metadata["parameter_draws"] == 2


def test_real_wealth_is_not_invented_from_annual_inflation():
    returns, macro = _sample_history()
    result = validation.walk_forward_validation(
        returns.iloc[:90], macro.iloc[:90], **OPTIONS, horizons=(1, 3)
    )
    assert "real_wealth_crps" not in result.horizon_scores
    assert result.metadata["real_wealth_outcome_basis"] == "unavailable"
    assert any("provide observed CPI" in warning for warning in result.warnings)


def test_missing_months_are_not_treated_as_monthly_paths():
    returns, macro = _sample_history()
    # First evaluation begins at index 72 after the causal classifier warmup.
    returns = returns.iloc[:100].drop(returns.index[75])
    result = validation.walk_forward_validation(returns, macro.iloc[:100], **OPTIONS, horizons=(1, 3, 12))
    assert 12 not in result.horizon_scores.horizon_months.values
    assert any("missing calendar months" in warning for warning in result.warnings)


@pytest.mark.parametrize(
    "options, message",
    [
        ({"validation_paths": 1}, "at least 2"),
        ({"horizons": (0, 12)}, "integer months"),
        ({"step": 0}, "positive integer"),
        ({"simulation_options": {"start_state": "low_growth_low_inflation"}}, "forecast origins"),
    ],
)
def test_invalid_path_validation_options(options, message):
    returns, macro = _sample_history()
    with pytest.raises(ValueError, match=message):
        validation.walk_forward_validation(returns, macro, **{**OPTIONS, **options})


def test_csv_cpi_keeps_observation_dates_when_predictors_move_to_release_dates():
    from mc_quadrants.api import load_data_source

    macro, *_ = load_data_source(
        {
            "source": "csv",
            "asset_input": "Log returns",
            "csv_prices": "Date,A\n2020-01-31,0.01\n2020-02-29,0.02\n",
            "csv_macro": "Date,growth,inflation,price_index,AvailableDate\n2020-01-31,2,3,100,2020-02-15\n2020-02-29,2,3,101,2020-03-15\n",
        }
    )
    assert macro.index[0] == pd.Timestamp("2020-02-29")
    assert pd.Timestamp(next(iter(macro.attrs["observed_price_index"]))) == pd.Timestamp("2020-01-31")
    assert "price_index" not in macro.columns


def test_validation_cache_includes_cpi_outcomes_and_vintage(monkeypatch):
    from mc_quadrants import pipeline

    pipeline._clear_walk_forward_cache()
    calls = []

    def fake(*args, **kwargs):
        calls.append(1)
        return validation.WalkForwardResult(pd.DataFrame(), pd.Series(dtype=float))

    monkeypatch.setattr(pipeline, "walk_forward_validation", fake)
    returns, macro = _sample_history()
    pipeline._cached_walk_forward_validation(returns, macro)
    pipeline._cached_walk_forward_validation(returns, macro)
    macro.attrs["observed_price_index"] = {str(macro.index[0]): 100}
    pipeline._cached_walk_forward_validation(returns, macro)
    macro.attrs["observed_price_index"][str(macro.index[0])] = 101
    pipeline._cached_walk_forward_validation(returns, macro)
    macro.attrs["point_in_time"] = True
    pipeline._cached_walk_forward_validation(returns, macro)
    assert len(calls) == 4
    pipeline._clear_walk_forward_cache()


def test_pipeline_forwards_model_and_ledger_settings_to_validation(monkeypatch):
    from test_pipeline import _scenario_kwargs

    from mc_quadrants import pipeline

    calls = []

    def fake(*args, **kwargs):
        calls.append(kwargs)
        return validation.WalkForwardResult(pd.DataFrame(), pd.Series(dtype=float))

    monkeypatch.setattr(pipeline, "_cached_walk_forward_validation", fake)
    pipeline.run_scenario(
        **_scenario_kwargs(
            {
                "joint_macro": True,
                "macro_model": "ridge_var",
                "duration_model": "markov",
                "rebalance_frequency": 3,
                "transaction_cost_bps": 20,
                "contribution": 2,
                "asset_expense_ratios": {"Stocks": 0.01},
                "calibrate_dependence": False,
                "garch_alpha": 0.05,
                "dcc_alpha": 0.03,
                "parameter_draws": 2,
            }
        )
    )
    assert len(calls) == 1
    settings = calls[0]
    assert settings["calibration_options"]["joint_macro"] is True
    assert settings["calibration_options"]["macro_model"] == "ridge_var"
    assert settings["simulation_options"]["duration_model"] == "markov"
    assert settings["simulation_options"]["garch_alpha"] == 0.05
    assert settings["simulation_options"]["dcc_alpha"] == 0.03
    assert settings["portfolio_options"]["rebalance_frequency"] == 3
    assert settings["portfolio_options"]["transaction_cost_bps"] == 20
    assert settings["portfolio_options"]["contribution"] == 2
    assert settings["portfolio_options"]["asset_expense_ratios"] == {"Stocks": 0.01}
    assert settings["parameter_draws"] == 2


def test_energy_score_matches_all_pairs_and_is_invariant_to_mixture_order():
    values = np.array([[-3.0, -2.0], [-2.0, -1.0], [1.0, 3.0], [2.0, 4.0]])
    observation = np.array([0.3, 0.5])
    expected = (
        np.linalg.norm(values - observation, axis=1).mean()
        - 0.5 * np.linalg.norm(values[:, None] - values[None, :], axis=2).mean()
    )
    assert validation._energy_score(observation, values) == pytest.approx(expected)
    assert validation._energy_score(observation, values[[0, 2, 1, 3]]) == pytest.approx(expected)


def test_observed_cpi_drives_inflation_indexed_spending_in_historical_replay():
    from mc_quadrants.decumulation import normalize_decumulation

    assets = ["A"]
    matrix = pd.DataFrame([[1.0]], index=assets, columns=assets)
    moments = RegimeMoments(pd.Series([0.0], index=assets), matrix, matrix, 30)
    model = ScenarioModel(["s"], pd.DataFrame([[1.0]], index=["s"], columns=["s"]), {"s": moments})
    forecast = SimulationResult(np.zeros((1, 4, 1)), np.full((1, 4), "s"), assets, ["s"], "M")
    observed = replace(forecast, returns=forecast.returns[:, :1], regimes=forecast.regimes[:, :1])
    dates = pd.date_range("2020-01-31", periods=2, freq="ME")
    plan = normalize_decumulation(
        {
            "enabled": True,
            "phases": [
                {"start_month": 1, "end_month": 1, "annual_real_amount": 120},
            ],
        },
        periods=1,
    )
    row = score_horizon(
        model=model,
        forecast=forecast,
        benchmark=forecast,
        observed=observed,
        weights={"A": 1.0},
        portfolio_options={"rebalance_frequency": 1, "decumulation": plan},
        inflation_col="inflation",
        rate_col=None,
        observed_macro=pd.DataFrame(index=dates),
        price_index=pd.Series([100, 120], index=dates),
        origin=dates[0],
        horizon=1,
        outcome_dates=dates[1:],
        forecast_components=[(model, forecast)],
        macro_cutoff=dates[0],
        energy_score=validation._energy_score,
    )
    assert row["wealth_mean"] == pytest.approx(90)
    assert row["wealth_actual"] == pytest.approx(88)
    assert row["wealth_crps"] == pytest.approx(2)
    assert row["real_wealth_actual"] == pytest.approx(88 / 1.2)
