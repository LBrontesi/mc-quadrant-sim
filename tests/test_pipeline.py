import numpy as np
import pandas as pd
import pytest

from mc_quadrants import pipeline
from mc_quadrants.native import native_available
from mc_quadrants.pipeline import run_scenario
from mc_quadrants.validation import WalkForwardResult


def _scenario_kwargs(extra: dict | None = None) -> dict:
    dates = pd.date_range("2020-01-31", periods=48, freq="ME")
    macro = pd.DataFrame(
        {
            "growth": np.tile([2.0, 2.5, -1.0, -1.5], 12),
            "inflation": np.tile([1.0, 4.0, 4.5, 1.2], 12),
            "interest_rate": np.tile([1.0, 4.5, 5.0, 0.5], 12),
        },
        index=dates,
    )
    returns = pd.DataFrame(
        {
            "Stocks": np.linspace(-0.03, 0.04, len(dates)),
            "Bonds": np.linspace(0.02, -0.01, len(dates)),
        },
        index=dates,
    )
    kwargs = {
        "returns": returns,
        "macro": macro,
        "selected_tickers": ["Stocks", "Bonds"],
        "growth_col": "growth",
        "inflation_col": "inflation",
        "growth_threshold": 0.0,
        "inflation_threshold": 3.0,
        "periods": 4,
        "paths": 8,
        "random_seed": 3,
        "start_state": None,
        "weights": {"Stocks": 0.6, "Bonds": 0.4},
    }
    kwargs.update(extra or {})
    return kwargs


def test_pipeline_supports_hmm_model_kind():
    scenario = run_scenario(**_scenario_kwargs({"model_kind": "hmm", "hmm_states": 2}))

    assert scenario.model.metadata["model_kind"] == "hmm"
    assert scenario.model.states == ["state_0", "state_1"]
    assert scenario.result.returns.shape == (4, 8, 2)
    assert any("HMM fitted" in warning for warning in scenario.diagnostics.warnings)


@pytest.mark.parametrize("calibrated", [False, True])
def test_pipeline_passes_dependence_and_counts_macro_uncertainty_once(monkeypatch, calibrated):
    calls = []
    original = pipeline._simulate_chunked

    def recording(model, **kwargs):
        calls.append((model, kwargs.copy()))
        return original(model, **kwargs)

    monkeypatch.setattr(pipeline, "_simulate_chunked", recording)
    scenario = run_scenario(**_scenario_kwargs({
        "walk_forward": False, "parameter_draws": 2, "joint_macro": True,
        "macro_parameter_uncertainty": True, "calibrate_dependence": calibrated,
        "garch_alpha": .07, "garch_beta": .8, "dynamic_correlation": True,
    }))
    assert len(calls) == 2
    for model, kwargs in calls:
        assert kwargs["macro_parameter_uncertainty"] is False
        assert model.metadata["forecast_origin"] == "observed_reference_state"
        expected = model.metadata["dependence_fit"]["garch_alpha"] if calibrated else .07
        assert kwargs["garch_alpha"] == expected
    assert scenario.model.metadata["macro_parameter_uncertainty_mode"] == "bootstrap"


def test_pipeline_supports_semi_markov_duration_model():
    scenario = run_scenario(**_scenario_kwargs({"duration_model": "semi_markov"}))

    assert scenario.result.regimes.shape == (4, 8)
    assert scenario.result.regimes.dtype.kind in "iu"
    assert scenario.reporting_wealth is scenario.wealth
    assert scenario.gross_wealth is scenario.wealth
    assert scenario.gross_reporting_wealth is scenario.reporting_wealth
    assert scenario.model.metadata["model_kind"] == "quadrant"


def test_neutral_gross_reporting_reuses_inflation_adjusted_wealth():
    scenario = run_scenario(
        **_scenario_kwargs(
            {
                "annual_inflation": 0.02,
                "walk_forward": False,
            }
        )
    )

    assert scenario.gross_wealth is scenario.wealth
    assert scenario.gross_reporting_wealth is scenario.reporting_wealth


def test_pipeline_reports_italian_tax_accounting_across_chunks():
    scenario = run_scenario(
        **_scenario_kwargs(
            {
                "periods": 12,
                "paths": 12,
                "chunk_size": 4,
                "workers": 1,
                "walk_forward": False,
                "rebalance_frequency": 3,
                "tax_country": "IT",
                "tax_regime": "italy_administered",
                "asset_tax_categories": {
                    "Stocks": "fund",
                    "Bonds": "government_bond",
                },
                "italy_annual_wealth_tax": 0.002,
            }
        )
    )

    assert scenario.model.metadata["tax_country"] == "IT"
    assert scenario.model.metadata["tax_regime"] == "italy_administered"
    assert scenario.model.metadata["asset_tax_categories"] == {
        "Stocks": "fund",
        "Bonds": "government_bond",
    }
    assert scenario.summary["taxes_paid"] > 0
    assert scenario.summary["wealth_tax"] > 0
    assert scenario.summary["annual_wealth_tax_rate"] == 0.002
    assert scenario.wealth.attrs["taxes_paid_total"] == scenario.summary["taxes_paid"] * 12
    assert scenario.gross_wealth is not None
    assert scenario.gross_wealth.attrs["tax_country"] == "none"
    assert scenario.summary["gross_terminal_wealth_median"] >= scenario.summary[
        "after_tax_terminal_wealth_median"
    ]
    assert scenario.summary["terminal_tax_drag_median"] > 0


@pytest.mark.skipif(not native_available(), reason="native backend unavailable")
@pytest.mark.parametrize("paths", [12, 26_000])
def test_income_paying_assets_use_fused_ledger_with_annual_income_reporting(paths):
    scenario = run_scenario(**_scenario_kwargs({
        "periods": 13, "paths": paths, "workers": 4, "walk_forward": False,
        "rebalance_frequency": 3, "tax_country": "IT", "tax_regime": "italy_administered",
        "asset_tax_metadata": {"Stocks": {
            "category": "fund", "annual_income_yield": .04,
            "foreign_withholding_rate": .1, "foreign_tax_credit_rate": .05,
        }},
    }))
    assert scenario.wealth.attrs["native_backend"] is True
    assert scenario.wealth.attrs["native_fused_backend"] is True
    for yearly_key, total_key in (("investment_income_tax", "investment_income_tax_total"),
                                  ("foreign_withholding_tax", "foreign_withholding_tax_total")):
        annual_sum = sum(year[yearly_key] for year in scenario.wealth.attrs["tax_by_year"].values())
        assert annual_sum > 0
        assert annual_sum == pytest.approx(scenario.wealth.attrs[total_key], rel=1e-12)


@pytest.mark.skipif(not native_available(), reason="native backend unavailable")
def test_large_native_tax_run_uses_compact_exact_terminal_reporting():
    scenario = run_scenario(
        **_scenario_kwargs(
            {
                "periods": 6,
                "paths": 30_000,
                "workers": 4,
                "walk_forward": False,
                "rebalance_frequency": 3,
                "tax_country": "IT",
                "tax_regime": "italy_administered",
                "annual_inflation": 0.02,
            }
        )
    )

    terminal = np.asarray(scenario.reporting_wealth.attrs["full_terminal_values"])
    assert scenario.wealth.attrs["compact_reporting"] is True
    assert scenario.wealth.attrs["total_simulated_paths"] == 30_000
    assert scenario.wealth.shape == (6, 25_000)
    assert scenario.result.regimes.shape == (6, 25_000)
    assert terminal.shape == (30_000,)
    assert scenario.summary["p50"] == pytest.approx(np.median(terminal))
    assert np.asarray(scenario.wealth.attrs["native_max_drawdowns"]).shape == (30_000,)


@pytest.mark.skipif(not native_available(), reason="native backend unavailable")
def test_large_native_joint_macro_run_uses_streamed_compact_reporting():
    scenario = run_scenario(
        **_scenario_kwargs(
            {
                "periods": 6,
                "paths": 30_000,
                "workers": 4,
                "walk_forward": False,
                "rebalance_frequency": 3,
                "tax_country": "IT",
                "tax_regime": "italy_administered",
                "joint_macro": True,
                "duration_model": "semi_markov",
            }
        )
    )

    assert scenario.wealth.attrs["compact_reporting"] is True
    assert scenario.wealth.attrs["total_simulated_paths"] == 30_000
    assert scenario.result.regimes.shape == (6, 25_000)
    assert scenario.result.macro_paths.shape == (6, 25_000, 3)
    assert scenario.result.macro_columns == ["growth", "inflation", "interest_rate"]
    assert np.isfinite(scenario.result.macro_paths).all()
    assert np.all(scenario.result.macro_paths[:, :, 2] >= -5.0)
    assert np.all(scenario.result.macro_paths[:, :, 2] <= 50.0)
    assert scenario.result.native_portfolio["macro_paths"] is scenario.result.macro_paths


@pytest.mark.skipif(not native_available(), reason="native backend unavailable")
def test_large_neutral_joint_macro_run_uses_compact_real_reporting():
    scenario = run_scenario(
        **_scenario_kwargs(
            {
                "periods": 12,
                "paths": 30_000,
                "workers": 4,
                "walk_forward": False,
                "rebalance_frequency": 3,
                "tax_country": "none",
                "joint_macro": True,
                "duration_model": "semi_markov",
            }
        )
    )

    assert scenario.wealth.attrs["compact_reporting"] is True
    assert scenario.wealth.shape == (12, 25_000)
    assert scenario.result.regimes.shape == (12, 25_000)
    terminal = np.asarray(scenario.reporting_wealth.attrs["full_terminal_values"])
    sample_indices = np.asarray(scenario.wealth.attrs["sample_indices"], dtype=int)
    assert terminal.shape == (30_000,)
    assert np.allclose(terminal[sample_indices], scenario.reporting_wealth.iloc[-1])
    assert int(np.asarray(scenario.wealth.attrs["native_regime_counts"]).sum()) == 360_000
    assert scenario.result.native_portfolio is not None


@pytest.mark.skipif(not native_available(), reason="native backend unavailable")
@pytest.mark.parametrize("tax_country,tax_regime", [("none", "none"), ("IT", "italy_administered")])
def test_parameter_ensemble_shares_one_compact_reporting_budget(tax_country, tax_regime):
    scenario = run_scenario(
        **_scenario_kwargs(
            {
                "periods": 12,
                "paths": 30_004,
                "parameter_draws": 3,
                "workers": 4,
                "walk_forward": False,
                "rebalance_frequency": 3,
                "tax_country": tax_country,
                "tax_regime": tax_regime,
                "tax_wrapper_benchmark": tax_country == "IT",
                "joint_macro": True,
                "dynamic_correlation": True,
            }
        )
    )

    assert scenario.wealth.attrs["compact_reporting"] is True
    assert scenario.wealth.shape == (12, 25_000)
    assert np.asarray(scenario.wealth.attrs["terminal_values"]).shape == (30_004,)
    assert np.asarray(scenario.wealth.attrs["terminal_deflators"]).shape == (30_004,)
    assert np.asarray(scenario.wealth.attrs["native_max_drawdowns"]).shape == (30_004,)
    assert scenario.parameter_uncertainty is not None
    assert len(scenario.parameter_uncertainty) == 3
    sample_indices = np.asarray(scenario.wealth.attrs["sample_indices"])
    terminal = scenario.reporting_wealth.attrs["full_terminal_values"]
    assert len(np.unique(sample_indices)) == 25_000
    assert np.allclose(terminal[sample_indices], scenario.reporting_wealth.iloc[-1])
    if tax_country == "IT":
        wrapper_terminal = scenario.wealth.attrs["wrapper_terminal_values"]
        discounts = scenario.wealth.attrs["terminal_deflators"]
        assert scenario.summary["wrapper_terminal_median"] == pytest.approx(
            np.median(wrapper_terminal * discounts)
        )
        assert np.isfinite(scenario.summary["wrapper_annual_drag_bps"])


@pytest.mark.parametrize("flag", ["MC_DISABLE_NATIVE_SIM", "MC_DISABLE_NATIVE_FUSED"])
def test_disabled_native_neutral_run_keeps_reference_accounting(monkeypatch, flag):
    monkeypatch.setenv(flag, "1")
    scenario = run_scenario(**_scenario_kwargs({
        "periods": 2, "paths": 25_001, "walk_forward": False,
        "tax_country": "none", "rebalance_frequency": 1,
    }))
    assert not scenario.wealth.attrs.get("compact_reporting", False)
    assert scenario.wealth.shape == (2, 25_001)
    assert np.isfinite(scenario.wealth.to_numpy()).all()


@pytest.mark.skipif(not native_available(), reason="native backend unavailable")
@pytest.mark.parametrize("change,message", [
    ({"transaction_cost_bps": -1.0}, "transaction_cost_bps"),
    ({"rebalance_frequency": -1}, "rebalance_frequency"),
    ({"rebalance_frequency": 0, "transaction_cost_bps": 5.0}, "Transaction costs require"),
    ({"asset_expense_ratios": {"Stocks": -0.01}}, "expense ratios"),
    ({"asset_expense_ratios": {"Stocks": 1.0}}, "expense ratios"),
])
def test_compact_neutral_execution_preserves_portfolio_validation(change, message):
    with pytest.raises(ValueError, match=message):
        run_scenario(**_scenario_kwargs({
            "periods": 2, "paths": 25_001, "walk_forward": False,
            "tax_country": "none", "rebalance_frequency": 1, **change,
        }))


@pytest.mark.skipif(not native_available(), reason="native backend unavailable")
def test_neutral_compact_kernel_decline_falls_back_to_reference_ledger(monkeypatch):
    from mc_quadrants import simulation

    monkeypatch.setattr(
        simulation, "simulate_parametric_italian_portfolios_compact_native",
        lambda **kwargs: None,
    )
    scenario = run_scenario(**_scenario_kwargs({
        "periods": 2, "paths": 25_001, "walk_forward": False,
        "tax_country": "none", "rebalance_frequency": 1,
    }))
    assert not scenario.wealth.attrs.get("compact_reporting", False)
    assert scenario.wealth.shape == (2, 25_001)
    assert np.isfinite(scenario.wealth.to_numpy()).all()


@pytest.mark.skipif(not native_available(), reason="native backend unavailable")
@pytest.mark.parametrize("tax_country,tax_regime", [("none", "none"), ("IT", "italy_administered")])
@pytest.mark.parametrize("joint_macro,contribution", [(False, 0.0), (True, 2.0)])
def test_compact_summary_skips_reference_recalculation_with_identical_metrics(
    monkeypatch, tax_country, tax_regime, joint_macro, contribution,
):
    from mc_quadrants.simulation import summarize_wealth_risk

    def unexpected_reference(*args, **kwargs):
        pytest.fail("Complete native reductions must not recalculate sampled risk.")

    monkeypatch.setattr(pipeline, "summarize_wealth_risk", unexpected_reference)
    scenario = run_scenario(**_scenario_kwargs({
        "periods": 6, "paths": 25_001, "walk_forward": False,
        "tax_country": tax_country, "tax_regime": tax_regime,
        "rebalance_frequency": 1, "joint_macro": joint_macro,
        "annual_inflation": 0.02, "contribution": contribution,
    }))
    # Reproduce the old reference-then-native-overrides path. All reference
    # metrics are overwritten, including stochastic real terminal statistics.
    reference = summarize_wealth_risk(
        scenario.wealth, annual_inflation=0.02, contribution=contribution,
    )
    pipeline._apply_compact_risk_summary(
        reference, scenario.wealth, periods=6, contribution=contribution,
    )
    terminal = scenario.reporting_wealth.attrs["full_terminal_values"]
    lower, median, upper = np.quantile(terminal, [0.05, 0.50, 0.95])
    reference.update({
        "mean": float(terminal.mean()), "std": float(terminal.std()),
        "p05": lower, "p50": median, "p95": upper,
        "probability_of_loss": float(np.mean(terminal < 100.0)),
        "var_95": 100.0 - lower,
        "expected_shortfall_95": 100.0 - float(terminal[terminal <= lower].mean()),
        "terminal_skewness": float(pd.Series(terminal).skew()),
        "terminal_kurtosis": float(pd.Series(terminal).kurt()),
    })
    pd.testing.assert_series_equal(scenario.summary.iloc[:len(reference)], reference, check_exact=True)


@pytest.mark.skipif(not native_available(), reason="native backend unavailable")
def test_compact_summary_falls_back_when_reductions_are_unavailable(monkeypatch):
    reference_summary = pipeline.summarize_wealth_risk
    calls = []

    def tracked_reference(*args, **kwargs):
        calls.append(True)
        return reference_summary(*args, **kwargs)

    monkeypatch.setattr(pipeline, "_compact_risk_summary", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "summarize_wealth_risk", tracked_reference)
    scenario = run_scenario(**_scenario_kwargs({
        "periods": 2, "paths": 25_001, "walk_forward": False,
        "tax_country": "none", "rebalance_frequency": 1,
    }))
    assert scenario.wealth.attrs["compact_reporting"]
    assert calls == [True]
    assert np.isfinite(scenario.summary["annualized_return"])


@pytest.mark.parametrize("statistics,drawdowns,terminal", [
    (None, np.zeros(3), np.ones(3)),
    (np.zeros((9, 3)), None, np.ones(3)),
    (np.zeros((8, 3)), np.zeros(3), np.ones(3)),
    (np.zeros((9, 0)), np.zeros(0), np.ones(0)),
    (np.zeros((9, 3)), np.zeros(2), np.ones(3)),
    (np.zeros((9, 3)), np.zeros(3), np.ones(2)),
    (np.full((9, 3), np.nan), np.zeros(3), np.ones(3)),
    (np.zeros((9, 3)), np.full(3, np.inf), np.ones(3)),
    ([["invalid"]], np.zeros(3), np.ones(3)),
])
def test_compact_summary_declines_incomplete_reductions(statistics, drawdowns, terminal):
    wealth = pd.DataFrame(np.ones((2, 2)))
    wealth.attrs.update({
        "native_risk_statistics": statistics, "native_max_drawdowns": drawdowns,
        "terminal_values": terminal,
    })
    assert pipeline._compact_risk_summary(wealth, periods=2, contribution=0.0) is None


@pytest.mark.skipif(not native_available(), reason="native backend unavailable")
def test_compact_summary_preserves_scalar_inflation_validation():
    with pytest.raises(ValueError, match="annual_inflation"):
        run_scenario(**_scenario_kwargs({
            "periods": 2, "paths": 25_001, "walk_forward": False,
            "tax_country": "none", "rebalance_frequency": 1,
            "annual_inflation": -0.02,
        }))


@pytest.mark.skipif(not native_available(), reason="native backend unavailable")
def test_disabled_decumulation_keeps_native_fused_tax_execution():
    common = {
        "periods": 12,
        "paths": 12,
        "walk_forward": False,
        "rebalance_frequency": 3,
        "tax_country": "IT",
        "tax_regime": "italy_administered",
    }
    legacy = run_scenario(**_scenario_kwargs(common))
    disabled = run_scenario(
        **_scenario_kwargs(
            {
                **common,
                "decumulation": {
                    "enabled": False,
                    "mode": "manual",
                    "phases": [],
                    "one_time_expenses": [],
                },
            }
        )
    )

    assert disabled.wealth.attrs["native_fused_backend"] is True
    assert np.array_equal(disabled.result.regimes, legacy.result.regimes)
    assert np.array_equal(disabled.wealth.to_numpy(), legacy.wealth.to_numpy())


def test_taxed_scenario_reuses_the_country_neutral_gross_paths():
    shared = {
        "periods": 12,
        "paths": 12,
        "chunk_size": 4,
        "workers": 1,
        "walk_forward": False,
        "rebalance_frequency": 3,
    }
    neutral = run_scenario(**_scenario_kwargs(shared))
    italian = run_scenario(
        **_scenario_kwargs(
            {
                **shared,
                "tax_country": "IT",
                "tax_regime": "italy_administered",
                "italy_annual_wealth_tax": 0.002,
            }
        )
    )

    assert italian.gross_wealth is not None
    assert np.array_equal(italian.result.regimes, neutral.result.regimes)
    assert np.allclose(italian.gross_wealth, neutral.wealth)
    assert np.all(italian.gross_wealth.to_numpy() >= italian.wealth.to_numpy())


def test_pipeline_supports_garch_and_threshold_window():
    scenario = run_scenario(
        **_scenario_kwargs({"garch": True, "threshold_window": 12, "walk_forward": False})
    )

    assert scenario.model.metadata["threshold_window"] == 12
    assert np.isfinite(scenario.result.returns).all()
    assert any("causal expanding windows" in warning for warning in scenario.diagnostics.warnings)


def test_pipeline_does_not_double_lag_availability_aligned_macro():
    kwargs = _scenario_kwargs({"macro_lag_periods": 2, "walk_forward": False})
    kwargs["macro"].attrs.update(
        {
            "data_vintage": "user_point_in_time",
            "point_in_time": True,
            "availability_aligned": True,
        }
    )

    scenario = run_scenario(**kwargs)

    assert scenario.model.metadata["macro_lag_periods"] == 0
    assert scenario.model.metadata["requested_macro_lag_periods"] == 2


def test_pipeline_combines_parameter_macro_and_dependence_uncertainty():
    scenario = run_scenario(
        **_scenario_kwargs(
            {
                "paths": 12,
                "periods": 6,
                "walk_forward": False,
                "distribution": "mnts",
                "probabilistic_regimes": True,
                "mean_prior_strength": 24.0,
                "parameter_draws": 3,
                "parameter_block_size": 6,
                "joint_macro": True,
                "dynamic_correlation": True,
            }
        )
    )

    assert scenario.parameter_uncertainty is not None
    assert len(scenario.parameter_uncertainty) == 3
    assert scenario.result.regimes.shape == (6, 12)
    assert scenario.result.macro_paths.shape == (6, 12, 3)
    assert scenario.result.macro_columns == ["growth", "inflation", "interest_rate"]
    assert scenario.reporting_wealth.shape == scenario.wealth.shape
    assert scenario.model.metadata["regime_assignment"] == "probabilistic"
    assert scenario.model.metadata["inflation_model"] == "joint_macro_path"
    assert scenario.model.metadata["rate_model"] == "joint_macro_path"
    assert np.isfinite(scenario.summary["effective_risk_free_rate"])


def test_pipeline_runs_walk_forward_validation_on_long_history():
    rng = np.random.default_rng(2)
    dates = pd.date_range("1990-01-31", periods=240, freq="ME")
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
    scenario = run_scenario(
        returns=returns,
        macro=macro,
        selected_tickers=["Stocks", "Bonds"],
        growth_col="growth",
        inflation_col="inflation",
        growth_threshold="median",
        inflation_threshold="median",
        periods=6,
        paths=10,
        random_seed=3,
        start_state=None,
        weights={"Stocks": 0.6, "Bonds": 0.4},
    )

    assert scenario.walk_forward is not None
    assert scenario.walk_forward is not None
    assert scenario.walk_forward.summary["splits"] > 0


def test_walk_forward_cache_reuses_identical_inputs_and_invalidates_changes(monkeypatch):
    pipeline._clear_walk_forward_cache()
    calls = 0

    def fake_validation(returns, macro, **kwargs):
        nonlocal calls
        calls += 1
        return WalkForwardResult(
            splits=pd.DataFrame({"score": [float(returns.iloc[0, 0])]}),
            summary=pd.Series({"splits": 1}),
            warnings=[str(kwargs["growth_threshold"])],
        )

    monkeypatch.setattr(pipeline, "walk_forward_validation", fake_validation)
    dates = pd.date_range("2020-01-31", periods=3, freq="ME")
    returns = pd.DataFrame({"Stocks": [0.01, 0.02, 0.03]}, index=dates)
    macro = pd.DataFrame(
        {"growth": [1.0, 2.0, 3.0], "inflation": [2.0, 2.5, 3.0]},
        index=dates,
    )
    kwargs = {
        "growth_col": "growth",
        "inflation_col": "inflation",
        "growth_threshold": "median",
        "inflation_threshold": "median",
        "weights": {"Stocks": 1.0},
    }

    first = pipeline._cached_walk_forward_validation(returns, macro, **kwargs)
    first.warnings.append("caller mutation")
    second = pipeline._cached_walk_forward_validation(
        returns.copy(), macro.copy(), **kwargs
    )

    assert calls == 1
    assert "caller mutation" not in second.warnings

    changed_returns = returns.copy()
    changed_returns.iloc[0, 0] += 0.001
    pipeline._cached_walk_forward_validation(changed_returns, macro, **kwargs)
    pipeline._cached_walk_forward_validation(
        returns,
        macro,
        **{**kwargs, "growth_threshold": 0.0},
    )

    assert calls == 3
    pipeline._clear_walk_forward_cache()


def test_parameter_cache_isolated_and_invalidated_by_data_options_and_metadata(monkeypatch):
    pipeline._clear_parameter_model_cache()
    calls = []

    def fake_bootstrap(returns, macro, **kwargs):
        calls.append(kwargs)
        return [{"caller_notes": []}]

    monkeypatch.setattr(pipeline, "bootstrap_quadrant_models", fake_bootstrap)
    kwargs = _scenario_kwargs()
    returns, macro = kwargs["returns"], kwargs["macro"]
    options = {"random_seed": 42, "draws": 3, "asset_classes": {"Stocks": "equity"}}
    first = pipeline._cached_bootstrap_quadrant_models(returns, macro, **options)
    first[0]["caller_notes"].append("mutation")
    second = pipeline._cached_bootstrap_quadrant_models(returns.copy(), macro.copy(), **options)
    assert len(calls) == 1
    assert second[0]["caller_notes"] == []

    changed = returns.copy()
    changed.iloc[-1, -1] += 0.001
    pipeline._cached_bootstrap_quadrant_models(changed, macro, **options)
    changed_macro = macro.copy()
    changed_macro.attrs["point_in_time"] = True
    pipeline._cached_bootstrap_quadrant_models(returns, changed_macro, **options)
    pipeline._cached_bootstrap_quadrant_models(returns, macro, **{**options, "random_seed": 43})
    pipeline._cached_bootstrap_quadrant_models(returns, macro, **{**options, "draws": 4})
    assert len(calls) == 5
    assert len(pipeline._PARAMETER_MODEL_CACHE) <= pipeline._PARAMETER_MODEL_CACHE_MAX_ENTRIES

    unseeded = {**options, "random_seed": None}
    pipeline._cached_bootstrap_quadrant_models(returns, macro, **unseeded)
    pipeline._cached_bootstrap_quadrant_models(returns, macro, **unseeded)
    assert len(calls) == 7
    pipeline._clear_parameter_model_cache()


def test_pipeline_reports_when_walk_forward_validation_is_unavailable():
    scenario = run_scenario(**_scenario_kwargs())

    assert scenario.walk_forward is None
    assert any(
        "Walk-forward validation unavailable" in warning
        for warning in scenario.diagnostics.warnings
    )


def test_pipeline_applies_macro_lag_and_normalizes_asset_names():
    dates = pd.date_range("2020-01-31", periods=48, freq="ME")
    macro = pd.DataFrame(
        {
            "growth": np.tile([2.0, 2.5, -1.0, -1.5], 12),
            "inflation": np.tile([1.0, 4.0, 4.5, 1.2], 12),
        },
        index=dates,
    )
    returns = pd.DataFrame(
        {
            "Stocks": np.linspace(-0.03, 0.04, len(dates)),
            "Bonds": np.linspace(0.02, -0.01, len(dates)),
        },
        index=dates,
    )
    fx_rates = pd.DataFrame({"EUR": np.linspace(1.05, 1.15, len(dates))}, index=dates)

    scenario = run_scenario(
        returns=returns,
        macro=macro,
        selected_tickers=["STOCKS", "BONDS"],
        growth_col="growth",
        inflation_col="inflation",
        growth_threshold=0.0,
        inflation_threshold=3.0,
        periods=4,
        paths=8,
        random_seed=3,
        start_state=None,
        weights={"STOCKS": 0.6, "BONDS": 0.4},
        macro_lag_periods=1,
        transition_uncertainty=0.2,
        distribution="mnts",
        rebalance_frequency=1,
        transaction_cost_bps=10,
        base_currency="USD",
        asset_currencies={"Stocks": "EUR"},
        fx_rates=fx_rates,
    )

    assert scenario.result.assets == ["Stocks", "Bonds"]
    assert scenario.result.returns.shape == (4, 8, 2)
    assert scenario.model.metadata["macro_lag_periods"] == 1
    assert scenario.result.transition_concentration is not None
    assert scenario.model.metadata["base_currency"] == "USD"
    assert any("lagged" in warning for warning in scenario.diagnostics.warnings)
