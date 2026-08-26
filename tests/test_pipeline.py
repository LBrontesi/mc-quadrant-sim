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
