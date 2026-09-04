import io
import json

import numpy as np
import pandas as pd
import pytest

import mc_quadrants.api as api
from mc_quadrants.native import native_available
from mc_quadrants.types import SimulationResult

ASSET_TICKERS = ["SPY", "IEF", "GLD", "DBC", "EFA", "VNQ", "TIP", "SHY"]


def _csv_payload(**overrides):
    rng = np.random.default_rng(7)
    dates = pd.date_range("2010-01-31", periods=120, freq="ME")
    prices = pd.DataFrame(
        {
            ticker: 100 * np.exp(np.cumsum(rng.normal(0.004, 0.02, len(dates))))
            for ticker in ASSET_TICKERS
        },
        index=pd.Index(dates, name="Date"),
    )
    macro = pd.DataFrame(
        {
            "growth": rng.normal(0.0, 0.6, len(dates)),
            "inflation": rng.normal(0.0, 0.4, len(dates)),
            "interest_rate": np.clip(rng.normal(3.0, 1.0, len(dates)), 0.0, None),
        },
        index=pd.Index(dates, name="Date"),
    )
    payload = {
        "source": "csv",
        "csv_prices": prices.reset_index().to_csv(index=False),
        "csv_macro": macro.reset_index().to_csv(index=False),
        "asset_input": "Price levels",
        "monthly": True,
        "growth_col": "growth",
        "inflation_col": "inflation",
        "rate_col": "interest_rate",
        "selected_tickers": ["SPY", "IEF", "GLD", "DBC"],
        "weights": {"SPY": 40, "IEF": 20, "GLD": 10, "DBC": 10},
        "periods": 12,
        "paths": 50,
        "random_seed": 7,
        "initial_value": 100.0,
        "target_wealth": 200.0,
        "base_currency": "USD",
        "currency_map": "",
        "growth_threshold": "median",
        "inflation_threshold": "median",
        "macro_lag": 1,
        "transition_uncertainty": 0,
        "distribution": "mnts",
        "rebalance": "monthly",
        "cost_bps": 10,
        "start_state": "Stationary",
    }
    payload.update(overrides)
    return payload


def test_parse_tickers_handles_strings_and_lists():
    assert api.parse_tickers("spy, IEF, GLD, GLD") == ["SPY", "IEF", "GLD"]
    assert api.parse_tickers(["spy", "IEF"]) == ["SPY", "IEF"]


def test_parse_pair_map_rejects_invalid_pairs():
    assert api.parse_pair_map("efa:euro, GLD:USD", "currency") == {"EFA": "EURO", "GLD": "USD"}
    with pytest.raises(ValueError, match="Invalid currency"):
        api.parse_pair_map("EFA", "currency")


def test_default_selected_tickers_prefers_stitched_series():
    assert api.default_selected_tickers(["IEF", "IEF_SIM", "IEFSIM"]) == ["IEFSIM"]
    assert api.default_selected_tickers(["SPY", "IEF"]) == ["SPY", "IEF"]


def test_correlation_overrides_helper():
    overrides, blend = api.correlation_overrides({}, ["SPY", "IEF"])
    assert overrides is None and blend == 1.0

    payload = {
        "use_correlation_override": True,
        "correlation_blend": 0.4,
        "correlation_override_targets": {"high_growth_low_inflation": -0.2},
    }
    overrides, blend = api.correlation_overrides(payload, ["SPY", "IEF"])
    assert blend == 0.4
    assert overrides["high_growth_low_inflation"] == {("SPY", "IEF"): -0.2}
    assert overrides["low_growth_low_inflation"] == {("SPY", "IEF"): -0.40}

    overrides, blend = api.correlation_overrides(payload, ["SPY"])
    assert overrides is None and blend == 1.0
    with pytest.raises(ValueError, match="between -1 and 1"):
        api.correlation_overrides(
            {
                "use_correlation_override": True,
                "correlation_override_targets": {"high_growth_low_inflation": 2.0},
            },
            ["SPY", "IEF"],
        )


def test_load_csv_source():
    macro, returns, tickers, growth_col, inflation_col, message = api.load_data_source(
        _csv_payload()
    )
    assert len(tickers) == 8
    assert list(returns.columns) == tickers
    assert growth_col == "growth" and inflation_col == "inflation"
    assert "CSV" in message


def test_load_rejects_unknown_source():
    with pytest.raises(ValueError, match="Unknown data source"):
        api.load_data_source({"source": "unknown"})


def test_load_response_has_coverage_and_preview():
    macro, returns, tickers, growth_col, inflation_col, message = api.load_data_source(
        _csv_payload()
    )
    response = api.build_load_response(macro, returns, tickers, growth_col, inflation_col, message)
    assert response["ok"] is True
    assert response["coverage"]["SPY"]["first"] == "2010-02-28"
    assert response["macro"]["columns"] == ["Date", "growth", "inflation", "interest_rate"]
    assert response["rate_col"] == "interest_rate"
    assert response["returns"]["columns"] == ["Date"] + tickers


def test_load_response_includes_portfolio_presets():
    macro, returns, tickers, growth_col, inflation_col, message = api.load_data_source(
        _csv_payload()
    )
    response = api.build_load_response(macro, returns, tickers, growth_col, inflation_col, message)
    presets = response["presets"]
    assert len(presets) >= 5
    for preset in presets:
        assert "name" in preset and preset["weights"]
    assert sum(preset["weights"].values()) == pytest.approx(100.0)
    assert any(preset["name"] == "Classic 60/40" for preset in presets)
    assert "synthetic" in response
    assert response["synthetic"] == {}


def test_simulate_reports_real_terms_with_inflation():
    payload = _csv_payload(annual_inflation=2.5, risk_free_rate=1.0)
    response = api.build_simulate_response(payload)
    assert response["ok"] is True
    assert response["terms"] == "real"
    assert response["summary"]["sharpe_ratio"] == response["summary"]["sharpe_ratio"]
    assert response["resources"]["chunk_size"] > 0


def test_scenario_kwargs_include_long_term_fields():
    kwargs = api.scenario_kwargs(
        {
            "weights": {"SPY": 100},
            "risk_free_rate": 2.0,
            "annual_inflation": 3.0,
            "initial_value": 250.0,
        }
    )
    assert kwargs["risk_free_rate"] == pytest.approx(0.02)
    assert kwargs["annual_inflation"] == pytest.approx(0.03)
    assert kwargs["initial_value"] == pytest.approx(250.0)


def test_scenario_kwargs_include_periodic_cash_flows():
    kwargs = api.scenario_kwargs(
        {
            "weights": {"SPY": 100},
            "periods": 36,
            "contribution": 25.0,
            "withdrawal": 10.0,
            "withdrawal_start_period": 10,
        }
    )
    assert kwargs["contribution"] == pytest.approx(25.0)
    assert kwargs["withdrawal"] == pytest.approx(10.0)
    assert kwargs["withdrawal_start_period"] == 10


def test_scenario_kwargs_rejects_withdrawal_start_outside_horizon():
    with pytest.raises(ValueError, match="withdrawal_start_period"):
        api.scenario_kwargs(
            {
                "weights": {"SPY": 100},
                "periods": 12,
                "withdrawal_start_period": 13,
            }
        )


def test_scenario_kwargs_normalizes_advanced_decumulation():
    kwargs = api.scenario_kwargs(
        _csv_payload(
            periods=24,
            decumulation={
                "enabled": True,
                "mode": "manual",
                "policy": "guyton_klinger",
                "phases": [
                    {
                        "start_month": 2,
                        "end_month": 12,
                        "frequency": "quarterly",
                        "annual_real_amount": 1200,
                    }
                ],
                "one_time_expenses": [{"month": 6, "real_amount": 500}],
            },
        )
    )
    assert kwargs["decumulation"]["policy"]["type"] == "guyton_klinger"
    assert kwargs["decumulation"]["phases"][0]["frequency"] == "quarterly"


def test_safe_rate_solver_compares_policies_on_the_same_paths():
    payload = _csv_payload(
        paths=40,
        periods=12,
        decumulation={
            "enabled": True,
            "mode": "safe_rate",
            "policy": "guyton_klinger",
            "phases": [
                {
                    "start_month": 1,
                    "end_month": 12,
                    "frequency": "monthly",
                    "spending_multiplier": 1.0,
                }
            ],
            "safe_rate": {
                "objective": "survival",
                "target_probability": 0.90,
            },
        },
    )
    response = api.build_safe_rate_response(payload)
    policies = response["retirement"]["safe_rate"]["policies"]
    assert response["same_market_paths"] is True
    assert set(policies) == {"fixed", "guyton_klinger"}
    assert policies["fixed"]["curve"]
    assert len(policies["guyton_klinger"]["wilson_95"]) == 2
    assert {row["policy"] for row in response["retirement"]["paired_comparison"]} == {
        "fixed",
        "guyton_klinger",
    }


def test_scenario_kwargs_include_methodology_options():
    kwargs = api.scenario_kwargs(
        {
            "weights": {"SPY": 100},
            "model": "hmm",
            "hmm_states": 3,
            "threshold_window": 12,
            "duration_model": "semi_markov",
            "min_regime_duration": 5,
            "regime_smoothing_window": 3,
            "regime_hysteresis": 0.2,
            "regime_confirmation_periods": 2,
            "duration_prior_strength": 10,
            "garch": True,
            "garch_alpha": 0.2,
            "garch_beta": 0.7,
            "walk_forward": False,
        }
    )
    assert kwargs["model_kind"] == "hmm"
    assert kwargs["hmm_states"] == 3
    assert kwargs["threshold_window"] == 12
    assert kwargs["duration_model"] == "semi_markov"
    assert kwargs["min_regime_duration"] == 5
    assert kwargs["regime_smoothing_window"] == 3
    assert kwargs["regime_hysteresis"] == pytest.approx(0.2)
    assert kwargs["regime_confirmation_periods"] == 2
    assert kwargs["duration_prior_strength"] == pytest.approx(10)
    assert kwargs["garch"] is True
    assert kwargs["garch_alpha"] == pytest.approx(0.2)
    assert kwargs["walk_forward"] is False
    assert kwargs["start_state"] is None


def test_scenario_kwargs_reject_invalid_methodology_options():
    with pytest.raises(ValueError, match="model"):
        api.scenario_kwargs({"weights": {"SPY": 100}, "model": "black_box"})
    with pytest.raises(ValueError, match="duration model"):
        api.scenario_kwargs({"weights": {"SPY": 100}, "duration_model": "weibull"})
    with pytest.raises(ValueError, match="hmm_states"):
        api.scenario_kwargs({"weights": {"SPY": 100}, "hmm_states": 1})
    with pytest.raises(ValueError, match="garch_alpha"):
        api.scenario_kwargs({"weights": {"SPY": 100}, "garch": True, "garch_alpha": 0.5, "garch_beta": 0.6})
    with pytest.raises(ValueError, match="regime_smoothing_window"):
        api.scenario_kwargs({"weights": {"SPY": 100}, "regime_smoothing_window": 0})


def test_scenario_kwargs_reject_incompatible_settings():
    with pytest.raises(ValueError, match="Unknown return distribution"):
        api.scenario_kwargs(
            {"weights": {"SPY": 100}, "distribution": "legacy", "garch": True}
        )
    with pytest.raises(ValueError, match="transaction costs"):
        api.scenario_kwargs({"weights": {"SPY": 100}, "rebalance": "legacy", "cost_bps": 10})
    with pytest.raises(ValueError, match="transaction costs"):
        api.scenario_kwargs({"weights": {"SPY": 100}, "rebalance": "buy_hold", "cost_bps": 10})


def test_scenario_kwargs_support_true_buy_and_hold():
    kwargs = api.scenario_kwargs({"weights": {"SPY": 60, "IEF": 40}, "rebalance": "buy_hold"})

    assert kwargs["rebalance_frequency"] == 0
    assert kwargs["transaction_cost_bps"] == 0


def test_metric_formatter_respects_units():
    assert api.format_metric_value("probability_of_loss", 0.125) == "12.50%"
    assert api.format_metric_value("p50", 1234.5, "EUR") == "EUR 1,234.50"
    assert api.format_metric_value("target_wealth", 500.0, "EUR") == "EUR 500.00"
    assert api.format_metric_value("recovery_months_p95", 14.25) == "14.2 mo"
    assert api.format_metric_value("sharpe_ratio", 1.234) == "1.23"
    assert api.format_metric_value("annual_wealth_tax_rate", 0.002) == "0.20%"
    assert api.format_metric_value("taxes_paid", 42.5, "EUR") == "EUR 42.50"


def test_scenario_kwargs_reject_invalid_wealth_targets():
    with pytest.raises(ValueError, match="initial_value"):
        api.scenario_kwargs({"weights": {"SPY": 100}, "initial_value": 0})
    with pytest.raises(ValueError, match="target_wealth"):
        api.scenario_kwargs({"weights": {"SPY": 100}, "target_wealth": float("inf")})


def test_drawdown_duration_metrics_track_recovery_episodes():
    values = np.array(
        [
            [90.0, 110.0],
            [95.0, 90.0],
            [105.0, 100.0],
            [110.0, 120.0],
        ]
    )

    metrics = api._drawdown_duration_metrics(values, initial_value=100.0)

    assert metrics["max_underwater_months_mean"] == pytest.approx(2.0)
    assert metrics["max_underwater_months_p95"] == pytest.approx(2.0)
    assert metrics["recovery_months_median"] == pytest.approx(2.0)
    assert metrics["recovery_months_p95"] == pytest.approx(2.0)
    assert metrics["unrecovered_at_horizon"] == pytest.approx(0.0)


def test_drawdown_chart_analytics_handles_monotonic_and_recovery_paths():
    monotonic = api._drawdown_chart_analytics(
        np.array([[101.0], [102.0], [103.0]]),
        initial_value=100.0,
    )
    assert monotonic["drawdown_episodes"]["points"] == []

    recovered = api._drawdown_chart_analytics(
        np.array([[110.0], [90.0], [100.0], [120.0]]),
        initial_value=100.0,
    )
    point = recovered["drawdown_episodes"]["points"][0]
    assert point["path"] == 0
    assert point["duration_months"] == 2
    assert point["depth"] == pytest.approx(1.0 - 90.0 / 110.0)
    assert point["recovered"] is True


def test_goal_probability_curve_uses_terminal_distribution():
    curve = api._goal_probability_curve(
        np.array([50.0, 100.0, 150.0]),
        initial_value=100.0,
        target_wealth=125.0,
    )

    target_index = curve["targets"].index(125.0)
    initial_index = curve["targets"].index(100.0)
    assert curve["success_probability"][target_index] == pytest.approx(1 / 3)
    assert curve["success_probability"][initial_index] == pytest.approx(2 / 3)
    assert np.all(np.diff(curve["success_probability"]) <= 0)


def test_scenario_kwargs_parse_fees_and_leverage():
    kwargs = api.scenario_kwargs(
        {
            "weights": {"SPY": 100},
            "expense_ratios": "SPY:0.03",
            "leverage_multiple": 2.0,
            "financing_rate": 6.0,
            "financing_inflation_sensitivity": 1.0,
            "maintenance_margin": 25.0,
            "rebalance": "monthly",
        }
    )

    assert kwargs["asset_expense_ratios"] == {"SPY": pytest.approx(0.0003)}
    assert kwargs["leverage_multiple"] == pytest.approx(2.0)
    assert kwargs["financing_rate"] == pytest.approx(0.06)
    assert kwargs["financing_inflation_sensitivity"] == pytest.approx(1.0)
    assert kwargs["maintenance_margin"] == pytest.approx(0.25)


def test_scenario_kwargs_parse_italian_tax_settings():
    kwargs = api.scenario_kwargs(
        {
            "weights": {"SPY": 60, "BTP": 40},
            "rebalance": "quarterly",
            "tax_country": "IT",
            "tax_regime": "italy_administered",
            "asset_tax_categories": "SPY:FUND, BTP:GOVERNMENT_BOND",
            "italy_wealth_tax": 0.20,
            "tax_terminal_liquidation": False,
        }
    )

    assert kwargs["tax_country"] == "IT"
    assert kwargs["tax_regime"] == "italy_administered"
    assert kwargs["asset_tax_categories"] == {
        "SPY": "fund",
        "BTP": "government_bond",
    }
    assert kwargs["italy_annual_wealth_tax"] == pytest.approx(0.002)
    assert kwargs["tax_terminal_liquidation"] is False


def test_scenario_kwargs_keeps_legacy_regime_only_italy_selection_compatible():
    kwargs = api.scenario_kwargs(
        {
            "weights": {"SPY": 100},
            "rebalance": "monthly",
            "tax_regime": "italy_administered",
        }
    )

    assert kwargs["tax_country"] == "IT"
    assert kwargs["tax_regime"] == "italy_administered"


def test_scenario_kwargs_reject_invalid_italian_tax_settings():
    with pytest.raises(ValueError, match="tax category"):
        api.scenario_kwargs(
            {
                "weights": {"SPY": 100},
                "tax_regime": "italy_administered",
                "asset_tax_categories": "SPY:CRYPTO",
            }
        )
    with pytest.raises(ValueError, match="holdings-based"):
        api.scenario_kwargs(
            {"weights": {"SPY": 100}, "tax_regime": "italy_administered", "rebalance": "legacy"}
        )
    with pytest.raises(ValueError, match="leveraged"):
        api.scenario_kwargs(
            {
                "weights": {"SPY": 100},
                "tax_regime": "italy_administered",
                "rebalance": "monthly",
                "leverage_multiple": 2.0,
            }
        )


def test_scenario_kwargs_rejects_negative_inflation_sensitivity():
    with pytest.raises(ValueError, match="financing_inflation_sensitivity"):
        api.scenario_kwargs(
            {"weights": {"SPY": 100}, "rebalance": "monthly", "financing_inflation_sensitivity": -0.5}
        )


def test_simulate_response_reports_model_kind_and_validation():
    response = api.build_simulate_response(_csv_payload())
    assert response["model_kind"] == "quadrant"
    assert response["taxes"]["enabled"] is False
    assert response["gross_wealth"] is None
    validation = response["validation"]
    assert validation is not None
    assert validation["summary"]["splits"] > 0
    assert "advantage_mean" in validation["summary"]
    assert "predicted_switches_per_decade" in validation["summary"]
    assert validation["rows"]
    persistence = response["persistence"]
    assert persistence["expected_switches_per_decade"] >= 0
    assert len(persistence["states"]) == 4
    assert all(state["expected_months"] >= 5 for state in persistence["states"])


def test_simulate_reports_cash_flow_summary():
    payload = _csv_payload(
        contribution=20.0,
        withdrawal=5.0,
        withdrawal_start_period=5,
    )
    response = api.build_simulate_response(payload)
    assert response["ok"] is True
    assert response["summary"]["periodic_contribution"] == pytest.approx(20.0)
    assert response["summary"]["total_withdrawn"] == pytest.approx(5.0 * 8)
    assert "cash_flow_adjusted_annualized_return" in response["summary"]
    assert response["summary"]["periodic_withdrawal"] == pytest.approx(5.0)
    assert response["summary"]["withdrawal_start_period"] == 5
    assert response["summary"]["withdrawal_periods"] == 8
    assert response["summary"]["total_contributed"] == pytest.approx(20.0 * 12)


def test_simulate_reports_fee_and_leverage_assumptions():
    payload = _csv_payload(
        expense_ratios="SPY:0.03, IEF:0.15",
        leverage_multiple=1.5,
        financing_rate=6.0,
        maintenance_margin=20.0,
    )

    response = api.build_simulate_response(payload)

    assert response["costs"]["leverage_multiple"] == pytest.approx(1.5)
    assert response["costs"]["annual_financing_cost"] == pytest.approx(0.03)
    assert response["costs"]["weighted_expense_ratio"] > 0
    assert response["costs"]["effective_financing_rate"] == pytest.approx(0.06)


def test_simulate_reports_italian_tax_assumptions():
    response = api.build_simulate_response(
        _csv_payload(
            tax_country="IT",
            tax_regime="italy_administered",
            asset_tax_categories="SPY:FUND, IEF:GOVERNMENT_BOND",
            italy_wealth_tax=0.20,
        )
    )

    assert response["taxes"]["enabled"] is True
    assert response["taxes"]["country"] == "IT"
    assert response["taxes"]["regime"] == "italy_administered"
    assert response["taxes"]["available_countries"] == [
        {
            "code": "IT",
            "label": "Italy",
                "regimes": [
                    {"value": "italy_administered", "label": "Simplified administered regime"},
                    {"value": "italy_declarative", "label": "Declarative regime"},
                    {"value": "italy_managed", "label": "Managed regime"},
                ],
        }
    ]
    assert response["taxes"]["standard_rate"] == pytest.approx(0.26)
    assert response["taxes"]["government_bond_rate"] == pytest.approx(0.125)
    assert response["taxes"]["annual_wealth_tax_rate"] == pytest.approx(0.002)
    assert response["taxes"]["loss_carry_years"] == 4
    assert response["taxes"]["rule_snapshot"] == "IT-2026"
    assert response["costs"]["taxes_paid"] > 0
    assert response["costs"]["wealth_tax"] > 0
    assert response["costs"]["realized_gains"] >= 0
    assert response["costs"]["realized_losses"] >= 0
    assert response["costs"]["gross_terminal_wealth_median"] >= response["costs"][
        "after_tax_terminal_wealth_median"
    ]
    assert response["taxes"]["impact"]["terminal_drag_median"] > 0
    assert response["taxes"]["by_year"]
    assert sum(
        values.get("stamp_duty", 0.0)
        for values in response["taxes"]["by_year"].values()
    ) == pytest.approx(response["costs"]["stamp_duty"])
    assert response["gross_wealth"]["median"] != response["wealth"]["median"]
    assert any("planning approximation" in warning for warning in response["warnings"])
    assert any("IT-2026" in warning for warning in response["warnings"])


@pytest.mark.skipif(not native_available(), reason="native backend unavailable")
def test_large_simulate_response_uses_compact_native_histories_and_exact_terminal():
    response = api.build_simulate_response(
        _csv_payload(
            periods=6,
            paths=30_000,
            workers=4,
            walk_forward=False,
            tax_country="IT",
            tax_regime="italy_administered",
            rebalance="quarterly",
            annual_inflation=2.0,
        )
    )

    assert response["execution"] == {
        "native_backend": True,
        "compact_reporting": True,
        "simulated_paths": 30_000,
        "retained_history_paths": 25_000,
    }
    assert response["reporting_sample"]["total_paths"] == 30_000
    assert len(response["terminal"]) == api.MAX_REPORTING_PATHS
    assert response["wealth"]["median"][-1] == pytest.approx(response["summary"]["p50"])
    assert response["analytics_sample"]["sampled"] is True
    assert any("intermediate chart bands" in warning for warning in response["warnings"])


def test_simulate_reports_inflation_linked_financing():
    rng = np.random.default_rng(7)
    dates = pd.date_range("2010-01-31", periods=120, freq="ME")
    prices = pd.DataFrame(
        {ticker: 100 * np.exp(np.cumsum(rng.normal(0.004, 0.02, len(dates)))) for ticker in ASSET_TICKERS},
        index=pd.Index(dates, name="Date"),
    )
    macro = pd.DataFrame(
        {
            "growth": rng.normal(0.0, 0.6, len(dates)),
            "inflation": rng.normal(3.0, 0.4, len(dates)),
        },
        index=pd.Index(dates, name="Date"),
    )
    payload = _csv_payload(
        csv_prices=prices.reset_index().to_csv(index=False),
        csv_macro=macro.reset_index().to_csv(index=False),
        leverage_multiple=2.0,
        financing_rate=6.0,
        financing_inflation_sensitivity=1.0,
    )

    response = api.build_simulate_response(payload)

    assert response["costs"]["effective_financing_rate"] > 0.06
    assert response["costs"]["annual_financing_cost"] > 0.06
    assert response["summary"]["effective_financing_rate"] == pytest.approx(
        response["costs"]["effective_financing_rate"]
    )


def test_simulate_parallel_workers_match_sequential():
    seq = api.build_simulate_response(_csv_payload(periods=12, paths=100, chunk_size=25, workers=1))
    par = api.build_simulate_response(_csv_payload(periods=12, paths=100, chunk_size=25, workers=4))
    for key in ("mean", "p05", "p50", "p95", "annualized_return", "effective_financing_rate"):
        assert par["summary"][key] == pytest.approx(seq["summary"][key], abs=1e-12)
    assert np.array_equal(par["terminal"], seq["terminal"])
    assert par["regime_timelines"] == seq["regime_timelines"]


def test_scenario_kwargs_rejects_invalid_workers():
    with pytest.raises(ValueError, match="workers"):
        api.scenario_kwargs({"weights": {"SPY": 100}, "workers": 0})


def test_execution_plan_scales_with_scenario_dimensions():
    compact = api.simulation_resource_estimate(
        {"weights": {"SPY": 100}, "periods": 12, "paths": 1000, "workers": 1}
    )
    larger = api.simulation_resource_estimate(
        {
            "weights": {ticker: 1 for ticker in ASSET_TICKERS},
            "selected_tickers": ASSET_TICKERS,
            "periods": 120,
            "paths": 100000,
            "workers": 1,
        }
    )

    assert larger["work_units"] > compact["work_units"]
    assert larger["work_units"] == 120 * 100000 * len(ASSET_TICKERS)
    advanced = api.simulation_resource_estimate(
        {
            "weights": {ticker: 1 for ticker in ASSET_TICKERS},
            "selected_tickers": ASSET_TICKERS,
            "periods": 120,
            "paths": 100000,
            "workers": 1,
            "joint_macro": True,
            "dynamic_correlation": True,
        }
    )
    assert advanced["chunk_size"] == larger["chunk_size"]
    kwargs = api.scenario_kwargs(
        {
            "weights": {ticker: 1 for ticker in ASSET_TICKERS},
            "selected_tickers": ASSET_TICKERS,
            "periods": 360,
            "paths": 500000,
            "workers": 4,
        }
    )
    assert kwargs["paths"] == 500000

    with pytest.raises(ValueError, match="500,000"):
        api.scenario_kwargs({"weights": {"SPY": 100}, "paths": 500001})


def test_execution_plan_selects_workers_automatically(monkeypatch):
    monkeypatch.setattr(api.os, "cpu_count", lambda: 8)
    monkeypatch.setenv("MC_SIM_MAX_AUTO_WORKERS", "4")

    large = api.simulation_resource_estimate(
        {
            "weights": {ticker: 1 for ticker in ASSET_TICKERS},
            "selected_tickers": ASSET_TICKERS,
            "periods": 120,
            "paths": 100000,
        }
    )
    compact = api.simulation_resource_estimate(
        {"weights": {"SPY": 1}, "periods": 12, "paths": 1000}
    )

    assert large["workers"] == 4
    assert compact["workers"] == 1


def test_execution_plan_uses_full_native_tax_batch(monkeypatch):
    monkeypatch.setattr(api, "native_available", lambda: True)
    monkeypatch.setattr(api.os, "cpu_count", lambda: 10)
    monkeypatch.delenv("MC_SIM_MAX_AUTO_WORKERS", raising=False)
    payload = {
        "weights": {ticker: 1 for ticker in ASSET_TICKERS[:4]},
        "selected_tickers": ASSET_TICKERS[:4],
        "periods": 120,
        "paths": 500_000,
        "tax_country": "IT",
        "tax_regime": "italy_administered",
        "decumulation": {
            "enabled": False,
            "mode": "manual",
            "phases": [],
            "one_time_expenses": [],
        },
    }

    plan = api.simulation_resource_estimate(payload)

    assert plan["chunk_size"] == 500_000
    assert plan["workers"] == 8


def test_analytics_inputs_bound_diagnostics_and_slice_path_metadata():
    periods = 2
    paths = api.MAX_ANALYTICS_PATHS + 3
    values = np.arange(periods * paths, dtype=float).reshape(periods, paths)
    wealth = pd.DataFrame(values)
    wealth.attrs["withdrawal_funded"] = values + 1.0
    wealth.attrs["withdrawal_cpi"] = np.ones_like(values)
    result = SimulationResult(
        returns=np.empty((periods, 0, 1)),
        regimes=np.zeros((periods, paths), dtype=np.uint8),
        assets=["SPY"],
        states=["state"],
        frequency="M",
        macro_paths=np.zeros((periods, paths, 1)),
        macro_columns=["inflation"],
    )
    drawdowns = np.linspace(0.0, 1.0, paths)

    sampled_wealth, sampled_result, sampled_drawdowns, metadata = api._analytics_inputs(
        wealth, result, drawdowns
    )

    assert sampled_wealth.shape == (periods, api.MAX_ANALYTICS_PATHS)
    assert sampled_wealth.attrs["withdrawal_funded"].shape == sampled_wealth.shape
    assert sampled_result.regimes.shape == sampled_wealth.shape
    assert sampled_result.macro_paths.shape == (periods, api.MAX_ANALYTICS_PATHS, 1)
    assert sampled_drawdowns.shape == (api.MAX_ANALYTICS_PATHS,)
    assert metadata == {
        "paths": api.MAX_ANALYTICS_PATHS,
        "total_paths": paths,
        "sampled": True,
        "selection": "deterministic_even_spacing",
    }


def test_wealth_export_is_bounded_sample():
    response = api.build_wealth_csv(_csv_payload(paths=50, export_paths=7))
    exported = pd.read_csv(io.StringIO(response["csv"]))

    assert response["exported_paths"] == 7
    assert response["requested_paths"] == 50
    assert response["replayed_paths"] == 50
    assert response["sampled"] is True
    assert exported.shape == (12, 8)  # period plus seven sampled paths


def test_simulate_inflation_linked_financing_needs_leverage_state_inflation():
    payload = _csv_payload(
        leverage_multiple=2.0,
        financing_rate=6.0,
        financing_inflation_sensitivity=1.0,
    )
    payload["model"] = "hmm"
    response = api.build_simulate_response(payload)
    assert response["costs"]["effective_financing_rate"] == pytest.approx(0.06)


def test_simulate_csv_returns_full_result():
    response = api.build_simulate_response(_csv_payload())
    assert response["ok"] is True
    summary = response["summary"]
    for key in ("mean", "p05", "p50", "p95", "annualized_return", "annualized_volatility", "sharpe_ratio"):
        assert key in summary
    assert response["wealth"]["periods"] == list(range(1, 13))
    assert len(response["monthly_returns"]) == 12
    assert np.isfinite(response["monthly_returns"]).all()
    assert len(response["terminal"]) == 50
    assert response["reporting_sample"] == {
        "paths": 50,
        "total_paths": 50,
        "sampled": False,
    }
    assert len(response["transition"]["values"]) == 4
    assert response["currency"] == "USD"
    assert response["selected_tickers"] == ["SPY", "IEF", "GLD", "DBC"]


def test_reporting_indices_bound_browser_payload_and_preserve_endpoints():
    indices = api._reporting_indices(120_000)

    assert len(indices) == api.MAX_REPORTING_PATHS
    assert indices[0] == 0
    assert indices[-1] == 119_999


def test_simulate_response_separates_model_and_market_uncertainty():
    response = api.build_simulate_response(
        _csv_payload(
            paths=24,
            distribution="mnts",
            probabilistic_regimes=True,
            mean_prior_strength=24,
            parameter_draws=3,
            parameter_block_size=8,
            joint_macro=True,
            dynamic_correlation=True,
            walk_forward=False,
        )
    )

    assert response["parameter_uncertainty"]["draws"] == 3
    assert response["macro_paths"]["series"]["inflation"]
    assert response["methodology"]["regime_assignment"] == "probabilistic"
    assert response["methodology"]["transition_estimator"] == "hsmm_forward_backward_joint_posteriors"
    assert response["methodology"]["duration_model_kind"] == "hidden_semi_markov_explicit_duration"
    assert np.isfinite(response["methodology"]["hsmm_log_likelihood"])
    assert response["methodology"]["hsmm_iterations"] >= 1
    assert response["methodology"]["hsmm_max_duration"] >= 5
    assert response["methodology"]["joint_macro"] is True
    assert response["methodology"]["dynamic_correlation"] is True
    assert response["terms"] == "real"
    assert sum(item["probability"] for item in response["regime_probabilities"]) == pytest.approx(1.0)


def test_simulate_reports_regime_timelines_for_percentile_paths():
    payload = _csv_payload(periods=24, paths=100)
    response = api.build_simulate_response(payload)
    timelines = response["regime_timelines"]
    assert set(timelines) == {"p05", "median", "p95"}
    for label, timeline in timelines.items():
        assert len(timeline) == 24
        assert all(isinstance(state, str) and state for state in timeline)
    assert timelines["p05"] != timelines["p95"]
    assert response["regime_timeline"] == timelines["median"]


def test_simulate_reports_decision_analytics_and_sequence_risk():
    response = api.build_simulate_response(
        _csv_payload(
            periods=24,
            paths=80,
            contribution=10.0,
            target_wealth=250.0,
            walk_forward=False,
        )
    )

    assert set(response["success"]) == {"periods", "survival", "preservation", "profit", "target"}
    assert len(response["success"]["survival"]) == 24
    assert all(0 <= probability <= 1 for probability in response["success"]["profit"])
    assert all(0 <= probability <= 1 for probability in response["success"]["target"])
    decision = response["decision_metrics"]
    assert decision["target_wealth"] == pytest.approx(250.0)
    assert 0 <= decision["goal_success_probability"] <= 1
    assert decision["expected_goal_shortfall"] >= 0
    assert 0 <= decision["risk_of_ruin"] <= 1
    assert decision["omega_ratio"] >= 0
    assert 0 <= decision["max_underwater_months_p95"] <= 24
    assert 0 <= decision["recovery_months_p95"] <= 24
    assert -1 <= decision["worst_rolling_return_p05"]
    assert response["summary"]["goal_success_probability"] == pytest.approx(
        decision["goal_success_probability"]
    )
    assert len(response["drawdown_fan"]["periods"]) == 24
    assert len(response["drawdown_fan"]["p05"]) == 24
    assert len(response["recovery_required"]["median"]) == 24
    assert response["drawdown_episodes"]["source_paths"] == 80
    assert isinstance(response["drawdown_episodes"]["points"], list)
    assert response["rolling_horizons"]["months"] == [12, 24]
    assert len(response["rolling_horizons"]["p05"]) == 2
    assert response["goal_curve"]["targets"]
    assert len(response["goal_curve"]["targets"]) == len(
        response["goal_curve"]["success_probability"]
    )
    assert {scenario["label"] for scenario in response["representative_scenarios"]} == {
        "worst",
        "p05",
        "median",
        "p95",
        "best",
    }
    assert len(response["representative_scenarios"][0]["wealth"]) == 24
    assert "terminal_wealth" in response["metric_distributions"]
    assert "geometric_annualized_return" in response["metric_distributions"]
    assert response["sequence_risk"] is not None
    assert len(response["sequence_risk"]["points"]) == 80


def test_simulate_csv_with_correlation_overrides():
    payload = _csv_payload(
        use_correlation_override=True,
        correlation_blend=0.4,
        correlation_override_targets={"high_growth_high_inflation": 0.30},
    )
    response = api.build_simulate_response(payload)
    assert response["ok"] is True
    assert response["summary"]["mean"] > 0


def test_simulate_rejects_missing_weights():
    payload = _csv_payload(weights={})
    with pytest.raises(ValueError, match="weight"):
        api.build_simulate_response(payload)


def test_simulate_rejects_unknown_source():
    payload = _csv_payload(source="unknown")
    with pytest.raises(ValueError, match="Unknown data source"):
        api.build_simulate_response(payload)


def test_simulate_rejects_missing_selected_tickers():
    payload = _csv_payload(selected_tickers=[])
    with pytest.raises(ValueError, match="at least one ticker"):
        api.build_simulate_response(payload)


def test_csv_source_load_and_simulate():
    prices_csv = (
        "Date,SPY,BONDS\n2020-01-31,100,50\n2020-02-29,110,51\n2020-03-31,120,52\n2020-04-30,115,53\n"
    )
    macro_csv = "Date,growth,inflation\n2020-01-31,2.0,1.0\n2020-02-29,2.5,4.0\n2020-03-31,-1.0,4.5\n2020-04-30,-1.5,1.2\n"
    payload = _csv_payload(
        csv_prices=prices_csv,
        csv_macro=macro_csv,
        selected_tickers=["SPY", "BONDS"],
        weights={"SPY": 60, "BONDS": 40},
        periods=3,
    )
    response = api.build_simulate_response(payload)
    assert response["ok"] is True
    assert response["selected_tickers"] == ["SPY", "BONDS"]
    assert len(response["wealth"]["periods"]) == 3


def test_csv_simple_returns_are_compounded_monthly_and_converted_to_log_returns():
    payload = _csv_payload(
        asset_input="Simple returns",
        monthly=True,
        csv_prices=(
            "Date,SPY\n"
            "2020-01-01,0.10\n"
            "2020-01-31,-0.05\n"
            "2020-02-29,0.02\n"
        ),
    )

    _, returns, _, _, _, _ = api.load_data_source(payload)

    assert returns.loc[pd.Timestamp("2020-01-31"), "SPY"] == pytest.approx(np.log(1.10 * 0.95))


def test_threshold_value_parsing():
    assert api._threshold_value("fixed:2.5") == 2.5
    assert api._threshold_value("median") == "median"
    assert api._threshold_value(1.5) == 1.5


def test_simulate_response_is_json_serializable():
    response = api.build_simulate_response(_csv_payload())
    json.dumps(response)
    assert np.isfinite(response["summary"]["mean"])
