import json

import numpy as np
import pytest

import mc_quadrants.api as api

DEMO_PAYLOAD = {
    "source": "demo",
    "seed": 42,
    "selected_tickers": ["SPY", "IEF", "GLD", "DBC"],
    "weights": {"SPY": 40, "IEF": 20, "GLD": 10, "DBC": 10},
    "periods": 12,
    "paths": 50,
    "random_seed": 7,
    "base_currency": "USD",
    "currency_map": "",
    "growth_threshold": "median",
    "inflation_threshold": "median",
    "macro_lag": 1,
    "transition_uncertainty": 0,
    "distribution": "normal",
    "degrees_of_freedom": 5,
    "block_size": 3,
    "rebalance": "monthly",
    "cost_bps": 10,
    "start_state": "Stationary",
}


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


def test_load_demo_source():
    macro, returns, tickers, growth_col, inflation_col, message = api.load_data_source(
        {"source": "demo", "seed": 7}
    )
    assert len(tickers) == 8
    assert list(returns.columns) == tickers
    assert growth_col == "growth" and inflation_col == "inflation"
    assert "demo" in message


def test_load_response_has_coverage_and_preview():
    macro, returns, tickers, growth_col, inflation_col, message = api.load_data_source(
        {"source": "demo", "seed": 7}
    )
    response = api.build_load_response(macro, returns, tickers, growth_col, inflation_col, message)
    assert response["ok"] is True
    assert response["coverage"]["SPY"]["first"] == "1990-01-31"
    assert response["macro"]["columns"] == ["Date", "growth", "inflation"]
    assert response["returns"]["columns"] == ["Date"] + tickers


def test_load_response_includes_portfolio_presets():
    macro, returns, tickers, growth_col, inflation_col, message = api.load_data_source(
        {"source": "demo", "seed": 7}
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
    payload = dict(DEMO_PAYLOAD)
    payload["annual_inflation"] = 2.5
    payload["risk_free_rate"] = 1.0
    response = api.build_simulate_response(payload)
    assert response["ok"] is True
    assert response["terms"] == "real"
    assert response["summary"]["sharpe_ratio"] == response["summary"]["sharpe_ratio"]


def test_scenario_kwargs_include_long_term_fields():
    kwargs = api.scenario_kwargs({"weights": {"SPY": 100}, "risk_free_rate": 2.0, "annual_inflation": 3.0})
    assert kwargs["risk_free_rate"] == pytest.approx(0.02)
    assert kwargs["annual_inflation"] == pytest.approx(0.03)


def test_scenario_kwargs_include_periodic_cash_flows():
    kwargs = api.scenario_kwargs({"weights": {"SPY": 100}, "contribution": 25.0, "withdrawal": 10.0})
    assert kwargs["contribution"] == pytest.approx(25.0)
    assert kwargs["withdrawal"] == pytest.approx(10.0)


def test_scenario_kwargs_include_methodology_options():
    kwargs = api.scenario_kwargs(
        {
            "weights": {"SPY": 100},
            "model": "hmm",
            "hmm_states": 3,
            "threshold_window": 12,
            "duration_model": "semi_markov",
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


def test_scenario_kwargs_reject_incompatible_settings():
    with pytest.raises(ValueError, match="Normal return distribution"):
        api.scenario_kwargs(
            {"weights": {"SPY": 100}, "distribution": "student_t", "garch": True}
        )
    with pytest.raises(ValueError, match="transaction costs"):
        api.scenario_kwargs({"weights": {"SPY": 100}, "rebalance": "legacy", "cost_bps": 10})


def test_metric_formatter_respects_units():
    assert api.format_metric_value("probability_of_loss", 0.125) == "12.50%"
    assert api.format_metric_value("p50", 1234.5, "EUR") == "EUR 1,234.50"
    assert api.format_metric_value("sharpe_ratio", 1.234) == "1.23"


def test_scenario_kwargs_parse_fees_and_leverage():
    kwargs = api.scenario_kwargs(
        {
            "weights": {"SPY": 100},
            "expense_ratios": "SPY:0.03",
            "leverage_multiple": 2.0,
            "financing_rate": 6.0,
            "maintenance_margin": 25.0,
            "rebalance": "monthly",
        }
    )

    assert kwargs["asset_expense_ratios"] == {"SPY": pytest.approx(0.0003)}
    assert kwargs["leverage_multiple"] == pytest.approx(2.0)
    assert kwargs["financing_rate"] == pytest.approx(0.06)
    assert kwargs["maintenance_margin"] == pytest.approx(0.25)


def test_simulate_response_reports_model_kind_and_validation():
    response = api.build_simulate_response(dict(DEMO_PAYLOAD))
    assert response["model_kind"] == "quadrant"
    validation = response["validation"]
    assert validation is not None
    assert validation["summary"]["splits"] > 0
    assert "advantage_mean" in validation["summary"]
    assert validation["rows"]


def test_simulate_reports_cash_flow_summary():
    payload = dict(DEMO_PAYLOAD)
    payload["contribution"] = 20.0
    payload["withdrawal"] = 5.0
    response = api.build_simulate_response(payload)
    assert response["ok"] is True
    assert response["summary"]["periodic_contribution"] == pytest.approx(20.0)
    assert response["summary"]["total_withdrawn"] == pytest.approx(5.0 * 12)
    assert "cash_flow_adjusted_annualized_return" in response["summary"]
    assert response["summary"]["periodic_withdrawal"] == pytest.approx(5.0)
    assert response["summary"]["total_contributed"] == pytest.approx(20.0 * 12)


def test_simulate_reports_fee_and_leverage_assumptions():
    payload = dict(DEMO_PAYLOAD)
    payload.update(
        {
            "expense_ratios": "SPY:0.03, IEF:0.15",
            "leverage_multiple": 1.5,
            "financing_rate": 6.0,
            "maintenance_margin": 20.0,
        }
    )

    response = api.build_simulate_response(payload)

    assert response["costs"]["leverage_multiple"] == pytest.approx(1.5)
    assert response["costs"]["annual_financing_cost"] == pytest.approx(0.03)
    assert response["costs"]["weighted_expense_ratio"] > 0


def test_simulate_demo_returns_full_result():
    response = api.build_simulate_response(dict(DEMO_PAYLOAD))
    assert response["ok"] is True
    summary = response["summary"]
    for key in ("mean", "p05", "p50", "p95", "annualized_return", "annualized_volatility", "sharpe_ratio"):
        assert key in summary
    assert response["wealth"]["periods"] == list(range(1, 13))
    assert len(response["terminal"]) == 50
    assert len(response["transition"]["values"]) == 4
    assert response["currency"] == "USD"
    assert response["selected_tickers"] == ["SPY", "IEF", "GLD", "DBC"]


def test_simulate_demo_with_correlation_overrides():
    payload = dict(DEMO_PAYLOAD)
    payload["use_correlation_override"] = True
    payload["correlation_blend"] = 0.4
    payload["correlation_override_targets"] = {"high_growth_high_inflation": 0.30}
    response = api.build_simulate_response(payload)
    assert response["ok"] is True
    assert response["summary"]["mean"] > 0


def test_compare_demo_returns_two_rows():
    payload = dict(DEMO_PAYLOAD)
    payload["periods"] = 12
    payload["paths"] = 30
    response = api.build_compare_response(payload)
    assert response["ok"] is True
    assert response["columns"] == [
        "distribution",
        "mean",
        "p05",
        "median",
        "p95",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "probability_of_loss",
        "var_95",
        "expected_shortfall_95",
        "worst_max_drawdown",
        "ulcer_index_mean",
        "sortino_ratio",
        "calmar_ratio",
        "geometric_annualized_return",
    ]
    assert [row[0] for row in response["rows"]] == ["Normal", "Student-t"]


def test_simulate_rejects_missing_weights():
    payload = dict(DEMO_PAYLOAD)
    payload["weights"] = {}
    with pytest.raises(ValueError, match="weight"):
        api.build_simulate_response(payload)


def test_simulate_rejects_unknown_source():
    payload = dict(DEMO_PAYLOAD)
    payload["source"] = "unknown"
    with pytest.raises(ValueError, match="Unknown data source"):
        api.build_simulate_response(payload)


def test_simulate_rejects_missing_selected_tickers():
    payload = dict(DEMO_PAYLOAD)
    payload["selected_tickers"] = []
    with pytest.raises(ValueError, match="at least one ticker"):
        api.build_simulate_response(payload)


def test_csv_source_load_and_simulate():
    prices_csv = (
        "Date,SPY,BONDS\n2020-01-31,100,50\n2020-02-29,110,51\n2020-03-31,120,52\n2020-04-30,115,53\n"
    )
    macro_csv = "Date,growth,inflation\n2020-01-31,2.0,1.0\n2020-02-29,2.5,4.0\n2020-03-31,-1.0,4.5\n2020-04-30,-1.5,1.2\n"
    payload = dict(DEMO_PAYLOAD)
    payload.update(
        {
            "source": "csv",
            "csv_prices": prices_csv,
            "csv_macro": macro_csv,
            "asset_input": "Price levels",
            "monthly": True,
            "growth_col": "growth",
            "inflation_col": "inflation",
            "selected_tickers": ["SPY", "BONDS"],
            "weights": {"SPY": 60, "BONDS": 40},
            "periods": 3,
        }
    )
    response = api.build_simulate_response(payload)
    assert response["ok"] is True
    assert response["selected_tickers"] == ["SPY", "BONDS"]
    assert len(response["wealth"]["periods"]) == 3


def test_threshold_value_parsing():
    assert api._threshold_value("fixed:2.5") == 2.5
    assert api._threshold_value("median") == "median"
    assert api._threshold_value(1.5) == 1.5


def test_simulate_response_is_json_serializable():
    response = api.build_simulate_response(dict(DEMO_PAYLOAD))
    json.dumps(response)
    assert np.isfinite(response["summary"]["mean"])
