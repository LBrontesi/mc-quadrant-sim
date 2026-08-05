import json

import numpy as np
import pytest

import web_app

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
    assert web_app.parse_tickers("spy, IEF, GLD, GLD") == ["SPY", "IEF", "GLD"]
    assert web_app.parse_tickers(["spy", "IEF"]) == ["SPY", "IEF"]


def test_parse_pair_map_rejects_invalid_pairs():
    assert web_app.parse_pair_map("efa:euro, GLD:USD", "currency") == {"EFA": "EURO", "GLD": "USD"}
    with pytest.raises(ValueError, match="Invalid currency"):
        web_app.parse_pair_map("EFA", "currency")


def test_default_selected_tickers_prefers_stitched_series():
    assert web_app.default_selected_tickers(["IEF", "IEF_SIM", "IEFSIM"]) == ["IEFSIM"]
    assert web_app.default_selected_tickers(["SPY", "IEF"]) == ["SPY", "IEF"]


def test_correlation_overrides_helper():
    overrides, blend = web_app.correlation_overrides({}, ["SPY", "IEF"])
    assert overrides is None and blend == 1.0

    payload = {
        "use_correlation_override": True,
        "correlation_blend": 0.4,
        "correlation_override_targets": {"high_growth_low_inflation": -0.2},
    }
    overrides, blend = web_app.correlation_overrides(payload, ["SPY", "IEF"])
    assert blend == 0.4
    assert overrides["high_growth_low_inflation"] == {("SPY", "IEF"): -0.2}
    assert overrides["low_growth_low_inflation"] == {("SPY", "IEF"): -0.40}

    overrides, blend = web_app.correlation_overrides(payload, ["SPY"])
    assert overrides is None and blend == 1.0
    with pytest.raises(ValueError, match="between -1 and 1"):
        web_app.correlation_overrides({"use_correlation_override": True, "correlation_override_targets": {"high_growth_low_inflation": 2.0}}, ["SPY", "IEF"])


def test_load_demo_source():
    macro, returns, tickers, growth_col, inflation_col, message = web_app.load_data_source({"source": "demo", "seed": 7})
    assert len(tickers) == 8
    assert list(returns.columns) == tickers
    assert growth_col == "growth" and inflation_col == "inflation"
    assert "demo" in message


def test_load_response_has_coverage_and_preview():
    macro, returns, tickers, growth_col, inflation_col, message = web_app.load_data_source({"source": "demo", "seed": 7})
    response = web_app.build_load_response(macro, returns, tickers, growth_col, inflation_col, message)
    assert response["ok"] is True
    assert response["coverage"]["SPY"]["first"] == "1990-01-31"
    assert response["macro"]["columns"] == ["Date", "growth", "inflation"]
    assert response["returns"]["columns"] == ["Date"] + tickers


def test_load_response_includes_portfolio_presets():
    macro, returns, tickers, growth_col, inflation_col, message = web_app.load_data_source({"source": "demo", "seed": 7})
    response = web_app.build_load_response(macro, returns, tickers, growth_col, inflation_col, message)
    presets = response["presets"]
    assert len(presets) >= 5
    for preset in presets:
        assert "name" in preset and preset["weights"]
        assert sum(preset["weights"].values()) == pytest.approx(100.0)
    assert any(preset["name"] == "Classic 60/40" for preset in presets)


def test_simulate_reports_real_terms_with_inflation():
    payload = dict(DEMO_PAYLOAD)
    payload["annual_inflation"] = 2.5
    payload["risk_free_rate"] = 1.0
    response = web_app.build_simulate_response(payload)
    assert response["ok"] is True
    assert response["terms"] == "real"
    assert response["summary"]["sharpe_ratio"] == response["summary"]["sharpe_ratio"]


def test_scenario_kwargs_include_long_term_fields():
    kwargs = web_app.scenario_kwargs(
        {"weights": {"SPY": 100}, "risk_free_rate": 2.0, "annual_inflation": 3.0}
    )
    assert kwargs["risk_free_rate"] == pytest.approx(0.02)
    assert kwargs["annual_inflation"] == pytest.approx(0.03)


def test_simulate_demo_returns_full_result():
    response = web_app.build_simulate_response(dict(DEMO_PAYLOAD))
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
    response = web_app.build_simulate_response(payload)
    assert response["ok"] is True
    assert response["summary"]["mean"] > 0


def test_compare_demo_returns_two_rows():
    payload = dict(DEMO_PAYLOAD)
    payload["periods"] = 12
    payload["paths"] = 30
    response = web_app.build_compare_response(payload)
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
        web_app.build_simulate_response(payload)


def test_simulate_rejects_unknown_source():
    payload = dict(DEMO_PAYLOAD)
    payload["source"] = "unknown"
    with pytest.raises(ValueError, match="Unknown data source"):
        web_app.build_simulate_response(payload)


def test_simulate_rejects_missing_selected_tickers():
    payload = dict(DEMO_PAYLOAD)
    payload["selected_tickers"] = []
    with pytest.raises(ValueError, match="at least one ticker"):
        web_app.build_simulate_response(payload)


def test_csv_source_load_and_simulate():
    prices_csv = "Date,SPY,BONDS\n2020-01-31,100,50\n2020-02-29,110,51\n2020-03-31,120,52\n2020-04-30,115,53\n"
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
    response = web_app.build_simulate_response(payload)
    assert response["ok"] is True
    assert response["selected_tickers"] == ["SPY", "BONDS"]
    assert len(response["wealth"]["periods"]) == 3


def test_threshold_value_parsing():
    assert web_app._threshold_value("fixed:2.5") == 2.5
    assert web_app._threshold_value("median") == "median"
    assert web_app._threshold_value(1.5) == 1.5


def test_simulate_response_is_json_serializable():
    response = web_app.build_simulate_response(dict(DEMO_PAYLOAD))
    json.dumps(response)
    assert np.isfinite(response["summary"]["mean"])
