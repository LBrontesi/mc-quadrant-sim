import pytest

pytest.importorskip("gradio")

import os  # noqa: E402

import gradio_app as app  # noqa: E402

BASE_ARGS = dict(
    state=None,
    tickers=["SPY", "IEF", "GLD", "DBC"],
    weights_table=[["SPY", 40.0], ["IEF", 20.0], ["GLD", 20.0], ["DBC", 20.0]],
    periods=12,
    paths=100,
    seed=7,
    start_state="Stationary",
    distribution="normal",
    degrees_of_freedom=5,
    block_size=3,
    rebalance="monthly",
    cost_bps=10,
    risk_free_rate=0.0,
    annual_inflation=0.0,
    base_currency="USD",
    currency_map="",
    use_corr_override=True,
    corr_blend=0.4,
    corr_growth_low=-0.1,
    corr_growth_high=0.35,
    corr_stagflation=0.25,
    corr_recession=-0.4,
    growth_threshold="median",
    inflation_threshold="median",
    macro_lag=1,
    transition_uncertainty=0.0,
)


def _loaded_args() -> dict:
    message, ticker_group, weights, presets, state = app.on_load(
        "demo", 42, "", "", "", "", [], 42, None, None, "Price levels", True, "growth", "inflation"
    )
    args = dict(BASE_ARGS)
    args["state"] = state
    args["tickers"] = ticker_group.get("choices")
    args["weights_table"] = weights.get("value")
    return args


def test_gradio_app_loads_demo_data():
    message, ticker_group, weights, presets, state = app.on_load(
        "demo", 42, "", "", "", "", [], 42, None, None, "Price levels", True, "growth", "inflation"
    )
    assert message.startswith("Loaded 8 tickers")
    assert ticker_group.get("choices") == ["SPY", "IEF", "GLD", "DBC", "EFA", "VNQ", "TIP", "SHY"]
    assert weights.get("value") == [["SPY", 40.0], ["IEF", 20.0], ["GLD", 10.0], ["DBC", 10.0], ["EFA", 10.0], ["VNQ", 5.0], ["TIP", 3.0], ["SHY", 2.0]]


def test_gradio_app_full_simulation_flow():
    out = app.on_run(**_loaded_args())
    status, metrics, *figures, diagnostics, _, results = out
    assert "Simulation complete" in status.get("value")
    assert "Mean terminal wealth" in metrics.get("value")
    assert len(figures) == 6
    assert diagnostics.value and diagnostics.headers
    assert results["selected_tickers"] == ["SPY", "IEF", "GLD", "DBC", "EFA", "VNQ", "TIP", "SHY"]


def test_gradio_app_compare_and_downloads():
    args = _loaded_args()
    comparison = app.on_compare(**args)
    assert comparison.value["headers"][0] == "distribution"
    assert len(comparison.value["data"]) == 2

    results = app.on_run(**args)[-1]
    wealth_path = app.download_wealth_paths(**args)
    assert os.path.exists(wealth_path)
    summary_path = app.download_summary(results)
    assert os.path.exists(summary_path)
    json_path = app.download_json(results)
    assert os.path.exists(json_path)
