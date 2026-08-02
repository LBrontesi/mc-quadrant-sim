import numpy as np
import pandas as pd

from mc_quadrants.pipeline import compare_distributions, run_scenario


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
        distribution="student_t",
        degrees_of_freedom=5,
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

    comparison = compare_distributions(
        {"Normal": "normal", "Student-t": "student_t"},
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
        rebalance_frequency=1,
        transaction_cost_bps=10,
    )
    assert comparison["distribution"].tolist() == ["Normal", "Student-t"]
    assert {"probability_of_loss", "worst_max_drawdown"}.issubset(comparison.columns)
