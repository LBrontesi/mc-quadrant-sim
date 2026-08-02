"""Example calibration using Yahoo Finance prices and FRED macro data.

Install optional dependencies first:

    python -m pip install -e ".[data]"
"""

from __future__ import annotations

import pandas as pd

from mc_quadrants.calibration import calibrate_quadrant_model
from mc_quadrants.data import fetch_fred_macro, fetch_yahoo_prices, prices_to_returns, yoy_change
from mc_quadrants.regimes import Regime
from mc_quadrants.simulation import simulate_portfolio_paths, simulate_returns, summarize_terminal_wealth


def main() -> None:
    tickers = ["SPY", "IEF", "GLD", "DBC", "EFA", "VNQ", "TIP", "SHY"]
    prices = fetch_yahoo_prices(tickers, start="2006-01-01")
    returns = prices_to_returns(prices, method="log").resample("ME").sum().dropna()

    raw_macro = fetch_fred_macro(
        {
            "industrial_production": "INDPRO",
            "cpi": "CPIAUCSL",
        },
        start="2005-01-01",
    )
    macro_yoy = yoy_change(raw_macro, periods=12)
    macro = pd.DataFrame(
        {
            "growth": macro_yoy["industrial_production"],
            "inflation": macro_yoy["cpi"],
        }
    ).dropna()
    macro = macro.resample("ME").last()

    model = calibrate_quadrant_model(
        returns=returns,
        macro=macro,
        growth_col="growth",
        inflation_col="inflation",
        growth_threshold="median",
        inflation_threshold="median",
        correlation_overrides={
            Regime.HIGH_GROWTH_HIGH_INFLATION.value: {("SPY", "IEF"): 0.30},
            Regime.LOW_GROWTH_HIGH_INFLATION.value: {("SPY", "IEF"): 0.25},
            Regime.LOW_GROWTH_LOW_INFLATION.value: {("SPY", "IEF"): -0.30},
        },
        override_weight=0.35,
    )

    result = simulate_returns(
        model,
        periods=120,
        paths=5000,
        random_seed=11,
        distribution="student_t",
        degrees_of_freedom=5,
    )
    wealth = simulate_portfolio_paths(
        result,
        weights={
            "SPY": 0.40,
            "IEF": 0.20,
            "GLD": 0.10,
            "DBC": 0.10,
            "EFA": 0.10,
            "VNQ": 0.05,
            "TIP": 0.03,
            "SHY": 0.02,
        },
        rebalance_frequency=1,
        transaction_cost_bps=10,
    )

    print("\nTransition matrix")
    print(model.transition_matrix.round(3))
    print("\nTerminal wealth summary")
    print(summarize_terminal_wealth(wealth).round(2))


if __name__ == "__main__":
    main()
