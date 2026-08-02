from __future__ import annotations

import numpy as np
import pandas as pd

from mc_quadrants.calibration import calibrate_quadrant_model
from mc_quadrants.regimes import Regime
from mc_quadrants.simulation import simulate_portfolio_paths, simulate_returns, summarize_terminal_wealth


def _demo_history(seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("1990-01-31", periods=420, freq="ME")
    regimes = np.array(
        [
            Regime.HIGH_GROWTH_LOW_INFLATION.value,
            Regime.HIGH_GROWTH_HIGH_INFLATION.value,
            Regime.LOW_GROWTH_HIGH_INFLATION.value,
            Regime.LOW_GROWTH_LOW_INFLATION.value,
        ]
    )
    transition = np.array(
        [
            [0.86, 0.08, 0.01, 0.05],
            [0.14, 0.74, 0.09, 0.03],
            [0.03, 0.10, 0.77, 0.10],
            [0.12, 0.02, 0.08, 0.78],
        ]
    )
    state_index = np.empty(len(dates), dtype=int)
    state_index[0] = 0
    for i in range(1, len(dates)):
        state_index[i] = rng.choice(len(regimes), p=transition[state_index[i - 1]])

    growth_level = {
        Regime.HIGH_GROWTH_LOW_INFLATION.value: 3.0,
        Regime.HIGH_GROWTH_HIGH_INFLATION.value: 3.5,
        Regime.LOW_GROWTH_HIGH_INFLATION.value: -0.5,
        Regime.LOW_GROWTH_LOW_INFLATION.value: -1.0,
    }
    inflation_level = {
        Regime.HIGH_GROWTH_LOW_INFLATION.value: 1.8,
        Regime.HIGH_GROWTH_HIGH_INFLATION.value: 4.5,
        Regime.LOW_GROWTH_HIGH_INFLATION.value: 5.0,
        Regime.LOW_GROWTH_LOW_INFLATION.value: 1.0,
    }
    macro = pd.DataFrame(
        {
            "growth": [growth_level[regimes[i]] + rng.normal(0, 0.35) for i in state_index],
            "inflation": [inflation_level[regimes[i]] + rng.normal(0, 0.30) for i in state_index],
        },
        index=dates,
    )

    assets = [
        "Stocks",
        "Bonds",
        "Gold",
        "Commodities",
        "International Stocks",
        "Real Estate",
        "TIPS",
        "Short Treasuries",
    ]
    means = {
        Regime.HIGH_GROWTH_LOW_INFLATION.value: [0.008, 0.003, 0.002, 0.002, 0.006, 0.006, 0.0025, 0.0015],
        Regime.HIGH_GROWTH_HIGH_INFLATION.value: [0.006, -0.003, 0.005, 0.010, 0.004, 0.005, 0.0010, 0.0012],
        Regime.LOW_GROWTH_HIGH_INFLATION.value: [-0.006, -0.002, 0.008, 0.009, -0.005, -0.004, 0.0020, 0.0010],
        Regime.LOW_GROWTH_LOW_INFLATION.value: [-0.008, 0.007, 0.004, -0.004, -0.008, -0.006, 0.0040, 0.0015],
    }
    vols = np.array([0.045, 0.018, 0.040, 0.055, 0.050, 0.050, 0.025, 0.005])
    stock_bond_corr = {
        Regime.HIGH_GROWTH_LOW_INFLATION.value: -0.10,
        Regime.HIGH_GROWTH_HIGH_INFLATION.value: 0.35,
        Regime.LOW_GROWTH_HIGH_INFLATION.value: 0.25,
        Regime.LOW_GROWTH_LOW_INFLATION.value: -0.45,
    }
    factor_loadings = np.array(
        [
            [0.75, 0.15, 0.10],
            [0.00, 0.65, 0.10],
            [0.10, 0.20, 0.65],
            [0.45, 0.05, 0.35],
            [0.70, 0.10, 0.10],
            [0.55, 0.15, 0.25],
            [0.20, 0.55, 0.20],
            [0.05, 0.60, 0.05],
        ]
    )
    idiosyncratic_vol = np.array([0.35, 0.35, 0.45, 0.40, 0.40, 0.45, 0.40, 0.35])

    rows = []
    for regime in regimes[state_index]:
        loadings = factor_loadings.copy()
        loadings[1, 0] = stock_bond_corr[regime]
        raw_covariance = loadings @ loadings.T + np.diag(idiosyncratic_vol**2)
        raw_vols = np.sqrt(np.diag(raw_covariance))
        corr = raw_covariance / np.outer(raw_vols, raw_vols)
        cov = corr * np.outer(vols, vols)
        rows.append(rng.multivariate_normal(means[regime], cov))

    returns = pd.DataFrame(rows, index=dates, columns=assets)
    return macro, returns


def main() -> None:
    macro, returns = _demo_history()
    model = calibrate_quadrant_model(
        returns,
        macro,
        growth_col="growth",
        inflation_col="inflation",
        growth_threshold="median",
        inflation_threshold="median",
        correlation_overrides={
            Regime.HIGH_GROWTH_HIGH_INFLATION.value: {("Stocks", "Bonds"): 0.35},
            Regime.LOW_GROWTH_HIGH_INFLATION.value: {("Stocks", "Bonds"): 0.25},
            Regime.LOW_GROWTH_LOW_INFLATION.value: {("Stocks", "Bonds"): -0.40},
        },
        override_weight=0.40,
    )
    result = simulate_returns(
        model,
        periods=120,
        paths=3000,
        random_seed=7,
        distribution="student_t",
        degrees_of_freedom=5,
    )
    wealth = simulate_portfolio_paths(
        result,
        weights={
            "Stocks": 0.40,
            "Bonds": 0.20,
            "Gold": 0.10,
            "Commodities": 0.10,
            "International Stocks": 0.10,
            "Real Estate": 0.05,
            "TIPS": 0.03,
            "Short Treasuries": 0.02,
        },
        rebalance_frequency=1,
        transaction_cost_bps=10,
    )

    print("\nTransition matrix")
    print(model.transition_matrix.round(2))
    print("\nObservations by regime")
    print(pd.Series({state: moments.observations for state, moments in model.moments.items()}))
    print("\nTerminal wealth summary after 120 months")
    print(summarize_terminal_wealth(wealth).round(2))


if __name__ == "__main__":
    main()
