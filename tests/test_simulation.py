import numpy as np
import pandas as pd
import pytest

from mc_quadrants.calibration import calibrate_quadrant_model
from mc_quadrants.regimes import Regime
from mc_quadrants.simulation import simulate_portfolio_paths, simulate_returns
from mc_quadrants.types import SimulationResult


def _calibrated_model():
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
    return calibrate_quadrant_model(
        returns,
        macro,
        growth_threshold=0.0,
        inflation_threshold=3.0,
        min_observations=3,
        correlation_overrides={
            Regime.HIGH_GROWTH_HIGH_INFLATION.value: {("Stocks", "Bonds"): 0.30},
            Regime.LOW_GROWTH_LOW_INFLATION.value: {("Stocks", "Bonds"): -0.30},
        },
        override_weight=0.50,
    )


def test_calibration_and_simulation_shapes():
    model = _calibrated_model()

    result = simulate_returns(model, periods=6, paths=10, random_seed=1)
    wealth = simulate_portfolio_paths(result, {"Stocks": 0.6, "Bonds": 0.4})

    assert result.returns.shape == (6, 10, 2)
    assert result.regimes.shape == (6, 10)
    assert wealth.shape == (6, 10)


def test_student_t_sampling_is_reproducible():
    model = _calibrated_model()

    first = simulate_returns(
        model,
        periods=6,
        paths=10,
        random_seed=1,
        distribution="student_t",
        degrees_of_freedom=5,
    )
    second = simulate_returns(
        model,
        periods=6,
        paths=10,
        random_seed=1,
        distribution="student_t",
        degrees_of_freedom=5,
    )

    assert np.array_equal(first.returns, second.returns)
    assert first.distribution == "student_t"
    assert first.degrees_of_freedom == 5.0

    with pytest.raises(ValueError, match="greater than 2"):
        simulate_returns(model, periods=1, paths=1, distribution="student_t", degrees_of_freedom=2)


def test_rebalancing_transaction_costs_reduce_wealth():
    result = SimulationResult(
        returns=np.array(
            [
                [[0.10, 0.00]],
                [[0.00, 0.10]],
            ]
        ),
        regimes=np.empty((2, 1), dtype=object),
        assets=["Stocks", "Bonds"],
        states=[],
        frequency="M",
    )

    without_costs = simulate_portfolio_paths(
        result,
        {"Stocks": 0.5, "Bonds": 0.5},
        rebalance_frequency=1,
    )
    with_costs = simulate_portfolio_paths(
        result,
        {"Stocks": 0.5, "Bonds": 0.5},
        rebalance_frequency=1,
        transaction_cost_bps=100,
    )

    assert with_costs.iloc[-1, 0] < without_costs.iloc[-1, 0]
