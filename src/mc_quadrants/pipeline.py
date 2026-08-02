from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

from mc_quadrants.calibration import CorrelationOverrides, calibrate_quadrant_model
from mc_quadrants.data import convert_returns_to_base_currency
from mc_quadrants.diagnostics import CalibrationDiagnostics, build_calibration_diagnostics
from mc_quadrants.regimes import classify_quadrants
from mc_quadrants.simulation import simulate_portfolio_paths, simulate_returns, summarize_wealth_risk
from mc_quadrants.types import ScenarioModel, SimulationResult


@dataclass(frozen=True)
class SimulationRun:
    """All model and output objects produced by one scenario run."""

    model: ScenarioModel
    regimes: pd.Series
    result: SimulationResult
    wealth: pd.DataFrame
    summary: pd.Series
    diagnostics: CalibrationDiagnostics


def run_scenario(
    returns: pd.DataFrame,
    macro: pd.DataFrame,
    selected_tickers: list[str],
    growth_col: str,
    inflation_col: str,
    growth_threshold: str | float,
    inflation_threshold: str | float,
    periods: int,
    paths: int,
    random_seed: int,
    start_state: str | None,
    weights: Mapping[str, float],
    correlation_overrides: CorrelationOverrides | None = None,
    override_weight: float = 1.0,
    macro_lag_periods: int = 0,
    distribution: str = "normal",
    degrees_of_freedom: float = 5.0,
    block_size: int = 3,
    transition_uncertainty: float = 0.0,
    rebalance_frequency: int | None = None,
    transaction_cost_bps: float = 0.0,
    initial_value: float = 100.0,
    base_currency: str = "USD",
    asset_currencies: Mapping[str, str] | None = None,
    fx_rates: pd.DataFrame | None = None,
    fx_quote: str = "base_per_foreign",
    risk_free_rate: float = 0.0,
    annual_inflation: float = 0.0,
) -> SimulationRun:
    """Calibrate and simulate one fully specified investment scenario."""

    transition_uncertainty = float(transition_uncertainty)
    if not 0 <= transition_uncertainty <= 1:
        raise ValueError("transition_uncertainty must be between 0 and 1.")
    macro_lag_periods = int(macro_lag_periods)
    if macro_lag_periods < 0:
        raise ValueError("macro_lag_periods must be non-negative.")
    available_columns = {str(column).strip().upper(): column for column in returns.columns}
    selected = [
        available_columns.get(str(ticker).strip().upper(), str(ticker).strip())
        for ticker in selected_tickers
    ]
    if not selected:
        raise ValueError("Select at least one ticker.")
    missing = [ticker for ticker in selected if ticker not in returns.columns]
    if missing:
        raise ValueError(f"Selected tickers are missing from returns: {', '.join(missing)}")
    normalized_weights = {
        available_columns.get(str(asset).strip().upper(), asset): weight
        for asset, weight in weights.items()
    }
    scenario_returns = convert_returns_to_base_currency(
        returns.loc[:, selected],
        asset_currencies=asset_currencies,
        base_currency=base_currency,
        fx_rates=fx_rates,
        fx_quote=fx_quote,
    )

    model = calibrate_quadrant_model(
        returns=scenario_returns,
        macro=macro,
        growth_col=growth_col,
        inflation_col=inflation_col,
        growth_threshold=growth_threshold,
        inflation_threshold=inflation_threshold,
        correlation_overrides=correlation_overrides,
        override_weight=override_weight,
        macro_lag_periods=macro_lag_periods,
    )
    regimes = classify_quadrants(
        macro,
        growth_col=growth_col,
        inflation_col=inflation_col,
        growth_threshold=growth_threshold,
        inflation_threshold=inflation_threshold,
    )
    result = simulate_returns(
        model,
        periods=int(periods),
        paths=int(paths),
        start_state=start_state,
        random_seed=int(random_seed),
        distribution=distribution,
        degrees_of_freedom=float(degrees_of_freedom),
        block_size=int(block_size),
        transition_concentration=(
            None
            if transition_uncertainty == 0.0
            else max(1.0, 1.0 / transition_uncertainty**2)
        ),
    )
    wealth = simulate_portfolio_paths(
        result,
        weights=normalized_weights,
        initial_value=initial_value,
        rebalance_frequency=rebalance_frequency,
        transaction_cost_bps=float(transaction_cost_bps),
    )
    diagnostics = build_calibration_diagnostics(
        model,
        scenario_returns,
        macro,
        growth_col,
        inflation_col,
        growth_threshold,
        inflation_threshold,
        macro_lag_periods=macro_lag_periods,
    )
    if transition_uncertainty > 0:
        diagnostics.warnings.append(
            f"Transition probabilities are sampled with uncertainty {transition_uncertainty:.2f}."
        )
    model.metadata["base_currency"] = str(base_currency).strip().upper()
    model.metadata["fx_quote"] = fx_quote
    model.metadata["initial_value"] = float(initial_value)
    if asset_currencies:
        model.metadata["asset_currencies"] = dict(asset_currencies)
    return SimulationRun(
        model=model,
        regimes=regimes,
        result=result,
        wealth=wealth,
        summary=summarize_wealth_risk(
            wealth,
            initial_value=initial_value,
            risk_free_rate=risk_free_rate,
            annual_inflation=annual_inflation,
        ),
        diagnostics=diagnostics,
    )


def compare_distributions(
    distributions: Mapping[str, str],
    **scenario_kwargs: Any,
) -> pd.DataFrame:
    """Run identical inputs under several return distributions."""

    rows: list[dict[str, Any]] = []
    for label, distribution in distributions.items():
        scenario = run_scenario(distribution=distribution, **scenario_kwargs)
        summary = scenario.summary
        rows.append(
            {
                "distribution": label,
                "mean": summary["mean"],
                "p05": summary["p05"],
                "median": summary["p50"],
                "p95": summary["p95"],
                "annualized_return": summary["annualized_return"],
                "annualized_volatility": summary["annualized_volatility"],
                "sharpe_ratio": summary["sharpe_ratio"],
                "probability_of_loss": summary["probability_of_loss"],
                "var_95": summary["var_95"],
                "expected_shortfall_95": summary["expected_shortfall_95"],
                "worst_max_drawdown": summary["max_drawdown_worst"],
            }
        )
    return pd.DataFrame(rows)
