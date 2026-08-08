from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

from mc_quadrants.calibration import CorrelationOverrides, calibrate_quadrant_model
from mc_quadrants.data import convert_returns_to_base_currency
from mc_quadrants.diagnostics import (
    CalibrationDiagnostics,
    build_calibration_diagnostics,
    build_hmm_diagnostics,
)
from mc_quadrants.hmm import fit_hmm_model
from mc_quadrants.regimes import classify_quadrants
from mc_quadrants.simulation import simulate_portfolio_paths, simulate_returns, summarize_wealth_risk
from mc_quadrants.types import ScenarioModel, SimulationResult
from mc_quadrants.validation import WalkForwardResult, walk_forward_validation


@dataclass(frozen=True)
class SimulationRun:
    """All model and output objects produced by one scenario run."""

    model: ScenarioModel
    regimes: pd.Series
    result: SimulationResult
    wealth: pd.DataFrame
    summary: pd.Series
    diagnostics: CalibrationDiagnostics
    walk_forward: WalkForwardResult | None = None


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
    asset_expense_ratios: Mapping[str, float] | None = None,
    leverage_multiple: float = 1.0,
    financing_rate: float = 0.0,
    maintenance_margin: float = 0.0,
    contribution: float = 0.0,
    withdrawal: float = 0.0,
    initial_value: float = 100.0,
    base_currency: str = "USD",
    asset_currencies: Mapping[str, str] | None = None,
    fx_rates: pd.DataFrame | None = None,
    fx_quote: str = "base_per_foreign",
    risk_free_rate: float = 0.0,
    annual_inflation: float = 0.0,
    model_kind: str = "quadrant",
    hmm_states: int = 4,
    threshold_window: int | None = None,
    duration_model: str = "markov",
    garch: bool = False,
    garch_alpha: float = 0.10,
    garch_beta: float = 0.85,
    walk_forward: bool = True,
) -> SimulationRun:
    """Calibrate and simulate one fully specified investment scenario.

    ``model_kind="quadrant"`` builds the four-quadrant macro model from growth
    and inflation thresholds; ``model_kind="hmm"`` fits a Gaussian-emission
    hidden Markov model directly on the asset returns instead. ``duration_model``
    controls whether regime run lengths follow the Markov chain or the
    empirical sojourn distribution, and ``garch`` adds within-regime
    GARCH(1,1) conditional variance dynamics. ``walk_forward`` runs a
    strictly out-of-sample predictive check of the regime model against an
    unconditional benchmark.
    """

    transition_uncertainty = float(transition_uncertainty)
    if not 0 <= transition_uncertainty <= 1:
        raise ValueError("transition_uncertainty must be between 0 and 1.")
    macro_lag_periods = int(macro_lag_periods)
    if macro_lag_periods < 0:
        raise ValueError("macro_lag_periods must be non-negative.")
    available_columns = {str(column).strip().upper(): column for column in returns.columns}
    selected = [
        available_columns.get(str(ticker).strip().upper(), str(ticker).strip()) for ticker in selected_tickers
    ]
    if not selected:
        raise ValueError("Select at least one ticker.")
    missing = [ticker for ticker in selected if ticker not in returns.columns]
    if missing:
        raise ValueError(f"Selected tickers are missing from returns: {', '.join(missing)}")
    normalized_weights = {
        available_columns.get(str(asset).strip().upper(), asset): weight for asset, weight in weights.items()
    }
    normalized_expense_ratios = {
        available_columns.get(str(asset).strip().upper(), asset): float(rate)
        for asset, rate in (asset_expense_ratios or {}).items()
    }
    scenario_returns = convert_returns_to_base_currency(
        returns.loc[:, selected],
        asset_currencies=asset_currencies,
        base_currency=base_currency,
        fx_rates=fx_rates,
        fx_quote=fx_quote,
    )

    if model_kind == "hmm":
        model, fit = fit_hmm_model(
            scenario_returns,
            n_states=int(hmm_states),
            random_seed=int(random_seed),
        )
        regimes = fit.regimes
        model.metadata["base_currency"] = str(base_currency).strip().upper()
        model.metadata["fx_quote"] = fx_quote
        model.metadata["initial_value"] = float(initial_value)
        if asset_currencies:
            model.metadata["asset_currencies"] = dict(asset_currencies)
        diagnostics = build_hmm_diagnostics(model, fit.regimes)
    else:
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
            threshold_window=threshold_window,
        )
        regimes = classify_quadrants(
            macro,
            growth_col=growth_col,
            inflation_col=inflation_col,
            growth_threshold=growth_threshold,
            inflation_threshold=inflation_threshold,
            threshold_window=threshold_window,
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
            threshold_window=threshold_window,
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
            None if transition_uncertainty == 0.0 else max(1.0, 1.0 / transition_uncertainty**2)
        ),
        duration_model=duration_model,
        garch=garch,
        garch_alpha=float(garch_alpha),
        garch_beta=float(garch_beta),
    )
    wealth = simulate_portfolio_paths(
        result,
        weights=normalized_weights,
        initial_value=initial_value,
        rebalance_frequency=rebalance_frequency,
        transaction_cost_bps=float(transaction_cost_bps),
        asset_expense_ratios=normalized_expense_ratios,
        leverage_multiple=float(leverage_multiple),
        financing_rate=float(financing_rate),
        maintenance_margin=float(maintenance_margin),
        contribution=float(contribution),
        withdrawal=float(withdrawal),
    )
    walk_forward_result = None
    if walk_forward and model_kind == "quadrant":
        try:
            walk_forward_result = walk_forward_validation(
                scenario_returns,
                macro,
                growth_col=growth_col,
                inflation_col=inflation_col,
                growth_threshold=growth_threshold,
                inflation_threshold=inflation_threshold,
                macro_lag_periods=macro_lag_periods,
                threshold_window=threshold_window,
            )
            diagnostics.warnings.extend(walk_forward_result.warnings)
        except ValueError:
            walk_forward_result = None
    weight_series = pd.Series(normalized_weights, dtype=float).reindex(selected).fillna(0.0)
    weight_total = float(weight_series.sum())
    if not pd.notna(weight_total) or abs(weight_total) < 1e-12:
        raise ValueError("Portfolio weights must have a non-zero sum.")
    weight_series = weight_series / weight_total
    fee_series = pd.Series(normalized_expense_ratios, dtype=float).reindex(selected).fillna(0.0)
    summary = summarize_wealth_risk(
        wealth,
        initial_value=initial_value,
        risk_free_rate=risk_free_rate,
        annual_inflation=annual_inflation,
        contribution=float(contribution),
        withdrawal=float(withdrawal),
    )
    summary = summary.copy()
    for key, value in {
        "weighted_expense_ratio": float((weight_series.abs() * fee_series).sum()),
        "annual_fee_drag": float((weight_series.abs() * fee_series).sum() * float(leverage_multiple)),
        "annual_financing_cost": float(max(float(leverage_multiple) - 1.0, 0.0) * float(financing_rate)),
        "leverage_multiple": float(leverage_multiple),
        "maintenance_margin": float(maintenance_margin),
        "margin_calls": int(wealth.attrs.get("margin_calls", 0)),
    }.items():
        summary[key] = value
    return SimulationRun(
        model=model,
        regimes=regimes,
        result=result,
        wealth=wealth,
        summary=summary,
        diagnostics=diagnostics,
        walk_forward=walk_forward_result,
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
                "ulcer_index_mean": summary["ulcer_index_mean"],
                "sortino_ratio": summary["sortino_ratio"],
                "calmar_ratio": summary["calmar_ratio"],
                "geometric_annualized_return": summary["geometric_annualized_return"],
            }
        )
    return pd.DataFrame(rows)
