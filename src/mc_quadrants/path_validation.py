"""Distribution scores and portfolio replay for production-path validation.

Only simulated paths and held-out outcomes enter this module. Nothing scored
here is used to fit a model or select a forecast-origin state.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from mc_quadrants.cpi import accounting_inflation_paths, monthly_cpi_from_yoy
from mc_quadrants.simulation import inflation_deflators, simulate_portfolio_paths
from mc_quadrants.types import SimulationResult


def crps(observation: float, draws: np.ndarray) -> float:
    """Exact CRPS of an empirical ensemble, O(n log n), in outcome units."""
    values = np.sort(np.asarray(draws, dtype=float))
    n = len(values)
    return float(np.abs(values - observation).mean() - ((2 * np.arange(1, n + 1) - n - 1) @ values) / n**2)


def variogram_score(observation: np.ndarray, draws: np.ndarray, power: float = 0.5) -> float:
    """Equal-weight, pair-averaged variogram score; zero for one asset."""
    i, j = np.triu_indices(len(observation), k=1)
    if not len(i):
        return 0.0
    observed = np.abs(observation[i] - observation[j]) ** power
    predicted = (np.abs(draws[:, i] - draws[:, j]) ** power).mean(axis=0)
    return float(np.square(observed - predicted).mean())


def scalar_scores(name: str, observation: float, draws: np.ndarray) -> dict[str, float]:
    """Forecast summaries plus CRPS, central 90% coverage and midpoint PIT."""
    lower, median, upper = np.quantile(draws, [0.05, 0.5, 0.95])
    return {
        f"{name}_actual": float(observation),
        f"{name}_mean": float(np.mean(draws)),
        f"{name}_p05": float(lower),
        f"{name}_p50": float(median),
        f"{name}_p95": float(upper),
        f"{name}_crps": crps(observation, draws),
        f"{name}_coverage_90": float(lower <= observation <= upper),
        f"{name}_pit": float((np.mean(draws < observation) + np.mean(draws <= observation)) / 2),
    }


def annual_macro(model, result: SimulationResult, column: str | None, flag: str):
    if result.macro_paths is None or column not in result.macro_columns:
        return None
    values = result.macro_paths[:, :, result.macro_columns.index(column)].copy()
    dynamics = model.metadata.get("macro_dynamics", {})
    if dynamics.get(flag, model.metadata.get(flag, False)):
        values /= 100
    return values


def replay_portfolio(result, weights, options, *, inflation=None, rates=None, withdrawal_inflation=None):
    """Run the production ledger from the same starting balance at each origin."""
    options = dict(options)
    periods = len(result.returns)
    # A withdrawal scheduled beyond the scored horizon has not started yet.
    if options.get("withdrawal_start_period", 1) > periods:
        options.update(withdrawal=0.0, withdrawal_start_period=1)
    options["financing_rate_paths"] = rates
    options["financing_inflation_paths"] = inflation
    options["withdrawal_inflation_paths"] = (
        inflation if withdrawal_inflation is None else withdrawal_inflation
    )
    return simulate_portfolio_paths(result, weights, **options).to_numpy(dtype=float)


def truncate_result(result, horizon):
    return replace(
        result,
        returns=result.returns[:horizon],
        regimes=result.regimes[:horizon],
        macro_paths=None if result.macro_paths is None else result.macro_paths[:horizon],
    )


def max_drawdown(wealth, initial_value):
    values = np.vstack([np.full((1, wealth.shape[1]), initial_value), wealth])
    peaks = np.maximum.accumulate(values, axis=0)
    return np.max(1 - values / peaks, axis=0)


def score_horizon(
    *,
    model,
    forecast,
    benchmark,
    observed,
    weights,
    portfolio_options,
    inflation_col,
    rate_col,
    observed_macro,
    price_index,
    origin,
    horizon,
    outcome_dates,
    forecast_components,
    macro_cutoff,
    energy_score,
):
    """Score one fully observed horizon; replay terminal taxes at that horizon."""
    forecast, benchmark, observed = (
        truncate_result(result, horizon) for result in (forecast, benchmark, observed)
    )
    options = dict(portfolio_options)
    options["tax_start_date"] = str((pd.Timestamp(origin) + pd.offsets.MonthEnd(1)).date())
    options["state_inflation"] = model.metadata.get("state_inflation")
    initial = float(options.get("initial_value", 100.0))
    components = [(m, truncate_result(f, horizon)) for m, f in forecast_components]

    def component_macro(column, flag):
        arrays = [annual_macro(m, f, column, flag) for m, f in components]
        return np.concatenate(arrays, axis=1) if all(a is not None for a in arrays) else None

    inflation = component_macro(inflation_col, "inflation_is_percent")
    spending_arrays = [accounting_inflation_paths(m, f) for m, f in components]
    spending_inflation = np.concatenate(spending_arrays, axis=1) if all(a is not None for a in spending_arrays) else None
    rates = component_macro(rate_col, "rate_is_percent")
    actual_inflation = annual_macro(model, observed, inflation_col, "inflation_is_percent")
    actual_rates = annual_macro(model, observed, rate_col, "rate_is_percent")
    # The iid benchmark has no macro model: persist the origin's last release.
    history = observed_macro.loc[observed_macro.index <= macro_cutoff]
    benchmark_macro = {}
    for column, flag in ((inflation_col, "inflation_is_percent"), (rate_col, "rate_is_percent")):
        if column and column in history and not history[column].dropna().empty:
            value = float(history[column].dropna().iloc[-1])
            if model.metadata.get("macro_dynamics", {}).get(flag, model.metadata.get(flag, False)):
                value /= 100
            benchmark_macro[column] = np.full(benchmark.returns.shape[:2], value)
    wealth = np.concatenate(
        [
            replay_portfolio(
                component,
                weights,
                {**options, "state_inflation": fitted.metadata.get("state_inflation")},
                inflation=annual_macro(fitted, component, inflation_col, "inflation_is_percent"),
                rates=annual_macro(fitted, component, rate_col, "rate_is_percent"),
                withdrawal_inflation=accounting_inflation_paths(fitted, component),
            )
            for fitted, component in components
        ],
        axis=1,
    )
    realized_cpi = None
    actual_spending_inflation = None
    if price_index is not None:
        realized = price_index.reindex(pd.DatetimeIndex([origin]).append(outcome_dates)).to_numpy(dtype=float)
        if len(realized) == horizon + 1 and np.isfinite(realized).all() and (realized > 0).all():
            realized_cpi = realized
            # The production spending controller takes annualized monthly rates.
            # Derive those from actual CPI increments for the observed replay,
            # keeping the published annual rate for financing and macro scores.
            actual_spending_inflation = np.expm1(12 * np.diff(np.log(realized)))[:, None]
    actual_wealth = replay_portfolio(
        observed,
        weights,
        options,
        inflation=actual_inflation,
        rates=actual_rates,
        withdrawal_inflation=actual_spending_inflation,
    )
    benchmark_spending = benchmark_macro.get(inflation_col)
    if benchmark_spending is not None and model.metadata.get("cpi_process"):
        benchmark_spending = np.expm1(12 * monthly_cpi_from_yoy(benchmark_spending, model.metadata["cpi_process"]["monthly_log_history"]))
    benchmark_wealth = replay_portfolio(
        benchmark,
        weights,
        options,
        inflation=benchmark_macro.get(inflation_col) if inflation is not None else None,
        rates=benchmark_macro.get(rate_col) if rates is not None else None,
        withdrawal_inflation=benchmark_spending,
    )
    # Asset cumulative log returns retain stable units for multivariate scoring.
    draws = forecast.returns.sum(axis=0)
    baseline = benchmark.returns.sum(axis=0)
    actual = observed.returns[:, 0].sum(axis=0)
    if not all(
        np.isfinite(values).all()
        for values in (draws, baseline, actual, wealth, actual_wealth, benchmark_wealth)
    ):
        raise ValueError(
            "Production-path validation produced non-finite outcomes; inspect the model's tails."
        )
    row = {
        "origin": pd.Timestamp(origin),
        "target_end": pd.Timestamp(outcome_dates[-1]),
        "horizon_months": horizon,
        "paths": forecast.returns.shape[1],
        "energy_score": energy_score(actual, draws),
        "benchmark_energy_score": energy_score(actual, baseline),
        "variogram_score": variogram_score(actual, draws),
        "benchmark_variogram_score": variogram_score(actual, baseline),
        "marginal_crps": float(np.mean([crps(y, draws[:, i]) for i, y in enumerate(actual)])),
        "benchmark_marginal_crps": float(np.mean([crps(y, baseline[:, i]) for i, y in enumerate(actual)])),
    }
    # Cumulative return is for a monthly rebalanced, pre-cost portfolio, separate
    # from wealth under the user's ledger, fees, leverage and external cash flows.
    vector = pd.Series(weights).reindex(forecast.assets).to_numpy(dtype=float)

    def cumulative_return(result):
        return np.prod(1 + np.expm1(result.returns) @ vector, axis=0) - 1

    outcomes = {
        "portfolio_return": (
            cumulative_return(observed)[0],
            cumulative_return(forecast),
            cumulative_return(benchmark),
        ),
        "wealth": (actual_wealth[-1, 0], wealth[-1], benchmark_wealth[-1]),
        "drawdown": (
            max_drawdown(actual_wealth, initial)[0],
            max_drawdown(wealth, initial),
            max_drawdown(benchmark_wealth, initial),
        ),
    }
    if inflation is not None and actual_inflation is not None:
        outcomes["inflation"] = (actual_inflation[-1, 0], inflation[-1], benchmark_macro[inflation_col][-1])
    # CPI is an outcome only. Never reconstruct monthly realized inflation from
    # trailing annual rates, and never feed a future price level into forecasting.
    if realized_cpi is not None:
        actual_real = actual_wealth * (realized_cpi[0] / realized_cpi[1:])[:, None]
        predicted_deflator = inflation_deflators(
            horizon,
            wealth.shape[1],
            annual_inflation=options.get("annual_inflation", 0.0),
            inflation_paths=spending_inflation,
        )
        baseline_deflator = inflation_deflators(
            horizon,
            benchmark_wealth.shape[1],
            annual_inflation=options.get("annual_inflation", 0.0),
            inflation_paths=benchmark_spending if inflation is not None else None,
        )
        real = wealth * predicted_deflator
        base_real = benchmark_wealth * baseline_deflator
        outcomes["cumulative_inflation"] = (
            realized_cpi[-1] / realized_cpi[0] - 1,
            1 / predicted_deflator[-1] - 1,
            1 / baseline_deflator[-1] - 1,
        )
        outcomes["real_wealth"] = (actual_real[-1, 0], real[-1], base_real[-1])
        outcomes["real_drawdown"] = (
            max_drawdown(actual_real, initial)[0],
            max_drawdown(real, initial),
            max_drawdown(base_real, initial),
        )
    for name, (actual_value, values, baseline_values) in outcomes.items():
        row.update(scalar_scores(name, actual_value, values))
        row[f"benchmark_{name}_crps"] = crps(actual_value, baseline_values)
    row["joint_loss_probability"] = float(np.mean((draws < 0).all(axis=1)))
    row["joint_loss_actual"] = float((actual < 0).all())
    row["joint_loss_brier"] = (row["joint_loss_probability"] - row["joint_loss_actual"]) ** 2
    row["benchmark_joint_loss_brier"] = (
        float(np.mean((baseline < 0).all(axis=1))) - row["joint_loss_actual"]
    ) ** 2
    return row


def summarize_horizons(scores, horizons):
    """Report counts alongside scores; overlapping origins are not independent."""
    rows = []
    for horizon in horizons:
        group = scores.loc[scores["horizon_months"] == horizon].sort_values("origin")
        last_end = None
        nonoverlapping = 0
        for origin in group["origin"]:
            if last_end is None or origin >= last_end:
                nonoverlapping += 1
                last_end = origin + pd.offsets.MonthEnd(horizon)
        row = {"horizon_months": horizon, "origins": len(group), "nonoverlapping_origins": nonoverlapping}
        for column in scores.columns:
            if column.endswith(("_score", "_crps", "_coverage_90", "_brier")):
                row[column] = float(group[column].mean())
                row[f"{column}_count"] = int(group[column].notna().sum())
        row["energy_advantage"] = row.get("benchmark_energy_score", np.nan) - row.get("energy_score", np.nan)
        row["portfolio_crps_advantage"] = row.get("benchmark_portfolio_return_crps", np.nan) - row.get(
            "portfolio_return_crps", np.nan
        )
        rows.append(row)
    return pd.DataFrame(rows)
