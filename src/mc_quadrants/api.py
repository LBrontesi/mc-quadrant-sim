"""UI-agnostic API layer shared by every frontend (web, Streamlit, Gradio).

All data loading, scenario building, and result shaping lives here so that
the simulation methodology is identical regardless of the interface. Frontends
should only call these functions and render the returned dicts/frames.
"""

from __future__ import annotations

import io
import re
from collections.abc import Mapping
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from mc_quadrants.data import (
    fetch_yahoo_fx_rates,
    load_market_data,
    prices_to_returns,
)
from mc_quadrants.demo import _demo_history
from mc_quadrants.pipeline import compare_distributions, run_scenario
from mc_quadrants.regimes import REGIME_ORDER

REGIME_NAMES = {
    "high_growth_low_inflation": "High growth / low inflation",
    "high_growth_high_inflation": "High growth / high inflation",
    "low_growth_high_inflation": "Low growth / high inflation",
    "low_growth_low_inflation": "Low growth / low inflation",
}
REGIME_LOOKUP = {name: state for state, name in REGIME_NAMES.items()}

DEMO_TICKERS = {
    "Stocks": "SPY",
    "Bonds": "IEF",
    "Gold": "GLD",
    "Commodities": "DBC",
    "International Stocks": "EFA",
    "Real Estate": "VNQ",
    "TIPS": "TIP",
    "Short Treasuries": "SHY",
}

DEFAULT_TICKER_ORDER = ["SPY", "IEF", "GLD", "DBC", "EFA", "VNQ", "TIP", "SHY"]
SYNTHETIC_TICKER_OPTIONS = DEFAULT_TICKER_ORDER + ["DBMF", "KMLM", "TLT", "QQQ"]

DEFAULT_CORRELATIONS = {
    "high_growth_low_inflation": -0.10,
    "high_growth_high_inflation": 0.35,
    "low_growth_high_inflation": 0.25,
    "low_growth_low_inflation": -0.40,
}

# Portfolio presets inspired by PortfolioCharts, mapped onto the available
# ticker universe. Approximations are noted per preset (for example, IEF
# stands in for long-term treasuries and SHY for short-term/cash holdings).
PORTFOLIO_PRESETS: dict[str, dict[str, float]] = {
    "Classic 60/40": {"SPY": 60.0, "IEF": 40.0},
    "Three-Fund": {"SPY": 60.0, "EFA": 30.0, "IEF": 10.0},
    "Permanent Portfolio": {"SPY": 25.0, "IEF": 25.0, "SHY": 25.0, "GLD": 25.0},
    "Golden Butterfly (approx)": {"SPY": 40.0, "IEF": 20.0, "SHY": 20.0, "GLD": 20.0},
    "All Seasons (approx)": {"SPY": 30.0, "IEF": 40.0, "TIP": 15.0, "GLD": 7.5, "DBC": 7.5},
    "Core Four": {"SPY": 48.0, "EFA": 24.0, "IEF": 20.0, "VNQ": 8.0},
    "Risk Parity (simplified)": {"SPY": 30.0, "IEF": 40.0, "GLD": 15.0, "SHY": 15.0},
}

DISTRIBUTION_KEYS = {
    "normal": "normal",
    "student_t": "student_t",
    "bootstrap": "bootstrap",
    "block_bootstrap": "block_bootstrap",
}
REBALANCE_KEYS = {
    "legacy": None,
    "monthly": 1,
    "quarterly": 3,
    "annual": 12,
}


def parse_tickers(raw_tickers: str | list[str]) -> list[str]:
    raw_values = (
        re.split(r"[,;\s]+", str(raw_tickers).strip().upper())
        if isinstance(raw_tickers, str)
        else [str(ticker).strip().upper() for ticker in raw_tickers]
    )
    parsed: list[str] = []
    for ticker in raw_values:
        if ticker and ticker not in parsed:
            parsed.append(ticker)
    return parsed


def parse_pair_map(raw: str, kind: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for pair in re.split(r"[,;\s]+", str(raw).strip().upper()):
        if not pair:
            continue
        if ":" not in pair:
            raise ValueError(f"Invalid {kind} '{pair}'. Use ASSET:{kind.upper()}.")
        asset, value = pair.split(":", 1)
        if not asset or not value:
            raise ValueError(f"Invalid {kind} '{pair}'. Use ASSET:{kind.upper()}.")
        parsed[asset] = value
    return parsed


def _currency_for_asset(asset: str, asset_currencies: Mapping[str, str]) -> str:
    normalized = str(asset).strip().upper()
    base_asset = normalized.removesuffix("_SIM").removesuffix("SIM")
    return asset_currencies.get(normalized, asset_currencies.get(base_asset, "USD"))


def prepare_fx_rates(
    returns: pd.DataFrame,
    selected_tickers: list[str],
    base_currency: str,
    currency_map: Mapping[str, str],
) -> tuple[dict[str, str], pd.DataFrame | None]:
    foreign_currencies = sorted(
        {
            _currency_for_asset(ticker, currency_map)
            for ticker in selected_tickers
        }
        - {base_currency}
    )
    if not foreign_currencies:
        return dict(currency_map), None
    fx_start = pd.Timestamp(returns.index.min()) - pd.DateOffset(months=1)
    fx_rates = fetch_yahoo_fx_rates(
        foreign_currencies,
        base_currency,
        start=fx_start.strftime("%Y-%m-%d"),
        end=pd.Timestamp(returns.index.max()).strftime("%Y-%m-%d"),
    )
    return dict(currency_map), fx_rates


def default_weights(ticker: str) -> float:
    defaults = {
        "SPY": 40.0,
        "IEF": 20.0,
        "GLD": 10.0,
        "DBC": 10.0,
        "EFA": 10.0,
        "VNQ": 5.0,
        "TIP": 3.0,
        "SHY": 2.0,
        "DBMF": 5.0,
        "KMLM": 5.0,
    }
    base = str(ticker).removesuffix("_SIM").removesuffix("SIM")
    return defaults.get(base, 0.0)


def default_selected_tickers(tickers: list[str]) -> list[str]:
    preferred = [
        f"{ticker}SIM" if f"{ticker}SIM" in tickers else ticker
        for ticker in DEFAULT_TICKER_ORDER
        if ticker in tickers or f"{ticker}SIM" in tickers
    ]
    if preferred:
        preferred.extend(
            ticker
            for ticker in tickers
            if ticker.endswith("SIM") and not ticker.endswith("_SIM") and ticker not in preferred
        )
        return preferred
    stitched = [ticker for ticker in tickers if ticker.endswith("SIM") and not ticker.endswith("_SIM")]
    if stitched:
        return stitched
    return tickers[: min(4, len(tickers))]


def correlation_overrides(
    payload: Mapping[str, Any],
    selected_tickers: list[str],
) -> tuple[dict[str, dict[tuple[str, str], float]] | None, float]:
    """Build per-regime pairwise correlation targets for the first two assets."""

    if not bool(payload.get("use_correlation_override", False)):
        return None, 1.0
    if len(selected_tickers) < 2:
        return None, 1.0
    blend = float(payload.get("correlation_blend", 0.40))
    if not 0 <= blend <= 1:
        raise ValueError("Correlation blend must be between 0 and 1.")
    targets = payload.get("correlation_override_targets") or {}
    pair = (selected_tickers[0], selected_tickers[1])
    overrides: dict[str, dict[tuple[str, str], float]] = {}
    for state in REGIME_ORDER:
        raw = targets.get(state, DEFAULT_CORRELATIONS[state])
        value = float(raw)
        if not -1 <= value <= 1:
            raise ValueError(f"Correlation override for {state} must be between -1 and 1.")
        overrides[state] = {pair: value}
    return overrides, blend


def _read_csv_text(content: str | None) -> pd.DataFrame | None:
    if not content:
        return None
    data = pd.read_csv(io.StringIO(content))
    if "Date" not in data.columns:
        raise ValueError("CSV files need a Date column.")
    data["Date"] = pd.to_datetime(data["Date"])
    return data.set_index("Date").sort_index()


def _normalize_columns(data: pd.DataFrame) -> pd.DataFrame:
    normalized = data.copy()
    normalized.columns = [str(column).strip().upper() for column in normalized.columns]
    return normalized


def load_data_source(payload: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, list[str], str, str, str]:
    source = str(payload.get("source", "demo"))
    if source == "demo":
        seed = int(payload.get("seed", 42))
        macro, returns = _demo_history(seed)
        returns = _normalize_columns(returns.rename(columns=DEMO_TICKERS))
        return macro, returns, list(returns.columns), "growth", "inflation", f"Loaded demo data with seed {seed}."

    if source == "yahoo":
        tickers = parse_tickers(payload.get("tickers", []))
        if not tickers:
            raise ValueError("Enter at least one Yahoo Finance ticker.")
        historical_proxies = parse_pair_map(payload.get("proxies", ""), "proxy")
        synthetic_assets = parse_tickers(payload.get("synthetic", []))
        tickers.extend(asset for asset in synthetic_assets if asset not in tickers)
        synthetic_seed = int(payload.get("synthetic_seed", 42))
        start = str(payload.get("start", "1990-01-01"))
        end = str(payload.get("end", date.today().isoformat()))
        macro, returns, available = load_market_data(
            tickers,
            start,
            end,
            historical_proxies=historical_proxies or None,
            synthetic_assets=synthetic_assets,
            synthetic_seed=synthetic_seed,
        )
        available_list = list(available)
        msg = f"Loaded {len(available_list)} tickers from Yahoo Finance."
        if historical_proxies:
            msg += f" Backfilled proxies: {', '.join(historical_proxies.values())}."
        if synthetic_assets:
            msg += f" Simulated sources: {', '.join(f'{asset}SIM' for asset in synthetic_assets)}."
        return macro, returns, available_list, "growth", "inflation", msg

    if source == "csv":
        asset_data = _read_csv_text(payload.get("csv_prices"))
        macro_data = _read_csv_text(payload.get("csv_macro"))
        if asset_data is None or macro_data is None:
            raise ValueError("Upload both an asset CSV and a macro CSV.")
        growth_col = str(payload.get("growth_col", "growth"))
        inflation_col = str(payload.get("inflation_col", "inflation"))
        if growth_col not in macro_data.columns:
            raise ValueError(f"Growth column not found in macro CSV: {growth_col}")
        if inflation_col not in macro_data.columns:
            raise ValueError(f"Inflation column not found in macro CSV: {inflation_col}")
        if growth_col == inflation_col:
            raise ValueError("Growth and inflation must use different macro columns.")
        if str(payload.get("asset_input", "Price levels")) == "Price levels":
            returns = prices_to_returns(asset_data, method="log")
        else:
            returns = asset_data.apply(pd.to_numeric, errors="coerce")
        if bool(payload.get("monthly", True)):
            returns = returns.resample("ME").sum(min_count=1)
        macro_data = macro_data.apply(pd.to_numeric, errors="coerce").resample("ME").last()
        returns = _normalize_columns(returns)
        returns = returns.dropna(how="all")
        if returns.empty or not any(returns[column].notna().any() for column in returns.columns):
            raise ValueError("The asset CSV has no usable numeric data.")
        return macro_data, returns, list(returns.columns), growth_col, inflation_col, "Loaded data from CSV uploads."

    raise ValueError(f"Unknown data source: {source}")


def _frame_preview(frame: pd.DataFrame, columns: list[str] | None = None, rows: int = 60) -> dict[str, Any]:
    preview = frame.tail(rows)
    selected = list(preview.columns) if columns is None else columns
    records = preview.loc[:, [col for col in selected if col in preview.columns]].reset_index(names="Date")
    return {
        "columns": [str(column) for column in records.columns],
        "rows": [[None if pd.isna(value) else _json_value(value) for value in record] for record in records.itertuples(index=False, name=None)],
    }


def _coverage(returns: pd.DataFrame) -> dict[str, dict[str, str]]:
    coverage: dict[str, dict[str, str]] = {}
    for column in returns.columns:
        valid = returns[column].dropna()
        if valid.empty:
            continue
        coverage[str(column)] = {
            "first": pd.Timestamp(valid.index[0]).strftime("%Y-%m-%d"),
            "last": pd.Timestamp(valid.index[-1]).strftime("%Y-%m-%d"),
        }
    return coverage


def build_load_response(
    macro: pd.DataFrame,
    returns: pd.DataFrame,
    tickers: list[str],
    growth_col: str,
    inflation_col: str,
    message: str,
) -> dict[str, Any]:
    return {
        "ok": True,
        "tickers": tickers,
        "default_tickers": default_selected_tickers(tickers),
        "growth_col": growth_col,
        "inflation_col": inflation_col,
        "message": message,
        "coverage": _coverage(returns),
        "presets": [
            {"name": name, "weights": dict(weights)}
            for name, weights in PORTFOLIO_PRESETS.items()
        ],
        "macro": _frame_preview(macro, columns=[growth_col, inflation_col]),
        "returns": _frame_preview(returns),
    }


def _threshold_value(raw: Any) -> str | float:
    if isinstance(raw, str) and raw.startswith("fixed:"):
        return float(raw.split(":", 1)[1])
    return raw if raw in {"median", "mean"} else float(raw)


def scenario_kwargs(payload: Mapping[str, Any]) -> dict[str, Any]:
    distribution = str(payload.get("distribution", "normal")).lower().replace("-", "_")
    distribution = DISTRIBUTION_KEYS.get(distribution, distribution)
    if distribution not in DISTRIBUTION_KEYS.values():
        raise ValueError("Unknown return distribution.")
    rebalance_label = str(payload.get("rebalance", "monthly")).lower()
    if rebalance_label not in REBALANCE_KEYS:
        raise ValueError(f"Unknown rebalancing frequency: {rebalance_label}")
    start_state = None
    start_label = str(payload.get("start_state", "Stationary"))
    if start_label != "Stationary":
        start_state = REGIME_LOOKUP.get(start_label)
        if start_state is None:
            raise ValueError(f"Unknown start state: {start_label}")
    weights = {str(ticker).strip().upper(): float(weight) for ticker, weight in (payload.get("weights") or {}).items()}
    if not weights:
        raise ValueError("Set at least one ticker weight above zero.")
    base_currency = str(payload.get("base_currency", "USD")).strip().upper()
    if len(base_currency) != 3:
        raise ValueError("Portfolio currency must be a three-letter ISO code.")
    return {
        "growth_threshold": _threshold_value(payload.get("growth_threshold", "median")),
        "inflation_threshold": _threshold_value(payload.get("inflation_threshold", "median")),
        "periods": int(payload.get("periods", 120)),
        "paths": int(payload.get("paths", 3000)),
        "random_seed": int(payload.get("seed", 7)),
        "start_state": start_state,
        "weights": weights,
        "macro_lag_periods": int(payload.get("macro_lag", 1)),
        "distribution": distribution,
        "degrees_of_freedom": float(payload.get("degrees_of_freedom", 5.0)),
        "block_size": int(payload.get("block_size", 3)),
        "transition_uncertainty": float(payload.get("transition_uncertainty", 0.0)),
        "rebalance_frequency": REBALANCE_KEYS[rebalance_label],
        "transaction_cost_bps": float(payload.get("cost_bps", 10.0)),
        "contribution": float(payload.get("contribution", 0.0)),
        "withdrawal": float(payload.get("withdrawal", 0.0)),
        "base_currency": base_currency,
        "risk_free_rate": float(payload.get("risk_free_rate", 0.0)) / 100.0,
        "annual_inflation": float(payload.get("annual_inflation", 0.0)) / 100.0,
    }


def run_scenario_payload(payload: Mapping[str, Any]) -> tuple[Any, list[str], pd.DataFrame]:
    """Load, select, and run a single scenario from a client payload."""

    macro, returns, tickers, growth_col, inflation_col, _ = load_data_source(payload)
    selected_tickers = parse_tickers(payload.get("selected_tickers", tickers))
    if not selected_tickers:
        raise ValueError("Select at least one ticker.")
    missing = [ticker for ticker in selected_tickers if ticker not in returns.columns]
    if missing:
        raise ValueError(f"Selected tickers are missing from the loaded returns: {', '.join(missing)}")
    returns = returns.loc[:, selected_tickers]

    kwargs = scenario_kwargs(payload)
    currency_map = parse_pair_map(payload.get("currency_map", ""), "currency")
    asset_currencies, fx_rates = prepare_fx_rates(
        returns,
        selected_tickers,
        kwargs["base_currency"],
        currency_map,
    )
    correlation_targets, override_weight = correlation_overrides(payload, selected_tickers)
    scenario = run_scenario(
        returns=returns,
        macro=macro,
        selected_tickers=selected_tickers,
        growth_col=growth_col,
        inflation_col=inflation_col,
        correlation_overrides=correlation_targets,
        override_weight=override_weight,
        **kwargs,
        asset_currencies=asset_currencies,
        fx_rates=fx_rates,
    )
    return scenario, selected_tickers, macro


def _max_drawdown_paths(wealth: pd.DataFrame, initial_value: float = 100.0) -> np.ndarray:
    wealth_with_initial = pd.concat(
        [pd.DataFrame([[initial_value] * wealth.shape[1]], columns=wealth.columns), wealth],
        ignore_index=True,
    )
    running_max = wealth_with_initial.cummax(axis=0)
    return -(wealth_with_initial / running_max - 1.0).min(axis=0).to_numpy(dtype=float)


def _simulated_regime_summary(result: Any) -> pd.DataFrame:
    counts = pd.Series(result.regimes.ravel(), dtype="string").value_counts()
    total = max(int(counts.sum()), 1)
    return pd.DataFrame(
        {
            "regime": result.states,
            "simulated_observations": [int(counts.get(state, 0)) for state in result.states],
            "simulated_share": [float(counts.get(state, 0)) / total for state in result.states],
        }
    )


def _json_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.strftime("%Y-%m-%d")
    return value


def build_simulate_response(payload: Mapping[str, Any]) -> dict[str, Any]:
    scenario, selected_tickers, macro = run_scenario_payload(payload)
    model = scenario.model
    result = scenario.result
    wealth = scenario.wealth
    summary = scenario.summary
    growth_col = scenario.model.metadata.get("growth_col", "growth")
    inflation_col = scenario.model.metadata.get("inflation_col", "inflation")

    percentiles = wealth.quantile([0.05, 0.50, 0.95], axis=1).T
    regime_mix = (
        pd.Series(result.regimes.ravel())
        .value_counts(normalize=True)
        .reindex(REGIME_ORDER)
        .fillna(0.0)
        .rename(index=REGIME_NAMES)
    )
    scatter = macro[[growth_col, inflation_col]].copy()
    scatter["regime"] = scenario.regimes.map(REGIME_NAMES)
    scatter["date"] = scatter.index.astype(str)
    scatter_records = [
        {
            "date": str(getattr(record, "date")),
            "growth": float(getattr(record, growth_col)),
            "inflation": float(getattr(record, inflation_col)),
            "regime": str(getattr(record, "regime")),
        }
        for record in scatter.dropna().tail(240).itertuples(index=False)
    ]
    observations = {
        REGIME_NAMES[state]: int(moments.observations) for state, moments in model.moments.items()
    }
    diagnostics = scenario.diagnostics.regime_summary.copy()
    simulated_diagnostics = _simulated_regime_summary(result)
    diagnostics = diagnostics.merge(simulated_diagnostics, on="regime", how="left")
    diagnostics["regime"] = diagnostics["regime"].map(REGIME_NAMES)

    summary_values = {str(key): _json_value(value) for key, value in summary.items()}
    contribution = float(payload.get("contribution", 0.0))
    withdrawal = float(payload.get("withdrawal", 0.0))
    if contribution or withdrawal:
        summary_values["periodic_contribution"] = contribution
        summary_values["periodic_withdrawal"] = withdrawal
        summary_values["total_contributed"] = contribution * len(wealth)

    return {
        "ok": True,
        "summary": summary_values,
        "currency": scenario.model.metadata.get("base_currency", "USD"),
        "terms": "real" if scenario_kwargs(payload)["annual_inflation"] > 0 else "nominal",
        "warnings": list(scenario.diagnostics.warnings),
        "wealth": {
            "periods": list(range(1, len(wealth) + 1)),
            "p05": percentiles[0.05].tolist(),
            "median": percentiles[0.50].tolist(),
            "p95": percentiles[0.95].tolist(),
        },
        "terminal": wealth.iloc[-1].tolist(),
        "drawdowns": _max_drawdown_paths(wealth).tolist(),
        "regime_timeline": [str(state) for state in result.regimes[0]],
        "regime_mix": [{"label": label, "share": float(share)} for label, share in regime_mix.items()],
        "transition": {
            "labels": [REGIME_NAMES[state] for state in model.transition_matrix.index],
            "values": model.transition_matrix.to_numpy(dtype=float).tolist(),
        },
        "macro_scatter": scatter_records,
        "observations": observations,
        "correlations": {
            REGIME_NAMES[state]: {
                "labels": list(model.moments[state].correlation.columns),
                "values": model.moments[state].correlation.to_numpy(dtype=float).tolist(),
            }
            for state in REGIME_ORDER
        },
        "diagnostics": {
            "columns": [str(column) for column in diagnostics.columns],
            "rows": [
                [None if pd.isna(value) else _json_value(value) for value in record]
                for record in diagnostics.itertuples(index=False, name=None)
            ],
        },
        "selected_tickers": selected_tickers,
        "message": (
            f"Simulation complete: {len(wealth)} periods x {wealth.shape[1]} paths. "
            f"Distribution: {scenario.result.distribution}."
        ),
    }


def build_wealth_csv(payload: Mapping[str, Any]) -> dict[str, Any]:
    scenario, selected_tickers, _ = run_scenario_payload(payload)
    wealth = scenario.wealth.copy()
    wealth.insert(0, "period", range(1, len(wealth) + 1))
    return {"ok": True, "csv": wealth.to_csv(index=False), "tickers": selected_tickers}


def build_compare_response(payload: Mapping[str, Any]) -> dict[str, Any]:
    macro, returns, tickers, growth_col, inflation_col, _ = load_data_source(payload)
    selected_tickers = parse_tickers(payload.get("selected_tickers", tickers))
    if not selected_tickers:
        raise ValueError("Select at least one ticker.")
    missing = [ticker for ticker in selected_tickers if ticker not in returns.columns]
    if missing:
        raise ValueError(f"Selected tickers are missing from the loaded returns: {', '.join(missing)}")
    returns = returns.loc[:, selected_tickers]
    kwargs = scenario_kwargs(payload)
    kwargs.pop("distribution", None)
    currency_map = parse_pair_map(payload.get("currency_map", ""), "currency")
    asset_currencies, fx_rates = prepare_fx_rates(returns, selected_tickers, kwargs["base_currency"], currency_map)
    correlation_targets, override_weight = correlation_overrides(payload, selected_tickers)
    comparison = compare_distributions(
        {"Normal": "normal", "Student-t": "student_t"},
        returns=returns,
        macro=macro,
        selected_tickers=selected_tickers,
        growth_col=growth_col,
        inflation_col=inflation_col,
        correlation_overrides=correlation_targets,
        override_weight=override_weight,
        **kwargs,
        asset_currencies=asset_currencies,
        fx_rates=fx_rates,
    )
    return {
        "ok": True,
        "columns": [str(column) for column in comparison.columns],
        "rows": [
            [None if pd.isna(value) else _json_value(value) for value in record]
            for record in comparison.itertuples(index=False, name=None)
        ],
    }
