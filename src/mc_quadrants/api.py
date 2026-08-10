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
from mc_quadrants.pipeline import compare_distributions, run_scenario
from mc_quadrants.regimes import REGIME_ORDER

REGIME_NAMES = {
    "high_growth_low_inflation": "High growth / low inflation",
    "high_growth_high_inflation": "High growth / high inflation",
    "low_growth_high_inflation": "Low growth / high inflation",
    "low_growth_low_inflation": "Low growth / low inflation",
}
REGIME_LOOKUP = {name: state for state, name in REGIME_NAMES.items()}

_PERCENT_METRICS = {
    "annualized_return",
    "annualized_volatility",
    "cash_flow_adjusted_annualized_return",
    "cash_flow_adjusted_volatility",
    "geometric_annualized_return",
    "weighted_expense_ratio",
    "annual_fee_drag",
    "annual_financing_cost",
    "effective_financing_rate",
    "maintenance_margin",
    "probability_of_loss",
    "max_drawdown_mean",
    "max_drawdown_p95",
    "max_drawdown_worst",
    "ulcer_index_mean",
    "ulcer_index_p95",
}
_CURRENCY_METRICS = {
    "mean",
    "std",
    "p05",
    "p50",
    "p95",
    "var_95",
    "expected_shortfall_95",
    "periodic_contribution",
    "periodic_withdrawal",
    "total_contributed",
    "total_withdrawn",
    "net_external_cash_flow",
}


def format_metric_value(key: str, value: Any, currency: str = "USD") -> str:
    """Format a result value according to its semantic unit for frontend use."""

    if value is None:
        return "-"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(numeric):
        return "-"
    if key == "leverage_multiple":
        return f"{numeric:.1f}x"
    if key == "margin_calls":
        return f"{int(numeric):,}"
    if key in _PERCENT_METRICS:
        return f"{numeric * 100:.2f}%"
    if key in _CURRENCY_METRICS:
        return f"{currency} {numeric:,.2f}"
    return f"{numeric:,.2f}"


def _state_label(state: str) -> str:
    """Human-readable label for quadrant or HMM states."""

    return REGIME_NAMES.get(state, f"Regime {state.removeprefix('state_')}")


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

MAX_PERIODS = 360
MAX_PATHS = 120_000
MAX_WORKERS = 16
DEFAULT_EXPORT_PATHS = 1_000
MAX_EXPORT_PATHS = 5_000


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


def parse_expense_ratios(raw: str | Mapping[str, Any] | None) -> dict[str, float]:
    """Parse annual ETF expense ratios supplied as percentages into decimals."""

    if not raw:
        return {}
    if isinstance(raw, Mapping):
        items = raw.items()
    else:
        items = []
        for pair in re.split(r"[,;\s]+", str(raw).strip().upper()):
            if not pair:
                continue
            if ":" not in pair:
                raise ValueError(f"Invalid expense ratio '{pair}'. Use ASSET:PERCENT.")
            asset, value = pair.split(":", 1)
            items.append((asset, value))
    ratios: dict[str, float] = {}
    for asset, raw_value in items:
        normalized_asset = str(asset).strip().upper()
        if not normalized_asset:
            raise ValueError("Expense ratio asset names must not be empty.")
        percentage = float(raw_value)
        if not np.isfinite(percentage) or not 0 <= percentage < 100:
            raise ValueError(f"Expense ratio for {normalized_asset} must be between 0 and 100 percent.")
        ratios[normalized_asset] = percentage / 100.0
    return ratios


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
        {_currency_for_asset(ticker, currency_map) for ticker in selected_tickers} - {base_currency}
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


def load_data_source(
    payload: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], str, str, str]:
    source = str(payload.get("source", "yahoo"))
    if source == "yahoo":
        tickers = parse_tickers(payload.get("tickers", []))
        if not tickers:
            raise ValueError("Enter at least one Yahoo Finance ticker.")
        historical_proxies = parse_pair_map(payload.get("proxies", ""), "proxy")
        synthetic_assets = parse_tickers(payload.get("synthetic", []))
        tickers.extend(asset for asset in synthetic_assets if asset not in tickers)
        synthetic_seed = int(payload.get("synthetic_seed", 42))
        synthetic_method = str(payload.get("synthetic_method", "regime"))
        synthetic_categories = parse_pair_map(payload.get("synthetic_categories", ""), "category")
        growth_threshold = _threshold_value(payload.get("growth_threshold", "median"))
        inflation_threshold = _threshold_value(payload.get("inflation_threshold", "median"))
        threshold_window = int(payload.get("threshold_window", 0) or 0) or None
        macro_lag = int(payload.get("macro_lag", 1))
        macro_vintage = str(payload.get("macro_vintage", "latest"))
        start = str(payload.get("start", "1990-01-01"))
        end = str(payload.get("end", date.today().isoformat()))
        macro, returns, available, synthetic_report = load_market_data(
            tickers,
            start,
            end,
            historical_proxies=historical_proxies or None,
            synthetic_assets=synthetic_assets,
            synthetic_seed=synthetic_seed,
            synthetic_method=synthetic_method,
            synthetic_categories=synthetic_categories or None,
            growth_threshold=growth_threshold,
            inflation_threshold=inflation_threshold,
            threshold_window=threshold_window,
            macro_lag_periods=macro_lag,
            macro_vintage=macro_vintage,
        )
        returns.attrs["synthetic_report"] = synthetic_report
        available_list = list(available)
        timing_label = (
            "ALFRED initial-release, availability-aligned macro data"
            if bool(macro.attrs.get("point_in_time", False))
            else "latest-revised FRED macro data"
        )
        msg = f"Loaded {len(available_list)} tickers from Yahoo Finance with {timing_label}."
        if historical_proxies:
            msg += f" Backfilled proxies: {', '.join(historical_proxies.values())}."
        if synthetic_assets:
            msg += f" Simulated sources: {', '.join(f'{asset}SIM' for asset in synthetic_assets)}."
        if synthetic_report:
            grades = ", ".join(f"{name}:{info['grade']}" for name, info in synthetic_report.items())
            msg += f" Synthetic feasibility: {grades}."
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
        asset_input = str(payload.get("asset_input", "Price levels"))
        if asset_input == "Price levels":
            returns = prices_to_returns(asset_data, method="log")
        else:
            returns = asset_data.apply(pd.to_numeric, errors="coerce")
        if bool(payload.get("monthly", True)):
            if asset_input == "Simple returns":
                returns = (1.0 + returns).resample("ME").prod(min_count=1) - 1.0
            else:
                returns = returns.resample("ME").sum(min_count=1)
        if asset_input == "Simple returns":
            if (returns <= -1.0).any().any():
                raise ValueError("Simple returns must be greater than -100%.")
            returns = np.log1p(returns)
        elif asset_input not in {"Price levels", "Log returns", "Returns"}:
            raise ValueError("Asset input must be Price levels, Log returns, or Simple returns.")
        available_date = None
        if "AvailableDate" in macro_data.columns:
            available_date = pd.to_datetime(macro_data.pop("AvailableDate"), errors="coerce")
        macro_data = macro_data.apply(pd.to_numeric, errors="coerce")
        if available_date is not None:
            macro_data = macro_data.loc[available_date.notna()].copy()
            macro_data.index = pd.DatetimeIndex(available_date.dropna()).to_period("M").to_timestamp("M")
        macro_data = macro_data.resample("ME").last()
        macro_data.attrs.update(
            {
                "data_vintage": "user_point_in_time" if available_date is not None else "user_supplied",
                "point_in_time": available_date is not None,
                "availability_aligned": available_date is not None,
            }
        )
        returns = _normalize_columns(returns)
        returns = returns.dropna(how="all")
        if returns.empty or not any(returns[column].notna().any() for column in returns.columns):
            raise ValueError("The asset CSV has no usable numeric data.")
        return (
            macro_data,
            returns,
            list(returns.columns),
            growth_col,
            inflation_col,
            "Loaded data from CSV uploads.",
        )

    raise ValueError(f"Unknown data source: {source}")


def _frame_preview(frame: pd.DataFrame, columns: list[str] | None = None, rows: int = 60) -> dict[str, Any]:
    preview = frame.tail(rows)
    selected = list(preview.columns) if columns is None else columns
    records = preview.loc[:, [col for col in selected if col in preview.columns]].reset_index(names="Date")
    return {
        "columns": [str(column) for column in records.columns],
        "rows": [
            [None if pd.isna(value) else _json_value(value) for value in record]
            for record in records.itertuples(index=False, name=None)
        ],
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
        "presets": [{"name": name, "weights": dict(weights)} for name, weights in PORTFOLIO_PRESETS.items()],
        "macro": _frame_preview(macro, columns=[growth_col, inflation_col]),
        "returns": _frame_preview(returns),
        "synthetic": returns.attrs.get("synthetic_report", {}),
        "data_timing": {
            "vintage": macro.attrs.get("data_vintage", "user_supplied"),
            "point_in_time": bool(macro.attrs.get("point_in_time", False)),
            "availability_aligned": bool(macro.attrs.get("availability_aligned", False)),
        },
    }


def _threshold_value(raw: Any) -> str | float:
    if isinstance(raw, str) and raw.startswith("fixed:"):
        return float(raw.split(":", 1)[1])
    return raw if raw in {"median", "mean"} else float(raw)


def _asset_count(payload: Mapping[str, Any]) -> int:
    selected = parse_tickers(payload.get("selected_tickers", []))
    if selected:
        return len(selected)
    weights = payload.get("weights") or {}
    if isinstance(weights, Mapping) and weights:
        return len(weights)
    return max(len(parse_tickers(payload.get("tickers", []))), 1)


def _chunk_size_value(payload: Mapping[str, Any]) -> int | None:
    raw = payload.get("chunk_size")
    paths = int(payload.get("paths", 3000))
    periods = int(payload.get("periods", 120))
    if raw is None or raw == "":
        # Hold the dominant periods x chunk x assets transient near the default
        # eight-asset, 120-period footprint instead of scaling by horizon alone.
        assets = _asset_count(payload)
        target_chunk = max(500, int(round(5000 * 120 * 8 / max(periods * assets, 1))))
        target_chunk = min(target_chunk, 5000)
        return target_chunk if paths > target_chunk else None
    chunk_size = int(raw)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive or empty (no chunking).")
    return chunk_size


def _workers_value(payload: Mapping[str, Any]) -> int:
    raw = payload.get("workers")
    if raw is None or raw == "":
        return 1
    workers = int(raw)
    if workers < 1 or workers > MAX_WORKERS:
        raise ValueError(f"workers must be between 1 and {MAX_WORKERS}.")
    return workers


def simulation_resource_estimate(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Describe the adaptive execution plan for one simulation request."""

    periods = int(payload.get("periods", 120))
    paths = int(payload.get("paths", 3000))
    assets = _asset_count(payload)
    workers = _workers_value(payload)
    chunk_size = _chunk_size_value(payload) or paths
    chunk_size = min(chunk_size, paths)
    return {
        "periods": periods,
        "paths": paths,
        "assets": assets,
        "workers": workers,
        "chunk_size": chunk_size,
        "joint_macro": bool(payload.get("joint_macro", False)),
        "dynamic_correlation": bool(payload.get("dynamic_correlation", False)),
        "work_units": periods * paths * assets,
    }


def _validate_simulation_size(payload: Mapping[str, Any]) -> dict[str, Any]:
    estimate = simulation_resource_estimate(payload)
    if estimate["periods"] < 1 or estimate["periods"] > MAX_PERIODS:
        raise ValueError(f"periods must be between 1 and {MAX_PERIODS}.")
    if estimate["paths"] < 1 or estimate["paths"] > MAX_PATHS:
        raise ValueError(f"paths must be between 1 and {MAX_PATHS:,}.")
    return estimate


def scenario_kwargs(payload: Mapping[str, Any]) -> dict[str, Any]:
    _validate_simulation_size(payload)
    distribution = str(payload.get("distribution", "normal")).lower().replace("-", "_")
    distribution = DISTRIBUTION_KEYS.get(distribution, distribution)
    if distribution not in DISTRIBUTION_KEYS.values():
        raise ValueError("Unknown return distribution.")
    model_kind = str(payload.get("model", "quadrant")).lower()
    if model_kind not in {"quadrant", "hmm"}:
        raise ValueError("Unknown model kind (expected 'quadrant' or 'hmm').")
    duration_model = str(payload.get("duration_model", "markov")).lower()
    if duration_model not in {"markov", "semi_markov"}:
        raise ValueError("Unknown duration model (expected 'markov' or 'semi_markov').")
    min_regime_duration = int(payload.get("min_regime_duration", 3))
    if min_regime_duration < 1:
        raise ValueError("min_regime_duration must be at least 1.")
    hmm_states = int(payload.get("hmm_states", 4))
    if not 2 <= hmm_states <= 8:
        raise ValueError("hmm_states must be between 2 and 8.")
    threshold_window = int(payload.get("threshold_window", 0) or 0) or None
    if threshold_window is not None and threshold_window <= 0:
        raise ValueError("threshold_window must be positive or zero.")
    rebalance_label = str(payload.get("rebalance", "monthly")).lower()
    if rebalance_label not in REBALANCE_KEYS:
        raise ValueError(f"Unknown rebalancing frequency: {rebalance_label}")
    garch = bool(payload.get("garch", False))
    cost_bps = float(payload.get("cost_bps", 10.0))
    leverage_multiple = float(payload.get("leverage_multiple", 1.0))
    financing_rate = float(payload.get("financing_rate", 0.0)) / 100.0
    financing_inflation_sensitivity = float(payload.get("financing_inflation_sensitivity", 0.0))
    maintenance_margin = float(payload.get("maintenance_margin", 0.0)) / 100.0
    if rebalance_label == "legacy" and not np.isclose(cost_bps, 0.0):
        raise ValueError("Legacy rebalancing does not support transaction costs; set cost_bps to 0.")
    if garch and distribution != "normal":
        raise ValueError("GARCH volatility clustering requires the Normal return distribution.")
    if not np.isfinite(leverage_multiple) or leverage_multiple < 1:
        raise ValueError("leverage_multiple must be at least 1.0.")
    if leverage_multiple != 1.0 and rebalance_label == "legacy":
        raise ValueError("Leverage requires an explicit rebalancing frequency, not legacy accounting.")
    if not np.isfinite(financing_rate) or financing_rate < 0:
        raise ValueError("financing_rate must be a finite, non-negative percentage.")
    if not np.isfinite(financing_inflation_sensitivity) or financing_inflation_sensitivity < 0:
        raise ValueError("financing_inflation_sensitivity must be a finite, non-negative number.")
    if not np.isfinite(maintenance_margin) or not 0 <= maintenance_margin < 1:
        raise ValueError("maintenance_margin must be between 0 and 100 percent.")
    if leverage_multiple == 1.0 and not np.isclose(maintenance_margin, 0.0):
        raise ValueError("maintenance_margin only applies when leverage_multiple is greater than 1.0.")
    if leverage_multiple > 1.0 and maintenance_margin >= 1.0 / leverage_multiple:
        raise ValueError("maintenance_margin must be below the initial equity margin for the selected leverage.")
    start_state = None
    start_label = str(payload.get("start_state", "Stationary"))
    if start_label != "Stationary":
        start_state = REGIME_LOOKUP.get(start_label)
        if start_state is None:
            raise ValueError(f"Unknown start state: {start_label}")
    if model_kind == "hmm":
        start_state = None
    weights = {
        str(ticker).strip().upper(): float(weight)
        for ticker, weight in (payload.get("weights") or {}).items()
    }
    if not weights:
        raise ValueError("Set at least one ticker weight above zero.")
    if not all(np.isfinite(weight) for weight in weights.values()):
        raise ValueError("Portfolio weights must be finite numbers.")
    if np.isclose(sum(weights.values()), 0.0):
        raise ValueError("Portfolio weights must have a non-zero sum.")
    expense_ratios = parse_expense_ratios(payload.get("expense_ratios"))
    base_currency = str(payload.get("base_currency", "USD")).strip().upper()
    if len(base_currency) != 3:
        raise ValueError("Portfolio currency must be a three-letter ISO code.")
    garch_alpha = float(payload.get("garch_alpha", 0.10))
    garch_beta = float(payload.get("garch_beta", 0.85))
    if not 0 <= garch_alpha < 1 or not 0 <= garch_beta < 1 or garch_alpha + garch_beta >= 1:
        raise ValueError("garch_alpha and garch_beta must satisfy 0 <= alpha, beta < 1 and alpha + beta < 1.")
    probabilistic_regimes = bool(payload.get("probabilistic_regimes", False))
    regime_temperature = float(payload.get("regime_temperature", 0.35))
    if not np.isfinite(regime_temperature) or regime_temperature <= 0:
        raise ValueError("regime_temperature must be positive and finite.")
    mean_prior_strength = float(payload.get("mean_prior_strength", 0.0))
    if not np.isfinite(mean_prior_strength) or mean_prior_strength < 0:
        raise ValueError("mean_prior_strength must be finite and non-negative.")
    parameter_draws = int(payload.get("parameter_draws", 0))
    parameter_block_size = int(payload.get("parameter_block_size", 12))
    if not 0 <= parameter_draws <= 100:
        raise ValueError("parameter_draws must be between 0 and 100.")
    if parameter_block_size < 1:
        raise ValueError("parameter_block_size must be positive.")
    joint_macro = bool(payload.get("joint_macro", False))
    macro_transition_weight = float(payload.get("macro_transition_weight", 0.35))
    if not 0 <= macro_transition_weight <= 1:
        raise ValueError("macro_transition_weight must be between 0 and 1.")
    dynamic_correlation = bool(payload.get("dynamic_correlation", False))
    dcc_alpha = float(payload.get("dcc_alpha", 0.04))
    dcc_beta = float(payload.get("dcc_beta", 0.94))
    dcc_asymmetry = float(payload.get("dcc_asymmetry", 0.01))
    if min(dcc_alpha, dcc_beta, dcc_asymmetry) < 0 or dcc_alpha + dcc_beta + dcc_asymmetry >= 1:
        raise ValueError("DCC parameters must be non-negative and sum to less than 1.")
    if model_kind == "hmm" and (parameter_draws or joint_macro or probabilistic_regimes):
        raise ValueError(
            "Probabilistic quadrants, parameter bootstrap, and joint macro paths require the quadrant model."
        )
    if distribution in {"bootstrap", "block_bootstrap"} and (joint_macro or dynamic_correlation):
        raise ValueError("Joint macro paths and dynamic correlation require a parametric return distribution.")
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
        "transaction_cost_bps": cost_bps,
        "asset_expense_ratios": expense_ratios,
        "leverage_multiple": leverage_multiple,
        "financing_rate": financing_rate,
        "financing_inflation_sensitivity": financing_inflation_sensitivity,
        "maintenance_margin": maintenance_margin,
        "contribution": float(payload.get("contribution", 0.0)),
        "withdrawal": float(payload.get("withdrawal", 0.0)),
        "base_currency": base_currency,
        "risk_free_rate": float(payload.get("risk_free_rate", 0.0)) / 100.0,
        "annual_inflation": float(payload.get("annual_inflation", 0.0)) / 100.0,
        "model_kind": model_kind,
        "hmm_states": hmm_states,
        "threshold_window": threshold_window,
        "duration_model": duration_model,
        "min_regime_duration": min_regime_duration,
        "garch": garch,
        "garch_alpha": garch_alpha,
        "garch_beta": garch_beta,
        "walk_forward": bool(payload.get("walk_forward", True)),
        "probabilistic_regimes": probabilistic_regimes,
        "regime_temperature": regime_temperature,
        "mean_prior_strength": mean_prior_strength,
        "parameter_draws": parameter_draws,
        "parameter_block_size": parameter_block_size,
        "joint_macro": joint_macro,
        "macro_transition_weight": macro_transition_weight,
        "dynamic_correlation": dynamic_correlation,
        "dcc_alpha": dcc_alpha,
        "dcc_beta": dcc_beta,
        "dcc_asymmetry": dcc_asymmetry,
        "chunk_size": _chunk_size_value(payload),
        "workers": _workers_value(payload),
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


def _wealth_percentiles(wealth: pd.DataFrame) -> pd.DataFrame:
    """Compute per-period wealth percentiles without copying the full matrix."""
    values = wealth.to_numpy(dtype=float)
    quantiles = np.quantile(values, [0.05, 0.50, 0.95], axis=1)
    return pd.DataFrame(quantiles.T, columns=[0.05, 0.50, 0.95])


def _median_period_returns(wealth: pd.DataFrame, payload: Mapping[str, Any]) -> list[float]:
    """Calculate cross-sectional median time-weighted return for each period."""

    values = wealth.to_numpy(dtype=float)
    annual_inflation = float(payload.get("annual_inflation", 0.0)) / 100.0
    contribution = float(payload.get("contribution", 0.0))
    withdrawal = float(payload.get("withdrawal", 0.0))
    medians: list[float] = []
    for period in range(len(values)):
        previous = 100.0 if period == 0 else values[period - 1]
        previous_deflator = (1.0 + annual_inflation) ** (-period / 12.0)
        current_deflator = (1.0 + annual_inflation) ** (-(period + 1) / 12.0)
        denominator = previous * previous_deflator + contribution * previous_deflator
        numerator = values[period] * current_deflator + withdrawal * current_deflator
        with np.errstate(divide="ignore", invalid="ignore"):
            returns = numerator / denominator - 1.0
        returns = np.asarray(returns, dtype=float)
        returns[(denominator <= 0) | (numerator < 0)] = np.nan
        finite = returns[np.isfinite(returns)]
        medians.append(float(np.median(finite)) if finite.size else 0.0)
    return medians


def _max_drawdown_paths(wealth: pd.DataFrame, initial_value: float = 100.0) -> np.ndarray:
    drawdowns = np.empty(wealth.shape[1], dtype=float)
    block = max(1, int(4096))
    for start in range(0, wealth.shape[1], block):
        chunk = wealth.iloc[:, start:start + block].to_numpy(dtype=float)
        chunk_with_initial = np.vstack([np.full(chunk.shape[1], initial_value), chunk])
        running_max = np.maximum.accumulate(chunk_with_initial, axis=0)
        drawdowns[start:start + chunk.shape[1]] = -(chunk_with_initial / running_max - 1.0).min(axis=0)
    return drawdowns


def _sample_distribution(values: np.ndarray, limit: int = 4_000) -> list[float]:
    """Return a deterministic bounded sample while preserving the full range."""

    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size <= limit:
        return clean.tolist()
    indices = np.linspace(0, clean.size - 1, limit, dtype=int)
    return clean[indices].tolist()


def _distribution_summary(values: np.ndarray) -> dict[str, float]:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if not clean.size:
        return {key: 0.0 for key in ("min", "p05", "p25", "median", "p75", "p95", "max", "mean", "std")}
    quantiles = np.quantile(clean, [0.05, 0.25, 0.50, 0.75, 0.95])
    return {
        "min": float(clean.min()),
        "p05": float(quantiles[0]),
        "p25": float(quantiles[1]),
        "median": float(quantiles[2]),
        "p75": float(quantiles[3]),
        "p95": float(quantiles[4]),
        "max": float(clean.max()),
        "mean": float(clean.mean()),
        "std": float(clean.std(ddof=0)),
    }


def _path_analytics(
    wealth: pd.DataFrame,
    result: Any,
    payload: Mapping[str, Any],
    initial_value: float = 100.0,
) -> dict[str, Any]:
    """Build decision-focused path analytics without retaining asset return cubes."""

    values = wealth.to_numpy(dtype=float)
    periods, paths = values.shape
    contribution = float(payload.get("contribution", 0.0))
    withdrawal = float(payload.get("withdrawal", 0.0))
    risk_free_rate = float(payload.get("risk_free_rate", 0.0)) / 100.0
    previous = np.vstack([np.full(paths, initial_value), values[:-1]])
    denominator = previous + contribution
    numerator = values + withdrawal
    with np.errstate(divide="ignore", invalid="ignore"):
        period_returns = numerator / denominator - 1.0
    period_returns[(denominator <= 0) | (numerator < 0)] = np.nan
    annual_return = np.nanmean(period_returns, axis=0) * 12.0
    annual_volatility = np.nanstd(period_returns, axis=0) * np.sqrt(12.0)
    valid_log_returns = np.where(period_returns > -1.0, np.log1p(period_returns), np.nan)
    valid_counts = np.sum(np.isfinite(valid_log_returns), axis=0)
    log_sums = np.nansum(valid_log_returns, axis=0)
    annual_cagr = np.where(
        valid_counts > 0,
        np.exp(log_sums / np.maximum(valid_counts, 1) * 12.0) - 1.0,
        0.0,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        sharpe = (annual_return - risk_free_rate) / annual_volatility
    sharpe[~np.isfinite(sharpe)] = 0.0
    drawdowns = _max_drawdown_paths(wealth, initial_value=initial_value)
    terminal = values[-1]

    invested = initial_value + (contribution - withdrawal) * np.arange(1, periods + 1)
    success = {
        "periods": list(range(1, periods + 1)),
        "survival": np.mean(values > 0.0, axis=1).tolist(),
        "preservation": np.mean(values >= initial_value, axis=1).tolist(),
        "profit": np.mean(values >= np.maximum(invested, 0.0)[:, None], axis=1).tolist(),
    }

    metric_values = {
        "terminal_wealth": terminal,
        "max_drawdown": drawdowns,
        "annualized_return": annual_return,
        "geometric_annualized_return": annual_cagr,
        "annualized_volatility": annual_volatility,
        "sharpe_ratio": sharpe,
    }
    metric_distributions = {
        key: {
            "sample": _sample_distribution(metric),
            "summary": _distribution_summary(metric),
        }
        for key, metric in metric_values.items()
    }

    scenario_targets = (
        ("worst", 0.0),
        ("p05", 0.05),
        ("median", 0.50),
        ("p95", 0.95),
        ("best", 1.0),
    )
    scenarios = []
    for label, quantile in scenario_targets:
        target = float(np.quantile(terminal, quantile))
        path_index = int(np.argmin(np.abs(terminal - target)))
        regime_column = result.regimes[:, path_index]
        if result.regimes.dtype.kind in "iu":
            states = np.asarray(result.states, dtype=object)
            regimes = [str(state) for state in states[regime_column]]
        else:
            regimes = [str(state) for state in regime_column]
        scenarios.append(
            {
                "label": label,
                "terminal": float(terminal[path_index]),
                "wealth": values[:, path_index].tolist(),
                "regimes": regimes,
            }
        )
    sequence_risk = None
    if contribution > 0 and withdrawal == 0:
        low = np.full(paths, -0.99, dtype=float)
        high = np.full(paths, 10.0, dtype=float)
        periods_index = np.arange(1, periods + 1, dtype=float)[:, None]
        interim_cashflow = -contribution
        for _ in range(64):
            midpoint = (low + high) / 2.0
            discount = np.power(1.0 + midpoint[None, :], periods_index)
            npv = -initial_value + np.sum(interim_cashflow / discount, axis=0) + terminal / discount[-1]
            low = np.where(npv > 0, midpoint, low)
            high = np.where(npv > 0, high, midpoint)
        money_weighted = np.power(1.0 + (low + high) / 2.0, 12.0) - 1.0
        money_weighted = np.clip(money_weighted, -1.0, 100.0)
        sequence_drag = money_weighted - annual_cagr
        sample_indices = np.linspace(0, paths - 1, min(paths, 1_000), dtype=int)
        sequence_risk = {
            "points": [
                {
                    "cagr": float(annual_cagr[index]),
                    "mwrr": float(money_weighted[index]),
                    "drag": float(sequence_drag[index]),
                }
                for index in sample_indices
            ],
            "median_drag": float(np.median(sequence_drag)),
            "probability_negative_drag": float(np.mean(sequence_drag < 0.0)),
        }

    return {
        "success": success,
        "metric_distributions": metric_distributions,
        "representative_scenarios": scenarios,
        "sequence_risk": sequence_risk,
    }


def _regime_counts(result: Any) -> dict[str, int]:
    regimes = result.regimes
    if regimes.dtype.kind in "iu":
        codes = regimes.ravel()
        hist = np.bincount(codes, minlength=len(result.states))
        return {str(state): int(count) for state, count in zip(result.states, hist)}
    values, counts = np.unique(regimes.ravel(), return_counts=True)
    return {str(state): int(count) for state, count in zip(values, counts)}


def _simulated_regime_summary(result: Any) -> pd.DataFrame:
    counts = _regime_counts(result)
    total = max(sum(counts.values()), 1)
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


def _validation_response(walk_forward: Any) -> dict[str, Any] | None:
    """Shape the walk-forward validation result for JSON output."""

    if walk_forward is None:
        return None
    splits = walk_forward.splits
    return {
        "summary": {str(key): _json_value(value) for key, value in walk_forward.summary.items()},
        "columns": [str(column) for column in splits.columns],
        "rows": [
            [
                value.strftime("%Y-%m-%d") if isinstance(value, pd.Timestamp) else _json_value(value)
                for value in record
            ]
            for record in splits.tail(60).itertuples(index=False, name=None)
        ],
    }


def _parameter_uncertainty_response(frame: pd.DataFrame | None) -> dict[str, Any] | None:
    if frame is None or frame.empty:
        return None
    metric_columns = [column for column in frame.columns if column != "draw"]
    bands = {
        str(column): {
            "p05": float(frame[column].quantile(0.05)),
            "median": float(frame[column].quantile(0.50)),
            "p95": float(frame[column].quantile(0.95)),
        }
        for column in metric_columns
    }
    return {
        "draws": int(len(frame)),
        "bands": bands,
        "columns": [str(column) for column in frame.columns],
        "rows": [
            [_json_value(value) for value in record]
            for record in frame.itertuples(index=False, name=None)
        ],
    }


def _macro_path_response(result: Any) -> dict[str, Any] | None:
    if result.macro_paths is None or not result.macro_columns:
        return None
    response: dict[str, Any] = {"periods": list(range(1, len(result.macro_paths) + 1)), "series": {}}
    for index, column in enumerate(result.macro_columns):
        values = result.macro_paths[:, :, index]
        quantiles = np.quantile(values, [0.05, 0.50, 0.95], axis=1)
        response["series"][str(column)] = {
            "p05": quantiles[0].tolist(),
            "median": quantiles[1].tolist(),
            "p95": quantiles[2].tolist(),
        }
    return response


def _simulation_start_date(macro: pd.DataFrame) -> str | None:
    if macro is None or macro.empty:
        return None
    last_observed = pd.Timestamp(macro.index.max())
    return (last_observed + pd.DateOffset(months=1)).strftime("%Y-%m-%d")


def build_simulate_response(payload: Mapping[str, Any]) -> dict[str, Any]:
    scenario, selected_tickers, macro = run_scenario_payload(payload)
    model = scenario.model
    result = scenario.result
    wealth = scenario.reporting_wealth if scenario.reporting_wealth is not None else scenario.wealth
    summary = scenario.summary
    growth_col = scenario.model.metadata.get("growth_col", "growth")
    inflation_col = scenario.model.metadata.get("inflation_col", "inflation")
    percentiles = _wealth_percentiles(wealth)
    terminal_values = wealth.iloc[-1].to_numpy(dtype=float)
    regime_timelines: dict[str, list[str]] = {}
    for label, target in (("p05", 0.05), ("median", 0.50), ("p95", 0.95)):
        target_value = float(np.quantile(terminal_values, target))
        path_index = int(np.argmin(np.abs(terminal_values - target_value)))
        column = result.regimes[:, path_index]
        if result.regimes.dtype.kind in "iu":
            states = np.asarray(result.states, dtype=object)
            regime_timelines[label] = [str(state) for state in states[column]]
        else:
            regime_timelines[label] = [str(state) for state in column]
    regime_counts = _regime_counts(result)
    regime_total = max(sum(regime_counts.values()), 1)
    regime_mix = (
        pd.Series(regime_counts, dtype=float)
        .reindex(model.states)
        .fillna(0.0)
        .div(regime_total)
        .rename(index={state: _state_label(state) for state in model.states})
    )
    scatter = macro[[growth_col, inflation_col]].copy()
    scatter["regime"] = scenario.regimes.map(_state_label, na_action="ignore")
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
        _state_label(state): int(moments.observations) for state, moments in model.moments.items()
    }
    diagnostics = scenario.diagnostics.regime_summary.copy()
    simulated_diagnostics = _simulated_regime_summary(result)
    diagnostics = diagnostics.merge(simulated_diagnostics, on="regime", how="left")
    diagnostics["regime"] = diagnostics["regime"].map(_state_label)

    summary_values = {str(key): _json_value(value) for key, value in summary.items()}
    contribution = float(payload.get("contribution", 0.0))
    withdrawal = float(payload.get("withdrawal", 0.0))
    if contribution or withdrawal:
        summary_values["periodic_contribution"] = contribution
        summary_values["periodic_withdrawal"] = withdrawal
        summary_values["total_contributed"] = contribution * len(wealth)
        summary_values["total_withdrawn"] = withdrawal * len(wealth)
        summary_values["net_external_cash_flow"] = (contribution - withdrawal) * len(wealth)

    costs = {
        "weighted_expense_ratio": summary_values.get("weighted_expense_ratio", 0.0),
        "annual_fee_drag": summary_values.get("annual_fee_drag", 0.0),
        "annual_financing_cost": summary_values.get("annual_financing_cost", 0.0),
        "effective_financing_rate": summary_values.get("effective_financing_rate", 0.0),
        "leverage_multiple": summary_values.get("leverage_multiple", 1.0),
        "maintenance_margin": summary_values.get("maintenance_margin", 0.0),
        "margin_calls": summary_values.get("margin_calls", 0),
    }
    path_analytics = _path_analytics(
        wealth,
        result,
        payload,
        initial_value=float(payload.get("initial_value", 100.0)),
    )

    return {
        "ok": True,
        "summary": summary_values,
        "currency": scenario.model.metadata.get("base_currency", "USD"),
        "terms": (
            "real"
            if scenario_kwargs(payload)["annual_inflation"] > 0
            or model.metadata.get("inflation_model") == "joint_macro_path"
            else "nominal"
        ),
        "warnings": list(scenario.diagnostics.warnings),
        "costs": costs,
        "wealth": {
            "periods": list(range(1, len(wealth) + 1)),
            "p05": percentiles[0.05].tolist(),
            "median": percentiles[0.50].tolist(),
            "p95": percentiles[0.95].tolist(),
        },
        "monthly_returns": _median_period_returns(
            wealth,
            {**dict(payload), "annual_inflation": 0.0},
        ),
        "terminal": terminal_values.tolist(),
        "drawdowns": _max_drawdown_paths(wealth).tolist(),
        **path_analytics,
        "regime_timeline": regime_timelines["median"],
        "regime_timelines": regime_timelines,
        "regime_mix": [{"label": label, "share": float(share)} for label, share in regime_mix.items()],
        "transition": {
            "labels": [_state_label(state) for state in model.transition_matrix.index],
            "values": model.transition_matrix.to_numpy(dtype=float).tolist(),
        },
        "macro_scatter": scatter_records,
        "observations": observations,
        "correlations": {
            _state_label(state): {
                "labels": list(model.moments[state].correlation.columns),
                "values": model.moments[state].correlation.to_numpy(dtype=float).tolist(),
            }
            for state in model.states
        },
        "validation": _validation_response(scenario.walk_forward),
        "parameter_uncertainty": _parameter_uncertainty_response(scenario.parameter_uncertainty),
        "regime_probabilities": [
            {
                "state": state,
                "label": _state_label(state),
                "probability": float(
                    model.metadata.get("latest_regime_probabilities", {}).get(state, 0.0)
                ),
            }
            for state in model.states
        ],
        "macro_paths": _macro_path_response(result),
        "methodology": {
            "data_vintage": model.metadata.get("data_vintage", "user_supplied"),
            "point_in_time": bool(model.metadata.get("point_in_time", False)),
            "availability_aligned": bool(model.metadata.get("availability_aligned", False)),
            "macro_lag_periods": int(model.metadata.get("macro_lag_periods", 0)),
            "regime_assignment": model.metadata.get("regime_assignment", "hard"),
            "mean_prior_strength": float(model.metadata.get("mean_prior_strength", 0.0)),
            "parameter_draws": int(payload.get("parameter_draws", 0)),
            "joint_macro": bool(payload.get("joint_macro", False)),
            "dynamic_correlation": bool(payload.get("dynamic_correlation", False)),
            "inflation_model": model.metadata.get("inflation_model", "deterministic"),
        },
        "model_kind": scenario.model.metadata.get("model_kind", "quadrant"),
        "diagnostics": {
            "columns": [str(column) for column in diagnostics.columns],
            "rows": [
                [None if pd.isna(value) else _json_value(value) for value in record]
                for record in diagnostics.itertuples(index=False, name=None)
            ],
        },
        "selected_tickers": selected_tickers,
        "resources": simulation_resource_estimate(payload),
        "start_date": _simulation_start_date(macro),
        "message": (
            f"Simulation complete: {len(wealth)} periods x {wealth.shape[1]} paths. "
            f"Distribution: {scenario.result.distribution}."
        ),
    }


def build_wealth_csv(payload: Mapping[str, Any]) -> dict[str, Any]:
    requested_paths = int(payload.get("paths", DEFAULT_EXPORT_PATHS))
    if requested_paths < 1 or requested_paths > MAX_PATHS:
        raise ValueError(f"paths must be between 1 and {MAX_PATHS:,}.")
    export_paths = int(payload.get("export_paths", DEFAULT_EXPORT_PATHS))
    export_paths = max(1, min(export_paths, requested_paths, MAX_EXPORT_PATHS))
    original_chunk_size = _chunk_size_value(payload) or requested_paths
    # Re-run at least the original first chunk so seeded vectorized draws line
    # up exactly with the completed simulation, then retain only the bounded
    # number of columns requested for the CSV.
    replayed_paths = min(
        requested_paths,
        max(export_paths, min(original_chunk_size, MAX_EXPORT_PATHS)),
    )
    export_payload = dict(payload)
    export_payload["paths"] = replayed_paths
    export_payload["workers"] = 1
    export_payload["walk_forward"] = False
    scenario, selected_tickers, _ = run_scenario_payload(export_payload)
    source_wealth = scenario.reporting_wealth if scenario.reporting_wealth is not None else scenario.wealth
    wealth = source_wealth.iloc[:, :export_paths].copy()
    wealth.insert(0, "period", range(1, len(wealth) + 1))
    return {
        "ok": True,
        "csv": wealth.to_csv(index=False),
        "tickers": selected_tickers,
        "exported_paths": export_paths,
        "requested_paths": requested_paths,
        "replayed_paths": replayed_paths,
        "sampled": export_paths < requested_paths,
    }


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
    asset_currencies, fx_rates = prepare_fx_rates(
        returns, selected_tickers, kwargs["base_currency"], currency_map
    )
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
