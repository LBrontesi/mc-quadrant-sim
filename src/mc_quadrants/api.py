"""UI-agnostic API layer shared by every frontend (web, Streamlit, Gradio).

All data loading, scenario building, and result shaping lives here so that
the simulation methodology is identical regardless of the interface. Frontends
should only call these functions and render the returned dicts/frames.
"""

from __future__ import annotations

import io
import os
import re
from collections.abc import Mapping
from dataclasses import replace
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from mc_quadrants.data import (
    fetch_yahoo_fx_rates,
    load_market_data,
    prices_to_returns,
)
from mc_quadrants.decumulation import (
    GuardrailSettings,
    inflation_index as withdrawal_inflation_index,
    normalize_decumulation,
    success_mask,
    wilson_interval,
)
from mc_quadrants.pipeline import compare_distributions, run_scenario
from mc_quadrants.regimes import REGIME_ORDER
from mc_quadrants.simulation import simulate_portfolio_paths
from mc_quadrants.tax_policy import available_tax_countries, resolve_tax_selection
from mc_quadrants.taxes import (
    CONTRIBUTION_ALLOCATION_MODES,
    ITALY_TAX_CATEGORIES,
    normalize_italy_tax_metadata,
)

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
    "effective_risk_free_rate",
    "maintenance_margin",
    "annual_wealth_tax_rate",
    "terminal_tax_drag_percent",
    "wrapper_advantage_percent",
    "tax_drag_cagr",
    "effective_tax_rate",
    "probability_of_loss",
    "goal_success_probability",
    "risk_of_ruin",
    "unrecovered_at_horizon",
    "worst_rolling_return",
    "worst_rolling_return_p05",
    "median_worst_rolling_return",
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
    "target_wealth",
    "expected_goal_shortfall",
    "capital_gains_tax",
    "wealth_tax",
    "terminal_liquidation_tax",
    "taxes_paid",
    "realized_gains",
    "realized_losses",
    "loss_carryforward",
    "investment_income_tax",
    "foreign_withholding_tax",
    "financial_transaction_tax",
    "stamp_duty",
    "ivafe",
    "expired_losses",
    "gross_terminal_wealth_median",
    "after_tax_terminal_wealth_median",
    "terminal_tax_drag_median",
    "wrapper_terminal_p05",
    "wrapper_terminal_median",
    "wrapper_terminal_p95",
    "wrapper_advantage_median",
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
    if "_months" in key:
        return f"{numeric:.1f} mo"
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

# Standard portfolio presets mapped onto the available ticker universe.
# Approximations are noted per preset (for example, IEF stands in for
# long-term treasuries and SHY for short-term/cash holdings).
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
    "buy_hold": 0,
    "monthly": 1,
    "quarterly": 3,
    "annual": 12,
}
MAX_PERIODS = 360
MAX_PATHS = 500_000
MAX_WORKERS = 16
MAX_REPORTING_PATHS = 5_000
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


def parse_numeric_map(
    raw: str | Mapping[str, Any] | None,
    kind: str,
    *,
    scale: float = 1.0,
) -> dict[str, float]:
    if not raw:
        return {}
    items = raw.items() if isinstance(raw, Mapping) else parse_pair_map(str(raw), kind).items()
    parsed: dict[str, float] = {}
    for asset, raw_value in items:
        value = float(raw_value) * scale
        if not np.isfinite(value):
            raise ValueError(f"{kind} for {asset} must be finite.")
        parsed[str(asset).strip().upper()] = value
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
        requested_rate_col = str(payload.get("rate_col", "interest_rate")).strip()
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
                "rate_col": (
                    requested_rate_col
                    if requested_rate_col and requested_rate_col in macro_data.columns
                    else None
                ),
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
    rate_col = macro.attrs.get("rate_col")
    macro_columns = [growth_col, inflation_col]
    if rate_col and rate_col in macro.columns:
        macro_columns.append(str(rate_col))
    return {
        "ok": True,
        "tickers": tickers,
        "default_tickers": default_selected_tickers(tickers),
        "growth_col": growth_col,
        "inflation_col": inflation_col,
        "rate_col": rate_col,
        "message": message,
        "coverage": _coverage(returns),
        "presets": [{"name": name, "weights": dict(weights)} for name, weights in PORTFOLIO_PRESETS.items()],
        "macro": _frame_preview(macro, columns=macro_columns),
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
        paths = int(payload.get("paths", 3000))
        periods = int(payload.get("periods", 120))
        assets = _asset_count(payload)
        chunk_size = min(_chunk_size_value(payload) or paths, paths)
        chunk_count = max(1, (paths + chunk_size - 1) // chunk_size)
        work_units = periods * paths * assets
        if chunk_count < 2 or work_units < 2_000_000:
            return 1
        configured_cap = int(os.getenv("MC_SIM_MAX_AUTO_WORKERS", "4"))
        auto_cap = max(1, min(configured_cap, MAX_WORKERS))
        return min(max(os.cpu_count() or 1, 1), auto_cap, chunk_count)
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
    for key in ("initial_value", "target_wealth"):
        if key in payload:
            value = float(payload[key])
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be positive and finite.")
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
    duration_model = str(payload.get("duration_model", "semi_markov")).lower()
    if duration_model not in {"markov", "semi_markov"}:
        raise ValueError("Unknown duration model (expected 'markov' or 'semi_markov').")
    min_regime_duration = int(payload.get("min_regime_duration", 5))
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
    default_cost_bps = 0.0 if rebalance_label in {"legacy", "buy_hold"} else 10.0
    cost_bps = float(payload.get("cost_bps", default_cost_bps))
    leverage_multiple = float(payload.get("leverage_multiple", 1.0))
    financing_rate = float(payload.get("financing_rate", 0.0)) / 100.0
    financing_inflation_sensitivity = float(payload.get("financing_inflation_sensitivity", 0.0))
    maintenance_margin = float(payload.get("maintenance_margin", 0.0)) / 100.0
    if rebalance_label in {"legacy", "buy_hold"} and not np.isclose(cost_bps, 0.0):
        raise ValueError(
            "Legacy and buy-and-hold accounting do not support transaction costs; set cost_bps to 0."
        )
    if garch and distribution != "normal":
        raise ValueError("GARCH volatility clustering requires the Normal return distribution.")
    if not np.isfinite(leverage_multiple) or leverage_multiple < 1:
        raise ValueError("leverage_multiple must be at least 1.0.")
    if leverage_multiple != 1.0 and rebalance_label in {"legacy", "buy_hold"}:
        raise ValueError("Leverage requires monthly, quarterly, or annual rebalancing.")
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
    tax_selection = resolve_tax_selection(
        payload.get("tax_country"),
        payload.get("tax_regime"),
    )
    tax_country = tax_selection.country
    tax_regime = tax_selection.regime
    raw_tax_categories = payload.get("asset_tax_categories", "")
    if isinstance(raw_tax_categories, Mapping):
        tax_categories = {
            str(asset).strip().upper(): str(category).strip().lower()
            for asset, category in raw_tax_categories.items()
        }
    else:
        tax_categories = {
            asset: category.lower()
            for asset, category in parse_pair_map(str(raw_tax_categories or ""), "tax category").items()
        }
    unknown_tax_categories = sorted(set(tax_categories.values()) - ITALY_TAX_CATEGORIES)
    if tax_country == "IT" and unknown_tax_categories:
        allowed = ", ".join(sorted(ITALY_TAX_CATEGORIES))
        raise ValueError(
            f"Unknown Italian tax category '{unknown_tax_categories[0]}'. Expected one of: {allowed}."
        )
    raw_tax_metadata = payload.get("asset_tax_metadata") or {}
    if not isinstance(raw_tax_metadata, Mapping):
        raise ValueError("asset_tax_metadata must be an object keyed by asset symbol.")
    tax_metadata = {
        str(asset).strip().upper(): dict(values)
        for asset, values in raw_tax_metadata.items()
        if isinstance(values, Mapping)
    }
    if len(tax_metadata) != len(raw_tax_metadata):
        raise ValueError("Every asset_tax_metadata value must be an object.")
    if tax_country == "IT":
        tax_metadata = normalize_italy_tax_metadata(
            list(weights),
            tax_categories,
            tax_metadata,
        )
        tax_categories = {
            asset: str(values["category"])
            for asset, values in tax_metadata.items()
        }
    italy_wealth_tax = float(payload.get("italy_wealth_tax", 0.20)) / 100.0
    if not np.isfinite(italy_wealth_tax) or not 0 <= italy_wealth_tax < 1:
        raise ValueError("italy_wealth_tax must be between 0 and 100 percent.")
    italy_wealth_tax_mode = str(payload.get("italy_wealth_tax_mode", "auto")).strip().lower()
    if italy_wealth_tax_mode not in {"auto", "stamp_duty", "ivafe", "both", "none"}:
        raise ValueError("italy_wealth_tax_mode must be auto, stamp_duty, ivafe, both, or none.")
    contribution_allocation = str(payload.get("contribution_allocation", "target")).strip().lower()
    if contribution_allocation not in CONTRIBUTION_ALLOCATION_MODES:
        allowed = ", ".join(sorted(CONTRIBUTION_ALLOCATION_MODES))
        raise ValueError(f"contribution_allocation must be one of: {allowed}.")
    if contribution_allocation == "underweight_first" and rebalance_label == "legacy":
        raise ValueError("Underweight-first contributions require holdings-based accounting.")
    if contribution_allocation == "underweight_first" and not np.isclose(leverage_multiple, 1.0):
        raise ValueError("Underweight-first contributions are not available with leverage.")
    if tax_selection.policy is not None:
        tax_selection.policy.validate(
            rebalance_frequency=REBALANCE_KEYS[rebalance_label],
            leverage_multiple=leverage_multiple,
            target_weights=np.asarray(list(weights.values()), dtype=float),
        )
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
    regime_smoothing_window = int(payload.get("regime_smoothing_window", 3))
    regime_hysteresis = float(payload.get("regime_hysteresis", 0.15))
    regime_confirmation_periods = int(payload.get("regime_confirmation_periods", 2))
    duration_prior_strength = float(payload.get("duration_prior_strength", 8.0))
    if not 1 <= regime_smoothing_window <= 24:
        raise ValueError("regime_smoothing_window must be between 1 and 24.")
    if not np.isfinite(regime_hysteresis) or not 0 <= regime_hysteresis <= 2:
        raise ValueError("regime_hysteresis must be between 0 and 2.")
    if not 1 <= regime_confirmation_periods <= 12:
        raise ValueError("regime_confirmation_periods must be between 1 and 12.")
    if not np.isfinite(duration_prior_strength) or duration_prior_strength <= 0:
        raise ValueError("duration_prior_strength must be positive and finite.")
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
    macro_parameter_uncertainty = bool(payload.get("macro_parameter_uncertainty", True))
    macro_model = str(payload.get("macro_model", "bvar_ensemble")).strip().lower()
    if macro_model not in {"ridge_var", "bvar", "bvar_ensemble"}:
        raise ValueError("macro_model must be ridge_var, bvar, or bvar_ensemble.")
    structural_returns = bool(payload.get("structural_returns", False))
    raw_asset_classes = payload.get("asset_classes", "")
    parsed_asset_classes = (
        {str(asset).strip().upper(): str(value).strip().upper() for asset, value in raw_asset_classes.items()}
        if isinstance(raw_asset_classes, Mapping)
        else parse_pair_map(raw_asset_classes, "asset class")
    )
    asset_classes = {asset: value.lower() for asset, value in parsed_asset_classes.items()}
    asset_durations = parse_numeric_map(payload.get("asset_durations"), "duration")
    asset_income_yields = parse_numeric_map(
        payload.get("asset_income_yields"), "income yield", scale=0.01
    )
    state_dependent_liquidity = bool(payload.get("state_dependent_liquidity", False))
    raw_liquidity = payload.get("state_transaction_cost_multipliers") or {}
    if not isinstance(raw_liquidity, Mapping):
        raise ValueError("state_transaction_cost_multipliers must be an object.")
    liquidity_multipliers = {str(state): float(value) for state, value in raw_liquidity.items()}
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
    periods = int(payload.get("periods", 120))
    withdrawal_start_raw = payload.get("withdrawal_start_period", 1)
    try:
        withdrawal_start_value = float(withdrawal_start_raw)
        withdrawal_start_period = int(withdrawal_start_value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("withdrawal_start_period must be an integer.") from exc
    if not np.isfinite(withdrawal_start_value) or not np.isclose(
        withdrawal_start_value, withdrawal_start_period
    ):
        raise ValueError("withdrawal_start_period must be an integer.")
    if not 1 <= withdrawal_start_period <= periods:
        raise ValueError(
            "withdrawal_start_period must be between 1 and the simulation periods."
        )
    raw_decumulation = payload.get("decumulation")
    decumulation = (
        normalize_decumulation(
            raw_decumulation,
            periods=periods,
            legacy_withdrawal=float(payload.get("withdrawal", 0.0)),
            legacy_start_period=withdrawal_start_period,
            annual_inflation_fallback=float(payload.get("annual_inflation", 0.0)) / 100.0,
        ).to_dict()
        if raw_decumulation is not None
        else None
    )
    return {
        "growth_threshold": _threshold_value(payload.get("growth_threshold", "median")),
        "inflation_threshold": _threshold_value(payload.get("inflation_threshold", "median")),
        "rate_col": str(payload.get("rate_col", "interest_rate")).strip() or None,
        "periods": periods,
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
        "state_dependent_liquidity": state_dependent_liquidity,
        "state_transaction_cost_multipliers": liquidity_multipliers,
        "asset_expense_ratios": expense_ratios,
        "tax_country": tax_country,
        "tax_regime": tax_regime,
        "asset_tax_categories": tax_categories,
        "asset_tax_metadata": tax_metadata,
        "italy_annual_wealth_tax": italy_wealth_tax,
        "italy_wealth_tax_mode": italy_wealth_tax_mode,
        "tax_terminal_liquidation": bool(payload.get("tax_terminal_liquidation", True)),
        "tax_start_date": str(payload.get("tax_start_date", "")).strip() or None,
        "tax_wrapper_benchmark": bool(payload.get("tax_wrapper_benchmark", False)),
        "leverage_multiple": leverage_multiple,
        "financing_rate": financing_rate,
        "financing_inflation_sensitivity": financing_inflation_sensitivity,
        "maintenance_margin": maintenance_margin,
        "contribution": float(payload.get("contribution", 0.0)),
        "contribution_allocation": contribution_allocation,
        "withdrawal": float(payload.get("withdrawal", 0.0)),
        "withdrawal_start_period": withdrawal_start_period,
        "decumulation": decumulation,
        "initial_value": float(payload.get("initial_value", 100.0)),
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
        "regime_smoothing_window": regime_smoothing_window,
        "regime_hysteresis": regime_hysteresis,
        "regime_confirmation_periods": regime_confirmation_periods,
        "duration_prior_strength": duration_prior_strength,
        "mean_prior_strength": mean_prior_strength,
        "parameter_draws": parameter_draws,
        "parameter_block_size": parameter_block_size,
        "joint_macro": joint_macro,
        "macro_transition_weight": macro_transition_weight,
        "macro_parameter_uncertainty": macro_parameter_uncertainty,
        "macro_model": macro_model,
        "structural_returns": structural_returns,
        "asset_classes": asset_classes,
        "asset_durations": asset_durations,
        "asset_income_yields": asset_income_yields,
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
    withdrawal_start_period = int(payload.get("withdrawal_start_period", 1))
    funded_paths = wealth.attrs.get("withdrawal_funded")
    withdrawal_cpi = wealth.attrs.get("withdrawal_cpi")
    if funded_paths is not None:
        funded_paths = np.asarray(funded_paths, dtype=float)
        if funded_paths.shape != values.shape:
            funded_paths = None
    if withdrawal_cpi is not None:
        withdrawal_cpi = np.asarray(withdrawal_cpi, dtype=float)
        if withdrawal_cpi.shape != values.shape:
            withdrawal_cpi = None
    medians: list[float] = []
    for period in range(len(values)):
        previous = float(payload.get("initial_value", 100.0)) if period == 0 else values[period - 1]
        previous_deflator = (1.0 + annual_inflation) ** (-period / 12.0)
        current_deflator = (1.0 + annual_inflation) ** (-(period + 1) / 12.0)
        denominator = previous * previous_deflator + contribution * previous_deflator
        active_withdrawal = (
            funded_paths[period] / np.maximum(withdrawal_cpi[period], 1e-300)
            if funded_paths is not None and withdrawal_cpi is not None
            else funded_paths[period]
            if funded_paths is not None
            else withdrawal if period + 1 >= withdrawal_start_period else 0.0
        )
        numerator = (
            values[period] * current_deflator
            + active_withdrawal * current_deflator
        )
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


def _reporting_indices(paths: int, limit: int = MAX_REPORTING_PATHS) -> np.ndarray:
    """Select stable path indexes for browser reporting and paired analysis."""

    if paths <= 0:
        return np.empty(0, dtype=int)
    return np.linspace(0, paths - 1, min(paths, limit), dtype=int)


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


def _drawdown_duration_metrics(values: np.ndarray, initial_value: float) -> dict[str, float]:
    """Summarize time underwater and completed recovery episodes."""

    periods, paths = values.shape
    running_peak = np.full(paths, initial_value, dtype=float)
    current_duration = np.zeros(paths, dtype=np.int32)
    maximum_duration = np.zeros(paths, dtype=np.int32)
    recovery_histogram = np.zeros(periods + 1, dtype=np.int64)
    for period_values in values:
        recovered = period_values >= running_peak
        completed = current_duration[recovered & (current_duration > 0)]
        if completed.size:
            recovery_histogram += np.bincount(completed, minlength=periods + 1)[: periods + 1]
        running_peak = np.maximum(running_peak, period_values)
        underwater = period_values < running_peak
        current_duration = np.where(underwater, current_duration + 1, 0)
        maximum_duration = np.maximum(maximum_duration, current_duration)

    def recovery_quantile(probability: float) -> float:
        total = int(recovery_histogram.sum())
        if total == 0:
            return 0.0
        threshold = max(1, int(np.ceil(total * probability)))
        return float(np.searchsorted(np.cumsum(recovery_histogram), threshold))

    return {
        "max_underwater_months_mean": float(maximum_duration.mean()),
        "max_underwater_months_p95": float(np.quantile(maximum_duration, 0.95)),
        "max_underwater_months_worst": float(maximum_duration.max()),
        "recovery_months_median": recovery_quantile(0.50),
        "recovery_months_p95": recovery_quantile(0.95),
        "unrecovered_at_horizon": float((current_duration > 0).mean()),
    }


def _rolling_return_metrics(period_returns: np.ndarray, window: int = 12) -> dict[str, float]:
    """Summarize each path's worst compounded rolling return."""

    periods = period_returns.shape[0]
    window = max(1, min(int(window), periods))
    finite = np.isfinite(period_returns)
    clipped = np.clip(np.where(finite, period_returns, 0.0), -0.999999999, None)
    logs = np.where(finite, np.log1p(clipped), 0.0)
    cumulative_logs = np.vstack([np.zeros((1, logs.shape[1])), np.cumsum(logs, axis=0)])
    cumulative_counts = np.vstack([np.zeros((1, logs.shape[1]), dtype=int), np.cumsum(finite, axis=0)])
    rolling_logs = cumulative_logs[window:] - cumulative_logs[:-window]
    rolling_counts = cumulative_counts[window:] - cumulative_counts[:-window]
    rolling_returns = np.where(rolling_counts == window, np.expm1(rolling_logs), np.inf)
    path_worst = np.min(rolling_returns, axis=0)
    path_worst = path_worst[np.isfinite(path_worst)]
    if not path_worst.size:
        return {
            "rolling_window_months": float(window),
            "worst_rolling_return": 0.0,
            "worst_rolling_return_p05": 0.0,
            "median_worst_rolling_return": 0.0,
        }
    return {
        "rolling_window_months": float(window),
        "worst_rolling_return": float(path_worst.min()),
        "worst_rolling_return_p05": float(np.quantile(path_worst, 0.05)),
        "median_worst_rolling_return": float(np.median(path_worst)),
    }


def _drawdown_chart_analytics(
    values: np.ndarray,
    initial_value: float,
    max_episode_paths: int = 1_000,
    max_episodes: int = 4_000,
) -> dict[str, Any]:
    """Build bounded drawdown bands and representative depth-duration episodes."""

    periods, paths = values.shape
    running_peak = np.full(paths, initial_value, dtype=float)
    drawdown_bands = {key: [] for key in ("p05", "median", "p95")}
    recovery_bands = {key: [] for key in ("p05", "median", "p95")}
    recovery_capped = False
    for period_values in values:
        running_peak = np.maximum(running_peak, period_values)
        with np.errstate(divide="ignore", invalid="ignore"):
            drawdown = period_values / running_peak - 1.0
            raw_recovery = running_peak / np.maximum(period_values, initial_value * 1e-12) - 1.0
        drawdown = np.nan_to_num(drawdown, nan=-1.0, neginf=-1.0, posinf=0.0)
        recovery_capped = recovery_capped or bool(np.any(raw_recovery > 99.0))
        recovery = np.nan_to_num(raw_recovery, nan=99.0, neginf=99.0, posinf=99.0)
        recovery = np.clip(recovery, 0.0, 99.0)
        drawdown_quantiles = np.quantile(drawdown, [0.05, 0.50, 0.95])
        recovery_quantiles = np.quantile(recovery, [0.05, 0.50, 0.95])
        for key, value in zip(drawdown_bands, drawdown_quantiles, strict=True):
            drawdown_bands[key].append(float(value))
        for key, value in zip(recovery_bands, recovery_quantiles, strict=True):
            recovery_bands[key].append(float(value))

    source_indices = _reporting_indices(paths, max_episode_paths)
    episodes: list[tuple[int, int, float, bool]] = []
    for path_index in source_indices:
        peak = initial_value
        duration = 0
        maximum_depth = 0.0
        for period_value in values[:, path_index]:
            if period_value >= peak:
                if duration:
                    episodes.append((int(path_index), duration, maximum_depth, True))
                peak = float(period_value)
                duration = 0
                maximum_depth = 0.0
                continue
            duration += 1
            maximum_depth = max(maximum_depth, 1.0 - float(period_value) / peak)
        if duration:
            episodes.append((int(path_index), duration, maximum_depth, False))
    if len(episodes) > max_episodes:
        episode_indices = np.linspace(0, len(episodes) - 1, max_episodes, dtype=int)
        episodes = [episodes[index] for index in episode_indices]

    return {
        "drawdown_fan": {
            "periods": list(range(1, periods + 1)),
            **drawdown_bands,
        },
        "recovery_required": {
            "periods": list(range(1, periods + 1)),
            **recovery_bands,
            "capped": recovery_capped,
            "cap": 99.0,
        },
        "drawdown_episodes": {
            "points": [
                {
                    "path": path_index,
                    "duration_months": duration,
                    "depth": depth,
                    "recovered": recovered,
                }
                for path_index, duration, depth, recovered in episodes
            ],
            "source_paths": int(len(source_indices)),
            "total_paths": int(paths),
            "sampled": bool(len(source_indices) < paths or len(episodes) == max_episodes),
        },
    }


def _rolling_horizon_analytics(period_returns: np.ndarray) -> dict[str, Any]:
    """Summarize annualized returns across rolling investment horizons."""

    periods, paths = period_returns.shape
    reporting_indices = _reporting_indices(paths)
    sampled = period_returns[:, reporting_indices]
    finite = np.isfinite(sampled) & (sampled > -1.0)
    log_returns = np.where(finite, np.log1p(np.where(finite, sampled, 0.0)), 0.0)
    cumulative_logs = np.vstack(
        [np.zeros((1, sampled.shape[1])), np.cumsum(log_returns, axis=0)]
    )
    cumulative_counts = np.vstack(
        [np.zeros((1, sampled.shape[1]), dtype=int), np.cumsum(finite, axis=0)]
    )
    horizons = sorted(
        {horizon for horizon in (12, 36, 60, 120, 240, 360, periods) if 12 <= horizon <= periods}
    )
    response: dict[str, Any] = {
        "months": [],
        "p05": [],
        "median": [],
        "p95": [],
        "probability_of_loss": [],
        "sample_paths": int(len(reporting_indices)),
        "total_paths": int(paths),
    }
    for horizon in horizons:
        rolling_logs = cumulative_logs[horizon:] - cumulative_logs[:-horizon]
        rolling_counts = cumulative_counts[horizon:] - cumulative_counts[:-horizon]
        valid_logs = rolling_logs[rolling_counts == horizon]
        if not valid_logs.size:
            continue
        with np.errstate(over="ignore", invalid="ignore"):
            annualized = np.expm1(np.clip(valid_logs * 12.0 / horizon, -20.0, 20.0))
        annualized = annualized[np.isfinite(annualized)]
        if not annualized.size:
            continue
        quantiles = np.quantile(annualized, [0.05, 0.50, 0.95])
        response["months"].append(int(horizon))
        response["p05"].append(float(quantiles[0]))
        response["median"].append(float(quantiles[1]))
        response["p95"].append(float(quantiles[2]))
        response["probability_of_loss"].append(float(np.mean(annualized < 0.0)))
    return response


def _goal_probability_curve(
    terminal: np.ndarray,
    initial_value: float,
    target_wealth: float,
) -> dict[str, list[float]]:
    """Return an exact terminal target-success curve over useful wealth levels."""

    clean = np.sort(np.asarray(terminal, dtype=float)[np.isfinite(terminal)])
    if not clean.size:
        return {"targets": [], "success_probability": []}
    lower, upper = np.quantile(clean, [0.01, 0.99])
    targets = np.unique(
        np.concatenate(
            [
                np.linspace(max(0.0, float(lower)), float(upper), 31),
                np.asarray([initial_value, target_wealth], dtype=float),
            ]
        )
    )
    successes = 1.0 - np.searchsorted(clean, targets, side="left") / clean.size
    return {
        "targets": targets.tolist(),
        "success_probability": successes.tolist(),
    }


def _path_analytics(
    wealth: pd.DataFrame,
    result: Any,
    payload: Mapping[str, Any],
    model: Any | None = None,
    initial_value: float = 100.0,
    drawdowns: np.ndarray | None = None,
) -> dict[str, Any]:
    """Build decision-focused path analytics without retaining asset return cubes."""

    values = wealth.to_numpy(dtype=float)
    periods, paths = values.shape
    contribution = float(payload.get("contribution", 0.0))
    withdrawal = float(payload.get("withdrawal", 0.0))
    withdrawal_start_period = int(payload.get("withdrawal_start_period", 1))
    risk_free_rate = float(payload.get("risk_free_rate", 0.0)) / 100.0
    target_wealth = float(payload.get("target_wealth", initial_value * 2.0))
    if not np.isfinite(target_wealth) or target_wealth <= 0:
        raise ValueError("target_wealth must be positive and finite.")
    previous = np.vstack([np.full(paths, initial_value), values[:-1]])
    denominator = previous + contribution
    funded_paths = wealth.attrs.get("withdrawal_funded")
    withdrawal_cpi = wealth.attrs.get("withdrawal_cpi")
    if funded_paths is not None:
        funded_values = np.asarray(funded_paths, dtype=float)
        if funded_values.shape != values.shape:
            funded_values = None
    else:
        funded_values = None
    if funded_values is not None and withdrawal_cpi is not None:
        cpi_values = np.asarray(withdrawal_cpi, dtype=float)
        if cpi_values.shape == values.shape:
            funded_values = funded_values / np.maximum(cpi_values, 1e-300)
    if funded_values is None:
        withdrawal_schedule = (
            np.arange(1, periods + 1, dtype=int) >= withdrawal_start_period
        ).astype(float)
        funded_values = withdrawal * withdrawal_schedule[:, None]
    numerator = values + funded_values
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
    nominal_risk_free = np.full((periods, paths), risk_free_rate, dtype=float)
    annual_inflation = np.full(
        (periods, paths),
        float(payload.get("annual_inflation", 0.0)) / 100.0,
        dtype=float,
    )
    if model is not None and result.macro_paths is not None:
        dynamics = model.metadata.get("macro_dynamics", {})
        rate_col = model.metadata.get("rate_col")
        inflation_col = model.metadata.get("inflation_col")
        if rate_col in result.macro_columns:
            nominal_risk_free = result.macro_paths[:, :, result.macro_columns.index(rate_col)].astype(float)
            if bool(dynamics.get("rate_is_percent", model.metadata.get("rate_is_percent", False))):
                nominal_risk_free /= 100.0
            nominal_risk_free = np.clip(nominal_risk_free, -0.05, 0.50)
        if inflation_col in result.macro_columns:
            annual_inflation = result.macro_paths[
                :, :, result.macro_columns.index(inflation_col)
            ].astype(float)
            if bool(dynamics.get("inflation_is_percent", False)):
                annual_inflation /= 100.0
            annual_inflation = np.clip(annual_inflation, -0.10, 0.50)
    real_risk_free = (1.0 + nominal_risk_free) / (1.0 + annual_inflation) - 1.0
    path_risk_free = np.mean(real_risk_free, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        sharpe = (annual_return - path_risk_free) / annual_volatility
    sharpe[~np.isfinite(sharpe)] = 0.0
    if drawdowns is None:
        drawdowns = _max_drawdown_paths(wealth, initial_value=initial_value)
    terminal = values[-1]
    misses = terminal < target_wealth
    expected_goal_shortfall = (
        float(np.mean(target_wealth - terminal[misses])) if misses.any() else 0.0
    )
    periodic_target = np.power(1.0 + real_risk_free, 1.0 / 12.0) - 1.0
    finite_mask = np.isfinite(period_returns)
    finite_period_returns = period_returns[finite_mask]
    excess = finite_period_returns - periodic_target[finite_mask]
    omega_gains = float(np.maximum(excess, 0.0).sum())
    omega_losses = float(np.maximum(-excess, 0.0).sum())
    duration_metrics = _drawdown_duration_metrics(values, initial_value)
    rolling_metrics = _rolling_return_metrics(period_returns)
    drawdown_charts = _drawdown_chart_analytics(values, initial_value)
    rolling_horizons = _rolling_horizon_analytics(period_returns)
    goal_curve = _goal_probability_curve(terminal, initial_value, target_wealth)
    decision_metrics = {
        "target_wealth": target_wealth,
        "goal_success_probability": float((terminal >= target_wealth).mean()),
        "expected_goal_shortfall": expected_goal_shortfall,
        "risk_of_ruin": float(np.any(values <= 0.0, axis=0).mean()),
        "omega_ratio": min(omega_gains / omega_losses, 999.0) if omega_losses > 1e-12 else 999.0,
        **duration_metrics,
        **rolling_metrics,
    }

    invested = (
        initial_value
        + contribution * np.arange(1, periods + 1, dtype=float)[:, None]
        - np.cumsum(funded_values, axis=0)
    )
    success = {
        "periods": list(range(1, periods + 1)),
        "survival": np.mean(values > 0.0, axis=1).tolist(),
        "preservation": np.mean(values >= initial_value, axis=1).tolist(),
        "profit": np.mean(values >= np.maximum(invested, 0.0), axis=1).tolist(),
        "target": np.mean(values >= target_wealth, axis=1).tolist(),
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
        "decision_metrics": decision_metrics,
        **drawdown_charts,
        "rolling_horizons": rolling_horizons,
        "goal_curve": goal_curve,
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


def _persistence_response(model: Any, result: Any) -> dict[str, Any]:
    """Summarize calibrated duration hazards and simulated switching."""

    durations = model.metadata.get("sojourn_durations", {})
    expected = model.metadata.get("expected_duration_months", {})
    counts = _regime_counts(result)
    total = max(sum(counts.values()), 1)
    states: list[dict[str, Any]] = []
    expected_switch_rate = 0.0
    for state in model.states:
        observed = np.asarray(durations.get(state, []), dtype=float)
        expected_months = float(expected.get(state, np.nan))
        share = float(counts.get(state, 0)) / total
        if np.isfinite(expected_months) and expected_months > 0:
            expected_switch_rate += share / expected_months
        states.append(
            {
                "state": state,
                "label": _state_label(state),
                "expected_months": expected_months,
                "historical_mean_months": float(observed.mean()) if len(observed) else None,
                "historical_median_months": float(np.median(observed)) if len(observed) else None,
                "historical_episodes": int(len(observed)),
            }
        )
    regimes = np.asarray(result.regimes)
    simulated_switch_rate = (
        float(np.mean(regimes[1:] != regimes[:-1])) if regimes.shape[0] > 1 else 0.0
    )
    switches_per_decade = 120.0 * expected_switch_rate
    low_persistence = bool(
        switches_per_decade > 24.0
        or any(
            np.isfinite(item["expected_months"]) and item["expected_months"] < 5.0
            for item in states
        )
    )
    return {
        "expected_switches_per_decade": float(switches_per_decade),
        "simulated_switches_per_decade": float(simulated_switch_rate * 120.0),
        "expected_months_between_switches": (
            float(1.0 / expected_switch_rate) if expected_switch_rate > 0 else None
        ),
        "low_persistence_warning": low_persistence,
        "states": states,
    }


def _json_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        resolved = float(value)
        return resolved if np.isfinite(resolved) else None
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


def _retirement_response(
    wealth: pd.DataFrame,
    payload: Mapping[str, Any],
    *,
    tax_by_year: Mapping[str, Mapping[str, float]] | None = None,
) -> dict[str, Any]:
    """Build spending, exhaustion, guardrail, and paired-policy reporting."""

    periods, paths = wealth.shape
    plan = wealth.attrs.get("decumulation")
    if not isinstance(plan, Mapping):
        plan = normalize_decumulation(
            payload.get("decumulation"),
            periods=periods,
            legacy_withdrawal=float(payload.get("withdrawal", 0.0)),
            legacy_start_period=int(payload.get("withdrawal_start_period", 1)),
            annual_inflation_fallback=float(payload.get("annual_inflation", 0.0)) / 100.0,
        ).to_dict()
    requested = np.asarray(
        wealth.attrs.get("withdrawal_requested", np.zeros((periods, paths))),
        dtype=float,
    )
    funded = np.asarray(
        wealth.attrs.get("withdrawal_funded", np.zeros((periods, paths))),
        dtype=float,
    )
    events = np.asarray(
        wealth.attrs.get("guardrail_events", np.zeros((periods, paths))),
        dtype=np.int8,
    )
    cpi = np.asarray(
        wealth.attrs.get("withdrawal_cpi", np.ones((periods, paths))),
        dtype=float,
    )
    if requested.shape != (periods, paths):
        requested = np.zeros((periods, paths), dtype=float)
    if funded.shape != (periods, paths):
        funded = np.zeros((periods, paths), dtype=float)
    if events.shape != (periods, paths):
        events = np.zeros((periods, paths), dtype=np.int8)
    if cpi.shape != (periods, paths):
        cpi = np.ones((periods, paths), dtype=float)
    requested_real = requested / np.maximum(cpi, 1e-300)
    funded_real = funded / np.maximum(cpi, 1e-300)
    fully_funded_by_month = np.cumprod(funded + 1e-8 >= requested, axis=0).astype(bool)
    survival_curve = fully_funded_by_month.mean(axis=1)
    requested_total = requested_real.sum(axis=0)
    funded_total = funded_real.sum(axis=0)
    funded_ratio = np.divide(
        funded_total,
        requested_total,
        out=np.ones(paths, dtype=float),
        where=requested_total > 1e-12,
    )
    funded_fan = np.quantile(funded_real, [0.05, 0.50, 0.95], axis=1)
    cumulative_funded = np.cumsum(funded_real, axis=0)
    cumulative_fan = np.quantile(cumulative_funded, [0.05, 0.50, 0.95], axis=1)
    values = wealth.to_numpy(dtype=float)
    underfunded = funded + 1e-8 < requested
    first_shortfall = np.where(
        underfunded.any(axis=0),
        np.argmax(underfunded, axis=0) + 1,
        0,
    )
    cut_counts = (events < 0).sum(axis=0)
    increase_counts = (events > 0).sum(axis=0)

    def policy_summary(
        policy: str,
        policy_wealth: np.ndarray,
        policy_requested: np.ndarray,
        policy_funded: np.ndarray,
        policy_events: np.ndarray,
    ) -> dict[str, Any]:
        policy_cpi = cpi
        real_requested = policy_requested / np.maximum(policy_cpi, 1e-300)
        real_funded = policy_funded / np.maximum(policy_cpi, 1e-300)
        totals = real_requested.sum(axis=0)
        ratios = np.divide(
            real_funded.sum(axis=0),
            totals,
            out=np.ones(paths, dtype=float),
            where=totals > 1e-12,
        )
        success = np.all(policy_funded + 1e-8 >= policy_requested, axis=0)
        terminal = policy_wealth[-1]
        return {
            "policy": policy,
            "success_probability": float(success.mean()),
            "funded_spending_ratio": float(np.mean(ratios)),
            "median_real_spending": float(np.median(real_funded.sum(axis=0))),
            "terminal_p05": float(np.quantile(terminal, 0.05)),
            "terminal_median": float(np.quantile(terminal, 0.50)),
            "terminal_p95": float(np.quantile(terminal, 0.95)),
            "probability_of_cut": float(np.mean(np.any(policy_events < 0, axis=0))),
            "probability_of_increase": float(np.mean(np.any(policy_events > 0, axis=0))),
        }

    policy_values = plan.get("policy", {})
    current_policy = (
        str(policy_values.get("type", "fixed"))
        if isinstance(policy_values, Mapping)
        else str(policy_values)
    )
    comparison = [policy_summary(current_policy, values, requested, funded, events)]
    paired_policy = wealth.attrs.get("paired_policy")
    paired_wealth = wealth.attrs.get("paired_wealth")
    if paired_policy and paired_wealth is not None:
        comparison.append(
            policy_summary(
                str(paired_policy),
                np.asarray(paired_wealth, dtype=float),
                np.asarray(wealth.attrs.get("paired_withdrawal_requested"), dtype=float),
                np.asarray(wealth.attrs.get("paired_withdrawal_funded"), dtype=float),
                np.asarray(wealth.attrs.get("paired_guardrail_events"), dtype=np.int8),
            )
        )
    comparison.sort(key=lambda item: item["policy"])

    return {
        "enabled": bool(plan.get("enabled", False)),
        "mode": str(plan.get("mode", "manual")),
        "config": dict(plan),
        "periods": list(range(1, periods + 1)),
        "survival_probability": survival_curve.tolist(),
        "funded_spending": {
            "p05": funded_fan[0].tolist(),
            "median": funded_fan[1].tolist(),
            "p95": funded_fan[2].tolist(),
        },
        "cumulative_real_spending": {
            "p05": cumulative_fan[0].tolist(),
            "median": cumulative_fan[1].tolist(),
            "p95": cumulative_fan[2].tolist(),
        },
        "metrics": {
            "success_probability": float(survival_curve[-1]),
            "funded_spending_ratio": float(np.mean(funded_ratio)),
            "median_cumulative_real_spending": float(np.median(funded_total)),
            "probability_of_exhaustion": float(np.mean(underfunded.any(axis=0))),
            "median_first_shortfall_month": (
                float(np.median(first_shortfall[first_shortfall > 0]))
                if np.any(first_shortfall > 0)
                else None
            ),
            "probability_of_cut": float(np.mean(cut_counts > 0)),
            "probability_of_increase": float(np.mean(increase_counts > 0)),
            "mean_cuts": float(np.mean(cut_counts)),
            "mean_increases": float(np.mean(increase_counts)),
            "terminal_p05": float(np.quantile(values[-1], 0.05)),
            "terminal_median": float(np.quantile(values[-1], 0.50)),
            "terminal_p95": float(np.quantile(values[-1], 0.95)),
        },
        "paired_comparison": comparison,
        "tax_by_year": {
            str(year): {str(name): float(value) for name, value in metrics.items()}
            for year, metrics in (tax_by_year or {}).items()
        },
        "safe_rate": None,
    }


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
    gross_wealth = (
        scenario.gross_reporting_wealth
        if scenario.gross_reporting_wealth is not None
        else wealth
    )
    summary = scenario.summary
    growth_col = scenario.model.metadata.get("growth_col", "growth")
    inflation_col = scenario.model.metadata.get("inflation_col", "inflation")
    percentiles = _wealth_percentiles(wealth)
    gross_percentiles = (
        percentiles if gross_wealth is wealth else _wealth_percentiles(gross_wealth)
    )
    terminal_values = wealth.iloc[-1].to_numpy(dtype=float)
    drawdown_values = _max_drawdown_paths(
        wealth,
        initial_value=float(payload.get("initial_value", 100.0)),
    )
    reporting_indices = _reporting_indices(len(terminal_values))
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
    withdrawal_start_period = int(payload.get("withdrawal_start_period", 1))
    if contribution or withdrawal:
        withdrawal_periods = len(wealth) - withdrawal_start_period + 1
        summary_values["periodic_contribution"] = contribution
        summary_values["periodic_withdrawal"] = withdrawal
        summary_values["total_contributed"] = contribution * len(wealth)
        summary_values["total_withdrawn"] = withdrawal * withdrawal_periods
        summary_values["net_external_cash_flow"] = (
            contribution * len(wealth) - withdrawal * withdrawal_periods
        )
        if withdrawal:
            summary_values["withdrawal_start_period"] = withdrawal_start_period
            summary_values["withdrawal_periods"] = withdrawal_periods

    costs = {
        "weighted_expense_ratio": summary_values.get("weighted_expense_ratio", 0.0),
        "annual_fee_drag": summary_values.get("annual_fee_drag", 0.0),
        "annual_financing_cost": summary_values.get("annual_financing_cost", 0.0),
        "effective_financing_rate": summary_values.get("effective_financing_rate", 0.0),
        "leverage_multiple": summary_values.get("leverage_multiple", 1.0),
        "maintenance_margin": summary_values.get("maintenance_margin", 0.0),
        "margin_calls": summary_values.get("margin_calls", 0),
        "capital_gains_tax": summary_values.get("capital_gains_tax", 0.0),
        "wealth_tax": summary_values.get("wealth_tax", 0.0),
        "terminal_liquidation_tax": summary_values.get("terminal_liquidation_tax", 0.0),
        "taxes_paid": summary_values.get("taxes_paid", 0.0),
        "realized_gains": summary_values.get("realized_gains", 0.0),
        "realized_losses": summary_values.get("realized_losses", 0.0),
        "loss_carryforward": summary_values.get("loss_carryforward", 0.0),
        "investment_income_tax": summary_values.get("investment_income_tax", 0.0),
        "foreign_withholding_tax": summary_values.get("foreign_withholding_tax", 0.0),
        "financial_transaction_tax": summary_values.get("financial_transaction_tax", 0.0),
        "stamp_duty": summary_values.get("stamp_duty", 0.0),
        "ivafe": summary_values.get("ivafe", 0.0),
        "expired_losses": summary_values.get("expired_losses", 0.0),
        "annual_wealth_tax_rate": summary_values.get("annual_wealth_tax_rate", 0.0),
        "gross_terminal_wealth_median": summary_values.get(
            "gross_terminal_wealth_median", 0.0
        ),
        "after_tax_terminal_wealth_median": summary_values.get(
            "after_tax_terminal_wealth_median", 0.0
        ),
        "terminal_tax_drag_median": summary_values.get("terminal_tax_drag_median", 0.0),
        "terminal_tax_drag_percent": summary_values.get(
            "terminal_tax_drag_percent", 0.0
        ),
        "tax_drag_cagr": summary_values.get("tax_drag_cagr", 0.0),
        "effective_tax_rate": summary_values.get("effective_tax_rate", 0.0),
        "wrapper_terminal_p05": summary_values.get("wrapper_terminal_p05", 0.0),
        "wrapper_terminal_median": summary_values.get("wrapper_terminal_median", 0.0),
        "wrapper_terminal_p95": summary_values.get("wrapper_terminal_p95", 0.0),
        "wrapper_advantage_median": summary_values.get("wrapper_advantage_median", 0.0),
        "wrapper_advantage_percent": summary_values.get("wrapper_advantage_percent", 0.0),
        "wrapper_annual_drag_bps": summary_values.get("wrapper_annual_drag_bps", 0.0),
    }
    path_analytics = _path_analytics(
        wealth,
        result,
        payload,
        model=model,
        initial_value=float(payload.get("initial_value", 100.0)),
        drawdowns=drawdown_values,
    )
    summary_values.update(path_analytics["decision_metrics"])
    persistence = _persistence_response(model, result)
    warnings = list(scenario.diagnostics.warnings)
    if persistence["low_persistence_warning"]:
        warnings.append(
            "Regime persistence is unusually low for a macro-state model; review the "
            "smoothing, hysteresis, confirmation, and duration assumptions."
        )
    tax_country = str(model.metadata.get("tax_country", "none"))
    tax_regime = str(model.metadata.get("tax_regime", "none"))
    tax_selection = resolve_tax_selection(tax_country, tax_regime)
    if tax_selection.enabled:
        warnings.append(
            "Italian taxes are a planning approximation; verify instrument metadata, withholding, tax regime, "
            "and personal circumstances with a qualified tax adviser."
        )
        warnings.append(
            "Italian tax rules use the versioned IT-2026 planning snapshot; future simulation years "
            "assume those rules remain unchanged."
        )
    path_count = max(int(result.regimes.shape[1]), 1)
    tax_by_year = {
        str(year): {
            str(metric): float(value) / path_count
            for metric, value in values.items()
        }
        for year, values in model.metadata.get("tax_by_year", {}).items()
    }

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
        "warnings": warnings,
        "costs": costs,
        "taxes": {
            **(
                tax_selection.policy.metadata()
                if tax_selection.policy is not None
                else {
                    "country": "none",
                    "country_label": "No taxation",
                    "regime": "none",
                    "label": "No tax model",
                    "standard_rate": 0.0,
                    "government_bond_rate": 0.0,
                    "loss_carry_years": 0,
                }
            ),
            "enabled": tax_selection.enabled,
            "available_countries": available_tax_countries(),
            "asset_categories": model.metadata.get("asset_tax_categories", {}),
            "asset_metadata": model.metadata.get("asset_tax_metadata", {}),
            "annual_wealth_tax_rate": summary_values.get("annual_wealth_tax_rate", 0.0),
            "wealth_tax_mode": model.metadata.get("italy_wealth_tax_mode", "auto"),
            "tax_start_date": model.metadata.get("tax_start_date"),
            "by_year": tax_by_year,
            "terminal_liquidation": bool(model.metadata.get("tax_terminal_liquidation", False)),
            "impact": {
                "gross_terminal_median": summary_values.get(
                    "gross_terminal_wealth_median", 0.0
                ),
                "after_tax_terminal_median": summary_values.get(
                    "after_tax_terminal_wealth_median", 0.0
                ),
                "terminal_drag_median": summary_values.get(
                    "terminal_tax_drag_median", 0.0
                ),
                "terminal_drag_percent": summary_values.get(
                    "terminal_tax_drag_percent", 0.0
                ),
                "tax_drag_cagr": summary_values.get("tax_drag_cagr", 0.0),
                "effective_tax_rate": summary_values.get("effective_tax_rate", 0.0),
            },
            "wrapper": {
                "requested": bool(wealth.attrs.get("tax_wrapper_benchmark_requested", False)),
                "available": bool(wealth.attrs.get("tax_wrapper_benchmark_available", False)),
                "unavailable_reason": wealth.attrs.get("tax_wrapper_unavailable_reason"),
                "terminal_p05": summary_values.get("wrapper_terminal_p05", 0.0),
                "terminal_median": summary_values.get("wrapper_terminal_median", 0.0),
                "terminal_p95": summary_values.get("wrapper_terminal_p95", 0.0),
                "advantage_median": summary_values.get("wrapper_advantage_median", 0.0),
                "advantage_percent": summary_values.get("wrapper_advantage_percent", 0.0),
                "annual_drag_bps": summary_values.get("wrapper_annual_drag_bps", 0.0),
            },
        },
        "retirement": _retirement_response(
            wealth,
            payload,
            tax_by_year=tax_by_year if tax_selection.enabled else None,
        ),
        "wealth": {
            "periods": list(range(1, len(wealth) + 1)),
            "p05": percentiles[0.05].tolist(),
            "median": percentiles[0.50].tolist(),
            "p95": percentiles[0.95].tolist(),
        },
        "gross_wealth": (
            {
                "periods": list(range(1, len(gross_wealth) + 1)),
                "p05": gross_percentiles[0.05].tolist(),
                "median": gross_percentiles[0.50].tolist(),
                "p95": gross_percentiles[0.95].tolist(),
            }
            if tax_selection.enabled
            else None
        ),
        "monthly_returns": _median_period_returns(
            wealth,
            {**dict(payload), "annual_inflation": 0.0},
        ),
        "terminal": terminal_values[reporting_indices].tolist(),
        "drawdowns": drawdown_values[reporting_indices].tolist(),
        "reporting_sample": {
            "paths": int(len(reporting_indices)),
            "total_paths": int(len(terminal_values)),
            "sampled": bool(len(reporting_indices) < len(terminal_values)),
        },
        **path_analytics,
        "regime_timeline": regime_timelines["median"],
        "regime_timelines": regime_timelines,
        "regime_mix": [{"label": label, "share": float(share)} for label, share in regime_mix.items()],
        "persistence": persistence,
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
        "uncertainty_decomposition": model.metadata.get("uncertainty_decomposition", {}),
        "asset_profiles": model.metadata.get("asset_profiles", {}),
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
            "transition_estimator": model.metadata.get("transition_estimator", "hard_labels"),
            "regime_smoothing_window": int(model.metadata.get("regime_smoothing_window", 1)),
            "regime_hysteresis": float(model.metadata.get("regime_hysteresis", 0.0)),
            "regime_confirmation_periods": int(
                model.metadata.get("regime_confirmation_periods", 1)
            ),
            "duration_model_kind": model.metadata.get("duration_model_kind", "markov"),
            "min_regime_duration": int(payload.get("min_regime_duration", 5)),
            "hsmm_log_likelihood": model.metadata.get("hsmm_log_likelihood"),
            "hsmm_iterations": model.metadata.get("hsmm_iterations"),
            "hsmm_converged": model.metadata.get("hsmm_converged"),
            "hsmm_max_duration": model.metadata.get("hsmm_max_duration"),
            "mean_prior_strength": float(model.metadata.get("mean_prior_strength", 0.0)),
            "parameter_draws": int(payload.get("parameter_draws", 0)),
            "joint_macro": bool(payload.get("joint_macro", False)),
            "macro_model": model.metadata.get("macro_model", "ridge_var"),
            "macro_instability_score": float(
                model.metadata.get("macro_dynamics", {}).get("macro_instability_score", 0.0)
            ),
            "macro_parameter_uncertainty": bool(payload.get("macro_parameter_uncertainty", True)),
            "structural_returns": bool(model.metadata.get("structural_returns", False)),
            "state_dependent_liquidity": bool(model.metadata.get("state_dependent_liquidity", False)),
            "dynamic_correlation": bool(payload.get("dynamic_correlation", False)),
            "inflation_model": model.metadata.get("inflation_model", "deterministic"),
            "rate_model": model.metadata.get("rate_model", "deterministic"),
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


def build_safe_rate_response(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Solve fixed and guardrail safe rates on one shared set of market paths."""

    requested_paths = int(payload.get("paths", 3000))
    periods = int(payload.get("periods", 120))
    plan = normalize_decumulation(
        payload.get("decumulation"),
        periods=periods,
        legacy_withdrawal=float(payload.get("withdrawal", 0.0)),
        legacy_start_period=int(payload.get("withdrawal_start_period", 1)),
        annual_inflation_fallback=float(payload.get("annual_inflation", 0.0)) / 100.0,
    )
    if not plan.enabled or not plan.phases:
        raise ValueError("Safe-rate analysis requires decumulation with at least one recurring phase.")
    if plan.mode != "safe_rate":
        raise ValueError("Set decumulation.mode to safe_rate before calculating a safe rate.")

    # Force one in-memory result so every rate and policy sees the exact same
    # returns, regimes, macro paths, and parameter draw assignments.
    solver_payload = dict(payload)
    solver_payload["chunk_size"] = requested_paths
    solver_payload["workers"] = 1
    solver_payload["walk_forward"] = False
    scenario, selected_tickers, _ = run_scenario_payload(solver_payload)
    result = scenario.result
    if result.returns.shape[:2] != (periods, requested_paths):
        raise RuntimeError("Safe-rate solver requires retained market paths.")
    kwargs = scenario_kwargs(solver_payload)
    model = scenario.model

    def macro_rate(column: str | None, percent_key: str) -> np.ndarray | None:
        if result.macro_paths is None or not column or column not in result.macro_columns:
            return None
        values = result.macro_paths[:, :, result.macro_columns.index(column)].astype(float)
        dynamics = model.metadata.get("macro_dynamics", {})
        if bool(dynamics.get(percent_key, model.metadata.get(percent_key, False))):
            values /= 100.0
        return values

    inflation_column = model.metadata.get("inflation_col")
    inflation_paths = macro_rate(inflation_column, "inflation_is_percent")
    if inflation_paths is not None:
        inflation_paths = np.clip(inflation_paths, -0.10, 0.50)
    cpi = withdrawal_inflation_index(
        periods,
        requested_paths,
        annual_inflation=plan.annual_inflation_fallback,
        inflation_paths=inflation_paths,
    )
    rate_paths = macro_rate(model.metadata.get("rate_col"), "rate_is_percent")
    tax_selection = resolve_tax_selection(kwargs["tax_country"], kwargs["tax_regime"])
    portfolio_kwargs = {
        "weights": kwargs["weights"],
        "initial_value": kwargs["initial_value"],
        "return_kind": "log",
        "rebalance_frequency": kwargs["rebalance_frequency"],
        "transaction_cost_bps": kwargs["transaction_cost_bps"],
        "state_transaction_cost_multipliers": kwargs["state_transaction_cost_multipliers"],
        "asset_expense_ratios": kwargs["asset_expense_ratios"],
        "leverage_multiple": kwargs["leverage_multiple"],
        "financing_rate": kwargs["financing_rate"],
        "financing_inflation_sensitivity": kwargs["financing_inflation_sensitivity"],
        "state_inflation": model.metadata.get("state_inflation"),
        "financing_rate_paths": rate_paths,
        "financing_inflation_paths": inflation_paths,
        "maintenance_margin": kwargs["maintenance_margin"],
        "contribution": kwargs["contribution"],
        "contribution_allocation": kwargs["contribution_allocation"],
        "withdrawal": 0.0,
        "withdrawal_start_period": 1,
        "withdrawal_inflation_paths": inflation_paths,
        "annual_inflation": kwargs["annual_inflation"],
        "tax_country": tax_selection.country,
        "tax_regime": tax_selection.regime,
        "asset_tax_categories": kwargs["asset_tax_categories"],
        "asset_tax_metadata": kwargs["asset_tax_metadata"],
        "italy_annual_wealth_tax": kwargs["italy_annual_wealth_tax"],
        "italy_wealth_tax_mode": kwargs["italy_wealth_tax_mode"],
        "tax_terminal_liquidation": kwargs["tax_terminal_liquidation"],
        "tax_start_date": kwargs["tax_start_date"],
        "tax_wrapper_benchmark": False,
        "native_threads": max(1, int(kwargs["workers"])),
    }

    policy_results: dict[str, dict[str, Any]] = {}
    recommended_frames: dict[str, pd.DataFrame] = {}
    warnings: list[str] = []
    max_step = int(round(plan.safe_rate.maximum_rate / plan.safe_rate.precision))

    for policy in ("fixed", "guyton_klinger"):
        policy_plan = replace(
            plan,
            guardrails=replace(plan.guardrails, policy=policy),
        )
        evaluated: dict[int, dict[str, Any]] = {}

        def evaluate(step: int, *, keep_frame: bool = False) -> dict[str, Any]:
            step = max(0, min(max_step, int(step)))
            if step in evaluated and (not keep_frame or step in recommended_frames):
                return evaluated[step]
            rate = step * plan.safe_rate.precision
            frame = simulate_portfolio_paths(
                result,
                **portfolio_kwargs,
                decumulation=policy_plan,
                safe_withdrawal_rate=rate,
            )
            mask = success_mask(
                frame.to_numpy(dtype=float),
                np.asarray(frame.attrs["withdrawal_requested"], dtype=float),
                np.asarray(frame.attrs["withdrawal_funded"], dtype=float),
                objective=plan.safe_rate.objective,
                initial_value=kwargs["initial_value"],
                minimum_bequest=plan.safe_rate.minimum_bequest,
                terminal_cpi=cpi[-1],
            )
            successes = int(mask.sum())
            lower, upper = wilson_interval(successes, requested_paths)
            row = {
                "rate": rate,
                "probability": successes / requested_paths,
                "wilson_low": lower,
                "wilson_high": upper,
                "successes": successes,
            }
            evaluated[step] = row
            if keep_frame:
                recommended_frames[policy] = frame
            return row

        low = 0
        high = max_step
        low_result = evaluate(low)
        high_result = evaluate(high)
        capped = high_result["probability"] >= plan.safe_rate.target_probability
        baseline_failure = low_result["probability"] < plan.safe_rate.target_probability
        if capped:
            recommended_step = high
            warnings.append(
                f"{policy}: the 25% search ceiling still satisfies the target; the safe rate is at least 25%."
            )
        elif baseline_failure:
            recommended_step = 0
            warnings.append(
                f"{policy}: one-time expenses or the terminal objective miss the target even at a 0% recurring rate."
            )
        else:
            while high - low > 1:
                middle = (low + high) // 2
                result_at_middle = evaluate(middle)
                if result_at_middle["probability"] >= plan.safe_rate.target_probability:
                    low = middle
                else:
                    high = middle
            recommended_step = low
        recommended = evaluate(recommended_step, keep_frame=True)
        policy_results[policy] = {
            "recommended_rate": recommended["rate"],
            "display_rate": (
                f"≥{plan.safe_rate.maximum_rate:.0%}" if capped else f"{recommended['rate']:.2%}"
            ),
            "at_search_ceiling": capped,
            "baseline_below_target": baseline_failure,
            "probability": recommended["probability"],
            "wilson_95": [recommended["wilson_low"], recommended["wilson_high"]],
            "curve": [evaluated[key] for key in sorted(evaluated)],
        }

    selected_policy = plan.policy
    selected_frame = recommended_frames[selected_policy]
    reporting_frame = pd.DataFrame(
        selected_frame.to_numpy(dtype=float) / np.maximum(cpi, 1e-300),
        columns=selected_frame.columns,
    )
    reporting_frame.attrs.update(selected_frame.attrs)
    alternate_policy = "guyton_klinger" if selected_policy == "fixed" else "fixed"
    alternate_frame = recommended_frames[alternate_policy]
    reporting_frame.attrs.update(
        {
            "paired_policy": alternate_policy,
            "paired_wealth": alternate_frame.to_numpy(dtype=float)
            / np.maximum(cpi, 1e-300),
            "paired_withdrawal_requested": np.asarray(
                alternate_frame.attrs["withdrawal_requested"], dtype=float
            ),
            "paired_withdrawal_funded": np.asarray(
                alternate_frame.attrs["withdrawal_funded"], dtype=float
            ),
            "paired_guardrail_events": np.asarray(
                alternate_frame.attrs["guardrail_events"], dtype=np.int8
            ),
        }
    )
    average_tax_by_year = {
        str(year): {
            str(name): float(value) / requested_paths for name, value in metrics.items()
        }
        for year, metrics in selected_frame.attrs.get("tax_by_year", {}).items()
    }
    retirement = _retirement_response(
        reporting_frame,
        solver_payload,
        tax_by_year=average_tax_by_year,
    )
    retirement["safe_rate"] = {
        "selected_policy": selected_policy,
        "objective": plan.safe_rate.objective,
        "target_probability": plan.safe_rate.target_probability,
        "minimum_bequest": plan.safe_rate.minimum_bequest,
        "precision": plan.safe_rate.precision,
        "policies": policy_results,
    }
    return {
        "ok": True,
        "retirement": retirement,
        "warnings": warnings,
        "paths": requested_paths,
        "periods": periods,
        "selected_tickers": selected_tickers,
        "same_market_paths": True,
        "message": "Safe-rate analysis complete on paired fixed and guardrail policies.",
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
