from __future__ import annotations

import io
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from datetime import date

import gradio as gr
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from mc_quadrants.data import (
    fetch_yahoo_fx_rates,
    load_market_data,
    prices_to_returns,
)
from mc_quadrants.demo import _demo_history
from mc_quadrants.diagnostics import simulation_regime_summary
from mc_quadrants.pipeline import compare_distributions, run_scenario
from mc_quadrants.regimes import REGIME_ORDER

REGIME_NAMES = {
    "high_growth_low_inflation": "High growth / low inflation",
    "high_growth_high_inflation": "High growth / high inflation",
    "low_growth_high_inflation": "Low growth / high inflation",
    "low_growth_low_inflation": "Low growth / low inflation",
}

REGIME_COLORS = {
    "high_growth_low_inflation": "#2f855a",
    "high_growth_high_inflation": "#d97706",
    "low_growth_high_inflation": "#c2410c",
    "low_growth_low_inflation": "#2563eb",
}

DEFAULT_CORRELATIONS = {
    "high_growth_low_inflation": -0.10,
    "high_growth_high_inflation": 0.35,
    "low_growth_high_inflation": 0.25,
    "low_growth_low_inflation": -0.40,
}

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

REGIME_LABELS = [REGIME_NAMES[state] for state in REGIME_ORDER]
REGIME_LOOKUP = {REGIME_NAMES[state]: state for state in REGIME_ORDER}
RETURN_DISTRIBUTIONS = {
    "Normal": "normal",
    "Student-t": "student_t",
    "Historical bootstrap": "bootstrap",
    "Block bootstrap": "block_bootstrap",
}
REBALANCE_FREQUENCIES = {
    "Weighted log (legacy)": None,
    "Monthly": 1,
    "Quarterly": 3,
    "Annual": 12,
}


def demo_history(seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    macro, returns = _demo_history(seed)
    return macro, returns.rename(columns=DEMO_TICKERS)


def read_uploaded_csv(uploaded_file) -> pd.DataFrame | None:
    if uploaded_file is None:
        return None

    # Gradio 4/5 may pass a filepath, FileData object, or a mapping containing
    # the uploaded path depending on how the event is invoked.
    source = uploaded_file
    if isinstance(source, Mapping):
        source = source.get("path") or source.get("name") or source.get("data")
    else:
        source = getattr(source, "path", None) or getattr(source, "name", source)
    if source is None:
        raise ValueError("The uploaded CSV did not include readable file data.")
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)

    data = pd.read_csv(source)
    if "Date" not in data.columns:
        raise ValueError("CSV files need a Date column.")
    try:
        data["Date"] = pd.to_datetime(data["Date"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("The CSV Date column contains invalid dates.") from exc
    if data["Date"].isna().any():
        raise ValueError("The CSV Date column contains missing dates.")
    return data.set_index("Date").sort_index()


def normalize_ticker_columns(data: pd.DataFrame) -> pd.DataFrame:
    normalized = data.copy()
    normalized.columns = [str(column).strip().upper() for column in normalized.columns]
    return normalized


def parse_tickers(raw_tickers: str | Sequence[str]) -> list[str]:
    parsed: list[str] = []
    raw_values = (
        re.split(r"[,;\s]+", raw_tickers.strip().upper())
        if isinstance(raw_tickers, str)
        else [str(ticker).strip().upper() for ticker in raw_tickers]
    )
    for ticker in raw_values:
        if ticker and ticker not in parsed:
            parsed.append(ticker)
    return parsed


def parse_proxy_map(raw_proxies: str) -> dict[str, str]:
    """Parse ``ASSET:PROXY`` pairs used to extend pre-inception history."""

    parsed: dict[str, str] = {}
    for pair in re.split(r"[,;\s]+", raw_proxies.strip().upper()):
        if not pair:
            continue
        if ":" not in pair:
            raise ValueError(f"Invalid proxy '{pair}'. Use ASSET:PROXY.")
        asset, proxy = pair.split(":", 1)
        if not asset or not proxy:
            raise ValueError(f"Invalid proxy '{pair}'. Use ASSET:PROXY.")
        parsed[asset] = proxy
    return parsed


def parse_currency_map(raw_currencies: str) -> dict[str, str]:
    """Parse ``ASSET:CURRENCY`` pairs, defaulting unspecified assets to USD."""

    parsed: dict[str, str] = {}
    for pair in re.split(r"[,;\s]+", raw_currencies.strip().upper()):
        if not pair:
            continue
        if ":" not in pair:
            raise ValueError(f"Invalid currency '{pair}'. Use ASSET:CURRENCY.")
        asset, currency = pair.split(":", 1)
        if not asset or len(currency) != 3:
            raise ValueError(f"Invalid currency '{pair}'. Use ASSET:CURRENCY.")
        parsed[asset] = currency
    return parsed


def _currency_for_asset(asset: str, asset_currencies: Mapping[str, str]) -> str:
    normalized = str(asset).strip().upper()
    base_asset = normalized.removesuffix("_SIM").removesuffix("SIM")
    return asset_currencies.get(normalized, asset_currencies.get(base_asset, "USD"))


def prepare_fx_rates(
    returns: pd.DataFrame,
    selected_tickers: list[str],
    base_currency: str,
    currency_text: str,
) -> tuple[dict[str, str], pd.DataFrame | None]:
    """Load historical FX levels needed to express selected returns in base currency."""

    base_currency = str(base_currency).strip().upper()
    if len(base_currency) != 3:
        raise ValueError("Portfolio currency must be a three-letter ISO code.")
    asset_currencies = parse_currency_map(currency_text)
    source_currencies = {
        _currency_for_asset(ticker, asset_currencies)
        for ticker in selected_tickers
    }
    foreign_currencies = tuple(sorted(currency for currency in source_currencies if currency != base_currency))
    if not foreign_currencies:
        return asset_currencies, None
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise ValueError("Historical FX conversion requires datetime-indexed returns.")
    fx_start = pd.Timestamp(returns.index.min()) - pd.DateOffset(months=1)
    fx_rates = fetch_yahoo_fx_rates(
        foreign_currencies,
        base_currency,
        start=fx_start.strftime("%Y-%m-%d"),
        end=pd.Timestamp(returns.index.max()).strftime("%Y-%m-%d"),
    )
    return asset_currencies, fx_rates


def default_weight(asset: str, assets: list[str]) -> float:
    base_asset = asset.removesuffix("_EXTENDED").removesuffix("_SIM").removesuffix("SIM")
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
    return defaults.get(base_asset, round(100.0 / max(len(assets), 1), 2))


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


def default_correlation_pair(tickers: list[str]) -> tuple[str | None, str | None]:
    if len(tickers) < 2:
        return (tickers[0], None) if tickers else (None, None)

    asset_a = "SPYSIM" if "SPYSIM" in tickers else "SPY" if "SPY" in tickers else tickers[0]
    remaining = [ticker for ticker in tickers if ticker != asset_a]
    asset_b = "IEFSIM" if "IEFSIM" in remaining else "IEF" if "IEF" in remaining else remaining[0]
    return asset_a, asset_b


def macro_column_updates(macro_file) -> tuple[dict, dict]:
    """Return dropdown updates for the columns found in an uploaded macro CSV."""

    try:
        macro = read_uploaded_csv(macro_file)
        if macro is None:
            return gr.update(choices=[], value=None), gr.update(choices=[], value=None)

        columns = [str(column) for column in macro.columns]
        if len(columns) < 2:
            raise ValueError("The macro CSV needs at least two indicator columns.")

        growth = "growth" if "growth" in columns else columns[0]
        inflation = "inflation" if "inflation" in columns and "inflation" != growth else None
        if inflation is None:
            inflation = next(column for column in columns if column != growth)
        return (
            gr.update(choices=columns, value=growth),
            gr.update(choices=columns, value=inflation),
        )
    except Exception as exc:
        raise gr.Error(f"Could not inspect macro CSV: {exc}")


# ---------- Chart builders ----------

def wealth_percentile_chart(wealth: pd.DataFrame) -> go.Figure:
    percentiles = wealth.quantile([0.05, 0.50, 0.95], axis=1).T
    percentiles.columns = ["P05", "Median", "P95"]
    fig = go.Figure()
    for col, color in [("P05", "#c2410c"), ("Median", "#2563eb"), ("P95", "#2f855a")]:
        fig.add_trace(
            go.Scatter(
                x=percentiles.index,
                y=percentiles[col],
                mode="lines",
                name=col,
                line=dict(color=color, width=2),
            )
        )
    fig.update_layout(
        title="Wealth Percentiles",
        xaxis_title="Period",
        yaxis_title="Wealth",
        height=400,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def terminal_distribution_chart(terminal: pd.Series) -> go.Figure:
    fig = px.histogram(
        terminal,
        nbins=45,
        title="Terminal Wealth Distribution",
        labels={"value": "Terminal wealth", "count": "Paths"},
        color_discrete_sequence=["#2563eb"],
        opacity=0.82,
    )
    fig.update_layout(height=400, template="plotly_white", showlegend=False)
    return fig


def regime_mix_chart(regimes: np.ndarray) -> go.Figure:
    mix = (
        pd.Series(regimes.ravel())
        .value_counts(normalize=True)
        .reindex(REGIME_ORDER)
        .fillna(0.0)
        .rename(index=REGIME_NAMES)
    )
    fig = px.bar(
        x=mix.index,
        y=mix.values,
        title="Simulated Regime Mix",
        labels={"x": "Regime", "y": "Proportion"},
        color=mix.index,
        color_discrete_map=REGIME_COLORS,
    )
    fig.update_layout(
        height=400,
        template="plotly_white",
        showlegend=False,
        xaxis_tickangle=-20,
    )
    return fig


def transition_matrix_chart(matrix: pd.DataFrame) -> go.Figure:
    renamed = matrix.rename(index=REGIME_NAMES, columns=REGIME_NAMES)
    fig = px.imshow(
        renamed,
        text_auto=".2f",
        title="Transition Matrix",
        labels=dict(x="To", y="From", color="Probability"),
        color_continuous_scale="Blues",
        zmin=0,
        zmax=1,
        aspect="auto",
    )
    fig.update_layout(height=400, template="plotly_white")
    return fig


def macro_scatter_chart(
    macro: pd.DataFrame,
    regimes: pd.Series,
    growth_col: str,
    inflation_col: str,
) -> go.Figure:
    data = macro[[growth_col, inflation_col]].copy()
    data["regime"] = regimes.map(REGIME_NAMES)
    data["date"] = data.index.astype(str)

    fig = px.scatter(
        data.dropna(),
        x=growth_col,
        y=inflation_col,
        color="regime",
        hover_data=["date"],
        title="Macro Quadrants",
        labels={growth_col: "Growth", inflation_col: "Inflation", "regime": "Regime"},
        color_discrete_map=REGIME_COLORS,
        opacity=0.72,
    )
    fig.update_traces(marker=dict(size=8))
    fig.update_layout(height=400, template="plotly_white")
    return fig


def observations_chart(model) -> go.Figure:
    observations = pd.Series(
        {REGIME_NAMES[state]: moments.observations for state, moments in model.moments.items()},
        name="Observations",
    )
    fig = px.bar(
        x=observations.index,
        y=observations.values,
        title="Historical Observations by Regime",
        labels={"x": "Regime", "y": "Observations"},
        color=observations.index,
        color_discrete_map=REGIME_COLORS,
    )
    fig.update_layout(height=400, template="plotly_white", showlegend=False, xaxis_tickangle=-20)
    return fig


def correlation_matrix_chart(
    correlation: pd.DataFrame,
    title: str = "Correlation Matrix",
) -> go.Figure:
    fig = px.imshow(
        correlation,
        text_auto=".2f",
        title=title,
        labels=dict(color="Correlation"),
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto",
    )
    fig.update_layout(height=400, template="plotly_white")
    return fig


def _write_csv_download(data: pd.DataFrame, prefix: str, index: bool = False) -> str:
    handle = tempfile.NamedTemporaryFile(prefix=prefix, suffix=".csv", delete=False)
    handle.close()
    data.to_csv(handle.name, index=index)
    return handle.name


# ---------- Core logic ----------


def _weights_from_dataframe(
    weights_df: pd.DataFrame | Mapping | list | None,
    selected_tickers: list[str],
) -> dict[str, float]:
    if weights_df is None:
        raise ValueError("Set at least one ticker weight above zero.")

    if isinstance(weights_df, Mapping):
        rows = weights_df.get("data", [])
        headers = weights_df.get("headers", [])
        frame = pd.DataFrame(rows, columns=headers or None)
    elif isinstance(weights_df, pd.DataFrame):
        frame = weights_df
    else:
        frame = pd.DataFrame(weights_df)

    if frame.empty or frame.shape[1] < 2:
        raise ValueError("Set at least one ticker weight above zero.")

    weights: dict[str, float] = {}
    for ticker_value, weight_value in frame.iloc[:, :2].itertuples(index=False, name=None):
        if pd.isna(ticker_value) or str(ticker_value).strip() == "":
            continue
        ticker = str(ticker_value).strip().upper()
        if ticker not in selected_tickers:
            raise ValueError(f"Weight table contains ticker not selected: {ticker}.")
        try:
            weight = float(weight_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Weight for {ticker} must be numeric.") from exc
        if not np.isfinite(weight) or weight < 0:
            raise ValueError(f"Weight for {ticker} must be a non-negative number.")
        weights[ticker] = weight

    if np.isclose(sum(weights.values()), 0.0):
        raise ValueError("Set at least one ticker weight above zero.")
    return weights

def load_data(
    source: str,
    demo_seed: int,
    ticker_text: str,
    proxy_text: str,
    synthetic_text: str | Sequence[str],
    synthetic_seed: int,
    start_date: str,
    end_date: str,
    prices_file,
    macro_file,
    asset_input: str,
    monthly_returns: bool,
    growth_col: str,
    inflation_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], str, str, str]:
    """Load data based on the selected source. Returns (macro, returns, tickers, growth_col, inflation_col, message)."""
    try:
        if source == "Demo":
            macro, returns = demo_history(int(demo_seed))
            returns = normalize_ticker_columns(returns)
            tickers = list(returns.columns)
            return macro, returns, tickers, "growth", "inflation", f"Loaded demo data with seed {demo_seed}."

        if source == "Yahoo Finance":
            tickers = parse_tickers(ticker_text)
            historical_proxies = parse_proxy_map(proxy_text)
            synthetic_assets = parse_tickers(synthetic_text)
            tickers.extend(asset for asset in synthetic_assets if asset not in tickers)
            if not tickers:
                raise ValueError("Enter at least one Yahoo Finance ticker.")
            macro, returns, available = load_market_data(
                tickers,
                start_date,
                end_date,
                historical_proxies=historical_proxies,
                synthetic_assets=synthetic_assets,
                synthetic_seed=int(synthetic_seed),
            )
            unavailable = [t for t in tickers if t not in available]
            base_available = [
                ticker
                for ticker in available
                if not ticker.endswith("_SIM") and not ticker.endswith("SIM")
            ]
            msg = f"Loaded {len(base_available)} tickers from Yahoo Finance."
            if unavailable:
                msg += f" No usable history for: {', '.join(unavailable)}"
            if historical_proxies:
                msg += f" Backfilled proxies: {', '.join(historical_proxies.values())}."
            if synthetic_assets:
                msg += f" Simulated sources: {', '.join(f'{asset}SIM' for asset in synthetic_assets)}."
            return macro, returns, available, "growth", "inflation", msg

        # CSV upload
        asset_data = read_uploaded_csv(prices_file)
        macro_data = read_uploaded_csv(macro_file)
        if asset_data is None or macro_data is None:
            raise ValueError("Upload both an asset CSV and a macro CSV.")

        if growth_col not in macro_data.columns:
            raise ValueError(f"Growth column not found in macro CSV: {growth_col}")
        if inflation_col not in macro_data.columns:
            raise ValueError(f"Inflation column not found in macro CSV: {inflation_col}")
        if growth_col == inflation_col:
            raise ValueError("Growth and inflation must use different macro columns.")

        if asset_input == "Price levels":
            returns = prices_to_returns(asset_data, method="log")
        else:
            returns = asset_data.apply(pd.to_numeric, errors="coerce")

        if monthly_returns:
            returns = returns.resample("ME").sum(min_count=1)
        macro_data = macro_data.apply(pd.to_numeric, errors="coerce").resample("ME").last()
        returns = normalize_ticker_columns(returns)
        returns = returns.dropna(how="all")
        if returns.empty or not any(returns[column].notna().any() for column in returns.columns):
            raise ValueError("The asset CSV has no usable numeric data.")

        tickers = list(returns.columns)
        return macro_data, returns, tickers, growth_col, inflation_col, "Loaded data from CSV uploads."

    except Exception as exc:
        raise gr.Error(f"Could not load data: {exc}")


def run_simulation(
    macro: pd.DataFrame,
    returns: pd.DataFrame,
    selected_tickers: list[str],
    growth_col: str,
    inflation_col: str,
    growth_threshold: str | float,
    inflation_threshold: str | float,
    periods: int,
    paths: int,
    seed: int,
    start_state_label: str,
    weights_df: pd.DataFrame,
    correlation_regime_label: str = REGIME_LABELS[0],
    use_correlation_override: bool = False,
    correlation_asset_a: str | None = None,
    correlation_asset_b: str | None = None,
    correlation_blend: float = 0.40,
    corr_high_growth_low_inflation: float = DEFAULT_CORRELATIONS["high_growth_low_inflation"],
    corr_high_growth_high_inflation: float = DEFAULT_CORRELATIONS["high_growth_high_inflation"],
    corr_low_growth_high_inflation: float = DEFAULT_CORRELATIONS["low_growth_high_inflation"],
    corr_low_growth_low_inflation: float = DEFAULT_CORRELATIONS["low_growth_low_inflation"],
    distribution_label: str = "Normal",
    degrees_of_freedom: float = 5.0,
    block_size: int = 3,
    rebalance_frequency_label: str = "Weighted log (legacy)",
    transaction_cost_bps: float = 0.0,
    macro_lag_periods: int = 0,
    transition_uncertainty: float = 0.0,
    base_currency: str = "USD",
    currency_text: str = "",
) -> tuple[go.Figure, go.Figure, go.Figure, go.Figure, go.Figure, go.Figure, go.Figure, str, str, str, str, str]:
    """Run the full calibration + simulation pipeline and return all outputs."""
    try:
        if macro is None or returns is None:
            raise ValueError("Load Data before running the simulation.")
        selected_tickers = [str(ticker).strip().upper() for ticker in (selected_tickers or [])]
        if not selected_tickers:
            raise ValueError("Select at least one ticker.")
        missing_tickers = [ticker for ticker in selected_tickers if ticker not in returns.columns]
        if missing_tickers:
            raise ValueError(f"Selected tickers are missing from the loaded returns: {', '.join(missing_tickers)}")

        returns = returns.loc[:, selected_tickers]
        base_currency = str(base_currency).strip().upper()
        asset_currencies, fx_rates = prepare_fx_rates(
            returns,
            selected_tickers,
            base_currency,
            currency_text,
        )

        periods = int(periods)
        paths = int(paths)
        if periods <= 0 or paths <= 0:
            raise ValueError("Periods and paths must be positive.")

        distribution_key = str(distribution_label).strip()
        distribution = RETURN_DISTRIBUTIONS.get(
            distribution_key,
            distribution_key.lower().replace("-", "_"),
        )
        if distribution not in {"normal", "student_t", "bootstrap", "block_bootstrap"}:
            raise ValueError("Unknown return distribution.")
        if rebalance_frequency_label not in REBALANCE_FREQUENCIES:
            raise ValueError(f"Unknown rebalancing frequency: {rebalance_frequency_label}")
        rebalance_frequency = REBALANCE_FREQUENCIES[rebalance_frequency_label]

        # Parse thresholds
        growth_thr: str | float = growth_threshold
        inflation_thr: str | float = inflation_threshold
        if isinstance(growth_threshold, str) and growth_threshold.startswith("fixed:"):
            growth_thr = float(growth_threshold.split(":", 1)[1])
        if isinstance(inflation_threshold, str) and inflation_threshold.startswith("fixed:"):
            inflation_thr = float(inflation_threshold.split(":", 1)[1])

        weights = _weights_from_dataframe(weights_df, selected_tickers)

        # Start state
        start_state = None
        if start_state_label != "Stationary":
            start_state = REGIME_LOOKUP[start_state_label]

        correlation_overrides = None
        if use_correlation_override:
            asset_a = str(correlation_asset_a or "").strip().upper()
            asset_b = str(correlation_asset_b or "").strip().upper()
            if asset_a == asset_b:
                raise ValueError("Correlation override tickers must be different.")
            if asset_a not in selected_tickers or asset_b not in selected_tickers:
                raise ValueError("Correlation override tickers must be selected in the portfolio.")
            if not 0 <= float(correlation_blend) <= 1:
                raise ValueError("Correlation blend must be between 0 and 1.")
            correlation_values = {
                REGIME_ORDER[0]: corr_high_growth_low_inflation,
                REGIME_ORDER[1]: corr_high_growth_high_inflation,
                REGIME_ORDER[2]: corr_low_growth_high_inflation,
                REGIME_ORDER[3]: corr_low_growth_low_inflation,
            }
            if any(not -1 <= float(value) <= 1 for value in correlation_values.values()):
                raise ValueError("Correlation overrides must be between -1 and 1.")
            correlation_overrides = {
                state: {(asset_a, asset_b): float(value)}
                for state, value in correlation_values.items()
            }

        scenario = run_scenario(
            returns=returns,
            macro=macro,
            selected_tickers=selected_tickers,
            growth_col=growth_col,
            inflation_col=inflation_col,
            growth_threshold=growth_thr,
            inflation_threshold=inflation_thr,
            periods=periods,
            paths=paths,
            random_seed=int(seed),
            start_state=start_state,
            weights=weights,
            correlation_overrides=correlation_overrides,
            override_weight=float(correlation_blend),
            macro_lag_periods=int(macro_lag_periods),
            distribution=distribution,
            degrees_of_freedom=float(degrees_of_freedom),
            block_size=int(block_size),
            transition_uncertainty=float(transition_uncertainty),
            rebalance_frequency=rebalance_frequency,
            transaction_cost_bps=float(transaction_cost_bps),
            base_currency=base_currency,
            asset_currencies=asset_currencies,
            fx_rates=fx_rates,
        )
        model = scenario.model
        regimes = scenario.regimes
        result = scenario.result
        wealth = scenario.wealth
        summary = scenario.summary

        # Build charts
        wealth_fig = wealth_percentile_chart(wealth)
        terminal_fig = terminal_distribution_chart(wealth.iloc[-1])
        mix_fig = regime_mix_chart(result.regimes)
        transition_fig = transition_matrix_chart(model.transition_matrix)
        scatter_fig = macro_scatter_chart(macro, regimes, growth_col, inflation_col)
        obs_fig = observations_chart(model)

        selected_correlation_state = REGIME_LOOKUP.get(correlation_regime_label)
        if selected_correlation_state is None:
            raise ValueError(f"Unknown correlation regime: {correlation_regime_label}")
        selected_correlation = model.moments[selected_correlation_state].correlation
        corr_fig = correlation_matrix_chart(
            selected_correlation,
            title=f"Correlation Matrix: {correlation_regime_label}",
        )

        # Transition matrix as text
        transition_text = model.transition_matrix.rename(index=REGIME_NAMES, columns=REGIME_NAMES).round(3).to_string()

        # Observations as text
        obs_text = pd.Series(
            {REGIME_NAMES[state]: moments.observations for state, moments in model.moments.items()},
            name="Observations",
        ).to_string()

        # Correlation as text
        corr_text = selected_correlation.round(3).to_string()

        summary_text = (
            f"**Currency:** {base_currency} | "
            f"**Mean:** {summary['mean']:.2f} | "
            f"**P05:** {summary['p05']:.2f} | "
            f"**Median:** {summary['p50']:.2f} | "
            f"**P95:** {summary['p95']:.2f} | "
            f"**Volatility:** {summary['std']:.2f}"
        )
        distribution_text = next(
            label for label, value in RETURN_DISTRIBUTIONS.items() if value == distribution
        )
        friction_text = (
            f"{rebalance_frequency_label}, {float(transaction_cost_bps):.1f} bps"
            if rebalance_frequency is not None
            else "Weighted log (no rebalancing)"
        )
        risk_text = (
            f"**Probability of loss:** {summary['probability_of_loss']:.1%} | "
            f"**VaR (95%):** {summary['var_95']:.2f} | "
            f"**Expected shortfall (95%):** {summary['expected_shortfall_95']:.2f} | "
            f"**Mean max drawdown:** {summary['max_drawdown_mean']:.1%} | "
            f"**Worst max drawdown:** {summary['max_drawdown_worst']:.1%}"
        )
        diagnostics_table = scenario.diagnostics.regime_summary.copy()
        simulated_diagnostics = simulation_regime_summary(result).rename(
            columns={
                "observations": "simulated_observations",
                "share": "simulated_share",
            }
        )
        diagnostics_table = diagnostics_table.merge(simulated_diagnostics, on="regime", how="left")
        diagnostics_table["regime"] = diagnostics_table["regime"].map(REGIME_NAMES)
        warnings_text = ""
        if scenario.diagnostics.warnings:
            warnings_text = "**Calibration warnings**\n\n" + "\n".join(
                f"- {warning}" for warning in scenario.diagnostics.warnings
            )
        wealth_export = wealth.copy()
        wealth_export.insert(0, "period", range(1, len(wealth_export) + 1))
        summary_export = summary.rename("value").rename_axis("metric").reset_index()
        wealth_download = _write_csv_download(wealth_export, "mc-wealth-")
        summary_download = _write_csv_download(summary_export, "mc-summary-")
        diagnostics_download = _write_csv_download(diagnostics_table, "mc-diagnostics-")

        return (
            wealth_fig, terminal_fig, mix_fig,
            transition_fig, scatter_fig, obs_fig, corr_fig,
            transition_text, obs_text, corr_text,
            summary_text,
            f"Simulation complete: {paths} paths x {periods} periods. "
            f"Distribution: {distribution_text}. Portfolio: {friction_text}.",
            f"Selected tickers: {', '.join(selected_tickers)}",
            risk_text,
            diagnostics_table,
            warnings_text,
            wealth_download,
            summary_download,
            diagnostics_download,
        )

    except Exception as exc:
        raise gr.Error(f"Simulation failed: {exc}")


def compare_scenarios(
    macro: pd.DataFrame,
    returns: pd.DataFrame,
    selected_tickers: list[str],
    growth_col: str,
    inflation_col: str,
    growth_threshold: str | float,
    inflation_threshold: str | float,
    periods: int,
    paths: int,
    seed: int,
    start_state_label: str,
    weights_df: pd.DataFrame,
    use_correlation_override: bool,
    correlation_asset_a: str | None,
    correlation_asset_b: str | None,
    correlation_blend: float,
    corr_high_growth_low_inflation: float,
    corr_high_growth_high_inflation: float,
    corr_low_growth_high_inflation: float,
    corr_low_growth_low_inflation: float,
    rebalance_frequency_label: str,
    transaction_cost_bps: float,
    macro_lag_periods: int,
    transition_uncertainty: float,
    base_currency: str,
    currency_text: str,
) -> pd.DataFrame:
    """Compare Gaussian and Student-t outcomes using identical inputs."""

    try:
        if macro is None or returns is None:
            raise ValueError("Load Data before comparing scenarios.")
        selected_tickers = [str(ticker).strip().upper() for ticker in (selected_tickers or [])]
        weights = _weights_from_dataframe(weights_df, selected_tickers)
        base_currency = str(base_currency).strip().upper()
        asset_currencies, fx_rates = prepare_fx_rates(
            returns.loc[:, selected_tickers],
            selected_tickers,
            base_currency,
            currency_text,
        )
        growth_thr: str | float = growth_threshold
        inflation_thr: str | float = inflation_threshold
        if isinstance(growth_threshold, str) and growth_threshold.startswith("fixed:"):
            growth_thr = float(growth_threshold.split(":", 1)[1])
        if isinstance(inflation_threshold, str) and inflation_threshold.startswith("fixed:"):
            inflation_thr = float(inflation_threshold.split(":", 1)[1])

        start_state = None if start_state_label == "Stationary" else REGIME_LOOKUP[start_state_label]
        rebalance_frequency = REBALANCE_FREQUENCIES[rebalance_frequency_label]
        correlation_overrides = None
        if use_correlation_override:
            asset_a = str(correlation_asset_a or "").strip().upper()
            asset_b = str(correlation_asset_b or "").strip().upper()
            if asset_a == asset_b:
                raise ValueError("Correlation override tickers must be different.")
            correlation_values = {
                REGIME_ORDER[0]: corr_high_growth_low_inflation,
                REGIME_ORDER[1]: corr_high_growth_high_inflation,
                REGIME_ORDER[2]: corr_low_growth_high_inflation,
                REGIME_ORDER[3]: corr_low_growth_low_inflation,
            }
            correlation_overrides = {
                state: {(asset_a, asset_b): float(value)}
                for state, value in correlation_values.items()
            }

        return compare_distributions(
            {"Normal": "normal", "Student-t": "student_t"},
            returns=returns,
            macro=macro,
            selected_tickers=selected_tickers,
            growth_col=growth_col,
            inflation_col=inflation_col,
            growth_threshold=growth_thr,
            inflation_threshold=inflation_thr,
            periods=int(periods),
            paths=int(paths),
            random_seed=int(seed),
            start_state=start_state,
            weights=weights,
            correlation_overrides=correlation_overrides,
            override_weight=float(correlation_blend),
            macro_lag_periods=int(macro_lag_periods),
            transition_uncertainty=float(transition_uncertainty),
            degrees_of_freedom=5.0,
            rebalance_frequency=rebalance_frequency,
            transaction_cost_bps=float(transaction_cost_bps),
            base_currency=base_currency,
            asset_currencies=asset_currencies,
            fx_rates=fx_rates,
        )
    except Exception as exc:
        raise gr.Error(f"Scenario comparison failed: {exc}")


# ---------- Build the Gradio app ----------

with gr.Blocks(title="Four-Quadrant Monte Carlo Simulator") as demo:
    gr.Markdown(
        """
        # Four-Quadrant Monte Carlo Simulator

        A Monte Carlo simulator built around the classic four macro quadrants:
        **Goldilocks**, **Overheating**, **Stagflation**, and **Recession**.
        Calibrate from real data, simulate regime-dependent asset returns, and
        analyze portfolio outcomes.
        """
    )

    # State
    macro_state = gr.State()
    returns_state = gr.State()
    tickers_state = gr.State()
    growth_col_state = gr.State(value="growth")
    inflation_col_state = gr.State(value="inflation")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Data Source")
            source = gr.Radio(
                ["Demo", "Yahoo Finance", "CSV upload"],
                value="Demo",
                label="Data source",
            )

            with gr.Group(visible=True) as demo_group:
                demo_seed = gr.Number(value=42, label="Demo seed", precision=0, minimum=1)

            with gr.Group(visible=False) as yahoo_group:
                ticker_text = gr.Textbox(
                    value="SPY, IEF, GLD, DBC, EFA, VNQ, TIP, SHY",
                    label="Market tickers",
                    info="Yahoo Finance symbols separated by commas or spaces.",
                )
                proxy_text = gr.Textbox(
                    value="",
                    label="Historical proxies (optional)",
                    info="ASSET:PROXY pairs, for example SPY:^GSPC, GLD:GC=F. Proxy levels are scaled during the overlap.",
                )
                synthetic_text = gr.Dropdown(
                    choices=SYNTHETIC_TICKER_OPTIONS,
                    value=[],
                    multiselect=True,
                    label="Synthetic backfill assets (optional)",
                    info="Selecting an asset automatically adds its live ticker. IEF, SHY, or DBMF become IEF_SIM/IEFSIM-style source pairs.",
                )
                synthetic_seed = gr.Number(value=42, label="Synthetic history seed", precision=0, minimum=1)
                start_date = gr.Textbox(value="1990-01-01", label="History start (YYYY-MM-DD)")
                end_date = gr.Textbox(value=date.today().isoformat(), label="History end (YYYY-MM-DD)")

            with gr.Group(visible=False) as csv_group:
                prices_file = gr.File(label="Asset CSV", file_types=[".csv"])
                macro_file = gr.File(label="Macro CSV", file_types=[".csv"])
                asset_input = gr.Radio(["Price levels", "Returns"], value="Price levels", label="Asset CSV values")
                monthly_returns = gr.Checkbox(value=True, label="Monthly asset returns")
                growth_col = gr.Dropdown(choices=["growth", "inflation"], value="growth", label="Growth column")
                inflation_col = gr.Dropdown(choices=["growth", "inflation"], value="inflation", label="Inflation column")

            load_btn = gr.Button("Load Data", variant="primary")
            load_msg = gr.Markdown()
            gr.Markdown(
                "**Simulated assets:** choose a live ticker above, then click **Load Data**. "
                "The portfolio list will show observed and `SIM` versions after loading."
            )

            gr.Markdown("### Currency")
            base_currency = gr.Textbox(
                value="USD",
                label="Portfolio currency",
                info="Any three-letter ISO code supported by Yahoo Finance, for example USD, EUR, GBP, JPY, or CHF.",
            )
            asset_currency_text = gr.Textbox(
                value="",
                label="Asset currencies (optional)",
                info="Use ASSET:CURRENCY pairs, for example EFA:EUR. Unspecified assets are treated as USD.",
            )

            gr.Markdown("### Calibration")
            growth_threshold = gr.Dropdown(
                ["median", "mean", "fixed:0.0", "fixed:1.0", "fixed:2.0", "fixed:3.0"],
                value="median",
                label="Growth threshold",
            )
            inflation_threshold = gr.Dropdown(
                ["median", "mean", "fixed:0.0", "fixed:1.0", "fixed:2.0", "fixed:3.0"],
                value="median",
                label="Inflation threshold",
            )
            macro_lag_periods = gr.Slider(
                0,
                3,
                value=1,
                step=1,
                label="Macro release lag (periods)",
                info="Use prior macro observations to reduce look-ahead bias.",
            )
            transition_uncertainty = gr.Slider(
                0.0,
                1.0,
                value=0.0,
                step=0.05,
                label="Transition uncertainty",
                info="0 uses the calibrated matrix; higher values sample more uncertain transitions.",
            )

            gr.Markdown("### Simulation")
            periods = gr.Slider(12, 360, value=120, step=12, label="Periods (months)")
            paths = gr.Slider(250, 20000, value=3000, step=250, label="Paths")
            seed = gr.Number(value=7, label="Random seed", precision=0, minimum=1)
            start_state = gr.Dropdown(
                ["Stationary"] + REGIME_LABELS,
                value="Stationary",
                label="Start state",
            )
            return_distribution = gr.Dropdown(
                choices=list(RETURN_DISTRIBUTIONS),
                value="Normal",
                label="Return distribution",
            )
            degrees_of_freedom = gr.Slider(
                3.0,
                30.0,
                value=5.0,
                step=1.0,
                label="Student-t degrees of freedom",
                info="Lower values produce heavier tails.",
                visible=False,
            )
            block_size = gr.Slider(
                2,
                12,
                value=3,
                step=1,
                label="Bootstrap block size",
                info="Consecutive observations sampled by block bootstrap.",
                visible=False,
            )
            rebalance_frequency = gr.Dropdown(
                choices=list(REBALANCE_FREQUENCIES),
                value="Monthly",
                label="Portfolio rebalancing",
            )
            transaction_cost_bps = gr.Slider(
                0.0,
                100.0,
                value=10.0,
                step=1.0,
                label="Transaction cost (basis points)",
                info="Charged on traded notional at each rebalance.",
            )

        with gr.Column(scale=2):
            gr.Markdown("### Portfolio")
            ticker_selector = gr.Dropdown(
                choices=DEFAULT_TICKER_ORDER,
                value=[],
                multiselect=True,
                interactive=False,
                label="Portfolio tickers",
                info="Select tickers to include in the simulation. Load Data refreshes this list for the selected source.",
            )
            weights_table = gr.Dataframe(
                headers=["Ticker", "Weight (%)"],
                datatype=["str", "number"],
                interactive=True,
                label="Portfolio Weights",
            )

            with gr.Accordion("Correlation Overrides", open=False):
                use_correlation_override = gr.Checkbox(
                    value=True,
                    label="Blend custom correlation view",
                    info="Blend the selected pair's empirical correlation with these investment-view targets.",
                )
                with gr.Row():
                    correlation_asset_a = gr.Dropdown(
                        choices=DEFAULT_TICKER_ORDER,
                        value="SPY",
                        label="First ticker",
                    )
                    correlation_asset_b = gr.Dropdown(
                        choices=DEFAULT_TICKER_ORDER,
                        value="IEF",
                        label="Second ticker",
                    )
                correlation_blend = gr.Slider(
                    0.0,
                    1.0,
                    value=0.40,
                    step=0.05,
                    label="View blend",
                )
                corr_high_growth_low_inflation = gr.Slider(
                    -1.0,
                    1.0,
                    value=DEFAULT_CORRELATIONS[REGIME_ORDER[0]],
                    step=0.05,
                    label=REGIME_NAMES[REGIME_ORDER[0]],
                )
                corr_high_growth_high_inflation = gr.Slider(
                    -1.0,
                    1.0,
                    value=DEFAULT_CORRELATIONS[REGIME_ORDER[1]],
                    step=0.05,
                    label=REGIME_NAMES[REGIME_ORDER[1]],
                )
                corr_low_growth_high_inflation = gr.Slider(
                    -1.0,
                    1.0,
                    value=DEFAULT_CORRELATIONS[REGIME_ORDER[2]],
                    step=0.05,
                    label=REGIME_NAMES[REGIME_ORDER[2]],
                )
                corr_low_growth_low_inflation = gr.Slider(
                    -1.0,
                    1.0,
                    value=DEFAULT_CORRELATIONS[REGIME_ORDER[3]],
                    step=0.05,
                    label=REGIME_NAMES[REGIME_ORDER[3]],
                )

            run_btn = gr.Button("Run Simulation", variant="primary", size="lg", interactive=False)
            run_msg = gr.Markdown()
            selection_msg = gr.Markdown()

            gr.Markdown("### Results")
            summary_text = gr.Markdown()
            risk_text = gr.Markdown()
            with gr.Row():
                wealth_download = gr.File(label="Download wealth paths")
                summary_download = gr.File(label="Download risk summary")
                diagnostics_download = gr.File(label="Download diagnostics")
            compare_btn = gr.Button("Compare Normal vs Student-t", interactive=False)
            comparison_table = gr.Dataframe(
                label="Scenario Comparison",
                interactive=False,
            )

            with gr.Tabs():
                with gr.Tab("Simulation"):
                    with gr.Row():
                        with gr.Column():
                            wealth_plot = gr.Plot(label="Wealth Percentiles")
                        with gr.Column():
                            terminal_plot = gr.Plot(label="Terminal Wealth")
                    mix_plot = gr.Plot(label="Regime Mix")

                with gr.Tab("Regimes"):
                    with gr.Row():
                        with gr.Column():
                            transition_plot = gr.Plot(label="Transition Matrix")
                        with gr.Column():
                            scatter_plot = gr.Plot(label="Macro Quadrants")
                    obs_plot = gr.Plot(label="Observations")
                    diagnostics_table = gr.Dataframe(
                        label="Calibration Diagnostics",
                        interactive=False,
                    )
                    warnings_text = gr.Markdown()
                    with gr.Row():
                        transition_text = gr.Textbox(
                            label="Transition Matrix (text)",
                            lines=8,
                            interactive=False,
                        )
                        observations_text = gr.Textbox(
                            label="Historical Observations (text)",
                            lines=8,
                            interactive=False,
                        )

                with gr.Tab("Correlations"):
                    correlation_regime = gr.Dropdown(
                        REGIME_LABELS,
                        value=REGIME_LABELS[0],
                        label="Regime",
                    )
                    corr_plot = gr.Plot(label="Correlation Matrix")
                    corr_text = gr.Textbox(label="Correlation Matrix (text)", lines=8)

                with gr.Tab("Data"):
                    with gr.Row():
                        with gr.Column():
                            macro_table = gr.Dataframe(label="Macro Data", interactive=False)
                        with gr.Column():
                            returns_table = gr.Dataframe(label="Returns Data", interactive=False)

    # ---------- Event handlers ----------

    def toggle_groups(source):
        return (
            gr.update(visible=source == "Demo"),
            gr.update(visible=source == "Yahoo Finance"),
            gr.update(visible=source == "CSV upload"),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(value=[], interactive=False),
        )

    source.change(
        toggle_groups,
        inputs=[source],
        outputs=[demo_group, yahoo_group, csv_group, run_btn, compare_btn, ticker_selector],
    )

    def toggle_student_t(value):
        return (
            gr.update(visible=value == "Student-t"),
            gr.update(visible=value == "Block bootstrap"),
        )

    return_distribution.change(
        toggle_student_t,
        inputs=[return_distribution],
        outputs=[degrees_of_freedom, block_size],
    )

    def toggle_transaction_cost(value):
        if value == "Weighted log (legacy)":
            return gr.update(value=0.0, interactive=False)
        return gr.update(interactive=True)

    rebalance_frequency.change(
        toggle_transaction_cost,
        inputs=[rebalance_frequency],
        outputs=[transaction_cost_bps],
    )

    macro_file.change(
        macro_column_updates,
        inputs=[macro_file],
        outputs=[growth_col, inflation_col],
    )

    def on_load(
        source, demo_seed, ticker_text, proxy_text, synthetic_text, synthetic_seed, start_date, end_date,
        prices_file, macro_file, asset_input, monthly_returns,
        growth_col, inflation_col,
    ):
        macro, returns, tickers, gcol, icol, msg = load_data(
            source, demo_seed, ticker_text, proxy_text, synthetic_text, synthetic_seed, start_date, end_date,
            prices_file, macro_file, asset_input, monthly_returns,
            growth_col, inflation_col,
        )
        # Build default weights dataframe
        defaults = default_selected_tickers(tickers)
        weights_data = [[t, default_weight(t, defaults)] for t in defaults]
        correlation_a, correlation_b = default_correlation_pair(defaults)
        return (
            macro, returns, tickers, gcol, icol, msg,
            gr.update(choices=tickers, value=defaults, interactive=True),
            gr.update(value=weights_data, headers=["Ticker", "Weight (%)"], datatype=["str", "number"], interactive=True),
            gr.update(value=macro.tail(120)),
            gr.update(value=returns.tail(120)),
            gr.update(choices=defaults, value=correlation_a, interactive=correlation_a is not None),
            gr.update(choices=defaults, value=correlation_b, interactive=correlation_b is not None),
            gr.update(value=len(defaults) >= 2, interactive=len(defaults) >= 2),
            gr.update(interactive=bool(defaults)),
            gr.update(interactive=bool(defaults)),
        )

    load_btn.click(
        on_load,
        inputs=[
            source, demo_seed, ticker_text, proxy_text, synthetic_text, synthetic_seed, start_date, end_date,
            prices_file, macro_file, asset_input, monthly_returns,
            growth_col, inflation_col,
        ],
        outputs=[
            macro_state, returns_state, tickers_state,
            growth_col_state, inflation_col_state, load_msg,
            ticker_selector, weights_table, macro_table, returns_table,
            correlation_asset_a, correlation_asset_b, use_correlation_override,
            run_btn, compare_btn,
        ],
    )

    def on_ticker_change(tickers):
        if not tickers:
            return (
                gr.update(value=[], headers=["Ticker", "Weight (%)"], datatype=["str", "number"], interactive=True),
                gr.update(choices=[], value=None, interactive=False),
                gr.update(choices=[], value=None, interactive=False),
                gr.update(value=False, interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
            )
        weights_data = [[t, default_weight(t, tickers)] for t in tickers]
        correlation_a, correlation_b = default_correlation_pair(tickers)
        return (
            gr.update(value=weights_data, headers=["Ticker", "Weight (%)"], datatype=["str", "number"], interactive=True),
            gr.update(choices=tickers, value=correlation_a, interactive=correlation_a is not None),
            gr.update(choices=tickers, value=correlation_b, interactive=correlation_b is not None),
            gr.update(value=len(tickers) >= 2, interactive=len(tickers) >= 2),
            gr.update(interactive=True),
            gr.update(interactive=True),
        )

    ticker_selector.change(
        on_ticker_change,
        inputs=[ticker_selector],
        outputs=[weights_table, correlation_asset_a, correlation_asset_b, use_correlation_override, run_btn, compare_btn],
    )

    def on_run(
        macro, returns, selected_tickers, growth_col, inflation_col,
        growth_threshold, inflation_threshold, periods, paths, seed, start_state,
        weights_df, correlation_regime, use_corr_override, correlation_a, correlation_b,
        correlation_weight, corr_hgli, corr_hghi, corr_lghi, corr_lgli,
        return_distribution, degrees_df, block_size_value, rebalance_label, transaction_cost,
        macro_lag, transition_uncertainty_value,
        base_currency_value, asset_currency_text_value,
        progress=gr.Progress(track_tqdm=True),
    ):
        progress(0.05, desc="Calibrating model")
        return run_simulation(
            macro, returns, selected_tickers, growth_col, inflation_col,
            growth_threshold, inflation_threshold, periods, paths, seed, start_state,
            weights_df, correlation_regime, use_corr_override, correlation_a, correlation_b,
            correlation_weight, corr_hgli, corr_hghi, corr_lghi, corr_lgli,
            return_distribution, degrees_df, block_size_value, rebalance_label, transaction_cost,
            macro_lag, transition_uncertainty_value,
            base_currency_value, asset_currency_text_value,
        )

    run_btn.click(
        on_run,
        inputs=[
            macro_state, returns_state, ticker_selector,
            growth_col_state, inflation_col_state,
            growth_threshold, inflation_threshold,
            periods, paths, seed, start_state,
            weights_table,
            correlation_regime, use_correlation_override, correlation_asset_a, correlation_asset_b,
            correlation_blend,
            corr_high_growth_low_inflation, corr_high_growth_high_inflation,
            corr_low_growth_high_inflation, corr_low_growth_low_inflation,
            return_distribution, degrees_of_freedom, block_size, rebalance_frequency, transaction_cost_bps,
            macro_lag_periods, transition_uncertainty,
            base_currency, asset_currency_text,
        ],
        outputs=[
            wealth_plot, terminal_plot, mix_plot,
            transition_plot, scatter_plot, obs_plot, corr_plot,
            transition_text, observations_text, corr_text,
            summary_text, run_msg, selection_msg,
            risk_text, diagnostics_table, warnings_text,
            wealth_download, summary_download, diagnostics_download,
        ],
    )

    compare_btn.click(
        compare_scenarios,
        inputs=[
            macro_state, returns_state, ticker_selector,
            growth_col_state, inflation_col_state,
            growth_threshold, inflation_threshold,
            periods, paths, seed, start_state, weights_table,
            use_correlation_override, correlation_asset_a, correlation_asset_b,
            correlation_blend,
            corr_high_growth_low_inflation, corr_high_growth_high_inflation,
            corr_low_growth_high_inflation, corr_low_growth_low_inflation,
            rebalance_frequency, transaction_cost_bps, macro_lag_periods,
            transition_uncertainty,
            base_currency, asset_currency_text,
        ],
        outputs=[comparison_table],
    )


if __name__ == "__main__":
    demo.launch(
        theme=gr.themes.Soft(),
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.getenv("PORT", "7860")),
    )
