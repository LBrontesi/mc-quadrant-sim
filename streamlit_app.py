from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import date

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from mc_quadrants.data import (
    load_market_data as load_market_data_shared,
)
from mc_quadrants.data import (
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


st.set_page_config(
    page_title="Four-Quadrant Monte Carlo",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def demo_history(seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    macro, returns = _demo_history(seed)
    return macro, returns.rename(columns=DEMO_TICKERS)


def read_uploaded_csv(uploaded_file) -> pd.DataFrame | None:
    if uploaded_file is None:
        return None

    data = pd.read_csv(uploaded_file)
    if "Date" not in data.columns:
        st.error("CSV files need a Date column.")
        st.stop()

    data["Date"] = pd.to_datetime(data["Date"])
    return data.set_index("Date").sort_index()


def threshold_control(label: str, key: str) -> str | float:
    mode = st.sidebar.selectbox(
        label,
        ["median", "mean", "fixed"],
        key=f"{key}_mode",
    )
    if mode == "fixed":
        return st.sidebar.number_input(
            f"{label} value",
            value=0.0,
            step=0.25,
            key=f"{key}_value",
        )
    return mode


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
    }
    return defaults.get(base_asset, round(100.0 / max(len(assets), 1), 2))


def normalize_ticker_columns(data: pd.DataFrame) -> pd.DataFrame:
    normalized = data.copy()
    normalized.columns = [str(column).strip().upper() for column in normalized.columns]
    return normalized


def default_selected_tickers(tickers: list[str]) -> list[str]:
    preferred = [
        f"{ticker}SIM" if f"{ticker}SIM" in tickers else ticker
        for ticker in DEFAULT_TICKER_ORDER
        if ticker in tickers or f"{ticker}SIM" in tickers
    ]
    if preferred:
        return preferred
    return tickers[: min(4, len(tickers))]


def preferred_asset(asset: str, assets: list[str]) -> str | None:
    for candidate in (f"{asset}SIM", asset, f"{asset}_SIM"):
        if candidate in assets:
            return candidate
    return None


def parse_tickers(raw_tickers: str) -> list[str]:
    """Return unique Yahoo Finance tickers from a comma or space-separated field."""

    parsed: list[str] = []
    for ticker in re.split(r"[,;\s]+", raw_tickers.strip().upper()):
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


@st.cache_data(ttl=3_600, show_spinner=False)
def load_market_data(
    tickers: tuple[str, ...],
    start: date,
    end: date,
    historical_proxies: tuple[tuple[str, str], ...] = (),
    synthetic_assets: tuple[str, ...] = (),
    synthetic_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[str, ...]]:
    """Load market prices plus FRED industrial production and CPI macro inputs."""
    macro, returns, available = load_market_data_shared(
        tickers,
        start=start.isoformat(),
        end=end.isoformat(),
        historical_proxies=dict(historical_proxies),
        synthetic_assets=synthetic_assets,
        synthetic_seed=synthetic_seed,
    )
    return macro, returns, tuple(available)


def ticker_selector(returns: pd.DataFrame) -> list[str]:
    tickers = list(returns.dropna(how="all").columns)
    if not tickers:
        st.error("No ticker columns were found in the asset data.")
        st.stop()

    selected = st.sidebar.multiselect(
        "Portfolio tickers",
        options=tickers,
        default=default_selected_tickers(tickers),
        help="The model will calibrate and simulate only the selected tickers.",
    )
    if not selected:
        st.warning("Select at least one ticker to run the simulation.")
        st.stop()
    return selected


def matrix_chart(
    matrix: pd.DataFrame,
    value_name: str,
    domain: tuple[float, float],
    scheme: str,
    height: int = 320,
) -> alt.Chart:
    chart_data = (
        matrix.reset_index(names="from")
        .melt(id_vars="from", var_name="to", value_name=value_name)
        .replace({"from": REGIME_NAMES, "to": REGIME_NAMES})
    )
    base = (
        alt.Chart(chart_data)
        .mark_rect()
        .encode(
            x=alt.X("to:N", title=None, sort=None),
            y=alt.Y("from:N", title=None, sort=None),
            color=alt.Color(
                f"{value_name}:Q",
                scale=alt.Scale(domain=list(domain), scheme=scheme),
                title=value_name.replace("_", " ").title(),
            ),
            tooltip=[
                alt.Tooltip("from:N", title="From"),
                alt.Tooltip("to:N", title="To"),
                alt.Tooltip(f"{value_name}:Q", title="Value", format=".3f"),
            ],
        )
    )
    labels = (
        alt.Chart(chart_data)
        .mark_text(fontSize=12)
        .encode(
            x=alt.X("to:N", sort=None),
            y=alt.Y("from:N", sort=None),
            text=alt.Text(f"{value_name}:Q", format=".2f"),
            color=alt.condition(
                alt.datum[value_name] > (domain[0] + domain[1]) / 2,
                alt.value("white"),
                alt.value("#111827"),
            ),
        )
    )
    return (base + labels).properties(height=height)


def correlation_overrides(
    assets: list[str],
) -> tuple[Mapping[str, Mapping[tuple[str, str], float]] | None, float]:
    if len(assets) < 2:
        return None, 0.0

    use_override = st.sidebar.toggle("Blend custom correlation", value=True)
    if not use_override:
        return None, 0.0

    default_a = preferred_asset("SPY", assets) or assets[0]
    default_b = preferred_asset("IEF", assets) or assets[min(1, len(assets) - 1)]
    asset_a = st.sidebar.selectbox("First ticker", assets, index=assets.index(default_a))
    asset_b_options = [asset for asset in assets if asset != asset_a]
    asset_b = st.sidebar.selectbox(
        "Second ticker",
        asset_b_options,
        index=asset_b_options.index(default_b) if default_b in asset_b_options else 0,
    )
    weight = st.sidebar.slider("View blend", 0.0, 1.0, 0.40, 0.05)

    overrides: dict[str, dict[tuple[str, str], float]] = {}
    for state in REGIME_ORDER:
        value = st.sidebar.slider(
            REGIME_NAMES[state],
            -1.0,
            1.0,
            float(DEFAULT_CORRELATIONS[state]),
            0.05,
            key=f"corr_{state}",
        )
        overrides[state] = {(asset_a, asset_b): value}

    return overrides, weight


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, str, str]:
    source = st.sidebar.radio(
        "Data source",
        ["Demo", "Yahoo Finance", "CSV upload"],
        horizontal=True,
    )

    if source == "Demo":
        seed = st.sidebar.number_input("Demo seed", value=42, min_value=1, step=1)
        macro, returns = demo_history(int(seed))
        return macro, normalize_ticker_columns(returns), "growth", "inflation"

    if source == "Yahoo Finance":
        ticker_text = st.sidebar.text_area(
            "Market tickers",
            value="SPY, IEF, GLD, DBC, EFA, VNQ, TIP, SHY",
            help="Use Yahoo Finance symbols separated by commas or spaces.",
        )
        proxy_text = st.sidebar.text_input(
            "Historical proxies (optional)",
            value="",
            help="Use ASSET:PROXY pairs, for example SPY:^GSPC, GLD:GC=F. Proxy levels are scaled to the asset during the overlap.",
        )
        synthetic_text = st.sidebar.text_input(
            "Synthetic backfill assets (optional)",
            value="",
            help="Enter observed tickers such as IEF, SHY, or DBMF; each must also be in Market tickers. The loader creates *_SIM and *SIM columns using reproducible Student-t draws.",
        )
        synthetic_seed = st.sidebar.number_input(
            "Synthetic history seed",
            value=42,
            min_value=1,
            step=1,
            disabled=not synthetic_text.strip(),
        )
        tickers = parse_tickers(ticker_text)
        if not tickers:
            st.error("Enter at least one Yahoo Finance ticker.")
            st.stop()

        start = st.sidebar.date_input("History start", value=date(1990, 1, 1))
        end = st.sidebar.date_input("History end", value=date.today())
        if end <= start:
            st.error("History end must be after history start.")
            st.stop()

        try:
            historical_proxies = parse_proxy_map(proxy_text)
            synthetic_assets = parse_tickers(synthetic_text)
            with st.spinner("Downloading prices and macro data..."):
                macro, returns, available = load_market_data(
                    tuple(tickers),
                    start,
                    end,
                    tuple(historical_proxies.items()),
                    tuple(synthetic_assets),
                    int(synthetic_seed),
                )
        except (ImportError, ValueError, RuntimeError) as exc:
            st.error(f"Could not load market data: {exc}")
            st.stop()

        unavailable = [ticker for ticker in tickers if ticker not in available]
        if unavailable:
            st.warning(f"No usable price history was returned for: {', '.join(unavailable)}")
        if synthetic_assets:
            st.info(
                "Synthetic sources loaded: "
                + ", ".join(f"{asset}_SIM -> {asset}SIM" for asset in synthetic_assets)
            )
        return macro, returns, "growth", "inflation"

    prices_file = st.sidebar.file_uploader("Asset CSV", type="csv")
    macro_file = st.sidebar.file_uploader("Macro CSV", type="csv")
    asset_data = read_uploaded_csv(prices_file)
    macro = read_uploaded_csv(macro_file)

    if asset_data is None or macro is None:
        st.info("Upload an asset CSV and macro CSV to calibrate from your own data.")
        st.stop()

    asset_input = st.sidebar.selectbox("Asset CSV values", ["Price levels", "Returns"])
    if asset_input == "Price levels":
        returns = prices_to_returns(asset_data, method="log")
    else:
        returns = asset_data.astype(float)

    if st.sidebar.toggle("Monthly asset returns", value=True):
        returns = returns.resample("ME").sum()
    macro = macro.resample("ME").last()
    returns = normalize_ticker_columns(returns)

    growth_col = st.sidebar.selectbox("Growth column", list(macro.columns))
    inflation_options = [col for col in macro.columns if col != growth_col]
    inflation_col = st.sidebar.selectbox(
        "Inflation column",
        inflation_options or list(macro.columns),
    )
    return macro, returns, growth_col, inflation_col


def macro_scatter(
    macro: pd.DataFrame,
    regimes: pd.Series,
    growth_col: str,
    inflation_col: str,
) -> alt.Chart:
    data = macro[[growth_col, inflation_col]].copy()
    data["regime"] = regimes.map(REGIME_NAMES)
    data["date"] = data.index.astype(str)
    colors = [REGIME_COLORS[state] for state in REGIME_ORDER]
    labels = [REGIME_NAMES[state] for state in REGIME_ORDER]

    return (
        alt.Chart(data.dropna())
        .mark_circle(size=58, opacity=0.72)
        .encode(
            x=alt.X(f"{growth_col}:Q", title="Growth"),
            y=alt.Y(f"{inflation_col}:Q", title="Inflation"),
            color=alt.Color(
                "regime:N",
                scale=alt.Scale(domain=labels, range=colors),
                title="Regime",
            ),
            tooltip=[
                alt.Tooltip("date:N", title="Date"),
                alt.Tooltip("regime:N", title="Regime"),
                alt.Tooltip(f"{growth_col}:Q", title="Growth", format=".2f"),
                alt.Tooltip(f"{inflation_col}:Q", title="Inflation", format=".2f"),
            ],
        )
        .properties(height=420)
    )


def terminal_distribution(terminal: pd.Series) -> alt.Chart:
    data = pd.DataFrame({"terminal_wealth": terminal.to_numpy(dtype=float)})
    return (
        alt.Chart(data)
        .mark_bar(color="#2563eb", opacity=0.82)
        .encode(
            x=alt.X("terminal_wealth:Q", bin=alt.Bin(maxbins=45), title="Terminal wealth"),
            y=alt.Y("count():Q", title="Paths"),
            tooltip=[alt.Tooltip("count():Q", title="Paths")],
        )
        .properties(height=320)
    )


st.title("Four-Quadrant Monte Carlo Simulator")

with st.sidebar:
    st.header("Calibration")
    macro, returns, growth_col, inflation_col = load_inputs()
    selected_tickers = ticker_selector(returns)
    returns = returns[selected_tickers]
    growth_threshold = threshold_control("Growth threshold", "growth")
    inflation_threshold = threshold_control("Inflation threshold", "inflation")
    macro_lag_periods = st.sidebar.slider(
        "Macro release lag (periods)",
        min_value=0,
        max_value=3,
        value=1,
        help="Use prior macro observations to reduce look-ahead bias.",
    )
    transition_uncertainty = st.sidebar.slider(
        "Transition uncertainty",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05,
        help="0 uses the calibrated matrix; higher values sample more uncertain transitions.",
    )

    overrides, override_weight = correlation_overrides(selected_tickers)

    st.header("Simulation")
    periods = st.slider("Periods", min_value=12, max_value=360, value=120, step=12)
    paths = st.slider("Paths", min_value=250, max_value=20000, value=3000, step=250)
    seed = st.number_input("Random seed", value=7, min_value=1, step=1)
    start_options = ["Stationary"] + [REGIME_NAMES[state] for state in REGIME_ORDER]
    start_label = st.selectbox("Start state", start_options)
    start_state = None
    if start_label != "Stationary":
        start_state = {REGIME_NAMES[state]: state for state in REGIME_ORDER}[start_label]

    distribution_label = st.selectbox(
        "Return distribution",
        ["Normal", "Student-t", "Historical bootstrap", "Block bootstrap"],
    )
    degrees_of_freedom = 5.0
    block_size = 3
    if distribution_label == "Student-t":
        degrees_of_freedom = st.slider("Student-t degrees of freedom", 3.0, 30.0, 5.0, 1.0)
    if distribution_label == "Block bootstrap":
        block_size = st.slider("Bootstrap block size", 2, 12, 3, 1)
    rebalance_label = st.selectbox(
        "Portfolio rebalancing",
        ["Monthly", "Quarterly", "Annual", "Weighted log (legacy)"],
    )
    rebalance_frequency = {
        "Monthly": 1,
        "Quarterly": 3,
        "Annual": 12,
        "Weighted log (legacy)": None,
    }[rebalance_label]
    transaction_cost_bps = st.number_input(
        "Transaction cost (basis points)",
        min_value=0.0,
        max_value=100.0,
        value=0.0 if rebalance_frequency is None else 10.0,
        step=1.0,
        disabled=rebalance_frequency is None,
    )

    st.header("Portfolio")
    st.caption("Weights are entered for the selected tickers only.")
    weights = {
        ticker: st.number_input(
            ticker,
            min_value=0.0,
            max_value=100.0,
            value=default_weight(ticker, selected_tickers),
            step=1.0,
        )
        for ticker in selected_tickers
    }
    weight_total = sum(weights.values())
    st.caption(f"Total weight: {weight_total:.1f}%. The simulator normalizes this to 100%.")
    if np.isclose(weight_total, 0.0):
        st.warning("Set at least one ticker weight above zero.")
        st.stop()

st.caption(f"Selected tickers: {', '.join(selected_tickers)}")

try:
    scenario = run_scenario(
        returns=returns,
        macro=macro,
        selected_tickers=selected_tickers,
        growth_col=growth_col,
        inflation_col=inflation_col,
        growth_threshold=growth_threshold,
        inflation_threshold=inflation_threshold,
        periods=periods,
        paths=paths,
        random_seed=int(seed),
        start_state=start_state,
        weights=weights,
        correlation_overrides=overrides,
        override_weight=override_weight,
        macro_lag_periods=macro_lag_periods,
        distribution={
            "Normal": "normal",
            "Student-t": "student_t",
            "Historical bootstrap": "bootstrap",
            "Block bootstrap": "block_bootstrap",
        }[distribution_label],
        degrees_of_freedom=degrees_of_freedom,
        block_size=block_size,
        transition_uncertainty=transition_uncertainty,
        rebalance_frequency=rebalance_frequency,
        transaction_cost_bps=transaction_cost_bps,
    )
except (KeyError, TypeError, ValueError) as exc:
    st.error(f"Simulation failed: {exc}")
    st.stop()
model = scenario.model
regimes = scenario.regimes
result = scenario.result
wealth = scenario.wealth
summary = scenario.summary

metric_cols = st.columns(5)
metric_cols[0].metric("Mean", f"{summary['mean']:.2f}")
metric_cols[1].metric("P05", f"{summary['p05']:.2f}")
metric_cols[2].metric("Median", f"{summary['p50']:.2f}")
metric_cols[3].metric("P95", f"{summary['p95']:.2f}")
metric_cols[4].metric("Volatility", f"{summary['std']:.2f}")
st.caption(
    f"Probability of loss: {summary['probability_of_loss']:.1%} | "
    f"VaR (95%): {summary['var_95']:.2f} | "
    f"Expected shortfall (95%): {summary['expected_shortfall_95']:.2f} | "
    f"Worst max drawdown: {summary['max_drawdown_worst']:.1%}"
)

tab_simulation, tab_regimes, tab_correlations, tab_data = st.tabs(
    ["Simulation", "Regimes", "Correlations", "Data"]
)

with tab_simulation:
    left, right = st.columns([1.2, 1.0])
    with left:
        percentiles = wealth.quantile([0.05, 0.50, 0.95], axis=1).T
        percentiles.columns = ["P05", "Median", "P95"]
        st.subheader("Wealth Percentiles")
        st.line_chart(percentiles)
    with right:
        st.subheader("Terminal Wealth")
        st.altair_chart(terminal_distribution(wealth.iloc[-1]), width="stretch")

    regime_mix = (
        pd.Series(result.regimes.ravel())
        .value_counts(normalize=True)
        .reindex(REGIME_ORDER)
        .fillna(0.0)
        .rename(index=REGIME_NAMES)
    )
    st.subheader("Simulated Regime Mix")
    st.bar_chart(regime_mix)
    wealth_export = wealth.copy()
    wealth_export.insert(0, "period", range(1, len(wealth_export) + 1))
    summary_export = summary.rename("value").rename_axis("metric").reset_index()
    st.download_button(
        "Download wealth paths",
        wealth_export.to_csv(index=False),
        file_name="wealth_paths.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download risk summary",
        summary_export.to_csv(index=False),
        file_name="risk_summary.csv",
        mime="text/csv",
    )
    if st.button("Compare Normal vs Student-t"):
        with st.spinner("Running comparison scenarios..."):
            comparison = compare_distributions(
                {"Normal": "normal", "Student-t": "student_t"},
                returns=returns,
                macro=macro,
                selected_tickers=selected_tickers,
                growth_col=growth_col,
                inflation_col=inflation_col,
                growth_threshold=growth_threshold,
                inflation_threshold=inflation_threshold,
                periods=periods,
                paths=paths,
                random_seed=int(seed),
                start_state=start_state,
                weights=weights,
                correlation_overrides=overrides,
                override_weight=override_weight,
                macro_lag_periods=macro_lag_periods,
                transition_uncertainty=transition_uncertainty,
                degrees_of_freedom=degrees_of_freedom,
                rebalance_frequency=rebalance_frequency,
                transaction_cost_bps=transaction_cost_bps,
            )
        st.subheader("Scenario Comparison")
        st.dataframe(comparison, width="stretch")

with tab_regimes:
    left, right = st.columns([1.0, 1.1])
    with left:
        st.subheader("Transition Matrix")
        transition_chart = matrix_chart(
            model.transition_matrix.rename(index=REGIME_NAMES, columns=REGIME_NAMES),
            value_name="probability",
            domain=(0.0, 1.0),
            scheme="blues",
        )
        st.altair_chart(transition_chart, width="stretch")
    with right:
        st.subheader("Macro Quadrants")
        st.altair_chart(
            macro_scatter(macro, regimes, growth_col, inflation_col),
            width="stretch",
        )

    observations = pd.Series(
        {REGIME_NAMES[state]: moments.observations for state, moments in model.moments.items()},
        name="Observations",
    )
    st.subheader("Historical Observations")
    st.bar_chart(observations)
    diagnostics = scenario.diagnostics.regime_summary.copy()
    simulated_diagnostics = simulation_regime_summary(result).rename(
        columns={
            "observations": "simulated_observations",
            "share": "simulated_share",
        }
    )
    diagnostics = diagnostics.merge(simulated_diagnostics, on="regime", how="left")
    diagnostics["regime"] = diagnostics["regime"].map(REGIME_NAMES)
    st.subheader("Calibration Diagnostics")
    st.dataframe(diagnostics, width="stretch")
    if scenario.diagnostics.warnings:
        st.warning("\n".join(scenario.diagnostics.warnings))
    st.download_button(
        "Download calibration diagnostics",
        diagnostics.to_csv(index=False),
        file_name="calibration_diagnostics.csv",
        mime="text/csv",
    )

with tab_correlations:
    regime_label = st.selectbox(
        "Regime",
        [REGIME_NAMES[state] for state in REGIME_ORDER],
    )
    regime_lookup = {REGIME_NAMES[state]: state for state in REGIME_ORDER}
    selected_regime = regime_lookup[regime_label]
    correlation = model.moments[selected_regime].correlation
    st.subheader("Correlation Matrix")
    st.altair_chart(
        matrix_chart(correlation, value_name="correlation", domain=(-1.0, 1.0), scheme="redblue"),
        width="stretch",
    )
    st.dataframe(correlation.round(3), width="stretch")

with tab_data:
    left, right = st.columns(2)
    with left:
        st.subheader("Macro")
        st.dataframe(macro.tail(120), width="stretch")
    with right:
        st.subheader("Returns")
        st.dataframe(returns.tail(120), width="stretch")
