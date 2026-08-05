"""Streamlit frontend for the Four-Quadrant Monte Carlo Simulator.

All data loading, scenario building, and result shaping is delegated to
``mc_quadrants.api`` so the simulation methodology is identical across
frontends. Run with:

    streamlit run streamlit_app.py
"""

from __future__ import annotations

import json
from datetime import date

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from mc_quadrants import api

REGIME_NAMES = api.REGIME_NAMES
REGIME_ORDER_NAMES = [api.REGIME_NAMES[state] for state in api.REGIME_ORDER]
REGIME_COLORS = {
    "High growth / low inflation": "#2f855a",
    "High growth / high inflation": "#d97706",
    "Low growth / high inflation": "#c2410c",
    "Low growth / low inflation": "#3b82f6",
}

st.set_page_config(page_title="Four-Quadrant Monte Carlo Simulator", page_icon="📈", layout="wide")

SUMMARY_METRICS = [
    ("mean", "Mean terminal wealth"),
    ("p05", "P05"),
    ("p50", "Median"),
    ("p95", "P95"),
    ("annualized_return", "Annualized return"),
    ("annualized_volatility", "Annualized volatility"),
    ("sharpe_ratio", "Sharpe ratio"),
    ("sortino_ratio", "Sortino"),
    ("calmar_ratio", "Calmar"),
    ("ulcer_index_mean", "Ulcer index"),
    ("geometric_annualized_return", "CAGR"),
    ("probability_of_loss", "Prob. of loss"),
    ("var_95", "VaR (95%)"),
    ("expected_shortfall_95", "Expected shortfall (95%)"),
    ("max_drawdown_worst", "Worst max drawdown"),
]


def pct(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}%"


def load_payload() -> dict:
    """Build the payload for /api/load-equivalent calls from the sidebar widgets."""
    source = st.session_state["source"]
    payload: dict = {"source": source}
    if source == "demo":
        payload["seed"] = st.session_state["demo_seed"]
    elif source == "yahoo":
        payload["tickers"] = st.session_state["yahoo_tickers"]
        payload["start"] = st.session_state["yahoo_start"]
        payload["end"] = st.session_state["yahoo_end"]
        payload["proxies"] = st.session_state["yahoo_proxies"]
        payload["synthetic"] = st.session_state["yahoo_synthetic"]
        payload["synthetic_seed"] = st.session_state["synthetic_seed"]
    else:
        payload["csv_prices"] = st.session_state.get("csv_prices_text")
        payload["csv_macro"] = st.session_state.get("csv_macro_text")
        payload["asset_input"] = st.session_state["asset_input"]
        payload["monthly"] = st.session_state["csv_monthly"]
        payload["growth_col"] = st.session_state["csv_growth"]
        payload["inflation_col"] = st.session_state["csv_inflation"]
    return payload


def scenario_payload() -> dict:
    return {
        "growth_threshold": st.session_state["growth_threshold"],
        "inflation_threshold": st.session_state["inflation_threshold"],
        "macro_lag": st.session_state["macro_lag"],
        "transition_uncertainty": st.session_state["transition_uncertainty"],
        "periods": st.session_state["periods"],
        "paths": st.session_state["paths"],
        "seed": st.session_state["seed"],
        "start_state": st.session_state["start_state"],
        "distribution": st.session_state["distribution"],
        "degrees_of_freedom": st.session_state["degrees_of_freedom"],
        "block_size": st.session_state["block_size"],
        "rebalance": st.session_state["rebalance"],
        "cost_bps": st.session_state["cost_bps"],
        "risk_free_rate": st.session_state["risk_free_rate"] / 100.0,
        "annual_inflation": st.session_state["annual_inflation"] / 100.0,
        "base_currency": st.session_state["base_currency"],
        "currency_map": st.session_state["currency_map"],
        "use_correlation_override": st.session_state["use_corr_override"],
        "correlation_blend": st.session_state["corr_blend"],
        "correlation_override_targets": {
            name: st.session_state[f"corr_{name}"] for name in REGIME_ORDER_NAMES
        },
    }


def sim_payload(selected: list[str], weights: dict[str, float]) -> dict:
    payload = load_payload()
    payload.update(scenario_payload())
    payload["selected_tickers"] = selected
    payload["weights"] = weights
    return payload


# ---------- Sidebar: data source ----------

st.sidebar.header("Data source")
st.sidebar.radio("Source", ["demo", "yahoo", "csv"], key="source", horizontal=True)
if st.session_state["source"] == "demo":
    st.sidebar.number_input("Demo seed", min_value=1, value=42, key="demo_seed")
elif st.session_state["source"] == "yahoo":
    st.sidebar.text_area(
        "Market tickers",
        value="SPY, IEF, GLD, DBC, EFA, VNQ, TIP, SHY",
        key="yahoo_tickers",
    )
    st.sidebar.date_input("Start", value=date(1990, 1, 1), key="yahoo_start")
    st.sidebar.date_input("End", value=date.today(), key="yahoo_end")
    st.sidebar.text_input(
        "Historical proxies (ASSET:PROXY)",
        key="yahoo_proxies",
        placeholder="SPY:^GSPC, GLD:GC=F",
    )
    st.sidebar.multiselect(
        "Synthetic backfill assets",
        api.SYNTHETIC_TICKER_OPTIONS,
        key="yahoo_synthetic",
    )
    st.sidebar.number_input("Synthetic history seed", min_value=1, value=42, key="synthetic_seed")
else:
    st.sidebar.file_uploader("Asset CSV", type=["csv"], key="csv_prices_file")
    st.sidebar.file_uploader("Macro CSV", type=["csv"], key="csv_macro_file")
    st.sidebar.radio("Asset input", ["Price levels", "Returns"], key="asset_input", horizontal=True)
    st.sidebar.checkbox("Monthly asset returns", value=True, key="csv_monthly")
    st.sidebar.text_input("Growth column", value="growth", key="csv_growth")
    st.sidebar.text_input("Inflation column", value="inflation", key="csv_inflation")
    if st.session_state.get("csv_prices_file"):
        st.session_state["csv_prices_text"] = st.session_state["csv_prices_file"].getvalue().decode("utf-8")
    if st.session_state.get("csv_macro_file"):
        st.session_state["csv_macro_text"] = st.session_state["csv_macro_file"].getvalue().decode("utf-8")

st.sidebar.header("Calibration")
st.sidebar.selectbox("Growth threshold", ["median", "mean"], key="growth_threshold")
st.sidebar.selectbox("Inflation threshold", ["median", "mean"], key="inflation_threshold")
st.sidebar.slider("Macro release lag", 0, 3, 1, key="macro_lag")
st.sidebar.slider("Transition uncertainty", 0.0, 1.0, 0.0, 0.05, key="transition_uncertainty")

st.sidebar.header("Simulation")
st.sidebar.number_input("Periods (months)", min_value=12, max_value=360, value=120, step=12, key="periods")
st.sidebar.number_input("Paths", min_value=250, max_value=20000, value=3000, step=250, key="paths")
st.sidebar.number_input("Random seed", min_value=1, value=7, key="seed")
st.sidebar.selectbox("Start state", ["Stationary"] + REGIME_ORDER_NAMES, key="start_state")
st.sidebar.selectbox(
    "Return distribution",
    ["normal", "student_t", "bootstrap", "block_bootstrap"],
    format_func=lambda value: {
        "normal": "Normal",
        "student_t": "Student-t",
        "bootstrap": "Historical bootstrap",
        "block_bootstrap": "Block bootstrap",
    }[value],
    key="distribution",
)
st.sidebar.number_input("Student-t dof", min_value=3, max_value=30, value=5, key="degrees_of_freedom")
st.sidebar.number_input("Block size", min_value=2, max_value=12, value=3, key="block_size")
st.sidebar.selectbox("Rebalancing", ["monthly", "quarterly", "annual", "legacy"], key="rebalance")
st.sidebar.number_input("Cost (bps)", min_value=0, max_value=100, value=10, key="cost_bps")
st.sidebar.number_input("Risk-free rate (annual %)", value=0.0, step=0.1, key="risk_free_rate")
st.sidebar.number_input("Inflation assumption (annual %)", value=0.0, step=0.1, key="annual_inflation")
st.sidebar.text_input("Portfolio currency", value="USD", key="base_currency")
st.sidebar.text_input("Asset currencies (ASSET:CURRENCY)", key="currency_map", placeholder="EFA:EUR")

st.sidebar.header("Correlation overrides")
st.sidebar.checkbox("Blend custom correlation view", value=True, key="use_corr_override")
st.sidebar.slider("Blend", 0.0, 1.0, 0.4, 0.05, key="corr_blend")
for name in REGIME_ORDER_NAMES:
    default = api.DEFAULT_CORRELATIONS[next(state for state in api.REGIME_ORDER if api.REGIME_NAMES[state] == name)]
    st.sidebar.slider(name, -1.0, 1.0, default, 0.05, key=f"corr_{name}")


# ---------- Load data ----------

st.title("Four-Quadrant Monte Carlo Simulator")
st.caption("Goldilocks · Overheating · Stagflation · Recession — regime-based portfolio simulation")

with st.spinner("Loading data..."):
    try:
        load = api.build_load_response(*api.load_data_source(load_payload())[:5], "")
        st.session_state["tickers"] = load["tickers"]
        st.session_state["default_tickers"] = load["default_tickers"]
        st.session_state["presets"] = load["presets"]
        st.session_state["load_ok"] = True
    except Exception as exc:  # noqa: BLE001
        st.session_state["load_ok"] = False
        st.error(str(exc))

if not st.session_state.get("load_ok"):
    st.stop()

tickers = st.session_state["tickers"]

st.subheader("Portfolio")
preset_names = {preset["name"]: preset["weights"] for preset in st.session_state["presets"]}
preset_pick = st.selectbox("Preset (applies weights to matching tickers)", ["— choose —", *preset_names])
selected = st.multiselect("Tickers", tickers, default=st.session_state["default_tickers"])

weights: dict[str, float] = {}
if preset_pick != "— choose —":
    matched = {ticker: weight for ticker, weight in preset_names[preset_pick].items() if ticker in selected}
    total = sum(matched.values())
    for ticker in selected:
        weights[ticker] = matched.get(ticker, 0.0) / total * 100 if total else 0.0
else:
    for ticker in selected:
        weights[ticker] = api.default_weights(ticker)

weight_cols = st.columns(len(selected)) if selected else [st.columns(1)[0]]
for column, ticker in zip(weight_cols, selected):
    weights[ticker] = column.number_input(
        ticker, min_value=0.0, max_value=100.0, value=float(weights[ticker]), step=1.0, key=f"w_{ticker}"
    )
if selected:
    total = sum(weights.values())
    st.caption(f"Total weight: {total:.1f}% (the simulator normalizes this to 100%).")

run = st.button("Run Simulation", type="primary", disabled=not selected or sum(weights.values()) <= 0)

# ---------- Results ----------

if run:
    payload = sim_payload(selected, weights)
    with st.spinner("Running simulation..."):
        try:
            results = api.build_simulate_response(payload)
            st.session_state["results"] = results
            st.session_state["last_sim_payload"] = payload
        except Exception as exc:  # noqa: BLE001
            st.session_state["results"] = None
            st.error(str(exc))

results = st.session_state.get("results")
if results is None:
    st.info("Load data, pick tickers and weights, then press **Run Simulation**.")
    st.stop()

summary = results["summary"]
st.success(results["message"])
st.caption(
    f"{'Real (inflation-adjusted)' if results['terms'] == 'real' else 'Nominal'} · "
    f"Currency: {results['currency']} · "
    f"VaR (95%): {summary['var_95']:.2f} · "
    f"Worst max drawdown: {pct(summary['max_drawdown_worst'])}"
)
if results["warnings"]:
    st.warning("\n".join(results["warnings"]))

metric_cols = st.columns(5)
for index, (key, label) in enumerate(SUMMARY_METRICS):
    value = summary.get(key)
    metric_cols[index % 5].metric(label, "—" if value is None or not np.isfinite(value) else f"{value:,.2f}")

wealth = pd.DataFrame(
    {
        "Period": results["wealth"]["periods"],
        "P05": results["wealth"]["p05"],
        "Median": results["wealth"]["median"],
        "P95": results["wealth"]["p95"],
    }
)

chart_left, chart_right = st.columns(2)
with chart_left:
    st.subheader("Wealth Percentiles")
    long_wealth = wealth.melt(id_vars="Period", var_name="Percentile", value_name="Wealth")
    st.altair_chart(
        alt.Chart(long_wealth)
        .mark_line(strokeWidth=2.5)
        .encode(
            x=alt.X("Period:Q", title="Period", axis=alt.Axis(format="d")),
            y=alt.Y("Wealth:Q", title="Wealth"),
            color=alt.Color(
                "Percentile:N",
                scale=alt.Scale(domain=["P05", "Median", "P95"], range=["#f97316", "#3b82f6", "#10b981"]),
            ),
        )
        .properties(height=300),
        width="stretch",
    )

with chart_right:
    st.subheader("Terminal Wealth Distribution")
    st.altair_chart(
        alt.Chart(pd.DataFrame({"Terminal wealth": results["terminal"]}))
        .mark_bar(opacity=0.82)
        .encode(
            x=alt.X("Terminal wealth:Q", bin=alt.Bin(maxbins=45), title="Terminal wealth"),
            y=alt.Y("count()", title="Paths"),
        )
        .properties(height=300),
        width="stretch",
    )

chart_row2_left, chart_row2_right = st.columns(2)
with chart_row2_left:
    st.subheader("Maximum Drawdown Distribution")
    st.altair_chart(
        alt.Chart(pd.DataFrame({"Maximum drawdown": results["drawdowns"]}))
        .mark_bar(opacity=0.82)
        .encode(
            x=alt.X("Maximum drawdown:Q", bin=alt.Bin(maxbins=45), title="Maximum drawdown"),
            y=alt.Y("count()", title="Paths"),
        )
        .properties(height=300),
        width="stretch",
    )

with chart_row2_right:
    st.subheader("Simulated Regime Mix")
    regime_mix = pd.DataFrame(results["regime_mix"])
    st.altair_chart(
        alt.Chart(regime_mix)
        .mark_bar()
        .encode(
            x=alt.X("label:N", title=None, sort=REGIME_ORDER_NAMES),
            y=alt.Y("share:Q", title="Share"),
            color=alt.Color(
                "label:N",
                scale=alt.Scale(domain=REGIME_ORDER_NAMES, range=[REGIME_COLORS[name] for name in REGIME_ORDER_NAMES]),
                legend=None,
            ),
        )
        .properties(height=300),
        width="stretch",
    )

chart_row3_left, chart_row3_right = st.columns(2)
with chart_row3_left:
    st.subheader("Macro Quadrants (historical)")
    scatter = pd.DataFrame(results["macro_scatter"])
    st.altair_chart(
        alt.Chart(scatter)
        .mark_circle(opacity=0.75, size=60)
        .encode(
            x=alt.X("growth:Q", title="Growth"),
            y=alt.Y("inflation:Q", title="Inflation"),
            color=alt.Color(
                "regime:N",
                scale=alt.Scale(domain=REGIME_ORDER_NAMES, range=[REGIME_COLORS[name] for name in REGIME_ORDER_NAMES]),
            ),
            tooltip=["date", "growth", "inflation", "regime"],
        )
        .properties(height=300),
        width="stretch",
    )

with chart_row3_right:
    st.subheader("Transition Matrix")
    transition = results["transition"]
    matrix_rows = [
        {"From": row_label, "To": col_label, "Probability": transition["values"][row][col]}
        for row, row_label in enumerate(transition["labels"])
        for col, col_label in enumerate(transition["labels"])
    ]
    st.altair_chart(
        alt.Chart(pd.DataFrame(matrix_rows))
        .mark_rect()
        .encode(
            x=alt.X("To:N", sort=REGIME_ORDER_NAMES),
            y=alt.Y("From:N", sort=REGIME_ORDER_NAMES),
            color=alt.Color("Probability:Q", scale=alt.Scale(scheme="blues", domain=[0, 1])),
            tooltip=["From", "To", "Probability"],
        )
        .properties(height=300),
        width="stretch",
    )

st.subheader("Regime-Specific Correlations")
correlation_regime = st.selectbox("Regime", list(results["correlations"].keys()))
corr = results["correlations"][correlation_regime]
corr_rows = [
    {"From": row_label, "To": col_label, "Correlation": corr["values"][row][col]}
    for row, row_label in enumerate(corr["labels"])
    for col, col_label in enumerate(corr["labels"])
]
st.altair_chart(
    alt.Chart(pd.DataFrame(corr_rows))
    .mark_rect()
    .encode(
        x=alt.X("To:N", sort=corr["labels"]),
        y=alt.Y("From:N", sort=corr["labels"]),
        color=alt.Color("Correlation:Q", scale=alt.Scale(scheme="redblue", domain=[-1, 1])),
        tooltip=["From", "To", "Correlation"],
    )
    .properties(height=400),
    width="stretch",
)

st.subheader("Calibration Diagnostics")
diagnostics = pd.DataFrame(results["diagnostics"]["rows"], columns=results["diagnostics"]["columns"])
st.dataframe(diagnostics, width="stretch")

st.subheader("Scenario Comparison (Normal vs Student-t)")
compare_cols = st.columns([1, 1])
with compare_cols[0]:
    do_compare = st.button("Compare distributions")
if do_compare:
    with st.spinner("Comparing distributions..."):
        try:
            comparison = api.build_compare_response(sim_payload(selected, weights))
            st.dataframe(pd.DataFrame(comparison["rows"], columns=comparison["columns"]), width="stretch")
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))

st.subheader("Downloads")
summary_rows = [[key, value] for key, value in summary.items()]
st.download_button("Risk summary (CSV)", pd.DataFrame(summary_rows, columns=["metric", "value"]).to_csv(index=False), "risk_summary.csv", "text/csv")
st.download_button("Diagnostics (CSV)", diagnostics.to_csv(index=False), "calibration_diagnostics.csv", "text/csv")
if st.session_state.get("last_sim_payload"):
    st.download_button(
        "Wealth paths (CSV)",
        api.build_wealth_csv(st.session_state["last_sim_payload"])["csv"],
        "wealth_paths.csv",
        "text/csv",
    )
st.download_button(
    "Results (JSON)",
    json.dumps(results, indent=2),
    "results.json",
    "application/json",
)
