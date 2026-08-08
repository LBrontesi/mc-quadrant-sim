"""Gradio frontend for the Four-Quadrant Monte Carlo Simulator.

All data loading, scenario building, and result shaping is delegated to
``mc_quadrants.api`` so the simulation methodology is identical across
frontends. Run with:

    python gradio_app.py
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import date

import gradio as gr
import plotly.graph_objects as go

from mc_quadrants import api

REGIME_NAMES = api.REGIME_NAMES
ORDER_NAMES = [api.REGIME_NAMES[state] for state in api.REGIME_ORDER]
REGIME_COLORS = {
    "High growth / low inflation": "#2f855a",
    "High growth / high inflation": "#d97706",
    "Low growth / high inflation": "#c2410c",
    "Low growth / low inflation": "#3b82f6",
}
DISTRIBUTION_LABELS = {
    "normal": "Normal",
    "student_t": "Student-t",
    "bootstrap": "Historical bootstrap",
    "block_bootstrap": "Block bootstrap",
}

METRIC_KEYS = [
    ("mean", "Mean terminal wealth"),
    ("p05", "P05"),
    ("p50", "Median"),
    ("p95", "P95"),
    ("annualized_return", "Annualized return (wealth)"),
    ("annualized_volatility", "Annualized volatility (wealth)"),
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

TEMP_DIR = tempfile.mkdtemp(prefix="mcq-gradio-")


def pct(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}%"


def _download_path(prefix: str, suffix: str) -> str:
    descriptor, path = tempfile.mkstemp(prefix=prefix, suffix=suffix, dir=TEMP_DIR)
    os.close(descriptor)
    return path


# ---------- Payload builders ----------


def load_payload(
    source: str,
    demo_seed: int,
    yahoo_tickers: str,
    yahoo_start: str,
    yahoo_end: str,
    yahoo_proxies: str,
    synthetic: list[str],
    synthetic_seed: int,
    csv_prices: str | None,
    csv_macro: str | None,
    asset_input: str,
    monthly: bool,
    growth_col: str,
    inflation_col: str,
) -> dict:
    payload: dict = {"source": source}
    if source == "demo":
        payload["seed"] = int(demo_seed)
    elif source == "yahoo":
        payload["tickers"] = yahoo_tickers
        payload["start"] = str(yahoo_start)
        payload["end"] = str(yahoo_end)
        payload["proxies"] = yahoo_proxies
        payload["synthetic"] = synthetic or []
        payload["synthetic_seed"] = int(synthetic_seed)
    else:
        payload["csv_prices"] = csv_prices
        payload["csv_macro"] = csv_macro
        payload["asset_input"] = asset_input
        payload["monthly"] = monthly
        payload["growth_col"] = growth_col
        payload["inflation_col"] = inflation_col
    return payload


def scenario_payload(
    periods: int,
    paths: int,
    seed: int,
    start_state: str,
    distribution: str,
    degrees_of_freedom: int,
    block_size: int,
    rebalance: str,
    cost_bps: int,
    contribution: float,
    withdrawal: float,
    risk_free_rate: float,
    annual_inflation: float,
    base_currency: str,
    currency_map: str,
    use_corr_override: bool,
    corr_blend: float,
    corr_targets: dict[str, float],
    growth_threshold: str,
    inflation_threshold: str,
    macro_lag: int,
    transition_uncertainty: float,
    model_kind: str = "quadrant",
    hmm_states: int = 4,
    threshold_window: int = 0,
    duration_model: str = "semi_markov",
    garch: bool = False,
    walk_forward: bool = True,
    expense_ratios: str = "",
    leverage_multiple: float = 1.0,
    financing_rate: float = 0.0,
    maintenance_margin: float = 0.0,
) -> dict:
    return {
        "growth_threshold": growth_threshold,
        "inflation_threshold": inflation_threshold,
        "macro_lag": int(macro_lag),
        "transition_uncertainty": float(transition_uncertainty),
        "periods": int(periods),
        "paths": int(paths),
        "seed": int(seed),
        "start_state": start_state,
        "distribution": distribution,
        "degrees_of_freedom": int(degrees_of_freedom),
        "block_size": int(block_size),
        "rebalance": rebalance,
        "cost_bps": 0 if rebalance == "legacy" else int(cost_bps),
        "contribution": float(contribution),
        "withdrawal": float(withdrawal),
        "risk_free_rate": float(risk_free_rate),
        "annual_inflation": float(annual_inflation),
        "base_currency": base_currency.strip().upper(),
        "currency_map": currency_map,
        "use_correlation_override": bool(use_corr_override),
        "correlation_blend": float(corr_blend),
        "correlation_override_targets": corr_targets,
        "model": model_kind,
        "hmm_states": int(hmm_states),
        "threshold_window": int(threshold_window or 0),
        "duration_model": duration_model,
        "garch": bool(garch),
        "walk_forward": bool(walk_forward),
        "expense_ratios": expense_ratios,
        "leverage_multiple": float(leverage_multiple),
        "financing_rate": float(financing_rate),
        "maintenance_margin": float(maintenance_margin),
    }


def sim_payload(
    state: dict,
    selected_tickers: list[str],
    weights: dict[str, float],
    periods: int,
    paths: int,
    seed: int,
    start_state: str,
    distribution: str,
    degrees_of_freedom: int,
    block_size: int,
    rebalance: str,
    cost_bps: int,
    contribution: float,
    withdrawal: float,
    risk_free_rate: float,
    annual_inflation: float,
    base_currency: str,
    currency_map: str,
    use_corr_override: bool,
    corr_blend: float,
    corr_targets: dict[str, float],
    growth_threshold: str,
    inflation_threshold: str,
    macro_lag: int,
    transition_uncertainty: float,
    model_kind: str = "quadrant",
    hmm_states: int = 4,
    threshold_window: int = 0,
    duration_model: str = "semi_markov",
    garch: bool = False,
    walk_forward: bool = True,
    expense_ratios: str = "",
    leverage_multiple: float = 1.0,
    financing_rate: float = 0.0,
    maintenance_margin: float = 0.0,
) -> dict:
    payload = load_payload(
        state["source"],
        state["demo_seed"],
        state["yahoo_tickers"],
        state["yahoo_start"],
        state["yahoo_end"],
        state["yahoo_proxies"],
        state["yahoo_synthetic"],
        state["synthetic_seed"],
        state["csv_prices"],
        state["csv_macro"],
        state["asset_input"],
        state["csv_monthly"],
        state["csv_growth"],
        state["csv_inflation"],
    )
    payload.update(
        scenario_payload(
            periods,
            paths,
            seed,
            start_state,
            distribution,
            degrees_of_freedom,
            block_size,
            rebalance,
            cost_bps,
            contribution,
            withdrawal,
            risk_free_rate,
            annual_inflation,
            base_currency,
            currency_map,
            use_corr_override,
            corr_blend,
            corr_targets,
            growth_threshold,
            inflation_threshold,
            macro_lag,
            transition_uncertainty,
            model_kind,
            hmm_states,
            threshold_window,
            duration_model,
            garch,
            walk_forward,
            expense_ratios,
            leverage_multiple,
            financing_rate,
            maintenance_margin,
        )
    )
    payload["selected_tickers"] = selected_tickers
    payload["weights"] = weights
    return payload


# ---------- Plotly charts ----------


def wealth_chart(results: dict) -> go.Figure:
    figure = go.Figure()
    wealth = results["wealth"]
    for key, name, color in (
        ("p05", "P05", "#f97316"),
        ("median", "Median", "#3b82f6"),
        ("p95", "P95", "#10b981"),
    ):
        figure.add_trace(
            go.Scatter(
                x=wealth["periods"], y=wealth[key], mode="lines", name=name, line=dict(width=2.5, color=color)
            )
        )
    figure.update_layout(
        title="Wealth Percentiles",
        xaxis_title="Period",
        yaxis_title="Wealth",
        height=300,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return figure


def histogram(values: list[float], title: str, x_title: str, color: str) -> go.Figure:
    figure = go.Figure(go.Histogram(x=values, nbinsx=45, opacity=0.82, marker_color=color))
    figure.update_layout(
        title=title, xaxis_title=x_title, yaxis_title="Paths", height=300, margin=dict(l=40, r=20, t=40, b=40)
    )
    return figure


def regime_mix_chart(results: dict) -> go.Figure:
    items = results["regime_mix"]
    figure = go.Figure(
        go.Bar(
            x=[item["label"] for item in items],
            y=[item["share"] for item in items],
            marker_color=[REGIME_COLORS.get(item["label"], "#64748b") for item in items],
        )
    )
    figure.update_layout(
        title="Simulated Regime Mix", yaxis_title="Share", height=300, margin=dict(l=40, r=20, t=40, b=40)
    )
    return figure


def macro_scatter_chart(results: dict) -> go.Figure:
    points = results["macro_scatter"]
    figure = go.Figure()
    names = list(dict.fromkeys(point["regime"] for point in points))
    for name in names:
        subset = [p for p in points if p["regime"] == name]
        if subset:
            figure.add_trace(
                go.Scatter(
                    x=[p["growth"] for p in subset],
                    y=[p["inflation"] for p in subset],
                    mode="markers",
                    name=name,
                    marker=dict(color=REGIME_COLORS.get(name, "#64748b"), size=7, opacity=0.75),
                    text=[p["date"] for p in subset],
                )
            )
    figure.update_layout(
        title="Macro Quadrants (historical)",
        xaxis_title="Growth",
        yaxis_title="Inflation",
        height=300,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return figure


def matrix_heatmap(
    labels: list[str], values: list[list[float]], title: str, domain: list[float], scheme: str
) -> go.Figure:
    figure = go.Figure(
        go.Heatmap(
            x=labels,
            y=labels,
            z=values,
            colorscale=scheme,
            zmin=domain[0],
            zmax=domain[1],
            text=[[f"{value:.3f}" for value in row] for row in values],
            texttemplate="%{text}",
        )
    )
    figure.update_layout(
        title=title, height=360, margin=dict(l=40, r=20, t=40, b=40), yaxis=dict(autorange="reversed")
    )
    return figure


# ---------- Handlers ----------


def on_load(
    source: str,
    demo_seed: int,
    yahoo_tickers: str,
    yahoo_start: str,
    yahoo_end: str,
    yahoo_proxies: str,
    synthetic: list[str],
    synthetic_seed: int,
    csv_prices: bytes | None,
    csv_macro: bytes | None,
    asset_input: str,
    monthly: bool,
    growth_col: str,
    inflation_col: str,
) -> tuple[str, gr.Checkboxgroup, gr.Dataframe, gr.Dropdown, dict]:
    csv_prices_text = csv_prices.decode("utf-8") if csv_prices else None
    csv_macro_text = csv_macro.decode("utf-8") if csv_macro else None
    payload = load_payload(
        source,
        demo_seed,
        yahoo_tickers,
        yahoo_start,
        yahoo_end,
        yahoo_proxies,
        synthetic,
        synthetic_seed,
        csv_prices_text,
        csv_macro_text,
        asset_input,
        monthly,
        growth_col,
        inflation_col,
    )
    state = {
        **payload,
        "demo_seed": int(demo_seed),
        "yahoo_tickers": yahoo_tickers,
        "yahoo_start": str(yahoo_start),
        "yahoo_end": str(yahoo_end),
        "yahoo_proxies": yahoo_proxies,
        "yahoo_synthetic": synthetic or [],
        "synthetic_seed": int(synthetic_seed),
        "csv_prices": csv_prices_text,
        "csv_macro": csv_macro_text,
        "asset_input": asset_input,
        "csv_monthly": monthly,
        "csv_growth": growth_col,
        "csv_inflation": inflation_col,
    }
    try:
        load = api.build_load_response(*api.load_data_source(payload)[:5], "")
    except Exception as exc:  # noqa: BLE001
        return f"Error: {exc}", gr.update(), gr.update(), gr.update(), state
    tickers = load["tickers"]
    defaults = load["default_tickers"]
    presets = {preset["name"]: preset["weights"] for preset in load["presets"]}
    weights = [[ticker, api.default_weights(ticker)] for ticker in defaults]
    state["source"] = source
    state["presets"] = load["presets"]
    coverage = load["coverage"]
    coverage_dates = [
        date_value
        for item in coverage.values()
        for date_value in (item.get("first"), item.get("last"))
        if date_value
    ]
    coverage_text = (
        f"{min(coverage_dates)} to {max(coverage_dates)}"
        if coverage_dates
        else "no dated return observations"
    )
    message = (
        f"Loaded {len(tickers)} tickers. Select tickers and set weights.\n\n"
        f"**Return coverage:** {coverage_text} · **Macro preview:** {len(load['macro']['rows'])} rows"
    )
    return (
        message,
        gr.update(choices=tickers, value=defaults),
        gr.update(value=weights),
        gr.update(choices=list(presets)),
        state,
    )


def apply_preset(preset_name: str, state: dict, tickers: list[str]) -> tuple[gr.Dataframe, str]:
    if not preset_name or not state:
        return gr.update(), "Choose a preset first."
    presets = {preset["name"]: preset["weights"] for preset in state.get("presets", [])}
    if preset_name not in presets:
        return gr.update(), "Preset not available."
    preset = presets[preset_name]
    matched = {ticker: weight for ticker, weight in preset.items() if ticker in tickers}
    total = sum(matched.values())
    if total <= 0:
        return gr.update(), "Preset assets are not selected. Select matching tickers first."
    factor = 100.0 / total
    rows = [[ticker, matched.get(ticker, 0.0) * factor] for ticker in tickers]
    return gr.update(value=rows), f"Applied {preset_name}."


def on_run(
    state: dict,
    tickers: list[str],
    weights_table: list[list],
    periods: int,
    paths: int,
    seed: int,
    start_state: str,
    distribution: str,
    degrees_of_freedom: int,
    block_size: int,
    rebalance: str,
    cost_bps: int,
    contribution: float,
    withdrawal: float,
    risk_free_rate: float,
    annual_inflation: float,
    base_currency: str,
    currency_map: str,
    use_corr_override: bool,
    corr_blend: float,
    corr_growth_low: float,
    corr_growth_high: float,
    corr_stagflation: float,
    corr_recession: float,
    growth_threshold: str,
    inflation_threshold: str,
    macro_lag: int,
    transition_uncertainty: float,
    model_kind: str = "quadrant",
    hmm_states: int = 4,
    threshold_window: int = 0,
    duration_model: str = "semi_markov",
    garch: bool = False,
    walk_forward: bool = True,
    expense_ratios: str = "",
    leverage_multiple: float = 1.0,
    financing_rate: float = 0.0,
    maintenance_margin: float = 0.0,
) -> tuple[
    gr.Markdown,
    gr.Markdown,
    go.Figure,
    go.Figure,
    go.Figure,
    go.Figure,
    go.Figure,
    go.Figure,
    gr.Dataframe,
    gr.Dataframe,
    dict,
]:
    selected = list(tickers) if tickers else []
    weights: dict[str, float] = {}
    for row in weights_table or []:
        if row and len(row) >= 2:
            weights[str(row[0]).strip().upper()] = float(row[1] or 0)
    if not selected:
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            {},
        )
    corr_targets = {
        "high_growth_low_inflation": float(corr_growth_low),
        "high_growth_high_inflation": float(corr_growth_high),
        "low_growth_high_inflation": float(corr_stagflation),
        "low_growth_low_inflation": float(corr_recession),
    }
    payload = sim_payload(
        state,
        selected,
        weights,
        periods,
        paths,
        seed,
        start_state,
        distribution,
        degrees_of_freedom,
        block_size,
        rebalance,
        cost_bps,
        contribution,
        withdrawal,
        risk_free_rate,
        annual_inflation,
        base_currency,
        currency_map,
        use_corr_override,
        corr_blend,
        corr_targets,
        growth_threshold,
        inflation_threshold,
        macro_lag,
        transition_uncertainty,
        model_kind,
        hmm_states,
        threshold_window,
        duration_model,
        garch,
        walk_forward,
        expense_ratios,
        leverage_multiple,
        financing_rate,
        maintenance_margin,
    )
    try:
        results = api.build_simulate_response(payload)
    except Exception as exc:  # noqa: BLE001
        return (
            gr.update(value=f"Error: {exc}"),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            {},
        )

    summary = results["summary"]
    lines = [
        f"**{label}:** {api.format_metric_value(key, summary.get(key), results['currency'])}"
        for key, label in METRIC_KEYS
    ]
    if summary.get("periodic_contribution", 0) or summary.get("periodic_withdrawal", 0):
        lines.extend(
            [
                "",
                "**Investment performance excluding external cash flows**",
                (
                    "**Time-weighted annualized return:** "
                    + api.format_metric_value(
                        "cash_flow_adjusted_annualized_return",
                        summary.get("cash_flow_adjusted_annualized_return"),
                        results["currency"],
                    )
                ),
                (
                    "**Time-weighted volatility:** "
                    + api.format_metric_value(
                        "cash_flow_adjusted_volatility",
                        summary.get("cash_flow_adjusted_volatility"),
                        results["currency"],
                    )
                ),
                (
                    "**Time-weighted Sharpe:** "
                    + api.format_metric_value(
                        "cash_flow_adjusted_sharpe_ratio",
                        summary.get("cash_flow_adjusted_sharpe_ratio"),
                        results["currency"],
                    )
                ),
            ]
        )
    costs = results.get("costs", {})
    if costs.get("leverage_multiple", 1.0) > 1.0 or costs.get("weighted_expense_ratio", 0.0) > 0:
        lines.extend(
            [
                "",
                "**Cost and leverage assumptions**",
                f"**Leverage:** {costs.get('leverage_multiple', 1.0):.1f}x",
                f"**Weighted ETF fee:** {api.format_metric_value('weighted_expense_ratio', costs.get('weighted_expense_ratio'), results['currency'])}",
                f"**Annual fee drag:** {api.format_metric_value('annual_fee_drag', costs.get('annual_fee_drag'), results['currency'])}",
                f"**Annual financing cost:** {api.format_metric_value('annual_financing_cost', costs.get('annual_financing_cost'), results['currency'])}",
                f"**Margin calls:** {costs.get('margin_calls', 0)}",
            ]
        )
    metrics = "\n\n".join(lines)
    caption = (
        f"{'Real (inflation-adjusted)' if results['terms'] == 'real' else 'Nominal'} · "
        f"Currency: {results['currency']} · "
        f"VaR (95%): {api.format_metric_value('var_95', summary['var_95'], results['currency'])} · "
        f"Worst max drawdown: {api.format_metric_value('max_drawdown_worst', summary['max_drawdown_worst'])}"
    )
    validation = results.get("validation")
    if validation:
        validation_summary = validation["summary"]
        caption += (
            f"\n\n**Out-of-sample validation:** regime model "
            f"{'+' if validation_summary['advantage_mean'] > 0 else ''}"
            f"{validation_summary['advantage_mean']:.4f} log-likelihood units/period vs "
            f"unconditional benchmark · one-step regime hit rate "
            f"{pct(validation_summary['regime_hit_rate'], 0)}"
        )
    if results["warnings"]:
        caption += "\n\nWarnings:\n" + "\n".join(results["warnings"])
    transition = results["transition"]
    diagnostics = gr.Dataframe(
        value=results["diagnostics"]["rows"],
        headers=results["diagnostics"]["columns"],
        label="Calibration Diagnostics",
    )
    return (
        gr.update(value=f"**{results['message']}**\n\n{caption}"),
        gr.update(value=metrics),
        wealth_chart(results),
        histogram(results["terminal"], "Terminal Wealth Distribution", "Terminal wealth", "#3b82f6"),
        histogram(results["drawdowns"], "Maximum Drawdown Distribution", "Maximum drawdown", "#f97316"),
        regime_mix_chart(results),
        matrix_heatmap(transition["labels"], transition["values"], "Transition Matrix", [0, 1], "Blues"),
        macro_scatter_chart(results),
        diagnostics,
        gr.update(),
        results,
    )


def correlation_chart(results: dict | None, regime_label: str) -> go.Figure:
    if not results:
        return go.Figure()
    corr = results["correlations"].get(regime_label)
    if not corr:
        return go.Figure()
    return matrix_heatmap(corr["labels"], corr["values"], regime_label, [-1, 1], "RdBu")


def on_compare(
    state: dict,
    tickers: list[str],
    weights_table: list[list],
    periods: int,
    paths: int,
    seed: int,
    start_state: str,
    distribution: str,
    degrees_of_freedom: int,
    block_size: int,
    rebalance: str,
    cost_bps: int,
    contribution: float,
    withdrawal: float,
    risk_free_rate: float,
    annual_inflation: float,
    base_currency: str,
    currency_map: str,
    use_corr_override: bool,
    corr_blend: float,
    corr_growth_low: float,
    corr_growth_high: float,
    corr_stagflation: float,
    corr_recession: float,
    growth_threshold: str,
    inflation_threshold: str,
    macro_lag: int,
    transition_uncertainty: float,
    model_kind: str = "quadrant",
    hmm_states: int = 4,
    threshold_window: int = 0,
    duration_model: str = "semi_markov",
    garch: bool = False,
    walk_forward: bool = True,
    expense_ratios: str = "",
    leverage_multiple: float = 1.0,
    financing_rate: float = 0.0,
    maintenance_margin: float = 0.0,
) -> gr.Dataframe:
    selected = list(tickers) if tickers else []
    weights: dict[str, float] = {}
    for row in weights_table or []:
        if row and len(row) >= 2:
            weights[str(row[0]).strip().upper()] = float(row[1] or 0)
    corr_targets = {
        "high_growth_low_inflation": float(corr_growth_low),
        "high_growth_high_inflation": float(corr_growth_high),
        "low_growth_high_inflation": float(corr_stagflation),
        "low_growth_low_inflation": float(corr_recession),
    }
    payload = sim_payload(
        state,
        selected,
        weights,
        periods,
        paths,
        seed,
        start_state,
        distribution,
        degrees_of_freedom,
        block_size,
        rebalance,
        cost_bps,
        contribution,
        withdrawal,
        risk_free_rate,
        annual_inflation,
        base_currency,
        currency_map,
        use_corr_override,
        corr_blend,
        corr_targets,
        growth_threshold,
        inflation_threshold,
        macro_lag,
        transition_uncertainty,
        model_kind,
        hmm_states,
        threshold_window,
        duration_model,
        garch,
        walk_forward,
        expense_ratios,
        leverage_multiple,
        financing_rate,
        maintenance_margin,
    )
    try:
        comparison = api.build_compare_response(payload)
    except Exception as exc:  # noqa: BLE001
        return gr.Dataframe(value=[[str(exc)]], headers=["error"])
    return gr.Dataframe(value=comparison["rows"], headers=comparison["columns"])


def download_wealth_paths(
    state: dict,
    tickers: list[str],
    weights_table: list[list],
    periods: int,
    paths: int,
    seed: int,
    start_state: str,
    distribution: str,
    degrees_of_freedom: int,
    block_size: int,
    rebalance: str,
    cost_bps: int,
    contribution: float,
    withdrawal: float,
    risk_free_rate: float,
    annual_inflation: float,
    base_currency: str,
    currency_map: str,
    use_corr_override: bool,
    corr_blend: float,
    corr_growth_low: float,
    corr_growth_high: float,
    corr_stagflation: float,
    corr_recession: float,
    growth_threshold: str,
    inflation_threshold: str,
    macro_lag: int,
    transition_uncertainty: float,
    model_kind: str = "quadrant",
    hmm_states: int = 4,
    threshold_window: int = 0,
    duration_model: str = "semi_markov",
    garch: bool = False,
    walk_forward: bool = True,
    expense_ratios: str = "",
    leverage_multiple: float = 1.0,
    financing_rate: float = 0.0,
    maintenance_margin: float = 0.0,
) -> str | None:
    if not state:
        return None
    weights = {str(row[0]).strip().upper(): float(row[1] or 0) for row in (weights_table or []) if row}
    corr_targets = {
        "high_growth_low_inflation": float(corr_growth_low),
        "high_growth_high_inflation": float(corr_growth_high),
        "low_growth_high_inflation": float(corr_stagflation),
        "low_growth_low_inflation": float(corr_recession),
    }
    payload = sim_payload(
        state,
        list(tickers) if tickers else [],
        weights,
        periods,
        paths,
        seed,
        start_state,
        distribution,
        degrees_of_freedom,
        block_size,
        rebalance,
        cost_bps,
        contribution,
        withdrawal,
        risk_free_rate,
        annual_inflation,
        base_currency,
        currency_map,
        use_corr_override,
        corr_blend,
        corr_targets,
        growth_threshold,
        inflation_threshold,
        macro_lag,
        transition_uncertainty,
        model_kind,
        hmm_states,
        threshold_window,
        duration_model,
        garch,
        walk_forward,
        expense_ratios,
        leverage_multiple,
        financing_rate,
        maintenance_margin,
    )
    csv = api.build_wealth_csv(payload)["csv"]
    path = _download_path("wealth_paths-", ".csv")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(csv)
    return path


def download_summary(results: dict | None) -> str | None:
    if not results:
        return None
    path = _download_path("risk_summary-", ".csv")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(f"{key},{value}" for key, value in results["summary"].items()))
    return path


def download_json(results: dict | None) -> str | None:
    if not results:
        return None
    path = _download_path("results-", ".json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    return path


# ---------- UI ----------

with gr.Blocks(title="Four-Quadrant Monte Carlo Simulator") as demo:
    config_state = gr.State({})
    results_state = gr.State(None)

    gr.Markdown("# Four-Quadrant Monte Carlo Simulator")
    gr.Markdown("Goldilocks · Overheating · Stagflation · Recession — regime-based portfolio simulation")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Data source")
            source = gr.Radio(["demo", "yahoo", "csv"], value="demo", label="Source")
            demo_seed = gr.Number(value=42, label="Demo seed", minimum=1, precision=0)
            yahoo_tickers = gr.Textbox(
                value="SPY, IEF, GLD, DBC, EFA, VNQ, TIP, SHY",
                label="Market tickers",
                visible=False,
            )
            with gr.Row(visible=False) as yahoo_dates:
                yahoo_start = gr.Textbox(value="1990-01-01", label="Start")
                yahoo_end = gr.Textbox(value=date.today().isoformat(), label="End")
            yahoo_proxies = gr.Textbox(
                label="Historical proxies (ASSET:PROXY)", placeholder="SPY:^GSPC, GLD:GC=F", visible=False
            )
            synthetic = gr.CheckboxGroup(
                api.SYNTHETIC_TICKER_OPTIONS, label="Synthetic backfill assets", visible=False
            )
            synthetic_seed = gr.Number(
                value=42, label="Synthetic history seed", minimum=1, precision=0, visible=False
            )
            csv_prices = gr.File(label="Asset CSV", type="binary", visible=False)
            csv_macro = gr.File(label="Macro CSV", type="binary", visible=False)
            asset_input = gr.Radio(
                ["Price levels", "Returns"], value="Price levels", label="Asset input", visible=False
            )
            csv_monthly = gr.Checkbox(value=True, label="Monthly asset returns", visible=False)
            csv_growth = gr.Textbox(value="growth", label="Growth column", visible=False)
            csv_inflation = gr.Textbox(value="inflation", label="Inflation column", visible=False)
            load_btn = gr.Button("Load Data", variant="primary")
            load_status = gr.Markdown("")

            gr.Markdown("### Portfolio")
            tickers = gr.CheckboxGroup([], label="Tickers")
            weights_table = gr.Dataframe(
                headers=["ticker", "weight"],
                datatype=["str", "number"],
                column_count=(2, "fixed"),
                label="Weights (editable)",
                interactive=True,
            )
            preset_dropdown = gr.Dropdown([], label="Portfolio preset")
            preset_btn = gr.Button("Apply preset")

            gr.Markdown("### Simulation")
            with gr.Row():
                periods = gr.Number(value=120, label="Periods (months)", minimum=12, maximum=360, precision=0)
                paths = gr.Number(value=3000, label="Paths", minimum=250, maximum=20000, precision=0)
            with gr.Row():
                seed = gr.Number(value=7, label="Random seed", precision=0, minimum=1)
                start_state = gr.Dropdown(
                    ["Stationary"] + ORDER_NAMES, value="Stationary", label="Start state"
                )
            with gr.Row():
                distribution = gr.Dropdown(
                    [(label, key) for key, label in DISTRIBUTION_LABELS.items()],
                    value="normal",
                    label="Return distribution",
                    info="Normal, Student-t, historical bootstrap, or block bootstrap",
                )
                degrees_of_freedom = gr.Number(
                    value=5, label="Student-t dof", minimum=3, maximum=30, precision=0
                )
            with gr.Row():
                block_size = gr.Number(value=3, label="Block size", minimum=2, maximum=12, precision=0)
                rebalance = gr.Dropdown(
                    ["monthly", "quarterly", "annual", "legacy"], value="monthly", label="Rebalancing"
                )
            with gr.Row():
                cost_bps = gr.Number(value=10, label="Cost (bps)", minimum=0, maximum=100, precision=0)
                risk_free_rate = gr.Number(value=0.0, label="Risk-free rate (annual %)")
            with gr.Row():
                contribution = gr.Number(
                    value=0.0,
                    label="Contribution / period",
                    minimum=0,
                    info="Currency units invested at the target allocation each period (DCA).",
                )
                withdrawal = gr.Number(
                    value=0.0,
                    label="Withdrawal / period",
                    minimum=0,
                    info="Currency units funded pro-rata from holdings each period.",
                )
            with gr.Row():
                annual_inflation = gr.Number(value=0.0, label="Inflation assumption (annual %)")
                base_currency = gr.Textbox(value="USD", label="Portfolio currency", max_length=3)
            currency_map = gr.Textbox(label="Asset currencies (ASSET:CURRENCY)", placeholder="EFA:EUR")
            expense_ratios = gr.Textbox(
                label="ETF expense ratios (ASSET:annual %)",
                placeholder="SPY:0.03, IEF:0.15",
                info="Forward fee drag. Historical ETF prices may already include historical fund fees.",
            )
            with gr.Row():
                leverage_multiple = gr.Number(
                    value=1.0,
                    label="Leverage multiple",
                    minimum=1.0,
                    maximum=5.0,
                    step=0.1,
                    info="1.0x is unlevered; leverage requires explicit rebalancing.",
                )
                financing_rate = gr.Number(
                    value=0.0,
                    label="Financing rate (annual %)",
                    minimum=0.0,
                    maximum=30.0,
                    step=0.1,
                )
            maintenance_margin = gr.Number(
                value=0.0,
                label="Maintenance margin (%)",
                minimum=0.0,
                maximum=90.0,
                step=1.0,
                info="Optional liquidation threshold based on equity/assets.",
            )
            with gr.Row():
                macro_lag = gr.Dropdown([0, 1, 2, 3], value=1, label="Macro release lag")
                transition_uncertainty = gr.Slider(
                    0.0, 1.0, value=0.0, step=0.05, label="Transition uncertainty"
                )
            with gr.Row():
                growth_threshold = gr.Dropdown(["median", "mean"], value="median", label="Growth threshold")
                inflation_threshold = gr.Dropdown(
                    ["median", "mean"], value="median", label="Inflation threshold"
                )

            gr.Markdown("### Model methodology")
            with gr.Row():
                model_kind = gr.Dropdown(
                    ["quadrant", "hmm"],
                    value="quadrant",
                    label="Regime model",
                    info="Quadrant: macro thresholds on growth/inflation. HMM: states learned from returns.",
                )
                hmm_states = gr.Number(value=4, label="HMM states", minimum=2, maximum=8, precision=0)
            with gr.Row():
                duration_model = gr.Dropdown(
                    ["markov", "semi_markov"],
                    value="semi_markov",
                    label="Regime duration",
                    info="Semi-Markov uses empirical sojourn lengths instead of geometric chain run times.",
                )
                threshold_window = gr.Number(
                    value=12,
                    label="Causal threshold window",
                    minimum=0,
                    precision=0,
                    info="Prior observations required before classification; 0 uses the full sample (look-ahead).",
                )
            with gr.Row():
                garch = gr.Checkbox(
                    value=False,
                    label="GARCH(1,1) volatility clustering",
                    info="Within-regime conditional variance dynamics (Normal distribution only).",
                )
                walk_forward = gr.Checkbox(
                    value=True,
                    label="Walk-forward validation",
                    info="Out-of-sample check of the regime model against an unconditional benchmark.",
                )

            gr.Markdown("### Correlation overrides")
            use_corr_override = gr.Checkbox(value=True, label="Blend custom correlation view")
            corr_blend = gr.Slider(0.0, 1.0, value=0.4, step=0.05, label="Blend")
            corr_growth_low = gr.Slider(
                -1.0, 1.0, value=-0.10, step=0.05, label="High growth / low inflation"
            )
            corr_growth_high = gr.Slider(
                -1.0, 1.0, value=0.35, step=0.05, label="High growth / high inflation"
            )
            corr_stagflation = gr.Slider(
                -1.0, 1.0, value=0.25, step=0.05, label="Low growth / high inflation"
            )
            corr_recession = gr.Slider(-1.0, 1.0, value=-0.40, step=0.05, label="Low growth / low inflation")

        with gr.Column(scale=2):
            run_btn = gr.Button("Run Simulation", variant="primary")
            run_status = gr.Markdown("")
            metrics = gr.Markdown("")
            with gr.Tabs():
                with gr.Tab("Wealth"):
                    wealth_plot = gr.Plot(label="Wealth Percentiles")
                    terminal_plot = gr.Plot(label="Terminal Wealth Distribution")
                with gr.Tab("Risk"):
                    drawdown_plot = gr.Plot(label="Maximum Drawdown Distribution")
                    regime_mix_plot = gr.Plot(label="Simulated Regime Mix")
                with gr.Tab("Model"):
                    transition_plot = gr.Plot(label="Transition Matrix")
                    macro_plot = gr.Plot(label="Macro Quadrants")
                with gr.Tab("Correlations"):
                    with gr.Row():
                        correlation_regime = gr.Dropdown([], label="Regime")
                        correlation_plot = gr.Plot(label="Regime-Specific Correlation Matrix")
                with gr.Tab("Diagnostics"):
                    diagnostics_table = gr.Dataframe(label="Calibration Diagnostics")
                    compare_btn = gr.Button("Compare Normal vs Student-t")
                    comparison_table = gr.Dataframe(label="Scenario Comparison")
                with gr.Tab("Downloads"):
                    download_summary_btn = gr.DownloadButton("Download risk summary (CSV)")
                    download_json_btn = gr.DownloadButton("Download results (JSON)")
                    download_wealth_btn = gr.DownloadButton("Download wealth paths (CSV)")

    def toggle_source(source_value: str) -> tuple:
        show = source_value == "yahoo"
        csv_show = source_value == "csv"
        return (
            gr.update(visible=show),
            gr.update(visible=show),
            gr.update(visible=show),
            gr.update(visible=show),
            gr.update(visible=show),
            gr.update(visible=show),
            gr.update(visible=csv_show),
            gr.update(visible=csv_show),
            gr.update(visible=csv_show),
            gr.update(visible=csv_show),
            gr.update(visible=csv_show),
            gr.update(visible=csv_show),
        )

    source.change(
        toggle_source,
        source,
        [
            yahoo_tickers,
            yahoo_dates,
            yahoo_proxies,
            synthetic,
            synthetic_seed,
            yahoo_start,
            csv_prices,
            csv_macro,
            asset_input,
            csv_monthly,
            csv_growth,
            csv_inflation,
        ],
    )

    def toggle_methodology_controls(model_value: str, distribution_value: str, rebalance_value: str) -> tuple:
        quadrant = model_value == "quadrant"
        normal = distribution_value == "normal"
        rebalancing = rebalance_value != "legacy"
        return (
            gr.update(visible=quadrant),  # start state
            gr.update(visible=not quadrant),  # HMM states
            gr.update(visible=quadrant),  # macro lag
            gr.update(visible=quadrant),  # growth threshold
            gr.update(visible=quadrant),  # inflation threshold
            gr.update(visible=quadrant),  # causal threshold window
            gr.update(visible=quadrant),  # walk-forward validation
            gr.update(visible=quadrant),  # correlation override toggle
            gr.update(visible=quadrant),  # correlation blend
            gr.update(visible=quadrant),  # growth/low inflation target
            gr.update(visible=quadrant),  # growth/high inflation target
            gr.update(visible=quadrant),  # stagflation target
            gr.update(visible=quadrant),  # recession target
            gr.update(interactive=normal),  # GARCH
            gr.update(interactive=rebalancing),  # transaction cost
        )

    methodology_inputs = [model_kind, distribution, rebalance]
    methodology_outputs = [
        start_state,
        hmm_states,
        macro_lag,
        growth_threshold,
        inflation_threshold,
        threshold_window,
        walk_forward,
        use_corr_override,
        corr_blend,
        corr_growth_low,
        corr_growth_high,
        corr_stagflation,
        corr_recession,
        garch,
        cost_bps,
    ]
    model_kind.change(toggle_methodology_controls, methodology_inputs, methodology_outputs)
    distribution.change(toggle_methodology_controls, methodology_inputs, methodology_outputs)
    rebalance.change(toggle_methodology_controls, methodology_inputs, methodology_outputs)

    load_btn.click(
        on_load,
        [
            source,
            demo_seed,
            yahoo_tickers,
            yahoo_start,
            yahoo_end,
            yahoo_proxies,
            synthetic,
            synthetic_seed,
            csv_prices,
            csv_macro,
            asset_input,
            csv_monthly,
            csv_growth,
            csv_inflation,
        ],
        [load_status, tickers, weights_table, preset_dropdown, config_state],
    )
    preset_btn.click(apply_preset, [preset_dropdown, config_state, tickers], [weights_table, load_status])

    run_inputs = [
        config_state,
        tickers,
        weights_table,
        periods,
        paths,
        seed,
        start_state,
        distribution,
        degrees_of_freedom,
        block_size,
        rebalance,
        cost_bps,
        contribution,
        withdrawal,
        risk_free_rate,
        annual_inflation,
        base_currency,
        currency_map,
        use_corr_override,
        corr_blend,
        corr_growth_low,
        corr_growth_high,
        corr_stagflation,
        corr_recession,
        growth_threshold,
        inflation_threshold,
        macro_lag,
        transition_uncertainty,
        model_kind,
        hmm_states,
        threshold_window,
        duration_model,
        garch,
        walk_forward,
        expense_ratios,
        leverage_multiple,
        financing_rate,
        maintenance_margin,
    ]
    run_outputs = [
        run_status,
        metrics,
        wealth_plot,
        terminal_plot,
        drawdown_plot,
        regime_mix_plot,
        transition_plot,
        macro_plot,
        diagnostics_table,
        comparison_table,
        results_state,
    ]
    run_btn.click(on_run, run_inputs, run_outputs)
    compare_btn.click(on_compare, run_inputs, comparison_table)

    def update_correlation_plot(results: dict | None, regime_label: str | None) -> go.Figure:
        return correlation_chart(results, regime_label or ORDER_NAMES[0])

    def update_regime_choices(results: dict | None) -> gr.Dropdown:
        if not results:
            return gr.update(choices=[])
        return gr.update(
            choices=list(results["correlations"].keys()), value=list(results["correlations"].keys())[0]
        )

    results_state.change(update_regime_choices, results_state, correlation_regime)
    correlation_regime.change(update_correlation_plot, [results_state, correlation_regime], correlation_plot)

    download_wealth_btn.click(
        download_wealth_paths,
        run_inputs,
        download_wealth_btn,
    )
    download_summary_btn.click(download_summary, results_state, download_summary_btn)
    download_json_btn.click(download_json, results_state, download_json_btn)


def main() -> None:
    demo.queue().launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))


if __name__ == "__main__":
    main()
