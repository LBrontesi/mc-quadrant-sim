from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from multiprocessing import get_context
from typing import Any

import numpy as np
import pandas as pd

from mc_quadrants.calibration import CorrelationOverrides, calibrate_quadrant_model
from mc_quadrants.data import convert_returns_to_base_currency
from mc_quadrants.decumulation import DecumulationPlan, normalize_decumulation
from mc_quadrants.diagnostics import (
    CalibrationDiagnostics,
    build_calibration_diagnostics,
    build_hmm_diagnostics,
)
from mc_quadrants.hmm import fit_hmm_model
from mc_quadrants.native import native_available
from mc_quadrants.regimes import classify_persistent_quadrants
from mc_quadrants.simulation import (
    DEFAULT_LIQUIDITY_COST_MULTIPLIERS,
    inflation_adjust_wealth,
    simulate_portfolio_paths,
    simulate_returns,
    summarize_wealth_risk,
)
from mc_quadrants.tax_policy import resolve_tax_selection
from mc_quadrants.taxes import (
    italian_native_result_frame,
    normalize_italy_tax_metadata,
    prepare_italian_native_configuration,
)
from mc_quadrants.types import ScenarioModel, SimulationResult
from mc_quadrants.uncertainty import bootstrap_quadrant_models, summarize_parameter_models
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
    reporting_wealth: pd.DataFrame | None = None
    gross_wealth: pd.DataFrame | None = None
    gross_reporting_wealth: pd.DataFrame | None = None
    parameter_uncertainty: pd.DataFrame | None = None


_CHUNK_WORKER_STATE: dict[str, Any] = {}
_ADDITIVE_WEALTH_ATTRS = (
    "margin_calls",
    "capital_gains_tax_total",
    "wealth_tax_total",
    "terminal_liquidation_tax_total",
    "taxes_paid_total",
    "realized_gains_total",
    "realized_losses_total",
    "loss_carryforward_total",
    "investment_income_tax_total",
    "foreign_withholding_tax_total",
    "financial_transaction_tax_total",
    "stamp_duty_total",
    "ivafe_total",
    "expired_losses_total",
    "transaction_cost_total",
)
_PATH_WEALTH_ATTRS = (
    "withdrawal_requested",
    "withdrawal_funded",
    "guardrail_events",
    "withdrawal_cpi",
    "paired_wealth",
    "paired_withdrawal_requested",
    "paired_withdrawal_funded",
    "paired_guardrail_events",
)


def _alternative_decumulation(plan: DecumulationPlan) -> DecumulationPlan | None:
    if not plan.active or plan.legacy_nominal:
        return None
    alternative = "fixed" if plan.policy == "guyton_klinger" else "guyton_klinger"
    return replace(plan, guardrails=replace(plan.guardrails, policy=alternative))


def _annual_macro_paths(
    model: ScenarioModel,
    result: SimulationResult,
    column: str | None,
    percent_flag: str,
) -> np.ndarray | None:
    """Extract one simulated annual macro series as decimal rates."""

    if result.macro_paths is None or not column or column not in result.macro_columns:
        return None
    index = result.macro_columns.index(column)
    values = result.macro_paths[:, :, index].astype(float, copy=True)
    dynamics = model.metadata.get("macro_dynamics", {})
    if bool(dynamics.get(percent_flag, model.metadata.get(percent_flag, False))):
        values /= 100.0
    return values


def _init_chunk_worker(state: dict[str, Any]) -> None:
    """Initialize a worker process with the shared simulation state."""
    _CHUNK_WORKER_STATE.update(state)


def _merge_tax_by_year(
    target: dict[str, dict[str, float]],
    source: Mapping[str, Mapping[str, Any]] | None,
) -> None:
    for year, metrics in (source or {}).items():
        bucket = target.setdefault(str(year), {})
        for name, value in metrics.items():
            bucket[str(name)] = bucket.get(str(name), 0.0) + float(value)


def _cash_flow_adjusted_geometric_returns(
    wealth: pd.DataFrame,
    *,
    initial_value: float,
    contribution: float,
    withdrawal: float,
    withdrawal_start_period: int,
    annual_inflation: float,
    inflation_paths: np.ndarray | None,
    withdrawal_paths: np.ndarray | None = None,
) -> np.ndarray:
    """Return one annualized, cash-flow-adjusted geometric return per path."""

    values = wealth.to_numpy(dtype=float)
    periods, paths = values.shape
    if inflation_paths is None:
        steps = np.arange(1, periods + 1, dtype=float)
        deflator = ((1.0 + annual_inflation) ** (-steps / 12.0))[:, None]
    else:
        annual = np.asarray(inflation_paths, dtype=float)
        periodic = np.power(1.0 + annual, 1.0 / 12.0)
        deflator = 1.0 / np.cumprod(periodic, axis=0)
    real_values = values * deflator
    previous = np.vstack([np.full(paths, initial_value), real_values[:-1]])
    contribution_deflator = np.vstack([np.ones((1, deflator.shape[1])), deflator[:-1]])
    denominator = previous + contribution * contribution_deflator
    if withdrawal_paths is None:
        withdrawal_schedule = (
            np.arange(1, periods + 1, dtype=int) >= withdrawal_start_period
        ).astype(float)[:, None]
        nominal_withdrawals = withdrawal * withdrawal_schedule
    else:
        nominal_withdrawals = np.asarray(withdrawal_paths, dtype=float)
        if nominal_withdrawals.shape != (periods, paths):
            raise ValueError("withdrawal_paths must match the wealth path matrix.")
    numerator = real_values + nominal_withdrawals * deflator
    with np.errstate(divide="ignore", invalid="ignore"):
        returns = numerator / denominator - 1.0
    valid = np.isfinite(returns) & (returns > -1.0)
    log_values = np.where(valid, np.log1p(np.where(valid, returns, 0.0)), 0.0)
    counts = valid.sum(axis=0)
    annualized = np.zeros(paths, dtype=float)
    nonzero = counts > 0
    annualized[nonzero] = np.exp(log_values[:, nonzero].sum(axis=0) / counts[nonzero] * 12.0) - 1.0
    return annualized


def _native_tax_compatible(
    enabled: bool,
    metadata: Mapping[str, Mapping[str, object]] | None,
) -> bool:
    """Return whether the accumulating/total-return native ledger can be used."""

    return bool(
        enabled
        and native_available()
        and all(
            np.isclose(float(values.get("annual_income_yield", 0.0)), 0.0)
            for values in (metadata or {}).values()
        )
    )


def _run_chunk(
    args: tuple[int, int, int],
) -> tuple[
    int,
    np.ndarray,
    np.ndarray | None,
    np.ndarray,
    np.ndarray | None,
    dict[str, float],
    dict[str, float],
    dict[str, dict[str, float]],
    np.ndarray | None,
    np.ndarray | None,
    dict[str, np.ndarray],
]:
    """Simulate one path chunk in a worker process.

    Returns the start offset, active and gross wealth values, regime codes,
    macro paths, and additive accounting attributes. The same market paths
    feed both ledgers, so any difference is solely caused by the selected tax
    policy. Each chunk draws its own RNG from ``random_seed + chunk_index``.
    """
    start, count, seed = args
    state = _CHUNK_WORKER_STATE
    chunk_result = simulate_returns(
        state["model"],
        periods=state["periods"],
        paths=count,
        start_state=state["start_state"],
        random_seed=seed,
        distribution=state["distribution"],
        transition_concentration=state["transition_concentration"],
        duration_model=state["duration_model"],
        min_regime_duration=state["min_regime_duration"],
        garch=state["garch"],
        garch_alpha=state["garch_alpha"],
        garch_beta=state["garch_beta"],
        joint_macro=state["joint_macro"],
        macro_transition_weight=state["macro_transition_weight"],
        macro_parameter_uncertainty=state["macro_parameter_uncertainty"],
        dynamic_correlation=state["dynamic_correlation"],
        dcc_alpha=state["dcc_alpha"],
        dcc_beta=state["dcc_beta"],
        dcc_asymmetry=state["dcc_asymmetry"],
        return_regime_codes=True,
        native_threads=state.get("native_threads", 1),
    )
    financing_rate_paths = _annual_macro_paths(
        state["model"],
        chunk_result,
        state["rate_col"],
        "rate_is_percent",
    )
    financing_inflation_paths = _annual_macro_paths(
        state["model"],
        chunk_result,
        state["inflation_col"],
        "inflation_is_percent",
    )
    portfolio_kwargs = {
        "weights": state["weights"],
        "initial_value": state["initial_value"],
        "return_kind": state["return_kind"],
        "rebalance_frequency": state["rebalance_frequency"],
        "transaction_cost_bps": state["transaction_cost_bps"],
        "state_transaction_cost_multipliers": state["state_transaction_cost_multipliers"],
        "asset_expense_ratios": state["expense_ratios"],
        "leverage_multiple": state["leverage_multiple"],
        "financing_rate": state["financing_rate"],
        "financing_inflation_sensitivity": state["financing_inflation_sensitivity"],
        "state_inflation": state["state_inflation"],
        "financing_rate_paths": financing_rate_paths,
        "financing_inflation_paths": financing_inflation_paths,
        "maintenance_margin": state["maintenance_margin"],
        "contribution": state["contribution"],
        "contribution_allocation": state["contribution_allocation"],
        "withdrawal": state["withdrawal"],
        "withdrawal_start_period": state["withdrawal_start_period"],
        "decumulation": state["decumulation"],
        "withdrawal_inflation_paths": financing_inflation_paths,
        "annual_inflation": state["annual_inflation"],
        "safe_withdrawal_rate": state["safe_withdrawal_rate"],
        "native_threads": state.get("native_threads", 1),
    }
    if state["tax_enabled"]:
        chunk_wealth = simulate_portfolio_paths(
            chunk_result,
            **portfolio_kwargs,
            tax_country=state["tax_country"],
            tax_regime=state["tax_regime"],
            asset_tax_categories=state["asset_tax_categories"],
            asset_tax_metadata=state["asset_tax_metadata"],
            italy_annual_wealth_tax=state["italy_annual_wealth_tax"],
            italy_wealth_tax_mode=state["italy_wealth_tax_mode"],
            tax_terminal_liquidation=state["tax_terminal_liquidation"],
            tax_start_date=state["tax_start_date"],
            tax_wrapper_benchmark=state["tax_wrapper_benchmark"],
        )
        native_gross = chunk_wealth.attrs.pop("native_gross_wealth", None)
        if native_gross is not None:
            gross_wealth = pd.DataFrame(native_gross, columns=chunk_wealth.columns)
            gross_wealth.attrs.update(
                {
                    "margin_calls": 0,
                    "tax_country": "none",
                    "tax_regime": "none",
                    "transaction_cost_total": float(
                        chunk_wealth.attrs.pop("native_gross_transaction_cost_total", 0.0)
                    ),
                    "native_backend": True,
                }
            )
        else:
            gross_wealth = simulate_portfolio_paths(
                chunk_result,
                **portfolio_kwargs,
                tax_country="none",
                tax_regime="none",
            )
        gross_values: np.ndarray | None = gross_wealth.to_numpy(dtype=float)
    else:
        gross_wealth = simulate_portfolio_paths(
            chunk_result,
            **portfolio_kwargs,
            tax_country="none",
            tax_regime="none",
        )
        chunk_wealth = gross_wealth
        gross_values = None
    alternative_plan = _alternative_decumulation(state["decumulation"])
    if alternative_plan is not None:
        paired_kwargs = dict(portfolio_kwargs)
        paired_kwargs["decumulation"] = alternative_plan
        if state["tax_enabled"]:
            paired = simulate_portfolio_paths(
                chunk_result,
                **paired_kwargs,
                tax_country=state["tax_country"],
                tax_regime=state["tax_regime"],
                asset_tax_categories=state["asset_tax_categories"],
                asset_tax_metadata=state["asset_tax_metadata"],
                italy_annual_wealth_tax=state["italy_annual_wealth_tax"],
                italy_wealth_tax_mode=state["italy_wealth_tax_mode"],
                tax_terminal_liquidation=state["tax_terminal_liquidation"],
                tax_start_date=state["tax_start_date"],
                tax_wrapper_benchmark=False,
            )
        else:
            paired = simulate_portfolio_paths(
                chunk_result,
                **paired_kwargs,
                tax_country="none",
                tax_regime="none",
            )
        chunk_wealth.attrs["paired_policy"] = alternative_plan.policy
        chunk_wealth.attrs["paired_wealth"] = paired.to_numpy(dtype=float)
        chunk_wealth.attrs["paired_withdrawal_requested"] = paired.attrs[
            "withdrawal_requested"
        ]
        chunk_wealth.attrs["paired_withdrawal_funded"] = paired.attrs[
            "withdrawal_funded"
        ]
        chunk_wealth.attrs["paired_guardrail_events"] = paired.attrs["guardrail_events"]
    wealth_values = chunk_wealth.to_numpy(dtype=float)
    return (
        start,
        wealth_values,
        gross_values,
        chunk_result.regimes,
        chunk_result.macro_paths,
        {key: float(chunk_wealth.attrs.get(key, 0.0)) for key in _ADDITIVE_WEALTH_ATTRS},
        {key: float(gross_wealth.attrs.get(key, 0.0)) for key in _ADDITIVE_WEALTH_ATTRS},
        chunk_wealth.attrs.get("tax_by_year", {}),
        chunk_wealth.attrs.get("wrapper_terminal_values"),
        chunk_wealth.attrs.get("wrapper_annualized_returns"),
        {
            key: np.asarray(chunk_wealth.attrs[key])
            for key in _PATH_WEALTH_ATTRS
            if key in chunk_wealth.attrs
        },
    )


def _chunk_specs(total: int, chunk_size: int, random_seed: int) -> list[tuple[int, int, int]]:
    return [
        (start, min(chunk_size, total - start), int(random_seed) + start // chunk_size)
        for start in range(0, total, chunk_size)
    ]


def _simulate_chunked(
    model: ScenarioModel,
    periods: int,
    paths: int,
    random_seed: int,
    start_state: str | None,
    distribution: str,
    transition_concentration: float | None,
    duration_model: str,
    min_regime_duration: int,
    garch: bool,
    garch_alpha: float,
    garch_beta: float,
    joint_macro: bool,
    macro_transition_weight: float,
    macro_parameter_uncertainty: bool,
    dynamic_correlation: bool,
    dcc_alpha: float,
    dcc_beta: float,
    dcc_asymmetry: float,
    chunk_size: int | None,
    weight_series: pd.Series,
    initial_value: float,
    rebalance_frequency: int | None,
    transaction_cost_bps: float,
    state_transaction_cost_multipliers: Mapping[str, float] | None,
    expense_ratios: pd.Series,
    leverage_multiple: float,
    financing_rate: float,
    financing_inflation_sensitivity: float,
    state_inflation: Mapping[str, float] | None,
    maintenance_margin: float,
    contribution: float,
    contribution_allocation: str,
    withdrawal: float,
    withdrawal_start_period: int,
    decumulation: DecumulationPlan,
    annual_inflation: float,
    safe_withdrawal_rate: float,
    tax_country: str,
    tax_regime: str,
    asset_tax_categories: Mapping[str, str] | None,
    asset_tax_metadata: Mapping[str, Mapping[str, object]] | None,
    italy_annual_wealth_tax: float,
    italy_wealth_tax_mode: str,
    tax_terminal_liquidation: bool,
    tax_start_date: str | None,
    tax_wrapper_benchmark: bool,
    return_kind: str,
    rate_col: str | None,
    inflation_col: str,
    workers: int = 1,
) -> tuple[SimulationResult, pd.DataFrame, pd.DataFrame]:
    """Simulate returns and portfolio wealth, chunking the path dimension.

    ``simulate_returns`` materializes a ``(periods, paths, assets)`` array, so a
    single 100k-path run can require multiple gigabytes. This helper simulates
    ``chunk_size`` paths at a time and only keeps the accumulated wealth and
    regime paths, bounding peak memory to roughly one chunk plus the wealth
    frame. The regime arrays are concatenated so the caller sees the same
    ``SimulationResult`` shape as a one-shot run.

    Native tax runs use ``workers`` as one C++ thread pool. Python fallback runs
    use a ``ProcessPoolExecutor`` instead, avoiding nested parallelism.
    """

    tax_selection = resolve_tax_selection(tax_country, tax_regime)
    native_tax_execution = _native_tax_compatible(
        tax_selection.enabled,
        asset_tax_metadata,
    ) and (decumulation.legacy_nominal or not decumulation.active)
    native_threads = max(1, int(workers)) if native_tax_execution else 1
    native_portfolio_config: dict[str, object] | None = None
    if native_tax_execution:
        native_weights = weight_series.reindex(model.assets).fillna(0.0).to_numpy(dtype=float)
        native_weights = native_weights / native_weights.sum()
        native_portfolio_config = prepare_italian_native_configuration(
            periods=periods,
            assets=model.assets,
            target_weights=native_weights,
            initial_value=initial_value,
            rebalance_frequency=int(rebalance_frequency or 0),
            transaction_cost_bps=transaction_cost_bps,
            contribution=contribution,
            contribution_allocation=contribution_allocation,
            withdrawal=withdrawal,
            withdrawal_start_period=withdrawal_start_period,
            asset_tax_categories=asset_tax_categories,
            asset_tax_metadata=asset_tax_metadata,
            annual_wealth_tax=italy_annual_wealth_tax,
            terminal_liquidation=tax_terminal_liquidation,
            tax_regime=tax_selection.regime,
            wealth_tax_mode=italy_wealth_tax_mode,
            start_date=tax_start_date,
            wrapper_benchmark=tax_wrapper_benchmark,
        )
        native_portfolio_config.update(
            {
                "expense_ratios": expense_ratios.reindex(model.assets).fillna(0.0).to_numpy(
                    dtype=float
                ),
                "return_kind": return_kind,
                "state_transaction_cost_multipliers": dict(
                    state_transaction_cost_multipliers or {}
                ),
            }
        )

    def _single(
        paths_now: int,
        seed: int,
    ) -> tuple[SimulationResult, pd.DataFrame, pd.DataFrame]:
        chunk_result = simulate_returns(
            model,
            periods=periods,
            paths=paths_now,
            start_state=start_state,
            random_seed=seed,
            distribution=distribution,
            transition_concentration=transition_concentration,
            duration_model=duration_model,
            min_regime_duration=min_regime_duration,
            garch=garch,
            garch_alpha=garch_alpha,
            garch_beta=garch_beta,
            joint_macro=joint_macro,
            macro_transition_weight=macro_transition_weight,
            macro_parameter_uncertainty=macro_parameter_uncertainty,
            dynamic_correlation=dynamic_correlation,
            dcc_alpha=dcc_alpha,
            dcc_beta=dcc_beta,
            dcc_asymmetry=dcc_asymmetry,
            return_regime_codes=True,
            native_threads=native_threads,
            native_portfolio_config=native_portfolio_config,
        )
        if chunk_result.native_portfolio is not None:
            chunk_wealth = italian_native_result_frame(
                chunk_result.native_portfolio,
                chunk_result.native_portfolio["frame_metadata"],
                fused=True,
            )
            chunk_wealth.attrs.update(
                {
                    "tax_country": tax_selection.country,
                    "tax_regime": tax_selection.regime,
                }
            )
            native_gross = chunk_wealth.attrs.pop("native_gross_wealth")
            gross_wealth = pd.DataFrame(native_gross, columns=chunk_wealth.columns)
            gross_wealth.attrs.update(
                {
                    "margin_calls": 0,
                    "tax_country": "none",
                    "tax_regime": "none",
                    "transaction_cost_total": float(
                        chunk_wealth.attrs.pop("native_gross_transaction_cost_total", 0.0)
                    ),
                    "native_backend": True,
                    "native_fused_backend": True,
                }
            )
            return chunk_result, chunk_wealth, gross_wealth
        financing_rate_paths = _annual_macro_paths(
            model,
            chunk_result,
            rate_col,
            "rate_is_percent",
        )
        financing_inflation_paths = _annual_macro_paths(
            model,
            chunk_result,
            inflation_col,
            "inflation_is_percent",
        )
        portfolio_kwargs = {
            "weights": weight_series.to_dict(),
            "initial_value": initial_value,
            "return_kind": return_kind,
            "rebalance_frequency": rebalance_frequency,
            "transaction_cost_bps": transaction_cost_bps,
            "state_transaction_cost_multipliers": state_transaction_cost_multipliers,
            "asset_expense_ratios": expense_ratios.to_dict(),
            "leverage_multiple": leverage_multiple,
            "financing_rate": financing_rate,
            "financing_inflation_sensitivity": financing_inflation_sensitivity,
            "state_inflation": state_inflation,
            "financing_rate_paths": financing_rate_paths,
            "financing_inflation_paths": financing_inflation_paths,
            "maintenance_margin": maintenance_margin,
            "contribution": contribution,
            "contribution_allocation": contribution_allocation,
            "withdrawal": withdrawal,
            "withdrawal_start_period": withdrawal_start_period,
            "decumulation": decumulation,
            "withdrawal_inflation_paths": financing_inflation_paths,
            "annual_inflation": annual_inflation,
            "safe_withdrawal_rate": safe_withdrawal_rate,
            "native_threads": native_threads,
        }
        if tax_selection.enabled:
            chunk_wealth = simulate_portfolio_paths(
                chunk_result,
                **portfolio_kwargs,
                tax_country=tax_selection.country,
                tax_regime=tax_selection.regime,
                asset_tax_categories=asset_tax_categories,
                asset_tax_metadata=asset_tax_metadata,
                italy_annual_wealth_tax=italy_annual_wealth_tax,
                italy_wealth_tax_mode=italy_wealth_tax_mode,
                tax_terminal_liquidation=tax_terminal_liquidation,
                tax_start_date=tax_start_date,
                tax_wrapper_benchmark=tax_wrapper_benchmark,
            )
            native_gross = chunk_wealth.attrs.pop("native_gross_wealth", None)
            if native_gross is not None:
                gross_wealth = pd.DataFrame(native_gross, columns=chunk_wealth.columns)
                gross_wealth.attrs.update(
                    {
                        "margin_calls": 0,
                        "tax_country": "none",
                        "tax_regime": "none",
                        "transaction_cost_total": float(
                            chunk_wealth.attrs.pop(
                                "native_gross_transaction_cost_total",
                                0.0,
                            )
                        ),
                        "native_backend": True,
                    }
                )
            else:
                gross_wealth = simulate_portfolio_paths(
                    chunk_result,
                    **portfolio_kwargs,
                    tax_country="none",
                    tax_regime="none",
                )
        else:
            gross_wealth = simulate_portfolio_paths(
                chunk_result,
                **portfolio_kwargs,
                tax_country="none",
                tax_regime="none",
            )
            chunk_wealth = gross_wealth
        alternative_plan = _alternative_decumulation(decumulation)
        if alternative_plan is not None:
            paired_kwargs = dict(portfolio_kwargs)
            paired_kwargs["decumulation"] = alternative_plan
            if tax_selection.enabled:
                paired = simulate_portfolio_paths(
                    chunk_result,
                    **paired_kwargs,
                    tax_country=tax_selection.country,
                    tax_regime=tax_selection.regime,
                    asset_tax_categories=asset_tax_categories,
                    asset_tax_metadata=asset_tax_metadata,
                    italy_annual_wealth_tax=italy_annual_wealth_tax,
                    italy_wealth_tax_mode=italy_wealth_tax_mode,
                    tax_terminal_liquidation=tax_terminal_liquidation,
                    tax_start_date=tax_start_date,
                    tax_wrapper_benchmark=False,
                )
            else:
                paired = simulate_portfolio_paths(
                    chunk_result,
                    **paired_kwargs,
                    tax_country="none",
                    tax_regime="none",
                )
            chunk_wealth.attrs["paired_policy"] = alternative_plan.policy
            chunk_wealth.attrs["paired_wealth"] = paired.to_numpy(dtype=float)
            chunk_wealth.attrs["paired_withdrawal_requested"] = paired.attrs[
                "withdrawal_requested"
            ]
            chunk_wealth.attrs["paired_withdrawal_funded"] = paired.attrs[
                "withdrawal_funded"
            ]
            chunk_wealth.attrs["paired_guardrail_events"] = paired.attrs[
                "guardrail_events"
            ]
        return chunk_result, chunk_wealth, gross_wealth

    if chunk_size is None or paths <= chunk_size:
        return _single(int(paths), int(random_seed))

    total = int(paths)
    chunk_size = max(1, int(chunk_size))
    regime_codes = np.empty(
        (periods, total),
        dtype=np.min_scalar_type(max(len(model.states) - 1, 0)),
    )
    macro_columns = (
        list(model.metadata.get("macro_dynamics", {}).get("columns", []))
        if joint_macro
        else []
    )
    macro_paths = (
        np.empty((periods, total, len(macro_columns)), dtype=float)
        if macro_columns
        else None
    )
    wealth_values = np.empty((periods, total), dtype=float)
    gross_wealth_values = (
        np.empty((periods, total), dtype=float) if tax_selection.enabled else wealth_values
    )
    aggregate_attrs = {key: 0.0 for key in _ADDITIVE_WEALTH_ATTRS}
    aggregate_gross_attrs = {key: 0.0 for key in _ADDITIVE_WEALTH_ATTRS}
    aggregate_tax_by_year: dict[str, dict[str, float]] = {}
    wrapper_available = bool(
        tax_wrapper_benchmark
        and tax_selection.enabled
        and tax_terminal_liquidation
        and tax_selection.regime != "italy_managed"
    )
    wrapper_terminal_values = np.empty(total, dtype=float) if wrapper_available else None
    wrapper_annualized_returns = np.empty(total, dtype=float) if wrapper_available else None
    path_attrs: dict[str, np.ndarray] = {}
    if decumulation.active:
        path_attrs.update(
            {
                "withdrawal_requested": np.zeros((periods, total), dtype=float),
                "withdrawal_funded": np.zeros((periods, total), dtype=float),
                "guardrail_events": np.zeros((periods, total), dtype=np.int8),
                "withdrawal_cpi": np.ones((periods, total), dtype=float),
            }
        )
        if _alternative_decumulation(decumulation) is not None:
            path_attrs.update(
                {
                    "paired_wealth": np.zeros((periods, total), dtype=float),
                    "paired_withdrawal_requested": np.zeros(
                        (periods, total), dtype=float
                    ),
                    "paired_withdrawal_funded": np.zeros(
                        (periods, total), dtype=float
                    ),
                    "paired_guardrail_events": np.zeros(
                        (periods, total), dtype=np.int8
                    ),
                }
            )
    specs = _chunk_specs(total, chunk_size, random_seed)

    if workers is not None and workers > 1 and not native_tax_execution:
        worker_state = {
            "model": model,
            "periods": periods,
            "start_state": start_state,
            "distribution": distribution,
            "transition_concentration": transition_concentration,
            "duration_model": duration_model,
            "min_regime_duration": int(min_regime_duration),
            "garch": bool(garch),
            "garch_alpha": float(garch_alpha),
            "garch_beta": float(garch_beta),
            "joint_macro": bool(joint_macro),
            "macro_transition_weight": float(macro_transition_weight),
            "macro_parameter_uncertainty": bool(macro_parameter_uncertainty),
            "dynamic_correlation": bool(dynamic_correlation),
            "dcc_alpha": float(dcc_alpha),
            "dcc_beta": float(dcc_beta),
            "dcc_asymmetry": float(dcc_asymmetry),
            "weights": weight_series.to_dict(),
            "initial_value": float(initial_value),
            "return_kind": return_kind,
            "rebalance_frequency": rebalance_frequency,
            "transaction_cost_bps": float(transaction_cost_bps),
            "state_transaction_cost_multipliers": dict(state_transaction_cost_multipliers or {}),
            "expense_ratios": expense_ratios.to_dict(),
            "leverage_multiple": float(leverage_multiple),
            "financing_rate": float(financing_rate),
            "financing_inflation_sensitivity": float(financing_inflation_sensitivity),
            "state_inflation": dict(state_inflation) if state_inflation else None,
            "rate_col": rate_col,
            "inflation_col": inflation_col,
            "maintenance_margin": float(maintenance_margin),
            "contribution": float(contribution),
            "contribution_allocation": contribution_allocation,
            "withdrawal": float(withdrawal),
            "withdrawal_start_period": int(withdrawal_start_period),
            "decumulation": decumulation,
            "annual_inflation": float(annual_inflation),
            "safe_withdrawal_rate": float(safe_withdrawal_rate),
            "tax_enabled": tax_selection.enabled,
            "tax_country": tax_selection.country,
            "tax_regime": tax_selection.regime,
            "asset_tax_categories": dict(asset_tax_categories or {}),
            "asset_tax_metadata": {asset: dict(values) for asset, values in (asset_tax_metadata or {}).items()},
            "italy_annual_wealth_tax": float(italy_annual_wealth_tax),
            "italy_wealth_tax_mode": italy_wealth_tax_mode,
            "tax_terminal_liquidation": bool(tax_terminal_liquidation),
            "tax_start_date": tax_start_date,
            "tax_wrapper_benchmark": bool(tax_wrapper_benchmark),
            "native_threads": 1,
        }
        try:
            with ProcessPoolExecutor(
                max_workers=int(workers),
                mp_context=get_context("spawn"),
                initializer=_init_chunk_worker,
                initargs=(worker_state,),
            ) as executor:
                for (
                    start,
                    chunk_wealth_values,
                    chunk_gross_values,
                    chunk_regime_codes,
                    chunk_macro,
                    chunk_attrs,
                    chunk_gross_attrs,
                    chunk_tax_by_year,
                    chunk_wrapper_terminal,
                    chunk_wrapper_annualized,
                    chunk_path_attrs,
                ) in executor.map(
                    _run_chunk, specs
                ):
                    count = chunk_wealth_values.shape[1]
                    wealth_values[:, start:start + count] = chunk_wealth_values
                    if chunk_gross_values is not None:
                        gross_wealth_values[:, start:start + count] = chunk_gross_values
                    regime_codes[:, start:start + count] = chunk_regime_codes
                    for key in _ADDITIVE_WEALTH_ATTRS:
                        aggregate_attrs[key] += float(chunk_attrs.get(key, 0.0))
                        aggregate_gross_attrs[key] += float(
                            chunk_gross_attrs.get(key, 0.0)
                        )
                    if macro_paths is not None and chunk_macro is not None:
                        macro_paths[:, start:start + count, :] = chunk_macro
                    _merge_tax_by_year(aggregate_tax_by_year, chunk_tax_by_year)
                    if wrapper_terminal_values is not None and chunk_wrapper_terminal is not None:
                        wrapper_terminal_values[start:start + count] = chunk_wrapper_terminal
                    if wrapper_annualized_returns is not None and chunk_wrapper_annualized is not None:
                        wrapper_annualized_returns[start:start + count] = chunk_wrapper_annualized
                    for key, values in chunk_path_attrs.items():
                        if key in path_attrs:
                            path_attrs[key][:, start:start + count] = values
        except (NotImplementedError, PermissionError):
            for start, count, seed in specs:
                chunk_result, chunk_wealth, chunk_gross_wealth = _single(count, seed)
                wealth_values[:, start:start + count] = chunk_wealth.to_numpy(dtype=float)
                if tax_selection.enabled:
                    gross_wealth_values[:, start:start + count] = (
                        chunk_gross_wealth.to_numpy(dtype=float)
                    )
                regime_codes[:, start:start + count] = chunk_result.regimes
                for key in _ADDITIVE_WEALTH_ATTRS:
                    aggregate_attrs[key] += float(chunk_wealth.attrs.get(key, 0.0))
                    aggregate_gross_attrs[key] += float(
                        chunk_gross_wealth.attrs.get(key, 0.0)
                    )
                if macro_paths is not None and chunk_result.macro_paths is not None:
                    macro_paths[:, start:start + count, :] = chunk_result.macro_paths
                _merge_tax_by_year(aggregate_tax_by_year, chunk_wealth.attrs.get("tax_by_year"))
                if wrapper_terminal_values is not None:
                    wrapper_terminal_values[start:start + count] = chunk_wealth.attrs["wrapper_terminal_values"]
                    wrapper_annualized_returns[start:start + count] = chunk_wealth.attrs["wrapper_annualized_returns"]
                for key in path_attrs:
                    if key in chunk_wealth.attrs:
                        path_attrs[key][:, start:start + count] = chunk_wealth.attrs[key]
    else:
        for start, count, seed in specs:
            chunk_result, chunk_wealth, chunk_gross_wealth = _single(count, seed)
            wealth_values[:, start:start + count] = chunk_wealth.to_numpy(dtype=float)
            if tax_selection.enabled:
                gross_wealth_values[:, start:start + count] = (
                    chunk_gross_wealth.to_numpy(dtype=float)
                )
            regime_codes[:, start:start + count] = chunk_result.regimes
            for key in _ADDITIVE_WEALTH_ATTRS:
                aggregate_attrs[key] += float(chunk_wealth.attrs.get(key, 0.0))
                aggregate_gross_attrs[key] += float(
                    chunk_gross_wealth.attrs.get(key, 0.0)
                )
            if macro_paths is not None and chunk_result.macro_paths is not None:
                macro_paths[:, start:start + count, :] = chunk_result.macro_paths
            _merge_tax_by_year(aggregate_tax_by_year, chunk_wealth.attrs.get("tax_by_year"))
            if wrapper_terminal_values is not None:
                wrapper_terminal_values[start:start + count] = chunk_wealth.attrs["wrapper_terminal_values"]
                wrapper_annualized_returns[start:start + count] = chunk_wealth.attrs["wrapper_annualized_returns"]
            for key in path_attrs:
                if key in chunk_wealth.attrs:
                    path_attrs[key][:, start:start + count] = chunk_wealth.attrs[key]

    wealth = pd.DataFrame(
        wealth_values,
        columns=[f"path_{index}" for index in range(total)],
    )
    wealth.attrs.update(aggregate_attrs)
    wealth.attrs.update(
        {
            "tax_country": tax_selection.country,
            "tax_regime": tax_selection.regime,
            "asset_tax_categories": dict(asset_tax_categories or {}),
            "asset_tax_metadata": {asset: dict(values) for asset, values in (asset_tax_metadata or {}).items()},
            "annual_wealth_tax": float(italy_annual_wealth_tax),
            "wealth_tax_mode": italy_wealth_tax_mode,
            "tax_terminal_liquidation": bool(tax_terminal_liquidation),
            "tax_start_date": tax_start_date,
            "tax_by_year": aggregate_tax_by_year,
            "contribution_allocation": contribution_allocation,
            "tax_wrapper_benchmark_requested": bool(tax_wrapper_benchmark),
            "tax_wrapper_benchmark_available": wrapper_available,
            "tax_wrapper_unavailable_reason": (
                "managed_regime"
                if tax_wrapper_benchmark and tax_selection.regime == "italy_managed"
                else "terminal_liquidation_required"
                if tax_wrapper_benchmark and not tax_terminal_liquidation
                else None
            ),
            "wrapper_terminal_values": wrapper_terminal_values,
            "wrapper_annualized_returns": wrapper_annualized_returns,
            "native_backend": native_tax_execution,
            "native_fused_backend": bool(native_tax_execution),
            "decumulation": decumulation.to_dict(),
            "paired_policy": (
                _alternative_decumulation(decumulation).policy
                if _alternative_decumulation(decumulation) is not None
                else None
            ),
            **path_attrs,
        }
    )
    if tax_selection.enabled:
        gross_wealth = pd.DataFrame(
            gross_wealth_values,
            columns=wealth.columns,
        )
        gross_wealth.attrs.update(aggregate_gross_attrs)
        gross_wealth.attrs.update(
            {
                "tax_country": "none",
                "tax_regime": "none",
                "native_backend": native_tax_execution,
                "native_fused_backend": bool(native_tax_execution),
            }
        )
    else:
        gross_wealth = wealth
    combined = SimulationResult(
        returns=np.empty((periods, 0, len(model.assets)), dtype=float),
        regimes=regime_codes,
        assets=model.assets,
        states=model.states.copy(),
        frequency=model.frequency,
        distribution="mnts",
        transition_concentration=transition_concentration,
        macro_paths=macro_paths,
        macro_columns=macro_columns,
    )
    return combined, wealth, gross_wealth


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
    rate_col: str | None = "interest_rate",
    correlation_overrides: CorrelationOverrides | None = None,
    override_weight: float = 1.0,
    macro_lag_periods: int = 0,
    distribution: str = "mnts",
    transition_uncertainty: float = 0.0,
    rebalance_frequency: int | None = None,
    transaction_cost_bps: float = 0.0,
    state_dependent_liquidity: bool = False,
    state_transaction_cost_multipliers: Mapping[str, float] | None = None,
    asset_expense_ratios: Mapping[str, float] | None = None,
    leverage_multiple: float = 1.0,
    financing_rate: float = 0.0,
    financing_inflation_sensitivity: float = 0.0,
    maintenance_margin: float = 0.0,
    contribution: float = 0.0,
    contribution_allocation: str = "target",
    withdrawal: float = 0.0,
    withdrawal_start_period: int = 1,
    decumulation: Mapping[str, Any] | DecumulationPlan | None = None,
    safe_withdrawal_rate: float = 0.0,
    tax_country: str | None = None,
    tax_regime: str = "none",
    asset_tax_categories: Mapping[str, str] | None = None,
    asset_tax_metadata: Mapping[str, Mapping[str, object]] | None = None,
    italy_annual_wealth_tax: float = 0.002,
    italy_wealth_tax_mode: str = "auto",
    tax_terminal_liquidation: bool = True,
    tax_start_date: str | None = None,
    tax_wrapper_benchmark: bool = False,
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
    duration_model: str = "semi_markov",
    min_regime_duration: int = 5,
    garch: bool = True,
    garch_alpha: float = 0.10,
    garch_beta: float = 0.85,
    walk_forward: bool = True,
    probabilistic_regimes: bool = False,
    regime_temperature: float = 0.35,
    regime_smoothing_window: int = 3,
    regime_hysteresis: float = 0.15,
    regime_confirmation_periods: int = 2,
    duration_prior_strength: float = 8.0,
    mean_prior_strength: float = 0.0,
    parameter_draws: int = 0,
    parameter_block_size: int = 12,
    joint_macro: bool = False,
    macro_transition_weight: float = 0.35,
    macro_parameter_uncertainty: bool = True,
    macro_model: str = "bvar_ensemble",
    structural_returns: bool = False,
    asset_classes: Mapping[str, str] | None = None,
    asset_durations: Mapping[str, float] | None = None,
    asset_income_yields: Mapping[str, float] | None = None,
    dynamic_correlation: bool = False,
    dcc_alpha: float = 0.04,
    dcc_beta: float = 0.94,
    dcc_asymmetry: float = 0.01,
    chunk_size: int | None = None,
    return_kind: str = "log",
    workers: int = 1,
) -> SimulationRun:
    """Calibrate and simulate one fully specified investment scenario.

    ``model_kind="quadrant"`` builds the four macro states from high/low growth
    crossed with high/low inflation. Returns in every state follow the
    calibrated MNTS-GARCH process. ``duration_model`` controls whether regime
    run lengths follow the Markov chain or regularized state-specific duration
    hazards. ``walk_forward`` runs a strictly out-of-sample predictive check.
    """

    distribution = str(distribution).strip().lower().replace("-", "_")
    if distribution != "mnts":
        raise ValueError("distribution must be 'mnts'.")
    transition_uncertainty = float(transition_uncertainty)
    if not 0 <= transition_uncertainty <= 1:
        raise ValueError("transition_uncertainty must be between 0 and 1.")
    withdrawal_start_period_value = float(withdrawal_start_period)
    if not np.isfinite(withdrawal_start_period_value) or not withdrawal_start_period_value.is_integer():
        raise ValueError("withdrawal_start_period must be an integer.")
    withdrawal_start_period = int(withdrawal_start_period_value)
    if not 1 <= withdrawal_start_period <= int(periods):
        raise ValueError(
            "withdrawal_start_period must be between 1 and the simulation periods."
        )
    decumulation_plan = normalize_decumulation(
        decumulation,
        periods=int(periods),
        legacy_withdrawal=float(withdrawal),
        legacy_start_period=withdrawal_start_period,
        annual_inflation_fallback=float(annual_inflation),
    )
    requested_macro_lag_periods = int(macro_lag_periods)
    if requested_macro_lag_periods < 0:
        raise ValueError("macro_lag_periods must be non-negative.")
    macro_lag_periods = (
        0 if bool(macro.attrs.get("availability_aligned", False)) else requested_macro_lag_periods
    )
    parameter_draws = int(parameter_draws)
    if parameter_draws < 0 or parameter_draws > 100:
        raise ValueError("parameter_draws must be between 0 and 100.")
    if parameter_draws > int(paths):
        raise ValueError("parameter_draws cannot exceed simulated paths.")
    if parameter_draws and model_kind == "hmm":
        raise ValueError("Parameter bootstrap is currently available for the quadrant model only.")
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
    normalized_tax_categories = {
        available_columns.get(str(asset).strip().upper(), asset): str(category).strip().lower()
        for asset, category in (asset_tax_categories or {}).items()
    }
    normalized_tax_metadata = {
        available_columns.get(str(asset).strip().upper(), asset): dict(values)
        for asset, values in (asset_tax_metadata or {}).items()
    }
    normalized_asset_classes = {
        available_columns.get(str(asset).strip().upper(), asset): str(value).strip().lower()
        for asset, value in (asset_classes or {}).items()
    }
    normalized_asset_durations = {
        available_columns.get(str(asset).strip().upper(), asset): float(value)
        for asset, value in (asset_durations or {}).items()
    }
    normalized_asset_income_yields = {
        available_columns.get(str(asset).strip().upper(), asset): float(value)
        for asset, value in (asset_income_yields or {}).items()
    }
    liquidity_multipliers = (
        dict(state_transaction_cost_multipliers or DEFAULT_LIQUIDITY_COST_MULTIPLIERS)
        if state_dependent_liquidity
        else None
    )
    if tax_start_date is None and len(macro.index):
        tax_start_date = str((pd.Timestamp(macro.index.max()) + pd.offsets.MonthEnd(1)).date())
    tax_selection = resolve_tax_selection(tax_country, tax_regime)
    tax_country = tax_selection.country
    tax_regime = tax_selection.regime
    if tax_selection.enabled:
        normalized_tax_metadata = normalize_italy_tax_metadata(
            selected,
            normalized_tax_categories,
            normalized_tax_metadata,
        )
        normalized_tax_categories = {
            asset: str(values["category"])
            for asset, values in normalized_tax_metadata.items()
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
            min_regime_duration=int(min_regime_duration),
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
            rate_col=rate_col,
            growth_threshold=growth_threshold,
            inflation_threshold=inflation_threshold,
            correlation_overrides=correlation_overrides,
            override_weight=override_weight,
            macro_lag_periods=macro_lag_periods,
            threshold_window=threshold_window,
            min_regime_duration=int(min_regime_duration),
            probabilistic_regimes=probabilistic_regimes,
            regime_temperature=regime_temperature,
            regime_smoothing_window=int(regime_smoothing_window),
            regime_hysteresis=float(regime_hysteresis),
            regime_confirmation_periods=int(regime_confirmation_periods),
            duration_prior_strength=float(duration_prior_strength),
            mean_prior_strength=mean_prior_strength,
            joint_macro=joint_macro,
            structural_returns=structural_returns,
            asset_classes=normalized_asset_classes,
            asset_durations=normalized_asset_durations,
            asset_income_yields=normalized_asset_income_yields,
            macro_model=macro_model,
        )
        model.metadata["requested_macro_lag_periods"] = requested_macro_lag_periods
        regimes = classify_persistent_quadrants(
            macro,
            growth_col=growth_col,
            inflation_col=inflation_col,
            growth_threshold=growth_threshold,
            inflation_threshold=inflation_threshold,
            threshold_window=threshold_window,
            smoothing_window=int(regime_smoothing_window),
            hysteresis=float(regime_hysteresis),
            confirmation_periods=int(regime_confirmation_periods),
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
            regime_smoothing_window=int(regime_smoothing_window),
            regime_hysteresis=float(regime_hysteresis),
            regime_confirmation_periods=int(regime_confirmation_periods),
        )
        if transition_uncertainty > 0:
            diagnostics.warnings.append(
                f"Transition probabilities are sampled with uncertainty {transition_uncertainty:.2f}."
            )
        if parameter_draws > 0:
            diagnostics.warnings.append(
                f"Parameter uncertainty uses {parameter_draws} stationary-bootstrap recalibrations."
            )
        if not bool(model.metadata.get("point_in_time", False)):
            diagnostics.warnings.append(
                "Macro history is revised rather than point-in-time; select ALFRED initial-release data "
                "for strict pseudo-live calibration."
            )
        model.metadata["base_currency"] = str(base_currency).strip().upper()
        model.metadata["fx_quote"] = fx_quote
        model.metadata["initial_value"] = float(initial_value)
        if asset_currencies:
            model.metadata["asset_currencies"] = dict(asset_currencies)

    if tax_selection.enabled and structural_returns:
        for asset, profile in model.metadata.get("asset_profiles", {}).items():
            metadata = normalized_tax_metadata.setdefault(asset, {})
            metadata.setdefault("annual_income_yield", float(profile.get("income_yield", 0.0)))

    parameter_models: list[ScenarioModel] = []
    parameter_summary: pd.DataFrame | None = None
    if parameter_draws:
        parameter_models = bootstrap_quadrant_models(
            scenario_returns,
            macro,
            draws=parameter_draws,
            block_size=int(parameter_block_size),
            random_seed=int(random_seed) + 10_000,
            growth_col=growth_col,
            inflation_col=inflation_col,
            rate_col=rate_col,
            growth_threshold=growth_threshold,
            inflation_threshold=inflation_threshold,
            correlation_overrides=correlation_overrides,
            override_weight=override_weight,
            macro_lag_periods=macro_lag_periods,
            threshold_window=threshold_window,
            min_regime_duration=int(min_regime_duration),
            probabilistic_regimes=probabilistic_regimes,
            regime_temperature=float(regime_temperature),
            regime_smoothing_window=int(regime_smoothing_window),
            regime_hysteresis=float(regime_hysteresis),
            regime_confirmation_periods=int(regime_confirmation_periods),
            duration_prior_strength=float(duration_prior_strength),
            mean_prior_strength=float(mean_prior_strength),
            joint_macro=joint_macro,
            structural_returns=structural_returns,
            asset_classes=normalized_asset_classes,
            asset_durations=normalized_asset_durations,
            asset_income_yields=normalized_asset_income_yields,
            macro_model=macro_model,
        )
        parameter_summary = summarize_parameter_models(parameter_models, normalized_weights)

    simulation_models = parameter_models or [model]
    quotient, remainder = divmod(int(paths), len(simulation_models))
    model_path_counts = [quotient + (1 if index < remainder else 0) for index in range(len(simulation_models))]
    simulation_runs: list[tuple[SimulationResult, pd.DataFrame, pd.DataFrame]] = []
    for draw, (simulation_model, draw_paths) in enumerate(zip(simulation_models, model_path_counts)):
        draw_result, draw_wealth, draw_gross_wealth = _simulate_chunked(
                simulation_model,
                periods=int(periods),
                paths=draw_paths,
                random_seed=int(random_seed) + draw * 100_003,
                start_state=start_state,
                distribution=distribution,
                transition_concentration=(
                    None
                    if transition_uncertainty == 0.0 or parameter_models
                    else max(1.0, 1.0 / transition_uncertainty**2)
                ),
                duration_model=duration_model,
                min_regime_duration=int(min_regime_duration),
                garch=garch,
                garch_alpha=float(garch_alpha),
                garch_beta=float(garch_beta),
                joint_macro=joint_macro,
                macro_parameter_uncertainty=macro_parameter_uncertainty,
                macro_transition_weight=float(macro_transition_weight),
                dynamic_correlation=dynamic_correlation,
                dcc_alpha=float(dcc_alpha),
                dcc_beta=float(dcc_beta),
                dcc_asymmetry=float(dcc_asymmetry),
                chunk_size=chunk_size,
                weight_series=pd.Series(normalized_weights, dtype=float),
                initial_value=initial_value,
                rebalance_frequency=rebalance_frequency,
                transaction_cost_bps=float(transaction_cost_bps),
                state_transaction_cost_multipliers=liquidity_multipliers,
                expense_ratios=pd.Series(normalized_expense_ratios, dtype=float),
                leverage_multiple=float(leverage_multiple),
                financing_rate=float(financing_rate),
                financing_inflation_sensitivity=float(financing_inflation_sensitivity),
                state_inflation=simulation_model.metadata.get("state_inflation"),
                maintenance_margin=float(maintenance_margin),
                contribution=float(contribution),
                contribution_allocation=contribution_allocation,
                withdrawal=float(withdrawal),
                withdrawal_start_period=int(withdrawal_start_period),
                decumulation=decumulation_plan,
                annual_inflation=float(annual_inflation),
                safe_withdrawal_rate=float(safe_withdrawal_rate),
                tax_country=tax_country,
                tax_regime=tax_regime,
                asset_tax_categories=normalized_tax_categories,
                asset_tax_metadata=normalized_tax_metadata,
                italy_annual_wealth_tax=float(italy_annual_wealth_tax),
                italy_wealth_tax_mode=italy_wealth_tax_mode,
                tax_terminal_liquidation=bool(tax_terminal_liquidation),
                tax_start_date=tax_start_date,
                tax_wrapper_benchmark=bool(tax_wrapper_benchmark),
                return_kind=return_kind,
                rate_col=rate_col,
                inflation_col=inflation_col,
                workers=workers,
            )
        if parameter_models and draw_result.returns.shape[1]:
            draw_result = SimulationResult(
                returns=np.empty((int(periods), 0, len(model.assets)), dtype=float),
                regimes=draw_result.regimes,
                assets=draw_result.assets,
                states=draw_result.states,
                frequency=draw_result.frequency,
                distribution=draw_result.distribution,
                transition_concentration=draw_result.transition_concentration,
                macro_paths=draw_result.macro_paths,
                macro_columns=draw_result.macro_columns,
            )
        simulation_runs.append((draw_result, draw_wealth, draw_gross_wealth))

    if len(simulation_runs) == 1:
        result, wealth, gross_wealth = simulation_runs[0]
    else:
        wealth = pd.DataFrame(
            np.concatenate(
                [run_wealth.to_numpy(dtype=float) for _, run_wealth, _ in simulation_runs],
                axis=1,
            )
        )
        wealth.columns = [f"path_{index}" for index in range(wealth.shape[1])]
        for key in _ADDITIVE_WEALTH_ATTRS:
            wealth.attrs[key] = float(
                sum(
                    run_wealth.attrs.get(key, 0.0)
                    for _, run_wealth, _ in simulation_runs
                )
            )
        combined_tax_by_year: dict[str, dict[str, float]] = {}
        for _, run_wealth, _ in simulation_runs:
            _merge_tax_by_year(combined_tax_by_year, run_wealth.attrs.get("tax_by_year"))
        wealth.attrs.update(
            {
                "tax_country": tax_country,
                "tax_regime": tax_regime,
                "asset_tax_categories": normalized_tax_categories,
                "asset_tax_metadata": normalized_tax_metadata,
                "annual_wealth_tax": float(italy_annual_wealth_tax),
                "wealth_tax_mode": italy_wealth_tax_mode,
                "tax_terminal_liquidation": bool(tax_terminal_liquidation),
                "tax_start_date": tax_start_date,
                "tax_by_year": combined_tax_by_year,
                "contribution_allocation": contribution_allocation,
                "tax_wrapper_benchmark_requested": bool(tax_wrapper_benchmark),
                "tax_wrapper_benchmark_available": all(
                    bool(run_wealth.attrs.get("tax_wrapper_benchmark_available", False))
                    for _, run_wealth, _ in simulation_runs
                ),
                "tax_wrapper_unavailable_reason": next(
                    (
                        run_wealth.attrs.get("tax_wrapper_unavailable_reason")
                        for _, run_wealth, _ in simulation_runs
                        if run_wealth.attrs.get("tax_wrapper_unavailable_reason")
                    ),
                    None,
                ),
                "wrapper_terminal_values": (
                    np.concatenate(
                        [run_wealth.attrs["wrapper_terminal_values"] for _, run_wealth, _ in simulation_runs]
                    )
                    if all(run_wealth.attrs.get("wrapper_terminal_values") is not None for _, run_wealth, _ in simulation_runs)
                    else None
                ),
                "wrapper_annualized_returns": (
                    np.concatenate(
                        [run_wealth.attrs["wrapper_annualized_returns"] for _, run_wealth, _ in simulation_runs]
                    )
                    if all(run_wealth.attrs.get("wrapper_annualized_returns") is not None for _, run_wealth, _ in simulation_runs)
                    else None
                ),
                "decumulation": decumulation_plan.to_dict(),
                "paired_policy": (
                    _alternative_decumulation(decumulation_plan).policy
                    if _alternative_decumulation(decumulation_plan) is not None
                    else None
                ),
                **{
                    key: np.concatenate(
                        [np.asarray(run_wealth.attrs[key]) for _, run_wealth, _ in simulation_runs],
                        axis=1,
                    )
                    for key in _PATH_WEALTH_ATTRS
                    if all(key in run_wealth.attrs for _, run_wealth, _ in simulation_runs)
                },
            }
        )
        if tax_selection.enabled:
            gross_wealth = pd.DataFrame(
                np.concatenate(
                    [run_gross.to_numpy(dtype=float) for _, _, run_gross in simulation_runs],
                    axis=1,
                )
            )
            gross_wealth.columns = wealth.columns
            gross_wealth.attrs.update({"tax_country": "none", "tax_regime": "none"})
            for key in _ADDITIVE_WEALTH_ATTRS:
                gross_wealth.attrs[key] = float(
                    sum(
                        run_gross.attrs.get(key, 0.0)
                        for _, _, run_gross in simulation_runs
                    )
                )
        else:
            gross_wealth = wealth
        regimes_combined = np.concatenate(
            [run_result.regimes for run_result, _, _ in simulation_runs], axis=1
        )
        macro_parts = [
            run_result.macro_paths
            for run_result, _, _ in simulation_runs
            if run_result.macro_paths is not None
        ]
        result = SimulationResult(
            returns=np.empty((int(periods), 0, len(model.assets)), dtype=float),
            regimes=regimes_combined,
            assets=model.assets,
            states=model.states.copy(),
            frequency=model.frequency,
            distribution="mnts",
            macro_paths=(np.concatenate(macro_parts, axis=1) if macro_parts else None),
            macro_columns=(simulation_runs[0][0].macro_columns if macro_parts else []),
        )
        if parameter_summary is not None:
            terminal_offset = 0
            terminal_metrics: list[dict[str, float]] = []
            for draw_paths in model_path_counts:
                terminal = wealth.iloc[-1, terminal_offset:terminal_offset + draw_paths].to_numpy(dtype=float)
                terminal_metrics.append(
                    {
                        "terminal_p05": float(np.quantile(terminal, 0.05)),
                        "terminal_median": float(np.quantile(terminal, 0.50)),
                        "terminal_p95": float(np.quantile(terminal, 0.95)),
                    }
                )
                terminal_offset += draw_paths
            parameter_summary = pd.concat(
                [parameter_summary.reset_index(drop=True), pd.DataFrame(terminal_metrics)],
                axis=1,
            )
    terminal_groups = [run_wealth.iloc[-1].to_numpy(dtype=float) for _, run_wealth, _ in simulation_runs]
    total_terminal_count = max(sum(len(values) for values in terminal_groups), 1)
    group_weights = np.array([len(values) / total_terminal_count for values in terminal_groups], dtype=float)
    group_means = np.array([float(np.mean(values)) for values in terminal_groups], dtype=float)
    path_variance = float(
        sum(weight * float(np.var(values, ddof=0)) for weight, values in zip(group_weights, terminal_groups))
    )
    parameter_variance = float(np.sum(group_weights * np.square(group_means - np.sum(group_weights * group_means))))
    decomposed_variance = path_variance + parameter_variance
    uncertainty_decomposition = {
        "path_variance": path_variance,
        "parameter_variance": parameter_variance,
        "path_share": path_variance / decomposed_variance if decomposed_variance > 0 else 0.0,
        "parameter_share": parameter_variance / decomposed_variance if decomposed_variance > 0 else 0.0,
        "parameter_models": len(parameter_models),
        "macro_instability_score": float(
            model.metadata.get("macro_dynamics", {}).get("macro_instability_score", 0.0)
        ),
    }
    model.metadata["uncertainty_decomposition"] = uncertainty_decomposition

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
                probabilistic_regimes=probabilistic_regimes,
                regime_temperature=float(regime_temperature),
                regime_smoothing_window=int(regime_smoothing_window),
                regime_hysteresis=float(regime_hysteresis),
                regime_confirmation_periods=int(regime_confirmation_periods),
                duration_prior_strength=float(duration_prior_strength),
                min_regime_duration=int(min_regime_duration),
                mean_prior_strength=float(mean_prior_strength),
                weights=normalized_weights,
            )
            diagnostics.warnings.extend(walk_forward_result.warnings)
        except ValueError as exc:
            walk_forward_result = None
            diagnostics.warnings.append(f"Walk-forward validation unavailable: {exc}")
    weight_series = pd.Series(normalized_weights, dtype=float).reindex(selected).fillna(0.0)
    weight_total = float(weight_series.sum())
    if not pd.notna(weight_total) or abs(weight_total) < 1e-12:
        raise ValueError("Portfolio weights must have a non-zero sum.")
    weight_series = weight_series / weight_total
    fee_series = pd.Series(normalized_expense_ratios, dtype=float).reindex(selected).fillna(0.0)
    inflation_paths = None
    if result.macro_paths is not None and inflation_col in result.macro_columns:
        inflation_index = result.macro_columns.index(inflation_col)
        inflation_paths = result.macro_paths[:, :, inflation_index]
        dynamics = model.metadata.get("macro_dynamics", {})
        if bool(dynamics.get("inflation_is_percent", False)):
            inflation_paths = inflation_paths / 100.0
        inflation_paths = np.clip(inflation_paths, -0.10, 0.50)
        model.metadata["inflation_model"] = "joint_macro_path"
    else:
        model.metadata["inflation_model"] = "deterministic"
    risk_free_paths = _annual_macro_paths(
        model,
        result,
        model.metadata.get("rate_col"),
        "rate_is_percent",
    )
    if risk_free_paths is not None:
        risk_free_paths = np.clip(risk_free_paths, -0.05, 0.50)
        model.metadata["rate_model"] = "joint_macro_path"
    else:
        model.metadata["rate_model"] = "deterministic"
    reporting_wealth = (
        wealth
        if inflation_paths is None and np.isclose(annual_inflation, 0.0)
        else inflation_adjust_wealth(
            wealth,
            annual_inflation=annual_inflation,
            inflation_paths=inflation_paths,
        )
    )
    if gross_wealth is wealth:
        gross_reporting_wealth = reporting_wealth
    else:
        gross_reporting_wealth = (
            gross_wealth
            if inflation_paths is None and np.isclose(annual_inflation, 0.0)
            else inflation_adjust_wealth(
                gross_wealth,
                annual_inflation=annual_inflation,
                inflation_paths=inflation_paths,
            )
        )
    funded_withdrawal_paths = wealth.attrs.get("withdrawal_funded")
    summary = summarize_wealth_risk(
        wealth,
        initial_value=initial_value,
        risk_free_rate=risk_free_rate,
        annual_inflation=annual_inflation,
        contribution=float(contribution),
        withdrawal=float(withdrawal),
        withdrawal_start_period=int(withdrawal_start_period),
        withdrawal_paths=(
            np.asarray(funded_withdrawal_paths, dtype=float)
            if funded_withdrawal_paths is not None
            else None
        ),
        inflation_paths=inflation_paths,
        risk_free_paths=risk_free_paths,
    )
    summary = summary.copy()
    state_inflation = model.metadata.get("state_inflation", {})
    effective_financing = float(financing_rate)
    if risk_free_paths is not None:
        financing_paths = risk_free_paths + float(financing_rate)
        if float(financing_inflation_sensitivity) > 0 and inflation_paths is not None:
            financing_paths += float(financing_inflation_sensitivity) * inflation_paths
        effective_financing = float(np.clip(financing_paths, 0.0, 1.0).mean())
    elif float(financing_inflation_sensitivity) > 0 and state_inflation and len(result.states):
        simulated_regimes = result.regimes
        if simulated_regimes.dtype.kind in "iu":
            hist = np.bincount(simulated_regimes.ravel(), minlength=len(result.states))
            regime_counts = {state: int(count) for state, count in zip(result.states, hist)}
        else:
            unique_states, state_counts = np.unique(simulated_regimes.ravel(), return_counts=True)
            regime_counts = {str(state): int(count) for state, count in zip(unique_states, state_counts)}
        total = max(sum(regime_counts.values()), 1)
        effective_financing = float(
            sum(
                (float(financing_rate) + float(financing_inflation_sensitivity) * float(state_inflation.get(state, 0.0)))
                * int(regime_counts.get(state, 0)) / total
                for state in result.states
            )
        )
    for key, value in {
        "weighted_expense_ratio": float((weight_series.abs() * fee_series).sum()),
        "annual_fee_drag": float((weight_series.abs() * fee_series).sum() * float(leverage_multiple)),
        "annual_financing_cost": float(max(float(leverage_multiple) - 1.0, 0.0) * effective_financing),
        "effective_financing_rate": effective_financing,
        "leverage_multiple": float(leverage_multiple),
        "maintenance_margin": float(maintenance_margin),
        "margin_calls": int(wealth.attrs.get("margin_calls", 0)),
        "capital_gains_tax": float(wealth.attrs.get("capital_gains_tax_total", 0.0))
        / max(int(paths), 1),
        "wealth_tax": float(wealth.attrs.get("wealth_tax_total", 0.0)) / max(int(paths), 1),
        "terminal_liquidation_tax": float(
            wealth.attrs.get("terminal_liquidation_tax_total", 0.0)
        )
        / max(int(paths), 1),
        "taxes_paid": float(wealth.attrs.get("taxes_paid_total", 0.0)) / max(int(paths), 1),
        "realized_gains": float(wealth.attrs.get("realized_gains_total", 0.0))
        / max(int(paths), 1),
        "realized_losses": float(wealth.attrs.get("realized_losses_total", 0.0))
        / max(int(paths), 1),
        "loss_carryforward": float(wealth.attrs.get("loss_carryforward_total", 0.0))
        / max(int(paths), 1),
        "investment_income_tax": float(wealth.attrs.get("investment_income_tax_total", 0.0))
        / max(int(paths), 1),
        "foreign_withholding_tax": float(wealth.attrs.get("foreign_withholding_tax_total", 0.0))
        / max(int(paths), 1),
        "financial_transaction_tax": float(
            wealth.attrs.get("financial_transaction_tax_total", 0.0)
        ) / max(int(paths), 1),
        "stamp_duty": float(wealth.attrs.get("stamp_duty_total", 0.0)) / max(int(paths), 1),
        "ivafe": float(wealth.attrs.get("ivafe_total", 0.0)) / max(int(paths), 1),
        "expired_losses": float(wealth.attrs.get("expired_losses_total", 0.0))
        / max(int(paths), 1),
        "annual_wealth_tax_rate": (
            float(italy_annual_wealth_tax) if tax_selection.enabled else 0.0
        ),
    }.items():
        summary[key] = value
    gross_terminal = gross_reporting_wealth.iloc[-1].to_numpy(dtype=float)
    active_terminal = reporting_wealth.iloc[-1].to_numpy(dtype=float)
    terminal_tax_drag = gross_terminal - active_terminal
    gross_terminal_median = float(np.median(gross_terminal))
    summary["gross_terminal_wealth_median"] = gross_terminal_median
    summary["after_tax_terminal_wealth_median"] = float(np.median(active_terminal))
    summary["terminal_tax_drag_median"] = float(np.median(terminal_tax_drag))
    summary["terminal_tax_drag_percent"] = (
        float(np.median(terminal_tax_drag) / gross_terminal_median)
        if not np.isclose(gross_terminal_median, 0.0)
        else 0.0
    )
    wrapper_terminal = wealth.attrs.get("wrapper_terminal_values")
    wrapper_annualized = wealth.attrs.get("wrapper_annualized_returns")
    wrapper_available = bool(wealth.attrs.get("tax_wrapper_benchmark_available", False))
    if wrapper_available and wrapper_terminal is not None and wrapper_annualized is not None:
        wrapper_terminal = np.asarray(wrapper_terminal, dtype=float)
        wrapper_annualized = np.asarray(wrapper_annualized, dtype=float)
        if inflation_paths is None:
            final_deflator = float((1.0 + annual_inflation) ** (-len(wealth) / 12.0))
            effective_inflation = np.full(len(wrapper_terminal), float(annual_inflation))
        else:
            periodic_inflation = np.power(1.0 + inflation_paths, 1.0 / 12.0)
            cumulative_inflation = np.cumprod(periodic_inflation, axis=0)[-1]
            final_deflator = 1.0 / cumulative_inflation
            effective_inflation = np.power(cumulative_inflation, 12.0 / len(wealth)) - 1.0
        wrapper_reporting_terminal = wrapper_terminal * final_deflator
        wrapper_real_annualized = (1.0 + wrapper_annualized) / (1.0 + effective_inflation) - 1.0
        diy_annualized = _cash_flow_adjusted_geometric_returns(
            wealth,
            initial_value=float(initial_value),
            contribution=float(contribution),
            withdrawal=float(withdrawal),
            withdrawal_start_period=int(withdrawal_start_period),
            annual_inflation=float(annual_inflation),
            inflation_paths=inflation_paths,
            withdrawal_paths=np.asarray(
                wealth.attrs.get("withdrawal_funded", np.zeros(wealth.shape)),
                dtype=float,
            ),
        )
        wrapper_advantage = wrapper_reporting_terminal - active_terminal
        summary["wrapper_terminal_p05"] = float(np.quantile(wrapper_reporting_terminal, 0.05))
        summary["wrapper_terminal_median"] = float(np.median(wrapper_reporting_terminal))
        summary["wrapper_terminal_p95"] = float(np.quantile(wrapper_reporting_terminal, 0.95))
        summary["wrapper_advantage_median"] = float(np.median(wrapper_advantage))
        active_median = float(np.median(active_terminal))
        summary["wrapper_advantage_percent"] = (
            float(np.median(wrapper_advantage) / active_median)
            if not np.isclose(active_median, 0.0)
            else 0.0
        )
        summary["wrapper_annual_drag_bps"] = float(
            np.median(wrapper_real_annualized - diy_annualized) * 10_000.0
        )
    else:
        for key in (
            "wrapper_terminal_p05",
            "wrapper_terminal_median",
            "wrapper_terminal_p95",
            "wrapper_advantage_median",
            "wrapper_advantage_percent",
            "wrapper_annual_drag_bps",
        ):
            summary[key] = 0.0
    horizon_years = max(float(periods) / 12.0, 1.0 / 12.0)
    net_terminal_median = float(np.median(active_terminal))
    summary["tax_drag_cagr"] = (
        (gross_terminal_median / float(initial_value)) ** (1.0 / horizon_years)
        - (net_terminal_median / float(initial_value)) ** (1.0 / horizon_years)
        if gross_terminal_median > 0 and net_terminal_median > 0
        else 0.0
    )
    funded_withdrawals = np.asarray(
        wealth.attrs.get("withdrawal_funded", np.empty(0)), dtype=float
    )
    if funded_withdrawals.shape == wealth.shape:
        withdrawal_totals = funded_withdrawals.sum(axis=0)
    else:
        withdrawal_periods = int(periods) - int(withdrawal_start_period) + 1
        withdrawal_totals = np.full(
            int(paths), float(withdrawal) * float(withdrawal_periods), dtype=float
        )
    gross_taxable_gain = np.maximum(
        gross_terminal
        + withdrawal_totals
        - float(initial_value)
        - float(contribution) * float(periods),
        0.0,
    )
    average_taxes = float(wealth.attrs.get("taxes_paid_total", 0.0)) / max(int(paths), 1)
    summary["effective_tax_rate"] = (
        average_taxes / float(np.mean(gross_taxable_gain))
        if float(np.mean(gross_taxable_gain)) > 1e-12
        else 0.0
    )
    model.metadata["tax_country"] = tax_country
    model.metadata["tax_regime"] = tax_regime
    model.metadata["decumulation"] = decumulation_plan.to_dict()
    if tax_selection.enabled:
        model.metadata["asset_tax_categories"] = dict(
            wealth.attrs.get("asset_tax_categories", normalized_tax_categories)
        )
        model.metadata["asset_tax_metadata"] = {
            asset: dict(values)
            for asset, values in wealth.attrs.get("asset_tax_metadata", normalized_tax_metadata).items()
        }
        model.metadata["italy_annual_wealth_tax"] = float(italy_annual_wealth_tax)
        model.metadata["italy_wealth_tax_mode"] = italy_wealth_tax_mode
        model.metadata["tax_terminal_liquidation"] = bool(tax_terminal_liquidation)
        model.metadata["contribution_allocation"] = contribution_allocation
        model.metadata["withdrawal_start_period"] = int(withdrawal_start_period)
        model.metadata["tax_wrapper_benchmark"] = bool(tax_wrapper_benchmark)
        model.metadata["tax_start_date"] = tax_start_date
        model.metadata["tax_by_year"] = wealth.attrs.get("tax_by_year", {})
    model.metadata["state_dependent_liquidity"] = bool(liquidity_multipliers)
    model.metadata["state_transaction_cost_multipliers"] = liquidity_multipliers or {}
    if parameter_summary is not None:
        terminal_offset = 0
        real_terminal_metrics: list[dict[str, float]] = []
        for draw_paths in model_path_counts:
            terminal = reporting_wealth.iloc[
                -1, terminal_offset:terminal_offset + draw_paths
            ].to_numpy(dtype=float)
            real_terminal_metrics.append(
                {
                    "terminal_p05": float(np.quantile(terminal, 0.05)),
                    "terminal_median": float(np.quantile(terminal, 0.50)),
                    "terminal_p95": float(np.quantile(terminal, 0.95)),
                }
            )
            terminal_offset += draw_paths
        for column in ("terminal_p05", "terminal_median", "terminal_p95"):
            parameter_summary[column] = [row[column] for row in real_terminal_metrics]
    return SimulationRun(
        model=model,
        regimes=regimes,
        result=result,
        wealth=wealth,
        summary=summary,
        diagnostics=diagnostics,
        walk_forward=walk_forward_result,
        reporting_wealth=reporting_wealth,
        gross_wealth=gross_wealth,
        gross_reporting_wealth=gross_reporting_wealth,
        parameter_uncertainty=parameter_summary,
    )
