"""Validated bridge to the native tax-neutral portfolio ledger."""

from __future__ import annotations

import ctypes
import os

import numpy as np

from mc_quadrants.decumulation import DecumulationPlan
from mc_quadrants.native import _load_library, _pointer


class _NeutralPortfolioConfig(ctypes.Structure):
    _fields_ = (
        [(name, ctypes.c_int) for name in (
            "periods paths assets threads mode simple_returns rebalance_frequency "
            "contribution_mode guardrail_policy skip_inflation_after_loss"
        ).split()]
        + [(name, ctypes.c_double) for name in (
            "initial_value contribution leverage financing_growth maintenance_margin "
            "cost_rate upper_guardrail lower_guardrail adjustment floor ceiling"
        ).split()]
        + [(name, ctypes.c_void_p) for name in (
            "returns weights fee_logs cost_paths financing_paths cpi phase_ids "
            "phase_amounts due_factors one_times reviews"
        ).split()]
    )


def simulate_neutral_portfolios_native(
    returns: np.ndarray,
    weights: np.ndarray,
    fee_logs: np.ndarray,
    *,
    initial_value: float,
    return_kind: str,
    rebalance_frequency: int | None,
    contribution: float,
    contribution_allocation: str,
    transaction_cost_bps: float,
    transaction_cost_rate_paths: np.ndarray | None,
    leverage_multiple: float,
    financing_rate: float,
    financing_rate_paths: np.ndarray | None,
    maintenance_margin: float,
    plan: DecumulationPlan,
    cpi: np.ndarray,
    safe_rate: float,
    workers: int,
) -> dict | None:
    if os.getenv("MC_DISABLE_NATIVE_SIM", "").strip().lower() in {"1", "true", "yes", "on"}:
        return None
    library = _load_library()
    if library is None:
        return None
    values = np.ascontiguousarray(returns, dtype=np.float64)
    if values.ndim != 3 or min(values.shape) <= 0 or not np.isfinite(values).all():
        raise ValueError("returns must be a non-empty, finite periods x paths x assets matrix.")
    periods, paths, assets = values.shape
    weights = np.ascontiguousarray(weights, dtype=np.float64)
    fees = np.ascontiguousarray(fee_logs, dtype=np.float64)
    if weights.shape != (assets,) or fees.shape != (assets,):
        raise ValueError("weights and fee_logs must match the asset count.")
    if not np.isfinite(weights).all() or not np.isfinite(fees).all():
        raise ValueError("weights and fee_logs must be finite.")
    if safe_rate < 0:
        raise ValueError("safe_rate must be non-negative.")
    cost_paths = None
    if transaction_cost_rate_paths is not None:
        cost_paths = np.ascontiguousarray(transaction_cost_rate_paths, dtype=float)
        if cost_paths.shape != (periods, paths) or not np.isfinite(cost_paths).all() or (cost_paths < 0).any():
            raise ValueError("transaction_cost_rate_paths must be finite, non-negative and match wealth paths.")
    finance = None
    effective_financing = float(financing_rate)
    if financing_rate_paths is not None:
        rates = np.asarray(financing_rate_paths, dtype=float)
        if rates.shape != (periods, paths):
            raise ValueError("financing_rate_paths must have shape (periods, paths).")
        if not np.isfinite(rates).all() or (rates <= -1.0).any():
            raise ValueError("financing_rate_paths must contain finite annual rates above -100%.")
        finance = np.ascontiguousarray(np.power(1.0 + rates, 1.0 / 12.0))
        effective_financing = float(rates.mean())
    cpi_values = None
    if plan.active and not plan.legacy_nominal:
        cpi_values = np.ascontiguousarray(cpi, dtype=float)
        if cpi_values.shape != (periods, paths) or not np.isfinite(cpi_values).all() or (cpi_values <= 0).any():
            raise ValueError("withdrawal_cpi must be positive, finite and match wealth paths.")
    phases = np.full(periods, -1, dtype=np.int32)
    amounts = np.zeros(periods)
    due = np.zeros(periods)
    reviews = np.zeros(periods, dtype=np.uint8)
    one_times = np.zeros(periods)
    if plan.active:
        for index, phase in enumerate(plan.phases):
            section = slice(phase.start_month - 1, phase.end_month)
            phases[section] = index
            amounts[section] = phase.annual_amount(safe_rate=safe_rate, initial_value=initial_value, mode=plan.mode)
            months = np.arange(phase.end_month - phase.start_month + 1)
            due[section] = np.where(months % phase.frequency_months == 0, phase.frequency_months / 12.0, 0.0)
            reviews[section] = months % plan.guardrails.review_months == 0
        for expense in plan.one_time_expenses:
            one_times[expense.month - 1] += expense.real_amount
    leveraged = (
        not np.isclose(leverage_multiple, 1.0) or not np.isclose(financing_rate, 0.0)
        or not np.isclose(maintenance_margin, 0.0)
    )
    mode = 0 if rebalance_frequency is None else 2 if leveraged else 1
    guardrails = plan.guardrails
    config = _NeutralPortfolioConfig(
        periods, paths, assets, max(1, int(workers)), mode, int(return_kind == "simple"),
        int(rebalance_frequency or 0), int(contribution_allocation == "underweight_first"),
        int(plan.policy == "guyton_klinger"), int(guardrails.skip_inflation_after_negative_real_return),
        initial_value, contribution, leverage_multiple if mode == 2 else 1.0,
        (1.0 + financing_rate) ** (1.0 / 12.0), maintenance_margin,
        transaction_cost_bps / 10_000.0, guardrails.upper_guardrail, guardrails.lower_guardrail,
        guardrails.adjustment, guardrails.floor, guardrails.ceiling,
        *[_pointer(array) for array in (
            values, weights, fees, cost_paths, finance, cpi_values, phases, amounts, due, one_times, reviews,
        )],
    )
    wealth = np.empty((periods, paths))
    requested = np.empty_like(wealth)
    funded = np.empty_like(wealth)
    events = np.empty(wealth.shape, dtype=np.int8)
    costs = np.empty(paths)
    margin = np.empty(paths, dtype=np.uint8)
    function = library.mc_simulate_neutral_portfolios
    function.restype = ctypes.c_int
    function.argtypes = [ctypes.POINTER(_NeutralPortfolioConfig), *([ctypes.c_void_p] * 6)]
    status = function(ctypes.byref(config), *[_pointer(array) for array in (wealth, requested, funded, events, costs, margin)])
    if status == 2:
        raise ValueError("Simple returns must be greater than -100%; asset growth must be finite.")
    if status == 3:
        raise ValueError("Portfolio wealth contains non-finite values.")
    if status:
        raise RuntimeError(f"Native neutral portfolio ledger failed with status {status}.")
    return {
        "wealth": wealth, "withdrawal_requested": requested, "withdrawal_funded": funded,
        "guardrail_events": events, "transaction_cost_total": float(costs.sum()),
        "margin_calls": int(margin.sum()), "effective_financing_rate": effective_financing,
    }
