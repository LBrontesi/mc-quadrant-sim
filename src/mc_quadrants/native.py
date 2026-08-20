from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np

from mc_quadrants.decumulation import DecumulationPlan

NATIVE_ABI_VERSION = 4
NATIVE_TAX_STAT_NAMES = (
    "capital_gains_tax_total",
    "investment_income_tax_total",
    "foreign_withholding_tax_total",
    "financial_transaction_tax_total",
    "wealth_tax_total",
    "stamp_duty_total",
    "ivafe_total",
    "terminal_liquidation_tax_total",
    "taxes_paid_total",
    "realized_gains_total",
    "realized_losses_total",
    "loss_carryforward_total",
    "expired_losses_total",
    "transaction_cost_total",
)
NATIVE_YEAR_STAT_NAMES = (
    "capital_gains_tax",
    "managed_result_tax",
    "deferred_tax_payment",
    "expired_losses",
    "financial_transaction_tax",
    "stamp_duty",
    "ivafe",
    "terminal_liquidation_tax",
    "gross_sales_for_spending",
    "net_spending",
)


class _ParametricPortfolioConfig(ctypes.Structure):
    _fields_ = [
        ("periods", ctypes.c_int),
        ("paths", ctypes.c_int),
        ("assets", ctypes.c_int),
        ("states", ctypes.c_int),
        ("macro_dimensions", ctypes.c_int),
        ("requested_threads", ctypes.c_int),
        ("regimes", ctypes.c_void_p),
        ("means", ctypes.c_void_p),
        ("gaussian_correlation_cholesky", ctypes.c_void_p),
        ("gaussian_correlations", ctypes.c_void_p),
        ("volatilities", ctypes.c_void_p),
        ("tail_indexes", ctypes.c_void_p),
        ("temperings", ctypes.c_void_p),
        ("skewness", ctypes.c_void_p),
        ("gaussian_scales", ctypes.c_void_p),
        ("macro_shocks", ctypes.c_void_p),
        ("macro_betas", ctypes.c_void_p),
        ("seed", ctypes.c_uint64),
        ("garch", ctypes.c_int),
        ("garch_alpha", ctypes.c_double),
        ("garch_beta", ctypes.c_double),
        ("dynamic_correlation", ctypes.c_int),
        ("dcc_alpha", ctypes.c_double),
        ("dcc_beta", ctypes.c_double),
        ("dcc_asymmetry", ctypes.c_double),
        ("monthly_fee_log", ctypes.c_void_p),
        ("simple_returns", ctypes.c_int),
    ]


class _ItalianPortfolioConfig(ctypes.Structure):
    _fields_ = [
        ("weights", ctypes.c_void_p),
        ("initial_value", ctypes.c_double),
        ("rebalance_frequency", ctypes.c_int),
        ("transaction_cost_rate_paths", ctypes.c_void_p),
        ("default_transaction_cost_rate", ctypes.c_double),
        ("contribution", ctypes.c_double),
        ("contribution_mode", ctypes.c_int),
        ("withdrawal", ctypes.c_double),
        ("withdrawal_start_period", ctypes.c_int),
        ("tax_regime", ctypes.c_int),
        ("taxable_fraction", ctypes.c_void_p),
        ("offsettable", ctypes.c_void_p),
        ("ftt_rates", ctypes.c_void_p),
        ("stamp_mask", ctypes.c_void_p),
        ("ivafe_mask", ctypes.c_void_p),
        ("annual_wealth_tax", ctypes.c_double),
        ("terminal_liquidation", ctypes.c_int),
        ("wrapper_benchmark", ctypes.c_int),
        ("year_slots", ctypes.c_void_p),
        ("year_count", ctypes.c_int),
    ]

_LIBRARY: ctypes.CDLL | None | bool = None


def _load_library() -> ctypes.CDLL | None:
    global _LIBRARY
    if _LIBRARY is False:
        return None
    if isinstance(_LIBRARY, ctypes.CDLL):
        return _LIBRARY

    package_dir = Path(__file__).resolve().parent
    configured = os.getenv("MC_NATIVE_SIM_LIB")
    candidates = [Path(configured)] if configured else []
    candidates.extend(
        package_dir / name
        for name in ("_native_sim.so", "_native_sim.dylib", "_native_sim.dll")
    )
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            library = ctypes.CDLL(str(candidate))
            version = library.mc_native_version
            version.restype = ctypes.c_int
            version.argtypes = []
            if int(version()) != NATIVE_ABI_VERSION:
                continue
            function = library.mc_simulate_parametric
            function.restype = ctypes.c_int
            function.argtypes = [
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_uint64,
                ctypes.c_int,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_int,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_int,
                ctypes.c_void_p,
            ]
            subordinator_function = library.mc_sample_mnts_subordinators
            subordinator_function.restype = ctypes.c_int
            subordinator_function.argtypes = [
                ctypes.c_int,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_uint64,
                ctypes.c_void_p,
            ]
            tax_function = library.mc_simulate_italian_portfolios
            tax_function.restype = ctypes.c_int
            tax_function.argtypes = [
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_double,
                ctypes.c_int,
                ctypes.c_void_p,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_int,
                ctypes.c_double,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_double,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_double,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
            ]
            fused_function = library.mc_simulate_parametric_italian_portfolios
            fused_function.restype = ctypes.c_int
            fused_function.argtypes = [
                ctypes.POINTER(_ParametricPortfolioConfig),
                ctypes.POINTER(_ItalianPortfolioConfig),
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
            ]
            _LIBRARY = library
            return library
        except (AttributeError, OSError):
            continue
    _LIBRARY = False
    return None


def native_available() -> bool:
    """Return whether the optional compiled simulation backend is loadable."""

    return _load_library() is not None


def sample_mnts_subordinators_native(
    samples: int,
    tail_index: float,
    tempering: float,
    random_seed: int = 0,
) -> np.ndarray:
    """Draw standardized MNTS subordinators with the exact native sampler."""

    library = _load_library()
    if library is None:
        raise RuntimeError("The native MNTS backend is unavailable.")
    output = np.empty(int(samples), dtype=np.float64)
    status = library.mc_sample_mnts_subordinators(
        int(samples),
        float(tail_index),
        float(tempering),
        ctypes.c_uint64(int(random_seed) & ((1 << 64) - 1)),
        _pointer(output),
    )
    if status != 0:
        raise RuntimeError(f"Native MNTS subordinator sampler failed with status {status}.")
    return output


def _pointer(values: np.ndarray | None) -> ctypes.c_void_p:
    return ctypes.c_void_p(0 if values is None else values.ctypes.data)


def simulate_parametric_native(
    regime_codes: np.ndarray,
    means: np.ndarray,
    gaussian_correlation_cholesky: np.ndarray,
    gaussian_correlations: np.ndarray,
    volatilities: np.ndarray,
    tail_indexes: np.ndarray,
    temperings: np.ndarray,
    skewness: np.ndarray,
    gaussian_scales: np.ndarray,
    random_seed: int,
    garch: bool,
    garch_alpha: float,
    garch_beta: float,
    dynamic_correlation: bool,
    dcc_alpha: float,
    dcc_beta: float,
    dcc_asymmetry: float,
    macro_shocks: np.ndarray | None = None,
    macro_betas: np.ndarray | None = None,
    workers: int = 1,
) -> np.ndarray | None:
    """Run the parametric return kernel in C++, or return ``None`` if unavailable."""

    if os.getenv("MC_DISABLE_NATIVE_SIM", "").strip().lower() in {"1", "true", "yes", "on"}:
        return None
    library = _load_library()
    if library is None:
        return None
    regime_codes = np.asarray(regime_codes)
    means = np.asarray(means)
    if regime_codes.ndim != 2 or means.ndim != 2:
        raise ValueError("regime_codes and means must be two-dimensional.")
    periods, paths = regime_codes.shape
    states, assets = means.shape
    if states > 256:
        raise ValueError("The native backend supports at most 256 regimes.")
    if regime_codes.size and (
        np.min(regime_codes) < 0 or np.max(regime_codes) >= states
    ):
        raise ValueError("regime_codes contains an invalid state index.")
    expected_matrix_shape = (states, assets, assets)
    expected_vector_shape = (states, assets)
    if np.shape(gaussian_correlation_cholesky) != expected_matrix_shape:
        raise ValueError("gaussian_correlation_cholesky has an invalid shape.")
    if np.shape(gaussian_correlations) != expected_matrix_shape:
        raise ValueError("gaussian_correlations has an invalid shape.")
    if np.shape(volatilities) != expected_vector_shape:
        raise ValueError("volatilities has an invalid shape.")
    if np.shape(skewness) != expected_vector_shape or np.shape(gaussian_scales) != expected_vector_shape:
        raise ValueError("MNTS skewness and Gaussian scales have an invalid shape.")
    if np.shape(tail_indexes) != (states,) or np.shape(temperings) != (states,):
        raise ValueError("MNTS tail parameters have an invalid shape.")
    if (macro_shocks is None) != (macro_betas is None):
        raise ValueError("macro_shocks and macro_betas must be supplied together.")
    if macro_shocks is not None:
        if np.ndim(macro_shocks) != 3 or np.shape(macro_shocks)[:2] != (periods, paths):
            raise ValueError("macro_shocks has an invalid shape.")
        if np.shape(macro_betas) != (np.shape(macro_shocks)[2], assets):
            raise ValueError("macro_betas has an invalid shape.")
    codes = np.ascontiguousarray(regime_codes, dtype=np.uint8)
    means = np.ascontiguousarray(means, dtype=np.float64)
    gaussian_correlation_cholesky = np.ascontiguousarray(
        gaussian_correlation_cholesky,
        dtype=np.float64,
    )
    gaussian_correlations = np.ascontiguousarray(gaussian_correlations, dtype=np.float64)
    volatilities = np.ascontiguousarray(volatilities, dtype=np.float64)
    tail_indexes = np.ascontiguousarray(tail_indexes, dtype=np.float64)
    temperings = np.ascontiguousarray(temperings, dtype=np.float64)
    skewness = np.ascontiguousarray(skewness, dtype=np.float64)
    gaussian_scales = np.ascontiguousarray(gaussian_scales, dtype=np.float64)
    macro_shocks = (
        np.ascontiguousarray(macro_shocks, dtype=np.float64)
        if macro_shocks is not None
        else None
    )
    macro_betas = (
        np.ascontiguousarray(macro_betas, dtype=np.float64)
        if macro_betas is not None
        else None
    )
    macro_dimensions = 0 if macro_shocks is None else int(macro_shocks.shape[2])
    output = np.empty((periods, paths, assets), dtype=np.float64)
    status = library.mc_simulate_parametric(
        periods,
        paths,
        assets,
        states,
        macro_dimensions,
        _pointer(codes),
        _pointer(means),
        _pointer(gaussian_correlation_cholesky),
        _pointer(gaussian_correlations),
        _pointer(volatilities),
        _pointer(tail_indexes),
        _pointer(temperings),
        _pointer(skewness),
        _pointer(gaussian_scales),
        _pointer(macro_shocks),
        _pointer(macro_betas),
        ctypes.c_uint64(int(random_seed) & ((1 << 64) - 1)),
        int(garch),
        float(garch_alpha),
        float(garch_beta),
        int(dynamic_correlation),
        float(dcc_alpha),
        float(dcc_beta),
        float(dcc_asymmetry),
        max(1, int(workers)),
        _pointer(output),
    )
    if status != 0:
        raise RuntimeError(f"Native simulation kernel failed with status {status}.")
    return output


def simulate_italian_portfolios_native(
    asset_growth: np.ndarray,
    target_weights: np.ndarray,
    *,
    initial_value: float,
    rebalance_frequency: int,
    transaction_cost_bps: float,
    transaction_cost_rate_paths: np.ndarray | None,
    contribution: float,
    contribution_allocation: str,
    withdrawal: float,
    withdrawal_start_period: int,
    tax_regime: str,
    taxable_fraction: np.ndarray,
    gains_offsettable: np.ndarray,
    financial_transaction_tax_rate: np.ndarray,
    stamp_mask: np.ndarray,
    ivafe_mask: np.ndarray,
    annual_wealth_tax: float,
    terminal_liquidation: bool,
    wrapper_benchmark: bool,
    year_slots: np.ndarray,
    decumulation: DecumulationPlan | None = None,
    withdrawal_cpi: np.ndarray | None = None,
    safe_withdrawal_rate: float = 0.0,
    workers: int = 1,
) -> dict[str, object] | None:
    """Run the fused gross/Italian-tax ledger, or return ``None`` as fallback."""

    if os.getenv("MC_DISABLE_NATIVE_SIM", "").strip().lower() in {"1", "true", "yes", "on"}:
        return None
    library = _load_library()
    if library is None:
        return None

    growth = np.ascontiguousarray(asset_growth, dtype=np.float64)
    weights = np.ascontiguousarray(target_weights, dtype=np.float64)
    if growth.ndim != 3:
        raise ValueError("asset_growth must be three-dimensional.")
    periods, paths, assets = growth.shape
    if weights.shape != (assets,):
        raise ValueError("target_weights has an invalid shape.")
    cost_paths = None
    if transaction_cost_rate_paths is not None:
        cost_paths = np.ascontiguousarray(transaction_cost_rate_paths, dtype=np.float64)
        if cost_paths.shape != (periods, paths):
            raise ValueError("transaction_cost_rate_paths has an invalid shape.")
    taxable = np.ascontiguousarray(taxable_fraction, dtype=np.float64)
    offsettable = np.ascontiguousarray(gains_offsettable, dtype=np.uint8)
    ftt = np.ascontiguousarray(financial_transaction_tax_rate, dtype=np.float64)
    stamp = np.ascontiguousarray(stamp_mask, dtype=np.uint8)
    ivafe = np.ascontiguousarray(ivafe_mask, dtype=np.uint8)
    for name, values in {
        "taxable_fraction": taxable,
        "gains_offsettable": offsettable,
        "financial_transaction_tax_rate": ftt,
        "stamp_mask": stamp,
        "ivafe_mask": ivafe,
    }.items():
        if values.shape != (assets,):
            raise ValueError(f"{name} has an invalid shape.")
    slots = np.ascontiguousarray(year_slots, dtype=np.int32)
    if slots.shape != (periods,) or (slots < 0).any():
        raise ValueError("year_slots has an invalid shape or value.")
    year_count = int(slots.max()) + 1
    regime_codes = {
        "italy_administered": 0,
        "italy_declarative": 1,
        "italy_managed": 2,
    }
    try:
        regime_code = regime_codes[str(tax_regime).strip().lower()]
    except KeyError as error:
        raise ValueError(f"Unknown Italian tax regime '{tax_regime}'.") from error
    allocation_code = 1 if contribution_allocation == "underweight_first" else 0
    advanced = bool(
        decumulation is not None
        and decumulation.active
        and not decumulation.legacy_nominal
    )
    phase_starts = phase_ends = phase_frequencies = None
    phase_amounts = phase_multipliers = None
    one_times = cpi = None
    requested_spending = funded_spending = guardrail_events = None
    if advanced:
        phase_starts = np.ascontiguousarray(
            [phase.start_month for phase in decumulation.phases], dtype=np.int32
        )
        phase_ends = np.ascontiguousarray(
            [phase.end_month for phase in decumulation.phases], dtype=np.int32
        )
        phase_frequencies = np.ascontiguousarray(
            [phase.frequency_months for phase in decumulation.phases], dtype=np.int32
        )
        phase_amounts = np.ascontiguousarray(
            [phase.annual_real_amount for phase in decumulation.phases], dtype=np.float64
        )
        phase_multipliers = np.ascontiguousarray(
            [phase.spending_multiplier for phase in decumulation.phases], dtype=np.float64
        )
        one_times = np.zeros(periods, dtype=np.float64)
        for expense in decumulation.one_time_expenses:
            one_times[expense.month - 1] += expense.real_amount
        cpi = np.ascontiguousarray(withdrawal_cpi, dtype=np.float64)
        if cpi.shape != (periods, paths):
            raise ValueError("withdrawal_cpi must have shape (periods, paths).")
        requested_spending = np.zeros((periods, paths), dtype=np.float64)
        funded_spending = np.zeros((periods, paths), dtype=np.float64)
        guardrail_events = np.zeros((periods, paths), dtype=np.int8)

    gross = np.empty((periods, paths), dtype=np.float64)
    diy = np.empty((periods, paths), dtype=np.float64)
    wrapper_terminal = np.empty(paths, dtype=np.float64) if wrapper_benchmark else None
    wrapper_annualized = np.empty(paths, dtype=np.float64) if wrapper_benchmark else None
    stats = np.zeros((len(NATIVE_TAX_STAT_NAMES), paths), dtype=np.float64)
    gross_costs = np.zeros(paths, dtype=np.float64)
    year_stats = np.zeros((year_count, len(NATIVE_YEAR_STAT_NAMES)), dtype=np.float64)
    status = library.mc_simulate_italian_portfolios(
        periods,
        paths,
        assets,
        max(1, int(workers)),
        _pointer(growth),
        _pointer(weights),
        float(initial_value),
        int(rebalance_frequency),
        _pointer(cost_paths),
        float(transaction_cost_bps) / 10_000.0,
        float(contribution),
        allocation_code,
        float(withdrawal),
        int(withdrawal_start_period),
        int(advanced),
        len(decumulation.phases) if advanced and decumulation is not None else 0,
        _pointer(phase_starts),
        _pointer(phase_ends),
        _pointer(phase_frequencies),
        _pointer(phase_amounts),
        _pointer(phase_multipliers),
        int(bool(advanced and decumulation is not None and decumulation.mode == "safe_rate")),
        float(safe_withdrawal_rate),
        _pointer(one_times),
        _pointer(cpi),
        int(bool(advanced and decumulation is not None and decumulation.policy == "guyton_klinger")),
        int(decumulation.guardrails.review_months if advanced and decumulation is not None else 12),
        float(decumulation.guardrails.upper_guardrail if advanced and decumulation is not None else 1.20),
        float(decumulation.guardrails.lower_guardrail if advanced and decumulation is not None else 0.80),
        float(decumulation.guardrails.adjustment if advanced and decumulation is not None else 0.10),
        float(decumulation.guardrails.floor if advanced and decumulation is not None else 0.70),
        float(decumulation.guardrails.ceiling if advanced and decumulation is not None else 1.30),
        int(bool(
            decumulation.guardrails.skip_inflation_after_negative_real_return
            if advanced and decumulation is not None
            else True
        )),
        regime_code,
        _pointer(taxable),
        _pointer(offsettable),
        _pointer(ftt),
        _pointer(stamp),
        _pointer(ivafe),
        float(annual_wealth_tax),
        int(bool(terminal_liquidation)),
        int(bool(wrapper_benchmark)),
        _pointer(slots),
        year_count,
        _pointer(gross),
        _pointer(diy),
        _pointer(wrapper_terminal),
        _pointer(wrapper_annualized),
        _pointer(stats),
        _pointer(gross_costs),
        _pointer(year_stats),
        _pointer(requested_spending),
        _pointer(funded_spending),
        _pointer(guardrail_events),
    )
    if status != 0:
        raise RuntimeError(f"Native Italian tax ledger failed with status {status}.")
    return {
        "gross_wealth": gross,
        "wealth": diy,
        "wrapper_terminal_values": wrapper_terminal,
        "wrapper_annualized_returns": wrapper_annualized,
        "tax_stats": dict(zip(NATIVE_TAX_STAT_NAMES, stats, strict=True)),
        "gross_transaction_cost_total": float(gross_costs.sum()),
        "year_stats": year_stats,
        "withdrawal_requested": requested_spending,
        "withdrawal_funded": funded_spending,
        "guardrail_events": guardrail_events,
    }


def simulate_parametric_italian_portfolios_native(
    regime_codes: np.ndarray,
    means: np.ndarray,
    gaussian_correlation_cholesky: np.ndarray,
    gaussian_correlations: np.ndarray,
    volatilities: np.ndarray,
    tail_indexes: np.ndarray,
    temperings: np.ndarray,
    skewness: np.ndarray,
    gaussian_scales: np.ndarray,
    random_seed: int,
    garch: bool,
    garch_alpha: float,
    garch_beta: float,
    dynamic_correlation: bool,
    dcc_alpha: float,
    dcc_beta: float,
    dcc_asymmetry: float,
    monthly_fee_log: np.ndarray,
    return_kind: str,
    target_weights: np.ndarray,
    *,
    initial_value: float,
    rebalance_frequency: int,
    transaction_cost_bps: float,
    transaction_cost_rate_paths: np.ndarray | None,
    contribution: float,
    contribution_allocation: str,
    withdrawal: float,
    withdrawal_start_period: int,
    tax_regime: str,
    taxable_fraction: np.ndarray,
    gains_offsettable: np.ndarray,
    financial_transaction_tax_rate: np.ndarray,
    stamp_mask: np.ndarray,
    ivafe_mask: np.ndarray,
    annual_wealth_tax: float,
    terminal_liquidation: bool,
    wrapper_benchmark: bool,
    year_slots: np.ndarray,
    macro_shocks: np.ndarray | None = None,
    macro_betas: np.ndarray | None = None,
    workers: int = 1,
) -> dict[str, object] | None:
    """Generate parametric returns and update all ledgers without a return cube."""

    disabled_values = {"1", "true", "yes", "on"}
    if os.getenv("MC_DISABLE_NATIVE_SIM", "").strip().lower() in disabled_values:
        return None
    if os.getenv("MC_DISABLE_NATIVE_FUSED", "").strip().lower() in disabled_values:
        return None
    library = _load_library()
    if library is None:
        return None

    codes = np.ascontiguousarray(regime_codes, dtype=np.uint8)
    means = np.ascontiguousarray(means, dtype=np.float64)
    if codes.ndim != 2 or means.ndim != 2:
        raise ValueError("regime_codes and means must be two-dimensional.")
    periods, paths = codes.shape
    states, assets = means.shape
    if states > 256:
        raise ValueError("The native backend supports at most 256 regimes.")
    if codes.size and (np.min(codes) < 0 or np.max(codes) >= states):
        raise ValueError("regime_codes contains an invalid state index.")
    if return_kind not in {"log", "simple"}:
        raise ValueError("return_kind must be 'log' or 'simple'.")
    matrix_shape = (states, assets, assets)
    vector_shape = (states, assets)
    matrices = {
        "gaussian_correlation_cholesky": gaussian_correlation_cholesky,
        "gaussian_correlations": gaussian_correlations,
    }
    for name, values in matrices.items():
        if np.shape(values) != matrix_shape:
            raise ValueError(f"{name} has an invalid shape.")
    if np.shape(volatilities) != vector_shape:
        raise ValueError("volatilities has an invalid shape.")
    if np.shape(skewness) != vector_shape or np.shape(gaussian_scales) != vector_shape:
        raise ValueError("MNTS skewness and Gaussian scales have an invalid shape.")
    if np.shape(tail_indexes) != (states,) or np.shape(temperings) != (states,):
        raise ValueError("MNTS tail parameters have an invalid shape.")
    correlation = np.ascontiguousarray(gaussian_correlation_cholesky, dtype=np.float64)
    base = np.ascontiguousarray(gaussian_correlations, dtype=np.float64)
    volatility = np.ascontiguousarray(volatilities, dtype=np.float64)
    tail_indexes = np.ascontiguousarray(tail_indexes, dtype=np.float64)
    temperings = np.ascontiguousarray(temperings, dtype=np.float64)
    skewness = np.ascontiguousarray(skewness, dtype=np.float64)
    gaussian_scales = np.ascontiguousarray(gaussian_scales, dtype=np.float64)
    if (macro_shocks is None) != (macro_betas is None):
        raise ValueError("macro_shocks and macro_betas must be supplied together.")
    shocks = None
    betas = None
    macro_dimensions = 0
    if macro_shocks is not None:
        shocks = np.ascontiguousarray(macro_shocks, dtype=np.float64)
        betas = np.ascontiguousarray(macro_betas, dtype=np.float64)
        if shocks.ndim != 3 or shocks.shape[:2] != (periods, paths):
            raise ValueError("macro_shocks has an invalid shape.")
        macro_dimensions = int(shocks.shape[2])
        if betas.shape != (macro_dimensions, assets):
            raise ValueError("macro_betas has an invalid shape.")
    fees = np.ascontiguousarray(monthly_fee_log, dtype=np.float64)
    weights = np.ascontiguousarray(target_weights, dtype=np.float64)
    if fees.shape != (assets,) or weights.shape != (assets,):
        raise ValueError("Fees and target weights must match the asset count.")
    cost_paths = None
    if transaction_cost_rate_paths is not None:
        cost_paths = np.ascontiguousarray(transaction_cost_rate_paths, dtype=np.float64)
        if cost_paths.shape != (periods, paths):
            raise ValueError("transaction_cost_rate_paths has an invalid shape.")
    taxable = np.ascontiguousarray(taxable_fraction, dtype=np.float64)
    offsettable = np.ascontiguousarray(gains_offsettable, dtype=np.uint8)
    ftt = np.ascontiguousarray(financial_transaction_tax_rate, dtype=np.float64)
    stamp = np.ascontiguousarray(stamp_mask, dtype=np.uint8)
    ivafe = np.ascontiguousarray(ivafe_mask, dtype=np.uint8)
    for name, values in {
        "taxable_fraction": taxable,
        "gains_offsettable": offsettable,
        "financial_transaction_tax_rate": ftt,
        "stamp_mask": stamp,
        "ivafe_mask": ivafe,
    }.items():
        if values.shape != (assets,):
            raise ValueError(f"{name} has an invalid shape.")
    slots = np.ascontiguousarray(year_slots, dtype=np.int32)
    if slots.shape != (periods,) or (slots < 0).any():
        raise ValueError("year_slots has an invalid shape or value.")
    year_count = int(slots.max()) + 1
    regime_code = {
        "italy_administered": 0,
        "italy_declarative": 1,
        "italy_managed": 2,
    }[str(tax_regime).strip().lower()]

    parametric_config = _ParametricPortfolioConfig(
        periods,
        paths,
        assets,
        states,
        macro_dimensions,
        max(1, int(workers)),
        _pointer(codes),
        _pointer(means),
        _pointer(correlation),
        _pointer(base),
        _pointer(volatility),
        _pointer(tail_indexes),
        _pointer(temperings),
        _pointer(skewness),
        _pointer(gaussian_scales),
        _pointer(shocks),
        _pointer(betas),
        ctypes.c_uint64(int(random_seed) & ((1 << 64) - 1)),
        int(garch),
        float(garch_alpha),
        float(garch_beta),
        int(dynamic_correlation),
        float(dcc_alpha),
        float(dcc_beta),
        float(dcc_asymmetry),
        _pointer(fees),
        int(return_kind == "simple"),
    )
    tax_config = _ItalianPortfolioConfig(
        _pointer(weights),
        float(initial_value),
        int(rebalance_frequency),
        _pointer(cost_paths),
        float(transaction_cost_bps) / 10_000.0,
        float(contribution),
        int(contribution_allocation == "underweight_first"),
        float(withdrawal),
        int(withdrawal_start_period),
        regime_code,
        _pointer(taxable),
        _pointer(offsettable),
        _pointer(ftt),
        _pointer(stamp),
        _pointer(ivafe),
        float(annual_wealth_tax),
        int(bool(terminal_liquidation)),
        int(bool(wrapper_benchmark)),
        _pointer(slots),
        year_count,
    )
    gross = np.empty((periods, paths), dtype=np.float64)
    diy = np.empty((periods, paths), dtype=np.float64)
    wrapper_terminal = np.empty(paths, dtype=np.float64) if wrapper_benchmark else None
    wrapper_annualized = np.empty(paths, dtype=np.float64) if wrapper_benchmark else None
    stats = np.zeros((len(NATIVE_TAX_STAT_NAMES), paths), dtype=np.float64)
    gross_costs = np.zeros(paths, dtype=np.float64)
    year_stats = np.zeros((year_count, len(NATIVE_YEAR_STAT_NAMES)), dtype=np.float64)
    status = library.mc_simulate_parametric_italian_portfolios(
        ctypes.byref(parametric_config),
        ctypes.byref(tax_config),
        _pointer(gross),
        _pointer(diy),
        _pointer(wrapper_terminal),
        _pointer(wrapper_annualized),
        _pointer(stats),
        _pointer(gross_costs),
        _pointer(year_stats),
    )
    if status != 0:
        raise RuntimeError(f"Fused native portfolio kernel failed with status {status}.")
    return {
        "gross_wealth": gross,
        "wealth": diy,
        "wrapper_terminal_values": wrapper_terminal,
        "wrapper_annualized_returns": wrapper_annualized,
        "tax_stats": dict(zip(NATIVE_TAX_STAT_NAMES, stats, strict=True)),
        "gross_transaction_cost_total": float(gross_costs.sum()),
        "year_stats": year_stats,
    }
