from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import get_context
from typing import Any

import numpy as np
import pandas as pd

from mc_quadrants.calibration import CorrelationOverrides, calibrate_quadrant_model
from mc_quadrants.data import convert_returns_to_base_currency
from mc_quadrants.diagnostics import (
    CalibrationDiagnostics,
    build_calibration_diagnostics,
    build_hmm_diagnostics,
)
from mc_quadrants.hmm import fit_hmm_model
from mc_quadrants.regimes import classify_persistent_quadrants
from mc_quadrants.simulation import (
    inflation_adjust_wealth,
    simulate_portfolio_paths,
    simulate_returns,
    summarize_wealth_risk,
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
    parameter_uncertainty: pd.DataFrame | None = None


_CHUNK_WORKER_STATE: dict[str, Any] = {}


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


def _run_chunk(
    args: tuple[int, int, int],
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray | None]:
    """Simulate one path chunk in a worker process.

    Returns ``(start, wealth_values, regime_codes)`` for the chunk so the
    caller can scatter the results into the preallocated arrays. Each chunk
    draws its own RNG from ``random_seed + chunk_index``, so results are
    identical whether chunks run sequentially or in parallel.
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
        degrees_of_freedom=state["degrees_of_freedom"],
        block_size=state["block_size"],
        transition_concentration=state["transition_concentration"],
        duration_model=state["duration_model"],
        min_regime_duration=state["min_regime_duration"],
        garch=state["garch"],
        garch_alpha=state["garch_alpha"],
        garch_beta=state["garch_beta"],
        joint_macro=state["joint_macro"],
        macro_transition_weight=state["macro_transition_weight"],
        dynamic_correlation=state["dynamic_correlation"],
        dcc_alpha=state["dcc_alpha"],
        dcc_beta=state["dcc_beta"],
        dcc_asymmetry=state["dcc_asymmetry"],
    )
    chunk_wealth = simulate_portfolio_paths(
        chunk_result,
        weights=state["weights"],
        initial_value=state["initial_value"],
        return_kind=state["return_kind"],
        rebalance_frequency=state["rebalance_frequency"],
        transaction_cost_bps=state["transaction_cost_bps"],
        asset_expense_ratios=state["expense_ratios"],
        leverage_multiple=state["leverage_multiple"],
        financing_rate=state["financing_rate"],
        financing_inflation_sensitivity=state["financing_inflation_sensitivity"],
        state_inflation=state["state_inflation"],
        financing_rate_paths=_annual_macro_paths(
            state["model"],
            chunk_result,
            state["rate_col"],
            "rate_is_percent",
        ),
        financing_inflation_paths=_annual_macro_paths(
            state["model"],
            chunk_result,
            state["inflation_col"],
            "inflation_is_percent",
        ),
        maintenance_margin=state["maintenance_margin"],
        contribution=state["contribution"],
        withdrawal=state["withdrawal"],
    )
    wealth_values = chunk_wealth.to_numpy(dtype=float)
    regime_codes = np.empty((state["periods"], count), dtype=np.int8)
    state_codes = state["state_codes"]
    for period in range(state["periods"]):
        column = chunk_result.regimes[period]
        regime_codes[period] = [state_codes[state_name] for state_name in column]
    return start, wealth_values, regime_codes, chunk_result.macro_paths


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
    degrees_of_freedom: float,
    block_size: int,
    transition_concentration: float | None,
    duration_model: str,
    min_regime_duration: int,
    garch: bool,
    garch_alpha: float,
    garch_beta: float,
    joint_macro: bool,
    macro_transition_weight: float,
    dynamic_correlation: bool,
    dcc_alpha: float,
    dcc_beta: float,
    dcc_asymmetry: float,
    chunk_size: int | None,
    weight_series: pd.Series,
    initial_value: float,
    rebalance_frequency: int | None,
    transaction_cost_bps: float,
    expense_ratios: pd.Series,
    leverage_multiple: float,
    financing_rate: float,
    financing_inflation_sensitivity: float,
    state_inflation: Mapping[str, float] | None,
    maintenance_margin: float,
    contribution: float,
    withdrawal: float,
    return_kind: str,
    rate_col: str | None,
    inflation_col: str,
    workers: int = 1,
) -> tuple[SimulationResult, pd.DataFrame]:
    """Simulate returns and portfolio wealth, chunking the path dimension.

    ``simulate_returns`` materializes a ``(periods, paths, assets)`` array, so a
    single 100k-path run can require multiple gigabytes. This helper simulates
    ``chunk_size`` paths at a time and only keeps the accumulated wealth and
    regime paths, bounding peak memory to roughly one chunk plus the wealth
    frame. The regime arrays are concatenated so the caller sees the same
    ``SimulationResult`` shape as a one-shot run.

    With ``workers > 1`` the chunks run in a ``ProcessPoolExecutor``. Each chunk
    draws its RNG from ``random_seed + chunk_index``, so results are bit-for-bit
    identical to the sequential path while wall time drops roughly with the
    worker count (processes, not threads, so the GIL is bypassed).
    """

    def _single(paths_now: int, seed: int) -> tuple[SimulationResult, pd.DataFrame]:
        chunk_result = simulate_returns(
            model,
            periods=periods,
            paths=paths_now,
            start_state=start_state,
            random_seed=seed,
            distribution=distribution,
            degrees_of_freedom=degrees_of_freedom,
            block_size=block_size,
            transition_concentration=transition_concentration,
            duration_model=duration_model,
            min_regime_duration=min_regime_duration,
            garch=garch,
            garch_alpha=garch_alpha,
            garch_beta=garch_beta,
            joint_macro=joint_macro,
            macro_transition_weight=macro_transition_weight,
            dynamic_correlation=dynamic_correlation,
            dcc_alpha=dcc_alpha,
            dcc_beta=dcc_beta,
            dcc_asymmetry=dcc_asymmetry,
        )
        chunk_wealth = simulate_portfolio_paths(
            chunk_result,
            weights=weight_series.to_dict(),
            initial_value=initial_value,
            return_kind=return_kind,
            rebalance_frequency=rebalance_frequency,
            transaction_cost_bps=transaction_cost_bps,
            asset_expense_ratios=expense_ratios.to_dict(),
            leverage_multiple=leverage_multiple,
            financing_rate=financing_rate,
            financing_inflation_sensitivity=financing_inflation_sensitivity,
            state_inflation=state_inflation,
            financing_rate_paths=_annual_macro_paths(
                model,
                chunk_result,
                rate_col,
                "rate_is_percent",
            ),
            financing_inflation_paths=_annual_macro_paths(
                model,
                chunk_result,
                inflation_col,
                "inflation_is_percent",
            ),
            maintenance_margin=maintenance_margin,
            contribution=contribution,
            withdrawal=withdrawal,
        )
        return chunk_result, chunk_wealth

    if chunk_size is None or paths <= chunk_size:
        return _single(int(paths), int(random_seed))

    total = int(paths)
    chunk_size = max(1, int(chunk_size))
    state_codes = {state: index for index, state in enumerate(model.states)}
    regime_codes = np.empty((periods, total), dtype=np.int8)
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
    wealth = pd.DataFrame(
        np.nan,
        index=pd.RangeIndex(periods),
        columns=[f"path_{i}" for i in range(total)],
        dtype=float,
    )
    specs = _chunk_specs(total, chunk_size, random_seed)

    if workers is not None and workers > 1:
        worker_state = {
            "model": model,
            "periods": periods,
            "start_state": start_state,
            "distribution": distribution,
            "degrees_of_freedom": float(degrees_of_freedom),
            "block_size": int(block_size),
            "transition_concentration": transition_concentration,
            "duration_model": duration_model,
            "min_regime_duration": int(min_regime_duration),
            "garch": bool(garch),
            "garch_alpha": float(garch_alpha),
            "garch_beta": float(garch_beta),
            "joint_macro": bool(joint_macro),
            "macro_transition_weight": float(macro_transition_weight),
            "dynamic_correlation": bool(dynamic_correlation),
            "dcc_alpha": float(dcc_alpha),
            "dcc_beta": float(dcc_beta),
            "dcc_asymmetry": float(dcc_asymmetry),
            "weights": weight_series.to_dict(),
            "initial_value": float(initial_value),
            "return_kind": return_kind,
            "rebalance_frequency": rebalance_frequency,
            "transaction_cost_bps": float(transaction_cost_bps),
            "expense_ratios": expense_ratios.to_dict(),
            "leverage_multiple": float(leverage_multiple),
            "financing_rate": float(financing_rate),
            "financing_inflation_sensitivity": float(financing_inflation_sensitivity),
            "state_inflation": dict(state_inflation) if state_inflation else None,
            "rate_col": rate_col,
            "inflation_col": inflation_col,
            "maintenance_margin": float(maintenance_margin),
            "contribution": float(contribution),
            "withdrawal": float(withdrawal),
            "state_codes": state_codes,
        }
        try:
            with ProcessPoolExecutor(
                max_workers=int(workers),
                mp_context=get_context("spawn"),
                initializer=_init_chunk_worker,
                initargs=(worker_state,),
            ) as executor:
                for start, chunk_wealth_values, chunk_regime_codes, chunk_macro in executor.map(
                    _run_chunk, specs
                ):
                    count = chunk_wealth_values.shape[1]
                    wealth.iloc[:, start:start + count] = chunk_wealth_values
                    regime_codes[:, start:start + count] = chunk_regime_codes
                    if macro_paths is not None and chunk_macro is not None:
                        macro_paths[:, start:start + count, :] = chunk_macro
        except (NotImplementedError, PermissionError):
            for start, count, seed in specs:
                chunk_result, chunk_wealth = _single(count, seed)
                wealth.iloc[:, start:start + count] = chunk_wealth.to_numpy(dtype=float)
                for period in range(periods):
                    column = chunk_result.regimes[period]
                    regime_codes[period, start:start + count] = [
                        state_codes[state] for state in column
                    ]
                if macro_paths is not None and chunk_result.macro_paths is not None:
                    macro_paths[:, start:start + count, :] = chunk_result.macro_paths
    else:
        for start, count, seed in specs:
            chunk_result, chunk_wealth = _single(count, seed)
            wealth.iloc[:, start:start + count] = chunk_wealth.to_numpy(dtype=float)
            for period in range(periods):
                column = chunk_result.regimes[period]
                regime_codes[period, start:start + count] = [state_codes[state] for state in column]
            if macro_paths is not None and chunk_result.macro_paths is not None:
                macro_paths[:, start:start + count, :] = chunk_result.macro_paths

    combined = SimulationResult(
        returns=np.empty((periods, 0, len(model.assets)), dtype=float),
        regimes=regime_codes,
        assets=model.assets,
        states=model.states.copy(),
        frequency=model.frequency,
        distribution=distribution,
        degrees_of_freedom=(float(degrees_of_freedom) if distribution == "student_t" else None),
        transition_concentration=transition_concentration,
        macro_paths=macro_paths,
        macro_columns=macro_columns,
    )
    return combined, wealth


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
    distribution: str = "normal",
    degrees_of_freedom: float = 5.0,
    block_size: int = 3,
    transition_uncertainty: float = 0.0,
    rebalance_frequency: int | None = None,
    transaction_cost_bps: float = 0.0,
    asset_expense_ratios: Mapping[str, float] | None = None,
    leverage_multiple: float = 1.0,
    financing_rate: float = 0.0,
    financing_inflation_sensitivity: float = 0.0,
    maintenance_margin: float = 0.0,
    contribution: float = 0.0,
    withdrawal: float = 0.0,
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
    garch: bool = False,
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
    dynamic_correlation: bool = False,
    dcc_alpha: float = 0.04,
    dcc_beta: float = 0.94,
    dcc_asymmetry: float = 0.01,
    chunk_size: int | None = None,
    return_kind: str = "log",
    workers: int = 1,
) -> SimulationRun:
    """Calibrate and simulate one fully specified investment scenario.

    ``model_kind="quadrant"`` builds the four-quadrant macro model from growth
    and inflation thresholds; ``model_kind="hmm"`` fits a Gaussian-emission
    hidden Markov model directly on the asset returns instead. ``duration_model``
    controls whether regime run lengths follow the Markov chain or regularized
    state-specific duration hazards, and ``garch`` adds within-regime
    GARCH(1,1) conditional variance dynamics. ``walk_forward`` runs a
    strictly out-of-sample predictive check of the regime model against an
    unconditional benchmark.
    """

    transition_uncertainty = float(transition_uncertainty)
    if not 0 <= transition_uncertainty <= 1:
        raise ValueError("transition_uncertainty must be between 0 and 1.")
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
        )
        parameter_summary = summarize_parameter_models(parameter_models, normalized_weights)

    simulation_models = parameter_models or [model]
    quotient, remainder = divmod(int(paths), len(simulation_models))
    model_path_counts = [quotient + (1 if index < remainder else 0) for index in range(len(simulation_models))]
    simulation_runs: list[tuple[SimulationResult, pd.DataFrame]] = []
    for draw, (simulation_model, draw_paths) in enumerate(zip(simulation_models, model_path_counts)):
        draw_result, draw_wealth = _simulate_chunked(
                simulation_model,
                periods=int(periods),
                paths=draw_paths,
                random_seed=int(random_seed) + draw * 100_003,
                start_state=start_state,
                distribution=distribution,
                degrees_of_freedom=float(degrees_of_freedom),
                block_size=int(block_size),
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
                expense_ratios=pd.Series(normalized_expense_ratios, dtype=float),
                leverage_multiple=float(leverage_multiple),
                financing_rate=float(financing_rate),
                financing_inflation_sensitivity=float(financing_inflation_sensitivity),
                state_inflation=simulation_model.metadata.get("state_inflation"),
                maintenance_margin=float(maintenance_margin),
                contribution=float(contribution),
                withdrawal=float(withdrawal),
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
                degrees_of_freedom=draw_result.degrees_of_freedom,
                transition_concentration=draw_result.transition_concentration,
                macro_paths=draw_result.macro_paths,
                macro_columns=draw_result.macro_columns,
            )
        simulation_runs.append((draw_result, draw_wealth))

    if len(simulation_runs) == 1:
        result, wealth = simulation_runs[0]
    else:
        wealth = pd.concat([run_wealth for _, run_wealth in simulation_runs], axis=1, ignore_index=True)
        wealth.columns = [f"path_{index}" for index in range(wealth.shape[1])]
        wealth.attrs["margin_calls"] = int(
            sum(run_wealth.attrs.get("margin_calls", 0) for _, run_wealth in simulation_runs)
        )
        regimes_combined = np.concatenate(
            [run_result.regimes for run_result, _ in simulation_runs], axis=1
        )
        macro_parts = [
            run_result.macro_paths
            for run_result, _ in simulation_runs
            if run_result.macro_paths is not None
        ]
        result = SimulationResult(
            returns=np.empty((int(periods), 0, len(model.assets)), dtype=float),
            regimes=regimes_combined,
            assets=model.assets,
            states=model.states.copy(),
            frequency=model.frequency,
            distribution=distribution,
            degrees_of_freedom=(
                float(degrees_of_freedom) if distribution == "student_t" else None
            ),
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
    reporting_wealth = inflation_adjust_wealth(
        wealth,
        annual_inflation=annual_inflation,
        inflation_paths=inflation_paths,
    )
    summary = summarize_wealth_risk(
        wealth,
        initial_value=initial_value,
        risk_free_rate=risk_free_rate,
        annual_inflation=annual_inflation,
        contribution=float(contribution),
        withdrawal=float(withdrawal),
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
    }.items():
        summary[key] = value
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
        parameter_uncertainty=parameter_summary,
    )


def compare_distributions(
    distributions: Mapping[str, str],
    **scenario_kwargs: Any,
) -> pd.DataFrame:
    """Run identical inputs under several return distributions."""

    rows: list[dict[str, Any]] = []
    for label, distribution in distributions.items():
        scenario = run_scenario(distribution=distribution, **scenario_kwargs)
        summary = scenario.summary
        rows.append(
            {
                "distribution": label,
                "mean": summary["mean"],
                "p05": summary["p05"],
                "median": summary["p50"],
                "p95": summary["p95"],
                "annualized_return": summary["annualized_return"],
                "annualized_volatility": summary["annualized_volatility"],
                "sharpe_ratio": summary["sharpe_ratio"],
                "probability_of_loss": summary["probability_of_loss"],
                "var_95": summary["var_95"],
                "expected_shortfall_95": summary["expected_shortfall_95"],
                "worst_max_drawdown": summary["max_drawdown_worst"],
                "ulcer_index_mean": summary["ulcer_index_mean"],
                "sortino_ratio": summary["sortino_ratio"],
                "calmar_ratio": summary["calmar_ratio"],
                "geometric_annualized_return": summary["geometric_annualized_return"],
            }
        )
    return pd.DataFrame(rows)
