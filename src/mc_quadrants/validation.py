from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from mc_quadrants.calibration import _clean_aligned_returns, calibrate_quadrant_model
from mc_quadrants.matrix import nearest_psd
from mc_quadrants.mnts import (
    fit_mnts_parameters,
    resolved_mnts_parameters,
    sample_mnts_subordinators,
)
from mc_quadrants.native import native_available, sample_mnts_subordinators_native
from mc_quadrants.path_validation import score_horizon, summarize_horizons
from mc_quadrants.regimes import classify_persistent_quadrants
from mc_quadrants.simulation import simulate_returns
from mc_quadrants.types import RegimeMoments, SimulationResult
from mc_quadrants.uncertainty import anchor_parameter_models, bootstrap_quadrant_models


@dataclass(frozen=True)
class WalkForwardResult:
    """Out-of-sample predictive checks for the calibrated regime model."""

    splits: pd.DataFrame
    summary: pd.Series
    warnings: list[str] = field(default_factory=list)
    horizon_scores: pd.DataFrame = field(default_factory=pd.DataFrame)
    horizon_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    metadata: dict = field(default_factory=dict)


def _energy_score(
    observation: np.ndarray,
    draws: np.ndarray,
) -> float:
    """Exact empirical-ensemble energy score, invariant to bootstrap ordering."""

    values = np.asarray(draws, dtype=float)
    first = np.linalg.norm(values - np.asarray(observation, dtype=float), axis=1).mean()
    # All pairs matter for a mixture: pairing opposite halves would sample
    # only cross-model distances when two bootstrap ensembles are concatenated.
    centered = values - values.mean(axis=0)
    squares = np.sum(centered**2, axis=1)
    pair_sum = 0.0
    for start in range(0, len(values), 128):
        stop = min(start + 128, len(values))
        distances_squared = squares[start:stop, None] + squares[None, :] - 2 * centered[start:stop] @ centered.T
        distances_squared[np.arange(stop-start), np.arange(start, stop)] = 0.0
        pair_sum += float(np.sqrt(np.maximum(distances_squared, 0.0)).sum())
    return float(first - 0.5 * pair_sum / len(values)**2)


def _sample_mnts_observations(
    moments: RegimeMoments,
    samples: int,
    random_seed: int,
) -> np.ndarray:
    """Draw from one calibrated MNTS state for proper-score validation."""

    parameters = resolved_mnts_parameters(moments)
    rng = np.random.default_rng(random_seed)
    if native_available():
        subordinator = sample_mnts_subordinators_native(
            samples,
            parameters.tail_index,
            parameters.tempering,
            random_seed,
        )
    else:
        subordinator = sample_mnts_subordinators(
            rng,
            samples,
            parameters.tail_index,
            parameters.tempering,
        )
    assets = list(moments.mean.index)
    skewness = parameters.skewness.reindex(assets).to_numpy(dtype=float)
    variance_t = (2.0 - parameters.tail_index) / (2.0 * parameters.tempering)
    gaussian_scale = np.sqrt(np.maximum(1.0 - skewness * skewness * variance_t, 1e-10))
    correlation = parameters.gaussian_correlation.reindex(
        index=assets, columns=assets
    ).to_numpy(dtype=float)
    factor = np.linalg.cholesky(correlation + np.eye(len(assets)) * 1e-10)
    latent = rng.standard_normal((samples, len(assets))) @ factor.T
    standardized = (
        skewness * (subordinator[:, None] - 1.0)
        + np.sqrt(subordinator)[:, None] * gaussian_scale * latent
    )
    covariance = moments.covariance.reindex(index=assets, columns=assets).to_numpy(dtype=float)
    volatility = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    return moments.mean.reindex(assets).to_numpy(dtype=float) + standardized * volatility


def _unconditional_mnts_moments(training_returns: pd.DataFrame) -> RegimeMoments:  # Fit the unconditional MNTS benchmark moments.
    assets = list(training_returns.columns)
    covariance = nearest_psd(training_returns.cov().to_numpy(dtype=float))
    volatility = np.sqrt(np.maximum(np.diag(covariance), 1e-12))
    correlation = covariance / np.outer(volatility, volatility)
    np.fill_diagonal(correlation, 1.0)
    moments = RegimeMoments(
        mean=training_returns.mean().reindex(assets),
        covariance=pd.DataFrame(covariance, index=assets, columns=assets),
        correlation=pd.DataFrame(correlation, index=assets, columns=assets),
        observations=len(training_returns),
    )
    return replace(
        moments,
        mnts=fit_mnts_parameters(moments, training_returns, training_returns),
    )


def _newey_west_t_statistic(values: np.ndarray, max_lag: int = 3) -> float:  # Compute a Newey-West statistic for serial dependence.
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if len(clean) < 3:
        return 0.0
    centered = clean - clean.mean()
    long_run_variance = float(centered @ centered / len(clean))
    for lag in range(1, min(max_lag, len(clean) - 1) + 1):
        weight = 1.0 - lag / (max_lag + 1.0)
        covariance = float(centered[lag:] @ centered[:-lag] / len(clean))
        long_run_variance += 2.0 * weight * covariance
    standard_error = np.sqrt(max(long_run_variance, 1e-18) / len(clean))
    return float(clean.mean() / standard_error)


def _current_run_length(regimes: pd.Series) -> int:
    """Return the number of consecutive observations in the current state."""

    clean = regimes.dropna().astype(str)
    if clean.empty:
        return 0
    current = str(clean.iloc[-1])
    length = 0
    for state in reversed(clean.tolist()):
        if state != current:
            break
        length += 1
    return length


def _duration_hazard(model: object, state: str, age: int, minimum: int) -> float:  # Read one state-age exit hazard from model metadata.
    hazards = np.asarray(model.metadata.get("duration_hazards", {}).get(state, []), dtype=float)
    if not len(hazards):
        return float(1.0 - model.transition_matrix.loc[state, state])
    if age < minimum:
        return 0.0
    return float(np.clip(hazards[min(max(age, 1), len(hazards)) - 1], 0.0, 1.0))


def _semi_markov_state_probabilities(
    model: object,
    state: str,
    age: int,
    minimum: int,
) -> tuple[np.ndarray, float]:
    """Return one-step state probabilities conditional on current regime age."""

    index = model.states.index(state)
    hazard = _duration_hazard(model, state, age, minimum)
    transition = model.transition_matrix.loc[model.states, model.states].to_numpy(dtype=float)
    destinations = transition[index].copy()
    destinations[index] = 0.0
    if destinations.sum() <= 0:
        destinations[:] = 1.0
        destinations[index] = 0.0
    destinations /= destinations.sum()
    probabilities = destinations * hazard
    probabilities[index] = 1.0 - hazard
    return probabilities, hazard


def _duration_log_probability(
    model: object,
    state: str,
    duration: int,
    minimum: int,
) -> float:  # Compute the log probability of a state duration.
    hazards = np.asarray(model.metadata.get("duration_hazards", {}).get(state, []), dtype=float)
    if not len(hazards) or duration < 1:
        return float("nan")
    log_probability = 0.0
    for age in range(1, duration):
        hazard = 0.0 if age < minimum else hazards[min(age, len(hazards)) - 1]
        log_probability += float(np.log(max(1.0 - hazard, 1e-12)))
    exit_hazard = 0.0 if duration < minimum else hazards[min(duration, len(hazards)) - 1]
    return log_probability + float(np.log(max(exit_hazard, 1e-12)))


def walk_forward_validation(
    returns: pd.DataFrame,
    macro: pd.DataFrame,
    growth_col: str,
    inflation_col: str,
    growth_threshold: str | float,
    inflation_threshold: str | float,
    min_train_periods: int = 60,
    step: int | None = None,
    macro_lag_periods: int = 0,
    threshold_window: int | None = None,
    max_splits: int = 36,
    min_observations: int = 12,
    probabilistic_regimes: bool = False,
    regime_temperature: float = 0.35,
    regime_smoothing_window: int = 3,
    regime_hysteresis: float = 0.15,
    regime_confirmation_periods: int = 2,
    duration_prior_strength: float = 8.0,
    min_regime_duration: int = 5,
    mean_prior_strength: float = 24.0,
    weights: Mapping[str, float] | None = None,
    hsmm_max_iterations: int = 30,
    evaluation_start: str | None = None,
    use_fitted_dependence: bool = True,
    garch: bool = True,
    dynamic_correlation: bool = True,
    volatility_model: str = "garch",
    horizons: tuple[int, ...] = (1, 3, 12, 60),
    validation_paths: int = 512,
    random_seed: int = 42,
    calibration_options: Mapping | None = None,
    simulation_options: Mapping | None = None,
    portfolio_options: Mapping | None = None,
    parameter_draws: int = 0,
    parameter_block_size: int = 12,
    parameter_seed: int | None = None,
    observed_price_index: pd.Series | None = None,
) -> WalkForwardResult:
    """Refit chronologically and score paths from the production simulator.

    Simulated first-month returns supply the legacy one-step diagnostics.
    Fully observed 1/3/12/60-month windows additionally score cumulative asset
    returns, a monthly rebalanced portfolio, and the production portfolio ledger.
    Macro release history and optional CPI outcomes are scored separately.
    Point-in-time validity remains conditional on the supplied data vintage.
    """

    horizons = tuple(sorted(set(horizons) | {1}))
    if any(not isinstance(h, (int, np.integer)) or isinstance(h, bool) or not 1 <= h <= 360 for h in horizons):
        raise ValueError("horizons must contain integer months between 1 and 360.")
    if not isinstance(validation_paths, (int, np.integer)) or validation_paths < 2:
        raise ValueError("validation_paths must be an integer of at least 2.")
    if not isinstance(parameter_draws, (int, np.integer)) or not 0 <= parameter_draws <= validation_paths:
        raise ValueError("parameter_draws must be between zero and validation_paths.")
    if parameter_seed is not None and (
        isinstance(parameter_seed, (bool, np.bool_))
        or not isinstance(parameter_seed, (int, np.integer))
        or not 0 <= parameter_seed <= 2**32 - 1
    ):
        raise ValueError("parameter_seed must be an integer between 0 and 4294967295, or None.")
    if step is not None and (not isinstance(step, (int, np.integer)) or step < 1):
        raise ValueError("step must be a positive integer.")
    extra_calibration = dict(calibration_options or {})
    allowed_calibration = {"rate_col", "correlation_overrides", "override_weight", "joint_macro",
                           "structural_returns", "asset_classes", "asset_durations", "asset_income_yields",
                           "macro_model", "transition_smoothing", "shrinkage", "economic_returns"}
    if set(extra_calibration) - allowed_calibration:
        raise ValueError("Unsupported walk-forward calibration option.")
    extra_simulation = dict(simulation_options or {})
    allowed_simulation = {"duration_model", "transition_concentration", "macro_transition_weight",
                          "macro_parameter_uncertainty", "garch_alpha", "garch_beta", "dcc_alpha",
                          "dcc_beta", "dcc_asymmetry", "native_threads"}
    if set(extra_simulation) - allowed_simulation:
        raise ValueError("Unsupported walk-forward simulation option; forecast origins are inferred from training history.")
    ledger_options = dict(portfolio_options or {})
    # Forecast-origin dates and exogenous path arrays are owned by validation.
    if set(ledger_options) & {"financing_rate_paths", "financing_inflation_paths", "withdrawal_inflation_paths",
                              "state_inflation", "tax_start_date", "weights"}:
        raise ValueError("Portfolio path arrays and forecast dates cannot be overridden in validation.")
    if ledger_options.get("return_kind", "log") != "log":
        raise ValueError("Production-path validation requires log-return calibration inputs.")
    if observed_price_index is None:
        raw_index = macro.attrs.get("observed_price_index")
        if raw_index:
            observed_price_index = pd.Series(raw_index, dtype=float)
            observed_price_index.index = pd.to_datetime(observed_price_index.index)
        elif "price_index" in macro:
            observed_price_index = macro["price_index"]
    if observed_price_index is not None:
        observed_price_index = observed_price_index.sort_index()
        if observed_price_index.index.has_duplicates:
            raise ValueError("observed_price_index must have unique dates.")
    joint_macro = bool(extra_calibration.get("joint_macro", False))
    rate_col = extra_calibration.get("rate_col", "interest_rate")
    macro = macro.sort_index()
    # Availability-aligned data already carry their publication month.
    macro_lag_periods = 0 if macro.attrs.get("availability_aligned", False) else macro_lag_periods
    if not isinstance(macro_lag_periods, (int, np.integer)) or macro_lag_periods < 0:
        raise ValueError("macro_lag_periods must be a non-negative integer.")
    source_macro_lag = int(macro_lag_periods)
    if source_macro_lag:
        # Model the observable release sequence on its information calendar.
        # Slicing unshifted levels would either reveal unavailable observations
        # or start the macro process several months behind the forecast origin.
        macro = macro.copy()
        macro.index = macro.index + pd.offsets.MonthEnd(source_macro_lag)
        macro.attrs["availability_aligned"] = True
        macro_lag_periods = 0
    if volatility_model not in {"garch", "qarch"}:
        raise ValueError("volatility_model must be 'garch' or 'qarch'.")
    if min_train_periods < 12:
        raise ValueError("min_train_periods must be at least 12.")
    if max_splits < 1:
        raise ValueError("max_splits must be positive.")
    if hsmm_max_iterations < 1:
        raise ValueError("hsmm_max_iterations must be positive.")
    effective_threshold_window = threshold_window if threshold_window is not None else 12
    if effective_threshold_window <= 0:
        raise ValueError("threshold_window must be positive for walk-forward validation.")
    macro_regimes = classify_persistent_quadrants(
        macro,
        growth_col=growth_col,
        inflation_col=inflation_col,
        growth_threshold=growth_threshold,
        inflation_threshold=inflation_threshold,
        threshold_window=effective_threshold_window,
        smoothing_window=regime_smoothing_window,
        hysteresis=regime_hysteresis,
        confirmation_periods=regime_confirmation_periods,
    )
    aligned_returns, aligned_regimes = _clean_aligned_returns(
        returns,
        macro_regimes,
        lag_periods=macro_lag_periods,
    )
    observations = aligned_returns.to_numpy(dtype=float)
    if weights is None:
        portfolio_weights = np.full(
            aligned_returns.shape[1],
            1.0 / aligned_returns.shape[1],
            dtype=float,
        )
    else:
        weight_series = pd.Series(weights, dtype=float).reindex(aligned_returns.columns).fillna(0.0)
        portfolio_weights = weight_series.to_numpy(dtype=float, copy=True)
        weight_total = float(portfolio_weights.sum())
        if not np.isfinite(portfolio_weights).all() or not np.isfinite(weight_total):
            raise ValueError("Portfolio weights must be finite for walk-forward validation.")
        if abs(weight_total) < 1e-12:
            raise ValueError("Portfolio weights must have a non-zero sum for walk-forward validation.")
        portfolio_weights /= weight_total
    n = len(aligned_returns)
    if n < min_train_periods + 1:
        raise ValueError(
            f"Walk-forward validation requires at least {min_train_periods + 1} aligned observations."
        )
    available_splits = n - min_train_periods
    if step is None:
        step = max(1, int(np.ceil(available_splits / max_splits)))

    rows: list[dict[str, object]] = []
    path_rows: list[dict[str, object]] = []
    skipped_windows = 0
    for split in range(min_train_periods, n, step):
        if evaluation_start is not None and aligned_returns.index[split] < pd.Timestamp(evaluation_start):
            continue
        train_cutoff = aligned_returns.index[split - 1]
        train_returns = returns.sort_index().loc[:train_cutoff].dropna(how="any")
        available_horizons = []
        for horizon in horizons:
            if split + horizon > n:
                continue
            actual_months = aligned_returns.index[split:split + horizon].to_period("M")
            expected_months = pd.period_range(train_cutoff.to_period("M") + 1, periods=horizon, freq="M")
            if not actual_months.equals(expected_months):
                skipped_windows += 1
                continue
            available_horizons.append(horizon)
        if 1 not in available_horizons:
            continue
        macro_cutoff = train_cutoff
        train_macro = macro.loc[macro.index <= macro_cutoff].copy()
        # Outcome-only CPI is deliberately excluded from every model fit.
        train_macro.attrs.pop("observed_price_index", None)
        release_dates = train_macro.attrs.get("release_dates")
        if isinstance(release_dates, pd.DataFrame):
            train_macro.attrs["release_dates"] = release_dates.loc[release_dates.index <= train_cutoff]
        train_macro = train_macro.drop(columns=["price_index"], errors="ignore")
        if train_macro.empty:
            raise ValueError("No macro observations are available before a validation split.")
        fit_options = dict(
            growth_col=growth_col,
            inflation_col=inflation_col,
            growth_threshold=growth_threshold,
            inflation_threshold=inflation_threshold,
            min_observations=min_observations,
            macro_lag_periods=macro_lag_periods,
            threshold_window=effective_threshold_window,
            min_regime_duration=min_regime_duration,
            probabilistic_regimes=probabilistic_regimes,
            regime_temperature=regime_temperature,
            regime_smoothing_window=regime_smoothing_window,
            regime_hysteresis=regime_hysteresis,
            regime_confirmation_periods=regime_confirmation_periods,
            duration_prior_strength=duration_prior_strength,
            mean_prior_strength=mean_prior_strength,
            hsmm_max_iterations=hsmm_max_iterations,
            volatility_model=volatility_model if garch else "garch",
            **extra_calibration,
        )
        model = calibrate_quadrant_model(train_returns, train_macro, **fit_options)
        next_observation = observations[split]
        unconditional_moments = _unconditional_mnts_moments(train_returns)
        current_state = str(aligned_regimes.iloc[split - 1])
        current_age = _current_run_length(aligned_regimes.iloc[:split])
        state_probabilities, switch_probability = _semi_markov_state_probabilities(
            model,
            current_state,
            current_age,
            min_regime_duration,
        )
        predicted_state = model.states[int(np.argmax(state_probabilities))]
        actual_state = str(aligned_regimes.iloc[split])
        actual_switch = int(actual_state != current_state)
        actual_index = model.states.index(actual_state)
        one_hot = np.zeros(len(model.states), dtype=float)
        one_hot[actual_index] = 1.0
        historical_counts = aligned_regimes.iloc[:split].value_counts()
        benchmark_probabilities = np.array(
            [float(historical_counts.get(state, 0)) + 1.0 for state in model.states],
            dtype=float,
        )
        benchmark_probabilities /= benchmark_probabilities.sum()

        forecast_periods = max(available_horizons)
        seed = int(np.random.SeedSequence([random_seed, train_cutoff.year, train_cutoff.month]).generate_state(1)[0])
        simulation_models = [model]
        if parameter_draws:
            bootstrap_options = {key: value for key, value in fit_options.items() if key != "hsmm_max_iterations"}
            bank_seed = seed if parameter_seed is None else int(np.random.SeedSequence(
                [parameter_seed, train_cutoff.year, train_cutoff.month]
            ).generate_state(1)[0])
            simulation_models = anchor_parameter_models(bootstrap_quadrant_models(
                train_returns, train_macro, draws=parameter_draws, block_size=parameter_block_size,
                random_seed=bank_seed, **bootstrap_options), model)
        forecasts = []
        for draw, simulation_model in enumerate(simulation_models):
            simulation_model.metadata["calibrate_dependence"] = use_fitted_dependence
            count = validation_paths // len(simulation_models) + (draw < validation_paths % len(simulation_models))
            options = {"duration_model": "semi_markov", **extra_simulation}
            if use_fitted_dependence:
                fitted = simulation_model.metadata.get("dependence_fit", {})
                for key in ("garch_alpha", "garch_beta", "dcc_alpha", "dcc_beta", "dcc_asymmetry"):
                    if key in fitted:
                        options[key] = fitted[key]
            if parameter_draws:
                options.update(macro_parameter_uncertainty=False, transition_concentration=None)
            forecasts.append(simulate_returns(
                simulation_model, periods=forecast_periods, paths=count,
                random_seed=seed + draw * 100_003, min_regime_duration=min_regime_duration,
                garch=garch, dynamic_correlation=dynamic_correlation, joint_macro=joint_macro,
                volatility_model=volatility_model, **options))
        forecast = replace(forecasts[0], returns=np.concatenate([f.returns for f in forecasts], axis=1),
                           regimes=np.concatenate([f.regimes for f in forecasts], axis=1),
                           macro_paths=np.concatenate([f.macro_paths for f in forecasts], axis=1) if joint_macro else None)
        conditional_draws = forecast.returns[0]
        benchmark_returns = _sample_mnts_observations(
            unconditional_moments, forecast_periods * validation_paths, seed + 1,
        ).reshape(forecast_periods, validation_paths, -1)
        benchmark_draws = benchmark_returns[0]
        # A state-free benchmark: keep the training origin's classifier label
        # for state-dependent accounting, without borrowing future states.
        benchmark = replace(forecast, returns=benchmark_returns,
                            regimes=np.full(forecast.regimes.shape, current_state, dtype=object), macro_paths=None)
        outcome_dates = aligned_returns.index[split:split + forecast_periods]
        observed_macro = macro.reindex(outcome_dates, method="ffill")
        observed = SimulationResult(
            returns=observations[split:split + forecast_periods, None, :],
            regimes=aligned_regimes.iloc[split:split + forecast_periods].to_numpy()[:, None],
            assets=model.assets, states=model.states, frequency=model.frequency,
            macro_paths=observed_macro[forecast.macro_columns].to_numpy(dtype=float)[:, None, :] if joint_macro else None,
            macro_columns=forecast.macro_columns,
        )
        origin_scores = []
        for horizon in available_horizons:
            origin_scores.append(score_horizon(
                model=model, forecast=forecast, benchmark=benchmark, observed=observed,
                weights=dict(zip(model.assets, portfolio_weights)), portfolio_options=ledger_options,
                inflation_col=inflation_col, rate_col=rate_col, observed_macro=macro,
                price_index=observed_price_index, origin=train_cutoff, horizon=horizon,
                outcome_dates=outcome_dates[:horizon], forecast_components=list(zip(simulation_models, forecasts)),
                macro_cutoff=macro_cutoff, energy_score=_energy_score))
        path_rows.extend(origin_scores)
        regime_energy_score = _energy_score(next_observation, conditional_draws)
        benchmark_energy_score = _energy_score(next_observation, benchmark_draws)
        portfolio_draws = np.expm1(conditional_draws) @ portfolio_weights
        predicted_var = float(np.quantile(portfolio_draws, 0.05))
        predicted_es = float(portfolio_draws[portfolio_draws <= predicted_var].mean())
        actual_portfolio_return = float(np.expm1(next_observation) @ portfolio_weights)
        benchmark_portfolio_draws = np.expm1(benchmark_draws) @ portfolio_weights
        certainty_equivalent_advantage = float(
            portfolio_draws.mean() - 1.5 * portfolio_draws.var()
            - benchmark_portfolio_draws.mean() + 1.5 * benchmark_portfolio_draws.var())
        horizon_scores = {}
        for horizon in (3, 12, 60):
            scored = next((row for row in origin_scores if row["horizon_months"] == horizon), None)
            horizon_scores[f"forecast_error_{horizon}m"] = (
                scored["portfolio_return_mean"] - scored["portfolio_return_actual"] if scored else np.nan)
            if scored:
                base_return = np.prod(1 + np.expm1(benchmark_returns[:horizon]) @ portfolio_weights, axis=0) - 1
                horizon_scores[f"benchmark_error_{horizon}m"] = float(base_return.mean() - scored["portfolio_return_actual"])
            else:
                horizon_scores[f"benchmark_error_{horizon}m"] = np.nan
        expected_duration_map = model.metadata.get("expected_duration_months", {})
        expected_duration = float(expected_duration_map.get(current_state, np.nan))
        vintage_expected_duration = float(
            np.nanmean(
                [float(expected_duration_map.get(state, np.nan)) for state in model.states]
            )
        )
        vintage_switches_decade = 120.0 / max(vintage_expected_duration, 1e-12)
        rows.append(
            {
                "date": aligned_returns.index[split],
                "regime_energy_score": regime_energy_score,
                "unconditional_mnts_energy_score": benchmark_energy_score,
                "advantage": benchmark_energy_score - regime_energy_score,
                "regime_hit": int(predicted_state == actual_state),
                "regime_brier_score": float(np.square(state_probabilities - one_hot).sum()),
                "transition_brier_score": float(
                    np.square(state_probabilities - one_hot).sum()
                ),
                "transition_log_score": float(
                    np.log(max(state_probabilities[actual_index], 1e-12))
                ),
                "benchmark_brier_score": float(
                    np.square(benchmark_probabilities - one_hot).sum()
                ),
                "actual_state_probability": float(state_probabilities[actual_index]),
                "actual_switch": actual_switch,
                "predicted_switch_probability": switch_probability,
                "switch_brier_score": float((switch_probability - actual_switch) ** 2),
                "current_regime_age": int(current_age),
                "expected_state_duration": expected_duration,
                "completed_duration": float(current_age) if actual_switch else np.nan,
                "duration_log_score": (
                    _duration_log_probability(
                        model,
                        current_state,
                        current_age,
                        min_regime_duration,
                    )
                    if actual_switch
                    else np.nan
                ),
                "vintage_expected_duration": vintage_expected_duration,
                "vintage_switches_per_decade": vintage_switches_decade,
                "portfolio_pit": float((portfolio_draws <= actual_portfolio_return).mean()),
                "predicted_var_95": predicted_var,
                "predicted_es_95": predicted_es,
                "portfolio_return": actual_portfolio_return,
                "var_breach": int(actual_portfolio_return < predicted_var),
                "certainty_equivalent_advantage": certainty_equivalent_advantage,
                **horizon_scores,
                "predicted_state": predicted_state,
                "actual_state": actual_state,
            }
        )

    if not rows:
        raise ValueError("No observations are available in the requested evaluation window.")
    splits = pd.DataFrame(rows)
    summary = pd.Series(
        {
            "splits": int(len(splits)),
            "regime_energy_score_mean": float(splits["regime_energy_score"].mean()),
            "unconditional_mnts_energy_score_mean": float(
                splits["unconditional_mnts_energy_score"].mean()
            ),
            "advantage_mean": float(splits["advantage"].mean()),
            "advantage_positive_share": float((splits["advantage"] > 0).mean()),
            "dm_t_statistic": _newey_west_t_statistic(
                splits["advantage"].to_numpy(dtype=float)
            ),
            "regime_hit_rate": float(splits["regime_hit"].mean()),
            "regime_brier_score": float(splits["regime_brier_score"].mean()),
            "transition_brier_score": float(splits["transition_brier_score"].mean()),
            "transition_log_score_mean": float(splits["transition_log_score"].mean()),
            "benchmark_brier_score": float(splits["benchmark_brier_score"].mean()),
            "actual_switch_rate": float(splits["actual_switch"].mean()),
            "actual_switches_per_decade": float(splits["actual_switch"].mean() * 120.0),
            "predicted_switch_rate": float(splits["predicted_switch_probability"].mean()),
            "predicted_switches_per_decade": float(
                splits["predicted_switch_probability"].mean() * 120.0
            ),
            "switch_brier_score": float(splits["switch_brier_score"].mean()),
            "completed_duration_count": int(splits["completed_duration"].notna().sum()),
            "completed_duration_mean": float(splits["completed_duration"].mean()),
            "duration_log_score_mean": float(splits["duration_log_score"].mean()),
            "duration_mae_months": float(
                (splits["completed_duration"] - splits["expected_state_duration"]).abs().mean()
            ),
            "rolling_vintage_expected_duration_mean": float(
                splits["vintage_expected_duration"].mean()
            ),
            "rolling_vintage_expected_duration_std": float(
                splits["vintage_expected_duration"].std(ddof=0)
            ),
            "rolling_vintage_switches_decade_std": float(
                splits["vintage_switches_per_decade"].std(ddof=0)
            ),
            "actual_state_probability_mean": float(splits["actual_state_probability"].mean()),
            "portfolio_pit_mean": float(splits["portfolio_pit"].mean()),
            "portfolio_pit_std": float(splits["portfolio_pit"].std(ddof=0)),
            "var_95_breach_rate": float(splits["var_breach"].mean()),
            "var_95_breach_cluster_rate": float(
                (splits["var_breach"].shift(1).fillna(0).astype(int) * splits["var_breach"]).sum()
                / max(int(splits["var_breach"].sum()), 1)
            ),
        }
    )
    summary["certainty_equivalent_advantage_annualized"] = float(
        splits["certainty_equivalent_advantage"].mean() * 12.0
    )
    for horizon in (3, 12, 60):
        model_error = splits[f"forecast_error_{horizon}m"].dropna()
        benchmark_error = splits[f"benchmark_error_{horizon}m"].dropna()
        summary[f"forecast_mae_{horizon}m"] = float(model_error.abs().mean())
        summary[f"benchmark_mae_{horizon}m"] = float(benchmark_error.abs().mean())
        summary[f"forecast_mae_improvement_{horizon}m"] = (
            float(benchmark_error.abs().mean() - model_error.abs().mean())
            if len(model_error) and len(benchmark_error)
            else float("nan")
        )
    warnings: list[str] = []
    if summary["advantage_mean"] <= 0:
        warnings.append(
            "The regime-switching MNTS model does not beat an unconditional MNTS benchmark "
            f"out of sample (energy-score advantage {summary['advantage_mean']:.4f} per period)."
        )
    if summary["regime_hit_rate"] < 0.40:
        warnings.append(
            f"The one-step regime prediction matched reality on only "
            f"{summary['regime_hit_rate']:.0%} of splits."
        )
    if summary["regime_brier_score"] >= summary["benchmark_brier_score"]:
        warnings.append("Regime probabilities do not improve on historical-frequency probabilities.")
    if abs(summary["actual_switch_rate"] - summary["predicted_switch_rate"]) > 0.05:
        warnings.append(
            "Out-of-sample switch frequency differs materially from the duration model "
            f"({summary['actual_switches_per_decade']:.1f} observed versus "
            f"{summary['predicted_switches_per_decade']:.1f} predicted per decade)."
        )
    if max(
        summary["predicted_switches_per_decade"],
        summary["actual_switches_per_decade"],
    ) > 24:
        warnings.append(
            "Out-of-sample persistence is unusually low: "
            f"{summary['actual_switches_per_decade']:.1f} switches per decade were observed "
            f"versus {summary['predicted_switches_per_decade']:.1f} predicted."
        )
    duration_mean = max(float(summary["rolling_vintage_expected_duration_mean"]), 1e-12)
    if summary["rolling_vintage_expected_duration_std"] / duration_mean > 0.25:
        warnings.append("Persistence estimates are unstable across rolling calibration vintages.")
    if summary["completed_duration_count"] < 5:
        warnings.append("Too few completed out-of-sample regimes are available to validate durations reliably.")
    if not 0.025 <= summary["var_95_breach_rate"] <= 0.075:
        warnings.append(
            f"The 95% one-period VaR breach rate is {summary['var_95_breach_rate']:.1%}; "
            "the calibrated target is 5%."
        )
    path_scores = pd.DataFrame(path_rows)
    horizon_summary = summarize_horizons(path_scores, horizons)
    if observed_price_index is None:
        warnings.append("Real-wealth validation unavailable: provide observed CPI levels (price_index); annual inflation is not a monthly price index.")
    elif "real_wealth_crps" not in path_scores or path_scores["real_wealth_crps"].isna().any():
        warnings.append("Some real-wealth validation windows lack complete observed CPI levels.")
    if not bool(macro.attrs.get("point_in_time", False)):
        warnings.append("Validation refits are chronological, but supplied macro data are not certified point-in-time vintages.")
    if skipped_windows:
        warnings.append(f"Skipped {skipped_windows} validation windows containing missing calendar months.")
    for row in horizon_summary.itertuples():
        if row.nonoverlapping_origins < 10:
            warnings.append(f"{row.horizon_months}-month validation has {row.origins} scored origins and only {row.nonoverlapping_origins} non-overlapping windows; long-horizon evidence is limited.")
    return WalkForwardResult(
        splits=splits, summary=summary, warnings=warnings,
        horizon_scores=path_scores, horizon_summary=horizon_summary,
        metadata={
            "forecast_engine": "production_simulate_returns",
            "portfolio_engine": "production_simulate_portfolio_paths",
            "paths_per_origin": validation_paths, "horizons_months": list(horizons),
            "hsmm_max_iterations": hsmm_max_iterations, "joint_macro": joint_macro,
            "parameter_draws": parameter_draws, "seed": random_seed, "parameter_seed": parameter_seed,
            "duration_model": extra_simulation.get("duration_model", "semi_markov"),
            "volatility_model": volatility_model if garch else "constant",
            "dynamic_correlation": dynamic_correlation, "fitted_dependence": use_fitted_dependence,
            "portfolio_return_basis": "monthly_rebalanced_simple_returns_before_costs",
            "wealth_basis": "selected_portfolio_ledger_with_cash_flows",
            "inflation_basis": "annual_rate_on_supplied_macro_release_calendar",
            "real_wealth_outcome_basis": "observed_price_index" if observed_price_index is not None else "unavailable",
            "regime_diagnostics_basis": "persistent_classifier_proxy_not_observed_latent_state",
            "point_in_time": bool(macro.attrs.get("point_in_time", False)),
            "benchmark": "iid_unconditional_mnts_with_constant_origin_macro",
            "overlapping_windows": True,
            "forecast_origin": "training_filtered_state_and_age; no present-day state override",
            "macro_lag_periods": source_macro_lag,
            "macro_calendar": "publication_month",
            "tax_policy_basis": "selected_policy_replayed_on_historical_returns_not_historical_tax_law",
            "wealth_return_basis": "legacy_weighted_log" if ledger_options.get("rebalance_frequency") is None else "holdings_ledger",
            "score_units": "asset scores: cumulative log returns; portfolio returns: decimals; wealth: initial-value currency; inflation: decimal annual rate",

        })
