from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from mc_quadrants.calibration import _clean_aligned_returns, calibrate_quadrant_model
from mc_quadrants.matrix import nearest_psd
from mc_quadrants.regimes import classify_persistent_quadrants


@dataclass(frozen=True)
class WalkForwardResult:
    """Out-of-sample predictive checks for the calibrated regime model."""

    splits: pd.DataFrame
    summary: pd.Series
    warnings: list[str] = field(default_factory=list)


def _log_gaussian_density(
    observation: np.ndarray,
    mean: np.ndarray,
    covariance: np.ndarray,
) -> float:
    """Log density of one observation under a multivariate Gaussian."""

    chol = np.linalg.cholesky(nearest_psd(covariance))
    difference = observation - mean
    standardized = np.linalg.solve(chol, difference)
    log_determinant = 2.0 * np.log(np.diag(chol)).sum()
    dimension = len(mean)
    return -0.5 * (dimension * np.log(2.0 * np.pi) + log_determinant + standardized @ standardized)


def _log_student_t_density(
    observation: np.ndarray,
    mean: np.ndarray,
    covariance: np.ndarray,
    degrees_of_freedom: float = 5.0,
) -> float:
    """Multivariate Student-t log density parameterized by covariance."""

    dimension = len(mean)
    scale = nearest_psd(covariance * (degrees_of_freedom - 2.0) / degrees_of_freedom)
    chol = np.linalg.cholesky(scale)
    difference = observation - mean
    standardized = np.linalg.solve(chol, difference)
    quadratic = float(standardized @ standardized)
    log_determinant = 2.0 * np.log(np.diag(chol)).sum()
    return float(
        math.lgamma((degrees_of_freedom + dimension) / 2.0)
        - math.lgamma(degrees_of_freedom / 2.0)
        - 0.5 * (dimension * np.log(degrees_of_freedom * np.pi) + log_determinant)
        - 0.5 * (degrees_of_freedom + dimension) * np.log1p(quadratic / degrees_of_freedom)
    )


def _newey_west_t_statistic(values: np.ndarray, max_lag: int = 3) -> float:
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


def _duration_hazard(model: object, state: str, age: int, minimum: int) -> float:
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
) -> float:
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
    hsmm_max_iterations: int = 5,
) -> WalkForwardResult:
    """Evaluate the regime model strictly out of sample.

    Each split fits the quadrant model on data available up to period ``t``
    and scores the *next* return observation with regime-conditional and
    unconditional Gaussian and Student-t densities. The like-for-like
    Student-t comparison isolates the incremental value of regime conditioning
    from the value of simply using fatter tails. The regime hit rate measures
    whether the most likely next state matches the state actually realized.

    Thresholds are re-estimated causally on each training window, so no
    future information enters a split.
    """

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
    for split in range(min_train_periods, n, step):
        train_returns = aligned_returns.iloc[:split]
        train_cutoff = aligned_returns.index[split - 1]
        train_macro = macro.loc[macro.index <= train_cutoff]
        if train_macro.empty:
            raise ValueError("No macro observations are available before a validation split.")
        model = calibrate_quadrant_model(
            returns=train_returns,
            macro=train_macro,
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
        )
        next_observation = observations[split]
        unconditional_mean = observations[:split].mean(axis=0)
        unconditional_covariance = np.cov(observations[:split], rowvar=False)
        unconditional_covariance = np.atleast_2d(unconditional_covariance)
        benchmark_llk = _log_gaussian_density(
            next_observation,
            unconditional_mean,
            unconditional_covariance,
        )
        student_t_llk = _log_student_t_density(
            next_observation,
            unconditional_mean,
            unconditional_covariance,
        )
        current_state = str(aligned_regimes.iloc[split - 1])
        current_age = _current_run_length(aligned_regimes.iloc[:split])
        state_probabilities, switch_probability = _semi_markov_state_probabilities(
            model,
            current_state,
            current_age,
            min_regime_duration,
        )
        gaussian_densities = np.array(
            [
                _log_gaussian_density(
                    next_observation,
                    model.moments[state].mean.to_numpy(dtype=float),
                    model.moments[state].covariance.to_numpy(dtype=float),
                )
                for state in model.states
            ]
        )
        gaussian_weighted = gaussian_densities + np.log(
            np.maximum(state_probabilities, 1e-300)
        )
        gaussian_maximum = gaussian_weighted.max()
        regime_llk = gaussian_maximum + np.log(
            float(np.exp(gaussian_weighted - gaussian_maximum).sum())
        )
        conditional_t_densities = np.array(
            [
                _log_student_t_density(
                    next_observation,
                    model.moments[state].mean.to_numpy(dtype=float),
                    model.moments[state].covariance.to_numpy(dtype=float),
                )
                for state in model.states
            ]
        )
        student_t_weighted = conditional_t_densities + np.log(
            np.maximum(state_probabilities, 1e-300)
        )
        student_t_maximum = student_t_weighted.max()
        regime_student_t_llk = student_t_maximum + np.log(
            float(np.exp(student_t_weighted - student_t_maximum).sum())
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

        validation_rng = np.random.default_rng(split)
        sampled_states = validation_rng.choice(
            len(model.states), size=512, p=state_probabilities
        )
        portfolio_draws = np.empty(512, dtype=float)
        for state_index, state in enumerate(model.states):
            mask = sampled_states == state_index
            if not mask.any():
                continue
            draws = validation_rng.multivariate_normal(
                np.zeros(next_observation.shape[0]),
                model.moments[state].covariance.to_numpy(dtype=float),
                size=int(mask.sum()),
            )
            draws *= np.sqrt(
                3.0 / validation_rng.chisquare(5.0, size=int(mask.sum()))
            )[:, None]
            draws += model.moments[state].mean.to_numpy(dtype=float)
            portfolio_draws[mask] = draws @ portfolio_weights
        predicted_var = float(np.quantile(portfolio_draws, 0.05))
        predicted_es = float(portfolio_draws[portfolio_draws <= predicted_var].mean())
        actual_portfolio_return = float(next_observation @ portfolio_weights)
        state_portfolio_means = np.array(
            [float(model.moments[state].mean.to_numpy(dtype=float) @ portfolio_weights) for state in model.states]
        )
        state_portfolio_variances = np.array(
            [
                float(
                    portfolio_weights
                    @ model.moments[state].covariance.to_numpy(dtype=float)
                    @ portfolio_weights
                )
                for state in model.states
            ]
        )
        predicted_portfolio_mean = float(state_probabilities @ state_portfolio_means)
        predicted_portfolio_variance = float(
            state_probabilities
            @ (state_portfolio_variances + np.square(state_portfolio_means - predicted_portfolio_mean))
        )
        benchmark_portfolio_mean = float(unconditional_mean @ portfolio_weights)
        benchmark_portfolio_variance = float(
            portfolio_weights @ unconditional_covariance @ portfolio_weights
        )
        risk_aversion = 3.0
        certainty_equivalent_advantage = (
            predicted_portfolio_mean - 0.5 * risk_aversion * predicted_portfolio_variance
            - benchmark_portfolio_mean
            + 0.5 * risk_aversion * benchmark_portfolio_variance
        )
        horizon_scores: dict[str, float] = {}
        transition_values = model.transition_matrix.loc[model.states, model.states].to_numpy(dtype=float)
        for horizon in (3, 12, 60):
            if split + horizon > n:
                horizon_scores[f"forecast_error_{horizon}m"] = np.nan
                horizon_scores[f"benchmark_error_{horizon}m"] = np.nan
                continue
            probability = state_probabilities.copy()
            forecast = 0.0
            for _ in range(horizon):
                forecast += float(probability @ state_portfolio_means)
                probability = probability @ transition_values
            actual = float((observations[split:split + horizon] @ portfolio_weights).sum())
            horizon_scores[f"forecast_error_{horizon}m"] = forecast - actual
            horizon_scores[f"benchmark_error_{horizon}m"] = horizon * benchmark_portfolio_mean - actual
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
                "regime_log_likelihood": float(regime_llk),
                "regime_student_t_log_likelihood": float(regime_student_t_llk),
                "unconditional_log_likelihood": float(benchmark_llk),
                "student_t_log_likelihood": float(student_t_llk),
                "advantage": float(regime_llk - benchmark_llk),
                "advantage_vs_student_t": float(regime_student_t_llk - student_t_llk),
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

    splits = pd.DataFrame(rows)
    summary = pd.Series(
        {
            "splits": int(len(splits)),
            "regime_log_likelihood_mean": float(splits["regime_log_likelihood"].mean()),
            "regime_student_t_log_likelihood_mean": float(
                splits["regime_student_t_log_likelihood"].mean()
            ),
            "unconditional_log_likelihood_mean": float(splits["unconditional_log_likelihood"].mean()),
            "student_t_log_likelihood_mean": float(splits["student_t_log_likelihood"].mean()),
            "advantage_mean": float(splits["advantage"].mean()),
            "advantage_vs_student_t_mean": float(splits["advantage_vs_student_t"].mean()),
            "advantage_positive_share": float((splits["advantage"] > 0).mean()),
            "dm_t_statistic_vs_student_t": _newey_west_t_statistic(
                splits["advantage_vs_student_t"].to_numpy(dtype=float)
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
            "The regime model does not beat an unconditional benchmark out of sample "
            f"(advantage {summary['advantage_mean']:.4f} log-likelihood units per period)."
        )
    if summary["regime_hit_rate"] < 0.40:
        warnings.append(
            f"The one-step regime prediction matched reality on only "
            f"{summary['regime_hit_rate']:.0%} of splits."
        )
    if summary["advantage_vs_student_t_mean"] <= 0:
        warnings.append(
            "The regime model does not beat the stronger unconditional Student-t benchmark out of sample."
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
    return WalkForwardResult(splits=splits, summary=summary, warnings=warnings)
