from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from mc_quadrants.calibration import _clean_aligned_returns, calibrate_quadrant_model
from mc_quadrants.matrix import nearest_psd
from mc_quadrants.regimes import classify_quadrants


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
    max_splits: int = 120,
    min_observations: int = 12,
    probabilistic_regimes: bool = False,
    regime_temperature: float = 0.35,
    mean_prior_strength: float = 24.0,
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
    effective_threshold_window = threshold_window if threshold_window is not None else 12
    if effective_threshold_window <= 0:
        raise ValueError("threshold_window must be positive for walk-forward validation.")
    macro_regimes = classify_quadrants(
        macro,
        growth_col=growth_col,
        inflation_col=inflation_col,
        growth_threshold=growth_threshold,
        inflation_threshold=inflation_threshold,
        threshold_window=effective_threshold_window,
    )
    aligned_returns, aligned_regimes = _clean_aligned_returns(
        returns,
        macro_regimes,
        lag_periods=macro_lag_periods,
    )
    observations = aligned_returns.to_numpy(dtype=float)
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
            probabilistic_regimes=probabilistic_regimes,
            regime_temperature=regime_temperature,
            mean_prior_strength=mean_prior_strength,
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
        transition = model.transition_matrix.to_numpy(dtype=float)
        if probabilistic_regimes and model.metadata.get("latest_regime_probabilities"):
            current_probabilities = np.array(
                [
                    float(model.metadata["latest_regime_probabilities"].get(state, 0.0))
                    for state in model.states
                ],
                dtype=float,
            )
            current_probabilities /= max(float(current_probabilities.sum()), 1e-300)
            state_probabilities = current_probabilities @ transition
        else:
            last_state_index = model.states.index(str(aligned_regimes.iloc[split - 1]))
            state_probabilities = transition[last_state_index]
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
        gaussian_maximum = gaussian_densities.max()
        regime_llk = gaussian_maximum + np.log(
            float(
                np.sum(
                    state_probabilities
                    * np.exp(gaussian_densities - gaussian_maximum)
                )
            )
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
        student_t_maximum = conditional_t_densities.max()
        regime_student_t_llk = student_t_maximum + np.log(
            float(
                np.sum(
                    state_probabilities
                    * np.exp(conditional_t_densities - student_t_maximum)
                )
            )
        )
        predicted_state = model.states[int(np.argmax(state_probabilities))]
        actual_state = str(aligned_regimes.iloc[split])
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
        equal_weights = np.full(next_observation.shape[0], 1.0 / next_observation.shape[0])
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
            portfolio_draws[mask] = draws @ equal_weights
        predicted_var = float(np.quantile(portfolio_draws, 0.05))
        predicted_es = float(portfolio_draws[portfolio_draws <= predicted_var].mean())
        actual_portfolio_return = float(next_observation @ equal_weights)
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
                "benchmark_brier_score": float(
                    np.square(benchmark_probabilities - one_hot).sum()
                ),
                "actual_state_probability": float(state_probabilities[actual_index]),
                "portfolio_pit": float((portfolio_draws <= actual_portfolio_return).mean()),
                "predicted_var_95": predicted_var,
                "predicted_es_95": predicted_es,
                "portfolio_return": actual_portfolio_return,
                "var_breach": int(actual_portfolio_return < predicted_var),
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
            "benchmark_brier_score": float(splits["benchmark_brier_score"].mean()),
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
    if not 0.025 <= summary["var_95_breach_rate"] <= 0.075:
        warnings.append(
            f"The 95% one-period VaR breach rate is {summary['var_95_breach_rate']:.1%}; "
            "the calibrated target is 5%."
        )
    return WalkForwardResult(splits=splits, summary=summary, warnings=warnings)
