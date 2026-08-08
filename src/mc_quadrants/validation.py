from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from mc_quadrants.calibration import _clean_aligned_returns, calibrate_quadrant_model
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

    chol = np.linalg.cholesky(covariance)
    difference = observation - mean
    standardized = np.linalg.solve(chol, difference)
    log_determinant = 2.0 * np.log(np.diag(chol)).sum()
    dimension = len(mean)
    return -0.5 * (dimension * np.log(2.0 * np.pi) + log_determinant + standardized @ standardized)


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
) -> WalkForwardResult:
    """Evaluate the regime model strictly out of sample.

    Each split fits the quadrant model on data available up to period ``t``
    and scores the *next* return observation two ways: the regime-conditional
    mixture density (one-step predictive distribution through the transition
    matrix) and an unconditional Gaussian fitted on the same history. The
    difference is the model's out-of-sample predictive advantage, and the
    regime hit rate measures whether the most likely next state matches the
    state actually realized.

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
        transition = model.transition_matrix.to_numpy(dtype=float)
        last_state_index = model.states.index(str(aligned_regimes.iloc[split - 1]))
        state_probabilities = transition[last_state_index]
        densities = np.array(
            [
                _log_gaussian_density(
                    next_observation,
                    model.moments[state].mean.to_numpy(dtype=float),
                    model.moments[state].covariance.to_numpy(dtype=float),
                )
                for state in model.states
            ]
        )
        maximum = densities.max()
        regime_llk = maximum + np.log(float(np.sum(state_probabilities * np.exp(densities - maximum))))
        predicted_state = model.states[int(np.argmax(state_probabilities))]
        actual_state = str(aligned_regimes.iloc[split])
        rows.append(
            {
                "date": aligned_returns.index[split],
                "regime_log_likelihood": float(regime_llk),
                "unconditional_log_likelihood": float(benchmark_llk),
                "advantage": float(regime_llk - benchmark_llk),
                "regime_hit": int(predicted_state == actual_state),
                "predicted_state": predicted_state,
                "actual_state": actual_state,
            }
        )

    splits = pd.DataFrame(rows)
    summary = pd.Series(
        {
            "splits": int(len(splits)),
            "regime_log_likelihood_mean": float(splits["regime_log_likelihood"].mean()),
            "unconditional_log_likelihood_mean": float(splits["unconditional_log_likelihood"].mean()),
            "advantage_mean": float(splits["advantage"].mean()),
            "advantage_positive_share": float((splits["advantage"] > 0).mean()),
            "regime_hit_rate": float(splits["regime_hit"].mean()),
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
    return WalkForwardResult(splits=splits, summary=summary, warnings=warnings)
