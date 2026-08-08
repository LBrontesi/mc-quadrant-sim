from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from mc_quadrants.matrix import (
    covariance_to_correlation,
    nearest_correlation,
    nearest_psd,
    nearest_psd_higham,
)
from mc_quadrants.regimes import (
    REGIME_ORDER,
    ThresholdSpec,
    classify_quadrants,
    estimate_transition_matrix,
    sojourn_durations,
)
from mc_quadrants.types import RegimeMoments, ScenarioModel

CorrelationOverrides = Mapping[str, Mapping[tuple[str, str], float]]


def _align_regimes_to_returns(
    regimes: pd.Series,
    returns: pd.DataFrame,
    lag_periods: int = 0,
) -> pd.Series:
    if lag_periods < 0:
        raise ValueError("lag_periods must be non-negative.")
    sorted_regimes = regimes.dropna().sort_index().astype(str).shift(lag_periods)
    sorted_regimes = sorted_regimes.dropna()
    sorted_returns = returns.sort_index()
    aligned = sorted_regimes.reindex(sorted_returns.index, method="ffill")
    return aligned.loc[sorted_returns.index]


def _clean_aligned_returns(
    returns: pd.DataFrame,
    regimes: pd.Series,
    lag_periods: int = 0,
) -> tuple[pd.DataFrame, pd.Series]:
    if returns.index.has_duplicates or regimes.index.has_duplicates:
        raise ValueError("returns and macro regime indexes must not contain duplicates.")
    clean_returns = returns.sort_index().dropna(how="all")
    aligned_regimes = _align_regimes_to_returns(regimes, clean_returns, lag_periods=lag_periods)
    clean_returns = clean_returns.dropna(how="any")
    aligned_regimes = aligned_regimes.loc[clean_returns.index]
    valid_regime = aligned_regimes.notna()
    clean_returns = clean_returns.loc[valid_regime]
    aligned_regimes = aligned_regimes.loc[valid_regime]
    if clean_returns.empty:
        raise ValueError("No overlapping return and regime observations after alignment.")
    try:
        finite_returns = np.isfinite(clean_returns.to_numpy(dtype=float)).all()
    except (TypeError, ValueError) as exc:
        raise ValueError("returns must contain only numeric values.") from exc
    if not finite_returns:
        raise ValueError("returns must contain only finite values.")
    return clean_returns, aligned_regimes


def _ledoit_wolf_alpha(
    observations: np.ndarray,
    sample_covariance: np.ndarray,
    target: np.ndarray,
) -> float:
    """Return the Ledoit-Wolf optimal shrinkage intensity toward a target.

    ``alpha`` minimizes the expected quadratic loss of the blended estimate
    ``(1 - alpha) * S + alpha * T``. Its closed form (Ledoit & Wolf 2004)
    scales the variance of each sample covariance entry by the squared
    distance between ``S`` and ``T``, so sparse or short histories shrink
    aggressively while long samples keep their empirical covariance.
    """

    observations = np.asarray(observations, dtype=float)
    sample_covariance = np.asarray(sample_covariance, dtype=float)
    target = np.asarray(target, dtype=float)
    n, p = observations.shape
    if n < 2 or p == 0:
        return 1.0
    deviations = observations - observations.mean(axis=0)
    cross = np.einsum("ti,tj->tij", deviations, deviations)
    biased = cross.mean(axis=0)
    unbiased = biased * n / (n - 1.0)
    variance = n / (n - 1.0) ** 3 * np.square(cross - biased).sum(axis=0)
    numerator = float(variance.sum())
    denominator = float(np.square(unbiased - target).sum())
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= 0:
        return 1.0
    return float(np.clip(numerator / denominator, 0.0, 1.0))


def _blend_covariance(
    local_covariance: pd.DataFrame,
    global_covariance: pd.DataFrame,
    shrinkage: float | None,
    observations: np.ndarray | None = None,
) -> pd.DataFrame:
    if shrinkage is None:
        alpha = (
            _ledoit_wolf_alpha(
                observations, local_covariance.to_numpy(dtype=float), global_covariance.to_numpy(dtype=float)
            )
            if observations is not None and len(observations) > 1
            else 1.0
        )
        blended = (1.0 - alpha) * local_covariance + alpha * global_covariance
        psd = nearest_psd_higham(blended.to_numpy(dtype=float))
        return pd.DataFrame(psd, index=blended.index, columns=blended.columns)
    if not 0 <= shrinkage <= 1:
        raise ValueError("shrinkage must be between 0 and 1.")
    blended = (1.0 - shrinkage) * local_covariance + shrinkage * global_covariance
    psd = nearest_psd(blended.to_numpy(dtype=float))
    return pd.DataFrame(psd, index=blended.index, columns=blended.columns)


def _apply_correlation_overrides(
    covariance: pd.DataFrame,
    overrides: Mapping[tuple[str, str], float] | None,
    override_weight: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not overrides:
        correlation = covariance_to_correlation(covariance)
        return covariance, correlation

    if not 0 <= override_weight <= 1:
        raise ValueError("override_weight must be between 0 and 1.")

    correlation = covariance_to_correlation(covariance)
    for pair, target in overrides.items():
        asset_a, asset_b = pair
        if asset_a not in correlation.index or asset_b not in correlation.columns:
            raise KeyError(f"Unknown correlation override pair: {pair}")
        current = correlation.loc[asset_a, asset_b]
        blended_target = (1.0 - override_weight) * current + override_weight * float(target)
        correlation.loc[asset_a, asset_b] = blended_target
        correlation.loc[asset_b, asset_a] = blended_target

    correlation = nearest_correlation(correlation)
    volatility = np.sqrt(np.clip(np.diag(covariance.to_numpy(dtype=float)), 1e-12, None))
    adjusted_cov = correlation.to_numpy(dtype=float) * np.outer(volatility, volatility)
    adjusted_covariance = pd.DataFrame(adjusted_cov, index=covariance.index, columns=covariance.columns)
    return adjusted_covariance, correlation


def estimate_regime_moments(
    returns: pd.DataFrame,
    regimes: pd.Series,
    states: list[str] | None = None,
    min_observations: int = 12,
    shrinkage: float | None = None,
    correlation_overrides: CorrelationOverrides | None = None,
    override_weight: float = 1.0,
    regime_lag_periods: int = 0,
) -> dict[str, RegimeMoments]:
    """Estimate asset mean/covariance/correlation separately for each regime.

    ``shrinkage=None`` selects the Ledoit-Wolf optimal shrinkage intensity
    toward the full-sample covariance instead of a fixed blend.
    """

    if returns.empty:
        raise ValueError("returns must not be empty.")
    if min_observations <= 0:
        raise ValueError("min_observations must be positive.")

    state_list = states or REGIME_ORDER
    clean_returns, aligned_regimes = _clean_aligned_returns(
        returns,
        regimes,
        lag_periods=regime_lag_periods,
    )

    global_mean = clean_returns.mean()
    global_covariance = clean_returns.cov()
    if global_covariance.isna().any().any():
        raise ValueError("Global covariance contains NaN values. Check return history.")

    moments: dict[str, RegimeMoments] = {}
    for state in state_list:
        state_returns = clean_returns.loc[aligned_regimes == state]
        observations = int(len(state_returns))

        if observations >= min_observations:
            local_mean = state_returns.mean()
            local_covariance = state_returns.cov()
        elif observations > 1:
            weight = observations / max(min_observations, 1)
            local_mean = weight * state_returns.mean() + (1.0 - weight) * global_mean
            if shrinkage is None:
                local_covariance = state_returns.cov()
            else:
                local_covariance = weight * state_returns.cov() + (1.0 - weight) * global_covariance
        else:
            local_mean = global_mean
            local_covariance = global_covariance

        local_covariance = local_covariance.reindex(
            index=clean_returns.columns, columns=clean_returns.columns
        )
        local_covariance = local_covariance.fillna(global_covariance)
        covariance = _blend_covariance(
            local_covariance,
            global_covariance,
            shrinkage,
            observations=state_returns.to_numpy(dtype=float) if observations > 1 else None,
        )

        state_overrides = correlation_overrides.get(state) if correlation_overrides else None
        covariance, correlation = _apply_correlation_overrides(
            covariance,
            state_overrides,
            override_weight=override_weight,
        )

        moments[state] = RegimeMoments(
            mean=local_mean.reindex(clean_returns.columns),
            covariance=covariance,
            correlation=correlation,
            observations=observations,
        )

    return moments


def calibrate_quadrant_model(
    returns: pd.DataFrame,
    macro: pd.DataFrame,
    growth_col: str = "growth",
    inflation_col: str = "inflation",
    growth_threshold: ThresholdSpec = "median",
    inflation_threshold: ThresholdSpec = "median",
    transition_smoothing: float = 1.0,
    min_observations: int = 12,
    shrinkage: float | None = None,
    correlation_overrides: CorrelationOverrides | None = None,
    override_weight: float = 1.0,
    macro_lag_periods: int = 0,
    frequency: str = "M",
    threshold_window: int | None = None,
    min_regime_duration: int = 1,
) -> ScenarioModel:
    """Calibrate a full four-quadrant Markov Monte Carlo model."""

    regimes = classify_quadrants(
        macro,
        growth_col=growth_col,
        inflation_col=inflation_col,
        growth_threshold=growth_threshold,
        inflation_threshold=inflation_threshold,
        threshold_window=threshold_window,
    )
    transition_matrix = estimate_transition_matrix(
        regimes,
        states=REGIME_ORDER,
        smoothing=transition_smoothing,
    )
    moments = estimate_regime_moments(
        returns=returns,
        regimes=regimes,
        states=REGIME_ORDER,
        min_observations=min_observations,
        shrinkage=shrinkage,
        correlation_overrides=correlation_overrides,
        override_weight=override_weight,
        regime_lag_periods=macro_lag_periods,
    )

    clean_returns, aligned_regimes = _clean_aligned_returns(
        returns,
        regimes,
        lag_periods=macro_lag_periods,
    )
    historical_returns = {state: clean_returns.loc[aligned_regimes == state].copy() for state in REGIME_ORDER}

    model = ScenarioModel(
        states=REGIME_ORDER.copy(),
        transition_matrix=transition_matrix,
        moments=moments,
        frequency=frequency,
        historical_returns=historical_returns,
        metadata={
            "growth_col": growth_col,
            "inflation_col": inflation_col,
            "growth_threshold": growth_threshold,
            "inflation_threshold": inflation_threshold,
            "macro_lag_periods": macro_lag_periods,
            "min_observations": min_observations,
            "shrinkage": shrinkage,
            "transition_smoothing": transition_smoothing,
            "threshold_window": threshold_window,
            "model_kind": "quadrant",
            "sojourn_durations": sojourn_durations(regimes, REGIME_ORDER, min_length=min_regime_duration),
        },
    )
    model.validate()
    return model
