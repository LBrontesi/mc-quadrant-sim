from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from mc_quadrants.hsmm import fit_quadrant_hsmm
from mc_quadrants.matrix import (
    covariance_to_correlation,
    nearest_correlation,
    nearest_psd,
    nearest_psd_higham,
)
from mc_quadrants.regimes import (
    REGIME_ORDER,
    ThresholdSpec,
    classify_persistent_quadrants,
    resolve_threshold,
    smooth_macro_for_regimes,
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
    mean_prior_strength: float = 0.0,
) -> dict[str, RegimeMoments]:
    """Estimate asset mean/covariance/correlation separately for each regime.

    ``shrinkage=None`` selects the Ledoit-Wolf optimal shrinkage intensity
    toward the full-sample covariance instead of a fixed blend.
    """

    if returns.empty:
        raise ValueError("returns must not be empty.")
    if min_observations <= 0:
        raise ValueError("min_observations must be positive.")
    if not np.isfinite(mean_prior_strength) or mean_prior_strength < 0:
        raise ValueError("mean_prior_strength must be finite and non-negative.")

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

        if observations > 0 and mean_prior_strength > 0:
            mean_weight = observations / (observations + float(mean_prior_strength))
            local_mean = mean_weight * local_mean + (1.0 - mean_weight) * global_mean

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


def _align_probabilities_to_returns(
    probabilities: pd.DataFrame,
    returns: pd.DataFrame,
    lag_periods: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align causal macro state probabilities to complete return observations."""

    if lag_periods < 0:
        raise ValueError("lag_periods must be non-negative.")
    clean_returns = returns.sort_index().dropna(how="any")
    shifted = probabilities.sort_index().shift(lag_periods).dropna(how="all")
    aligned = shifted.reindex(clean_returns.index, method="ffill")
    valid = aligned.notna().all(axis=1)
    clean_returns = clean_returns.loc[valid]
    aligned = aligned.loc[valid]
    if clean_returns.empty:
        raise ValueError("No overlapping return and probabilistic regime observations after alignment.")
    return clean_returns, aligned


def estimate_weighted_regime_moments(
    returns: pd.DataFrame,
    probabilities: pd.DataFrame,
    states: list[str],
    min_observations: int = 12,
    shrinkage: float | None = None,
    correlation_overrides: CorrelationOverrides | None = None,
    override_weight: float = 1.0,
    regime_lag_periods: int = 0,
    mean_prior_strength: float = 24.0,
) -> dict[str, RegimeMoments]:
    """Estimate parametric moments using soft state-membership weights."""

    clean_returns, aligned = _align_probabilities_to_returns(
        probabilities,
        returns,
        lag_periods=regime_lag_periods,
    )
    values = clean_returns.to_numpy(dtype=float)
    global_mean = clean_returns.mean()
    global_covariance = clean_returns.cov()
    moments: dict[str, RegimeMoments] = {}
    for state in states:
        weights = aligned[state].to_numpy(dtype=float)
        weight_sum = float(weights.sum())
        effective = int(round(weight_sum))
        if weight_sum <= 1e-9:
            local_mean = global_mean
            local_covariance = global_covariance
            weighted_observations = values
        else:
            weighted_mean = np.average(values, axis=0, weights=weights)
            centered = values - weighted_mean
            denominator = weight_sum - float(np.square(weights).sum()) / weight_sum
            if denominator > 1e-9:
                covariance_values = (centered.T * weights) @ centered / denominator
                local_covariance = pd.DataFrame(
                    covariance_values,
                    index=clean_returns.columns,
                    columns=clean_returns.columns,
                )
            else:
                local_covariance = global_covariance
            reliability = weight_sum / max(weight_sum + float(mean_prior_strength), 1e-9)
            local_mean = pd.Series(weighted_mean, index=clean_returns.columns)
            local_mean = reliability * local_mean + (1.0 - reliability) * global_mean
            weighted_observations = values * np.sqrt(np.maximum(weights, 0.0))[:, None]
        covariance = _blend_covariance(
            local_covariance,
            global_covariance,
            shrinkage,
            observations=weighted_observations if len(weighted_observations) > 1 else None,
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
            observations=effective,
        )
    return moments


def _weighted_covariance(values: np.ndarray, weights: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    weight_sum = float(weights.sum())
    if weight_sum <= 1.0:
        return fallback.copy()
    mean = np.average(values, axis=0, weights=weights)
    centered = values - mean
    denominator = weight_sum - float(np.square(weights).sum()) / weight_sum
    if denominator <= 1e-9:
        return fallback.copy()
    return nearest_psd((centered.T * weights) @ centered / denominator)


def _fit_joint_macro_dynamics(
    macro: pd.DataFrame,
    returns: pd.DataFrame,
    regimes: pd.Series,
    probabilities: pd.DataFrame | None,
    moments: dict[str, RegimeMoments],
    growth_col: str,
    inflation_col: str,
    rate_col: str | None,
    growth_threshold: ThresholdSpec,
    inflation_threshold: ThresholdSpec,
    temperature: float,
    ridge: float = 1e-3,
) -> dict[str, object]:
    """Fit a compact regime-conditioned macro VAR and return-factor link.

    Growth and inflation always define the four economic quadrants.  When a
    short-rate column is available it joins the VAR as a third state variable,
    so policy-rate innovations can co-move with inflation, growth, and returns
    without changing the meaning of the quadrant labels.
    """

    columns = [growth_col, inflation_col]
    active_rate_col = None
    if rate_col and rate_col in macro.columns and rate_col not in columns:
        active_rate_col = str(rate_col)
        columns.append(active_rate_col)
    macro_clean = macro.loc[:, columns].apply(pd.to_numeric, errors="coerce").dropna().sort_index()
    if len(macro_clean) < 24:
        raise ValueError("Joint macro dynamics require at least 24 complete macro observations.")
    macro_values = macro_clean.to_numpy(dtype=float)
    global_mean = macro_values.mean(axis=0)
    x = macro_values[:-1] - global_mean
    y = macro_values[1:] - global_mean
    coefficient = np.linalg.solve(x.T @ x + ridge * np.eye(len(columns)), x.T @ y)
    eigenvalues = np.linalg.eigvals(coefficient)
    radius = float(np.max(np.abs(eigenvalues))) if len(eigenvalues) else 0.0
    if radius >= 0.98:
        coefficient *= 0.98 / radius
    residuals = y - x @ coefficient
    global_residual_covariance = nearest_psd(np.atleast_2d(np.cov(residuals, rowvar=False)))

    if probabilities is not None:
        macro_membership = probabilities.reindex(macro_clean.index, method="ffill")
    else:
        hard = regimes.reindex(macro_clean.index, method="ffill")
        macro_membership = pd.DataFrame(
            {state: (hard == state).astype(float) for state in REGIME_ORDER},
            index=macro_clean.index,
        )
    state_centers: dict[str, list[float]] = {}
    state_covariances: dict[str, list[list[float]]] = {}
    for state in REGIME_ORDER:
        weights = macro_membership[state].fillna(0.0).to_numpy(dtype=float)
        if weights.sum() > 1e-9:
            center = np.average(macro_values, axis=0, weights=weights)
        else:
            center = global_mean
        residual_weights = weights[1:]
        covariance = _weighted_covariance(
            residuals,
            residual_weights,
            global_residual_covariance,
        )
        state_centers[state] = center.tolist()
        state_covariances[state] = covariance.tolist()

    macro_changes = macro_clean.diff().dropna()
    aligned_changes = macro_changes.reindex(returns.index, method="ffill")
    joint = returns.join(aligned_changes.add_prefix("__macro_"), how="inner").dropna()
    return_values = joint.loc[:, returns.columns].to_numpy(dtype=float)
    change_values = joint.loc[:, [f"__macro_{column}" for column in columns]].to_numpy(dtype=float)
    centered_returns = return_values - return_values.mean(axis=0)
    betas = np.linalg.solve(
        change_values.T @ change_values + ridge * np.eye(len(columns)),
        change_values.T @ centered_returns,
    )
    explained = betas.T @ global_residual_covariance @ betas
    total_variance = np.maximum(np.diag(np.atleast_2d(np.cov(return_values, rowvar=False))), 1e-12)
    explained_share = np.max(np.diag(explained) / total_variance)
    if np.isfinite(explained_share) and explained_share > 0.25:
        betas *= np.sqrt(0.25 / explained_share)

    residual_covariances: dict[str, list[list[float]]] = {}
    for state in REGIME_ORDER:
        macro_effect = betas.T @ np.asarray(state_covariances[state], dtype=float) @ betas
        residual_covariances[state] = nearest_psd(
            moments[state].covariance.to_numpy(dtype=float) - macro_effect
        ).tolist()

    thresholds = [
        resolve_threshold(macro_clean[growth_col], growth_threshold),
        resolve_threshold(macro_clean[inflation_col], inflation_threshold),
    ]
    scales = np.maximum(
        macro_clean.loc[:, [growth_col, inflation_col]].std(ddof=1).to_numpy(dtype=float)
        * temperature,
        1e-6,
    )
    inflation_scale_hint = float(macro_clean[inflation_col].abs().quantile(0.90))
    rate_is_percent = False
    rate_bounds = None
    if active_rate_col is not None:
        rate_values = macro_clean[active_rate_col]
        rate_is_percent = float(rate_values.abs().quantile(0.90)) >= 0.50
        rate_bounds = [-5.0, 50.0] if rate_is_percent else [-0.05, 0.50]
    return {
        "columns": columns,
        "latest": macro_values[-1].tolist(),
        "global_mean": global_mean.tolist(),
        "var_coefficient": coefficient.tolist(),
        "state_centers": state_centers,
        "state_innovation_covariances": state_covariances,
        "return_betas": betas.tolist(),
        "return_residual_covariances": residual_covariances,
        "thresholds": thresholds,
        "probability_scales": scales.tolist(),
        "inflation_is_percent": inflation_scale_hint >= 0.50,
        "rate_col": active_rate_col,
        "rate_is_percent": rate_is_percent,
        "rate_bounds": rate_bounds,
    }


def calibrate_quadrant_model(
    returns: pd.DataFrame,
    macro: pd.DataFrame,
    growth_col: str = "growth",
    inflation_col: str = "inflation",
    rate_col: str | None = "interest_rate",
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
    min_regime_duration: int = 5,
    probabilistic_regimes: bool = False,
    regime_temperature: float = 0.35,
    regime_smoothing_window: int = 3,
    regime_hysteresis: float = 0.15,
    regime_confirmation_periods: int = 2,
    duration_prior_strength: float = 8.0,
    mean_prior_strength: float = 0.0,
    joint_macro: bool = False,
    hsmm_max_iterations: int = 30,
) -> ScenarioModel:
    """Calibrate a full four-quadrant Markov Monte Carlo model."""

    initial_regimes = classify_persistent_quadrants(
        macro,
        growth_col=growth_col,
        inflation_col=inflation_col,
        growth_threshold=growth_threshold,
        inflation_threshold=inflation_threshold,
        threshold_window=threshold_window,
        smoothing_window=regime_smoothing_window,
        hysteresis=regime_hysteresis,
        confirmation_periods=regime_confirmation_periods,
    )
    smoothed_macro = smooth_macro_for_regimes(
        macro,
        growth_col=growth_col,
        inflation_col=inflation_col,
        smoothing_window=regime_smoothing_window,
    )
    hsmm = fit_quadrant_hsmm(
        smoothed_macro,
        initial_regimes,
        states=REGIME_ORDER,
        columns=(growth_col, inflation_col),
        min_duration=min_regime_duration,
        duration_prior_strength=duration_prior_strength,
        transition_smoothing=transition_smoothing,
        max_iterations=hsmm_max_iterations,
    )
    regimes = hsmm.viterbi_path
    transition_matrix = hsmm.transition_matrix
    probabilities: pd.DataFrame | None = None
    if probabilistic_regimes:
        probabilities = hsmm.filtered_probabilities
        moments = estimate_weighted_regime_moments(
            returns=returns,
            probabilities=probabilities,
            states=REGIME_ORDER,
            min_observations=min_observations,
            shrinkage=shrinkage,
            correlation_overrides=correlation_overrides,
            override_weight=override_weight,
            regime_lag_periods=macro_lag_periods,
            mean_prior_strength=mean_prior_strength,
        )
    else:
        moments = estimate_regime_moments(
            returns=returns,
            regimes=regimes,
            states=REGIME_ORDER,
            min_observations=min_observations,
            shrinkage=shrinkage,
            correlation_overrides=correlation_overrides,
            override_weight=override_weight,
            regime_lag_periods=macro_lag_periods,
            mean_prior_strength=mean_prior_strength,
        )

    clean_returns, aligned_regimes = _clean_aligned_returns(
        returns,
        regimes,
        lag_periods=macro_lag_periods,
    )
    historical_returns = {state: clean_returns.loc[aligned_regimes == state].copy() for state in REGIME_ORDER}

    inflation_values = pd.to_numeric(macro[inflation_col], errors="coerce")
    if inflation_values.notna().any():
        inflation_scale_hint = float(inflation_values.abs().quantile(0.90))
        inflation_fraction = inflation_values / 100.0 if inflation_scale_hint >= 0.50 else inflation_values
        state_inflation = {
            state: float(inflation_fraction[regimes == state].mean())
            for state in REGIME_ORDER
            if (regimes == state).any()
        }
    else:
        state_inflation = {}

    active_rate_col = rate_col if rate_col and rate_col in macro.columns else None
    state_short_rate: dict[str, float] = {}
    rate_is_percent = False
    if active_rate_col is not None:
        rate_values = pd.to_numeric(macro[active_rate_col], errors="coerce")
        if rate_values.notna().any():
            rate_is_percent = float(rate_values.abs().quantile(0.90)) >= 0.50
            rate_fraction = rate_values / 100.0 if rate_is_percent else rate_values
            state_short_rate = {
                state: float(rate_fraction[regimes == state].mean())
                for state in REGIME_ORDER
                if (regimes == state).any()
            }

    duration_hazards = hsmm.duration_hazards
    expected_durations = hsmm.expected_duration_months
    metadata: dict[str, object] = {
        "growth_col": growth_col,
        "inflation_col": inflation_col,
        "rate_col": active_rate_col,
        "growth_threshold": growth_threshold,
        "inflation_threshold": inflation_threshold,
        "macro_lag_periods": macro_lag_periods,
        "min_observations": min_observations,
        "shrinkage": shrinkage,
        "mean_prior_strength": mean_prior_strength,
        "transition_smoothing": transition_smoothing,
        "threshold_window": threshold_window,
        "regime_smoothing_window": int(regime_smoothing_window),
        "regime_hysteresis": float(regime_hysteresis),
        "regime_confirmation_periods": int(regime_confirmation_periods),
        "model_kind": "quadrant",
        "regime_assignment": "probabilistic" if probabilistic_regimes else "hsmm_viterbi",
        "transition_estimator": "hsmm_forward_backward_joint_posteriors",
        "regime_temperature": regime_temperature,
        "duration_model_kind": "hidden_semi_markov_explicit_duration",
        "duration_prior_strength": float(duration_prior_strength),
        "min_regime_duration": int(min_regime_duration),
        "sojourn_durations": sojourn_durations(regimes, REGIME_ORDER, min_length=1),
        "duration_hazards": duration_hazards,
        "expected_duration_months": expected_durations,
        "hsmm_exit_transition_matrix": hsmm.exit_transition_matrix,
        "hsmm_filtered_probabilities": hsmm.filtered_probabilities,
        "hsmm_smoothed_probabilities": hsmm.smoothed_probabilities,
        "hsmm_latest_state_age_probabilities": hsmm.latest_state_age_probabilities,
        "hsmm_log_likelihood": hsmm.log_likelihood,
        "hsmm_iterations": hsmm.iterations,
        "hsmm_converged": hsmm.converged,
        "hsmm_max_duration": hsmm.max_duration,
        "hsmm_emission_means": hsmm.emission_means,
        "hsmm_emission_covariances": hsmm.emission_covariances,
        "state_inflation": state_inflation,
        "state_short_rate": state_short_rate,
        "rate_is_percent": rate_is_percent,
        "data_vintage": macro.attrs.get("data_vintage", "user_supplied"),
        "point_in_time": bool(macro.attrs.get("point_in_time", False)),
        "availability_aligned": bool(macro.attrs.get("availability_aligned", False)),
    }
    latest_probabilities = hsmm.filtered_probabilities.dropna().iloc[-1]
    metadata["latest_regime_probabilities"] = latest_probabilities.to_dict()
    if probabilities is not None:
        metadata["historical_regime_probabilities"] = probabilities
    if joint_macro:
        metadata["macro_dynamics"] = _fit_joint_macro_dynamics(
            macro,
            returns,
            regimes,
            probabilities,
            moments,
            growth_col,
            inflation_col,
            active_rate_col,
            growth_threshold,
            inflation_threshold,
            regime_temperature,
        )

    model = ScenarioModel(
        states=REGIME_ORDER.copy(),
        transition_matrix=transition_matrix,
        moments=moments,
        frequency=frequency,
        historical_returns=historical_returns,
        metadata=metadata,
    )
    model.validate()
    return model
