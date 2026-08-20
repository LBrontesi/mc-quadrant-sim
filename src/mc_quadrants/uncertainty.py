from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from mc_quadrants.calibration import CorrelationOverrides, calibrate_quadrant_model
from mc_quadrants.regimes import ThresholdSpec
from mc_quadrants.types import ScenarioModel


def stationary_bootstrap_indices(
    observations: int,
    mean_block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw stationary-bootstrap indexes with geometrically sized blocks."""

    if observations <= 0:
        raise ValueError("observations must be positive.")
    if mean_block_size <= 0:
        raise ValueError("mean_block_size must be positive.")
    restart_probability = min(1.0, 1.0 / float(mean_block_size))
    indexes = np.empty(observations, dtype=int)
    indexes[0] = int(rng.integers(observations))
    for position in range(1, observations):
        if rng.random() < restart_probability:
            indexes[position] = int(rng.integers(observations))
        else:
            indexes[position] = (indexes[position - 1] + 1) % observations
    return indexes


def bootstrap_quadrant_models(
    returns: pd.DataFrame,
    macro: pd.DataFrame,
    draws: int,
    block_size: int = 12,
    random_seed: int | None = None,
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
    threshold_window: int | None = None,
    min_regime_duration: int = 5,
    probabilistic_regimes: bool = False,
    regime_temperature: float = 0.35,
    regime_smoothing_window: int = 3,
    regime_hysteresis: float = 0.15,
    regime_confirmation_periods: int = 2,
    duration_prior_strength: float = 8.0,
    mean_prior_strength: float = 24.0,
    joint_macro: bool = False,
    structural_returns: bool = False,
    asset_classes: Mapping[str, str] | None = None,
    asset_durations: Mapping[str, float] | None = None,
    asset_income_yields: Mapping[str, float] | None = None,
    macro_model: str = "bvar_ensemble",
) -> list[ScenarioModel]:
    """Recalibrate complete parametric models on paired stationary bootstraps.

    Macro values are first aligned to the return calendar with the configured
    information lag. Each sampled row therefore keeps its macro information
    and cross-asset return vector together. The bootstrap changes calibration
    parameters; it does not replace the parametric return generator.
    """

    if draws <= 0:
        return []
    if block_size <= 0:
        raise ValueError("block_size must be positive.")
    macro_columns = [growth_col, inflation_col]
    active_rate_col = rate_col if rate_col and rate_col in macro.columns else None
    if active_rate_col is not None and active_rate_col not in macro_columns:
        macro_columns.append(active_rate_col)
    selected_macro = macro.loc[:, macro_columns].sort_index()
    aligned_macro = selected_macro.reindex(returns.sort_index().index, method="ffill")
    if macro_lag_periods:
        aligned_macro = aligned_macro.shift(int(macro_lag_periods))
    prefixed = aligned_macro.rename(columns=lambda column: f"__macro_{column}")
    panel = returns.sort_index().join(prefixed, how="inner").dropna()
    if len(panel) < max(min_observations * 2, 24):
        raise ValueError("Parameter uncertainty requires at least 24 complete paired observations.")

    rng = np.random.default_rng(random_seed)
    artificial_index = pd.date_range("2000-01-31", periods=len(panel), freq="ME")
    return_columns = list(returns.columns)
    models: list[ScenarioModel] = []
    for draw in range(int(draws)):
        indexes = stationary_bootstrap_indices(len(panel), int(block_size), rng)
        sampled = panel.iloc[indexes].copy()
        sampled.index = artificial_index
        sampled_returns = sampled.loc[:, return_columns]
        sampled_macro = sampled.loc[:, [f"__macro_{column}" for column in macro_columns]].rename(
            columns={f"__macro_{column}": column for column in macro_columns}
        )
        sampled_macro.attrs.update(
            {
                "data_vintage": macro.attrs.get("data_vintage", "user_supplied"),
                "point_in_time": bool(macro.attrs.get("point_in_time", False)),
                "availability_aligned": bool(macro.attrs.get("availability_aligned", False)),
            }
        )
        model = calibrate_quadrant_model(
            sampled_returns,
            sampled_macro,
            growth_col=growth_col,
            inflation_col=inflation_col,
            rate_col=active_rate_col,
            growth_threshold=growth_threshold,
            inflation_threshold=inflation_threshold,
            transition_smoothing=transition_smoothing,
            min_observations=min_observations,
            shrinkage=shrinkage,
            correlation_overrides=correlation_overrides,
            override_weight=override_weight,
            macro_lag_periods=0,
            threshold_window=threshold_window,
            min_regime_duration=min_regime_duration,
            probabilistic_regimes=probabilistic_regimes,
            regime_temperature=regime_temperature,
            regime_smoothing_window=regime_smoothing_window,
            regime_hysteresis=regime_hysteresis,
            regime_confirmation_periods=regime_confirmation_periods,
            duration_prior_strength=duration_prior_strength,
            mean_prior_strength=mean_prior_strength,
            joint_macro=joint_macro,
            structural_returns=structural_returns,
            asset_classes=asset_classes,
            asset_durations=asset_durations,
            asset_income_yields=asset_income_yields,
            macro_model=macro_model,
        )
        model.metadata["bootstrap_draw"] = draw
        model.metadata["source_macro_lag_periods"] = int(macro_lag_periods)
        models.append(model)
    return models


def summarize_parameter_models(
    models: list[ScenarioModel],
    weights: Mapping[str, float],
    periods_per_year: float = 12.0,
) -> pd.DataFrame:
    """Summarize economically important variation across calibration draws."""

    rows: list[dict[str, float | int]] = []
    for draw, model in enumerate(models):
        vector = pd.Series(weights, dtype=float).reindex(model.assets).fillna(0.0)
        vector = vector / vector.sum()
        stationary = np.full(len(model.states), 1.0 / len(model.states))
        transition = model.transition_matrix.to_numpy(dtype=float)
        for _ in range(500):
            following = stationary @ transition
            if np.max(np.abs(following - stationary)) < 1e-12:
                break
            stationary = following
        state_means = np.array(
            [float(model.moments[state].mean @ vector) for state in model.states],
            dtype=float,
        )
        state_variances = np.array(
            [
                float(vector @ model.moments[state].covariance @ vector)
                for state in model.states
            ],
            dtype=float,
        )
        monthly_mean = float(stationary @ state_means)
        total_variance = float(
            stationary @ (state_variances + np.square(state_means - monthly_mean))
        )
        rows.append(
            {
                "draw": draw,
                "annualized_return": monthly_mean * periods_per_year,
                "annualized_volatility": np.sqrt(max(total_variance, 0.0) * periods_per_year),
                "average_persistence": float(np.mean(np.diag(transition))),
            }
        )
    return pd.DataFrame(rows)
