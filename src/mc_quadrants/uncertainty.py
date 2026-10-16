from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace

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
    volatility_model: str = "garch",
    economic_returns: Mapping | None = None,
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
    from mc_quadrants.economic_returns import (
        PREVIOUS_INFLATION,
        economic_assumptions,
        economic_columns,
        previous_column,
    )
    from mc_quadrants.structural import build_asset_profiles
    economic_options = economic_assumptions(economic_returns)
    source_macro = macro.sort_index().copy()
    if "cpi_monthly_log" in source_macro:
        macro_columns.append("cpi_monthly_log")
    physical_columns = []
    if economic_options:
        profiles = build_asset_profiles(list(returns), asset_classes=asset_classes, durations=asset_durations, income_yields=asset_income_yields)
        for column in economic_columns(profiles, economic_options):
            if column not in source_macro:
                raise ValueError(f"Missing observed economic column {column}.")
            if column not in macro_columns:
                macro_columns.append(column)
            lag_column = previous_column(column)
            source_macro[lag_column] = source_macro[column].shift(1)
            macro_columns.append(lag_column)
            physical_columns.extend([column, lag_column])
        source_macro[PREVIOUS_INFLATION] = source_macro[inflation_col].shift(1)
        macro_columns.append(PREVIOUS_INFLATION)
        physical_columns.append(PREVIOUS_INFLATION)
        for column in economic_options["equity_valuation_columns"].values():
            if column not in macro_columns:
                macro_columns.append(column)
        if "tips_inflation_log" in source_macro:
            macro_columns.append("tips_inflation_log")
            physical_columns.append("tips_inflation_log")
    selected_macro = source_macro.loc[:, macro_columns].sort_index()
    aligned_macro = selected_macro.reindex(returns.sort_index().index, method="ffill")
    if macro_lag_periods:
        unlagged_macro = aligned_macro
        aligned_macro = aligned_macro.shift(int(macro_lag_periods))
        if economic_options:
            # Market repricing is contemporaneous even when macro predictors lag.
            for column in physical_columns:
                if column in unlagged_macro:
                    aligned_macro[column] = unlagged_macro[column]
    prefixed = aligned_macro.rename(columns=lambda column: f"__macro_{column}")
    panel = returns.sort_index().join(prefixed, how="inner").dropna()
    if len(panel) < max(min_observations * 2, 24):
        raise ValueError("Parameter uncertainty requires at least 24 complete paired observations.")

    rng = np.random.default_rng(random_seed)
    artificial_index = pd.date_range("2000-01-31", periods=len(panel), freq="ME")
    return_columns = list(returns.columns)
    models: list[ScenarioModel] = []
    anchor_rng = np.random.default_rng(None if random_seed is None else np.random.SeedSequence([random_seed, 98731]))
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
        draw_options = dict(economic_options) if economic_options else None
        if draw_options or "cpi_monthly_log" in sampled_macro:
            # Exclude transitions that cross a bootstrap block boundary.
            sampled_macro.attrs["valid_macro_transitions"] = (np.diff(indexes) == 1).tolist()
        if draw_options:
            if draw_options["return_anchors"] and draw_options["anchor_std"]:
                from statistics import NormalDist
                location = np.log1p(draw_options["equity_real_return"])
                scale = draw_options["anchor_std"]
                normal = NormalDist()
                lower = normal.cdf((np.log(0.5) - location) / scale)
                upper = normal.cdf((np.log(1.5) - location) / scale)
                probability = np.clip(anchor_rng.uniform(lower, upper), np.finfo(float).eps, 1 - np.finfo(float).eps)
                log_real = location + scale * normal.inv_cdf(float(probability))
                draw_options["equity_real_return"] = float(np.expm1(log_real))
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
            volatility_model=volatility_model,
            economic_returns=draw_options,
        )
        model.metadata["bootstrap_draw"] = draw
        model.metadata["source_macro_lag_periods"] = int(macro_lag_periods)
        models.append(model)
    return models


def anchor_parameter_models(models: list[ScenarioModel], reference: ScenarioModel) -> list[ScenarioModel]:
    """Vary fitted parameters, not the observed forecast origin; leave caches intact."""
    anchored = []
    for model in models:
        metadata = deepcopy(model.metadata)
        moments = model.moments
        if "economic_returns" in metadata and "economic_returns" in reference.metadata:
            economic = metadata["economic_returns"]
            current = reference.metadata["economic_returns"]
            difference = pd.Series(np.asarray(current["current_carry"]) - np.asarray(economic["current_carry"]), index=model.assets)
            if economic["assumptions"]["return_anchors"] and not economic["assumptions"]["rate_stress"]:
                # The bond anchor is today's observed yield, not the randomly
                # selected final row of a bootstrap history.
                for asset, profile in metadata["asset_profiles"].items():
                    if profile["asset_class"] in {"government_bond", "cash", "inflation_linked_bond", "corporate_bond"}:
                        channel = current["channels"][asset]
                        level = current["latest_yields"][channel["yield_column"]] + (current["latest_yields"][channel["spread_column"]] if channel["spread_column"] else 0)
                        target = float((np.log1p(level) + np.log1p(-channel["credit_loss"])) / 12)
                        if profile["asset_class"] == "inflation_linked_bond":
                            target += np.log1p(economic["assumptions"]["inflation_anchor"]) / 12
                        correction = economic["anchor_weight"] * (target - economic["monthly_prior_targets"][asset])
                        difference[asset] += correction
                        economic["monthly_prior_targets"][asset] = target
                        economic["monthly_mean_shift"][asset] += correction
            for asset, valuation in economic.get("valuation", {}).items():
                current_multiple = current["valuation"][asset]["latest"]
                assumptions = economic["assumptions"]
                adjustment = assumptions["valuation_weight"] * np.log(2) / assumptions["valuation_half_life_years"] * np.log(valuation["reference"] / current_multiple)
                correction = economic["anchor_weight"] * (adjustment - valuation["annual_log_adjustment"]) / 12
                difference[asset] += correction
                economic["monthly_prior_targets"][asset] += (adjustment - valuation["annual_log_adjustment"]) / 12
                economic["monthly_mean_shift"][asset] += correction
                valuation.update(latest=current_multiple, annual_log_adjustment=float(adjustment))
            moments = {state: replace(moment, mean=moment.mean + difference) for state, moment in moments.items()}
            for key in ("latest_yield", "latest_yields", "latest_inflation", "current_carry"):
                economic[key] = deepcopy(current[key])
        for key in ("latest_regime_probabilities", "hsmm_latest_state_age_probabilities", "cpi_process"):
            if key in reference.metadata:
                metadata[key] = deepcopy(reference.metadata[key])
        # Hazard tables can have different bootstrap lengths. Pad them to retain
        # every age represented by the reference posterior, using the terminal hazard.
        posterior = metadata.get("hsmm_latest_state_age_probabilities")
        if posterior is not None:
            width = max(len(values) for values in posterior.values()) if isinstance(posterior, Mapping) else np.asarray(posterior).shape[-1]
            metadata["duration_hazards"] = {
                state: np.pad(np.asarray(values), (0, max(0, width-len(values))), mode="edge").tolist()
                for state, values in metadata["duration_hazards"].items()
            }
        if "macro_dynamics" in metadata and "macro_dynamics" in reference.metadata:
            metadata["macro_dynamics"]["latest"] = deepcopy(reference.metadata["macro_dynamics"]["latest"])
            if metadata["macro_dynamics"].get("cpi_dynamics"):
                metadata["macro_dynamics"]["cpi_dynamics"]["latest_monthly_log"] = reference.metadata["macro_dynamics"]["cpi_dynamics"]["latest_monthly_log"]
            for column, specification in metadata["macro_dynamics"].get("yield_dynamics", {}).items():
                specification["latest_drift"] = reference.metadata["macro_dynamics"]["yield_dynamics"][column]["latest_drift"]
            stress = metadata["macro_dynamics"].get("stress_dynamics")
            if stress:
                observed_stress = reference.metadata["macro_dynamics"]["stress_dynamics"]
                for key in ("latest_news", "latest_variance"):
                    stress[key] = deepcopy(observed_stress[key])
        reference_qarch = reference.metadata.get("dependence_fit", {}).get("qarch")
        qarch_fit = metadata.get("dependence_fit", {}).get("qarch")
        if reference_qarch and qarch_fit:
            from mc_quadrants.qarch import kernel, variance_ratio
            for key in ("history", "history_observations"):
                qarch_fit[key] = deepcopy(reference_qarch[key])
            qarch_fit["next_variance_ratio"] = variance_ratio(
                np.asarray(qarch_fit["history"]), kernel(qarch_fit["weights"]),
                qarch_fit["history_observations"],
            ).tolist()
        metadata["forecast_origin"] = "observed_reference_state"
        anchored.append(replace(model, metadata=metadata, moments=moments))
    return anchored


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
                **{key: float(model.metadata.get("dependence_fit", {}).get(key, np.nan))
                   for key in ("garch_alpha", "garch_beta", "dcc_alpha", "dcc_beta", "dcc_asymmetry")},
                "mean_mnts_tail_index": float(np.mean([model.moments[state].mnts.tail_index for state in model.states])),
                "mean_mnts_tempering": float(np.mean([model.moments[state].mnts.tempering for state in model.states])),
            }
        )
    return pd.DataFrame(rows)
