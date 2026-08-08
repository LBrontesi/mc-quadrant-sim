from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mc_quadrants.regimes import REGIME_ORDER, classify_quadrants
from mc_quadrants.types import ScenarioModel, SimulationResult


@dataclass(frozen=True)
class CalibrationDiagnostics:
    """Historical coverage and stability checks for a calibrated model."""

    regime_summary: pd.DataFrame
    transition_counts: pd.DataFrame
    warnings: list[str]


def _transition_counts(regimes: pd.Series) -> pd.DataFrame:
    clean = regimes.dropna().sort_index().astype(str)
    counts = pd.DataFrame(0, index=REGIME_ORDER, columns=REGIME_ORDER, dtype=int)
    for current, next_state in zip(clean.iloc[:-1], clean.iloc[1:]):
        if current in counts.index and next_state in counts.columns:
            counts.loc[current, next_state] += 1
    return counts


def build_calibration_diagnostics(
    model: ScenarioModel,
    returns: pd.DataFrame,
    macro: pd.DataFrame,
    growth_col: str,
    inflation_col: str,
    growth_threshold: str | float,
    inflation_threshold: str | float,
    macro_lag_periods: int = 0,
    threshold_window: int | None = None,
) -> CalibrationDiagnostics:
    """Check regime coverage, covariance conditioning, and transition data."""

    regimes = classify_quadrants(
        macro,
        growth_col=growth_col,
        inflation_col=inflation_col,
        growth_threshold=growth_threshold,
        inflation_threshold=inflation_threshold,
        threshold_window=threshold_window,
    )
    lagged_regimes = regimes.sort_index().shift(macro_lag_periods)
    aligned = lagged_regimes.dropna().reindex(returns.sort_index().index, method="ffill")
    clean_returns = returns.sort_index().dropna(how="any")
    aligned = aligned.reindex(clean_returns.index)
    valid = aligned.notna()
    aligned = aligned.loc[valid].astype(str)

    total = max(len(aligned), 1)
    rows: list[dict[str, object]] = []
    warnings: list[str] = []
    minimum_observations = int(model.metadata.get("min_observations", 12))
    for state in REGIME_ORDER:
        moments = model.moments[state]
        observations = int((aligned == state).sum())
        covariance = moments.covariance.to_numpy(dtype=float)
        condition_number = float(np.linalg.cond(covariance)) if covariance.size else np.nan
        rows.append(
            {
                "regime": state,
                "observations": observations,
                "share": observations / total,
                "covariance_condition_number": condition_number,
                "shrinkage": (
                    float(model.metadata["shrinkage"])
                    if model.metadata.get("shrinkage") is not None
                    else np.nan
                ),
            }
        )
        if observations < minimum_observations:
            warnings.append(
                f"{state} has {observations} aligned observations; "
                f"the configured minimum is {minimum_observations}."
            )
        if not np.isfinite(condition_number) or condition_number > 1e8:
            warnings.append(f"{state} covariance is poorly conditioned.")

    if macro_lag_periods > 0:
        warnings.append(
            f"Macro regimes are lagged by {macro_lag_periods} period(s) to reduce look-ahead bias."
        )
    if threshold_window:
        warnings.append(
            f"Thresholds use causal expanding windows with {threshold_window} minimum prior "
            "observations; the earliest macro observations are left unclassified."
        )
    if len(aligned) < 24:
        warnings.append("The aligned calibration sample contains fewer than 24 observations.")

    return CalibrationDiagnostics(
        regime_summary=pd.DataFrame(rows),
        transition_counts=_transition_counts(regimes),
        warnings=warnings,
    )


def build_hmm_diagnostics(
    model: ScenarioModel,
    regimes: pd.Series,
) -> CalibrationDiagnostics:
    """Regime coverage and conditioning checks for a fitted HMM model."""

    rows: list[dict[str, object]] = []
    warnings: list[str] = []
    total = max(int(len(regimes)), 1)
    minimum_observations = max(int(model.metadata.get("n_states", 4)) * 4, 12)
    for state in model.states:
        moments = model.moments[state]
        observations = int(moments.observations)
        covariance = moments.covariance.to_numpy(dtype=float)
        condition_number = float(np.linalg.cond(covariance)) if covariance.size else np.nan
        rows.append(
            {
                "regime": state,
                "observations": observations,
                "share": observations / total,
                "covariance_condition_number": condition_number,
                "shrinkage": np.nan,
            }
        )
        if observations < minimum_observations:
            warnings.append(
                f"{state} has {observations} fitted observations; "
                f"the minimum is {minimum_observations} for stable moments."
            )
        if not np.isfinite(condition_number) or condition_number > 1e8:
            warnings.append(f"{state} covariance is poorly conditioned.")
    log_likelihood = model.metadata.get("log_likelihood")
    if log_likelihood is not None:
        warnings.append(
            f"HMM fitted {model.metadata.get('n_states')} states "
            f"in {model.metadata.get('iterations')} iterations "
            f"(log-likelihood {log_likelihood:.1f})."
        )
    return CalibrationDiagnostics(
        regime_summary=pd.DataFrame(rows),
        transition_counts=_transition_counts(regimes),
        warnings=warnings,
    )


def simulation_regime_summary(result: SimulationResult) -> pd.DataFrame:
    """Return simulated regime frequency by state."""

    counts = pd.Series(result.regimes.ravel(), dtype="string").value_counts()
    total = max(int(counts.sum()), 1)
    return pd.DataFrame(
        {
            "regime": result.states,
            "observations": [int(counts.get(state, 0)) for state in result.states],
            "share": [float(counts.get(state, 0)) / total for state in result.states],
        }
    )
