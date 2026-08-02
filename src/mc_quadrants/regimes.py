from __future__ import annotations

from enum import Enum
from typing import Iterable

import numpy as np
import pandas as pd


class Regime(str, Enum):
    """Macro regimes from the growth/inflation four-quadrant map."""

    HIGH_GROWTH_LOW_INFLATION = "high_growth_low_inflation"
    HIGH_GROWTH_HIGH_INFLATION = "high_growth_high_inflation"
    LOW_GROWTH_HIGH_INFLATION = "low_growth_high_inflation"
    LOW_GROWTH_LOW_INFLATION = "low_growth_low_inflation"


REGIME_ORDER: list[str] = [regime.value for regime in Regime]


ThresholdSpec = float | int | str | tuple[str, float]


def resolve_threshold(values: pd.Series, threshold: ThresholdSpec) -> float:
    """Resolve a numeric, median, mean, or quantile threshold specification."""

    clean = values.dropna()
    if clean.empty:
        raise ValueError("Cannot resolve threshold from an empty series.")

    if isinstance(threshold, (int, float)):
        return float(threshold)
    if threshold == "median":
        return float(clean.median())
    if threshold == "mean":
        return float(clean.mean())
    if isinstance(threshold, tuple) and len(threshold) == 2 and threshold[0] == "quantile":
        q = float(threshold[1])
        if not 0 <= q <= 1:
            raise ValueError("Quantile threshold must be between 0 and 1.")
        return float(clean.quantile(q))

    raise ValueError(
        "Threshold must be a number, 'median', 'mean', or ('quantile', q)."
    )


def classify_quadrants(
    macro: pd.DataFrame,
    growth_col: str = "growth",
    inflation_col: str = "inflation",
    growth_threshold: ThresholdSpec = "median",
    inflation_threshold: ThresholdSpec = "median",
) -> pd.Series:
    """Classify each observation into a growth/inflation quadrant."""

    missing = {growth_col, inflation_col}.difference(macro.columns)
    if missing:
        raise KeyError(f"Macro data is missing required columns: {sorted(missing)}")

    growth_cutoff = resolve_threshold(macro[growth_col], growth_threshold)
    inflation_cutoff = resolve_threshold(macro[inflation_col], inflation_threshold)

    growth_high = macro[growth_col] >= growth_cutoff
    inflation_high = macro[inflation_col] >= inflation_cutoff

    labels = np.select(
        [
            growth_high & ~inflation_high,
            growth_high & inflation_high,
            ~growth_high & inflation_high,
            ~growth_high & ~inflation_high,
        ],
        [
            Regime.HIGH_GROWTH_LOW_INFLATION.value,
            Regime.HIGH_GROWTH_HIGH_INFLATION.value,
            Regime.LOW_GROWTH_HIGH_INFLATION.value,
            Regime.LOW_GROWTH_LOW_INFLATION.value,
        ],
        default=None,
    )

    regimes = pd.Series(labels, index=macro.index, name="regime", dtype="object")
    regimes[macro[[growth_col, inflation_col]].isna().any(axis=1)] = pd.NA
    return regimes.astype("string")


def estimate_transition_matrix(
    regimes: pd.Series,
    states: Iterable[str] = REGIME_ORDER,
    smoothing: float = 1.0,
) -> pd.DataFrame:
    """Estimate a Markov transition matrix from a historical regime series."""

    if smoothing < 0:
        raise ValueError("smoothing must be non-negative.")

    state_list = list(states)
    clean = regimes.dropna().astype(str)
    counts = pd.DataFrame(
        smoothing,
        index=state_list,
        columns=state_list,
        dtype=float,
    )

    for current_state, next_state in zip(clean.iloc[:-1], clean.iloc[1:]):
        if current_state in counts.index and next_state in counts.columns:
            counts.loc[current_state, next_state] += 1.0

    row_sums = counts.sum(axis=1)
    if (row_sums == 0).any():
        raise ValueError("At least one transition row has no observations or smoothing.")

    return counts.div(row_sums, axis=0)
