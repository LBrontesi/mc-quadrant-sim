from __future__ import annotations

from enum import Enum
from numbers import Real
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


def _consecutive_periods(index: pd.Index) -> np.ndarray:
    """Return a mask identifying adjacent observations without date gaps."""

    if len(index) < 2 or not isinstance(index, pd.DatetimeIndex):
        return np.ones(max(len(index) - 1, 0), dtype=bool)
    differences = index[1:] - index[:-1]
    typical_gap = differences.to_numpy(dtype="timedelta64[ns]").astype("int64")
    typical = float(np.median(typical_gap))
    if typical <= 0:
        return typical_gap > 0
    return typical_gap <= typical * 1.5


def resolve_threshold(values: pd.Series, threshold: ThresholdSpec) -> float:
    """Resolve a numeric, median, mean, or quantile threshold specification."""

    clean = values.dropna()
    if clean.empty:
        raise ValueError("Cannot resolve threshold from an empty series.")

    if isinstance(threshold, Real):
        resolved = float(threshold)
        if not np.isfinite(resolved):
            raise ValueError("Numeric thresholds must be finite.")
        return resolved
    if threshold == "median":
        return float(clean.median())
    if threshold == "mean":
        return float(clean.mean())
    if isinstance(threshold, tuple) and len(threshold) == 2 and threshold[0] == "quantile":
        q = float(threshold[1])
        if not 0 <= q <= 1:
            raise ValueError("Quantile threshold must be between 0 and 1.")
        return float(clean.quantile(q))

    raise ValueError("Threshold must be a number, 'median', 'mean', or ('quantile', q).")


def _causal_cutoffs(values: pd.Series, threshold: ThresholdSpec, min_periods: int) -> pd.Series:
    """Compute threshold cutoffs that never use future or current observations.

    Each cutoff at time ``t`` is estimated from observations strictly before
    ``t``, so classification is reproducible out of sample. Rows with fewer
    than ``min_periods`` prior observations are left unclassified (NaN).
    """

    if isinstance(threshold, Real):
        resolved = float(threshold)
        if not np.isfinite(resolved):
            raise ValueError("Numeric thresholds must be finite.")
        return pd.Series(resolved, index=values.index)
    prior = values.shift(1)
    if threshold == "median":
        cutoffs = prior.expanding(min_periods=min_periods).median()
    elif threshold == "mean":
        cutoffs = prior.expanding(min_periods=min_periods).mean()
    elif isinstance(threshold, tuple) and len(threshold) == 2 and threshold[0] == "quantile":
        q = float(threshold[1])
        if not 0 <= q <= 1:
            raise ValueError("Quantile threshold must be between 0 and 1.")
        cutoffs = prior.expanding(min_periods=min_periods).quantile(q)
    else:
        raise ValueError("Threshold must be a number, 'median', 'mean', or ('quantile', q).")
    return cutoffs.reindex(values.index)


def _threshold_series(
    values: pd.Series,
    threshold: ThresholdSpec,
    threshold_window: int | None,
) -> pd.Series:
    """Return an index-aligned cutoff series for hard or soft classification."""

    if threshold_window is not None:
        threshold_window = int(threshold_window)
        if threshold_window <= 0:
            raise ValueError("threshold_window must be positive or None.")
        return _causal_cutoffs(values, threshold, threshold_window)
    return pd.Series(resolve_threshold(values, threshold), index=values.index, dtype=float)


def _probability_scale(
    values: pd.Series,
    cutoff: pd.Series,
    threshold_window: int | None,
    temperature: float,
) -> pd.Series:
    """Estimate a causal scale for smooth high/low probabilities.

    ``temperature`` is expressed in historical standard deviations. A value
    near zero approaches hard classification, while larger values make the
    boundary deliberately less certain.
    """

    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite.")
    prior = values.shift(1)
    minimum = max(int(threshold_window or 12), 2)
    scale = prior.expanding(min_periods=minimum).std(ddof=1)
    fallback = float(values.std(ddof=1))
    if not np.isfinite(fallback) or fallback <= 0:
        fallback = max(float(np.nanmedian(np.abs(values - cutoff))), 1.0)
    scale = scale.fillna(fallback).clip(lower=max(fallback * 1e-6, 1e-9))
    return scale * float(temperature)


def quadrant_probabilities(
    macro: pd.DataFrame,
    growth_col: str = "growth",
    inflation_col: str = "inflation",
    growth_threshold: ThresholdSpec = "median",
    inflation_threshold: ThresholdSpec = "median",
    threshold_window: int | None = None,
    temperature: float = 0.35,
) -> pd.DataFrame:
    """Return causal probabilities for the four growth/inflation quadrants.

    The high/low decisions are logistic rather than discontinuous. Their joint
    probabilities preserve the four familiar quadrant labels and sum to one
    on every classified row. Missing inputs and unavailable causal cutoffs are
    left as missing rather than silently imputed.
    """

    missing = {growth_col, inflation_col}.difference(macro.columns)
    if missing:
        raise KeyError(f"Macro data is missing required columns: {sorted(missing)}")
    growth = pd.to_numeric(macro[growth_col], errors="coerce")
    inflation = pd.to_numeric(macro[inflation_col], errors="coerce")
    growth_cutoff = _threshold_series(growth, growth_threshold, threshold_window)
    inflation_cutoff = _threshold_series(inflation, inflation_threshold, threshold_window)
    growth_scale = _probability_scale(growth, growth_cutoff, threshold_window, temperature)
    inflation_scale = _probability_scale(
        inflation,
        inflation_cutoff,
        threshold_window,
        temperature,
    )

    def logistic(values: pd.Series) -> pd.Series:
        clipped = values.clip(lower=-35.0, upper=35.0)
        return 1.0 / (1.0 + np.exp(-clipped))

    growth_high = logistic((growth - growth_cutoff) / growth_scale)
    inflation_high = logistic((inflation - inflation_cutoff) / inflation_scale)
    probabilities = pd.DataFrame(
        {
            Regime.HIGH_GROWTH_LOW_INFLATION.value: growth_high * (1.0 - inflation_high),
            Regime.HIGH_GROWTH_HIGH_INFLATION.value: growth_high * inflation_high,
            Regime.LOW_GROWTH_HIGH_INFLATION.value: (1.0 - growth_high) * inflation_high,
            Regime.LOW_GROWTH_LOW_INFLATION.value: (1.0 - growth_high) * (1.0 - inflation_high),
        },
        index=macro.index,
    )
    invalid = growth.isna() | inflation.isna() | growth_cutoff.isna() | inflation_cutoff.isna()
    probabilities.loc[invalid] = np.nan
    row_sums = probabilities.sum(axis=1, min_count=1)
    return probabilities.div(row_sums, axis=0)


def classify_quadrants(
    macro: pd.DataFrame,
    growth_col: str = "growth",
    inflation_col: str = "inflation",
    growth_threshold: ThresholdSpec = "median",
    inflation_threshold: ThresholdSpec = "median",
    threshold_window: int | None = None,
) -> pd.Series:
    """Classify each observation into a growth/inflation quadrant.

    With ``threshold_window=None`` thresholds are estimated on the full sample,
    which leaks future information into the classification. Set a positive
    ``threshold_window`` to estimate each cutoff from the prior observations
    only (causal, expanding window), leaving the earliest rows unclassified
    until enough history exists.
    """

    missing = {growth_col, inflation_col}.difference(macro.columns)
    if missing:
        raise KeyError(f"Macro data is missing required columns: {sorted(missing)}")

    growth_cutoff = _threshold_series(macro[growth_col], growth_threshold, threshold_window)
    inflation_cutoff = _threshold_series(macro[inflation_col], inflation_threshold, threshold_window)

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
    if threshold_window is not None:
        no_cutoff = pd.Series(growth_cutoff).isna() | pd.Series(inflation_cutoff).isna()
        regimes.loc[no_cutoff.index[no_cutoff.to_numpy()]] = pd.NA
    return regimes.astype("string")


def sojourn_durations(regime_series: pd.Series, states: Iterable[str], min_length: int = 1) -> dict[str, np.ndarray]:
    """Extract the empirical run-length distribution of each state.

    A sojourn is a maximal consecutive run of the same state. The resulting
    distributions are fatter-tailed than the geometric lengths implied by a
    first-order Markov chain, which is why regimes persist longer in reality
    than a naive chain suggests.  Set *min_length* to exclude threshold-noise
    single-month flips from the empirical distribution.
    """

    state_list = list(dict.fromkeys(states))
    clean = regime_series.dropna().sort_index().astype(str)
    durations: dict[str, list[int]] = {state: [] for state in state_list}
    if clean.empty:
        return {state: np.array([], dtype=int) for state in state_list}
    current_state = str(clean.iloc[0])
    length = 1
    consecutive = _consecutive_periods(clean.index)
    for position, observation in enumerate(clean.iloc[1:]):
        if consecutive[position] and str(observation) == current_state:
            length += 1
        else:
            if length >= min_length:
                durations[current_state].append(length)
            current_state = str(observation)
            length = 1
    if length >= min_length:
        durations[current_state].append(length)
    return {state: np.array(lengths, dtype=int) for state, lengths in durations.items()}


def estimate_transition_matrix(
    regimes: pd.Series,
    states: Iterable[str] = REGIME_ORDER,
    smoothing: float = 1.0,
) -> pd.DataFrame:
    """Estimate a Markov transition matrix from a historical regime series."""

    if not np.isfinite(smoothing) or smoothing < 0:
        raise ValueError("smoothing must be a finite, non-negative number.")

    state_list = list(dict.fromkeys(states))
    if not state_list:
        raise ValueError("At least one state is required.")
    clean = regimes.dropna().sort_index().astype(str)
    counts = pd.DataFrame(
        smoothing,
        index=state_list,
        columns=state_list,
        dtype=float,
    )

    consecutive = _consecutive_periods(clean.index)
    for position, (current_state, next_state) in enumerate(zip(clean.iloc[:-1], clean.iloc[1:])):
        if consecutive[position] and current_state in counts.index and next_state in counts.columns:
            counts.loc[current_state, next_state] += 1.0

    row_sums = counts.sum(axis=1)
    if (row_sums == 0).any():
        raise ValueError("At least one transition row has no observations or smoothing.")

    return counts.div(row_sums, axis=0)


def estimate_probabilistic_transition_matrix(
    probabilities: pd.DataFrame,
    states: Iterable[str] = REGIME_ORDER,
    smoothing: float = 1.0,
) -> pd.DataFrame:
    """Estimate expected transition counts from soft regime memberships."""

    if not np.isfinite(smoothing) or smoothing < 0:
        raise ValueError("smoothing must be a finite, non-negative number.")
    state_list = list(dict.fromkeys(states))
    missing = set(state_list).difference(probabilities.columns)
    if missing:
        raise KeyError(f"Regime probabilities are missing states: {sorted(missing)}")
    clean = probabilities.loc[:, state_list].dropna().sort_index()
    counts = np.full((len(state_list), len(state_list)), float(smoothing), dtype=float)
    consecutive = _consecutive_periods(clean.index)
    values = clean.to_numpy(dtype=float)
    for position in range(max(len(values) - 1, 0)):
        if consecutive[position]:
            counts += np.outer(values[position], values[position + 1])
    row_sums = counts.sum(axis=1, keepdims=True)
    if (row_sums == 0).any():
        raise ValueError("At least one transition row has no observations or smoothing.")
    return pd.DataFrame(counts / row_sums, index=state_list, columns=state_list)
