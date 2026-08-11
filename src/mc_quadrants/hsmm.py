from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from mc_quadrants.matrix import nearest_psd
from mc_quadrants.regimes import (
    REGIME_ORDER,
    estimate_duration_hazards,
    expected_duration_from_hazards,
)


@dataclass(frozen=True)
class HSMMFit:
    """Fitted explicit-duration hidden semi-Markov quadrant model."""

    filtered_probabilities: pd.DataFrame
    smoothed_probabilities: pd.DataFrame
    viterbi_path: pd.Series
    transition_matrix: pd.DataFrame
    exit_transition_matrix: pd.DataFrame
    duration_hazards: dict[str, np.ndarray]
    expected_duration_months: dict[str, float]
    log_likelihood: float
    iterations: int
    converged: bool
    max_duration: int
    emission_means: dict[str, np.ndarray]
    emission_covariances: dict[str, np.ndarray]
    latest_state_age_probabilities: dict[str, list[float]]


def _contiguous_slices(index: pd.Index) -> list[slice]:
    if len(index) == 0:
        return []
    if len(index) == 1 or not isinstance(index, pd.DatetimeIndex):
        return [slice(0, len(index))]
    gaps = (index[1:] - index[:-1]).to_numpy(dtype="timedelta64[ns]").astype("int64")
    positive = gaps[gaps > 0]
    if not len(positive):
        return [slice(0, len(index))]
    typical = float(np.median(positive))
    breaks = np.flatnonzero((gaps <= 0) | (gaps > typical * 1.5)) + 1
    bounds = np.concatenate(([0], breaks, [len(index)]))
    return [slice(int(start), int(stop)) for start, stop in zip(bounds[:-1], bounds[1:])]


def _quadrant_signs(state: str) -> np.ndarray:
    growth = 1.0 if state.startswith("high_growth") else -1.0
    inflation = 1.0 if state.endswith("high_inflation") else -1.0
    return np.array([growth, inflation], dtype=float)


def _estimate_emissions(
    values: np.ndarray,
    labels: pd.Series,
    states: list[str],
    prior_strength: float,
) -> tuple[np.ndarray, np.ndarray]:
    dimensions = values.shape[1]
    global_mean = np.mean(values, axis=0)
    global_covariance = np.atleast_2d(np.cov(values, rowvar=False, ddof=1))
    if global_covariance.shape != (dimensions, dimensions):
        global_covariance = np.eye(dimensions, dtype=float)
    global_covariance = nearest_psd(global_covariance)
    global_scale = np.sqrt(np.maximum(np.diag(global_covariance), 1e-8))
    means = np.empty((len(states), dimensions), dtype=float)
    covariances = np.empty((len(states), dimensions, dimensions), dtype=float)
    label_values = labels.astype("string").fillna("").to_numpy(dtype=str)
    for state_index, state in enumerate(states):
        mask = label_values == state
        observations = values[mask]
        count = len(observations)
        if count:
            local_mean = np.mean(observations, axis=0)
        elif dimensions >= 2 and state in REGIME_ORDER:
            local_mean = global_mean.copy()
            local_mean[:2] += 0.5 * global_scale[:2] * _quadrant_signs(state)
        else:
            local_mean = global_mean.copy()
        reliability = count / max(count + prior_strength, 1e-12)
        means[state_index] = reliability * local_mean + (1.0 - reliability) * global_mean
        if count >= 2:
            local_covariance = np.atleast_2d(np.cov(observations, rowvar=False, ddof=1))
            if local_covariance.shape != global_covariance.shape:
                local_covariance = global_covariance
        else:
            local_covariance = global_covariance
        covariance = reliability * local_covariance + (1.0 - reliability) * global_covariance
        ridge = max(float(np.trace(global_covariance)) / max(dimensions, 1), 1e-8) * 1e-6
        covariances[state_index] = nearest_psd(covariance + np.eye(dimensions) * ridge)
    return means, covariances


def _emission_log_densities(
    values: np.ndarray,
    means: np.ndarray,
    covariances: np.ndarray,
) -> np.ndarray:
    periods, dimensions = values.shape
    densities = np.empty((periods, len(means)), dtype=float)
    constant = dimensions * np.log(2.0 * np.pi)
    for state in range(len(means)):
        covariance = covariances[state]
        sign, log_determinant = np.linalg.slogdet(covariance)
        if sign <= 0 or not np.isfinite(log_determinant):
            covariance = nearest_psd(covariance + np.eye(dimensions) * 1e-8)
            sign, log_determinant = np.linalg.slogdet(covariance)
        inverse = np.linalg.pinv(covariance)
        centered = values - means[state]
        mahalanobis = np.einsum("ti,ij,tj->t", centered, inverse, centered)
        densities[:, state] = -0.5 * (constant + log_determinant + mahalanobis)
    return densities


def _initial_exit_matrix(labels: pd.Series, states: list[str], smoothing: float) -> np.ndarray:
    counts = np.full((len(states), len(states)), float(smoothing), dtype=float)
    np.fill_diagonal(counts, 0.0)
    clean = labels.dropna().astype(str)
    state_lookup = {state: index for index, state in enumerate(states)}
    if len(clean) > 1:
        consecutive = np.ones(len(clean) - 1, dtype=bool)
        if isinstance(clean.index, pd.DatetimeIndex):
            gaps = (clean.index[1:] - clean.index[:-1]).to_numpy(dtype="timedelta64[ns]").astype("int64")
            positive = gaps[gaps > 0]
            if len(positive):
                consecutive = (gaps > 0) & (gaps <= float(np.median(positive)) * 1.5)
        for position, (left, right) in enumerate(zip(clean.iloc[:-1], clean.iloc[1:])):
            if consecutive[position] and left != right and left in state_lookup and right in state_lookup:
                counts[state_lookup[left], state_lookup[right]] += 1.0
    if len(states) == 1:
        return np.ones((1, 1), dtype=float)
    row_sums = counts.sum(axis=1, keepdims=True)
    empty = row_sums[:, 0] <= 0
    counts[empty] = 1.0
    counts[empty, np.arange(len(states))[empty]] = 0.0
    return counts / counts.sum(axis=1, keepdims=True)


def _forward_backward(
    log_densities: np.ndarray,
    hazards: np.ndarray,
    exit_matrix: np.ndarray,
    initial: np.ndarray,
) -> tuple[
    float,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    periods, state_count = log_densities.shape
    duration_count = hazards.shape[1]
    offsets = np.max(log_densities, axis=1)
    emissions = np.exp(log_densities - offsets[:, None])
    alpha = np.zeros((periods, state_count, duration_count), dtype=float)
    scales = np.empty(periods, dtype=float)
    alpha[0, :, 0] = initial * emissions[0]
    scales[0] = max(float(alpha[0].sum()), 1e-300)
    alpha[0] /= scales[0]
    for period in range(1, periods):
        previous = alpha[period - 1]
        predicted = np.zeros((state_count, duration_count), dtype=float)
        predicted[:, 1:] += previous[:, :-1] * (1.0 - hazards[:, :-1])
        predicted[:, -1] += previous[:, -1] * (1.0 - hazards[:, -1])
        exit_mass = np.sum(previous * hazards, axis=1)
        predicted[:, 0] = exit_mass @ exit_matrix
        alpha[period] = predicted * emissions[period, :, None]
        scales[period] = max(float(alpha[period].sum()), 1e-300)
        alpha[period] /= scales[period]

    beta = np.ones_like(alpha)
    for period in range(periods - 2, -1, -1):
        following = beta[period + 1]
        continuation = np.empty_like(following)
        continuation[:, :-1] = (
            (1.0 - hazards[:, :-1])
            * emissions[period + 1, :, None]
            * following[:, 1:]
        )
        continuation[:, -1] = (
            (1.0 - hazards[:, -1]) * emissions[period + 1] * following[:, -1]
        )
        destination = emissions[period + 1] * following[:, 0]
        exit_value = exit_matrix @ destination
        beta[period] = continuation + hazards * exit_value[:, None]
        beta[period] /= scales[period + 1]

    gamma = alpha * beta
    gamma /= np.maximum(gamma.sum(axis=(1, 2), keepdims=True), 1e-300)
    exit_counts = np.zeros_like(hazards)
    risk_counts = gamma[:-1].sum(axis=0) if periods > 1 else np.zeros_like(hazards)
    destination_counts = np.zeros_like(exit_matrix)
    for period in range(1, periods):
        destination = emissions[period] * beta[period, :, 0]
        source_exit = alpha[period - 1] * hazards
        pair_counts = (
            source_exit.sum(axis=1)[:, None]
            * exit_matrix
            * destination[None, :]
            / scales[period]
        )
        destination_counts += pair_counts
        future_exit_value = exit_matrix @ destination
        exit_counts += source_exit * future_exit_value[:, None] / scales[period]
    filtered = alpha.sum(axis=2)
    smoothed = gamma.sum(axis=2)
    log_likelihood = float(np.sum(np.log(scales) + offsets))
    return log_likelihood, filtered, smoothed, gamma, exit_counts, risk_counts, destination_counts


def _viterbi(
    log_densities: np.ndarray,
    hazards: np.ndarray,
    exit_matrix: np.ndarray,
    initial: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    periods, state_count = log_densities.shape
    duration_count = hazards.shape[1]
    size = state_count * duration_count
    negative_infinity = -np.inf
    scores = np.full((state_count, duration_count), negative_infinity, dtype=float)
    scores[:, 0] = np.log(np.maximum(initial, 1e-300)) + log_densities[0]
    pointers = np.full((periods, state_count, duration_count), -1, dtype=np.int32)
    for period in range(1, periods):
        following = np.full_like(scores, negative_infinity)
        continuation = scores[:, :-1] + np.log(np.maximum(1.0 - hazards[:, :-1], 1e-300))
        following[:, 1:] = continuation
        pointers[period, :, 1:] = (
            np.arange(state_count)[:, None] * duration_count
            + np.arange(duration_count - 1)[None, :]
        )
        capped = scores[:, -1] + np.log(np.maximum(1.0 - hazards[:, -1], 1e-300))
        replace_tail = capped > following[:, -1]
        following[replace_tail, -1] = capped[replace_tail]
        pointers[period, replace_tail, -1] = (
            np.arange(state_count)[replace_tail] * duration_count + duration_count - 1
        )
        exit_scores = scores + np.log(np.maximum(hazards, 1e-300))
        for destination in range(state_count):
            candidates = exit_scores + np.log(np.maximum(exit_matrix[:, destination], 1e-300))[:, None]
            source = int(np.argmax(candidates))
            source_state, source_age = divmod(source, duration_count)
            value = candidates[source_state, source_age]
            if value > following[destination, 0]:
                following[destination, 0] = value
                pointers[period, destination, 0] = source
        scores = following + log_densities[period, :, None]
    expanded_path = np.empty(periods, dtype=int)
    expanded_path[-1] = int(np.argmax(scores.reshape(size)))
    for period in range(periods - 1, 0, -1):
        state, age = divmod(int(expanded_path[period]), duration_count)
        expanded_path[period - 1] = pointers[period, state, age]
    return expanded_path // duration_count, expanded_path % duration_count


def _summary_transition_matrix(
    states: list[str],
    hazards: dict[str, np.ndarray],
    exit_matrix: np.ndarray,
    min_duration: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    expected = {
        state: expected_duration_from_hazards(hazards[state], min_duration=min_duration)
        for state in states
    }
    matrix = np.zeros_like(exit_matrix)
    for state_index, state in enumerate(states):
        if len(states) == 1:
            matrix[state_index, state_index] = 1.0
            continue
        exit_rate = float(np.clip(1.0 / max(expected[state], 1.0), 0.0, 1.0))
        matrix[state_index] = exit_rate * exit_matrix[state_index]
        matrix[state_index, state_index] = 1.0 - exit_rate
    return pd.DataFrame(matrix, index=states, columns=states), expected


def fit_quadrant_hsmm(
    macro: pd.DataFrame,
    initial_regimes: pd.Series,
    states: Iterable[str] = REGIME_ORDER,
    columns: tuple[str, str] = ("growth", "inflation"),
    min_duration: int = 5,
    duration_prior_strength: float = 8.0,
    transition_smoothing: float = 1.0,
    max_duration: int | None = None,
    max_iterations: int = 30,
    tolerance: float = 1e-5,
) -> HSMMFit:
    """Fit a Gaussian explicit-duration HSMM to growth/inflation observations.

    The persistent quadrant labels initialize semantically identified emission
    distributions only. State probabilities, exit hazards, and destination
    transitions are then estimated jointly with forward-backward expected
    counts on the expanded ``(state, age)`` state space.
    """

    state_list = list(dict.fromkeys(states))
    if not state_list:
        raise ValueError("At least one HSMM state is required.")
    if min_duration < 1:
        raise ValueError("min_duration must be positive.")
    if not np.isfinite(duration_prior_strength) or duration_prior_strength <= 0:
        raise ValueError("duration_prior_strength must be positive and finite.")
    if not np.isfinite(transition_smoothing) or transition_smoothing <= 0:
        raise ValueError("transition_smoothing must be positive and finite.")
    missing = set(columns).difference(macro.columns)
    if missing:
        raise KeyError(f"Macro data is missing HSMM emission columns: {sorted(missing)}")
    numeric = macro.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce").sort_index()
    valid = numeric.notna().all(axis=1)
    observations = numeric.loc[valid]
    if len(observations) < 2:
        raise ValueError("HSMM calibration requires at least two complete macro observations.")
    labels = initial_regimes.reindex(observations.index).astype("string")
    duration_count = int(
        max_duration
        if max_duration is not None
        else min(120, max(24, int(np.ceil(len(observations) / 2))))
    )
    if duration_count < min_duration:
        raise ValueError("max_duration must be at least min_duration.")
    values = observations.to_numpy(dtype=float)
    means, covariances = _estimate_emissions(
        values,
        labels,
        state_list,
        prior_strength=duration_prior_strength,
    )
    log_densities = _emission_log_densities(values, means, covariances)
    initial_hazards = estimate_duration_hazards(
        initial_regimes,
        state_list,
        prior_strength=duration_prior_strength,
        max_duration=duration_count,
    )
    hazards = np.vstack([initial_hazards[state] for state in state_list])
    hazards[:, : max(min_duration - 1, 0)] = 0.0
    exit_matrix = _initial_exit_matrix(labels, state_list, transition_smoothing)
    initial = np.full(len(state_list), 1.0 / len(state_list), dtype=float)
    slices = _contiguous_slices(observations.index)
    converged = False
    previous_log_likelihood = -np.inf
    iterations = 0
    for iteration in range(1, int(max_iterations) + 1):
        total_log_likelihood = 0.0
        exit_counts = np.zeros_like(hazards)
        risk_counts = np.zeros_like(hazards)
        destination_counts = np.zeros_like(exit_matrix)
        initial_counts = np.zeros(len(state_list), dtype=float)
        for sequence in slices:
            result = _forward_backward(
                log_densities[sequence],
                hazards,
                exit_matrix,
                initial,
            )
            log_likelihood, _, _, gamma, sequence_exits, sequence_risk, sequence_destinations = result
            total_log_likelihood += log_likelihood
            exit_counts += sequence_exits
            risk_counts += sequence_risk
            destination_counts += sequence_destinations
            initial_counts += gamma[0].sum(axis=1)

        if len(state_list) > 1:
            destination_counts += transition_smoothing
            np.fill_diagonal(destination_counts, 0.0)
            exit_matrix = destination_counts / np.maximum(
                destination_counts.sum(axis=1, keepdims=True),
                1e-300,
            )
        initial = (initial_counts + 1.0) / (initial_counts.sum() + len(state_list))
        allowed = np.arange(duration_count) >= min_duration - 1
        pooled_exits = exit_counts.sum(axis=0)
        pooled_risk = risk_counts.sum(axis=0)
        baseline = float(
            np.clip(
                pooled_exits[allowed].sum() / max(pooled_risk[allowed].sum(), 1e-12),
                0.002,
                0.95,
            )
        )
        pooled_hazard = (pooled_exits + duration_prior_strength * baseline) / (
            pooled_risk + duration_prior_strength
        )
        updated_hazards = (exit_counts + duration_prior_strength * pooled_hazard) / (
            risk_counts + duration_prior_strength
        )
        updated_hazards = np.clip(updated_hazards, 0.002, 0.95)
        updated_hazards[:, ~allowed] = 0.0
        hazards = updated_hazards
        iterations = iteration
        if np.isfinite(previous_log_likelihood):
            improvement = abs(total_log_likelihood - previous_log_likelihood)
            if improvement <= tolerance * (1.0 + abs(previous_log_likelihood)):
                converged = True
                break
        previous_log_likelihood = total_log_likelihood

    filtered = pd.DataFrame(np.nan, index=macro.index, columns=state_list, dtype=float)
    smoothed = filtered.copy()
    path = pd.Series(pd.NA, index=macro.index, name="regime", dtype="string")
    state_age_latest: dict[str, list[float]] = {state: [] for state in state_list}
    final_log_likelihood = 0.0
    for sequence in slices:
        result = _forward_backward(
            log_densities[sequence],
            hazards,
            exit_matrix,
            initial,
        )
        log_likelihood, sequence_filtered, sequence_smoothed, gamma, _, _, _ = result
        final_log_likelihood += log_likelihood
        sequence_index = observations.index[sequence]
        filtered.loc[sequence_index] = sequence_filtered
        smoothed.loc[sequence_index] = sequence_smoothed
        decoded_states, _ = _viterbi(
            log_densities[sequence],
            hazards,
            exit_matrix,
            initial,
        )
        path.loc[sequence_index] = [state_list[index] for index in decoded_states]
        if sequence.stop == len(observations):
            latest = gamma[-1]
            for state_index, state in enumerate(state_list):
                state_age_latest[state] = latest[state_index].tolist()

    hazard_map = {state: hazards[index].copy() for index, state in enumerate(state_list)}
    transition_frame, expected = _summary_transition_matrix(
        state_list,
        hazard_map,
        exit_matrix,
        min_duration,
    )
    exit_frame = pd.DataFrame(exit_matrix, index=state_list, columns=state_list)
    return HSMMFit(
        filtered_probabilities=filtered,
        smoothed_probabilities=smoothed,
        viterbi_path=path,
        transition_matrix=transition_frame,
        exit_transition_matrix=exit_frame,
        duration_hazards=hazard_map,
        expected_duration_months=expected,
        log_likelihood=float(final_log_likelihood),
        iterations=iterations,
        converged=converged,
        max_duration=duration_count,
        emission_means={state: means[index].copy() for index, state in enumerate(state_list)},
        emission_covariances={
            state: covariances[index].copy() for index, state in enumerate(state_list)
        },
        latest_state_age_probabilities=state_age_latest,
    )
