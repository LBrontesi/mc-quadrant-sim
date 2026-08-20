from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mc_quadrants.calibration import _ledoit_wolf_alpha
from mc_quadrants.matrix import covariance_to_correlation
from mc_quadrants.mnts import attach_mnts_parameters
from mc_quadrants.regimes import (
    estimate_duration_hazards,
    expected_duration_from_hazards,
    sojourn_durations,
)
from mc_quadrants.types import RegimeMoments, ScenarioModel


@dataclass(frozen=True)
class HmmFit:
    """Fitted Gaussian-emission hidden Markov model parameters."""

    log_likelihood: float
    iterations: int
    states: list[str]
    regimes: pd.Series


def _log_gaussian_densities(
    observations: np.ndarray,
    means: np.ndarray,
    covariances: np.ndarray,
) -> np.ndarray:
    """Log density of every observation under every state (states x samples)."""

    n, p = observations.shape
    states = len(means)
    log_densities = np.empty((states, n), dtype=float)
    for state in range(states):
        chol = np.linalg.cholesky(covariances[state])
        log_determinant = 2.0 * np.log(np.diag(chol)).sum()
        standardized = np.linalg.solve(chol, (observations - means[state]).T)
        log_densities[state] = -0.5 * (
            p * np.log(2.0 * np.pi) + log_determinant + (standardized**2).sum(axis=0)
        )
    return log_densities


def _log_likelihood(
    log_densities: np.ndarray,
    transitions: np.ndarray,
    initial: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Forward-backward pass; returns scaled posteriors and transition counts."""

    n = log_densities.shape[1]
    log_transitions = np.log(np.maximum(transitions, 1e-300))
    log_initial = np.log(np.maximum(initial, 1e-300))
    log_alpha = np.empty((n, len(initial)), dtype=float)
    log_alpha[0] = log_initial + log_densities[:, 0]
    for period in range(1, n):
        log_alpha[period] = (
            np.logaddexp.reduce(
                log_alpha[period - 1][:, None] + log_transitions,
                axis=0,
            )
            + log_densities[:, period]
        )
    log_likelihood = float(np.logaddexp.reduce(log_alpha[-1]))
    log_beta = np.zeros((n, len(initial)), dtype=float)
    for period in range(n - 2, -1, -1):
        log_beta[period] = np.logaddexp.reduce(
            log_transitions + log_beta[period + 1] + log_densities[:, period + 1],
            axis=1,
        )
    posterior = np.exp(log_alpha + log_beta - log_likelihood).T
    transition_counts = np.zeros_like(transitions, dtype=float)
    for period in range(1, n):
        log_pair = (
            log_alpha[period - 1][:, None]
            + log_transitions
            + log_beta[period][None, :]
            + log_densities[:, period][None, :]
            - log_likelihood
        )
        transition_counts += np.exp(log_pair)
    return log_likelihood, posterior, transition_counts, log_alpha


def _viterbi_path(
    log_densities: np.ndarray,
    transitions: np.ndarray,
    initial: np.ndarray,
) -> np.ndarray:
    """Most likely state sequence given fitted parameters."""

    n = len(log_densities.T)
    states = len(initial)
    log_transitions = np.log(np.maximum(transitions, 1e-300))
    scores = np.log(np.maximum(initial, 1e-300)) + log_densities[:, 0]
    backpointers = np.zeros((n, states), dtype=int)
    for period in range(1, n):
        candidate = scores[:, None] + log_transitions
        backpointers[period] = np.argmax(candidate, axis=0)
        scores = np.max(candidate, axis=0) + log_densities[:, period]
    path = np.empty(n, dtype=int)
    path[-1] = int(np.argmax(scores))
    for period in range(n - 1, 0, -1):
        path[period - 1] = backpointers[period, path[period]]
    return path


def _kmeans_initialization(
    observations: np.ndarray,
    states: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Seeded k-means++ style initialization of means and labels."""

    n = len(observations)
    means = np.empty((states, observations.shape[1]), dtype=float)
    means[0] = observations[rng.integers(n)]
    for state in range(1, states):
        distances = np.sum(
            (observations[:, None, :] - means[:state][None, :, :]) ** 2,
            axis=2,
        ).min(axis=1)
        probabilities = distances / max(float(distances.sum()), 1e-300)
        means[state] = observations[rng.choice(n, p=probabilities)]
    labels = np.empty(n, dtype=int)
    for _ in range(10):
        labels = np.sum(
            (observations[:, None, :] - means[None, :, :]) ** 2,
            axis=2,
        ).argmin(axis=1)
        for state in range(states):
            mask = labels == state
            if mask.any():
                means[state] = observations[mask].mean(axis=0)
    return means, labels


def fit_hmm_model(
    returns: pd.DataFrame,
    n_states: int = 4,
    random_seed: int | None = None,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
    shrinkage: float | None = None,
    restarts: int = 3,
    frequency: str = "M",
    min_regime_duration: int = 5,
) -> tuple[ScenarioModel, HmmFit]:
    """Fit a Gaussian-emission hidden Markov model on asset returns.

    States are learned from the return distribution itself rather than from
    hand-chosen macro thresholds. The fitted means, covariances, and
    transition matrix are wrapped in a :class:`ScenarioModel`, so every
    downstream simulation and reporting step works unchanged. State labels are
    ``state_0``...``state_{n-1}``; their economic interpretation follows from
    the fitted moments (for example, a low-mean, high-covariance state is a
    stress regime).

    Covariances use the same shrinkage and PSD projection as quadrant
    calibration, which keeps estimates stable for small states.
    """

    if returns.empty:
        raise ValueError("returns must not be empty.")
    if min_regime_duration < 1:
        raise ValueError("min_regime_duration must be positive.")
    if not isinstance(n_states, int) or n_states < 2:
        raise ValueError("n_states must be an integer of at least 2.")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive.")
    if restarts <= 0:
        raise ValueError("restarts must be positive.")

    assets = list(returns.columns)
    observations = returns.dropna().sort_index().to_numpy(dtype=float)
    if len(observations) < n_states * 5:
        raise ValueError("Not enough observations to fit the requested number of states.")
    n, p = observations.shape
    global_mean = observations.mean(axis=0)
    global_covariance = np.atleast_2d(np.cov(observations, rowvar=False))

    best_log_likelihood = -np.inf
    best_parameters: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    best_historical: np.ndarray | None = None
    best_iterations = 0
    covariance_floor = max(float(np.diag(global_covariance).mean()) / 1e6, 1e-12)

    for restart in range(restarts):
        rng = np.random.default_rng(None if random_seed is None else random_seed + restart)
        means, labels = _kmeans_initialization(observations, n_states, rng)
        transitions = np.full((n_states, n_states), 0.1, dtype=float)
        for current, following in zip(labels[:-1], labels[1:]):
            transitions[current, following] += 1.0
        transitions /= transitions.sum(axis=1, keepdims=True)
        initial = np.full(n_states, 1.0 / n_states, dtype=float)
        covariances = np.empty((n_states, observations.shape[1], observations.shape[1]))
        for state in range(n_states):
            state_observations = observations[labels == state]
            covariances[state] = (
                np.atleast_2d(np.cov(state_observations, rowvar=False))
                if len(state_observations) > 1
                else global_covariance
            )

        previous_likelihood = -np.inf
        iterations = 0
        for iterations in range(1, max_iterations + 1):
            log_densities = _log_gaussian_densities(observations, means, covariances)
            log_likelihood, posterior, transition_counts, _ = _log_likelihood(
                log_densities,
                transitions,
                initial,
            )
            if log_likelihood - previous_likelihood < tolerance:
                break
            previous_likelihood = log_likelihood

            initial = posterior[:, 0] / max(float(posterior[:, 0].sum()), 1e-300)
            transitions = transition_counts / np.maximum(
                transition_counts.sum(axis=1, keepdims=True),
                1e-300,
            )
            responsibilities = posterior.sum(axis=1)
            means = (posterior @ observations) / np.maximum(
                responsibilities[:, None],
                1e-300,
            )
            for state in range(n_states):
                difference = observations - means[state]
                weighted = (difference.T * posterior[state]) @ difference
                sample_covariance = np.atleast_2d(weighted / max(responsibilities[state], 1e-300))
                if shrinkage is None:
                    alpha = _ledoit_wolf_alpha(
                        observations,
                        sample_covariance,
                        global_covariance,
                    )
                else:
                    alpha = float(shrinkage)
                blended = (1.0 - alpha) * sample_covariance + alpha * global_covariance
                eigenvalues, eigenvectors = np.linalg.eigh((blended + blended.T) / 2.0)
                clipped = np.clip(eigenvalues, covariance_floor, None)
                covariances[state] = (eigenvectors * clipped) @ eigenvectors.T

        log_densities = _log_gaussian_densities(observations, means, covariances)
        log_likelihood, _, _, _ = _log_likelihood(log_densities, transitions, initial)
        if log_likelihood > best_log_likelihood:
            best_log_likelihood = log_likelihood
            best_parameters = (initial, transitions, covariances)
            best_historical = _viterbi_path(log_densities, transitions, initial)
            best_iterations = iterations

    if best_parameters is None:
        raise RuntimeError("HMM fitting failed to converge.")
    final_initial, final_transitions, final_covariances = best_parameters

    states = [f"state_{index}" for index in range(n_states)]
    historical = {
        state: returns.dropna().sort_index().iloc[best_historical == index].copy()
        for index, state in enumerate(states)
    }
    moments: dict[str, RegimeMoments] = {}
    for index, state in enumerate(states):
        state_observations = observations[best_historical == index]
        state_mean = state_observations.mean(axis=0) if len(state_observations) else global_mean
        mean_series = pd.Series(state_mean, index=assets)
        covariance = pd.DataFrame(final_covariances[index], index=assets, columns=assets)
        moments[state] = RegimeMoments(
            mean=mean_series,
            covariance=covariance,
            correlation=covariance_to_correlation(covariance),
            observations=int(len(state_observations)),
        )
    moments = attach_mnts_parameters(moments, historical)

    transition_frame = pd.DataFrame(
        final_transitions,
        index=states,
        columns=states,
    )
    regime_series = pd.Series(
        [states[index] for index in best_historical],
        index=returns.dropna().sort_index().index,
        dtype="string",
    )
    duration_hazards = estimate_duration_hazards(regime_series, states)
    model = ScenarioModel(
        states=states,
        transition_matrix=transition_frame,
        moments=moments,
        frequency=frequency,
        historical_returns=historical,
        metadata={
            "model_kind": "hmm",
            "n_states": n_states,
            "log_likelihood": float(best_log_likelihood),
            "iterations": best_iterations,
            "shrinkage": shrinkage,
            "sojourn_durations": sojourn_durations(regime_series, states),
            "duration_model_kind": "regularized_state_specific_hazard",
            "min_regime_duration": int(min_regime_duration),
            "duration_hazards": duration_hazards,
            "expected_duration_months": {
                state: expected_duration_from_hazards(
                    duration_hazards[state],
                    min_duration=min_regime_duration,
                )
                for state in states
            },
        },
    )
    model.validate()
    return model, HmmFit(
        log_likelihood=float(best_log_likelihood),
        iterations=best_iterations,
        states=states,
        regimes=regime_series,
    )
