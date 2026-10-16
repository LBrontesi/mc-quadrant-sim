"""Positive, finite-memory QARCH on regime-normalized monthly residuals.

This restricted Chicheportiche/Bouchaud-inspired kernel combines a power-law
diagonal with rank-one 3- and 12-month blocks. Nonnegative component weights
make K positive semidefinite; trace(K) < 1 gives a positive intercept and a
finite second moment for unit-variance martingale innovations. It is not a
replication of the paper's daily-stock fit, and has no linear asymmetry term.
"""
from __future__ import annotations

import numpy as np

LAGS = 12
DEFAULT_WEIGHTS = np.array([0.35, 0.10, 0.10])


def kernel_components() -> np.ndarray:
    diagonal = np.arange(1, LAGS + 1, dtype=float) ** -1.2
    components = [np.diag(diagonal / diagonal.sum())]
    for window in (3, LAGS):
        vector = np.zeros(LAGS)
        vector[:window] = 1 / np.sqrt(window)
        components.append(np.outer(vector, vector))
    return np.array(components)


def kernel(weights) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    if (weights.shape != (3,) or not np.isfinite(weights).all()
            or (weights < 0).any() or weights.sum() >= 0.98):
        raise ValueError("QARCH requires three nonnegative weights with sum below 0.98.")
    return np.einsum("k,kij->ij", weights, kernel_components())


def variance_ratio(history: np.ndarray, matrix: np.ndarray, observed: int = LAGS) -> np.ndarray:
    """Predict normalized variance; unknown presample lags have unit variance.

    History is newest first, with shape (..., assets, lags). Unknown entries
    must be zero. Their expected diagonal contribution avoids a cold-start dip.
    """
    return (1.0 - np.trace(matrix)
            + np.einsum("...i,ij,...j->...", history, matrix, history)
            + np.trace(matrix[observed:, observed:]))


def initial_history(fit: dict, assets: int) -> tuple[np.ndarray, int]:
    history = np.asarray(fit.get("history", np.zeros((assets, LAGS))), dtype=float)
    observed = int(fit.get("history_observations", 0))
    if (history.shape != (assets, LAGS) or not np.isfinite(history).all()
            or not 0 <= observed <= LAGS or np.any(history[:, observed:] != 0)):
        raise ValueError("Invalid QARCH initial residual history.")
    return history.copy(), observed


def filtered_ratios(normalized: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Causal in-sample variances for filtering correlation/tail innovations."""
    history = np.zeros((normalized.shape[1], LAGS))
    ratios = np.empty_like(normalized)
    for t, residual in enumerate(normalized):
        ratios[t] = variance_ratio(history, matrix, min(t, LAGS))
        history[:, 1:] = history[:, :-1].copy()
        history[:, 0] = residual
    return ratios


def fit_qarch(residuals: np.ndarray, levels: np.ndarray, minimum_observations: int = 60) -> dict:
    """Fit three pooled feedback weights by regularized Gaussian QML.

    Levels are per-observation regime variances, not conditional variances.
    Forecast features use only previous residuals and retain memory at switches.
    """
    normalized = np.asarray(residuals, dtype=float) / np.sqrt(np.maximum(levels, 1e-12))
    if normalized.ndim != 2 or not len(normalized) or not np.isfinite(normalized).all():
        raise ValueError("QARCH calibration requires finite nonempty residuals.")
    history = np.zeros((normalized.shape[1], LAGS))
    components = kernel_components()
    features = np.empty((*normalized.shape, 3))
    for t, residual in enumerate(normalized):
        for k, component in enumerate(components):
            features[t, :, k] = variance_ratio(history, component, min(t, LAGS))
        history[:, 1:] = history[:, :-1].copy()
        history[:, 0] = residual
    # Drop the warm-up before scoring, while retaining it in forecast features.
    x, target = features[LAGS:] - 1.0, normalized[LAGS:] ** 2

    def score(weights):
        h = 1.0 + x @ weights
        return float(np.mean(np.log(h) + target / h) + 0.01 * (weights @ weights))

    weights = DEFAULT_WEIGHTS.copy()
    fitted = len(normalized) >= max(minimum_observations, LAGS + 24)
    if fitted:
        candidates = []
        for start in (DEFAULT_WEIGHTS, np.array([0.1, 0.0, 0.0]), np.array([0.6, 0.1, 0.1])):
            best, step = start.copy(), 0.12
            loss = score(best)
            for _ in range(16):
                previous = loss
                for index in range(3):
                    for direction in (-1, 1):
                        trial = best.copy()
                        trial[index] += direction * step
                        if (trial < 0).any() or trial.sum() >= 0.98:
                            continue
                        trial_loss = score(trial)
                        if trial_loss < loss:
                            best, loss = trial, trial_loss
                if loss == previous:
                    step *= 0.5
            candidates.append((loss, best))
        weights = min(candidates, key=lambda item: item[0])[1]
    matrix = kernel(weights)
    return {
        "method": "pooled_regularized_monthly_qarch_qml",
        "status": "fitted" if fitted else "insufficient_history",
        "observations": len(normalized), "minimum_observations": minimum_observations,
        "lags": LAGS, "weights": weights.tolist(), "kernel": matrix.tolist(),
        "history": history.tolist(), "history_observations": min(len(normalized), LAGS),
        "next_variance_ratio": variance_ratio(history, matrix, min(len(normalized), LAGS)).tolist(),
        "objective": score(weights) if fitted else None,
        "default_objective": score(DEFAULT_WEIGHTS) if fitted else None,
    }
