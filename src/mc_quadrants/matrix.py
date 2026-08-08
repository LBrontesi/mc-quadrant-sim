from __future__ import annotations

import numpy as np
import pandas as pd


def nearest_psd(matrix: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """Return a symmetric positive semidefinite approximation."""

    symmetric = (matrix + matrix.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    clipped = np.clip(eigenvalues, epsilon, None)
    psd = (eigenvectors * clipped) @ eigenvectors.T
    return (psd + psd.T) / 2.0


def nearest_psd_higham(
    matrix: np.ndarray,
    max_iterations: int = 100,
    tolerance: float = 1e-8,
) -> np.ndarray:
    """Project a symmetric matrix to the nearest positive semidefinite matrix.

    Dykstra's alternating projection (Higham 2002) converges to the
    Frobenius-nearest PSD matrix, unlike eigen-clipping which distorts the
    spectrum without minimizing distance. The diagonal is preserved exactly,
    which requires a non-negative diagonal (the covariance/correlation case).
    """

    symmetric = (matrix + matrix.T) / 2.0
    original_diagonal = np.diag(symmetric).copy()
    if (original_diagonal < 0).any():
        raise ValueError("nearest_psd_higham requires a non-negative diagonal.")
    current = symmetric.copy()
    psd_correction = np.zeros_like(symmetric)
    diagonal_correction = np.zeros_like(symmetric)
    previous = np.full_like(symmetric, np.inf)

    for _ in range(max_iterations):
        candidate = current + psd_correction
        eigenvalues, eigenvectors = np.linalg.eigh((candidate + candidate.T) / 2.0)
        eigenvalues = np.clip(eigenvalues, 0.0, None)
        psd_projection = (eigenvectors * eigenvalues) @ eigenvectors.T
        psd_correction = candidate - psd_projection

        candidate = psd_projection + diagonal_correction
        diagonal_projection = candidate.copy()
        np.fill_diagonal(diagonal_projection, original_diagonal)
        diagonal_correction = candidate - diagonal_projection
        current = diagonal_projection

        if np.abs(current - previous).max() < tolerance:
            break
        previous = current

    return (current + current.T) / 2.0


def covariance_to_correlation(covariance: pd.DataFrame) -> pd.DataFrame:
    """Convert a covariance matrix to a correlation matrix."""

    values = covariance.to_numpy(dtype=float)
    volatility = np.sqrt(np.clip(np.diag(values), 0.0, None))
    denominator = np.outer(volatility, volatility)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.divide(values, denominator, out=np.zeros_like(values), where=denominator > 0)
    np.fill_diagonal(corr, 1.0)
    return pd.DataFrame(corr, index=covariance.index, columns=covariance.columns)


def nearest_correlation(correlation: pd.DataFrame) -> pd.DataFrame:
    """Project a correlation-like matrix back to a valid correlation matrix."""

    values = nearest_psd(correlation.to_numpy(dtype=float))
    diagonal = np.sqrt(np.clip(np.diag(values), 1e-10, None))
    normalized = values / np.outer(diagonal, diagonal)
    normalized = np.clip(normalized, -1.0, 1.0)
    np.fill_diagonal(normalized, 1.0)
    return pd.DataFrame(normalized, index=correlation.index, columns=correlation.columns)
