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
