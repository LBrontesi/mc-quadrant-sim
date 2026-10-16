"""Filtered macro news and conditional volatility, without a yield-level attractor.

The innovations are statistical forecast errors, not identified causal shocks.
Every filter is one-sided and resets at missing months or bootstrap block joins.
"""

from __future__ import annotations

import numpy as np


def filter_news(errors, valid, decay=0.8):
    """Return information available before each error and after the final error."""
    errors = np.asarray(errors, dtype=float)
    memory = np.zeros(errors.shape[1])
    features = np.zeros_like(errors)
    for t, value in enumerate(errors):
        features[t] = memory
        memory = decay * memory + (1 - decay) * value if valid[t] else np.zeros_like(memory)
    return features, memory


def variance_filter(energy, valid, alpha, beta):
    """Unit-target GARCH variance; current shocks update the NEXT variance."""
    variance = 1.0
    predicted = np.empty(len(energy))
    for t, value in enumerate(energy):
        predicted[t] = variance
        variance = 1 - alpha - beta + beta * variance + alpha * value if valid[t] else 1.0
    return predicted, float(variance)


def fit_macro_stress(errors, valid, mode, membership=None):
    """Fit a regularized news response and a small, prespecified variance grid.

    errors are the errors of the unchanged base macro/yield/CPI forecasts. News
    features always precede their targets. Volatility selection uses sequential
    Gaussian likelihood plus a two-parameter BIC penalty against constant risk.
    """
    errors = np.asarray(errors, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    if mode not in {"persistent", "volatility", "combined"}:
        raise ValueError("Macro stress mode must be persistent, volatility, or combined.")
    if errors.ndim != 2 or errors.shape[1] == 0 or valid.shape != (len(errors),) or valid.sum() < 24:
        raise ValueError("Macro stress requires at least 24 valid monthly innovations.")
    if not np.isfinite(errors[valid]).all():
        raise ValueError("Macro stress innovations must be finite.")
    dimensions = errors.shape[1]
    scale = np.maximum(errors[valid].std(axis=0, ddof=1), 1e-9)
    standardized = np.where(valid[:, None], errors / scale, 0.0)
    features, latest = filter_news(standardized, valid)
    response = np.zeros((dimensions, dimensions))
    decay = 0.8
    if mode in {"persistent", "combined"}:
        x, y = features[valid], standardized[valid]
        # 24 pseudo-observations with zero response; filter variance is 1/9
        # for iid unit-variance news at decay=0.8.
        ridge = 24 * (1 - decay) / (1 + decay)
        response = np.linalg.solve(x.T @ x + ridge * np.eye(dimensions), x.T @ y)
        # Bound the actual state transition, not yields or the news coefficient.
        gain = 1.0
        while np.max(np.abs(np.linalg.eigvals(decay * np.eye(dimensions) + (1 - decay) * response))) > 0.97:
            response *= 0.9
            gain *= 0.9
            if gain < 1e-8:
                raise ValueError("Could not stabilize the macro-news transition.")
    corrected = errors - (features @ response) * scale
    corrected[~valid] = 0
    covariance = np.atleast_2d(np.cov(corrected[valid], rowvar=False))
    membership = np.ones((len(errors), 1)) if membership is None else np.asarray(membership, dtype=float)
    if (
        membership.ndim != 2
        or membership.shape[1] == 0
        or len(membership) != len(errors)
        or not np.isfinite(membership).all()
        or (membership < 0).any()
    ):
        raise ValueError("Macro stress membership must contain finite nonnegative state weights.")
    row_sums = membership.sum(axis=1, keepdims=True)
    membership = np.divide(
        membership, row_sums, out=np.full_like(membership, 1 / membership.shape[1]), where=row_sums > 0
    )
    normalized = corrected / scale
    inverses, ranks = [], []
    energy = np.zeros(len(errors))
    for weights in membership.T:
        weights = weights * valid
        total = weights.sum()
        effective = total**2 / max(float(weights @ weights), 1e-12)
        denominator = total - float(weights @ weights) / max(total, 1e-12)
        if denominator > 1e-9:
            centered = corrected - np.average(corrected, axis=0, weights=weights)
            local = (centered.T * weights) @ centered / denominator
        else:
            local = covariance
        reliability = effective / (effective + 24)
        state_covariance = reliability * local + (1 - reliability) * covariance
        # Ignore unidentifiable directions, including constant CPI. Numerical
        # PSD-floor noise must not feed an explosive volatility recursion.
        correlation = state_covariance / np.outer(scale, scale)
        eigenvalues = np.linalg.eigvalsh(correlation)
        rank = int(np.sum(eigenvalues > max(float(eigenvalues[-1]) * 1e-6, 1e-12)))
        inverse = (
            np.linalg.pinv(correlation, rcond=1e-6, hermitian=True) if rank else np.zeros_like(correlation)
        )
        inverses.append((inverse / np.outer(scale, scale)).tolist())
        ranks.append(max(rank, 1))
        state_energy = np.maximum(np.einsum("ni,ij,nj->n", normalized, inverse, normalized), 0) / max(rank, 1)
        energy += membership[:, len(ranks) - 1] * state_energy
    energy_dimensions = float(np.mean(np.asarray(ranks)[None, :] * membership, axis=0).sum())
    candidates = [(0.0, 0.0)]
    if np.any(covariance) and mode in {"volatility", "combined"}:
        candidates += [(0.05, 0.8), (0.1, 0.8), (0.1, 0.88), (0.2, 0.75)]
    scores = []
    for alpha, beta in candidates:
        predicted, _ = variance_filter(energy, valid, alpha, beta)
        loss = energy_dimensions * np.sum(np.log(predicted[valid]) + energy[valid] / predicted[valid])
        penalty = 2 * np.log(valid.sum()) if alpha else 0.0
        scores.append(float(loss + penalty))
    best = int(np.argmin(scores))
    alpha, beta = candidates[best]
    variances, latest_variance = variance_filter(energy, valid, alpha, beta)
    # Retain raw-news units for restoring an observed origin across bootstrap
    # models with different fitted scales and response coefficients.
    return (
        dict(
            mode=mode,
            decay=decay,
            response=response.tolist(),
            scale=scale.tolist(),
            latest_news=(latest * scale).tolist(),
            latest_variance=latest_variance,
            alpha=alpha,
            beta=beta,
            energy_inverse=inverses[0],
            energy_dimensions=ranks[0],
            state_energy_inverses=inverses,
            state_energy_dimensions=ranks,
            observations=int(valid.sum()),
            variance_candidates=[list(pair) for pair in candidates],
            variance_scores=scores,
            selection="sequential_gaussian_likelihood_plus_BIC",
            innovation_basis="statistical_forecast_errors_not_causal_shocks",
        ),
        corrected,
        variances,
    )


def news_correction(news, specification):
    scale = np.asarray(specification["scale"])
    return (news / scale) @ np.asarray(specification["response"]) * scale


def update_stress(news, variance, correction, shocks, specification, states=None):
    """Update memory from total base-forecast errors; variance from new shocks."""
    decay = specification["decay"]
    next_news = decay * news + (1 - decay) * (correction + shocks)
    if states is None:
        inverses = np.asarray(specification["energy_inverse"])
        dimensions = specification["energy_dimensions"]
    else:
        inverses = np.asarray(specification["state_energy_inverses"])[states]
        dimensions = np.asarray(specification["state_energy_dimensions"])[states]
    contraction = "ni,ij,nj->n" if states is None else "ni,nij,nj->n"
    energy = np.maximum(np.einsum(contraction, shocks, inverses, shocks), 0) / dimensions
    alpha, beta = specification["alpha"], specification["beta"]
    next_variance = 1 - alpha - beta + beta * variance + alpha * energy
    return next_news, next_variance
