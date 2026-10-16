"""Monte Carlo error from a finite bank of fitted parameter models.

The resampling unit is a complete fitted model, not an individual market path.
These intervals measure numerical stability of a simulated quantile; they are
not predictive intervals for investment outcomes or evidence of model accuracy.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def _mixture_quantiles(groups: list[np.ndarray], weights: np.ndarray, probability: float) -> np.ndarray:
    """Invert weighted empirical CDFs without expanding resampled path arrays."""
    lower = np.full(len(weights), min(group[0] for group in groups), dtype=float)
    upper = np.full(len(weights), max(group[-1] for group in groups), dtype=float)
    # Bisection supports atoms (including a mass at zero drawdown), unequal
    # group sizes and very large path counts without a resample-by-path matrix.
    for _ in range(52):
        midpoint = lower * 0.5 + upper * 0.5
        cdf = np.zeros(len(weights))
        for index, group in enumerate(groups):
            cdf += weights[:, index] * (np.searchsorted(group, midpoint, side="right") / len(group))
        above = cdf >= probability - 1e-14
        upper = np.where(above, midpoint, upper)
        lower = np.where(above, lower, midpoint)
    return upper


def assess_parameter_convergence(
    samples: Mapping[str, Sequence[np.ndarray]],
    specifications: Mapping[str, tuple[float, float, bool]],
    *,
    random_seed: int = 271828,
    resamples: int = 256,
    minimum_models: int = 32,
    minimum_paths_per_model: int = 128,
) -> dict:
    """Cluster-bootstrap quantile stability with equal weight per fitted model.

    Specifications are (quantile, tolerance, relative_tolerance). The bootstrap
    retains each model's empirical conditional distribution. It does not isolate
    inner-path Monte Carlo error; independently seeded path runs remain needed.
    """
    if not samples or set(samples) != set(specifications):
        raise ValueError("Samples and specifications must have the same nonempty metric keys.")
    if resamples < 32 or minimum_models < 2 or minimum_paths_per_model < 1:
        raise ValueError("Require at least 32 resamples, two models and one path per model.")
    prepared = {}
    count = None
    sizes = None
    for name, values in samples.items():
        groups = [np.sort(np.asarray(group, dtype=float)) for group in values]
        if not groups or any(
            group.ndim != 1 or not len(group) or not np.isfinite(group).all() for group in groups
        ):
            raise ValueError("Every model must supply nonempty finite one-dimensional samples.")
        if count is None:
            count = len(groups)
            sizes = [len(group) for group in groups]
        elif len(groups) != count or [len(group) for group in groups] != sizes:
            raise ValueError("Metrics must refer to the same models and path counts.")
        probability, tolerance, relative = specifications[name]
        if not np.isfinite([probability, tolerance]).all() or not 0 < probability < 1 or tolerance <= 0:
            raise ValueError("Quantiles must be inside (0, 1) and tolerances positive and finite.")
        prepared[name] = groups
    rng = np.random.default_rng(random_seed)
    weights = np.vstack(
        [
            np.full(count, 1 / count),
            rng.multinomial(count, np.full(count, 1 / count), size=resamples) / count,
        ]
    )
    metrics = {}
    for name, groups in prepared.items():
        probability, tolerance, relative = specifications[name]
        quantiles = _mixture_quantiles(groups, weights, probability)
        estimate = float(quantiles[0])
        low, high = np.quantile(quantiles[1:], [0.025, 0.975])
        error = float(max(abs(low - estimate), abs(high - estimate)))
        limit = float(tolerance * abs(estimate) if relative else tolerance)
        metrics[name] = dict(
            estimate=estimate,
            quantile=probability,
            mc_interval_95=[float(low), float(high)],
            max_mc_deviation=error,
            tolerance=limit,
            relative_tolerance=relative,
            within_tolerance=bool(error <= limit),
        )
    if count < minimum_models:
        status = "insufficient_models"
    elif min(sizes) < minimum_paths_per_model:
        status = "insufficient_paths"
    else:
        status = "within_tolerance" if all(m["within_tolerance"] for m in metrics.values()) else "unstable"
    return dict(
        status=status,
        parameter_models=count,
        paths_per_model=sizes,
        minimum_models=minimum_models,
        minimum_paths_per_model=minimum_paths_per_model,
        resamples=resamples,
        diagnostic_seed=random_seed,
        metrics=metrics,
        method="equal_model_cluster_bootstrap",
        scope="Outer parameter-bank Monte Carlo error conditional on sampled paths; not a predictive interval or a model-validation result.",
    )
