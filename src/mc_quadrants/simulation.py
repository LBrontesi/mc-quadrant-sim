from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from mc_quadrants.decumulation import (
    DecumulationPlan,
    SpendingController,
    funded_amount,
    inflation_index,
    normalize_decumulation,
)
from mc_quadrants.matrix import nearest_correlation, nearest_psd
from mc_quadrants.mnts import resolved_mnts_parameters, sample_mnts_subordinators
from mc_quadrants.native import (
    simulate_parametric_italian_portfolios_compact_native,
    simulate_parametric_italian_portfolios_native,
    simulate_parametric_native,
)
from mc_quadrants.tax_policy import TaxSimulationContext, resolve_tax_selection
from mc_quadrants.taxes import (
    CONTRIBUTION_ALLOCATION_MODES,
    ITALY_DEFAULT_WEALTH_TAX_RATE,
    contribution_allocations,
)
from mc_quadrants.types import ScenarioModel, SimulationResult

DEFAULT_LIQUIDITY_COST_MULTIPLIERS = {
    "high_growth_low_inflation": 0.75,
    "high_growth_high_inflation": 1.25,
    "low_growth_high_inflation": 2.00,
    "low_growth_low_inflation": 1.50,
}


def stationary_distribution(transition_matrix: pd.DataFrame) -> pd.Series:
    """Compute the long-run state distribution implied by a transition matrix."""

    if transition_matrix.empty:
        raise ValueError("transition_matrix must contain at least one state.")
    if transition_matrix.index.has_duplicates or transition_matrix.columns.has_duplicates:
        raise ValueError("Transition matrix labels must be unique.")
    if set(transition_matrix.index) != set(transition_matrix.columns):
        raise ValueError("Transition matrix rows and columns must contain the same states.")

    states = list(transition_matrix.index)
    matrix = transition_matrix.loc[states, states].to_numpy(dtype=float)
    if not np.isfinite(matrix).all() or (matrix < 0).any():
        raise ValueError("Transition matrix must contain finite, non-negative probabilities.")
    if not np.allclose(matrix.sum(axis=1), 1.0):
        raise ValueError("Transition matrix rows must sum to 1.")

    # Solve pi P = pi with a normalization row instead of selecting an
    # eigenvector, which is unstable for repeated or nearly repeated eigenvalues.
    system = matrix.T - np.eye(len(states))
    system[-1] = 1.0
    target = np.zeros(len(states))
    target[-1] = 1.0
    vector, *_ = np.linalg.lstsq(system, target, rcond=None)
    vector = np.clip(vector, 0.0, None)
    if not np.isfinite(vector).all() or np.isclose(vector.sum(), 0.0):
        vector = np.ones(len(states), dtype=float)
    vector /= vector.sum()
    return pd.Series(vector, index=states)


def _rng(random_seed: int | None = None) -> np.random.Generator:
    return np.random.default_rng(random_seed)


def _sample_transition_matrix(
    rng: np.random.Generator,
    transition: np.ndarray,
    concentration: float,
) -> np.ndarray:
    """Draw a transition matrix from row-wise Dirichlet distributions."""

    if not np.isfinite(concentration) or concentration <= 0:
        raise ValueError("transition_concentration must be positive and finite.")
    return np.vstack([rng.dirichlet(np.maximum(row, 1e-12) * concentration) for row in transition])


def _sample_sojourns(
    rng: np.random.Generator,
    state_indices: np.ndarray,
    hazard_map: dict[str, np.ndarray],
    states: list[str],
    min_duration: int = 5,
) -> np.ndarray:
    """Vectorized duration draws for a batch of state indexes."""

    state_indices = np.asarray(state_indices, dtype=int)
    durations = np.empty(len(state_indices), dtype=np.int32)
    for state_index, state in enumerate(states):
        positions = np.flatnonzero(state_indices == state_index)
        if not len(positions):
            continue
        hazards = np.asarray(hazard_map.get(state, []), dtype=float)
        if not len(hazards):
            raise ValueError(f"No duration hazards are available for state {state!r}.")
        eligible = hazards.copy()
        eligible[: max(min_duration - 1, 0)] = 0.0
        cumulative_exit = 1.0 - np.cumprod(1.0 - np.clip(eligible, 0.0, 1.0))
        sampled = np.searchsorted(cumulative_exit, rng.random(len(positions)), side="right") + 1
        durations[positions] = np.where(
            sampled > len(hazards),
            max(len(hazards), min_duration),
            sampled,
        )
    return durations


def _latest_state_age_posterior(model: ScenarioModel) -> np.ndarray | None:
    """Return the normalized latest HSMM posterior over ``(state, age)``."""

    raw = model.metadata.get("hsmm_latest_state_age_probabilities")
    if not isinstance(raw, Mapping):
        return None
    rows = [np.asarray(raw.get(state, []), dtype=float) for state in model.states]
    width = max((len(row) for row in rows), default=0)
    if width == 0:
        return None
    posterior = np.zeros((len(model.states), width), dtype=float)
    for state_index, row in enumerate(rows):
        posterior[state_index, : len(row)] = row
    if not np.isfinite(posterior).all() or (posterior < 0.0).any():
        raise ValueError("Latest state-age probabilities must be finite and non-negative.")
    total = float(posterior.sum())
    if total <= 0.0:
        return None
    return posterior / total


def _sample_initial_states_and_ages(
    rng: np.random.Generator,
    model: ScenarioModel,
    transition: np.ndarray,
    paths: int,
    start_state: str | None,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Sample initial regimes, preferring the latest filtered HSMM posterior."""

    states = model.states
    if start_state is not None:
        if start_state not in states:
            raise ValueError(f"Unknown start_state: {start_state}")
        return (
            np.full(paths, states.index(start_state), dtype=int),
            np.zeros(paths, dtype=int),
            False,
        )

    posterior = _latest_state_age_posterior(model)
    if posterior is not None:
        sampled = rng.choice(posterior.size, size=paths, p=posterior.ravel())
        return sampled // posterior.shape[1], sampled % posterior.shape[1], True

    sampled_transition = pd.DataFrame(transition, index=states, columns=states)
    start_probabilities = stationary_distribution(sampled_transition).to_numpy()
    return (
        rng.choice(len(states), size=paths, p=start_probabilities),
        np.zeros(paths, dtype=int),
        False,
    )


def _sample_remaining_sojourns(
    rng: np.random.Generator,
    state_indices: np.ndarray,
    state_ages: np.ndarray,
    hazard_map: dict[str, np.ndarray],
    states: list[str],
    min_duration: int = 5,
) -> np.ndarray:
    """Sample residual durations conditional on the current HSMM state age."""

    state_indices = np.asarray(state_indices, dtype=int)
    state_ages = np.asarray(state_ages, dtype=int)
    if state_indices.shape != state_ages.shape:
        raise ValueError("Initial state indexes and ages must have the same shape.")
    remaining = np.empty(len(state_indices), dtype=np.int32)
    for state_index, state in enumerate(states):
        state_positions = np.flatnonzero(state_indices == state_index)
        if not len(state_positions):
            continue
        hazards = np.asarray(hazard_map.get(state, []), dtype=float)
        if not len(hazards):
            raise ValueError(f"No duration hazards are available for state {state!r}.")
        for raw_age in np.unique(state_ages[state_positions]):
            positions = state_positions[state_ages[state_positions] == raw_age]
            age = int(np.clip(raw_age, 0, len(hazards) - 1))
            eligible = hazards[age:].copy()
            absolute_ages = np.arange(age, len(hazards))
            eligible[absolute_ages < max(min_duration - 1, 0)] = 0.0
            cumulative_exit = 1.0 - np.cumprod(1.0 - np.clip(eligible, 0.0, 1.0))
            sampled = np.searchsorted(
                cumulative_exit,
                rng.random(len(positions)),
                side="right",
            ) + 1
            maximum = max(len(hazards) - age, min_duration - age, 1)
            remaining[positions] = np.where(
                sampled > len(eligible),
                maximum,
                sampled,
            )
    return remaining


def _decode_regime_codes(codes: np.ndarray, states: list[str]) -> np.ndarray:
    """Map compact state codes to public string labels in one vectorized pass."""

    return np.asarray(states, dtype=object)[codes]


def _batched_cholesky(values: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """Cholesky-factor a stack of small matrices across paths.

    NumPy's general batched LAPACK dispatch has noticeable overhead for the
    simulator's many 4-12 dimensional DCC matrices. This implementation loops
    over the small asset dimension while vectorizing every operation across
    paths, which keeps the hot path inside NumPy kernels.
    """

    matrices = np.asarray(values, dtype=float)
    if matrices.ndim != 3 or matrices.shape[1] != matrices.shape[2]:
        raise ValueError("values must contain a stack of square matrices.")
    factors = np.zeros_like(matrices)
    dimension = matrices.shape[1]
    for column in range(dimension):
        previous = factors[:, column, :column]
        diagonal = matrices[:, column, column] - np.einsum("ni,ni->n", previous, previous)
        factors[:, column, column] = np.sqrt(np.maximum(diagonal, epsilon))
        if column + 1 >= dimension:
            continue
        if column:
            cross = np.einsum(
                "ni,nki->nk",
                previous,
                factors[:, column + 1 :, :column],
            )
        else:
            cross = 0.0
        factors[:, column + 1 :, column] = (
            matrices[:, column + 1 :, column] - cross
        ) / factors[:, column, column, None]
    return factors


def _transaction_cost_rate_paths(
    result: SimulationResult,
    transaction_cost_bps: float,
    multipliers: Mapping[str, float] | None,
) -> np.ndarray | None:
    if not multipliers:
        return None
    normalized = {str(state): float(value) for state, value in multipliers.items()}
    if any(not np.isfinite(value) or value < 0 for value in normalized.values()):
        raise ValueError("State transaction-cost multipliers must be finite and non-negative.")
    state_values = np.array([normalized.get(state, 1.0) for state in result.states], dtype=float)
    if result.regimes.dtype.kind in "iu":
        multiplier_paths = state_values[np.asarray(result.regimes, dtype=int)]
    else:
        multiplier_paths = np.ones(result.regimes.shape, dtype=float)
        regimes = np.asarray(result.regimes, dtype=object)
        for state, value in normalized.items():
            multiplier_paths[regimes == state] = value
    return multiplier_paths * (float(transaction_cost_bps) / 10_000.0)


def simulate_regime_paths(
    model: ScenarioModel,
    periods: int,
    paths: int,
    start_state: str | None = None,
    random_seed: int | None = None,
    transition_concentration: float | None = None,
    duration_model: str = "markov",
    min_regime_duration: int = 5,
    return_codes: bool = False,
) -> np.ndarray:
    """Simulate Markov (or semi-Markov) regime paths.

    ``duration_model="semi_markov"`` replaces the geometric sojourn times of a
    first-order Markov chain with regularized state- and age-specific exit
    hazards stored in ``model.metadata``. Transitions between states still
    follow the calibrated matrix with self-transition probabilities removed.
    """

    model.validate()
    if periods <= 0 or paths <= 0:
        raise ValueError("periods and paths must be positive.")
    if min_regime_duration <= 0:
        raise ValueError("min_regime_duration must be positive.")

    rng = _rng(random_seed)
    states = model.states
    transition = model.transition_matrix.loc[states, states].to_numpy(dtype=float)
    if transition_concentration is not None:
        transition = _sample_transition_matrix(rng, transition, transition_concentration)

    duration_model = str(duration_model).lower()
    if duration_model not in {"markov", "semi_markov"}:
        raise ValueError("duration_model must be 'markov' or 'semi_markov'.")
    duration_hazards: dict[str, np.ndarray] | None = None
    if duration_model == "semi_markov":
        duration_hazards = model.metadata.get("duration_hazards")
        if not isinstance(duration_hazards, dict):
            raise ValueError(
                "semi_markov requires the model to expose 'duration_hazards' "
                "in its metadata; recalibrate the model with duration support."
            )

    current, current_ages, _ = _sample_initial_states_and_ages(
        rng,
        model,
        transition,
        paths,
        start_state,
    )

    code_dtype = np.min_scalar_type(max(len(states) - 1, 0))
    if duration_model == "markov":
        simulated = np.empty((periods, paths), dtype=code_dtype)
        for period in range(periods):
            simulated[period] = current
            next_state = np.empty(paths, dtype=int)
            for state_index in range(len(states)):
                mask = current == state_index
                if mask.any():
                    next_state[mask] = rng.choice(len(states), size=mask.sum(), p=transition[state_index])
            current = next_state
        return simulated if return_codes else _decode_regime_codes(simulated, states)

    simulated = np.empty((periods, paths), dtype=code_dtype)
    remaining = np.empty(paths, dtype=int)
    remaining[:] = _sample_remaining_sojourns(
        rng,
        current,
        current_ages,
        duration_hazards,
        states,
        min_regime_duration,
    )
    for period in range(periods):
        simulated[period] = current
        remaining -= 1
        for state_index in range(len(states)):
            mask = (current == state_index) & (remaining <= 0)
            if not mask.any():
                continue
            other_states = [index for index in range(len(states)) if index != state_index]
            probabilities = transition[state_index, other_states]
            probabilities = probabilities / max(float(probabilities.sum()), 1e-300)
            following = rng.choice(other_states, size=mask.sum(), p=probabilities)
            next_sojourns = _sample_sojourns(
                rng,
                following,
                duration_hazards,
                states,
                min_regime_duration,
            )
            current[mask] = following
            remaining[mask] = next_sojourns
    return simulated if return_codes else _decode_regime_codes(simulated, states)


def _prepare_macro_emissions(
    states: list[str],
    emission_means: Mapping[str, object] | None,
    emission_covariances: Mapping[str, object] | None,
) -> np.ndarray | None:
    """Prepare quadratic bivariate-Gaussian scores for repeated path scoring."""

    if not isinstance(emission_means, Mapping) or not isinstance(
        emission_covariances,
        Mapping,
    ):
        return None
    dimensions = 2
    coefficients = np.empty((len(states), 6), dtype=float)
    for state_index, state in enumerate(states):
        mean = np.asarray(emission_means.get(state, []), dtype=float)[:dimensions]
        covariance = np.asarray(emission_covariances.get(state, []), dtype=float)
        if mean.shape != (dimensions,) or covariance.shape[:2] != (dimensions, dimensions):
            return None
        covariance = nearest_psd(covariance[:dimensions, :dimensions])
        sign, log_determinant = np.linalg.slogdet(covariance)
        if sign <= 0 or not np.isfinite(log_determinant):
            return None
        precision = np.linalg.pinv(covariance)
        precision_mean = precision @ mean
        coefficients[state_index] = (
            -0.5 * precision[0, 0],
            -precision[0, 1],
            -0.5 * precision[1, 1],
            precision_mean[0],
            precision_mean[1],
            -0.5
            * (
                float(mean @ precision_mean)
                + dimensions * np.log(2.0 * np.pi)
                + log_determinant
            ),
        )
    if not np.isfinite(coefficients).all():
        return None
    return coefficients


def _macro_quadrant_probabilities(
    values: np.ndarray,
    dynamics: Mapping[str, object],
    states: list[str] | None = None,
    emission_means: Mapping[str, object] | None = None,
    emission_covariances: Mapping[str, object] | None = None,
    prepared_emissions: np.ndarray | None = None,
) -> np.ndarray:
    """Return joint emission probabilities, with threshold logistics as fallback."""

    if prepared_emissions is None and states:
        prepared_emissions = _prepare_macro_emissions(
            states,
            emission_means,
            emission_covariances,
        )
    if prepared_emissions is not None:
        observations = np.asarray(values, dtype=float)[:, :2]
        growth = observations[:, 0]
        inflation = observations[:, 1]
        features = np.column_stack(
            (
                np.square(growth),
                growth * inflation,
                np.square(inflation),
                growth,
                inflation,
                np.ones(len(observations)),
            )
        )
        log_densities = features @ prepared_emissions.T
        if np.isfinite(log_densities).all():
            normalized = log_densities - log_densities.max(axis=1, keepdims=True)
            probabilities = np.exp(np.clip(normalized, -700.0, 0.0))
            return probabilities / np.maximum(probabilities.sum(axis=1, keepdims=True), 1e-300)

    thresholds = np.asarray(dynamics["thresholds"], dtype=float)
    scales = np.maximum(np.asarray(dynamics["probability_scales"], dtype=float), 1e-9)
    # Only growth and inflation define quadrant membership. Additional macro
    # state variables such as the short rate affect their joint dynamics and
    # returns, but do not create new quadrant dimensions.
    quadrant_values = values[:, : len(thresholds)]
    standardized = np.clip((quadrant_values - thresholds) / scales, -35.0, 35.0)
    high = 1.0 / (1.0 + np.exp(-standardized))
    growth_high = high[:, 0]
    inflation_high = high[:, 1]
    return np.column_stack(
        [
            growth_high * (1.0 - inflation_high),
            growth_high * inflation_high,
            (1.0 - growth_high) * inflation_high,
            (1.0 - growth_high) * (1.0 - inflation_high),
        ]
    )


def simulate_joint_regime_macro_paths(
    model: ScenarioModel,
    periods: int,
    paths: int,
    start_state: str | None = None,
    random_seed: int | None = None,
    transition_concentration: float | None = None,
    duration_model: str = "markov",
    min_regime_duration: int = 5,
    macro_transition_weight: float = 0.35,
    macro_parameter_uncertainty: bool = True,
    return_codes: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate mutually consistent macro paths and time-varying regimes."""

    dynamics = model.metadata.get("macro_dynamics")
    if not isinstance(dynamics, Mapping):
        raise ValueError("joint_macro requires calibrated macro dynamics.")
    if not 0 <= macro_transition_weight <= 1:
        raise ValueError("macro_transition_weight must be between 0 and 1.")
    duration_model = str(duration_model).lower()
    if duration_model not in {"markov", "semi_markov"}:
        raise ValueError("duration_model must be 'markov' or 'semi_markov'.")

    rng = _rng(random_seed)
    states = model.states
    if len(states) != 4:
        raise ValueError("joint_macro currently requires the four-quadrant state model.")
    transition = model.transition_matrix.loc[states, states].to_numpy(dtype=float)
    if transition_concentration is not None:
        transition = _sample_transition_matrix(rng, transition, transition_concentration)
    current, current_ages, _ = _sample_initial_states_and_ages(
        rng,
        model,
        transition,
        paths,
        start_state,
    )

    coefficient = np.asarray(dynamics["var_coefficient"], dtype=float)
    coefficient_paths: np.ndarray | None = None
    coefficient_std = dynamics.get("var_coefficient_std")
    if macro_parameter_uncertainty and coefficient_std is not None:
        posterior_scale = np.asarray(coefficient_std, dtype=float)
        if posterior_scale.shape == coefficient.shape and np.isfinite(posterior_scale).all():
            coefficient_paths = coefficient[None, :, :] + np.clip(
                rng.standard_normal((paths, *coefficient.shape)), -2.5, 2.5
            ) * posterior_scale[None, :, :] * 0.35
            # A cheap sufficient stability guard for large path sets. The
            # infinity norm bounds the spectral radius without per-path eigensolves.
            row_norm = np.max(np.sum(np.abs(coefficient_paths), axis=2), axis=1)
            unstable = row_norm > 0.995
            if unstable.any():
                coefficient_paths[unstable] *= (0.995 / row_norm[unstable])[:, None, None]
    centers = {
        state: np.asarray(dynamics["state_centers"][state], dtype=float) for state in states
    }
    covariances = {
        state: nearest_psd(
            np.asarray(dynamics["state_innovation_covariances"][state], dtype=float)
        )
        for state in states
    }
    current_macro = np.broadcast_to(
        np.asarray(dynamics["latest"], dtype=float),
        (paths, len(dynamics["columns"])),
    ).copy()
    rate_col = dynamics.get("rate_col")
    rate_index = (
        list(dynamics["columns"]).index(rate_col)
        if rate_col in dynamics["columns"]
        else None
    )
    rate_bounds = dynamics.get("rate_bounds")
    code_dtype = np.min_scalar_type(max(len(states) - 1, 0))
    regime_paths = np.empty((periods, paths), dtype=code_dtype)
    macro_paths = np.empty((periods, paths, current_macro.shape[1]), dtype=float)
    macro_shocks = np.empty_like(macro_paths)

    duration_hazards = model.metadata.get("duration_hazards")
    remaining = np.zeros(paths, dtype=int)
    if duration_model == "semi_markov":
        if not isinstance(duration_hazards, dict):
            raise ValueError("semi_markov requires calibrated duration hazards.")
        remaining[:] = _sample_remaining_sojourns(
            rng,
            current,
            current_ages,
            duration_hazards,
            states,
            min_regime_duration,
        )

    powered_transition = np.power(
        np.maximum(transition, 1e-12),
        1.0 - macro_transition_weight,
    )
    use_joint_emissions = (
        model.metadata.get("macro_membership_model", "joint_emission")
        == "joint_emission"
    )
    prepared_emissions = (
        _prepare_macro_emissions(
            states,
            model.metadata.get("hsmm_emission_means"),
            model.metadata.get("hsmm_emission_covariances"),
        )
        if use_joint_emissions
        else None
    )

    for period in range(periods):
        regime_paths[period] = current
        next_macro = np.empty_like(current_macro)
        for state_index, state in enumerate(states):
            mask = current == state_index
            if not mask.any():
                continue
            shocks = rng.multivariate_normal(
                np.zeros(current_macro.shape[1]),
                covariances[state],
                size=int(mask.sum()),
            )
            center = centers[state]
            centered_macro = current_macro[mask] - center
            if coefficient_paths is None:
                next_macro[mask] = center + centered_macro @ coefficient + shocks
            else:
                next_macro[mask] = center + np.einsum(
                    "ni,nij->nj",
                    centered_macro,
                    coefficient_paths[mask],
                ) + shocks
            macro_shocks[period, mask] = shocks
        if rate_index is not None and rate_bounds is not None:
            next_macro[:, rate_index] = np.clip(
                next_macro[:, rate_index],
                float(rate_bounds[0]),
                float(rate_bounds[1]),
            )
        current_macro = next_macro
        macro_paths[period] = current_macro

        if duration_model == "semi_markov":
            remaining -= 1
            eligible_paths = np.flatnonzero(remaining <= 0)
        else:
            eligible_paths = np.arange(paths)
        macro_probabilities = _macro_quadrant_probabilities(
            current_macro[eligible_paths],
            dynamics,
            prepared_emissions=prepared_emissions,
        )
        eligible_states = current[eligible_paths]
        following = current.copy()
        for state_index in range(len(states)):
            local_mask = eligible_states == state_index
            if not local_mask.any():
                continue
            path_indexes = eligible_paths[local_mask]
            macro_probability = macro_probabilities[local_mask]
            combined = powered_transition[state_index] * np.power(
                np.maximum(macro_probability, 1e-12), macro_transition_weight
            )
            if duration_model == "semi_markov":
                combined[:, state_index] = 0.0
            combined /= combined.sum(axis=1, keepdims=True)
            uniforms = rng.random(len(path_indexes))
            cumulative = np.cumsum(combined, axis=1)
            selected = (uniforms[:, None] > cumulative).sum(axis=1)
            following[path_indexes] = np.minimum(selected, len(states) - 1)
            if duration_model == "semi_markov":
                remaining[path_indexes] = _sample_sojourns(
                    rng,
                    following[path_indexes],
                    duration_hazards,
                    states,
                    min_regime_duration,
                )
        current = following
    public_regimes = regime_paths if return_codes else _decode_regime_codes(regime_paths, states)
    return public_regimes, macro_paths, macro_shocks


def simulate_returns(
    model: ScenarioModel,
    periods: int,
    paths: int,
    start_state: str | None = None,
    random_seed: int | None = None,
    distribution: str = "mnts",
    transition_concentration: float | None = None,
    duration_model: str = "markov",
    min_regime_duration: int = 5,
    garch: bool = True,
    garch_alpha: float = 0.10,
    garch_beta: float = 0.85,
    joint_macro: bool = False,
    macro_transition_weight: float = 0.35,
    macro_parameter_uncertainty: bool = True,
    dynamic_correlation: bool = False,
    dcc_alpha: float = 0.04,
    dcc_beta: float = 0.94,
    dcc_asymmetry: float = 0.01,
    return_regime_codes: bool = False,
    native_threads: int = 1,
    native_portfolio_config: Mapping[str, object] | None = None,
) -> SimulationResult:
    """Simulate four-state, regime-switching MNTS-GARCH asset returns.

    Every quadrant has its own multivariate normal tempered-stable parameters.
    A common tempered-stable subordinator creates joint tail dependence, while
    asset-specific skew parameters and a latent correlation matrix preserve the
    calibrated first two moments. GARCH(1,1) is enabled by default.

    ``duration_model="semi_markov"`` draws regime run lengths from regularized
    state-specific duration hazards instead of the geometric lengths implied by
    a first-order chain.

    Conditional variance is re-anchored to a quadrant's calibrated level when
    a path changes state, preventing stale volatility from leaking across
    macro regimes. The C++ backend is the production path; the Python sampler
    is an exact but intentionally slower reference implementation.
    """

    distribution = str(distribution).lower().replace("-", "_")
    if distribution != "mnts":
        raise ValueError("distribution must be 'mnts'.")
    if not np.isfinite(garch_alpha) or not 0 <= garch_alpha < 1:
        raise ValueError("garch_alpha must be between 0 and 1.")
    if not np.isfinite(garch_beta) or not 0 <= garch_beta < 1:
        raise ValueError("garch_beta must be between 0 and 1.")
    if garch_alpha + garch_beta >= 1:
        raise ValueError("garch_alpha + garch_beta must be less than 1.")
    if not 0 <= dcc_alpha < 1 or not 0 <= dcc_beta < 1 or not 0 <= dcc_asymmetry < 1:
        raise ValueError("DCC parameters must be between 0 and 1.")
    if dcc_alpha + dcc_beta + dcc_asymmetry >= 1:
        raise ValueError("dcc_alpha + dcc_beta + dcc_asymmetry must be less than 1.")

    assets = model.assets

    macro_dynamics = model.metadata.get("macro_dynamics") if joint_macro else None
    macro_betas = (
        np.asarray(macro_dynamics["return_betas"], dtype=float)
        if isinstance(macro_dynamics, Mapping)
        else None
    )

    def resolve_state_covariance(state: str) -> np.ndarray:
        if isinstance(macro_dynamics, Mapping):
            residuals = macro_dynamics.get("return_residual_covariances")
            if isinstance(residuals, Mapping) and state in residuals:
                return nearest_psd(np.asarray(residuals[state], dtype=float))
        return nearest_psd(
            model.moments[state].covariance.reindex(index=assets, columns=assets).to_numpy(dtype=float)
        )

    # All of these quantities are invariant across periods. Preparing them
    # once removes repeated Pandas alignment, eigendecompositions, and
    # covariance factorization from the simulation's inner loop.
    state_means = np.stack(
        [model.moments[state].mean.reindex(assets).to_numpy(dtype=float) for state in model.states]
    )
    state_covariances = np.stack(
        [resolve_state_covariance(state) for state in model.states]
    )
    state_volatilities = np.sqrt(
        np.clip(np.diagonal(state_covariances, axis1=1, axis2=2), 0.0, None)
    )
    correlation_denominator = state_volatilities[:, :, None] * state_volatilities[:, None, :]
    state_correlations = np.divide(
        state_covariances,
        correlation_denominator,
        out=np.zeros_like(state_covariances),
        where=correlation_denominator > 0,
    )
    diagonal_indexes = np.arange(len(assets))
    state_correlations[:, diagonal_indexes, diagonal_indexes] = 1.0
    mnts_parameters = [resolved_mnts_parameters(model.moments[state]) for state in model.states]
    state_tail_indexes = np.array(
        [parameters.tail_index for parameters in mnts_parameters], dtype=float
    )
    state_temperings = np.array(
        [parameters.tempering for parameters in mnts_parameters], dtype=float
    )
    state_skewness = np.stack(
        [parameters.skewness.reindex(assets).to_numpy(dtype=float) for parameters in mnts_parameters]
    )
    alphas = state_tail_indexes / 2.0
    subordinator_variances = (1.0 - alphas) / state_temperings
    state_gaussian_scales = np.sqrt(
        np.maximum(1.0 - state_skewness**2 * subordinator_variances[:, None], 1e-10)
    )

    gaussian_correlations: list[np.ndarray] = []
    for state_index, parameters in enumerate(mnts_parameters):
        if isinstance(macro_dynamics, Mapping):
            latent = (
                state_correlations[state_index]
                - subordinator_variances[state_index]
                * np.outer(state_skewness[state_index], state_skewness[state_index])
            ) / np.outer(state_gaussian_scales[state_index], state_gaussian_scales[state_index])
            latent_frame = nearest_correlation(
                pd.DataFrame(latent, index=assets, columns=assets)
            )
        else:
            latent_frame = parameters.gaussian_correlation.reindex(
                index=assets, columns=assets
            )
        gaussian_correlations.append(latent_frame.to_numpy(dtype=float))
    state_gaussian_correlations = np.stack(gaussian_correlations)
    state_gaussian_correlation_cholesky = np.linalg.cholesky(
        state_gaussian_correlations + np.eye(len(assets))[None, :, :] * 1e-10
    )

    native_seed = (
        int(np.random.SeedSequence().generate_state(1, dtype=np.uint64)[0])
        if random_seed is None
        else int(random_seed) + 1
    )
    native_kwargs: dict[str, Any] | None = None
    normalized_multipliers: dict[str, float] = {}
    state_cost_values: np.ndarray | None = None
    monthly_fee_log: np.ndarray | None = None
    if native_portfolio_config is not None:
        native_kwargs = dict(native_portfolio_config["native_kwargs"])
        transaction_multipliers = native_portfolio_config.get(
            "state_transaction_cost_multipliers"
        )
        if transaction_multipliers:
            normalized_multipliers = {
                str(state): float(value)
                for state, value in dict(transaction_multipliers).items()
            }
            state_cost_values = np.array(
                [normalized_multipliers.get(state, 1.0) for state in model.states],
                dtype=float,
            )
        expense_ratios = np.asarray(native_portfolio_config["expense_ratios"], dtype=float)
        monthly_fee_log = np.log1p(-expense_ratios) / 12.0

        compact_requested = bool(native_portfolio_config.get("compact_reporting", False))
        if compact_requested:
            transition_rng = _rng(random_seed)
            transition = model.transition_matrix.loc[
                model.states, model.states
            ].to_numpy(dtype=float)
            if transition_concentration is not None:
                transition = _sample_transition_matrix(
                    transition_rng,
                    transition,
                    transition_concentration,
                )
            latest_state_age = (
                _latest_state_age_posterior(model)
                if start_state is None
                else None
            )
            start_probabilities = (
                latest_state_age.sum(axis=1)
                if latest_state_age is not None
                else stationary_distribution(
                    pd.DataFrame(
                        transition,
                        index=model.states,
                        columns=model.states,
                    )
                ).to_numpy(dtype=float)
            )
            start_state_index = (
                None
                if start_state is None
                else model.states.index(start_state)
                if start_state in model.states
                else -2
            )
            duration_hazard_values = None
            duration_hazard_lengths = None
            start_state_age_probabilities = None
            if str(duration_model).strip().lower() == "semi_markov":
                duration_hazard_map = model.metadata.get("duration_hazards")
                if not isinstance(duration_hazard_map, Mapping):
                    raise ValueError(
                        "semi_markov requires the model to expose 'duration_hazards' "
                        "in its metadata; recalibrate the model with duration support."
                    )
                state_hazards = [
                    np.asarray(duration_hazard_map.get(state, []), dtype=float)
                    for state in model.states
                ]
                if any(not len(values) for values in state_hazards):
                    raise ValueError("Every state must expose at least one duration hazard.")
                duration_hazard_lengths = np.asarray(
                    [len(values) for values in state_hazards],
                    dtype=np.int32,
                )
                duration_hazard_values = np.zeros(
                    (len(model.states), int(duration_hazard_lengths.max())),
                    dtype=float,
                )
                for state_index, values in enumerate(state_hazards):
                    duration_hazard_values[state_index, : len(values)] = values
                if latest_state_age is not None:
                    if latest_state_age.shape[1] > duration_hazard_values.shape[1]:
                        raise ValueError(
                            "Latest state-age posterior exceeds the duration hazard width."
                        )
                    start_state_age_probabilities = np.zeros_like(duration_hazard_values)
                    start_state_age_probabilities[:, : latest_state_age.shape[1]] = (
                        latest_state_age
                    )

            compact_kwargs = dict(native_kwargs)
            if compact_kwargs.pop("transaction_cost_rate_paths", None) is not None:
                compact_requested = False
            macro_process = None
            if compact_requested and joint_macro:
                if not isinstance(macro_dynamics, Mapping) or macro_betas is None:
                    raise ValueError("joint_macro requires calibrated macro dynamics.")
                emission_coefficients = (
                    _prepare_macro_emissions(
                        model.states,
                        model.metadata.get("hsmm_emission_means"),
                        model.metadata.get("hsmm_emission_covariances"),
                    )
                    if model.metadata.get("macro_membership_model", "joint_emission")
                    == "joint_emission"
                    else None
                )
                if emission_coefficients is None:
                    compact_requested = False
                else:
                    macro_columns = list(macro_dynamics["columns"])
                    rate_col = macro_dynamics.get("rate_col")
                    rate_index = (
                        macro_columns.index(rate_col)
                        if rate_col in macro_columns
                        else -1
                    )
                    macro_process = {
                        "latest": np.asarray(macro_dynamics["latest"], dtype=float),
                        "var_coefficient": np.asarray(
                            macro_dynamics["var_coefficient"],
                            dtype=float,
                        ),
                        "var_coefficient_std": np.asarray(
                            macro_dynamics.get(
                                "var_coefficient_std",
                                np.zeros_like(macro_dynamics["var_coefficient"]),
                            ),
                            dtype=float,
                        ),
                        "parameter_uncertainty": macro_parameter_uncertainty,
                        "state_centers": np.stack(
                            [
                                np.asarray(
                                    macro_dynamics["state_centers"][state],
                                    dtype=float,
                                )
                                for state in model.states
                            ]
                        ),
                        "state_innovation_covariances": np.stack(
                            [
                                nearest_psd(
                                    np.asarray(
                                        macro_dynamics["state_innovation_covariances"][state],
                                        dtype=float,
                                    )
                                )
                                for state in model.states
                            ]
                        ),
                        "emission_coefficients": emission_coefficients,
                        "transition_weight": macro_transition_weight,
                        "return_betas": macro_betas,
                        "rate_index": rate_index,
                        "rate_bounds": macro_dynamics.get("rate_bounds"),
                    }
            if compact_requested:
                compact_portfolio = simulate_parametric_italian_portfolios_compact_native(
                    periods=periods,
                    paths=paths,
                    transition_matrix=transition,
                    start_probabilities=start_probabilities,
                    start_state_index=start_state_index,
                    duration_model=duration_model,
                    duration_hazards=duration_hazard_values,
                    duration_hazard_lengths=duration_hazard_lengths,
                    min_regime_duration=min_regime_duration,
                    start_state_age_probabilities=start_state_age_probabilities,
                    macro_process=macro_process,
                    state_transaction_cost_multipliers=state_cost_values,
                    means=state_means,
                    gaussian_correlation_cholesky=state_gaussian_correlation_cholesky,
                    gaussian_correlations=state_gaussian_correlations,
                    volatilities=state_volatilities,
                    tail_indexes=state_tail_indexes,
                    temperings=state_temperings,
                    skewness=state_skewness,
                    gaussian_scales=state_gaussian_scales,
                    random_seed=native_seed,
                    regime_random_seed=(
                        int(np.random.SeedSequence().generate_state(1, dtype=np.uint64)[0])
                        if random_seed is None
                        else int(random_seed)
                    ),
                    garch=garch,
                    garch_alpha=garch_alpha,
                    garch_beta=garch_beta,
                    dynamic_correlation=dynamic_correlation,
                    dcc_alpha=dcc_alpha,
                    dcc_beta=dcc_beta,
                    dcc_asymmetry=dcc_asymmetry,
                    monthly_fee_log=monthly_fee_log,
                    return_kind=str(native_portfolio_config["return_kind"]),
                    reporting_paths=int(
                        native_portfolio_config.get("compact_reporting_paths", 25_000)
                    ),
                    annual_reporting_inflation=float(
                        native_portfolio_config.get("annual_reporting_inflation", 0.0)
                    ),
                    workers=native_threads,
                    **compact_kwargs,
                )
                if compact_portfolio is not None:
                    compact_portfolio["frame_metadata"] = dict(
                        native_portfolio_config["frame_metadata"]
                    )
                    compact_regimes = np.asarray(compact_portfolio["regimes"], dtype=np.uint8)
                    compact_macro_paths = compact_portfolio.get("macro_paths")
                    return SimulationResult(
                        returns=np.empty((periods, 0, len(assets)), dtype=float),
                        regimes=(
                            compact_regimes
                            if return_regime_codes
                            else _decode_regime_codes(compact_regimes, model.states)
                        ),
                        assets=assets,
                        states=model.states.copy(),
                        frequency=model.frequency,
                        distribution="mnts",
                        transition_concentration=transition_concentration,
                        macro_paths=(
                            np.asarray(compact_macro_paths, dtype=float)
                            if compact_macro_paths is not None
                            else None
                        ),
                        macro_columns=(
                            list(macro_dynamics["columns"])
                            if isinstance(macro_dynamics, Mapping)
                            else []
                        ),
                        native_portfolio=compact_portfolio,
                    )

    macro_paths: np.ndarray | None = None
    macro_shocks: np.ndarray | None = None
    if joint_macro:
        regime_paths, macro_paths, macro_shocks = simulate_joint_regime_macro_paths(
            model,
            periods=periods,
            paths=paths,
            start_state=start_state,
            random_seed=random_seed,
            transition_concentration=transition_concentration,
            duration_model=duration_model,
            min_regime_duration=min_regime_duration,
            macro_transition_weight=macro_transition_weight,
            macro_parameter_uncertainty=macro_parameter_uncertainty,
            return_codes=True,
        )
    else:
        regime_paths = simulate_regime_paths(
            model,
            periods=periods,
            paths=paths,
            start_state=start_state,
            random_seed=random_seed,
            transition_concentration=transition_concentration,
            duration_model=duration_model,
            min_regime_duration=min_regime_duration,
            return_codes=True,
        )
    rng = _rng(None if random_seed is None else random_seed + 1)

    if native_portfolio_config is not None and native_kwargs is not None:
        if state_cost_values is not None:
            native_kwargs["transaction_cost_rate_paths"] = (
                state_cost_values[np.asarray(regime_paths, dtype=int)]
                * float(native_kwargs["transaction_cost_bps"])
                / 10_000.0
            )
        fused_portfolio = simulate_parametric_italian_portfolios_native(
            regime_codes=regime_paths,
            means=state_means,
            gaussian_correlation_cholesky=state_gaussian_correlation_cholesky,
            gaussian_correlations=state_gaussian_correlations,
            volatilities=state_volatilities,
            tail_indexes=state_tail_indexes,
            temperings=state_temperings,
            skewness=state_skewness,
            gaussian_scales=state_gaussian_scales,
            random_seed=native_seed,
            garch=garch,
            garch_alpha=garch_alpha,
            garch_beta=garch_beta,
            dynamic_correlation=dynamic_correlation,
            dcc_alpha=dcc_alpha,
            dcc_beta=dcc_beta,
            dcc_asymmetry=dcc_asymmetry,
            monthly_fee_log=np.asarray(monthly_fee_log, dtype=float),
            return_kind=str(native_portfolio_config["return_kind"]),
            macro_shocks=macro_shocks,
            macro_betas=macro_betas,
            workers=native_threads,
            **native_kwargs,
        )
        if fused_portfolio is not None:
            fused_portfolio["frame_metadata"] = dict(native_portfolio_config["frame_metadata"])
            return SimulationResult(
                returns=np.empty((periods, 0, len(assets)), dtype=float),
                regimes=(
                    regime_paths
                    if return_regime_codes
                    else _decode_regime_codes(regime_paths, model.states)
                ),
                assets=assets,
                states=model.states.copy(),
                frequency=model.frequency,
                distribution="mnts",
                transition_concentration=transition_concentration,
                macro_paths=macro_paths,
                macro_columns=(
                    list(macro_dynamics["columns"])
                    if isinstance(macro_dynamics, Mapping)
                    else []
                ),
                native_portfolio=fused_portfolio,
            )

    native_returns = simulate_parametric_native(
        regime_codes=regime_paths,
        means=state_means,
        gaussian_correlation_cholesky=state_gaussian_correlation_cholesky,
        gaussian_correlations=state_gaussian_correlations,
        volatilities=state_volatilities,
        tail_indexes=state_tail_indexes,
        temperings=state_temperings,
        skewness=state_skewness,
        gaussian_scales=state_gaussian_scales,
        random_seed=native_seed,
        garch=garch,
        garch_alpha=garch_alpha,
        garch_beta=garch_beta,
        dynamic_correlation=dynamic_correlation,
        dcc_alpha=dcc_alpha,
        dcc_beta=dcc_beta,
        dcc_asymmetry=dcc_asymmetry,
        macro_shocks=macro_shocks,
        macro_betas=macro_betas,
        workers=native_threads,
    )
    if native_returns is not None:
        return SimulationResult(
            returns=native_returns,
            regimes=(
                regime_paths
                if return_regime_codes
                else _decode_regime_codes(regime_paths, model.states)
            ),
            assets=assets,
            states=model.states.copy(),
            frequency=model.frequency,
            distribution="mnts",
            transition_concentration=transition_concentration,
            macro_paths=macro_paths,
            macro_columns=(
                list(macro_dynamics["columns"])
                if isinstance(macro_dynamics, Mapping)
                else []
            ),
        )

    returns = np.empty((periods, paths, len(assets)), dtype=float)
    previous_state_indices = np.full(paths, -1, dtype=int)
    conditional_variance: np.ndarray | None = None
    if garch:
        conditional_variance = np.empty((paths, len(assets)), dtype=float)

    dcc_q: np.ndarray | None = None
    previous_standardized: np.ndarray | None = None
    if dynamic_correlation:
        dcc_q = np.empty((paths, len(assets), len(assets)), dtype=float)
        previous_standardized = np.zeros((paths, len(assets)), dtype=float)

    for period in range(periods):
        for state_index, state in enumerate(model.states):
            mask = regime_paths[period] == state_index
            if not mask.any():
                continue
            path_indices = np.flatnonzero(mask)
            mean = state_means[state_index]
            macro_effect = (
                macro_shocks[period, path_indices] @ macro_betas
                if macro_shocks is not None and macro_betas is not None
                else 0.0
            )
            reanchored = (period == 0) | (previous_state_indices[path_indices] != state_index)
            independent = rng.standard_normal((len(path_indices), len(assets)))
            if dynamic_correlation:
                base_correlation = state_gaussian_correlations[state_index]
                if reanchored.any():
                    dcc_q[path_indices[reanchored]] = base_correlation
                continuing = ~reanchored
                if continuing.any():
                    continuing_paths = path_indices[continuing]
                    previous = previous_standardized[continuing_paths]
                    negative = np.minimum(previous, 0.0)
                    dcc_q[continuing_paths] = (
                        (1.0 - dcc_alpha - dcc_beta - dcc_asymmetry) * base_correlation
                        + dcc_alpha * (previous[:, :, None] * previous[:, None, :])
                        + dcc_beta * dcc_q[continuing_paths]
                        + dcc_asymmetry * (negative[:, :, None] * negative[:, None, :])
                    )
                q_values = dcc_q[path_indices]
                cholesky = _batched_cholesky(q_values)
                q_scale = np.sqrt(
                    np.clip(np.diagonal(q_values, axis1=1, axis2=2), 1e-10, None)
                )
                gaussian_draws = np.einsum(
                    "nij,nj->ni", cholesky / q_scale[:, :, None], independent
                )
            else:
                gaussian_draws = independent @ state_gaussian_correlation_cholesky[state_index].T

            subordinator = sample_mnts_subordinators(
                rng,
                len(path_indices),
                state_tail_indexes[state_index],
                state_temperings[state_index],
            )
            standardized = (
                state_skewness[state_index] * (subordinator[:, None] - 1.0)
                + np.sqrt(subordinator)[:, None]
                * state_gaussian_scales[state_index]
                * gaussian_draws
            )
            if garch:
                levels = state_volatilities[state_index] ** 2
                if reanchored.any():
                    conditional_variance[path_indices[reanchored]] = levels
                scale = np.sqrt(np.maximum(conditional_variance[path_indices], 0.0))
                residual_draws = standardized * scale
                conditional_variance[path_indices] = (
                    (1.0 - garch_alpha - garch_beta) * levels
                    + garch_alpha * residual_draws**2
                    + garch_beta * conditional_variance[path_indices]
                )
            else:
                residual_draws = standardized * state_volatilities[state_index]
            draws = mean + macro_effect + residual_draws
            if dynamic_correlation:
                previous_standardized[path_indices] = standardized
            returns[period, mask, :] = draws
            previous_state_indices[path_indices] = state_index

    return SimulationResult(
        returns=returns,
        regimes=(
            regime_paths
            if return_regime_codes
            else _decode_regime_codes(regime_paths, model.states)
        ),
        assets=assets,
        states=model.states.copy(),
        frequency=model.frequency,
        distribution="mnts",
        transition_concentration=transition_concentration,
        macro_paths=macro_paths,
        macro_columns=(list(macro_dynamics["columns"]) if isinstance(macro_dynamics, Mapping) else []),
    )


def _simulate_leveraged_portfolio_paths(
    asset_growth: np.ndarray,
    target_weights: np.ndarray,
    initial_value: float,
    rebalance_frequency: int,
    transaction_cost_bps: float,
    leverage_multiple: float,
    financing_rate: float,
    maintenance_margin: float,
    contribution: float,
    withdrawal: float,
    withdrawal_start_period: int,
    decumulation: DecumulationPlan | None = None,
    withdrawal_inflation_paths: np.ndarray | None = None,
    annual_inflation: float = 0.0,
    safe_withdrawal_rate: float = 0.0,
    regimes: np.ndarray | None = None,
    state_financing_rates: Mapping[str, float] | None = None,
    financing_rate_paths: np.ndarray | None = None,
) -> pd.DataFrame:
    """Simulate leveraged holdings with explicit debt and financing costs.

    ``financing_rate_paths`` supplies a stochastic annual financing rate for
    every simulated month and path.  It takes precedence over the legacy
    regime-average map and scalar fallback.
    """

    periods, paths, assets = asset_growth.shape
    holdings = np.broadcast_to(
        initial_value * leverage_multiple * target_weights,
        (paths, assets),
    ).copy()
    debt = np.full(paths, initial_value * (leverage_multiple - 1.0), dtype=float)
    wealth = np.empty((periods, paths), dtype=float)
    plan = normalize_decumulation(
        decumulation,
        periods=periods,
        legacy_withdrawal=withdrawal,
        legacy_start_period=withdrawal_start_period,
        annual_inflation_fallback=annual_inflation,
    )
    cpi = inflation_index(
        periods,
        paths,
        annual_inflation=plan.annual_inflation_fallback,
        inflation_paths=withdrawal_inflation_paths,
    )
    spending = SpendingController(
        plan,
        paths=paths,
        initial_value=initial_value,
        cpi=cpi,
        safe_rate=safe_withdrawal_rate,
    )
    requested_spending = np.zeros((periods, paths), dtype=float)
    funded_spending = np.zeros((periods, paths), dtype=float)
    guardrail_events = np.zeros((periods, paths), dtype=np.int8)
    margin_calls = np.zeros(paths, dtype=bool)
    cost_rate = float(transaction_cost_bps) / 10_000.0

    effective_financing_rate = float(financing_rate)
    if financing_rate_paths is not None:
        annual_rates = np.asarray(financing_rate_paths, dtype=float)
        if annual_rates.shape != (periods, paths):
            raise ValueError("financing_rate_paths must have shape (periods, paths).")
        if not np.isfinite(annual_rates).all() or (annual_rates <= -1.0).any():
            raise ValueError("financing_rate_paths must contain finite annual rates above -100%.")
        financing_growth = np.power(1.0 + annual_rates, 1.0 / 12.0)
        effective_financing_rate = float(annual_rates.mean())
    elif regimes is not None and state_financing_rates:
        unique_states = np.unique(regimes.ravel())
        rate_by_state = np.array(
            [float(state_financing_rates.get(state, financing_rate)) for state in unique_states]
        )
        growth_lookup = (1.0 + rate_by_state) ** (1.0 / 12.0)
        state_codes = np.searchsorted(unique_states, regimes.ravel()).reshape(regimes.shape)
        financing_growth = growth_lookup[state_codes]
        effective_financing_rate = float(np.mean(rate_by_state[state_codes]))
    else:
        financing_growth = (1.0 + financing_rate) ** (1.0 / 12.0)

    for period in range(periods):
        if contribution:
            holdings += contribution * leverage_multiple * target_weights
            debt += contribution * (leverage_multiple - 1.0)
        holdings *= asset_growth[period]
        debt *= financing_growth[period] if np.ndim(financing_growth) == 2 else financing_growth

        asset_value = holdings.sum(axis=1)
        equity = asset_value - debt
        requested, policy_events = spending.request(period + 1, np.maximum(equity, 0.0))
        requested_spending[period] = requested
        guardrail_events[period] = policy_events
        if np.any(requested > 0):
            available = np.maximum(equity, 0.0)
            funded = funded_amount(requested, available)
            funded_spending[period] = funded
            fraction = np.divide(
                funded,
                asset_value,
                out=np.zeros_like(funded),
                where=asset_value > 0,
            )
            holdings -= holdings * fraction[:, None]
            exhausted = requested > equity
            if exhausted.any():
                holdings[exhausted] = 0.0
                debt[exhausted] = 0.0
                margin_calls[exhausted] = True

        asset_value = holdings.sum(axis=1)
        equity = asset_value - debt
        margin_ratio = np.divide(
            equity,
            asset_value,
            out=np.zeros_like(equity),
            where=asset_value > 0,
        )
        breached = (equity <= 0) | ((asset_value > 0) & (margin_ratio < maintenance_margin))
        if breached.any():
            holdings[breached] = 0.0
            debt[breached] = 0.0
            equity[breached] = 0.0
            margin_calls[breached] = True

        if (period + 1) % rebalance_frequency == 0:
            active = ~margin_calls
            target_holdings = equity[:, None] * leverage_multiple * target_weights
            turnover = np.abs(target_holdings - holdings).sum(axis=1)
            costs = turnover * cost_rate
            equity_after_costs = equity - costs
            liquidate = active & (equity_after_costs <= 0)
            if liquidate.any():
                holdings[liquidate] = 0.0
                debt[liquidate] = 0.0
                equity_after_costs[liquidate] = 0.0
                margin_calls[liquidate] = True
            active = ~margin_calls
            holdings[active] = equity_after_costs[active, None] * leverage_multiple * target_weights
            debt[active] = equity_after_costs[active] * (leverage_multiple - 1.0)
            equity = equity_after_costs

        wealth[period] = np.maximum(equity, 0.0)

    frame = pd.DataFrame(wealth, columns=[f"path_{i}" for i in range(paths)])
    frame.attrs.update(
        {
            "margin_calls": int(margin_calls.sum()),
            "effective_financing_rate": effective_financing_rate,
            "tax_country": "none",
            "tax_regime": "none",
            "decumulation": plan.to_dict(),
            "withdrawal_requested": requested_spending,
            "withdrawal_funded": funded_spending,
            "guardrail_events": guardrail_events,
            "withdrawal_cpi": cpi,
        }
    )
    return frame


def simulate_portfolio_paths(
    result: SimulationResult,
    weights: Mapping[str, float],
    initial_value: float = 100.0,
    return_kind: str = "log",
    rebalance_frequency: int | None = None,
    transaction_cost_bps: float = 0.0,
    state_transaction_cost_multipliers: Mapping[str, float] | None = None,
    asset_expense_ratios: Mapping[str, float] | None = None,
    leverage_multiple: float = 1.0,
    financing_rate: float = 0.0,
    financing_inflation_sensitivity: float = 0.0,
    state_inflation: Mapping[str, float] | None = None,
    financing_rate_paths: np.ndarray | None = None,
    financing_inflation_paths: np.ndarray | None = None,
    maintenance_margin: float = 0.0,
    contribution: float = 0.0,
    contribution_allocation: str = "target",
    withdrawal: float = 0.0,
    withdrawal_start_period: int = 1,
    decumulation: Mapping[str, Any] | DecumulationPlan | None = None,
    withdrawal_inflation_paths: np.ndarray | None = None,
    annual_inflation: float = 0.0,
    safe_withdrawal_rate: float = 0.0,
    tax_regime: str = "none",
    asset_tax_categories: Mapping[str, str] | None = None,
    asset_tax_metadata: Mapping[str, Mapping[str, object]] | None = None,
    italy_annual_wealth_tax: float = ITALY_DEFAULT_WEALTH_TAX_RATE,
    italy_wealth_tax_mode: str = "auto",
    tax_terminal_liquidation: bool = True,
    tax_start_date: str | None = None,
    tax_wrapper_benchmark: bool = False,
    tax_country: str | None = None,
    native_threads: int = 1,
) -> pd.DataFrame:
    """Convert simulated asset returns into portfolio wealth paths.

    With ``rebalance_frequency=None`` the original weighted-return behavior is
    retained. ``0`` models true buy-and-hold with drifting asset weights, while
    a positive frequency models holdings, periodic rebalancing, and transaction
    costs in basis points charged on traded notional.

    ``contribution`` and ``withdrawal`` are periodic cash flows in the same
    currency as ``initial_value``. A contribution is invested at the target
    allocation at the start of every period (dollar-cost averaging); a
    withdrawal is funded by selling a pro-rata slice of current holdings at
    the end of every period from ``withdrawal_start_period`` onward. Period 1
    means immediately. Wealth is floored at zero, so a path cannot be driven
    negative by withdrawals.
    """

    if not np.isfinite(initial_value) or initial_value <= 0:
        raise ValueError("initial_value must be positive and finite.")
    if result.returns.ndim != 3 or result.returns.shape[2] != len(result.assets):
        raise ValueError("result.returns must have shape (periods, paths, assets).")
    if not np.isfinite(result.returns).all():
        raise ValueError("Simulated returns must contain only finite values.")
    if not np.isfinite(contribution) or contribution < 0:
        raise ValueError("contribution must be a finite, non-negative number.")
    contribution_allocation = str(contribution_allocation).strip().lower()
    if contribution_allocation not in CONTRIBUTION_ALLOCATION_MODES:
        allowed = ", ".join(sorted(CONTRIBUTION_ALLOCATION_MODES))
        raise ValueError(f"contribution_allocation must be one of: {allowed}.")
    if not np.isfinite(withdrawal) or withdrawal < 0:
        raise ValueError("withdrawal must be a finite, non-negative number.")
    periods = int(result.returns.shape[0])
    try:
        withdrawal_start_value = float(withdrawal_start_period)
        withdrawal_start_period = int(withdrawal_start_value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("withdrawal_start_period must be an integer.") from exc
    if not np.isfinite(withdrawal_start_value) or not np.isclose(
        withdrawal_start_value, withdrawal_start_period
    ):
        raise ValueError("withdrawal_start_period must be an integer.")
    if not 1 <= withdrawal_start_period <= periods:
        raise ValueError(
            "withdrawal_start_period must be between 1 and the simulation periods."
        )
    decumulation_plan = normalize_decumulation(
        decumulation,
        periods=periods,
        legacy_withdrawal=withdrawal,
        legacy_start_period=withdrawal_start_period,
        annual_inflation_fallback=annual_inflation,
    )
    cpi = inflation_index(
        periods,
        int(result.returns.shape[1]),
        annual_inflation=decumulation_plan.annual_inflation_fallback,
        inflation_paths=withdrawal_inflation_paths,
    )
    if not np.isfinite(leverage_multiple) or leverage_multiple < 1:
        raise ValueError("leverage_multiple must be at least 1.0.")
    if not np.isfinite(financing_rate) or financing_rate < 0:
        raise ValueError("financing_rate must be a finite, non-negative number.")
    if not np.isfinite(financing_inflation_sensitivity) or financing_inflation_sensitivity < 0:
        raise ValueError("financing_inflation_sensitivity must be a finite, non-negative number.")
    if not np.isfinite(maintenance_margin) or not 0 <= maintenance_margin < 1:
        raise ValueError("maintenance_margin must be between 0 and 1.")
    if np.isclose(leverage_multiple, 1.0) and not np.isclose(maintenance_margin, 0.0):
        raise ValueError("maintenance_margin only applies when leverage_multiple is greater than 1.")
    if leverage_multiple > 1.0 and maintenance_margin >= 1.0 / leverage_multiple:
        raise ValueError("maintenance_margin must be below the initial equity margin for the selected leverage.")
    if state_transaction_cost_multipliers and leverage_multiple > 1.0:
        raise ValueError("State-dependent transaction costs are not available with leveraged accounting.")
    tax_selection = resolve_tax_selection(tax_country, tax_regime)

    provided_weights = pd.Series(weights, dtype=float)
    weight_vector = provided_weights.reindex(result.assets)
    missing_assets = ~pd.Index(result.assets).isin(provided_weights.index)
    weight_vector.loc[missing_assets] = 0.0
    if not np.isfinite(weight_vector.to_numpy(dtype=float)).all():
        raise ValueError("Portfolio weights must be finite numbers.")
    weight_total = float(weight_vector.sum())
    if not np.isfinite(weight_total) or np.isclose(weight_total, 0.0):
        raise ValueError("Portfolio weights must have a non-zero sum.")
    if not np.isclose(weight_total, 1.0):
        weight_vector = weight_vector / weight_total

    return_kind = str(return_kind).lower()
    if return_kind not in {"log", "simple"}:
        raise ValueError("return_kind must be 'log' or 'simple'.")
    if rebalance_frequency is not None:
        rebalance_frequency = int(rebalance_frequency)
        if rebalance_frequency < 0:
            raise ValueError("rebalance_frequency must be non-negative or None.")
    if contribution_allocation == "underweight_first" and rebalance_frequency is None:
        raise ValueError("Underweight-first contributions require holdings-based accounting.")
    if contribution_allocation == "underweight_first" and not np.isclose(leverage_multiple, 1.0):
        raise ValueError("Underweight-first contributions are not available with leveraged accounting.")
    if not np.isfinite(transaction_cost_bps) or transaction_cost_bps < 0:
        raise ValueError("transaction_cost_bps must be a non-negative number.")
    if rebalance_frequency in {None, 0} and not np.isclose(transaction_cost_bps, 0.0):
        raise ValueError("Transaction costs require monthly, quarterly, or annual rebalancing.")
    if rebalance_frequency in {None, 0} and (
        not np.isclose(leverage_multiple, 1.0)
        or not np.isclose(financing_rate, 0.0)
        or not np.isclose(maintenance_margin, 0.0)
    ):
        raise ValueError("Leverage and financing require a rebalancing frequency.")

    target_weights = weight_vector.to_numpy(dtype=float)
    if tax_selection.policy is not None:
        tax_selection.policy.validate(
            rebalance_frequency=rebalance_frequency,
            leverage_multiple=leverage_multiple,
            target_weights=target_weights,
        )

    provided_expense_ratios = pd.Series(asset_expense_ratios or {}, dtype=float)
    expense_ratios = provided_expense_ratios.reindex(result.assets).fillna(0.0)
    if not np.isfinite(expense_ratios.to_numpy(dtype=float)).all() or (expense_ratios < 0).any() or (expense_ratios >= 1).any():
        raise ValueError("Asset expense ratios must be finite decimals between 0 and 1.")
    monthly_fee_log = np.log1p(-expense_ratios.to_numpy(dtype=float)) / 12.0
    monthly_fee_growth = np.exp(monthly_fee_log)

    if rebalance_frequency is None:
        if return_kind == "log":
            portfolio_returns = (result.returns + monthly_fee_log) @ weight_vector.to_numpy(dtype=float)
            growth = np.exp(portfolio_returns)
            wealth = initial_value * np.exp(np.cumsum(portfolio_returns, axis=0))
        else:
            net_asset_returns = (1.0 + result.returns) * monthly_fee_growth - 1.0
            portfolio_returns = net_asset_returns @ weight_vector.to_numpy(dtype=float)
            if (1.0 + portfolio_returns <= 0).any():
                raise ValueError("Simple returns must be greater than -100% for positive wealth.")
            growth = 1.0 + portfolio_returns
            wealth = initial_value * np.cumprod(1.0 + portfolio_returns, axis=0)
        if contribution or decumulation_plan.active:
            periods, paths = result.returns.shape[:2]
            value = np.full(paths, initial_value, dtype=float)
            wealth = np.empty((periods, paths), dtype=float)
            spending = SpendingController(
                decumulation_plan,
                paths=paths,
                initial_value=initial_value,
                cpi=cpi,
                safe_rate=safe_withdrawal_rate,
            )
            requested_spending = np.zeros((periods, paths), dtype=float)
            funded_spending = np.zeros((periods, paths), dtype=float)
            guardrail_events = np.zeros((periods, paths), dtype=np.int8)
            for period in range(periods):
                value = (value + contribution) * growth[period]
                requested, policy_events = spending.request(period + 1, value)
                funded = funded_amount(requested, value)
                value = np.maximum(value - funded, 0.0)
                requested_spending[period] = requested
                funded_spending[period] = funded
                guardrail_events[period] = policy_events
                wealth[period] = value
        else:
            paths = result.returns.shape[1]
            requested_spending = np.zeros((periods, paths), dtype=float)
            funded_spending = np.zeros((periods, paths), dtype=float)
            guardrail_events = np.zeros((periods, paths), dtype=np.int8)
        if not np.isfinite(wealth).all():
            raise ValueError("Portfolio wealth contains non-finite values.")
        frame = pd.DataFrame(wealth, columns=[f"path_{i}" for i in range(result.returns.shape[1])])
        frame.attrs.update(
            {
                "margin_calls": 0,
                "tax_country": "none",
                "tax_regime": "none",
                "withdrawal_start_period": withdrawal_start_period,
                "decumulation": decumulation_plan.to_dict(),
                "withdrawal_requested": requested_spending,
                "withdrawal_funded": funded_spending,
                "guardrail_events": guardrail_events,
                "withdrawal_cpi": cpi,
            }
        )
        return frame

    if return_kind == "log":
        asset_growth = np.exp(result.returns + monthly_fee_log)
    else:
        asset_growth = (1.0 + result.returns) * monthly_fee_growth
        if (asset_growth <= 0).any():
            raise ValueError("Simple returns must be greater than -100% for rebalancing.")
    if not np.isfinite(asset_growth).all():
        raise ValueError("Asset growth contains non-finite values.")

    periods, paths, assets = result.returns.shape
    transaction_cost_rate_paths = _transaction_cost_rate_paths(
        result,
        transaction_cost_bps,
        state_transaction_cost_multipliers,
    )
    if tax_selection.policy is not None:
        context = TaxSimulationContext(
            asset_growth=asset_growth,
            assets=result.assets,
            target_weights=target_weights,
            initial_value=initial_value,
            rebalance_frequency=int(rebalance_frequency),
            transaction_cost_bps=transaction_cost_bps,
            transaction_cost_rate_paths=transaction_cost_rate_paths,
            contribution=contribution,
            contribution_allocation=contribution_allocation,
            withdrawal=withdrawal,
            withdrawal_start_period=withdrawal_start_period,
            decumulation=decumulation_plan,
            withdrawal_inflation_paths=withdrawal_inflation_paths,
            annual_inflation=annual_inflation,
            safe_withdrawal_rate=safe_withdrawal_rate,
            asset_categories=asset_tax_categories,
            asset_metadata=asset_tax_metadata,
            annual_wealth_tax=italy_annual_wealth_tax,
            wealth_tax_mode=italy_wealth_tax_mode,
            terminal_liquidation=tax_terminal_liquidation,
            start_date=tax_start_date,
            wrapper_benchmark=tax_wrapper_benchmark,
            native_threads=max(1, int(native_threads)),
        )
        frame = tax_selection.policy.simulate(context)
        frame.attrs.update(
            {
                "tax_country": tax_selection.country,
                "tax_regime": tax_selection.regime,
            }
        )
        return frame
    if (
        not np.isclose(leverage_multiple, 1.0)
        or not np.isclose(financing_rate, 0.0)
        or not np.isclose(maintenance_margin, 0.0)
    ):
        state_financing_rates = None
        path_financing_rates = None
        if financing_rate_paths is not None:
            short_rates = np.asarray(financing_rate_paths, dtype=float)
            if short_rates.shape != (periods, paths):
                raise ValueError("financing_rate_paths must have shape (periods, paths).")
            path_financing_rates = short_rates + float(financing_rate)
            if not np.isclose(financing_inflation_sensitivity, 0.0):
                if financing_inflation_paths is None:
                    raise ValueError(
                        "financing_inflation_paths are required when dynamic rates use inflation sensitivity."
                    )
                inflation_rates = np.asarray(financing_inflation_paths, dtype=float)
                if inflation_rates.shape != (periods, paths):
                    raise ValueError(
                        "financing_inflation_paths must have shape (periods, paths)."
                    )
                path_financing_rates += financing_inflation_sensitivity * inflation_rates
            path_financing_rates = np.clip(path_financing_rates, 0.0, 1.0)
        elif not np.isclose(financing_inflation_sensitivity, 0.0) and result.states and state_inflation:
            state_financing_rates = {
                state: financing_rate + financing_inflation_sensitivity * float(state_inflation.get(state, 0.0))
                for state in result.states
            }
        return _simulate_leveraged_portfolio_paths(
            asset_growth,
            target_weights,
            initial_value,
            rebalance_frequency,
            transaction_cost_bps,
            leverage_multiple,
            financing_rate,
            maintenance_margin,
            contribution,
            withdrawal,
            withdrawal_start_period,
            decumulation=decumulation_plan,
            withdrawal_inflation_paths=withdrawal_inflation_paths,
            annual_inflation=annual_inflation,
            safe_withdrawal_rate=safe_withdrawal_rate,
            regimes=(
                _decode_regime_codes(result.regimes, result.states)
                if state_financing_rates and result.regimes.dtype.kind in "iu"
                else result.regimes if state_financing_rates else None
            ),
            state_financing_rates=state_financing_rates,
            financing_rate_paths=path_financing_rates,
        )
    holdings = np.broadcast_to(
        initial_value * target_weights,
        (paths, assets),
    ).copy()
    wealth = np.empty((periods, paths), dtype=float)
    cost_rate = float(transaction_cost_bps) / 10_000.0
    transaction_cost_total = np.zeros(paths, dtype=float)
    spending = SpendingController(
        decumulation_plan,
        paths=paths,
        initial_value=initial_value,
        cpi=cpi,
        safe_rate=safe_withdrawal_rate,
    )
    requested_spending = np.zeros((periods, paths), dtype=float)
    funded_spending = np.zeros((periods, paths), dtype=float)
    guardrail_events = np.zeros((periods, paths), dtype=np.int8)

    for period in range(periods):
        if contribution:
            holdings += contribution_allocations(
                holdings,
                target_weights,
                contribution,
                contribution_allocation,
            )
        holdings *= asset_growth[period]
        values = holdings.sum(axis=1)
        requested, policy_events = spending.request(period + 1, values)
        funded = funded_amount(requested, values)
        requested_spending[period] = requested
        funded_spending[period] = funded
        guardrail_events[period] = policy_events
        if np.any(funded > 0):
            fraction = funded / np.maximum(values, 1e-300)
            holdings -= holdings * fraction[:, None]
            holdings = np.maximum(holdings, 0.0)
        value_before_rebalance = holdings.sum(axis=1)
        if rebalance_frequency > 0 and (period + 1) % rebalance_frequency == 0:
            target_holdings = value_before_rebalance[:, None] * target_weights
            turnover = np.abs(target_holdings - holdings).sum(axis=1)
            active_cost_rate = (
                transaction_cost_rate_paths[period]
                if transaction_cost_rate_paths is not None
                else cost_rate
            )
            costs = turnover * active_cost_rate
            transaction_cost_total += costs
            value_after_costs = value_before_rebalance - costs
            holdings = value_after_costs[:, None] * target_weights
            wealth[period] = value_after_costs
        else:
            wealth[period] = value_before_rebalance

    if not np.isfinite(wealth).all():
        raise ValueError("Portfolio wealth contains non-finite values.")
    frame = pd.DataFrame(wealth, columns=[f"path_{i}" for i in range(result.returns.shape[1])])
    frame.attrs.update(
        {
            "margin_calls": 0,
            "tax_country": "none",
            "tax_regime": "none",
            "transaction_cost_total": float(transaction_cost_total.sum()),
            "state_dependent_transaction_costs": bool(state_transaction_cost_multipliers),
            "withdrawal_start_period": withdrawal_start_period,
            "decumulation": decumulation_plan.to_dict(),
            "withdrawal_requested": requested_spending,
            "withdrawal_funded": funded_spending,
            "guardrail_events": guardrail_events,
            "withdrawal_cpi": cpi,
        }
    )
    return frame


def summarize_terminal_wealth(wealth: pd.DataFrame) -> pd.Series:
    """Summarize terminal wealth and downside risk across Monte Carlo paths."""

    return summarize_wealth_risk(wealth)


def inflation_deflators(
    periods: int,
    paths: int,
    periods_per_year: float = 12.0,
    annual_inflation: float = 0.0,
    inflation_paths: np.ndarray | None = None,
) -> np.ndarray:
    """Return cumulative nominal-to-real deflators for every simulated path."""

    if inflation_paths is None:
        period = np.arange(1, periods + 1, dtype=float)
        scalar = (1.0 + annual_inflation) ** (-period / periods_per_year)
        return np.broadcast_to(scalar[:, None], (periods, paths))
    rates = np.asarray(inflation_paths, dtype=float)
    if rates.shape != (periods, paths):
        raise ValueError("inflation_paths must have shape (periods, paths).")
    if not np.isfinite(rates).all() or (rates <= -1.0).any():
        raise ValueError("inflation_paths must contain finite annual rates greater than -100%.")
    periodic_growth = np.power(1.0 + rates, 1.0 / periods_per_year)
    return 1.0 / np.cumprod(periodic_growth, axis=0)


def inflation_adjust_wealth(
    wealth: pd.DataFrame,
    periods_per_year: float = 12.0,
    annual_inflation: float = 0.0,
    inflation_paths: np.ndarray | None = None,
) -> pd.DataFrame:
    """Convert nominal wealth paths to path-consistent purchasing power."""

    deflators = inflation_deflators(
        len(wealth),
        wealth.shape[1],
        periods_per_year=periods_per_year,
        annual_inflation=annual_inflation,
        inflation_paths=inflation_paths,
    )
    adjusted = pd.DataFrame(
        wealth.to_numpy(dtype=float) * deflators,
        index=wealth.index,
        columns=wealth.columns,
    )
    adjusted.attrs.update(wealth.attrs)
    return adjusted


def summarize_wealth_risk(
    wealth: pd.DataFrame,
    initial_value: float = 100.0,
    confidence: float = 0.95,
    periods_per_year: float = 12.0,
    risk_free_rate: float = 0.0,
    annual_inflation: float = 0.0,
    contribution: float = 0.0,
    withdrawal: float = 0.0,
    withdrawal_start_period: int = 1,
    withdrawal_paths: np.ndarray | None = None,
    inflation_paths: np.ndarray | None = None,
    risk_free_paths: np.ndarray | None = None,
) -> pd.Series:
    """Calculate terminal, loss-tail, drawdown, and annualized metrics.

    With ``annual_inflation > 0`` all metrics are computed on inflation-adjusted
    wealth, so results are expressed in real (purchasing power) terms. The
    Sharpe ratio uses ``risk_free_rate`` as the annualized risk-free return.
    """

    if wealth.empty or wealth.shape[1] == 0:
        raise ValueError("wealth must contain at least one simulated path.")
    if not np.isfinite(initial_value) or initial_value <= 0:
        raise ValueError("initial_value must be positive and finite.")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1.")
    if not np.isfinite(periods_per_year) or periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive and finite.")
    if not np.isfinite(risk_free_rate):
        raise ValueError("risk_free_rate must be finite.")
    if not np.isfinite(annual_inflation) or annual_inflation < 0:
        raise ValueError("annual_inflation must be a finite, non-negative number.")
    if not np.isfinite(contribution) or contribution < 0:
        raise ValueError("contribution must be a finite, non-negative number.")
    if not np.isfinite(withdrawal) or withdrawal < 0:
        raise ValueError("withdrawal must be a finite, non-negative number.")
    try:
        wealth_values = wealth.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("wealth must contain numeric values.") from exc
    if not np.isfinite(wealth_values).all():
        raise ValueError("wealth must contain only finite values.")

    periods, paths = wealth_values.shape
    try:
        withdrawal_start_value = float(withdrawal_start_period)
        withdrawal_start_period = int(withdrawal_start_value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("withdrawal_start_period must be an integer.") from exc
    if not np.isfinite(withdrawal_start_value) or not np.isclose(
        withdrawal_start_value, withdrawal_start_period
    ):
        raise ValueError("withdrawal_start_period must be an integer.")
    if not 1 <= withdrawal_start_period <= periods:
        raise ValueError(
            "withdrawal_start_period must be between 1 and the available periods."
        )
    if inflation_paths is None:
        period = np.arange(1, periods + 1, dtype=float)
        deflator = ((1.0 + annual_inflation) ** (-period / periods_per_year))[:, None]
        if not np.isclose(annual_inflation, 0.0):
            wealth_values = wealth_values * deflator
    else:
        deflator = inflation_deflators(
            periods,
            paths,
            periods_per_year=periods_per_year,
            annual_inflation=annual_inflation,
            inflation_paths=inflation_paths,
        )
        wealth_values = wealth_values * deflator

    if risk_free_paths is not None:
        nominal_risk_free = np.asarray(risk_free_paths, dtype=float)
        if nominal_risk_free.shape != (periods, paths):
            raise ValueError("risk_free_paths must have shape (periods, paths).")
        if not np.isfinite(nominal_risk_free).all() or (nominal_risk_free <= -1.0).any():
            raise ValueError("risk_free_paths must contain finite annual rates above -100%.")
    if risk_free_paths is None and inflation_paths is None:
        real_risk_free = (1.0 + float(risk_free_rate)) / (1.0 + annual_inflation) - 1.0
        periodic_risk_free: float | np.ndarray = float(
            (1.0 + real_risk_free) ** (1.0 / periods_per_year) - 1.0
        )
        effective_risk_free_rate = float(real_risk_free)
    else:
        nominal_risk_free = (
            nominal_risk_free
            if risk_free_paths is not None
            else float(risk_free_rate)
        )
        annual_inflation_values = (
            np.asarray(inflation_paths, dtype=float)
            if inflation_paths is not None
            else float(annual_inflation)
        )
        real_risk_free_values = (
            (1.0 + nominal_risk_free) / (1.0 + annual_inflation_values) - 1.0
        )
        periodic_risk_free = (
            np.power(1.0 + real_risk_free_values, 1.0 / periods_per_year) - 1.0
        )
        effective_risk_free_rate = float(np.mean(real_risk_free_values))
    terminal = wealth_values[-1]
    tail_probability = 1.0 - confidence
    lower_tail = float(np.quantile(terminal, tail_probability))
    tail = terminal[terminal <= lower_tail]
    contribution_deflator = np.vstack([np.ones((1, deflator.shape[1])), deflator[:-1]])
    withdrawal_deflator = deflator
    real_contributions = contribution * contribution_deflator
    if withdrawal_paths is None:
        withdrawal_schedule = (
            np.arange(1, periods + 1, dtype=int) >= withdrawal_start_period
        ).astype(float)[:, None]
        nominal_withdrawals = withdrawal * withdrawal_schedule
    else:
        nominal_withdrawals = np.asarray(withdrawal_paths, dtype=float)
        if nominal_withdrawals.shape != (periods, paths):
            raise ValueError("withdrawal_paths must have shape (periods, paths).")
        if not np.isfinite(nominal_withdrawals).all() or (nominal_withdrawals < 0).any():
            raise ValueError("withdrawal_paths must be finite and non-negative.")
    real_withdrawals = nominal_withdrawals * withdrawal_deflator

    # Compute per-path drawdown and downside metrics in blocks so the full
    # (periods x paths) matrix never needs to be copied multiple times.
    max_drawdown = np.empty(paths, dtype=float)
    ulcer = np.empty(paths, dtype=float)
    return_sum = 0.0
    return_squares = 0.0
    return_count = 0
    excess_return_sum = 0.0
    log_return_sum = 0.0
    log_return_count = 0
    downside_sum = 0.0
    downside_count = 0
    block = max(1, int(4096))
    for start in range(0, paths, block):
        values = wealth_values[:, start:start + block]
        values_with_initial = np.vstack([np.full(values.shape[1], initial_value), values])
        running_max = np.maximum.accumulate(values_with_initial, axis=0)
        drawdown = values_with_initial / running_max - 1.0
        max_drawdown[start:start + values.shape[1]] = -drawdown.min(axis=0)
        ulcer[start:start + values.shape[1]] = np.sqrt(np.mean(drawdown**2, axis=0))

        previous = np.vstack([np.full(values.shape[1], initial_value), values[:-1]])
        contribution_values = (
            real_contributions
            if real_contributions.shape[1] == 1
            else real_contributions[:, start:start + values.shape[1]]
        )
        withdrawal_values = (
            real_withdrawals
            if real_withdrawals.shape[1] == 1
            else real_withdrawals[:, start:start + values.shape[1]]
        )
        denominator = previous + contribution_values
        numerator = values + withdrawal_values
        with np.errstate(divide="ignore", invalid="ignore"):
            period_returns = numerator / denominator - 1.0
        period_returns[(denominator <= 0) | (numerator < 0)] = np.nan
        finite_returns = period_returns[np.isfinite(period_returns)]
        finite_mask = np.isfinite(period_returns)
        if np.ndim(periodic_risk_free) == 0:
            excess_returns = finite_returns - float(periodic_risk_free)
        else:
            finite_risk_free = periodic_risk_free[:, start:start + values.shape[1]][finite_mask]
            excess_returns = finite_returns - finite_risk_free
        return_sum += float(finite_returns.sum())
        return_squares += float(np.square(finite_returns).sum())
        return_count += int(finite_returns.size)
        valid_log_returns = finite_returns[finite_returns > -1.0]
        log_return_sum += float(np.log1p(valid_log_returns).sum())
        log_return_count += int(valid_log_returns.size)
        excess_return_sum += float(excess_returns.sum())
        downside_sum += float(np.sum(np.where(excess_returns < 0, excess_returns**2, 0.0)))
        downside_count += int(excess_returns.size)

    mean_period_return = return_sum / return_count if return_count else 0.0
    period_variance = max(return_squares / return_count - mean_period_return**2, 0.0) if return_count else 0.0
    annualized_return = float(mean_period_return * periods_per_year)
    annualized_excess_return = float(
        excess_return_sum / return_count * periods_per_year
    ) if return_count else 0.0
    annualized_volatility = float(np.sqrt(period_variance) * np.sqrt(periods_per_year))
    sharpe_ratio = (
        float(annualized_excess_return / annualized_volatility)
        if annualized_volatility > 0
        else 0.0
    )
    downside_deviation = float(np.sqrt(downside_sum / downside_count)) if downside_count else 0.0
    annualized_downside = downside_deviation * np.sqrt(periods_per_year)
    sortino_ratio = (
        float(annualized_excess_return / annualized_downside) if annualized_downside > 0 else 0.0
    )
    mean_max_drawdown = float(max_drawdown.mean())
    geometric_annualized_return = (
        float(np.exp(log_return_sum / log_return_count * periods_per_year) - 1.0)
        if log_return_count
        else 0.0
    )
    calmar_ratio = (
        float(geometric_annualized_return / mean_max_drawdown) if mean_max_drawdown > 0 else 0.0
    )
    skewness = pd.Series(terminal).skew()
    kurtosis = pd.Series(terminal).kurt()
    summary = {
        "mean": float(terminal.mean()),
        "std": float(terminal.std(ddof=0)),
        "p05": float(np.quantile(terminal, 0.05)),
        "p50": float(np.quantile(terminal, 0.50)),
        "p95": float(np.quantile(terminal, 0.95)),
        "annualized_return": float(annualized_return),
        "effective_risk_free_rate": effective_risk_free_rate,
        "annualized_volatility": float(annualized_volatility),
        "geometric_annualized_return": geometric_annualized_return,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "calmar_ratio": calmar_ratio,
        "probability_of_loss": float((terminal < initial_value).mean()),
        "var_95": initial_value - lower_tail,
        "expected_shortfall_95": initial_value - float(tail.mean()),
        "max_drawdown_mean": mean_max_drawdown,
        "max_drawdown_p95": float(np.quantile(max_drawdown, 0.95)),
        "max_drawdown_worst": float(max_drawdown.max()),
        "ulcer_index_mean": float(ulcer.mean()),
        "ulcer_index_p95": float(np.quantile(ulcer, 0.95)),
        "terminal_skewness": float(skewness) if np.isfinite(skewness) else 0.0,
        "terminal_kurtosis": float(kurtosis) if np.isfinite(kurtosis) else 0.0,
    }

    has_withdrawals = bool(withdrawal or np.any(nominal_withdrawals > 0))
    if contribution or has_withdrawals:
        contribution_periods = periods
        if withdrawal_paths is not None:
            active_withdrawal_months = np.any(nominal_withdrawals > 0, axis=1)
            withdrawal_periods = int(active_withdrawal_months.sum())
            effective_withdrawal_start = (
                int(np.argmax(active_withdrawal_months)) + 1
                if np.any(active_withdrawal_months)
                else withdrawal_start_period
            )
        else:
            withdrawal_periods = periods - withdrawal_start_period + 1
            effective_withdrawal_start = withdrawal_start_period
        summary.update(
            {
                "cash_flow_adjusted_annualized_return": annualized_return,
                "cash_flow_adjusted_volatility": annualized_volatility,
                "cash_flow_adjusted_sharpe_ratio": sharpe_ratio,
                "total_contributed": float(contribution * contribution_periods),
                "total_withdrawn": float(np.mean(real_withdrawals.sum(axis=0))),
                "net_external_cash_flow": float(
                    contribution * contribution_periods
                    - np.mean(real_withdrawals.sum(axis=0))
                ),
            }
        )
        if has_withdrawals:
            summary["withdrawal_start_period"] = effective_withdrawal_start
            summary["withdrawal_periods"] = withdrawal_periods

    return pd.Series(summary)
