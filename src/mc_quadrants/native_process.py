"""Input preparation for native standalone regime and joint-macro paths."""
from __future__ import annotations

import ctypes
import os

import numpy as np

from mc_quadrants.matrix import nearest_psd
from mc_quadrants.native import _load_library, _MacroProcessConfig, _pointer, _RegimeProcessConfig


def simulate_regime_macro_native(
    model, transition, start_probabilities, posterior, *, periods, paths, start_state,
    random_seed, duration_model, min_regime_duration, macro_dynamics=None,
    emission_coefficients=None, macro_transition_weight=0.35,
    macro_parameter_uncertainty=True, workers=1,
):
    if os.getenv("MC_DISABLE_NATIVE_SIM", "").strip().lower() in {"1", "true", "yes", "on"}:
        return None
    library = _load_library()
    if library is None:
        return None
    states = len(model.states)
    if not 1 <= states <= 256:
        return None
    if periods <= 0 or paths <= 0 or min_regime_duration <= 0:
        raise ValueError("periods, paths and min_regime_duration must be positive.")
    if duration_model not in {"markov", "semi_markov"}:
        raise ValueError("Invalid duration_model.")
    buffers = []

    def keep(values, shape=None, dtype=np.float64):
        array = np.ascontiguousarray(values, dtype=dtype)
        if shape is not None and array.shape != shape:
            raise ValueError(f"Native process input must have shape {shape}, got {array.shape}.")
        if not np.isfinite(array).all():
            raise ValueError("Native process inputs must be finite.")
        buffers.append(array)
        return array

    transition = keep(transition, (states, states))
    if (transition < 0).any() or (transition.sum(axis=1) <= 0).any():
        raise ValueError("transition_matrix must have non-negative entries and positive row sums.")
    transition = keep(transition / transition.sum(axis=1, keepdims=True))
    starts = keep(start_probabilities, (states,))
    if (starts < 0).any() or starts.sum() <= 0:
        raise ValueError("start_probabilities must have non-negative entries and positive mass.")
    starts = keep(starts / starts.sum())
    if start_state is not None and start_state not in model.states:
        raise ValueError(f"Unknown start_state: {start_state}")
    start_index = -1 if start_state is None else model.states.index(start_state)
    hazards = lengths = age = None
    width = 0
    if duration_model == "semi_markov":
        hazard_map = model.metadata.get("duration_hazards")
        if not isinstance(hazard_map, dict):
            raise ValueError("semi_markov requires duration_hazards metadata.")
        rows = [np.asarray(hazard_map.get(state, []), dtype=float) for state in model.states]
        if any(row.ndim != 1 or not len(row) for row in rows):
            raise ValueError("Every state must expose a vector of duration_hazards.")
        lengths = keep([len(row) for row in rows], (states,), np.int32)
        width = int(lengths.max())
        hazards = keep(np.zeros((states, width)))
        for index, row in enumerate(rows):
            hazards[index, :len(row)] = row
        if not np.isfinite(hazards).all() or (hazards < 0).any() or (hazards > 1).any():
            raise ValueError("duration_hazards must contain finite probabilities.")
        if posterior is not None and start_state is None:
            if posterior.shape[1] > width:
                raise ValueError("Latest state-age posterior exceeds the duration hazard width.")
            age = keep(np.zeros((states, width)))
            age[:, :posterior.shape[1]] = posterior
            age /= age.sum()
    seed = int(np.random.SeedSequence().generate_state(1, dtype=np.uint64)[0]) if random_seed is None else int(random_seed)
    regime = _RegimeProcessConfig(
        _pointer(transition), _pointer(starts), _pointer(age), width if age is not None else 0,
        start_index, seed & ((1 << 64) - 1), int(duration_model == "semi_markov"),
        _pointer(hazards), _pointer(lengths), width, min_regime_duration, 0.0,
        None, None, 0, 0.0,
    )
    macro = None
    dimensions = 0
    if macro_dynamics is not None:
        dynamics = macro_dynamics
        dimensions = len(dynamics["columns"])
        if states != 4 or dimensions < 2:
            raise ValueError("joint_macro requires four states and at least two macro variables.")
        latest = keep(dynamics["latest"], (dimensions,))
        coefficient = keep(dynamics["var_coefficient"], (dimensions, dimensions))
        std = dynamics.get("var_coefficient_std")
        if std is None or np.shape(std) != coefficient.shape or not np.isfinite(std).all():
            std = np.zeros_like(coefficient)
        std = keep(std)
        centers = keep([dynamics["state_centers"][state] for state in model.states], (states, dimensions))
        covariance = keep([
            nearest_psd(np.asarray(dynamics["state_innovation_covariances"][state], dtype=float))
            for state in model.states
        ], (states, dimensions, dimensions))
        factor = keep(np.linalg.cholesky(covariance + np.eye(dimensions)[None] * 1e-10))
        emission = keep(emission_coefficients, (states, 6)) if emission_coefficients is not None else None
        rate_col = dynamics.get("rate_col")
        rate_bounds = dynamics.get("rate_bounds")
        rate_index = list(dynamics["columns"]).index(rate_col) if rate_col in dynamics["columns"] and rate_bounds is not None else -1
        bounds = tuple(map(float, rate_bounds)) if rate_index >= 0 else (0.0, 0.0)
        if len(bounds) != 2 or not np.isfinite(bounds).all() or bounds[0] > bounds[1]:
            raise ValueError("Invalid macro rate_bounds.")
        thresholds = keep(dynamics.get("thresholds", [0.0, 0.0]), (2,))
        scales = keep(np.maximum(dynamics.get("probability_scales", [1.0, 1.0]), 1e-9), (2,))
        if not np.isfinite(macro_transition_weight) or not 0 <= macro_transition_weight <= 1:
            raise ValueError("macro_transition_weight must be between 0 and 1.")
        macro = _MacroProcessConfig(
            dimensions, _pointer(latest), _pointer(coefficient), _pointer(std), int(macro_parameter_uncertainty),
            _pointer(centers), _pointer(factor), _pointer(emission), macro_transition_weight,
            rate_index, *bounds, -1, 1.0, 1.0, int(emission is None), *thresholds, *scales,
        )
    regimes = np.empty((periods, paths), dtype=np.uint8)
    values = np.empty((periods, paths, dimensions)) if macro is not None else None
    shocks = np.empty_like(values) if macro is not None else None
    function = library.mc_simulate_regime_macro_paths
    function.restype = ctypes.c_int
    function.argtypes = [ctypes.c_int] * 4 + [ctypes.POINTER(_RegimeProcessConfig), ctypes.POINTER(_MacroProcessConfig)] + [ctypes.c_void_p] * 3
    status = function(periods, paths, states, max(1, int(workers)), ctypes.byref(regime),
                      ctypes.byref(macro) if macro is not None else None,
                      _pointer(regimes), _pointer(values), _pointer(shocks))
    if status:
        raise RuntimeError(f"Native regime/macro simulation failed with status {status}.")
    return regimes, values, shocks
