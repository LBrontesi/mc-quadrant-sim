from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np

_LIBRARY: ctypes.CDLL | None | bool = None


def _load_library() -> ctypes.CDLL | None:
    global _LIBRARY
    if _LIBRARY is False:
        return None
    if isinstance(_LIBRARY, ctypes.CDLL):
        return _LIBRARY

    package_dir = Path(__file__).resolve().parent
    configured = os.getenv("MC_NATIVE_SIM_LIB")
    candidates = [Path(configured)] if configured else []
    candidates.extend(
        package_dir / name
        for name in ("_native_sim.so", "_native_sim.dylib", "_native_sim.dll")
    )
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            library = ctypes.CDLL(str(candidate))
            function = library.mc_simulate_parametric
            function.restype = ctypes.c_int
            function.argtypes = [
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_uint64,
                ctypes.c_int,
                ctypes.c_double,
                ctypes.c_int,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_int,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_void_p,
            ]
            _LIBRARY = library
            return library
        except (AttributeError, OSError):
            continue
    _LIBRARY = False
    return None


def native_available() -> bool:
    """Return whether the optional compiled simulation backend is loadable."""

    return _load_library() is not None


def _pointer(values: np.ndarray | None) -> ctypes.c_void_p:
    return ctypes.c_void_p(0 if values is None else values.ctypes.data)


def simulate_parametric_native(
    regime_codes: np.ndarray,
    means: np.ndarray,
    covariance_cholesky: np.ndarray,
    correlation_cholesky: np.ndarray,
    base_correlations: np.ndarray,
    volatilities: np.ndarray,
    random_seed: int,
    distribution: str,
    degrees_of_freedom: float,
    garch: bool,
    garch_alpha: float,
    garch_beta: float,
    dynamic_correlation: bool,
    dcc_alpha: float,
    dcc_beta: float,
    dcc_asymmetry: float,
    macro_shocks: np.ndarray | None = None,
    macro_betas: np.ndarray | None = None,
) -> np.ndarray | None:
    """Run the parametric return kernel in C++, or return ``None`` if unavailable."""

    if os.getenv("MC_DISABLE_NATIVE_SIM", "").strip().lower() in {"1", "true", "yes", "on"}:
        return None
    library = _load_library()
    if library is None:
        return None
    regime_codes = np.asarray(regime_codes)
    means = np.asarray(means)
    if regime_codes.ndim != 2 or means.ndim != 2:
        raise ValueError("regime_codes and means must be two-dimensional.")
    periods, paths = regime_codes.shape
    states, assets = means.shape
    if states > 256:
        raise ValueError("The native backend supports at most 256 regimes.")
    if regime_codes.size and (
        np.min(regime_codes) < 0 or np.max(regime_codes) >= states
    ):
        raise ValueError("regime_codes contains an invalid state index.")
    expected_matrix_shape = (states, assets, assets)
    expected_vector_shape = (states, assets)
    if np.shape(covariance_cholesky) != expected_matrix_shape:
        raise ValueError("covariance_cholesky has an invalid shape.")
    if np.shape(correlation_cholesky) != expected_matrix_shape:
        raise ValueError("correlation_cholesky has an invalid shape.")
    if np.shape(base_correlations) != expected_matrix_shape:
        raise ValueError("base_correlations has an invalid shape.")
    if np.shape(volatilities) != expected_vector_shape:
        raise ValueError("volatilities has an invalid shape.")
    if (macro_shocks is None) != (macro_betas is None):
        raise ValueError("macro_shocks and macro_betas must be supplied together.")
    if macro_shocks is not None:
        if np.ndim(macro_shocks) != 3 or np.shape(macro_shocks)[:2] != (periods, paths):
            raise ValueError("macro_shocks has an invalid shape.")
        if np.shape(macro_betas) != (np.shape(macro_shocks)[2], assets):
            raise ValueError("macro_betas has an invalid shape.")
    codes = np.ascontiguousarray(regime_codes, dtype=np.uint8)
    means = np.ascontiguousarray(means, dtype=np.float64)
    covariance_cholesky = np.ascontiguousarray(covariance_cholesky, dtype=np.float64)
    correlation_cholesky = np.ascontiguousarray(correlation_cholesky, dtype=np.float64)
    base_correlations = np.ascontiguousarray(base_correlations, dtype=np.float64)
    volatilities = np.ascontiguousarray(volatilities, dtype=np.float64)
    macro_shocks = (
        np.ascontiguousarray(macro_shocks, dtype=np.float64)
        if macro_shocks is not None
        else None
    )
    macro_betas = (
        np.ascontiguousarray(macro_betas, dtype=np.float64)
        if macro_betas is not None
        else None
    )
    macro_dimensions = 0 if macro_shocks is None else int(macro_shocks.shape[2])
    output = np.empty((periods, paths, assets), dtype=np.float64)
    status = library.mc_simulate_parametric(
        periods,
        paths,
        assets,
        states,
        macro_dimensions,
        _pointer(codes),
        _pointer(means),
        _pointer(covariance_cholesky),
        _pointer(correlation_cholesky),
        _pointer(base_correlations),
        _pointer(volatilities),
        _pointer(macro_shocks),
        _pointer(macro_betas),
        ctypes.c_uint64(int(random_seed) & ((1 << 64) - 1)),
        int(distribution == "student_t"),
        float(degrees_of_freedom),
        int(garch),
        float(garch_alpha),
        float(garch_beta),
        int(dynamic_correlation),
        float(dcc_alpha),
        float(dcc_beta),
        float(dcc_asymmetry),
        _pointer(output),
    )
    if status != 0:
        raise RuntimeError(f"Native simulation kernel failed with status {status}.")
    return output
