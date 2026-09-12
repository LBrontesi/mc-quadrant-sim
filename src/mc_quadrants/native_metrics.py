"""Thin array bridges for C++ risk and inflation calculations."""
from __future__ import annotations

import ctypes
import os

import numpy as np


def _library():
    # Lazy import avoids a cycle through the decumulation plan declarations.
    from mc_quadrants.native import _load_library
    if os.getenv("MC_DISABLE_NATIVE_SIM", "").strip().lower() in {"1", "true", "yes", "on"}:
        return None
    return _load_library()


def money_weighted_returns_native(terminal, periods, initial, contribution):
    library = _library()
    if library is None:
        return None
    from mc_quadrants.native import _pointer
    values = np.ascontiguousarray(terminal, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("terminal must be a non-empty vector of finite non-negative wealth.")
    if periods <= 0 or not np.isfinite(initial) or initial <= 0 or not np.isfinite(contribution) or contribution < 0:
        raise ValueError("Invalid money-weighted return configuration.")
    output = np.empty_like(values)
    function = library.mc_money_weighted_returns
    function.restype = ctypes.c_int
    function.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_void_p, ctypes.c_void_p]
    status = function(periods, len(values), initial, contribution, _pointer(values), _pointer(output))
    if status:
        raise RuntimeError(f"Native money-weighted return calculation failed with status {status}.")
    return output


def risk_path_statistics_native(wealth, contributions, withdrawals, risk_free, initial):
    library = _library()
    if library is None:
        return None
    from mc_quadrants.native import _pointer
    values = np.ascontiguousarray(wealth, dtype=float)
    if values.ndim != 2 or min(values.shape) <= 0:
        raise ValueError("wealth must be a non-empty matrix.")
    periods, paths = values.shape
    arrays = []
    for value in (contributions, withdrawals, risk_free):
        array = np.asarray(value, dtype=float)
        if array.ndim == 0:
            array = np.full((periods, 1), float(array))
        if array.shape not in {(periods, 1), (periods, paths)}:
            raise ValueError("Risk calculation arrays must match the wealth periods/paths.")
        arrays.append(np.ascontiguousarray(array))
    output = np.empty((9, paths))
    function = library.mc_risk_path_statistics
    function.restype = ctypes.c_int
    function.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_void_p] + [ctypes.c_void_p, ctypes.c_int] * 3 + [ctypes.c_void_p]
    arguments = []
    for array in arrays:
        arguments.extend([_pointer(array), array.shape[1]])
    status = function(periods, paths, initial, _pointer(values), *arguments, _pointer(output))
    if status:
        raise RuntimeError(f"Native risk calculation failed with status {status}.")
    return output


def inflation_index_native(periods, paths, annual_rate, rates=None, *, frequency=12.0, inverse=False):
    library = _library()
    if library is None:
        return None
    from mc_quadrants.native import _pointer
    if periods <= 0 or paths <= 0 or not np.isfinite(frequency) or frequency <= 0:
        raise ValueError("Inflation dimensions and frequency must be positive.")
    if not np.isfinite(annual_rate) or annual_rate <= -1:
        raise ValueError("annual_inflation must be finite and above -100%.")
    rate_values = None
    if rates is not None:
        rate_values = np.ascontiguousarray(rates, dtype=float)
        if rate_values.shape != (periods, paths) or not np.isfinite(rate_values).all() or (rate_values <= -1).any():
            raise ValueError("inflation_paths must contain finite annual rates above -100% and match the requested shape.")
    output = np.empty((periods, paths))
    function = library.mc_inflation_index
    function.restype = ctypes.c_int
    function.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_void_p, ctypes.c_int, ctypes.c_double, ctypes.c_int, ctypes.c_void_p]
    status = function(periods, paths, frequency, _pointer(rate_values), paths, annual_rate, int(inverse), _pointer(output))
    if status:
        raise RuntimeError(f"Native inflation calculation failed with status {status}.")
    return output
