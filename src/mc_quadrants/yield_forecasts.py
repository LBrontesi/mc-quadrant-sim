"""Persistent yield forecasts selected with strictly preceding observations."""

from __future__ import annotations

import numpy as np


def fit_yield_forecasts(values, columns, yield_columns, valid, mode="adaptive"):
    """RW and damped-change forecasts; score the next change before updating.

    A yield level has no fixed historical attractor. EWMA drift is damped at
    0.5 per month; uncertainty lives in fitted joint macro innovations. Model
    weights use past squared errors with a 24-observation equal-weight prior.
    Bootstrap joins reset drift and contribute neither updates nor scores.
    """
    output, predictions = {}, {}
    for column in yield_columns:
        index = columns.index(column)
        series = values[:, index]
        differences = np.diff(series)
        drift = 0.0
        forecasts = np.zeros((len(differences), 2))
        errors = []
        for t, change in enumerate(differences):
            forecasts[t] = [series[t], series[t] + 0.5 * drift]
            if valid[t]:
                errors.append((np.array([0.0, 0.5 * drift]) - change) ** 2)
                drift = 0.8 * 0.5 * drift + 0.2 * change
            else:
                drift = 0.0
        losses = np.asarray(errors)
        if mode == "random_walk" or len(losses) < 24:
            weight = 0.0
        else:
            prior = max(float(np.mean(losses)), 1e-12)
            risk = (losses.sum(axis=0) + 24 * prior) / (len(losses) + 24)
            precision = 1 / np.maximum(risk, 1e-12)
            weight = float(precision[1] / precision.sum())
        predictions[column] = (1 - weight) * forecasts[:, 0] + weight * forecasts[:, 1]
        output[column] = dict(
            index=index,
            trend_weight=weight,
            latest_drift=float(drift),
            damping=0.5,
            update_weight=0.2,
            observations=len(losses),
            selection="prequential_squared_error_with_24_month_prior",
            mode=mode,
        )
    return output, predictions


def step_yields(current, predicted, drifts, specifications):
    """Replace yield conditional means, preserving their joint random shocks."""
    for column, specification in specifications.items():
        index = specification["index"]
        predicted[:, index] = (
            current[:, index] + specification["trend_weight"] * specification["damping"] * drifts[column]
        )
    return predicted
