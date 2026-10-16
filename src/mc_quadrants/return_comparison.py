"""Matched-moment return-law diagnostics, separate from portfolio path forecasts."""
from __future__ import annotations

import numpy as np

from mc_quadrants.types import ScenarioModel
from mc_quadrants.validation import _sample_mnts_observations


def build_return_comparison(model: ScenarioModel, samples: int = 16384) -> dict:
    """Compare calibrated MNTS marginals with a variance-matched t(5) reference.

    These are single-period, fixed-state log returns at calibrated volatility.
    They exclude pathwise GARCH, macro shocks, costs and parameter uncertainty.
    A dedicated random stream keeps production simulation draws unchanged.
    """
    degrees = 5
    rows = []
    for state_index, state in enumerate(model.states):
        moments = model.moments[state]
        seed = 73129 + state_index
        mnts = _sample_mnts_observations(moments, samples, seed)
        rng = np.random.default_rng(seed + 10000)
        volatility = np.sqrt(np.maximum(np.diag(moments.covariance), 0))
        student = (moments.mean.to_numpy() + volatility *
                   rng.standard_t(degrees, size=mnts.shape) * np.sqrt((degrees - 2) / degrees))
        for index, asset in enumerate(model.assets):
            values = [mnts[:, index] * 100, student[:, index] * 100]
            # A common central range makes shape legible; tails remain in CDF/quantiles.
            low = min(float(np.quantile(v, .001)) for v in values)
            high = max(float(np.quantile(v, .999)) for v in values)
            if high - low < 1e-8:
                low, high = low - .01, high + .01
            edges = np.linspace(low, high, 101)
            laws = {}
            for name, v in zip(("mnts", "student_t"), values):
                counts, _ = np.histogram(v, bins=edges)
                laws[name] = {
                    "density": (counts / samples / np.diff(edges)).tolist(),
                    "cdf": (np.searchsorted(np.sort(v), edges[1:], side="right") / samples * 100).tolist(),
                    "quantiles": np.quantile(v, [.01, .05, .5, .95, .99]).tolist(),
                    "loss_probability": float(np.mean(v < 0) * 100),
                    "outside_range": float(np.mean((v < low) | (v > high)) * 100),
                }
            rows.append({"asset": asset, "state": state,
                         "x": ((edges[:-1] + edges[1:]) / 2).tolist(),
                         "cdf_x": edges[1:].tolist(), **laws})
    return {"assets": model.assets, "states": model.states, "frequency": model.frequency,
            "samples": samples, "degrees_of_freedom": degrees, "rows": rows}
