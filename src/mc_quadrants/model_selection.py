"""Chronological hyperparameter selection with an untouched final holdout."""
from __future__ import annotations

import numpy as np

from mc_quadrants.validation import walk_forward_validation


def select_calibration_hyperparameters(returns, macro, *, validation_kwargs, candidates=None,
                                     holdout_periods=12, validation_runner=walk_forward_validation):
    """Tune on development walk-forward scores, never on the final holdout.

    Default search is small: mean shrinkage and HSMM duration-prior strength.
    Returned scores are selection scores, not unbiased test performance.
    """
    clean = returns.sort_index().replace([np.inf, -np.inf], np.nan).dropna()
    if holdout_periods < 1 or len(clean) < 60 + holdout_periods + 12:
        raise ValueError("Hyperparameter selection requires at least 84 complete observations with a 12-month holdout.")
    training = clean.iloc[:-holdout_periods]
    cutoff = training.index[-1]
    development_macro = macro.loc[macro.index <= cutoff]
    if candidates is None:
        strengths = list(dict.fromkeys([float(validation_kwargs.get("mean_prior_strength", 24)), 12.0, 48.0]))
        candidates = [{"mean_prior_strength": strength} for strength in strengths]
        duration_prior = float(validation_kwargs.get("duration_prior_strength", 8.0))
        candidates += [
            {"mean_prior_strength": strengths[0], "duration_prior_strength": strength}
            for strength in (4.0, 16.0) if strength != duration_prior
        ]
    allowed = {"mean_prior_strength", "threshold_window", "regime_temperature", "duration_prior_strength"}
    if not candidates or len(candidates) > 12:
        raise ValueError("Supply between 1 and 12 calibration candidates.")
    records = []
    for index, parameters in enumerate(candidates):
        if set(parameters)-allowed:
            raise ValueError("Unsupported calibration hyperparameter candidate.")
        kwargs = {**validation_kwargs, **parameters, "max_splits": 4}
        kwargs.pop("evaluation_start", None)
        result = validation_runner(training, development_macro, **kwargs)
        score = float(result.summary["regime_energy_score_mean"])
        if not np.isfinite(score):
            raise ValueError("Hyperparameter candidate returned a non-finite validation score.")
        records.append({"candidate": index, "parameters": dict(parameters), "energy_score": score,
                        "splits": int(result.summary["splits"])})
    best = min(records, key=lambda record: record["energy_score"])
    return {"selected_parameters": best["parameters"], "candidates": records,
            "selection_metric": "development_walk_forward_energy_score",
            "development_end": str(cutoff), "holdout_start": str(clean.index[-holdout_periods]),
            "holdout_periods": holdout_periods, "selection_scores_are_test_scores": False}
