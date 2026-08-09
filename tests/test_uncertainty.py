import numpy as np
import pandas as pd

from mc_quadrants.uncertainty import (
    bootstrap_quadrant_models,
    stationary_bootstrap_indices,
    summarize_parameter_models,
)


def test_stationary_bootstrap_preserves_local_runs():
    rng = np.random.default_rng(8)
    indexes = stationary_bootstrap_indices(120, 12, rng)

    assert indexes.shape == (120,)
    assert indexes.min() >= 0
    assert indexes.max() < 120
    continued = indexes[1:] == (indexes[:-1] + 1) % 120
    assert continued.mean() > 0.75


def test_parameter_bootstrap_recalibrates_complete_models():
    rng = np.random.default_rng(4)
    dates = pd.date_range("2000-01-31", periods=120, freq="ME")
    macro = pd.DataFrame(
        {
            "growth": np.tile([2.0, 2.0, -2.0, -2.0], 30),
            "inflation": np.tile([1.0, 4.0, 4.0, 1.0], 30),
        },
        index=dates,
    )
    returns = pd.DataFrame(
        rng.normal(0.005, 0.03, size=(len(dates), 2)),
        index=dates,
        columns=["SPY", "IEF"],
    )

    models = bootstrap_quadrant_models(
        returns,
        macro,
        draws=3,
        block_size=8,
        random_seed=9,
        growth_threshold=0.0,
        inflation_threshold=3.0,
        probabilistic_regimes=True,
        mean_prior_strength=24.0,
    )
    summary = summarize_parameter_models(models, {"SPY": 0.6, "IEF": 0.4})

    assert len(models) == 3
    assert summary.shape == (3, 4)
    assert np.isfinite(summary.to_numpy(dtype=float)).all()
    assert all(model.metadata["regime_assignment"] == "probabilistic" for model in models)
