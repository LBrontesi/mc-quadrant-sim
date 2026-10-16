from types import SimpleNamespace

import numpy as np
import pandas as pd

from scripts.compare_qarch_forecasts import first_variance, score_forecast


def test_analytic_forecast_variance_includes_macro_and_regime_uncertainty():
    model = SimpleNamespace(
        states=["a", "b"],
        moments={"a": SimpleNamespace(mean=pd.Series([.01, .02])),
                 "b": SimpleNamespace(mean=pd.Series([-.01, .04]))},
        metadata={
            "hsmm_latest_state_age_probabilities": {"a": [.25], "b": [.75]},
            "macro_dynamics": {
                "return_betas": [[.2, .3]],
                "return_residual_covariances": {"a": [[.01, 0], [0, .02]],
                                                "b": [[.03, 0], [0, .04]]},
                "state_innovation_covariances": {"a": [[.1]], "b": [[.2]]},
            },
            "dependence_fit": {"status": "fitted", "last_state": "b", "next_variance": [.08, .09],
                               "qarch": {"next_variance_ratio": [2, 3]}},
        },
    )
    mean, garch, filtered = first_variance(model, "garch")
    _, qarch, _ = first_variance(model, "qarch")
    np.testing.assert_allclose(mean, [-.005, .035])
    between = np.array([.000075, .000075])
    macro = .175 * np.square([.2, .3])
    np.testing.assert_allclose(garch, [.025, .035] + macro + between)
    np.testing.assert_allclose(qarch, np.array([.025, .035]) * [2, 3] + macro + between)
    np.testing.assert_allclose(filtered, [.0625, .0725] + macro + between)


def test_forecast_scoring_respects_horizon_and_actual_return_compounding():
    samples = np.zeros((12, 8, 4))
    samples[:, :, :] = np.linspace(-.02, .03, 8)[None, :, None]
    actual = np.full((3, 4), np.log1p(.01))
    scores = score_forecast(samples, actual, np.zeros(4), ["SPY", "IEF", "GLD", "DBC"])
    assert scores["actual_1m"] == .01
    assert "crps_3m" in scores
    assert "crps_12m" not in scores
    changed = actual.copy()
    changed[1:] = -1
    future_changed = score_forecast(samples, changed, np.zeros(4), ["SPY", "IEF", "GLD", "DBC"])
    assert scores["crps_1m"] == future_changed["crps_1m"]
    assert scores["qlike_assets"] == future_changed["qlike_assets"]
    assert scores["crps_3m"] != future_changed["crps_3m"]
