import numpy as np
import pandas as pd

from mc_quadrants.regimes import Regime, classify_quadrants, estimate_transition_matrix


def test_classify_quadrants_maps_growth_and_inflation_states():
    macro = pd.DataFrame(
        {
            "growth": [2.0, 2.0, -1.0, -1.0],
            "inflation": [1.0, 4.0, 4.0, 1.0],
        }
    )

    regimes = classify_quadrants(
        macro,
        growth_threshold=0.0,
        inflation_threshold=3.0,
    )

    assert regimes.tolist() == [
        Regime.HIGH_GROWTH_LOW_INFLATION.value,
        Regime.HIGH_GROWTH_HIGH_INFLATION.value,
        Regime.LOW_GROWTH_HIGH_INFLATION.value,
        Regime.LOW_GROWTH_LOW_INFLATION.value,
    ]


def test_transition_matrix_rows_sum_to_one():
    regimes = pd.Series(
        [
            Regime.HIGH_GROWTH_LOW_INFLATION.value,
            Regime.HIGH_GROWTH_HIGH_INFLATION.value,
            Regime.LOW_GROWTH_HIGH_INFLATION.value,
            Regime.LOW_GROWTH_LOW_INFLATION.value,
            Regime.HIGH_GROWTH_LOW_INFLATION.value,
        ]
    )

    transition = estimate_transition_matrix(regimes, smoothing=0.5)

    assert np.allclose(transition.sum(axis=1).to_numpy(), 1.0)
