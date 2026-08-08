import numpy as np
import pandas as pd
import pytest

from mc_quadrants.regimes import (
    Regime,
    classify_quadrants,
    estimate_transition_matrix,
    sojourn_durations,
)


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


def test_causal_thresholds_leave_early_rows_unclassified():
    macro = pd.DataFrame(
        {
            "growth": [1.0, 2.0, 3.0, 4.0, 5.0],
            "inflation": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )

    regimes = classify_quadrants(
        macro,
        growth_threshold="median",
        inflation_threshold="median",
        threshold_window=2,
    )

    assert pd.isna(regimes.iloc[0])
    assert pd.isna(regimes.iloc[1])
    assert regimes.iloc[2:].notna().all()


def test_causal_thresholds_use_only_prior_observations():
    macro = pd.DataFrame(
        {
            "growth": [10.0, 10.0, -10.0, -10.0],
            "inflation": [1.0, 1.0, 1.0, 1.0],
        }
    )

    regimes = classify_quadrants(
        macro,
        growth_threshold="median",
        inflation_threshold="median",
        threshold_window=1,
    )

    assert regimes.tolist()[2] == Regime.LOW_GROWTH_HIGH_INFLATION.value
    assert regimes.tolist()[3] == Regime.LOW_GROWTH_HIGH_INFLATION.value


def test_causal_thresholds_reject_invalid_windows():
    macro = pd.DataFrame({"growth": [1.0], "inflation": [1.0]})

    with pytest.raises(ValueError, match="threshold_window"):
        classify_quadrants(macro, threshold_window=0)


def test_sojourn_durations_record_run_lengths():
    series = pd.Series(["a", "a", "b", "b", "b", "a"])

    durations = sojourn_durations(series, ["a", "b"])

    assert durations["a"].tolist() == [2, 1]
    assert durations["b"].tolist() == [3]


def test_sojourn_durations_do_not_bridge_missing_periods():
    series = pd.Series(
        ["a", "a", "a"],
        index=pd.to_datetime(["2020-01-31", "2020-02-29", "2020-05-31"]),
    )

    durations = sojourn_durations(series, ["a"])

    assert durations["a"].tolist() == [2, 1]


def test_numeric_thresholds_must_be_finite():
    macro = pd.DataFrame({"growth": [1.0], "inflation": [2.0]})

    with pytest.raises(ValueError, match="finite"):
        classify_quadrants(macro, growth_threshold=np.nan, inflation_threshold=1.0)


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


def test_transition_matrix_sorts_timestamped_observations():
    ordered = pd.Series(
        [
            Regime.HIGH_GROWTH_LOW_INFLATION.value,
            Regime.HIGH_GROWTH_HIGH_INFLATION.value,
            Regime.LOW_GROWTH_HIGH_INFLATION.value,
            Regime.LOW_GROWTH_LOW_INFLATION.value,
        ],
        index=pd.date_range("2020-01-31", periods=4, freq="ME"),
    )
    shuffled = ordered.sample(frac=1.0, random_state=4)

    pd.testing.assert_frame_equal(
        estimate_transition_matrix(ordered, smoothing=0.5),
        estimate_transition_matrix(shuffled, smoothing=0.5),
    )


def test_transition_matrix_does_not_count_transitions_across_date_gaps():
    states = [Regime.HIGH_GROWTH_LOW_INFLATION.value, Regime.HIGH_GROWTH_HIGH_INFLATION.value]
    regimes = pd.Series(
        [states[0], states[1], states[0]],
        index=pd.to_datetime(["2020-01-31", "2020-02-29", "2020-05-31"]),
    )

    transition = estimate_transition_matrix(regimes, states=states, smoothing=0.5)

    assert transition.loc[states[0], states[1]] == pytest.approx(0.75)
    assert transition.loc[states[1], states[0]] == pytest.approx(0.5)
