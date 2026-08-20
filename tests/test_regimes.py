import numpy as np
import pandas as pd
import pytest

from mc_quadrants.regimes import (
    REGIME_ORDER,
    Regime,
    classify_persistent_quadrants,
    classify_quadrants,
    estimate_duration_hazards,
    estimate_probabilistic_transition_matrix,
    estimate_transition_matrix,
    expected_duration_from_hazards,
    quadrant_probabilities,
    smooth_macro_for_regimes,
    sojourn_durations,
)


def test_persistent_classifier_smooths_one_month_macro_noise():
    macro = pd.DataFrame(
        {
            "growth": [1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0],
            "inflation": [-1.0] * 7,
        },
        index=pd.date_range("2020-01-31", periods=7, freq="ME"),
    )

    regimes = classify_persistent_quadrants(
        macro,
        growth_threshold=0.0,
        inflation_threshold=0.0,
        smoothing_window=3,
        hysteresis=0.0,
        confirmation_periods=2,
    )

    assert regimes.nunique() == 1
    assert regimes.iloc[-1] == Regime.HIGH_GROWTH_LOW_INFLATION.value


def test_macro_smoothing_promotes_integer_csv_columns_to_float():
    macro = pd.DataFrame({"growth": [2, 1, -1], "inflation": [1, 4, 4]})

    smoothed = smooth_macro_for_regimes(macro, smoothing_window=2)

    assert smoothed["growth"].dtype.kind == "f"
    assert smoothed["growth"].tolist() == pytest.approx([2.0, 1.5, 0.0])


def test_persistent_classifier_confirms_a_transition_before_switching():
    macro = pd.DataFrame(
        {"growth": [1.0, 1.0, -1.0, -1.0, -1.0], "inflation": [-1.0] * 5}
    )

    regimes = classify_persistent_quadrants(
        macro,
        growth_threshold=0.0,
        inflation_threshold=0.0,
        smoothing_window=1,
        hysteresis=0.0,
        confirmation_periods=2,
    )

    assert regimes.iloc[2] == Regime.HIGH_GROWTH_LOW_INFLATION.value
    assert regimes.iloc[3] == Regime.LOW_GROWTH_LOW_INFLATION.value


def test_persistent_classifier_hysteresis_prevents_boundary_chatter():
    macro = pd.DataFrame(
        {"growth": [1.0, 0.1, -0.1, 0.1, -0.1, 0.1], "inflation": [-1.0] * 6}
    )

    regimes = classify_persistent_quadrants(
        macro,
        growth_threshold=0.0,
        inflation_threshold=0.0,
        smoothing_window=1,
        hysteresis=0.5,
        confirmation_periods=1,
    )

    assert regimes.nunique() == 1


def test_persistent_classifier_is_prefix_invariant_with_causal_thresholds():
    rng = np.random.default_rng(19)
    macro = pd.DataFrame(
        {"growth": rng.normal(size=30), "inflation": rng.normal(size=30)},
        index=pd.date_range("2000-01-31", periods=30, freq="ME"),
    )
    kwargs = {
        "threshold_window": 4,
        "smoothing_window": 3,
        "hysteresis": 0.15,
        "confirmation_periods": 2,
    }

    prefix = classify_persistent_quadrants(macro.iloc[:20], **kwargs)
    full = classify_persistent_quadrants(macro, **kwargs).iloc[:20]

    pd.testing.assert_series_equal(prefix, full)


def test_duration_hazards_are_regularized_age_dependent_and_respect_floor():
    regimes = pd.Series(
        ["a"] * 2 + ["b"] * 3 + ["a"] * 5 + ["b"] * 4 + ["a"] * 8,
        index=pd.date_range("2000-01-31", periods=22, freq="ME"),
    )

    hazards = estimate_duration_hazards(regimes, states=["a", "b"], max_duration=24)

    assert set(hazards) == {"a", "b"}
    assert all(np.isfinite(values).all() for values in hazards.values())
    assert all(((values > 0) & (values < 1)).all() for values in hazards.values())
    assert not np.allclose(hazards["a"], hazards["a"][0])
    assert expected_duration_from_hazards(hazards["a"], min_duration=5) >= 5.0


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


def test_probabilistic_quadrants_are_normalized_and_causal():
    dates = pd.date_range("2010-01-31", periods=36, freq="ME")
    macro = pd.DataFrame(
        {
            "growth": np.linspace(-2.0, 3.0, len(dates)),
            "inflation": np.linspace(1.0, 5.0, len(dates)),
        },
        index=dates,
    )

    probabilities = quadrant_probabilities(
        macro,
        threshold_window=12,
        temperature=0.35,
    )
    classified = probabilities.dropna()

    assert len(classified) == 24
    assert np.allclose(classified.sum(axis=1), 1.0)
    assert ((classified > 0) & (classified < 1)).all().all()
    transition = estimate_probabilistic_transition_matrix(probabilities)
    assert list(transition.index) == REGIME_ORDER
    assert np.allclose(transition.sum(axis=1), 1.0)
