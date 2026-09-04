import numpy as np
import pandas as pd

from mc_quadrants.hsmm import fit_quadrant_hsmm
from mc_quadrants.regimes import REGIME_ORDER


def _synthetic_quadrants() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(42)
    centers = {
        REGIME_ORDER[0]: np.array([2.0, 1.0]),
        REGIME_ORDER[1]: np.array([2.0, 4.0]),
        REGIME_ORDER[2]: np.array([-1.0, 4.0]),
        REGIME_ORDER[3]: np.array([-1.0, 1.0]),
    }
    labels = [state for _ in range(3) for state in REGIME_ORDER for _ in range(12)]
    values = np.vstack([centers[state] + rng.normal(0.0, 0.15, 2) for state in labels])
    index = pd.date_range("2000-01-31", periods=len(labels), freq="ME")
    macro = pd.DataFrame(values, index=index, columns=["growth", "inflation"])
    return macro, pd.Series(labels, index=index, dtype="string")


def test_hsmm_estimates_normalized_latent_probabilities_and_explicit_durations():
    macro, labels = _synthetic_quadrants()

    result = fit_quadrant_hsmm(
        macro,
        labels,
        min_duration=5,
        max_duration=36,
        max_iterations=20,
    )

    assert np.allclose(result.filtered_probabilities.sum(axis=1), 1.0)
    assert np.allclose(result.smoothed_probabilities.sum(axis=1), 1.0)
    assert result.viterbi_path.notna().all()
    assert np.allclose(result.transition_matrix.sum(axis=1), 1.0)
    assert np.allclose(result.exit_transition_matrix.sum(axis=1), 1.0)
    assert np.allclose(np.diag(result.exit_transition_matrix), 0.0)
    assert all(np.allclose(hazard[:4], 0.0) for hazard in result.duration_hazards.values())
    assert all(duration >= 5.0 for duration in result.expected_duration_months.values())
    assert np.isfinite(result.log_likelihood)
    assert 1 <= result.iterations <= 20


def test_hsmm_viterbi_path_recovers_persistent_quadrant_structure():
    macro, labels = _synthetic_quadrants()

    result = fit_quadrant_hsmm(
        macro,
        labels,
        min_duration=5,
        max_duration=36,
        max_iterations=20,
    )

    accuracy = float((result.viterbi_path == labels).mean())
    assert accuracy > 0.95
    changes = result.viterbi_path.ne(result.viterbi_path.shift()).fillna(True)
    starts = np.flatnonzero(changes.to_numpy(dtype=bool))
    lengths = np.diff(np.append(starts, len(result.viterbi_path)))
    assert lengths.min() >= 5


def test_hsmm_does_not_bridge_disconnected_macro_sequences():
    macro, labels = _synthetic_quadrants()
    macro = macro.iloc[:48].copy()
    labels = labels.iloc[:48].copy()
    shifted_index = macro.index.to_list()
    shifted_index[24:] = [date + pd.DateOffset(months=6) for date in shifted_index[24:]]
    macro.index = pd.DatetimeIndex(shifted_index)
    labels.index = macro.index

    result = fit_quadrant_hsmm(
        macro,
        labels,
        min_duration=5,
        max_duration=24,
        max_iterations=5,
    )

    assert result.filtered_probabilities.notna().all().all()
    assert result.viterbi_path.notna().all()


def test_hsmm_updates_emissions_without_losing_quadrant_semantics():
    rng = np.random.default_rng(9)
    centers = {
        REGIME_ORDER[0]: np.array([1.5, 1.5]),
        REGIME_ORDER[1]: np.array([1.5, 3.5]),
        REGIME_ORDER[2]: np.array([-0.5, 3.5]),
        REGIME_ORDER[3]: np.array([-0.5, 1.5]),
    }
    labels = [state for _ in range(2) for state in REGIME_ORDER for _ in range(18)]
    values = np.vstack(
        [centers[state] + rng.multivariate_normal([0.0, 0.0], [[0.5, 0.2], [0.2, 0.5]]) for state in labels]
    )
    index = pd.date_range("2000-01-31", periods=len(labels), freq="ME")
    macro = pd.DataFrame(values, index=index, columns=["growth", "inflation"])
    truth = pd.Series(labels, index=index, dtype="string")
    noisy = truth.copy()
    noisy.iloc[8::11] = noisy.shift(18).iloc[8::11].fillna(REGIME_ORDER[3])

    fixed = fit_quadrant_hsmm(
        macro,
        noisy,
        min_duration=5,
        max_duration=36,
        max_iterations=20,
        update_emissions=False,
    )
    updated = fit_quadrant_hsmm(
        macro,
        noisy,
        min_duration=5,
        max_duration=36,
        max_iterations=20,
        update_emissions=True,
    )

    fixed_accuracy = float((fixed.viterbi_path == truth).mean())
    updated_accuracy = float((updated.viterbi_path == truth).mean())
    assert updated_accuracy >= fixed_accuracy
    means = updated.emission_means
    assert means[REGIME_ORDER[0]][0] > means[REGIME_ORDER[3]][0]
    assert means[REGIME_ORDER[1]][0] > means[REGIME_ORDER[2]][0]
    assert means[REGIME_ORDER[1]][1] > means[REGIME_ORDER[0]][1]
    assert means[REGIME_ORDER[2]][1] > means[REGIME_ORDER[3]][1]
