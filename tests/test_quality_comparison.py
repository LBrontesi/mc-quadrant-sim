import numpy as np

from scripts.compare_model_quality import paired_ci, scalar_score, vector_score


def test_scalar_score_matches_pairwise_definition():
    values = np.array([-1.0, 0.0, 2.0, 4.0])
    expected = np.abs(values - 0.5).mean() - 0.5 * np.abs(values[:, None] - values).mean()
    assert np.isclose(scalar_score(values, 0.5), expected)


def test_vector_score_matches_pairwise_definition_and_ignores_order():
    values = np.random.default_rng(5).normal(size=(140, 4))
    actual = np.ones(4)
    expected = (
        np.linalg.norm(values - actual, axis=1).mean()
        - 0.5 * np.linalg.norm(values[:, None] - values, axis=2).mean()
    )
    assert np.isclose(vector_score(values, actual), expected)
    assert np.isclose(vector_score(values[::-1], actual), expected)


def test_block_ci_preserves_constant_paired_differences():
    np.testing.assert_allclose(paired_ci(np.full(72, 0.02)), [0.02, 0.02])
