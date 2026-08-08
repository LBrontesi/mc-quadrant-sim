import numpy as np
import pytest

from mc_quadrants.matrix import nearest_psd_higham


def test_higham_projection_preserves_diagonal():
    matrix = np.array([[1.0, 2.0, 2.0], [2.0, 1.0, 2.0], [2.0, 2.0, 1.0]])

    projected = nearest_psd_higham(matrix)

    assert np.allclose(np.diag(projected), 1.0)
    eigenvalues = np.linalg.eigvalsh((projected + projected.T) / 2.0)
    assert (eigenvalues >= -1e-6).all()
    assert np.allclose(projected, np.ones((3, 3)), atol=1e-6)


def test_higham_projection_rejects_negative_diagonal():
    with pytest.raises(ValueError, match="diagonal"):
        nearest_psd_higham(np.array([[1.0, 0.0], [0.0, -1.0]]))


def test_higham_projection_is_idempotent_on_psd_input():
    matrix = np.array([[2.0, 0.5], [0.5, 1.0]])

    projected = nearest_psd_higham(matrix)

    assert np.allclose(projected, matrix, atol=1e-8)


def test_higham_projection_is_frobenius_nearest():
    matrix = np.array([[1.0, 2.0], [2.0, 1.0]])
    nearest_psd = matrix.copy()
    nearest_psd[0, 1] = nearest_psd[1, 0] = 1.0

    projected = nearest_psd_higham(matrix)

    assert np.allclose(projected, nearest_psd, atol=1e-6)


def test_higham_projection_handles_singular_matrices():
    matrix = np.zeros((3, 3))

    projected = nearest_psd_higham(matrix)

    assert np.allclose(projected, 0.0)


def test_higham_projection_rejects_empty_input():
    with pytest.raises(ValueError):
        nearest_psd_higham(np.empty((0, 0)))
