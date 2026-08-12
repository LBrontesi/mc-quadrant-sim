import numpy as np
import pytest

from mc_quadrants.native import native_available, simulate_parametric_native

pytestmark = pytest.mark.skipif(not native_available(), reason="native simulator is not compiled")


def _inputs(periods: int = 8, paths: int = 20_000):
    regimes = np.zeros((periods, paths), dtype=np.uint8)
    means = np.array([[0.01, -0.005]])
    covariance = np.array([[[0.04, 0.018], [0.018, 0.09]]])
    volatility = np.sqrt(np.diagonal(covariance, axis1=1, axis2=2))
    correlation = covariance / (
        volatility[:, :, None] * volatility[:, None, :]
    )
    return {
        "regime_codes": regimes,
        "means": means,
        "covariance_cholesky": np.linalg.cholesky(covariance),
        "correlation_cholesky": np.linalg.cholesky(correlation),
        "base_correlations": correlation,
        "volatilities": volatility,
        "random_seed": 42,
        "distribution": "normal",
        "degrees_of_freedom": 5.0,
        "garch": False,
        "garch_alpha": 0.10,
        "garch_beta": 0.85,
        "dynamic_correlation": False,
        "dcc_alpha": 0.04,
        "dcc_beta": 0.94,
        "dcc_asymmetry": 0.01,
    }


def test_native_normal_is_reproducible_and_preserves_moments():
    inputs = _inputs()

    first = simulate_parametric_native(**inputs)
    second = simulate_parametric_native(**inputs)

    assert np.array_equal(first, second)
    assert np.allclose(first.mean(axis=(0, 1)), inputs["means"][0], atol=0.002)
    assert np.allclose(
        np.cov(first.reshape(-1, 2), rowvar=False),
        inputs["covariance_cholesky"][0] @ inputs["covariance_cholesky"][0].T,
        atol=0.002,
    )


@pytest.mark.parametrize(
    ("distribution", "garch", "dynamic_correlation"),
    [
        ("student_t", False, True),
        ("normal", True, False),
        ("normal", True, True),
    ],
)
def test_native_advanced_models_are_finite(distribution, garch, dynamic_correlation):
    inputs = _inputs(periods=48, paths=2_000)
    inputs.update(
        distribution=distribution,
        garch=garch,
        dynamic_correlation=dynamic_correlation,
    )

    result = simulate_parametric_native(**inputs)

    assert result.shape == (48, 2_000, 2)
    assert np.isfinite(result).all()


def test_native_joint_macro_effect_is_applied():
    inputs = _inputs(periods=4, paths=100)
    baseline = simulate_parametric_native(**inputs)
    shocks = np.ones((4, 100, 1))
    betas = np.array([[0.03, -0.02]])

    shifted = simulate_parametric_native(
        **inputs,
        macro_shocks=shocks,
        macro_betas=betas,
    )

    assert np.allclose(shifted - baseline, np.broadcast_to(betas, shifted.shape))
