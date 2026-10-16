import numpy as np
import pytest

pytest.importorskip("scipy")
from scipy.integrate import quad  # noqa: E402

from scripts.complex_sv_research import (  # noqa: E402
    factor_chains,
    filtered,
    initial_variance,
    lift_evaluate,
    simulate_complex,
    simulate_lifted,
    simulate_rich_factor,
    skew_density_cdf,
    skew_ppf,
    split_rhat,
)
from scripts.stochastic_volatility_research import forward_filter, transition_matrix  # noqa: E402


def test_standardized_skew_t_density_moments_and_inverse():
    for nu, skew in [(5, 0.65), (10, 1), (20, 1.6)]:
        for power, target in [(0, 1), (1, 0), (2, 1)]:
            value = quad(
                lambda z: z**power * np.exp(skew_density_cdf(z, nu, skew)[0]), -np.inf, np.inf, epsabs=1e-7
            )[0]
            np.testing.assert_allclose(value, target, atol=2e-6)
        p = np.linspace(0.001, 0.999, 101)
        np.testing.assert_allclose(skew_density_cdf(skew_ppf(p, nu, skew), nu, skew)[1], p, atol=1e-10)


def test_native_separable_filter_equals_full_kronecker_filter():
    g = np.linspace(-4, 4, 9)
    a, prior = transition_matrix(0.4, g)
    b, _ = transition_matrix(0.8, g)
    emission = np.random.default_rng(31).normal(size=(30, len(g) ** 2))
    ll, posterior, _ = filtered(emission, np.r_[a.ravel(), b.ravel()], np.outer(prior, prior), 1)
    expected, reference, _ = forward_filter(emission, np.kron(a, b), np.outer(prior, prior).ravel())
    np.testing.assert_allclose(ll, expected, atol=1e-11)
    np.testing.assert_allclose(posterior, reference, atol=1e-12)


def test_observation_dependent_filter_zero_leverage_matches_static_filter():
    g = np.linspace(-5, 5, 31)
    a, prior = transition_matrix(0.8, g)
    emission = np.random.default_rng(8).normal(size=(35, len(g)))
    ll, posterior, _ = filtered(emission, np.r_[g, 0.8, 0], prior, 2, np.zeros_like(emission))
    expected, reference, _ = forward_filter(emission, a, prior)
    np.testing.assert_allclose(ll, expected, atol=1e-10)
    np.testing.assert_allclose(posterior, reference, atol=1e-12)


def test_first_variance_matches_forecast_samples_all_four_models():
    rng = np.random.default_rng(361)
    gaussian = rng.normal(size=(1, 250000, 1))
    specifications = [
        ("realized_skewt", [0.1, 0.7, 0.4, -0.4, 10, 0.8, 0, 0.5]),
        ("two_component", [0.1, 0.3, 0.9, 0.3, 0.5]),
        ("switching", [-0.2, 0.7, 0.4, 0.8, 0.3, 0.6, 0.9, 0.8]),
        ("joint_jumps", [-0.2, 0.8, 0.3, 0.1, -1, 0.7, 0.5]),
    ]
    g = np.linspace(-1, 1, 5)
    for kind, p in specifications:
        count = (
            len(g) ** 2
            if kind == "two_component"
            else len(g) * (2 if kind in ["switching", "joint_jumps"] else 1)
        )
        weights = rng.dirichlet(np.ones(count) * 3)
        fit = dict(kind=kind, parameters=p, grid=g, posterior_weights=weights, last_x=-0.5)
        samples = simulate_complex([fit], gaussian, 95)[0, :, 0]
        expected = initial_variance(fit)
        np.testing.assert_allclose(samples.var(), expected, rtol=0.025, err_msg=kind)
        assert abs(samples.mean()) < 0.015 * np.sqrt(expected)


def test_lifted_zero_vol_of_vol_is_constant_variance_and_martingale():
    x = np.random.default_rng(12).normal(size=30)
    p = [1.2, 0.2, 0.0, -0.6, 0.2]
    ll, posterior, diagnostics = lift_evaluate(x, p, particles=256, steps=10)
    assert np.isfinite(ll) and diagnostics[0] == 0
    fit = dict(parameters=p, posterior=posterior, steps=10)
    returns, variance, floors = simulate_lifted(fit, 100000, 2, 71)
    np.testing.assert_allclose(variance, 1.2, atol=1e-12)
    np.testing.assert_allclose(returns.var(axis=1), [1.2, 1.2], rtol=0.02)
    np.testing.assert_allclose(returns.mean(axis=1), 0.0, atol=0.012)
    assert floors == 0
    p[3] = 0
    exact, _, info = lift_evaluate(x, p, particles=64, steps=10)
    expected = (-0.5 * (np.log(2 * np.pi * 1.2) + x[12:] ** 2 / 1.2)).sum()
    np.testing.assert_allclose(exact, expected, atol=1e-11)
    np.testing.assert_allclose(info, [0, 1], atol=1e-12)


def test_factor_sampler_and_predictive_moments():
    rng = np.random.default_rng(19)
    b = np.array([[0.8, 0], [0.2, 0.7], [-0.2, 0.3], [0.5, 0.2]])
    d = np.array([0.3, 0.4, 0.5, 0.4])
    x = rng.normal(size=(45, 2)) @ b.T + rng.normal(size=(45, 4)) * np.sqrt(d)
    chains, rhat, predictive_rhat, diagnostics = factor_chains(x, b, d, burn=30, draws=40, thin=2)
    assert chains.shape == (2, 40, 6, 4) and np.isfinite(chains).all()
    assert all(info[1] == 0 for info in diagnostics)
    posterior = chains.reshape(-1, 6, 4)
    h, mu, phi, tau = [posterior[..., k] for k in range(4)]
    v = np.exp(mu + phi * (h - mu) + tau * tau / 2).mean(axis=0)
    expected = b * b @ v[:2] + v[2:]
    fit = dict(loadings=b, static_idiosyncratic_variance=d, posterior_draws=posterior, nu=8)
    samples = simulate_rich_factor(fit, 180000, 1, 91)[0]
    np.testing.assert_allclose(samples.var(axis=0), expected, rtol=0.035)
    assert np.all(np.isfinite(rhat)) and np.all(np.isfinite(predictive_rhat))
    independent = rng.normal(size=(4, 1000, 3))
    assert split_rhat(independent).max() < 1.02
    independent[0] += 2
    assert split_rhat(independent).min() > 1.2
