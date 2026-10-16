import itertools

import numpy as np
import pytest

pytest.importorskip('scipy')
from scripts.stochastic_volatility_research import (  # noqa: E402
    forward_filter,
    normal_mixture_logpdf,
    simulate_variance,
    transition_matrix,
)


def test_forward_filter_matches_brute_force_hidden_state_paths():
    transition = np.array([[.8, .2], [.3, .7]])
    prior = np.array([.4, .6])
    emission = np.array([[.1, .7], [.8, .2], [.3, .4]])
    total = 0.
    terminal = np.zeros(2)
    for states in itertools.product(range(2), repeat=3):
        p = prior[states[0]]*emission[0, states[0]]
        for t in range(1, 3):
            p *= transition[states[t-1], states[t]]*emission[t, states[t]]
        total += p
        terminal[states[-1]] += p
    likelihood, posterior, _ = forward_filter(np.log(emission), transition, prior, warmup=0)
    np.testing.assert_allclose(likelihood, np.log(total))
    np.testing.assert_allclose(posterior, terminal/total)
    first_ll, _, _ = forward_filter(np.log(emission[:1]), transition, prior, warmup=0)
    warm_ll, _, _ = forward_filter(np.log(emission), transition, prior, warmup=1)
    np.testing.assert_allclose(warm_ll, likelihood-first_ll)


def test_transition_is_normalized_and_zero_persistence_is_iid():
    grid = np.linspace(-7, 7, 81)
    matrix, prior = transition_matrix(.98, grid)
    np.testing.assert_allclose(matrix.sum(axis=1), 1.)
    assert matrix.min() >= 0
    np.testing.assert_allclose(prior.sum(), 1.)
    iid, prior = transition_matrix(0., grid)
    np.testing.assert_allclose(iid, np.broadcast_to(prior, iid.shape))


def test_mnts_conditional_normal_mixture_has_expected_moments():
    z = np.linspace(-12, 12, 12001)
    pdf = np.exp(normal_mixture_logpdf(z, np.array([.5, 1.5]), .4, .96))
    np.testing.assert_allclose(np.trapezoid(pdf, z), 1., atol=1e-9)
    np.testing.assert_allclose(np.trapezoid(z*pdf, z), 0., atol=1e-9)
    np.testing.assert_allclose(np.trapezoid(z*z*pdf, z), 1., atol=1e-8)


def test_forecast_integrates_state_uncertainty_and_lognormal_transition():
    fit = dict(parameters=[-.2, .8, .6], posterior_grid=[-1., .5], posterior_weights=[.3, .7])
    mu, phi, sd = fit['parameters']
    tau = sd*np.sqrt(1-phi*phi)
    expected = np.array(fit['posterior_weights']) @ np.exp(mu+phi*(np.array(fit['posterior_grid'])-mu)+.5*tau*tau)
    samples = simulate_variance([fit], 2, 200000, 761)
    np.testing.assert_allclose(samples[0].mean(), expected, rtol=.007)
    assert samples[0].std() > .1
    assert not np.array_equal(samples[0], samples[1])
    fixed = dict(parameters=[.2, .9, 0.], posterior_grid=[.2], posterior_weights=[1.])
    np.testing.assert_allclose(simulate_variance([fixed], 3, 10, 31), np.exp(.2))
