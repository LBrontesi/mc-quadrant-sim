import numpy as np
import pytest

pytest.importorskip('scipy')
from scripts.volatility_research import figarch_filter, figarch_weights, fit_figarch  # noqa: E402


def test_figarch_weights_match_published_scalar_recursion():
    parameters = [.08, .2, .4, .3]
    actual = figarch_weights(parameters, 1000)
    expected = np.empty(1000)
    delta = parameters[2]
    expected[0] = delta-parameters[3]+parameters[1]
    for j in range(2, 1001):
        new_delta = (j-1-parameters[2])/j*delta
        expected[j-1] = parameters[3]*expected[j-2]+new_delta-parameters[1]*delta
        delta = new_delta
    np.testing.assert_allclose(actual, expected, rtol=1e-11)
    assert actual.min() > 0 and actual.sum() < 1


def test_figarch_filter_is_causal_and_d_zero_reduces_to_geometric_arch():
    params = [.1, .4, 0., .2]
    weights = figarch_weights(params, 1000)
    np.testing.assert_allclose(weights[:20], .2*.2**np.arange(20))
    x = np.array([-2., .5, 3.])
    h, next_h = figarch_filter(params, x)
    expected = [params[0]/.8+weights.sum()]
    for t in range(1, len(x)+1):
        expected.append(params[0]/.8+np.dot(weights[:t], x[:t][::-1]**2)+weights[t:].sum())
    np.testing.assert_allclose(np.r_[h, next_h], expected)
    changed = x.copy()
    changed[-1] = 8
    changed_h, changed_next = figarch_filter(params, changed)
    np.testing.assert_array_equal(h, changed_h)
    assert next_h != changed_next


def test_figarch_fit_reports_consistent_objective_and_next_variance():
    x = np.random.default_rng(193).normal(size=80)
    fit = fit_figarch(x)
    h, next_h = figarch_filter(fit['parameters'], x)
    np.testing.assert_allclose(fit['objective'], np.mean(np.log(h[12:])+x[12:]**2/h[12:]))
    np.testing.assert_allclose(fit['next_variance_ratio'], [next_h])
    assert fit['optimizer_success'] and fit['converged_starts'] > 0
    _, phi, d, beta = fit['parameters']
    assert 1-2*phi-d >= -1e-8 and phi+d-beta >= -1e-8
