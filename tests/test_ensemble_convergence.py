import numpy as np
import pytest

from mc_quadrants.ensemble_convergence import _mixture_quantiles, assess_parameter_convergence


def test_weighted_quantiles_preserve_model_weights_and_point_masses():
    groups = [np.zeros(10), np.full(1000, 10.)]
    weights = np.array([[.75, .25], [.25, .75], [0., 1.]])
    np.testing.assert_allclose(_mixture_quantiles(groups, weights, .5), [0, 10, 10], atol=1e-12)
    np.testing.assert_allclose(_mixture_quantiles(groups, weights, .95), [10, 10, 10], atol=1e-12)


def test_many_paths_do_not_disguise_too_few_models():
    groups = [np.full(10000, value) for value in (1., 1.1, 2., 3.)]
    result = assess_parameter_convergence({"wealth": groups}, {"wealth": (.5, .02, True)})
    assert result["status"] == "insufficient_models"
    assert not result["metrics"]["wealth"]["within_tolerance"]
    assert result == assess_parameter_convergence({"wealth": groups}, {"wealth": (.5, .02, True)})


def test_repeating_paths_does_not_reduce_outer_error():
    groups = [np.linspace(0, 1, 128) + value for value in np.linspace(0, 1, 32)]
    first = assess_parameter_convergence({"x": groups}, {"x": (.95, .02, False)})
    repeated = assess_parameter_convergence({"x": [np.tile(g, 20) for g in groups]}, {"x": (.95, .02, False)})
    assert first["status"] == repeated["status"] == "unstable"
    assert first["metrics"] == repeated["metrics"]


def test_identical_conditional_distributions_have_no_outer_error_but_require_enough_paths():
    result = assess_parameter_convergence({"x": [np.ones(128)] * 32}, {"x": (.5, .02, True)})
    assert result["status"] == "within_tolerance"
    assert result["metrics"]["x"]["max_mc_deviation"] == 0
    result = assess_parameter_convergence({"x": [np.ones(10)] * 32}, {"x": (.5, .02, True)})
    assert result["status"] == "insufficient_paths"


@pytest.mark.parametrize("groups", [[], [np.array([])], [np.array([np.nan])], [np.ones((2, 2))]])
def test_invalid_samples_are_rejected(groups):
    with pytest.raises(ValueError):
        assess_parameter_convergence({"x": groups}, {"x": (.5, .02, True)})
