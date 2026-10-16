import numpy as np

from scripts.compare_volatility_models import coordinate_fit, filter_variance, update, valid


def test_gjr_nests_garch_for_arbitrary_skew_moments():
    h = np.array([.6, 1.8, 1.2])
    x = np.array([-2., .2, 1.])
    kappa = np.array([.2, .5, .8])
    actual = update('gjr', [.12, .12, .7], h, x, kappa, .8)
    np.testing.assert_allclose(actual, update('garch', [.12, .7], h, x, kappa, .8))
    expected = .18 + .12*x*x + .7*h
    np.testing.assert_allclose(actual, expected)


def test_gjr_targets_variance_under_asymmetric_innovations():
    # Two-point, mean-zero/unit-variance distribution: P(-2)=.2, P(.5)=.8.
    x = np.array([-2., .5])
    next_h = update('gjr', [.1, .3, .6], np.ones(2), x, .8, .8)
    np.testing.assert_allclose(next_h @ [.2, .8], 1.)
    assert next_h[0] > next_h[1]


def test_egarch_uses_standardized_shocks_and_signed_asymmetry():
    h = np.array([4., 4.])
    x = np.array([-2., 2.])
    actual = update('egarch', [.01, .2, -.1, .7], h, x, .5, .75)
    np.testing.assert_allclose(actual, np.exp(.01+.7*np.log(4)+.2*.25+np.array([.1, -.1])))
    assert actual[0] > actual[1]
    assert not valid('egarch', np.array([[0, .4, -.2, .8]]), np.array([.5]), (.5, .5))[0]


def test_filter_is_causal_and_preserves_terminal_update():
    x = np.array([[-1.], [.5], [3.]])
    h, next_h = filter_variance('garch', [.2, .6], x, np.full_like(x, .5), np.full_like(x, .8))
    np.testing.assert_allclose(h[:, 0, 0], [1., 1., .85])
    np.testing.assert_allclose(next_h, [[2.51]])
    changed = x.copy()
    changed[-1] = -8
    alternative, other_next = filter_variance('garch', [.2, .6], changed, np.full_like(x, .5), np.full_like(x, .8))
    np.testing.assert_array_equal(h, alternative)
    assert other_next[0, 0] != next_h[0, 0]


def test_pooled_gjr_fit_cannot_lose_nested_garch_start_in_sample():
    rng = np.random.default_rng(31)
    x = rng.standard_normal((100, 4))
    kappa = np.full_like(x, .5)
    absolute = np.full_like(x, .8)
    garch = coordinate_fit('garch', x, kappa, absolute, (2, 2), rounds=12)
    alpha, beta = garch['parameters']
    gjr = coordinate_fit('gjr', x, kappa, absolute, (2, 2), [[alpha, alpha, beta]], rounds=12)
    assert gjr['objective'] <= garch['objective']+1e-12
    assert np.all(np.asarray(gjr['next_variance_ratio']) > 0)


def test_egarch_solver_returns_conventional_intercept_and_matching_filter():
    x = np.random.default_rng(97).standard_normal((75, 3))
    kappa, absolute = np.full_like(x, .5), np.full_like(x, .8)
    fit = coordinate_fit('egarch', x, kappa, absolute, (2., 2.), rounds=32)
    h, next_h = filter_variance('egarch', fit['parameters'], x, kappa, absolute)
    objective = np.mean(np.log(h[12:, 0])+x[12:]**2/h[12:, 0])
    np.testing.assert_allclose(fit['objective'], objective)
    np.testing.assert_allclose(fit['next_variance_ratio'], next_h[0])


def test_independent_fits_do_not_use_other_assets_residuals():
    from scripts.compare_volatility_models import fit_models

    x = np.random.default_rng(318).standard_normal((65, 2))
    codes = np.zeros(len(x), dtype=int)
    kappa, absolute = np.full((1, 2), .5), np.full((1, 2), .8)
    limits = np.full((2, 2), 2.)
    original = fit_models(x, codes, kappa, absolute, limits, independent=True, rounds=16)
    changed = x.copy()
    changed[:, 1] *= 3
    alternate = fit_models(changed, codes, kappa, absolute, limits, independent=True, rounds=16)
    for name in original:
        np.testing.assert_array_equal(np.asarray(original[name]['parameters'])[:, 0],
                                      np.asarray(alternate[name]['parameters'])[:, 0])
        assert original[name]['next_variance_ratio'][0] == alternate[name]['next_variance_ratio'][0]
        assert len(original[name]['asset_fits']) == 2


def test_asset_specific_qarch_and_garch_simulation_match_separate_recursions():
    from mc_quadrants.qarch import kernel, variance_ratio
    from scripts.compare_volatility_models import qarch_asset_variances

    history = np.random.default_rng(6).normal(size=(8, 2, 12))
    params = np.array([[.2, .5], [.1, 0], [0, .15]])
    actual = qarch_asset_variances(history, params)
    for j in range(2):
        np.testing.assert_allclose(actual[:, j], variance_ratio(history[:, j], kernel(params[:, j])))
    same = np.tile(params[:, :1], (1, 2))
    np.testing.assert_allclose(qarch_asset_variances(history, same), qarch_asset_variances(history, params[:, 0]))
    h = np.array([[.5, 2.], [3., .4]])
    x = np.array([[-1., .5], [.2, -.4]])
    garch_params = np.array([[.1, .2], [.7, .5]])
    together = update('garch', garch_params, h, x, .5, .8)
    for j in range(2):
        np.testing.assert_allclose(together[:, j], update('garch', garch_params[:, j], h[:, j], x[:, j], .5, .8))
