import numpy as np
import pandas as pd
import pytest

pytest.importorskip("scipy")
from scipy.special import ndtr  # noqa: E402

from scripts.prepare_sv_realized_data import monthly_realized_variance  # noqa: E402
from scripts.stochastic_volatility_research import forward_filter  # noqa: E402
from scripts.sv_extension_models import (  # noqa: E402
    dynamic_evaluate,
    fit_factor,
    fit_rough,
    fractional_covariance,
    normal_emissions,
    simulate_dynamic,
    truncated_normal_mean,
)


def test_native_leverage_filter_matches_independent_python_quadrature():
    x = np.random.default_rng(71).normal(size=20)
    p = np.array([-0.1, 0.8, 0.5, -0.4])
    g = 31
    k = 9
    ll, post, grid, means, sd, expected, edge = dynamic_evaluate("leverage", p, x, g, k)
    edges = np.r_[-np.inf, (grid[:-1] + grid[1:]) / 2, np.inf]
    mass = np.diff(ndtr((edges - p[0]) / p[2]))
    nodes, weights = np.polynomial.hermite.hermgauss(k)
    nodes *= np.sqrt(2)
    weights /= np.sqrt(np.pi)
    tau = p[2] * np.sqrt(1 - p[1] ** 2)
    score = 0.0
    for t in range(len(x)):
        loge = normal_emissions(x[t], grid)
        top = loge.max()
        mass *= np.exp(loge - top)
        total = mass.sum()
        mass /= total
        if t >= 12:
            score += np.log(total) + top
        if t == len(x) - 1:
            break
        target = (
            p[0]
            + p[1] * (grid[:, None] - p[0])
            + tau * p[3] * x[t] / np.exp(grid[:, None] / 2)
            + tau * np.sqrt(1 - p[3] ** 2) * nodes
        )
        next_mass = np.zeros(g)
        for i in range(g):
            for q in range(k):
                location = target[i, q]
                w = mass[i] * weights[q]
                right = np.searchsorted(grid, location)
                if right == 0:
                    next_mass[0] += w
                elif right == g:
                    next_mass[-1] += w
                else:
                    f = (location - grid[right - 1]) / (grid[right] - grid[right - 1])
                    next_mass[right] += w * f
                    next_mass[right - 1] += w * (1 - f)
        mass = next_mass
    np.testing.assert_allclose(ll, score, atol=1e-11)
    np.testing.assert_allclose(post, mass, atol=1e-12)
    np.testing.assert_allclose(expected, post @ np.exp(means + sd * sd / 2))
    assert edge < 0.01


def test_zero_leverage_has_observation_independent_transition():
    x = np.random.default_rng(32).normal(size=20)
    p = np.array([0.1, 0.7, 0.4, 0.0])
    g = 41
    k = 11
    ll, post, grid, *_ = dynamic_evaluate("leverage", p, x, g, k)
    edges = np.r_[-np.inf, (grid[:-1] + grid[1:]) / 2, np.inf]
    prior = np.diff(ndtr((edges - p[0]) / p[2]))
    nodes, w = np.polynomial.hermite.hermgauss(k)
    nodes *= np.sqrt(2)
    w /= np.sqrt(np.pi)
    trans = np.zeros((g, g))
    tau = p[2] * np.sqrt(1 - p[1] ** 2)
    for i in range(g):
        targets = p[0] + p[1] * (grid[i] - p[0]) + tau * nodes
        for target, weight in zip(targets, w):
            trans[i] += weight * np.array([np.interp(target, grid, np.eye(g)[j]) for j in range(g)])
    check, posterior, _ = forward_filter(normal_emissions(x, grid), trans, prior)
    np.testing.assert_allclose([ll], [check], atol=1e-10)
    np.testing.assert_allclose(post, posterior, atol=1e-12)


def test_dynamic_forecast_matches_integrated_first_variance_and_leverage_sign():
    x = np.random.default_rng(18).normal(size=30)
    for kind, p in [("leverage", [-0.1, 0.8, 0.5, -0.5]), ("heston", [0.3, 1.0, 0.4, -0.5])]:
        _, post, grid, means, sd, expected, _ = dynamic_evaluate(kind, p, x, 61, 15)
        fit = dict(
            kind=kind,
            parameters=p,
            posterior_weights=post.tolist(),
            next_means=means.tolist(),
            next_sd=sd.tolist(),
        )
        shocks = np.full((2, 120000, 1), 2.0)
        positive = simulate_dynamic([fit], shocks, 61)
        negative = simulate_dynamic([fit], -shocks, 61)
        np.testing.assert_allclose(positive[0].mean(), expected, rtol=0.012)
        np.testing.assert_array_equal(positive[0], negative[0])
        assert negative[1].mean() > positive[1].mean()
        assert positive.min() > 0 and negative.min() > 0


def test_full_truncation_mean_and_fractional_covariance():
    rng = np.random.default_rng(32)
    draws = np.maximum(-0.2 + 0.7 * rng.standard_normal(300000), 1e-8)
    np.testing.assert_allclose(draws.mean(), truncated_normal_mean(np.array(-0.2), np.array(0.7)), rtol=0.008)
    t = np.arange(1, 20, dtype=float)
    np.testing.assert_allclose(
        fractional_covariance(t, t, 0.5, 0.3), 0.09 * np.minimum(t[:, None], t[None, :])
    )
    rough = fractional_covariance(t, t, 0.1, 0.3)
    assert np.linalg.eigvalsh(rough).min() > 0
    assert rough[1, 0] - rough[0, 0] < 0


def test_factor_and_rough_fits_have_valid_predictive_covariances():
    rng = np.random.default_rng(9)
    x = rng.normal(size=(75, 4)) * 0.7 + rng.normal(size=(75, 1)) * np.array([0.8, 0.3, -0.4, 0.6])
    fit = fit_factor(x, 41)
    b = np.asarray(fit["loadings"])
    d = np.asarray(fit["idiosyncratic_variance"])
    assert np.linalg.eigvalsh(np.diag(d) + np.outer(b, b)).min() > 0
    np.testing.assert_allclose(fit["marginal_next_variance"], d + b * b * fit["next_variance_ratio"])
    rough = fit_rough(x[:, 0])
    covariance = np.asarray(rough["forecast_log_variance_covariance"])
    assert np.linalg.eigvalsh(covariance).min() > -1e-10
    np.testing.assert_allclose(
        rough["next_variance_ratio"], np.exp(rough["forecast_log_variance_mean"][0] + covariance[0, 0] / 2)
    )


def test_daily_realized_proxy_uses_only_its_month_and_rejects_short_months():
    dates = pd.bdate_range("2020-01-01", "2020-03-06")
    returns = np.random.default_rng(24).normal(0.001, 0.015, len(dates))
    prices = pd.DataFrame({"ETF": 100 * np.exp(np.cumsum(returns))}, index=dates)
    result = monthly_realized_variance(prices)
    january = returns[1:][dates[1:].month == 1]
    np.testing.assert_allclose(result.iloc[0, 0], len(january) * np.var(january, ddof=1))
    changed = prices.copy()
    changed.loc[changed.index.month > 1, "ETF"] *= np.exp(np.arange(sum(changed.index.month > 1)) * 0.2)
    np.testing.assert_allclose(monthly_realized_variance(changed).iloc[0], result.iloc[0])
    assert np.isnan(result.iloc[-1, 0])
