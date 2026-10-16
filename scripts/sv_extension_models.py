"""Research SV extensions. Explicit monthly approximations, not pricing models."""

from __future__ import annotations

import ctypes
import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.special import gammainc, ndtr

from scripts.stochastic_volatility_research import forward_filter, transition_matrix

_NATIVE = None


def native_filter():
    global _NATIVE
    if _NATIVE is None:
        source = Path(__file__).with_name("sv_dynamic_filter.cpp")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
        target = Path(tempfile.gettempdir()) / f"mc_sv_{digest}.so"
        if not target.exists():
            temporary = target.with_name(target.name + f".{__import__('os').getpid()}")
            subprocess.run(
                [
                    "c++",
                    "-O3",
                    "-std=c++17",
                    "-dynamiclib" if sys.platform == "darwin" else "-shared",
                    "-fPIC",
                    str(source),
                    "-o",
                    str(temporary),
                ],
                check=True,
                capture_output=True,
            )
            temporary.replace(target)
        _NATIVE = ctypes.CDLL(str(target)).dynamic_filter
        vector = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
        _NATIVE.argtypes = [ctypes.c_int] * 5 + [vector] * 8
        _NATIVE.restype = ctypes.c_double
    return _NATIVE


def normal_emissions(x, log_variance):
    return -0.5 * (np.log(2 * np.pi) + log_variance + np.asarray(x)[..., None] ** 2 * np.exp(-log_variance))


def select_optimum(loss, starts, bounds, maxiter=160):
    candidates = []
    for start in starts:
        fit = minimize(
            loss,
            start,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": maxiter, "ftol": 1e-9, "gtol": 2e-5, "maxls": 40},
        )
        if fit.success and np.isfinite(fit.fun):
            candidates.append(fit)
    if not candidates:
        raise ValueError("No converged optimization start")
    best = min(candidates, key=lambda f: f.fun)
    return best, len(candidates)


def truncated_normal_mean(mean, sd, floor=1e-8):
    a = (mean - floor) / sd
    return floor * ndtr(-a) + mean * ndtr(a) + sd * np.exp(-a * a / 2) / np.sqrt(2 * np.pi)


def dynamic_evaluate(kind, p, x, grid_points=81, quadrature=15):
    p = np.ascontiguousarray(p, dtype=float)
    x = np.ascontiguousarray(x, dtype=float)
    if kind == "leverage":
        grid = p[0] + p[2] * np.linspace(-7, 7, grid_points)
        edges = np.r_[-np.inf, (grid[:-1] + grid[1:]) / 2, np.inf]
        prior = np.diff(ndtr((edges - p[0]) / p[2]))
        code = 0
    else:
        grid = np.r_[np.log(1e-8), np.linspace(-7, 4, grid_points - 1)]
        edges = np.r_[0, np.exp((grid[:-1] + grid[1:]) / 2), np.inf]
        shape = 2 * p[0] * p[1] / p[2] ** 2
        scale = p[2] ** 2 / (2 * p[0])
        prior = np.diff(gammainc(shape, edges / scale))
        code = 1
    nodes, weights = np.polynomial.hermite.hermgauss(quadrature)
    nodes = np.ascontiguousarray(nodes * np.sqrt(2))
    weights = np.ascontiguousarray(weights / np.sqrt(np.pi))
    posterior = np.zeros(grid_points)
    edge = np.zeros(1)
    ll = native_filter()(
        code,
        len(x),
        grid_points,
        quadrature,
        12,
        p,
        x,
        np.ascontiguousarray(grid),
        np.ascontiguousarray(prior),
        nodes,
        weights,
        posterior,
        edge,
    )
    v = np.exp(grid)
    if kind == "leverage":
        tau = p[2] * np.sqrt(1 - p[1] ** 2)
        means = p[0] + p[1] * (grid - p[0]) + tau * p[3] * x[-1] / np.sqrt(v)
        sd = np.full(len(v), tau * np.sqrt(1 - p[3] ** 2))
        expected = np.exp(means + sd * sd / 2)
    else:
        means = v + p[0] * (p[1] - v) + p[2] * p[3] * x[-1]
        sd = p[2] * np.sqrt(v * (1 - p[3] ** 2))
        expected = truncated_normal_mean(means, sd)
    return ll, posterior, grid, means, sd, float(posterior @ expected), float(edge[0])


def fit_dynamic(kind, x, gaussian_fit=None, grid_points=81, quadrature=15, allow_leverage=True):
    level = float(np.clip(np.log(np.mean(x * x)), -1, 1))
    if kind == "leverage":
        base = np.asarray(gaussian_fit["parameters"][:3]) if gaussian_fit else np.array([level, 0.8, 0.5])
        base[2] = max(base[2], 0.04)
        starts = [[*base, 0.0], [*base, -0.4], [level, 0.6, 0.5, -0.2]]
        bounds = [(-2, 2), (0, 0.98), (0.02, 1.5), (-0.9, 0.9)]
        if not allow_leverage:
            starts = [[level, 0.5, 0.3, 0.0], [level, 0.9, 0.6, 0.0], [level, 0.97, 0.9, 0.0]]
            bounds[-1] = (0.0, 0.0)
    else:
        starts = [[0.2, 1.0, 0.3, -0.3], [0.6, 1.0, 0.5, 0.0], [0.1, 1.0, 0.2, -0.6]]
        bounds = [(0.02, 0.95), (0.15, 3.0), (0.02, 1.2), (-0.9, 0.9)]
    best, converged = select_optimum(
        lambda p: -dynamic_evaluate(kind, p, x, grid_points, quadrature)[0] / (len(x) - 12), starts, bounds
    )
    _, posterior, grid, means, sd, expected, edge = dynamic_evaluate(kind, best.x, x, grid_points, quadrature)
    return dict(
        kind=kind,
        parameters=best.x.tolist(),
        posterior_weights=posterior.tolist(),
        posterior_grid=grid.tolist(),
        next_means=means.tolist(),
        next_sd=sd.tolist(),
        next_variance_ratio=expected,
        objective=float(best.fun),
        optimizer_success=True,
        converged_starts=converged,
        maximum_posterior_edge_mass=edge,
        grid_points=grid_points,
        quadrature=quadrature,
    )


def simulate_dynamic(fits, gaussian, seed):
    rng = np.random.default_rng(seed)
    periods, paths, assets = gaussian.shape
    result = np.empty_like(gaussian)
    state = np.empty((paths, assets))
    kind = fits[0]["kind"]
    p = np.asarray([f["parameters"] for f in fits]).T
    for j, fit in enumerate(fits):
        index = rng.choice(len(fit["posterior_weights"]), paths, p=fit["posterior_weights"])
        state[:, j] = np.asarray(fit["next_means"])[index] + np.asarray(fit["next_sd"])[
            index
        ] * rng.standard_normal(paths)
    if kind == "heston":
        state = np.maximum(state, 1e-8)
    for t in range(periods):
        v = np.exp(state) if kind == "leverage" else state
        result[t] = v
        shock = p[3] * gaussian[t] + np.sqrt(1 - p[3] ** 2) * rng.standard_normal(state.shape)
        if kind == "leverage":
            tau = p[2] * np.sqrt(1 - p[1] ** 2)
            state = p[0] + p[1] * (state - p[0]) + tau * shock
        else:
            state = np.maximum(state + p[0] * (p[1] - state) + p[2] * np.sqrt(state) * shock, 1e-8)
    return result


def fit_realized(x, rv_ratio, macro_ratio, grid_points=81):
    g = np.linspace(-7, 7, grid_points)
    observed = np.log(rv_ratio)

    def evaluate(p):
        mu, phi, s, bias, noise = p
        h = mu + s * g
        emission = normal_emissions(x, h)
        rv_mean = np.log(np.exp(h)[None, :] + macro_ratio[:, None]) + bias
        emission += -0.5 * (np.log(2 * np.pi * noise**2) + (observed[:, None] - rv_mean) ** 2 / noise**2)
        trans, prior = transition_matrix(phi, g)
        return forward_filter(emission, trans, prior)

    starts = [[0, 0.8, 0.5, 0, 0.4], [0, 0.4, 0.25, 0, 0.7], [0, 0.95, 0.7, -0.1, 0.3]]
    best, count = select_optimum(
        lambda p: -evaluate(p)[0] / (len(x) - 12), starts, [(-2, 2), (0, 0.98), (0, 1.5), (-1, 1), (0.1, 1.5)]
    )
    _, post, edge = evaluate(best.x)
    mu, phi, s, bias, noise = best.x
    h = mu + s * g
    tau = s * np.sqrt(1 - phi * phi)
    expected = post @ np.exp(mu + phi * (h - mu) + tau * tau / 2)
    return dict(
        parameters=best.x[:3].tolist(),
        measurement_parameters=[float(bias), float(noise)],
        posterior_grid=h.tolist(),
        posterior_weights=post.tolist(),
        next_variance_ratio=float(expected),
        objective=float(best.fun),
        optimizer_success=True,
        converged_starts=count,
        maximum_posterior_edge_mass=edge,
    )


def fit_factor(x, grid_points=81):
    cov = np.cov(x[12:].T)
    eig, vec = np.linalg.eigh(cov)
    load = vec[:, -1] * np.sqrt(max(eig[-1] - eig[:-1].mean(), 0.01))
    if load[0] < 0:
        load = -load
    diagonal = np.maximum(np.diag(cov) - load * load, 0.05)

    def constant_loss(p):
        b = p[:4]
        d = np.exp(p[4:])
        inv = 1 / d
        den = 1 + np.sum(b * b * inv)
        quad = np.sum(x[12:] ** 2 * inv, axis=1) - (x[12:] @ (b * inv)) ** 2 / den
        return float(0.5 * (np.sum(np.log(d)) + np.log(den) + quad.mean() + 4 * np.log(2 * np.pi)))

    fit = minimize(
        constant_loss,
        np.r_[load, np.log(diagonal)],
        method="L-BFGS-B",
        bounds=[(0, 3)] + [(-3, 3)] * 3 + [(-5, 2)] * 4,
        options={"maxiter": 400, "ftol": 1e-11},
    )
    if not fit.success:
        raise ValueError("Factor covariance calibration failed")
    load = fit.x[:4]
    diagonal = np.exp(fit.x[4:])
    inv = 1 / diagonal
    a = np.sum(load * load * inv)
    b = (x @ (load * inv)) ** 2
    q = np.sum(x * x * inv, axis=1)
    grid = np.linspace(-7, 7, grid_points)

    def evaluate(p):
        phi, s = p
        h = -0.5 * s * s + s * grid
        v = np.exp(h)
        e = -0.5 * (
            4 * np.log(2 * np.pi)
            + np.log(diagonal).sum()
            + np.log1p(v * a)[None, :]
            + q[:, None]
            - b[:, None] * v / (1 + v * a)
        )
        trans, prior = transition_matrix(phi, grid)
        return forward_filter(e, trans, prior)

    best, count = select_optimum(
        lambda p: -evaluate(p)[0] / (len(x) - 12), [[0.5, 0.3], [0.9, 0.7], [0.97, 1]], [(0, 0.98), (0, 1.5)]
    )
    _, post, edge = evaluate(best.x)
    phi, s = best.x
    mu = -0.5 * s * s
    h = mu + s * grid
    tau = s * np.sqrt(1 - phi * phi)
    expected = float(post @ np.exp(mu + phi * (h - mu) + tau * tau / 2))
    return dict(
        parameters=[float(mu), float(phi), float(s)],
        posterior_grid=h.tolist(),
        posterior_weights=post.tolist(),
        loadings=load.tolist(),
        idiosyncratic_variance=diagonal.tolist(),
        next_variance_ratio=expected,
        marginal_next_variance=(diagonal + load * load * expected).tolist(),
        objective=float(best.fun),
        optimizer_success=True,
        converged_starts=count,
        maximum_posterior_edge_mass=edge,
    )


def fractional_covariance(t, s, hurst, scale):
    return (
        0.5
        * scale**2
        * (
            t[:, None] ** (2 * hurst)
            + s[None, :] ** (2 * hurst)
            - np.abs(t[:, None] - s[None, :]) ** (2 * hurst)
        )
    )


def fit_rough(x, horizon=12):
    # Gaussian log-square QML for a fractional-Brownian log-variance process.
    y = np.log(np.maximum(np.asarray(x) ** 2, 1e-12)) + 1.2703628454614782
    n = len(y)
    t = np.arange(1, n + 1, dtype=float)
    noise = np.pi**2 / 2

    def evaluate(p, output=False):
        mu, scale, hurst = p
        cov = fractional_covariance(t, t, hurst, scale) + np.eye(n) * noise
        chol = cho_factor(cov, lower=True, check_finite=False)
        residual = y - mu
        solved = cho_solve(chol, residual, check_finite=False)
        ll = 0.5 * (2 * np.log(np.diag(chol[0])).sum() + residual @ solved + n * np.log(2 * np.pi))
        initial = cov[:12, :12]
        c = cho_factor(initial, lower=True, check_finite=False)
        r = residual[:12]
        ll -= 0.5 * (
            2 * np.log(np.diag(c[0])).sum() + r @ cho_solve(c, r, check_finite=False) + 12 * np.log(2 * np.pi)
        )
        if not output:
            return float(ll / (n - 12))
        future = np.arange(n + 1, n + horizon + 1, dtype=float)
        cross = fractional_covariance(future, t, hurst, scale)
        mean = mu + cross @ solved
        covariance = fractional_covariance(future, future, hurst, scale) - cross @ cho_solve(
            chol, cross.T, check_finite=False
        )
        covariance = (covariance + covariance.T) / 2
        return mean, covariance

    level = float(np.clip(y.mean(), -1, 1))
    best, count = select_optimum(
        evaluate,
        [[level, 0.2, 0.1], [level, 0.1, 0.4], [level, 0.5, 0.2]],
        [(-2, 2), (0.005, 1.5), (0.03, 0.49)],
    )
    mean, cov = evaluate(best.x, True)
    return dict(
        parameters=best.x.tolist(),
        parameter_names=["mu", "fractional_scale", "hurst"],
        forecast_log_variance_mean=mean.tolist(),
        forecast_log_variance_covariance=cov.tolist(),
        next_variance_ratio=float(np.exp(mean[0] + cov[0, 0] / 2)),
        objective=float(best.fun),
        optimizer_success=True,
        converged_starts=count,
    )
