"""Richer monthly SV research models; no production-engine integration."""

from __future__ import annotations

import ctypes
import hashlib
import itertools
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln, ndtr, ndtri, stdtr, stdtrit

from scripts.stochastic_volatility_research import transition_matrix
from scripts.sv_extension_models import normal_emissions, select_optimum

_NATIVE = None


def native():
    global _NATIVE
    if _NATIVE is None:
        source = Path(__file__).with_name("complex_sv_native.cpp")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
        target = Path(tempfile.gettempdir()) / f"complex_sv_{digest}.so"
        if not target.exists():
            temporary = target.with_name(target.name + f".{os.getpid()}")
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
        _NATIVE = ctypes.CDLL(str(target))
        array = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
        _NATIVE.sv_filter.argtypes = [ctypes.c_int] * 4 + [array] * 6
        _NATIVE.sv_filter.restype = ctypes.c_double
        _NATIVE.lift_filter.argtypes = [ctypes.c_int] * 3 + [ctypes.c_ulonglong] + [array] * 6
        _NATIVE.lift_filter.restype = ctypes.c_double
        _NATIVE.lift_simulate.argtypes = [ctypes.c_int] * 4 + [ctypes.c_ulonglong] + [array] * 7
        _NATIVE.lift_simulate.restype = None
    return _NATIVE


def filtered(emission, transition, prior, kind=0, extra=None, groups=1):
    prior = np.ascontiguousarray(prior, dtype=float).reshape(-1)
    post = np.zeros_like(prior)
    edge = np.zeros(1)
    g = int(round(np.sqrt(len(prior)))) if kind == 1 else len(prior)
    if extra is None:
        extra = np.zeros(len(prior))
        for group in range(groups):
            extra[group * len(prior) // groups] = 1
            extra[(group + 1) * len(prior) // groups - 1] = 1
    ll = native().sv_filter(
        kind,
        len(emission),
        g,
        12,
        np.ascontiguousarray(emission),
        np.ascontiguousarray(transition),
        prior,
        np.ascontiguousarray(extra if extra is not None else np.zeros(1)),
        post,
        edge,
    )
    return ll, post, float(edge[0])


def skew_moments(nu, gamma):
    absolute = 2 * np.sqrt(nu) * np.exp(gammaln((nu + 1) / 2) - gammaln(nu / 2)) / ((nu - 1) * np.sqrt(np.pi))
    mean = absolute * (gamma - 1 / gamma)
    variance = nu / (nu - 2) * (gamma * gamma + gamma**-2 - 1) - mean * mean
    return mean, np.sqrt(variance)


def skew_density_cdf(z, nu, gamma):
    mean, sd = skew_moments(nu, gamma)
    y = np.asarray(z) * sd + mean
    u = np.where(y < 0, y * gamma, y / gamma)
    logpdf = (
        np.log(2 * sd / (gamma + 1 / gamma))
        + gammaln((nu + 1) / 2)
        - gammaln(nu / 2)
        - 0.5 * np.log(nu * np.pi)
        - (nu + 1) / 2 * np.log1p(u * u / nu)
    )
    cdf = np.where(
        y < 0,
        2 * stdtr(nu, u) / (1 + gamma * gamma),
        (1 + 2 * gamma * gamma * (stdtr(nu, u) - 0.5)) / (1 + gamma * gamma),
    )
    return logpdf, np.clip(cdf, 1e-12, 1 - 1e-12)


def skew_ppf(probability, nu, gamma):
    probability = np.clip(probability, 1e-12, 1 - 1e-12)
    cutoff = 1 / (1 + gamma * gamma)
    a = np.where(
        probability < cutoff,
        probability * (1 + gamma * gamma) / 2,
        0.5 + (probability * (1 + gamma * gamma) - 1) / (2 * gamma * gamma),
    )
    y = stdtrit(nu, a) * np.where(probability < cutoff, 1 / gamma, gamma)
    mean, sd = skew_moments(nu, gamma)
    return (y - mean) / sd


def fit_complex(kind, x, rv=None, macro=None, grid_points=41, component_grid=21):
    x = np.asarray(x)
    g = np.linspace(-6, 6, grid_points)
    level = float(np.clip(np.log(np.mean(x * x)), -1, 1))
    if kind == "realized_skewt":
        # Standardized Fernandez–Steel skew-t. A Gaussian copula links the
        # innovation to next-period log-volatility; this is not linear t leverage.
        def evaluate(p):
            mu, phi, s, rho, nu, gamma, bias, noise = p
            h = mu + s * g
            logpdf, probability = skew_density_cdf(x[:, None] * np.exp(-h / 2), nu, gamma)
            emission = logpdf - h / 2
            measurement = np.log(np.exp(h)[None, :] + macro[:, None]) + bias
            emission += -0.5 * (
                np.log(2 * np.pi * noise**2) + (np.log(rv)[:, None] - measurement) ** 2 / noise**2
            )
            _, prior = transition_matrix(phi, g)
            return filtered(emission, np.r_[g, phi, rho], prior, 2, ndtri(probability))

        starts = [[level, 0.8, 0.5, -0.3, 8, 0.8, 0, 0.5], [level, 0.4, 0.3, 0, 20, 1, 0, 0.8]]
        bounds = [(-2, 2), (0, 0.98), (0.02, 1.5), (-0.85, 0.85), (4.2, 40), (0.5, 2), (-1, 1), (0.1, 1.5)]
    elif kind == "two_component":
        g = np.linspace(-6, 6, component_grid)

        def evaluate(p):
            mu, fast, slow, sf, ss = p
            pf, prior = transition_matrix(fast, g)
            ps, _ = transition_matrix(slow, g)
            h = mu + sf * g[:, None] + ss * g[None, :]
            return filtered(
                normal_emissions(x, h.reshape(-1)),
                np.r_[pf.reshape(-1), ps.reshape(-1)],
                np.outer(prior, prior),
                1,
            )

        starts = [[level, 0.2, 0.9, 0.2, 0.4], [level - 0.2, 0.6, 0.97, 0.5, 0.7]]
        bounds = [(-2, 2), (0, 0.75), (0.8, 0.99), (0.02, 1.2), (0.02, 1.5)]
    elif kind == "switching":

        def evaluate(p):
            mu, gap, p0, p1, s0, s1, stay0, stay1 = p
            a, prior = transition_matrix(p0, g)
            b, _ = transition_matrix(p1, g)
            trans = np.block([[stay0 * a, (1 - stay0) * b], [(1 - stay1) * a, stay1 * b]])
            probability = (1 - stay1) / (2 - stay0 - stay1)
            h = np.r_[mu + s0 * g, mu + gap + s1 * g]
            return filtered(
                normal_emissions(x, h), trans, np.r_[probability * prior, (1 - probability) * prior], groups=2
            )

        starts = [
            [level - 0.3, 0.7, 0.5, 0.8, 0.2, 0.4, 0.9, 0.8],
            [level - 0.5, 1, 0.9, 0.9, 0.4, 0.5, 0.97, 0.9],
        ]
        bounds = [
            (-2, 1),
            (0.05, 2.5),
            (0, 0.98),
            (0, 0.98),
            (0.02, 1.2),
            (0.02, 1.5),
            (0.5, 0.995),
            (0.5, 0.995),
        ]
    elif kind == "joint_jumps":

        def evaluate(p):
            mu, phi, s, probability, jumpmean, jumpsd, voljump = p
            total_sd = np.sqrt(s * s + voljump * voljump * probability * (1 - probability) / (1 - phi * phi))
            h = mu + total_sd * g
            edges = np.r_[-np.inf, (h[:-1] + h[1:]) / 2, np.inf]
            trans = []
            for jump in [0, 1]:
                nextmean = mu + phi * (h - mu) + voljump * (jump - probability)
                matrix = np.diff(
                    ndtr((edges[None, :] - nextmean[:, None]) / (s * np.sqrt(1 - phi * phi))), axis=1
                )
                trans.append(np.column_stack([(1 - probability) * matrix, probability * matrix]))
            trans = np.vstack(trans)
            prior = np.diff(ndtr(np.r_[-np.inf, (g[:-1] + g[1:]) / 2, np.inf]))
            emission = []
            for jump in [0, 1]:
                variance = np.exp(h) + jump * jumpsd * jumpsd
                emission.append(
                    -0.5
                    * (
                        np.log(2 * np.pi * variance)
                        + (x[:, None] - jumpmean * (jump - probability)) ** 2 / variance
                    )
                )
            return filtered(
                np.column_stack(emission),
                trans,
                np.r_[(1 - probability) * prior, probability * prior],
                groups=2,
            )

        starts = [[level - 0.1, 0.8, 0.3, 0.04, -1, 0.8, 0.4], [level - 0.2, 0.5, 0.4, 0.12, -0.5, 1.5, 0.8]]
        bounds = [(-2, 2), (0, 0.98), (0.03, 1.5), (0.002, 0.2), (-3, 0), (0.1, 3), (0, 2)]
    else:
        raise ValueError(kind)
    best, count = select_optimum(lambda p: -evaluate(p)[0] / (len(x) - 12), starts, bounds, maxiter=240)
    _, post, edge = evaluate(best.x)
    fit = dict(
        kind=kind,
        parameters=best.x.tolist(),
        posterior_weights=post.tolist(),
        grid=g.tolist(),
        objective=float(best.fun),
        optimizer_success=True,
        converged_starts=count,
        maximum_posterior_edge_mass=edge,
        parameter_bounds=bounds,
    )
    fit["last_x"] = float(x[-1])
    fit["next_variance_ratio"] = initial_variance(fit)
    return fit


def initial_variance(fit):
    p, g, w = np.array(fit["parameters"]), np.array(fit["grid"]), np.array(fit["posterior_weights"])
    kind = fit["kind"]
    if kind == "realized_skewt":
        mu, phi, s, rho, nu, gamma = p[:6]
        h = mu + s * g
        eta = ndtri(skew_density_cdf(fit["last_x"] * np.exp(-h / 2), nu, gamma)[1])
        tau = s * np.sqrt(1 - phi * phi)
        return float(w @ np.exp(mu + phi * (h - mu) + tau * rho * eta + 0.5 * tau * tau * (1 - rho * rho)))
    if kind == "two_component":
        mu, fast, slow, sf, ss = p
        means = mu + sf * fast * g[:, None] + ss * slow * g[None, :]
        return float(
            w @ np.exp(means + 0.5 * (sf * sf * (1 - fast * fast) + ss * ss * (1 - slow * slow))).reshape(-1)
        )
    if kind == "switching":
        mu, gap, p0, p1, s0, s1, stay0, stay1 = p
        v0 = np.exp(mu + s0 * p0 * g + 0.5 * s0 * s0 * (1 - p0 * p0))
        v1 = np.exp(mu + gap + s1 * p1 * g + 0.5 * s1 * s1 * (1 - p1 * p1))
        return float(w @ np.r_[stay0 * v0 + (1 - stay0) * v1, (1 - stay1) * v0 + stay1 * v1])
    mu, phi, s, probability, jm, js, a = p
    total_sd = np.sqrt(s * s + a * a * probability * (1 - probability) / (1 - phi * phi))
    h = mu + total_sd * g
    means = np.r_[mu + phi * (h - mu) - a * probability, mu + phi * (h - mu) + a * (1 - probability)]
    return float(
        w @ np.exp(means + 0.5 * s * s * (1 - phi * phi))
        + probability * js * js
        + probability * (1 - probability) * jm * jm
    )


def simulate_complex(fits, gaussian, seed):
    rng = np.random.default_rng(seed)
    periods, paths, assets = gaussian.shape
    output = np.empty_like(gaussian)
    for j, fit in enumerate(fits):
        p, grid = np.array(fit["parameters"]), np.array(fit["grid"])
        size = len(grid)
        index = rng.choice(len(fit["posterior_weights"]), paths, p=fit["posterior_weights"])
        kind = fit["kind"]
        if kind == "two_component":
            gf, gs = grid[index // size], grid[index % size]
            mu, fast, slow, sf, ss = p
            for t in range(periods):
                gf = fast * gf + np.sqrt(1 - fast * fast) * rng.normal(size=paths)
                gs = slow * gs + np.sqrt(1 - slow * slow) * rng.normal(size=paths)
                output[t, :, j] = np.exp((mu + sf * gf + ss * gs) / 2) * gaussian[t, :, j]
        elif kind == "switching":
            mu, gap, p0, p1, s0, s1, stay0, stay1 = p
            state, g = index // size, grid[index % size]
            for t in range(periods):
                probability_high = np.where(state == 0, 1 - stay0, stay1)
                state = (rng.random(paths) < probability_high).astype(int)
                phi, sd = np.where(state == 0, p0, p1), np.where(state == 0, s0, s1)
                g = phi * g + np.sqrt(1 - phi * phi) * rng.normal(size=paths)
                output[t, :, j] = np.exp((mu + gap * state + sd * g) / 2) * gaussian[t, :, j]
        elif kind == "joint_jumps":
            mu, phi, s, probability, jm, js, a = p
            sd = np.sqrt(s * s + a * a * probability * (1 - probability) / (1 - phi * phi))
            h, last_jump = mu + sd * grid[index % size], index // size
            for t in range(periods):
                h = (
                    mu
                    + phi * (h - mu)
                    + s * np.sqrt(1 - phi * phi) * rng.normal(size=paths)
                    + a * (last_jump - probability)
                )
                last_jump = (rng.random(paths) < probability).astype(int)
                output[t, :, j] = (
                    np.exp(h / 2) * gaussian[t, :, j]
                    + jm * (last_jump - probability)
                    + js * last_jump * rng.normal(size=paths)
                )
        else:
            mu, phi, s, rho, nu, gamma = p[:6]
            h = mu + s * grid[index]
            eta = ndtri(skew_density_cdf(fit["last_x"] * np.exp(-h / 2), nu, gamma)[1])
            tau = s * np.sqrt(1 - phi * phi)
            for t in range(periods):
                h = mu + phi * (h - mu) + tau * (rho * eta + np.sqrt(1 - rho * rho) * rng.normal(size=paths))
                eta = gaussian[t, :, j]
                output[t, :, j] = np.exp(h / 2) * skew_ppf(ndtr(eta), nu, gamma)
    return output


def lift_kernel(hurst):
    rates = np.array([0.05, 0.5, 5.0])
    weights = rates ** (0.5 - hurst)
    weights /= np.sum(weights * (-np.expm1(-rates)) / rates)
    return rates, weights


def lift_evaluate(x, parameters, particles=256, steps=21, seed=471):
    p = np.ascontiguousarray(parameters[:4], dtype=float)
    rates, weights = lift_kernel(parameters[4])
    posterior = np.zeros((particles, 3))
    diagnostics = np.zeros(2)
    ll = native().lift_filter(
        len(x), particles, steps, seed, np.ascontiguousarray(x), p, rates, weights, posterior, diagnostics
    )
    return ll, posterior, diagnostics


def fit_lifted(x, particles=256, steps=21):
    # Training-only, predeclared finite parameter grid avoids differentiating a
    # discontinuous resampling likelihood. Re-evaluate three finalists at 4x particles.
    theta = float(np.clip(np.mean(np.asarray(x)[12:] ** 2), 0.15, 3))
    candidates = []
    for kappa, xi, rho, hurst in itertools.product([0.1, 0.5], [0.15, 0.35], [-0.7, 0.0], [0.1, 0.3, 0.49]):
        p = [theta, kappa, xi, rho, hurst]
        ll, _, diagnostic = lift_evaluate(x, p, particles, steps)
        candidates.append((ll, p, diagnostic.tolist()))
    candidates.sort(key=lambda a: -a[0])
    finalists = []
    for _, p, _ in candidates[:3]:
        ll, posterior, diagnostics = lift_evaluate(x, p, particles * 4, steps)
        finalists.append((ll, p, posterior, diagnostics))
    ll, p, posterior, diagnostics = max(finalists, key=lambda a: a[0])
    fit = dict(
        kind="lifted_heston",
        parameters=p,
        posterior=posterior.tolist(),
        objective=-float(ll) / (len(x) - 12),
        search_candidates=len(candidates),
        particles=particles * 4,
        steps=steps,
        training_floor_fraction=float(diagnostics[0]),
        minimum_effective_particle_fraction=float(diagnostics[1]),
    )
    _, integral, floors = simulate_lifted(fit, 32768, 1, 9127)
    fit["next_variance_ratio"] = float(integral.mean())
    fit["variance_monte_carlo_se"] = float(integral.std(ddof=1) / np.sqrt(integral.size))
    fit["forecast_floor_fraction"] = floors
    return fit


def simulate_lifted(fit, paths, periods, seed, steps=None):
    posterior = np.ascontiguousarray(fit["posterior"])
    p = np.ascontiguousarray(fit["parameters"][:4])
    rates, weights = lift_kernel(fit["parameters"][4])
    returns = np.zeros((periods, paths))
    integrals = np.zeros_like(returns)
    diagnostics = np.zeros(1)
    native().lift_simulate(
        len(posterior),
        paths,
        periods,
        steps or fit["steps"],
        seed,
        posterior,
        p,
        rates,
        weights,
        returns,
        integrals,
        diagnostics,
    )
    return returns, integrals, float(diagnostics[0])


def static_two_factor(x):
    # Identifiable lower-triangular loadings; Gaussian ML training covariance.
    covariance = np.cov(x[12:].T)
    eigen, vectors = np.linalg.eigh(covariance)
    load = vectors[:, -2:] * np.sqrt(np.maximum(eigen[-2:] - 0.15, 0.05))
    # Orthogonal rotation makes the first row's second loading zero.
    a, b = load[0]
    rotation = np.array([[a, -b], [b, a]]) / np.sqrt(a * a + b * b)
    load = load @ rotation
    if load[1, 1] < 0:
        load[:, 1] *= -1
    positions = [(0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1)]

    def unpack(p):
        b = np.zeros((4, 2))
        for value, position in zip(p[:7], positions):
            b[position] = value
        return b, np.exp(p[7:])

    def loss(p):
        b, d = unpack(p)
        cov = b @ b.T + np.diag(d)
        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0:
            return 1e10
        return 0.5 * (logdet + np.trace(np.linalg.solve(cov, x[12:].T @ x[12:] / len(x[12:]))))

    start = np.r_[
        [load[i, j] for i, j in positions], np.log(np.maximum(np.diag(covariance - load @ load.T), 0.05))
    ]
    bounds = [(0.001, 3), (-3, 3), (0.001, 3), (-3, 3), (-3, 3), (-3, 3), (-3, 3)] + [(-4, 1)] * 4
    result = minimize(loss, start, bounds=bounds, method="L-BFGS-B", options={"maxiter": 500, "ftol": 1e-11})
    if not result.success:
        raise ValueError("Two-factor covariance fit did not converge")
    return (*unpack(result.x), float(result.fun))


def split_rhat(values):
    # Basic split-Rhat on scalar chains. Also report predictions, not only parameters.
    values = np.asarray(values)
    half = values.shape[1] // 2
    split = np.concatenate([values[:, :half], values[:, -half:]], axis=0)
    within = np.var(split, axis=1, ddof=1).mean(axis=0)
    between = half * np.var(split.mean(axis=1), axis=0, ddof=1)
    return np.sqrt(((half - 1) / half * within + between / half) / np.maximum(within, 1e-15))


def factor_chains(x, load, diagonal, burn=1200, draws=200, thin=5):
    library = native()
    array = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    library.factor_mcmc.argtypes = (
        [ctypes.c_int] * 4 + [ctypes.c_ulonglong] + [array] * 3 + [ctypes.c_double] + [array] * 2
    )
    library.factor_mcmc.restype = None
    chains, diagnostics = [], []
    for seed in [791, 1571]:
        samples = np.zeros((draws, 6, 4))
        info = np.zeros(2)
        library.factor_mcmc(
            len(x),
            burn,
            draws,
            thin,
            seed,
            np.ascontiguousarray(x),
            np.ascontiguousarray(load),
            np.ascontiguousarray(diagonal),
            8.0,
            samples,
            info,
        )
        chains.append(samples)
        diagnostics.append(info.tolist())
    chains = np.array(chains)
    h, mu, phi, tau = [chains[..., k] for k in range(4)]
    predictive = np.exp(mu + phi * (h - mu) + 0.5 * tau * tau)
    rhats = split_rhat(chains)
    predictive_rhat = split_rhat(predictive)
    return chains, rhats, predictive_rhat, diagnostics


def fit_rich_factor(x, mcmc_scale=1):
    load, diagonal, objective = static_two_factor(x)
    burn, draws, thin = 1200 * mcmc_scale, 200 * mcmc_scale, 5
    chains, rhats, predictive_rhat, diagnostics = factor_chains(x, load, diagonal, burn, draws, thin)
    # Prespecified convergence-based extension; never uses held-out scores.
    extended = bool(max(rhats.max(), predictive_rhat.max()) > 1.1)
    if extended:
        burn, draws, thin = 2500 * mcmc_scale, 400 * mcmc_scale, 10
        chains, rhats, predictive_rhat, diagnostics = factor_chains(x, load, diagonal, burn, draws, thin)
    samples = chains.reshape(-1, 6, 4)
    h, mu, phi, tau = [samples[..., k] for k in range(4)]
    variance = np.exp(mu + phi * (h - mu) + tau * tau / 2).mean(axis=0)
    return dict(
        kind="rich_factor",
        loadings=load.tolist(),
        static_idiosyncratic_variance=diagonal.tolist(),
        posterior_draws=samples.tolist(),
        nu=8.0,
        objective=objective,
        static_optimizer_success=True,
        marginal_next_variance=(load * load @ variance[:2] + variance[2:]).tolist(),
        maximum_split_rhat=float(rhats.max()),
        maximum_predictive_split_rhat=float(predictive_rhat.max()),
        split_rhat=rhats.tolist(),
        predictive_split_rhat=predictive_rhat.tolist(),
        chain_diagnostics=diagnostics,
        mcmc_extended=extended,
        burn=burn,
        draws_per_chain=draws,
        thin=thin,
    )


def simulate_rich_factor(fit, paths, periods, seed, constant=False):
    rng = np.random.default_rng(seed)
    b = np.asarray(fit["loadings"])
    d = np.asarray(fit["static_idiosyncratic_variance"])
    posterior = np.asarray(fit["posterior_draws"])
    picked = posterior[rng.integers(len(posterior), size=paths)]
    h, mu, phi, tau = [picked[..., k] for k in range(4)]
    result = np.empty((periods, paths, 4))
    for t in range(periods):
        h = mu + phi * (h - mu) + tau * rng.normal(size=h.shape)
        variance = np.broadcast_to(np.r_[np.ones(2), d], h.shape) if constant else np.exp(h)
        shocks = rng.standard_t(fit["nu"], size=h.shape) * np.sqrt((fit["nu"] - 2) / fit["nu"])
        components = np.sqrt(variance) * shocks
        result[t] = components[:, :2] @ b.T + components[:, 2:]
    return result
