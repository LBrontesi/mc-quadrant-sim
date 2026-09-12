"""Constrained, variance-targeted QML for the simulation's pooled GARCH/ADCC.

Coefficients are pooled across assets/regimes (the native engine's parameter
contract). State levels and reset rules match simulation, not a textbook
constant-target ADCC implementation. Gaussian QML is not an MNTS likelihood.
"""
from __future__ import annotations

import numpy as np

DEFAULT_DEPENDENCE = dict(garch_alpha=.10, garch_beta=.85, dcc_alpha=.04, dcc_beta=.94, dcc_asymmetry=.01)


def _search(score, initial, steps, rounds=5):
    best = np.asarray(initial, dtype=float)
    value = float(score(best[None])[0])
    increments = np.asarray(steps, dtype=float)
    for _ in range(rounds):
        candidates = [best]
        for coordinate in range(len(best)):
            for direction in (-1, 1):
                candidate = best.copy()
                candidate[coordinate] += direction * increments[coordinate]
                if (candidate >= 0).all() and candidate.sum() < .995:
                    candidates.append(candidate)
        candidates = np.asarray(candidates)
        losses = score(candidates)
        selected = int(np.argmin(losses))
        if float(losses[selected]) < value:
            best, value = candidates[selected], float(losses[selected])
        else:
            increments *= .5
    return best, value


def fit_dependence(returns, regimes, moments, *, minimum_observations=60):
    """Fit common coefficients on chronological, regime-standardized residuals."""
    states = list(moments)
    values = returns.to_numpy(dtype=float)
    codes = np.array([states.index(state) for state in regimes], dtype=int)
    means = np.stack([moments[state].mean.to_numpy(dtype=float) for state in states])
    levels = np.stack([np.maximum(np.diag(moments[state].covariance), 1e-12) for state in states])
    residuals = values - means[codes]
    base = np.stack([moments[state].mnts.gaussian_correlation.to_numpy(dtype=float)
                     if moments[state].mnts is not None else moments[state].correlation.to_numpy(dtype=float)
                     for state in states])
    result = {**DEFAULT_DEPENDENCE, "method": "pooled_regime_reset_gaussian_qml",
              "observations": len(values), "status": "insufficient_history",
              "minimum_observations": minimum_observations}
    if len(values) < minimum_observations or values.shape[1] == 0:
        return result
    assets = values.shape[1]

    def garch_score(parameters, output=False):
        alpha, beta = parameters[:, 0, None], parameters[:, 1, None]
        h = np.broadcast_to(levels[codes[0]], (len(parameters), assets)).copy()
        loss = np.zeros(len(parameters))
        standardized = np.empty_like(values) if output else None
        for t, residual in enumerate(residuals):
            if t == 0 or codes[t] != codes[t-1]:
                h[:] = levels[codes[t]]
            h = np.maximum(h, 1e-12)
            loss += np.mean(np.log(h) + residual**2 / h, axis=1)
            if output:
                standardized[t] = residual / np.sqrt(h[0])
            h = (1-alpha-beta)*levels[codes[t]] + alpha*residual**2 + beta*h
        if output:
            return standardized, h[0]
        return loss / len(values)

    garch, garch_loss = min(
        (_search(garch_score, start, (.06, .12), 8) for start in ((.1, .85), (.03, .5), (.2, .65))),
        key=lambda pair: pair[1],
    )
    z, next_h = garch_score(garch[None], output=True)

    def dcc_score(parameters, output=False):
        a, b, g = (parameters[:, i, None, None] for i in range(3))
        q = np.broadcast_to(base[codes[0]], (len(parameters), assets, assets)).copy()
        loss = np.zeros(len(parameters))
        for t, residual in enumerate(z):
            if t == 0 or codes[t] != codes[t-1]:
                q[:] = base[codes[t]]
            else:
                negative = np.minimum(z[t-1], 0)
                q = (1-a-b-g)*base[codes[t]] + a*np.outer(z[t-1], z[t-1]) + b*q + g*np.outer(negative, negative)
            scale = np.sqrt(np.maximum(np.diagonal(q, axis1=1, axis2=2), 1e-12))
            correlation = q / (scale[:, :, None]*scale[:, None, :])
            correlation = correlation + np.eye(assets)*1e-10
            sign, determinant = np.linalg.slogdet(correlation)
            solved = np.linalg.solve(correlation, np.broadcast_to(residual, (len(parameters), assets))[..., None])[..., 0]
            loss += np.where(sign > 0, determinant + np.sum(residual*solved, axis=1), np.inf)
        if output:
            negative = np.minimum(z[-1], 0)
            next_q = (1-a-b-g)*base[codes[-1]] + a*np.outer(z[-1], z[-1]) + b*q + g*np.outer(negative, negative)
            scale = np.sqrt(np.maximum(np.diag(next_q[0]), 1e-12))
            return next_q[0] / np.outer(scale, scale)
        return loss / len(values)

    dcc, dcc_loss = min(
        (_search(dcc_score, start, (.025, .10, .025), 6) for start in ((.04, .94, .01), (.1, .65, .05))),
        key=lambda pair: pair[1],
    ) if assets > 1 else (np.zeros(3), 0.0)
    result.update(garch_alpha=float(garch[0]), garch_beta=float(garch[1]),
                  dcc_alpha=float(dcc[0]), dcc_beta=float(dcc[1]), dcc_asymmetry=float(dcc[2]),
                  status="fitted", garch_objective=garch_loss, dcc_objective=dcc_loss,
                  garch_default_objective=float(garch_score(np.array([[.1, .85]]))[0]),
                  dcc_default_objective=float(dcc_score(np.array([[.04, .94, .01]]))[0]) if assets > 1 else 0.0,
                  last_state=states[codes[-1]], next_variance=next_h.tolist(),
                  next_gaussian_correlation=dcc_score(dcc[None], output=True).tolist())
    return result
