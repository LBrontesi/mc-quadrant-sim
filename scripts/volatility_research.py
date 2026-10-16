"""Research optimizers for asset-specific EGARCH and truncated FIGARCH."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.signal import lfilter

FIGARCH_LAGS = 1000


def figarch_weights(parameters, lags=FIGARCH_LAGS):
    """ARCH-infinity weights, following arch's FIGARCH(1,d,1) recurrence."""
    _, phi, d, beta = parameters
    delta = d*np.cumprod(np.r_[1., (np.arange(1, lags)-d)/np.arange(2, lags+1)])
    return lfilter([1., -phi], [1., -beta], delta)+(phi-beta)*beta**np.arange(lags)


def figarch_filter(parameters, x, lags=FIGARCH_LAGS):
    """Pre-observation and next variance; unavailable squared residuals = 1."""
    weights = figarch_weights(parameters, lags)
    intercept = parameters[0]/(1-parameters[3])
    # x^2-1 zero-pads the fixed presample baseline, never future observations.
    centered = np.r_[0., np.asarray(x).reshape(-1)**2-1]
    h = intercept+weights.sum()+np.convolve(centered, weights)[:len(centered)]
    return h[:-1], h[-1]


def fit_figarch(x, lags=FIGARCH_LAGS):
    x = np.asarray(x).reshape(-1)
    upper = 10*max(float(np.mean(x*x)), 1.)
    bounds = [(1e-8, upper), (0, .5), (0, .999999), (0, .999999)]
    constraints = {'type': 'ineq', 'fun': lambda p: np.array([1-2*p[1]-p[2], p[1]+p[2]-p[3]])}

    def score(parameters):
        h, _ = figarch_filter(parameters, x, lags)
        if not np.isfinite(h).all() or h.min() <= 1e-10:
            return 1e10
        return float(np.mean(np.log(h[12:])+x[12:]**2/h[12:]))

    starts = []
    for d in (.1, .4, .8):
        for ratio in (.2, .7):
            phi = (1-d)*ratio/2
            for br in (.2, .8):
                beta = (d+phi)*br
                params = np.array([1., phi, d, beta])
                params[0] = max(1e-5, (1-beta)*(1-figarch_weights(params, lags).sum())*np.mean(x*x))
                starts.append(params)
    # Six diverse deterministic starts, ranked using training loss only.
    starts = sorted(starts, key=score)[:6]
    candidates = []
    for start in starts:
        fit = minimize(score, start, method='SLSQP', bounds=bounds, constraints=constraints,
                       options={'ftol': 1e-10, 'maxiter': 500})
        if fit.success and constraints['fun'](fit.x).min() >= -1e-8 and np.isfinite(fit.fun):
            candidates.append(fit)
    if not candidates:
        raise ValueError('FIGARCH optimizer did not converge from any start')
    selected = min(candidates, key=lambda fit: fit.fun)
    _, next_h = figarch_filter(selected.x, x, lags)
    return dict(parameters=selected.x.tolist(), objective=float(selected.fun), next_variance_ratio=[float(next_h)],
                search_step=0., optimizer='SLSQP', optimizer_success=True, optimizer_iterations=int(selected.nit),
                truncation=lags, weight_sum=float(figarch_weights(selected.x, lags).sum()),
                converged_starts=len(candidates), attempted_starts=len(starts))


def refine_egarch(fit, x, kappa, absolute, limits, filter_variance):
    """Refine a difficult coordinate-search fit with constrained SLSQP."""
    def score(parameters):
        h, _ = filter_variance('egarch', parameters, x, kappa, absolute)
        h = h[:, 0]
        if not np.isfinite(h).all() or h.min() <= 1e-10:
            return 1e10
        return float(np.mean(np.log(h[12:])+x[12:]**2/h[12:]))

    constraints = {'type': 'ineq', 'fun': lambda p: np.array([
        .95*limits[0]-p[1]-p[2]-1e-9, .95*limits[1]-p[1]+p[2]-1e-9])}
    candidates = []
    for start in [fit['parameters'], [0., .1, -.1, .5], [.05, .2, 0., .1]]:
        result = minimize(score, start, method='SLSQP', bounds=[(-.5, .5), (0, 1), (-.75, .75), (0, .979999)],
                          constraints=constraints, options={'ftol': 1e-11, 'maxiter': 1500})
        if result.success and constraints['fun'](result.x).min() >= -1e-8:
            candidates.append(result)
    if not candidates:
        raise ValueError('EGARCH SLSQP refinement failed')
    selected = min(candidates, key=lambda result: result.fun)
    if selected.fun > fit['objective']+1e-8:
        raise ValueError('EGARCH refinement lost the original training optimum')
    _, next_h = filter_variance('egarch', selected.x, x, kappa, absolute)
    return dict(parameters=selected.x.tolist(), objective=float(selected.fun), next_variance_ratio=next_h[0].tolist(),
                search_step=0., optimizer='SLSQP refinement', optimizer_success=True,
                optimizer_iterations=int(selected.nit), refined=True)
