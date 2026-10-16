"""Independent log-AR(1) SV with MNTS emissions and numerical state filtering.

Research approximation: each asset is estimated marginally. Parameters are
point estimates; the filtered latent-state uncertainty is fully propagated.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, ndtr


def normal_mixture_logpdf(z, subordinators, skewness, gaussian_scale2):
    center = skewness*(subordinators-1)
    variance = gaussian_scale2*subordinators
    z = np.asarray(z)
    return logsumexp(-.5*(np.log(2*np.pi*variance)+(z[..., None]-center)**2/variance), axis=-1)-np.log(len(variance))


def density_tables(model, nodes=1024, points=4097, sample_size=262144):
    """Integrate the MNTS Gaussian mixture over fixed empirical T quantiles."""
    from mc_quadrants.native import sample_mnts_subordinators_native

    axis = np.linspace(-12, 12, points)
    z = np.sinh(axis)
    tables = np.empty((len(model.states), len(model.assets), points))
    diagnostics = []
    for i, state in enumerate(model.states):
        params = model.metadata['macro_dynamics']['residual_mnts'][state]
        t = sample_mnts_subordinators_native(sample_size, params.tail_index, params.tempering, 61731+i)
        t = np.quantile(t, (np.arange(nodes)+.5)/nodes)
        nu = params.skewness.reindex(model.assets).to_numpy()
        scale2 = np.maximum(1-nu*nu*(1-params.tail_index/2)/params.tempering, 1e-10)
        for j in range(len(nu)):
            for start in range(0, points, 256):
                tables[i, j, start:start+256] = normal_mixture_logpdf(z[start:start+256], t, nu[j], scale2[j])
            diagnostics.append(dict(state=state, asset=model.assets[j], mean=float(nu[j]*(t.mean()-1)),
                second_moment=float(np.mean(nu[j]**2*(t-1)**2+scale2[j]*t))))
    return axis, tables, diagnostics


def transition_matrix(phi, grid):
    edges = np.r_[-np.inf, (grid[:-1]+grid[1:])/2, np.inf]
    transition = np.diff(ndtr((edges[None, :]-phi*grid[:, None])/np.sqrt(1-phi*phi)), axis=1)
    prior = np.diff(ndtr(edges))
    return transition, prior


def forward_filter(log_emissions, transition, prior, warmup=12):
    """Scaled causal HMM filter; posterior is AFTER the last observation."""
    posterior = prior.copy()
    total = 0.
    edge_max = 0.
    for t, row in enumerate(log_emissions):
        if t:
            posterior = posterior @ transition
        shift = float(row.max())
        posterior *= np.exp(row-shift)
        mass = float(posterior.sum())
        if not np.isfinite(mass) or mass <= 0:
            return -np.inf, posterior, 1.
        posterior /= mass
        if t >= warmup:
            total += np.log(mass)+shift
        edge_max = max(edge_max, float(posterior[0]+posterior[-1]))
    return float(total), posterior, edge_max


def emissions(parameters, x, codes, axis, tables, grid):
    mu, _, stationary_sd = parameters
    h = mu+stationary_sd*grid
    transformed = np.arcsinh(x[:, None]*np.exp(-h/2))
    if np.abs(transformed).max() > axis[-1]:
        raise ValueError('MNTS density table support exceeded')
    result = np.empty_like(transformed)
    for code in np.unique(codes):
        mask = codes == code
        result[mask] = np.interp(transformed[mask], axis, tables[code])-h/2
    return result


def fit_sv(x, codes, axis, tables, grid_points=81):
    x = np.asarray(x).reshape(-1)
    grid = np.linspace(-7., 7., grid_points)

    def evaluate(parameters):
        transition, prior = transition_matrix(parameters[1], grid)
        log_emissions = emissions(parameters, x, codes, axis, tables, grid)
        return forward_filter(log_emissions, transition, prior)

    def loss(parameters):
        likelihood, _, _ = evaluate(parameters)
        return -likelihood/(len(x)-12)

    level = float(np.clip(np.log(max(np.mean(x*x), 1e-6)), -1., 1.))
    starts = [(level-.5*s*s, p, s) for p, s in [(0., .15), (.5, .5), (.9, .7), (.97, 1.)]]
    candidates = []
    for start in starts:
        result = minimize(loss, start, method='L-BFGS-B', bounds=[(-2., 2.), (0., .98), (0., 1.5)],
                          options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-5, 'maxls': 40})
        if result.success and np.isfinite(result.fun):
            candidates.append(result)
    if not candidates:
        raise ValueError('SV optimizer failed for all starts')
    best = min(candidates, key=lambda r: r.fun)
    _, posterior, edge_mass = evaluate(best.x)
    mu, phi, stationary_sd = best.x
    innovation_sd = stationary_sd*np.sqrt(1-phi*phi)
    h = mu+stationary_sd*grid
    # Integrate the continuous next lognormal transition, not exp(E[h]).
    next_variance = posterior @ np.exp(mu+phi*(h-mu)+.5*innovation_sd**2)
    return dict(parameters=best.x.tolist(), parameter_names=['mu_log_variance', 'phi', 'stationary_log_variance_sd'],
                objective=float(best.fun), optimizer_success=True, converged_starts=len(candidates),
                optimizer_iterations=int(best.nit), grid_points=grid_points,
                posterior_grid=h.tolist(), posterior_weights=posterior.tolist(),
                next_variance_ratio=float(next_variance), innovation_sd=float(innovation_sd),
                maximum_posterior_edge_mass=edge_mass)


def simulate_variance(fits, periods, paths, seed):
    """Draw the filtered state, then continuous Gaussian AR(1) innovations."""
    rng = np.random.default_rng(seed)
    h = np.column_stack([rng.choice(fit['posterior_grid'], size=paths, p=fit['posterior_weights']) for fit in fits])
    parameters = np.array([fit['parameters'] for fit in fits]).T
    mu, phi, stationary_sd = parameters
    tau = stationary_sd*np.sqrt(1-phi*phi)
    result = np.empty((periods, paths, len(fits)))
    for t in range(periods):
        h = mu+phi*(h-mu)+tau*rng.standard_normal(h.shape)
        result[t] = np.exp(h)
    return result
