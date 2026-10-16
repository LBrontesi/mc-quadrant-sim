#!/usr/bin/env python3
"""Controlled expanding-window comparison of four monthly volatility kernels.

Research only: all models carry normalized volatility memory across regimes.
Shared macro, MNTS tails, ADCC and primitive shocks isolate the volatility model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))
import mc_quadrants.dependence as dependence  # noqa: E402
import mc_quadrants.simulation as simulation  # noqa: E402
from mc_quadrants.calibration import calibrate_quadrant_model  # noqa: E402
from mc_quadrants.native import sample_mnts_subordinators_native  # noqa: E402
from mc_quadrants.qarch import kernel, kernel_components, variance_ratio  # noqa: E402
from scripts.compare_model_quality import paired_ci  # noqa: E402
from scripts.compare_qarch_forecasts import SETTINGS, WEIGHTS, first_variance, score_forecast  # noqa: E402

NAMES = ('garch', 'gjr', 'egarch', 'qarch')
ORIGINAL_FIT = dependence.fit_dependence
ORIGINAL_MACRO = simulation.simulate_joint_regime_macro_paths


def update(name, parameters, h, x, negative_moment, absolute_moment):
    """Update normalized variance after observing x; parameters broadcast to h."""
    if name == 'garch':
        alpha, beta = parameters
        return 1-alpha-beta + alpha*x*x + beta*h
    if name == 'gjr':
        positive, negative, beta = parameters
        persistence = positive*(1-negative_moment) + negative*negative_moment + beta
        return 1-persistence + np.where(x < 0, negative, positive)*x*x + beta*h
    omega, alpha, gamma, beta = parameters
    z = x/np.sqrt(h)
    return np.exp(omega + beta*np.log(h) + alpha*(np.abs(z)-absolute_moment) + gamma*z)


def valid(name, parameters, kappa, limits):
    if name == 'garch':
        return (parameters >= 0).all(axis=1) & (parameters.sum(axis=1) < .98)
    if name == 'qarch':
        return (parameters >= 0).all(axis=1) & (parameters.sum(axis=1) < .98)
    if name == 'gjr':
        pos, neg, beta = parameters.T
        persistence = pos[:, None]*(1-kappa.ravel()) + neg[:, None]*kappa.ravel() + beta[:, None]
        return (parameters >= 0).all(axis=1) & (persistence.max(axis=1) < .98)
    omega, alpha, gamma, beta = parameters.T
    return ((np.abs(omega) <= .5) & (alpha >= 0) & (alpha <= 1)
            & (np.abs(gamma) <= .75) & (beta >= 0) & (beta < .98)
            & (alpha+gamma < limits[0]*.95) & (alpha-gamma < limits[1]*.95))


def filter_variance(name, parameters, x, kappa, absolute):
    """Causal batch filter. Returns pre-observation variances and next variance."""
    parameters = np.atleast_2d(parameters)
    h = np.ones((len(parameters), x.shape[1]))
    history = []
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        for t, residual in enumerate(x):
            history.append(h.copy())
            h = update(name, parameters.T[:, :, None], h, residual, kappa[t], absolute[t])
    return np.asarray(history), h


def coordinate_fit(name, x, kappa, absolute, limits, starts=None, rounds=256):
    """Multi-start batched pattern search, identical unpenalized QML objective."""
    if name == 'qarch':
        history = np.zeros((x.shape[1], 12))
        features = []
        for t, residual in enumerate(x):
            features.append([variance_ratio(history, c, min(t, 12)) for c in kernel_components()])
            history[:, 1:] = history[:, :-1].copy()
            history[:, 0] = residual
        features = np.asarray(features).transpose(0, 2, 1)-1
    defaults = {
        'garch': ([[.1, .85], [.1, .4], [.25, .05]], [.08, .16]),
        'gjr': ([[.1, .2, .75], [.2, .05, .4], [.05, .3, .05]], [.08, .08, .16]),
        'egarch': ([[0, .15, -.1, .85], [0, .1, .1, .4], [0, .3, 0, .05]], [.08, .1, .08, .16]),
        'qarch': ([[.35, .1, .1], [.1, 0, 0], [.6, .1, .1]], [.12, .12, .12]),
    }
    base, increments = defaults[name]
    best = np.asarray(base if starts is None else starts, dtype=float)
    steps = np.broadcast_to(increments, best.shape).copy()

    def decode(parameters):
        parameters = np.asarray(parameters).copy()
        if name == 'egarch':
            # Search log-variance level instead of omega to reduce the
            # intercept/persistence ridge; return conventional omega below.
            parameters[..., 0] *= 1-parameters[..., 3]
        return parameters

    def score(parameters):
        parameters = decode(parameters)
        if name == 'qarch':
            h = 1 + np.einsum('tak,pk->tpa', features, parameters)
        else:
            h, _ = filter_variance(name, parameters, x, kappa, absolute)
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            losses = np.mean(np.log(h[12:]) + x[12:, None, :]**2/h[12:], axis=(0, 2))
        okay = valid(name, parameters, kappa, limits)
        okay &= np.isfinite(h).all(axis=(0, 2)) & (h > 1e-10).all(axis=(0, 2))
        return np.where(okay & np.isfinite(losses), losses, np.inf)

    losses = score(best)
    for _ in range(rounds):
        directions = np.vstack([np.zeros(best.shape[1]), np.eye(best.shape[1]), -np.eye(best.shape[1])])
        candidates = best[:, None, :] + steps[:, None, :]*directions
        scores = score(candidates.reshape(-1, best.shape[1])).reshape(candidates.shape[:2])
        selections = scores.argmin(axis=1)
        new_losses = scores[np.arange(len(best)), selections]
        improved = new_losses < losses-1e-10
        best = candidates[np.arange(len(best)), selections]
        steps[~improved] *= .5
        losses = new_losses
        if steps.max() < 1e-6:
            break
    selected = int(losses.argmin())
    params = decode(best[selected])
    if not np.isfinite(losses[selected]):
        raise ValueError(f'No finite {name} fit')
    if name == 'qarch':
        next_h = variance_ratio(history, kernel(params))
    else:
        _, next_h = filter_variance(name, params, x, kappa, absolute)
        next_h = next_h[0]
    return dict(parameters=params.tolist(), objective=float(losses[selected]),
                next_variance_ratio=next_h.tolist(), search_step=float(steps[selected].max()))


def innovation_moments(model, *, per_asset=False):
    """Fixed-seed antithetic MC moments of fitted MNTS, not Gaussian constants."""
    kappas, absolutes, positive_limits, negative_limits = [], [], [], []
    for index, state in enumerate(model.states):
        parameters = model.metadata['macro_dynamics']['residual_mnts'][state]
        nu = parameters.skewness.reindex(model.assets).to_numpy()
        theta = parameters.tempering
        scale2 = np.maximum(1-nu*nu*(1-parameters.tail_index/2)/theta, 1e-10)
        t = sample_mnts_subordinators_native(32768, parameters.tail_index, theta, 491+index)
        normal = np.random.default_rng(391+index).standard_normal((len(t), len(nu)))
        center, spread = nu*(t[:, None]-1), np.sqrt(t[:, None]*scale2)*normal
        z = np.concatenate([center+spread, center-spread])
        kappas.append(np.mean(z*z*(z < 0), axis=0))
        absolutes.append(np.mean(np.abs(z), axis=0))
        root = np.sqrt(nu*nu+2*scale2*theta)
        positive_limits.append((-nu+root)/scale2)
        negative_limits.append((nu+root)/scale2)
    limits = np.array([positive_limits, negative_limits])
    limits = limits.min(axis=1) if per_asset else limits.min(axis=(1, 2))
    return np.asarray(kappas), np.asarray(absolutes), limits


def fit_models(x, codes, kappa, absolute, limits, *, independent=False, rounds=256, names=NAMES):
    """Fit each ETF using only its residual column, or share coefficients."""
    fits = {}
    for name in names:
        asset_fits = []
        for j in range(x.shape[1] if independent else 1):
            starts = None
            if name == 'gjr':
                alpha, beta = np.asarray(fits['garch']['parameters'])
                if independent:
                    alpha, beta = alpha[j], beta[j]
                starts = [[alpha, alpha, beta], [.1, .2, .75], [.2, .05, .4], [.05, .3, .05]]
            columns = slice(j, j+1) if independent else slice(None)
            if name == 'figarch':
                from scripts.volatility_research import fit_figarch
                if not independent:
                    raise ValueError('FIGARCH benchmark currently requires independent fitting')
                fit = fit_figarch(x[:, columns])
            else:
                fit = coordinate_fit(name, x[:, columns], kappa[codes, columns], absolute[codes, columns],
                                     limits[:, j] if independent else limits, starts, rounds=rounds)
            # Sparse single-ETF EGARCH fits can have shallow objective ridges.
            # Refine based on training convergence only, never forecast scores.
            if independent and rounds >= 256 and fit['search_step'] >= 1e-6:
                if name == 'egarch':
                    from scripts.volatility_research import refine_egarch
                    refined = refine_egarch(fit, x[:, columns], kappa[codes, columns], absolute[codes, columns],
                                            limits[:, j], filter_variance)
                else:
                    refined = coordinate_fit(name, x[:, columns], kappa[codes, columns], absolute[codes, columns],
                                             limits[:, j], [fit['parameters']], rounds=4096)
                assert refined['objective'] <= fit['objective']+1e-10
                fit = refined
                fit['refined'] = True
                if fit['search_step'] >= 1e-6:
                    raise ValueError(f'{name} asset {j} optimizer failed to converge')
            asset_fits.append(fit)
        if independent:
            fits[name] = dict(parameters=np.asarray([f['parameters'] for f in asset_fits]).T.tolist(),
                objective=float(np.mean([f['objective'] for f in asset_fits])),
                next_variance_ratio=[f['next_variance_ratio'][0] for f in asset_fits],
                search_step=max(f['search_step'] for f in asset_fits), asset_fits=asset_fits)
        else:
            fits[name] = asset_fits[0]
    return fits


def qarch_asset_variances(history, parameters):
    """Apply one shared QARCH kernel or one separate kernel per asset."""
    parameters = np.asarray(parameters)
    if parameters.ndim == 1:
        return variance_ratio(history, kernel(parameters))
    return np.stack([variance_ratio(history[..., j, :], kernel(parameters[:, j]))
                     for j in range(history.shape[-2])], axis=-1)


def run_origin(task):
    p, macro, returns, paths, seed, *options = task
    independent = bool(options and options[0])
    names = (*NAMES, 'figarch') if len(options) > 1 and options[1] else NAMES
    captured = {}

    def capture_fit(r, regimes, moments, **kwargs):
        states = list(moments)
        codes = np.array([states.index(s) for s in regimes])
        means = np.stack([moments[s].mean.to_numpy() for s in states])
        levels = np.stack([np.diag(moments[s].covariance) for s in states])
        captured['x'] = (r.to_numpy()-means[codes])/np.sqrt(levels[codes])
        captured['codes'] = codes
        captured['states'] = states
        return ORIGINAL_FIT(r, regimes, moments, **kwargs)

    def capture_macro(*args, **kwargs):
        result = ORIGINAL_MACRO(*args, **kwargs)
        captured['macro_shocks'] = result[2]
        return result

    dependence.fit_dependence = capture_fit
    simulation.simulate_joint_regime_macro_paths = capture_macro
    simulation.simulate_parametric_native = lambda **kwargs: None
    simulation.sample_mnts_subordinators = lambda rng, n, a, t: sample_mnts_subordinators_native(
        n, a, t, int(rng.integers(0, 2**62)))
    training = returns.iloc[:p]
    model = calibrate_quadrant_model(training, macro.loc[:training.index[-1]], volatility_model='garch', **SETTINGS)
    assert captured['states'] == model.states
    x, codes = captured['x'], captured['codes']
    kappa, absolute, limits = innovation_moments(model, per_asset=independent)
    fits = fit_models(x, codes, kappa, absolute, limits, independent=independent, names=names)
    dep = model.metadata['dependence_fit']
    result = simulation.simulate_returns(model, periods=12, paths=paths, random_seed=seed+p*101,
        duration_model='semi_markov', joint_macro=True, macro_parameter_uncertainty=False,
        dynamic_correlation=True, garch=False, return_regime_codes=True,
        **{key: dep[key] for key in ('dcc_alpha', 'dcc_beta', 'dcc_asymmetry')})
    dynamics = model.metadata['macro_dynamics']
    levels = np.stack([np.diag(dynamics['return_residual_covariances'][s]) for s in model.states])
    state_means = np.stack([model.moments[s].mean.to_numpy() for s in model.states])
    shared_mean = state_means[result.regimes] + captured['macro_shocks'] @ np.asarray(dynamics['return_betas'])
    vols = np.sqrt(levels[result.regimes])
    z = (result.returns-shared_mean)/vols
    mean, baseline_variance, _ = first_variance(model, 'garch')
    posterior = simulation._latest_state_age_posterior(model)
    probabilities = posterior.sum(axis=1) if posterior is not None else simulation.stationary_distribution(model.transition_matrix).to_numpy()
    actual = returns.iloc[p:p+12].to_numpy()
    rows = []
    for name in names:
        fit = fits[name]
        params = np.asarray(fit['parameters'])
        h = np.broadcast_to(fit['next_variance_ratio'], (paths, len(model.assets))).copy()
        initial_h = h[0].copy()
        history = np.broadcast_to(x[-12:][::-1].T, (paths, len(model.assets), 12)).copy()
        samples = np.empty_like(result.returns)
        if name == 'figarch':
            from scripts.volatility_research import figarch_weights
            fig_weights = np.stack([figarch_weights(params[:, j]) for j in range(len(model.assets))])
            # Historical contribution for each future step is deterministic;
            # add realized simulated squares for the short forward horizon.
            past_squares = x[::-1].T**2
            fig_intercept = params[0]/(1-params[3])
            simulated_squares = []
        for t in range(12):
            residual = np.sqrt(h)*z[t]
            samples[t] = shared_mean[t] + vols[t]*residual
            if name == 'qarch':
                history[:, :, 1:] = history[:, :, :-1].copy()
                history[:, :, 0] = residual
                h = qarch_asset_variances(history, params)
            elif name == 'figarch':
                simulated_squares.insert(0, residual**2)
                future_lags = t+1
                historical = np.sum(fig_weights[:, future_lags:future_lags+len(x)]*past_squares, axis=1)
                presample = fig_weights[:, future_lags+len(x):].sum(axis=1)
                h = fig_intercept+historical+presample
                h = h + np.einsum('lpa,al->pa', np.asarray(simulated_squares), fig_weights[:, :future_lags])
            else:
                h = update(name, params, h, residual, kappa[result.regimes[t]], absolute[result.regimes[t]])
            if not np.isfinite(h).all() or np.any(h <= 0):
                raise ValueError(f'Invalid {name} simulated variance at {returns.index[p]}')
        variance = baseline_variance + (probabilities @ levels)*(initial_h-1)
        qlike = np.log(variance)+(actual[0]-mean)**2/variance
        row = dict(model=name, date=str(returns.index[p].date()), training_months=p,
                   **score_forecast(samples, actual, mean, model.assets), **fit)
        row['qlike_assets_mc'] = row['qlike_assets']
        row['qlike_assets'] = float(qlike.mean())
        for j, ticker in enumerate(model.assets):
            row[f'qlike_{ticker}'] = float(qlike[j])
            row[f'analytic_variance_{ticker}'] = float(variance[j])
        rows.append(row)
    return rows


def summarize(rows):
    names = tuple(dict.fromkeys(row['model'] for row in rows))
    frames = {name: pd.DataFrame([r for r in rows if r['model'] == name]).set_index('date') for name in names}
    base = frames['garch']
    metrics = ['qlike_assets', 'crps_1m', 'crps_3m', 'crps_12m', 'drawdown_crps_12m', 'loss_brier']
    metrics += [f'qlike_{ticker}' for ticker in ['SPY', 'IEF', 'GLD', 'DBC']]
    metrics += [f'crps_{ticker}' for ticker in ['SPY', 'IEF', 'GLD', 'DBC']]
    output = {'metrics': {}, 'coverage': {}, 'pairwise': {}, 'periods': {}}
    for metric in metrics:
        output['metrics'][metric] = {}
        for name, frame in frames.items():
            pairs = pd.concat([base[metric], frame[metric]], axis=1).dropna()
            delta = pairs.iloc[:, 1]-pairs.iloc[:, 0]
            output['metrics'][metric][name] = dict(mean=float(pairs.iloc[:, 1].mean()),
                delta_vs_garch=float(delta.mean()), ci95=paired_ci(delta),
                ci95_block6=paired_ci(delta, length=6), n=len(pairs),
                improvement_percent=None if metric.startswith('qlike') else float(-100*delta.mean()/pairs.iloc[:, 0].mean()))
    for name, frame in frames.items():
        output['coverage'][name] = {key: float(frame[key].mean()) for key in ['coverage90', 'breach05', 'width90']}
    for i, a in enumerate(names):
        for b in names[i+1:]:
            delta = frames[b]['qlike_assets']-frames[a]['qlike_assets']
            output['pairwise'][f'{b}_minus_{a}'] = dict(delta=float(delta.mean()), ci95=paired_ci(delta))
    for start, end in [('2011', '2015'), ('2016', '2019'), ('2020', '2022'), ('2023', '2025')]:
        output['periods'][f'{start}-{end}'] = {}
        for name, frame in frames.items():
            mask = (frame.index.str[:4] >= start) & (frame.index.str[:4] <= end)
            if mask.any():
                output['periods'][f'{start}-{end}'][name] = {m: float(frame.loc[mask, m].mean()) for m in metrics[:2]}
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output')
    parser.add_argument('--independent', action='store_true', help='Fit separate volatility coefficients per ETF')
    parser.add_argument('--include-figarch', action='store_true', help='Add independent FIGARCH(1,d,1), truncated at 1000 lags')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--paths', type=int, default=8192)
    parser.add_argument('--seed', type=int, default=20261004)
    args = parser.parse_args()
    if args.include_figarch and not args.independent:
        parser.error('--include-figarch requires --independent')
    source = ROOT/'research/model_quality_2025'
    data = (source/'data.pkl').read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    assert digest == json.loads((source/'manifest.json').read_text())['data_sha256']
    macro, returns = pickle.loads(data)
    returns = returns.loc['2006-03-01':].dropna()
    assert returns.columns.tolist() == ['SPY', 'IEF', 'GLD', 'DBC']
    positions = list(range(60, len(returns)))
    if args.limit:
        positions = positions[:args.limit]
    output = ROOT/(args.output or ('research/independent_volatility_models_2026-10-04' if args.independent
                                  else 'research/four_volatility_models_2026-10-04'))
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(data_sha256=digest, models=(*NAMES, 'figarch') if args.include_figarch else NAMES, forecast_start=str(returns.index[positions[0]].date()),
        forecast_end=str(returns.index[positions[-1]].date()), origins=len(positions), paths=args.paths, seed=args.seed,
        weights=WEIGHTS.tolist(), settings=SETTINGS, pooled_parameters=not args.independent,
        parameter_scope='separate ETF coefficients, shared across regimes' if args.independent else 'shared across ETFs and regimes',
        estimation='unpenalized Gaussian QML; common 12-month warmup; multistart pattern search',
        initialization='filtered recent normalized residuals for every model; memory persists across regimes',
        qarch_difference='same PSD basis as experimental implementation, without its 0.01 penalty; common optimizer',
        macro_vintage='latest revised, already availability shifted one month; NOT point in time',
        shared_components='regime levels, means, macro, MNTS and GARCH-fitted ADCC held fixed across all four models',
        innovation_moments='65536 fixed-seed antithetic MNTS draws per regime; EGARCH moment-domain constraints',
        simulation_backend='shared Python return recurrence with native macro and MNTS sampling',
        ci='paired circular 12-month block bootstrap, 4000 replicates; pointwise, not adjusted for model selection',
        source_hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in list((ROOT/'src/mc_quadrants').glob('*.py'))+[Path(__file__), ROOT/'scripts/compare_qarch_forecasts.py', ROOT/'scripts/volatility_research.py']})
    if args.include_figarch:
        manifest['figarch'] = dict(truncation=1000, presample_squared_normalized_residual=1.,
                                  intercept='free omega; no finite unconditional variance target', optimizer='multistart SLSQP')
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    began = time.perf_counter()
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn')) as executor:
        for n, batch in enumerate(executor.map(run_origin, [(p, macro, returns, args.paths, args.seed, args.independent, args.include_figarch) for p in positions]), 1):
            rows.extend(batch)
            (output/'forecasts.json').write_text(json.dumps(rows, indent=2, allow_nan=False))
            if n % 6 == 0 or n == len(positions):
                print(f'{n}/{len(positions)} origins; {time.perf_counter()-began:.1f}s', flush=True)
    manifest['elapsed_seconds'] = time.perf_counter()-began
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    pd.DataFrame(rows).to_csv(output/'forecasts.csv', index=False)
    summary = summarize(rows)
    (output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps({k: summary['metrics'][k] for k in ['qlike_assets', 'crps_1m', 'crps_12m']}, indent=2))


if __name__ == '__main__':
    main()
