#!/usr/bin/env python3
"""Evaluate independent stochastic volatility against frozen five-model scores."""
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
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT))
import mc_quadrants.dependence as dependence  # noqa: E402
import mc_quadrants.simulation as simulation  # noqa: E402
from mc_quadrants.calibration import calibrate_quadrant_model  # noqa: E402
from mc_quadrants.native import sample_mnts_subordinators_native  # noqa: E402
from scripts.compare_qarch_forecasts import SETTINGS, first_variance, score_forecast  # noqa: E402
from scripts.compare_volatility_models import summarize, update  # noqa: E402
from scripts.stochastic_volatility_research import density_tables, fit_sv, simulate_variance  # noqa: E402

ORIGINAL_FIT = dependence.fit_dependence
ORIGINAL_MACRO = simulation.simulate_joint_regime_macro_paths


def run_origin(task):
    p, macro, returns, baseline, paths, seed, grid_points, density_nodes, density_sample_size = task
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
    simulation.sample_mnts_subordinators = lambda rng, n, a, t: sample_mnts_subordinators_native(n, a, t, int(rng.integers(0, 2**62)))
    training = returns.iloc[:p]
    model = calibrate_quadrant_model(training, macro.loc[:training.index[-1]], volatility_model='garch', **SETTINGS)
    assert captured['states'] == model.states
    axis, tables, density_diagnostics = density_tables(model, nodes=density_nodes, sample_size=density_sample_size)
    x, codes = captured['x'], captured['codes']
    fits = [fit_sv(x[:, j], codes, axis, tables[:, j], grid_points=grid_points) for j in range(len(model.assets))]
    dep = model.metadata['dependence_fit']
    result = simulation.simulate_returns(model, periods=12, paths=paths, random_seed=seed+p*101,
        duration_model='semi_markov', joint_macro=True, macro_parameter_uncertainty=False,
        dynamic_correlation=True, garch=False, return_regime_codes=True,
        **{key: dep[key] for key in ('dcc_alpha', 'dcc_beta', 'dcc_asymmetry')})
    dynamics = model.metadata['macro_dynamics']
    levels = np.stack([np.diag(dynamics['return_residual_covariances'][s]) for s in model.states])
    means = np.stack([model.moments[s].mean.to_numpy() for s in model.states])
    shared_mean = means[result.regimes]+captured['macro_shocks'] @ np.asarray(dynamics['return_betas'])
    vols = np.sqrt(levels[result.regimes])
    z = (result.returns-shared_mean)/vols
    mean, baseline_variance, _ = first_variance(model, 'garch')
    posterior = simulation._latest_state_age_posterior(model)
    probabilities = posterior.sum(axis=1) if posterior is not None else simulation.stationary_distribution(model.transition_matrix).to_numpy()
    actual = returns.iloc[p:p+12].to_numpy()
    # Replay frozen GARCH with the shared draws to audit cross-run comparability.
    params = np.asarray(baseline['parameters'])
    h = np.broadcast_to(baseline['next_variance_ratio'], z[0].shape).copy()
    replay = np.empty_like(z)
    for t in range(12):
        residual = np.sqrt(h)*z[t]
        replay[t] = shared_mean[t]+vols[t]*residual
        h = update('garch', params, h, residual, .5, .8)
    audit = score_forecast(replay, actual, mean, model.assets)
    differences = [abs(audit[key]-baseline[key]) for key in ['crps_1m','crps_3m','crps_12m'] if key in audit]
    assert max(differences) < 1e-12, 'GARCH primitive-draw replay changed'
    ratios = simulate_variance(fits, 12, paths, seed+10000019+p*101)
    samples = shared_mean+vols*np.sqrt(ratios)*z
    initial_h = np.array([f['next_variance_ratio'] for f in fits])
    variance = baseline_variance+(probabilities @ levels)*(initial_h-1)
    qlike = np.log(variance)+(actual[0]-mean)**2/variance
    row = dict(model='sv', date=str(returns.index[p].date()), training_months=p,
               **score_forecast(samples, actual, mean, model.assets), asset_fits=fits,
               density_diagnostics=density_diagnostics, baseline_replay_max_error=max(differences))
    row['qlike_assets_mc'] = row['qlike_assets']
    row['qlike_assets'] = float(qlike.mean())
    for j, ticker in enumerate(model.assets):
        row[f'qlike_{ticker}'] = float(qlike[j])
        row[f'analytic_variance_{ticker}'] = float(variance[j])
        row[f'initial_variance_mc_{ticker}'] = float(ratios[0, :, j].mean())
        row[f'initial_variance_expected_{ticker}'] = float(initial_h[j])
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='research/stochastic_volatility_2026-10-04')
    parser.add_argument('--baseline', default='research/independent_volatility_models_2026-10-04')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--grid-points', type=int, default=81)
    parser.add_argument('--density-nodes', type=int, default=1024)
    parser.add_argument('--density-sample-size', type=int, default=262144)
    args = parser.parse_args()
    baseline = ROOT/args.baseline
    bm = json.loads((baseline/'manifest.json').read_text())
    data = (ROOT/'research/model_quality_2025/data.pkl').read_bytes()
    assert hashlib.sha256(data).hexdigest() == bm['data_sha256']
    assert bm['settings'] == SETTINGS
    for path, digest in bm['source_hashes'].items():
        if path.startswith('src/'):
            assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == digest
    macro, returns = pickle.loads(data)
    returns = returns.loc['2006-03-01':].dropna()
    positions = list(range(60, len(returns)))[::args.stride]
    if args.limit:
        positions = positions[:args.limit]
    frozen_rows = json.loads((baseline/'forecasts.json').read_text())
    frozen_garch = {r['date']: r for r in frozen_rows if r['model'] == 'garch'}
    output = ROOT/args.output
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(data_sha256=bm['data_sha256'], origins=len(positions), paths=bm['paths'], seed=bm['seed'],
        baseline=str(baseline), settings=SETTINGS, assets=returns.columns.tolist(),
        forecast_dates=[str(returns.index[p].date()) for p in positions], grid_points=args.grid_points,
        grid_range=[-7,7], density_nodes=args.density_nodes, density_sample_size=args.density_sample_size,
        density_table_points=4097, latent_parameters='mu, phi, stationary_sd; bounds [-2,2], [0,.98], [0,1.5]',
        estimation='per-ETF approximate marginal MNTS likelihood, four-start L-BFGS-B, first12 warmup',
        filtering='causal discrete-state forward filter; continuous Gaussian AR1 transitions in forecasts',
        uncertainty='filtered latent state and future shocks; no parameter posterior or volatility leverage effect',
        macro_vintage=bm['macro_vintage'],
        source_hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
            list((ROOT/'src/mc_quadrants').glob('*.py'))+[Path(__file__), ROOT/'scripts/stochastic_volatility_research.py', ROOT/'scripts/compare_volatility_models.py', ROOT/'scripts/compare_qarch_forecasts.py']})
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    start = time.perf_counter()
    rows = []
    tasks = [(p, macro, returns, frozen_garch[str(returns.index[p].date())], bm['paths'], bm['seed'], args.grid_points, args.density_nodes, args.density_sample_size) for p in positions]
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn')) as executor:
        for n, row in enumerate(executor.map(run_origin, tasks), 1):
            rows.append(row)
            (output/'forecasts.json').write_text(json.dumps(rows, indent=2, allow_nan=False))
            if n % 3 == 0 or n == len(positions):
                print(f'{n}/{len(positions)} origins; {time.perf_counter()-start:.1f}s', flush=True)
    manifest['elapsed_seconds'] = time.perf_counter()-start
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    pd.DataFrame(rows).to_csv(output/'forecasts.csv', index=False)
    dates = set(r['date'] for r in rows)
    combined = [r for r in frozen_rows if r['date'] in dates]+rows
    summary = summarize(combined)
    (output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps({key: summary['metrics'][key]['sv'] for key in ['qlike_assets', 'crps_1m', 'crps_12m']}, indent=2))


if __name__ == '__main__':
    main()
