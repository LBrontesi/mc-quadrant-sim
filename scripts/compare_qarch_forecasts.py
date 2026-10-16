#!/usr/bin/env python3
"""Expanding-window QARCH/GARCH forecast comparison on frozen real ETF history.

No production changes, holdout tuning, or fresh downloads. GARCH is evaluated
as deployed (regime-level initialization); QARCH retains filtered history.
Both use identical calibration options and common primitive random draws.
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
import mc_quadrants.simulation as simulation  # noqa: E402
from mc_quadrants.calibration import calibrate_quadrant_model  # noqa: E402
from mc_quadrants.native import sample_mnts_subordinators_native  # noqa: E402
from scripts.compare_model_quality import paired_ci, scalar_score  # noqa: E402

WEIGHTS = np.array([.4, .3, .15, .15])
SETTINGS = dict(growth_threshold='median', inflation_threshold='median',
                threshold_window=12, macro_lag_periods=0, probabilistic_regimes=True,
                mean_prior_strength=24., duration_prior_strength=8., joint_macro=True,
                macro_model='bvar_ensemble')


def score_forecast(samples, realized, mean, tickers):
    first = samples[0]
    variance = first.var(axis=0, ddof=1)
    errors = realized[0] - mean
    qlike = np.log(variance) + errors**2 / variance
    portfolio = np.expm1(samples) @ WEIGHTS
    actual = np.expm1(realized) @ WEIGHTS
    q05, q95 = np.quantile(portfolio[0], [.05, .95])
    result = {'qlike_assets': float(qlike.mean()),
              'crps_1m': scalar_score(portfolio[0], actual[0]),
              'coverage90': int(q05 <= actual[0] <= q95),
              'breach05': int(actual[0] < q05), 'width90': float(q95-q05),
              'loss_brier': float((np.mean(portfolio[0] < 0) - (actual[0] < 0))**2),
              'actual_1m': float(actual[0]), 'q05': float(q05), 'q95': float(q95)}
    for j, ticker in enumerate(tickers):
        result[f'qlike_{ticker}'] = float(qlike[j])
        result[f'crps_{ticker}'] = scalar_score(first[:, j], realized[0, j])
        result[f'variance_{ticker}'] = float(variance[j])
    for horizon in (3, 12):
        if len(realized) < horizon:
            continue
        nav = np.vstack([np.ones(portfolio.shape[1]), np.cumprod(1+portfolio[:horizon], axis=0)])
        actual_nav = np.r_[1., np.cumprod(1+actual[:horizon])]
        result[f'crps_{horizon}m'] = scalar_score(nav[-1]-1, actual_nav[-1]-1)
        dd = (1-nav/np.maximum.accumulate(nav, axis=0)).max(axis=0)
        actual_dd = (1-actual_nav/np.maximum.accumulate(actual_nav)).max()
        result[f'drawdown_crps_{horizon}m'] = scalar_score(dd, actual_dd)
    return result


def summarize(rows, manifest):
    frames = {name: pd.DataFrame([row for row in rows if row['model']==name]).set_index('date')
              for name in ('garch', 'qarch')}
    a, b = frames['garch'], frames['qarch']
    assert a.index.equals(b.index)
    metrics = ['qlike_assets', 'qlike_filtered_assets', 'crps_1m', 'crps_3m', 'crps_12m',
               'drawdown_crps_3m', 'drawdown_crps_12m', 'loss_brier']
    metrics += [f'{metric}_{ticker}' for metric in ('qlike', 'qlike_filtered', 'crps') for ticker in manifest['tickers']]
    summary = {'manifest': manifest, 'metrics': {}, 'coverage': {}, 'periods': {}}
    for metric in metrics:
        if metric not in a or metric not in b:
            continue
        pair = pd.concat([a[metric], b[metric]], axis=1).dropna()
        delta = pair.iloc[:,1]-pair.iloc[:,0]
        summary['metrics'][metric] = dict(garch=float(pair.iloc[:,0].mean()), qarch=float(pair.iloc[:,1].mean()),
            qarch_minus_garch=float(delta.mean()), delta_ci95=paired_ci(delta),
            delta_ci95_block6=paired_ci(delta, length=6),
            improvement_percent=None if metric.startswith('qlike') else float(-100*delta.mean()/pair.iloc[:,0].mean()),
            n=len(pair), qarch_win_fraction=float((delta<0).mean()))
    for name, frame in frames.items():
        summary['coverage'][name] = {key: float(frame[key].mean()) for key in ('coverage90','breach05','width90')}
        summary['coverage'][name]['breaches'] = int(frame['breach05'].sum())
    for start, end in [('2011','2015'), ('2016','2019'), ('2020','2022'), ('2023','2025')]:
        index = a.index.str[:4].astype(int)
        mask = (index>=int(start)) & (index<=int(end))
        if mask.any():
            summary['periods'][f'{start}-{end}'] = {
                metric: float((b.loc[mask,metric]-a.loc[mask,metric]).mean())
                for metric in ('qlike_assets','crps_1m')}
    return summary


def first_variance(model, name):
    """Exact first-step marginal moments, including macro and regime mixture.

    The supplemental filtered-GARCH forecast only changes its initial variance
    in the continuing state, as the existing validation code does. No refitting.
    """
    posterior = simulation._latest_state_age_posterior(model)
    probabilities = (posterior.sum(axis=1) if posterior is not None else
                     simulation.stationary_distribution(model.transition_matrix).to_numpy())
    means = np.stack([model.moments[state].mean.to_numpy() for state in model.states])
    dynamics = model.metadata['macro_dynamics']
    beta = np.asarray(dynamics['return_betas'])
    levels = np.stack([np.diag(dynamics['return_residual_covariances'][state]) for state in model.states])
    macro_variance = np.stack([np.diag(beta.T @ np.asarray(dynamics['state_innovation_covariances'][state]) @ beta)
                               for state in model.states])
    fit = model.metadata['dependence_fit']
    conditional = levels.copy()
    if name == 'qarch':
        conditional *= np.asarray(fit['qarch']['next_variance_ratio'])
    filtered = conditional.copy()
    if name == 'garch' and fit.get('status') == 'fitted':
        filtered[model.states.index(fit['last_state'])] = fit['next_variance']
    mean = probabilities @ means
    mixture_variance = probabilities @ np.square(means-mean)
    variance = probabilities @ (conditional+macro_variance) + mixture_variance
    filtered_variance = probabilities @ (filtered+macro_variance) + mixture_variance
    return mean, variance, filtered_variance


def run_origin(task):
    p, macro, returns, paths, seed = task
    # Both reference recurrences share native exact MNTS subordinator draws.
    # This changes sampling speed, not the distribution or forecast recurrence.
    simulation.simulate_parametric_native = lambda **kwargs: None
    simulation.sample_mnts_subordinators = lambda rng, n, a, t: sample_mnts_subordinators_native(
        n, a, t, int(rng.integers(0, 2**62)))
    rows=[]
    target=returns.index[p]
    training=returns.iloc[:p]
    training_macro=macro.loc[macro.index<=training.index[-1]]
    actual=returns.iloc[p:p+12].to_numpy()
    means=[]
    for name in ('garch','qarch'):
        started=time.perf_counter()
        model=calibrate_quadrant_model(training, training_macro, volatility_model=name, **SETTINGS)
        fit=model.metadata['dependence_fit']
        model.metadata['calibrate_dependence']=True
        calibrated=time.perf_counter()
        result=simulation.simulate_returns(model, periods=12, paths=paths,
            random_seed=seed+p*101, duration_model='semi_markov', joint_macro=True,
            macro_parameter_uncertainty=False, dynamic_correlation=True, garch=True,
            volatility_model=name, return_regime_codes=True,
            **{key:fit[key] for key in ('garch_alpha','garch_beta','dcc_alpha','dcc_beta','dcc_asymmetry')})
        simulated=time.perf_counter()
        state_means=np.stack([model.moments[s].mean.to_numpy() for s in model.states])
        mean=state_means[result.regimes[0]].mean(axis=0)
        mean, analytic_variance, filtered_variance = first_variance(model, name)
        means.append(mean)
        row=dict(model=name, date=str(target.date()), training_months=len(training),
                 calibration_seconds=calibrated-started, simulation_seconds=simulated-calibrated,
                 **score_forecast(result.returns, actual, mean, model.assets))
        error = actual[0] - mean
        analytic_qlike = np.log(analytic_variance) + error**2 / analytic_variance
        filtered_qlike = np.log(filtered_variance) + error**2 / filtered_variance
        row['qlike_assets_mc'] = row['qlike_assets']
        row['qlike_assets'] = float(analytic_qlike.mean())
        row['qlike_filtered_assets'] = float(filtered_qlike.mean())
        for j, ticker in enumerate(model.assets):
            row[f'qlike_{ticker}'] = float(analytic_qlike[j])
            row[f'qlike_filtered_{ticker}'] = float(filtered_qlike[j])
            row[f'analytic_variance_{ticker}'] = float(analytic_variance[j])
        if name=='qarch':
            row['qarch_weights']=fit['qarch']['weights']
        rows.append(row)
    np.testing.assert_allclose(means[0], means[1], rtol=0, atol=0)
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='research/model_quality_2025')
    parser.add_argument('--output', default='research/qarch_forecast_2026-10-04')
    parser.add_argument('--paths', type=int, default=8192)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--start', default='2011-03-01')
    parser.add_argument('--seed', type=int, default=20261004)
    args=parser.parse_args()
    source=ROOT/args.data
    output=ROOT/args.output
    output.mkdir(parents=True, exist_ok=True)
    data=(source/'data.pkl').read_bytes()
    source_manifest=json.loads((source/'manifest.json').read_text())
    digest=hashlib.sha256(data).hexdigest()
    assert digest==source_manifest['data_sha256'], 'Frozen data checksum changed'
    macro, returns=pickle.loads(data)
    returns=returns.loc['2006-03-01':].dropna()
    assert returns.columns.tolist()==['SPY','IEF','GLD','DBC']
    positions=[int(p) for p in np.flatnonzero(returns.index>=args.start) if p>=60]
    if args.limit:
        positions=positions[:args.limit]
    manifest=dict(data_sha256=digest, data_source=str(source), tickers=returns.columns.tolist(),
                  weights=WEIGHTS.tolist(), forecast_start=str(returns.index[positions[0]].date()),
                  forecast_end=str(returns.index[positions[-1]].date()), origins=len(positions),
                  paths=args.paths, seed=args.seed, settings=SETTINGS,
                  macro_vintage='latest revised; fixed one-month availability shift already applied',
                  macro_point_in_time=False, parameter_bootstrap=False, hyperparameter_tuning=False,
                  common_random_numbers=True, simulation_backend='Python recurrences with shared native macro and exact MNTS samplers', workers=args.workers,
                  garch_initialization='deployed regime baseline', qarch_initialization='filtered recent history',
                  variance_proxy='squared next-month log return minus shared forecast mean',
                  primary_metric='equal-asset mean one-month QLIKE using analytic variance (log variance + squared error / variance)', filtered_garch_check='same QLIKE using end-of-training variance in continuing state',
                  ci='paired circular 12-month block bootstrap, 4000 replicates; 6-month sensitivity',
                  source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in list((ROOT/'src/mc_quadrants').glob('*.py'))+[Path(__file__)]})
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    rows=[]
    began=time.perf_counter()
    tasks=[(p, macro, returns, args.paths, args.seed) for p in positions]
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn')) as executor:
        for number, origin_rows in enumerate(executor.map(run_origin, tasks), 1):
            rows.extend(origin_rows)
            (output/'forecasts.json').write_text(json.dumps(rows, indent=2, allow_nan=False))
            if number%6==0 or number==len(positions):
                print(f"{number}/{len(positions)} origins, {origin_rows[0]['date']}, elapsed {time.perf_counter()-began:.1f}s",flush=True)
    manifest['elapsed_seconds']=time.perf_counter()-began
    summary=summarize(rows,manifest)
    (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    pd.DataFrame(rows).to_csv(output/'forecasts.csv',index=False)
    print(json.dumps({k:summary['metrics'][k] for k in ('qlike_assets','crps_1m','crps_12m') if k in summary['metrics']},indent=2),flush=True)

if __name__=='__main__':
    main()
