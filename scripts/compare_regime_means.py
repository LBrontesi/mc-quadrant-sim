#!/usr/bin/env python3
"""Controlled walk-forward ablation of regime-dependent expected log returns.

One current-engine calibration per origin. Only state means change; covariance,
GARCH/ADCC, MNTS, macro coefficients, regimes and primitive shocks stay fixed.
This is not a full re-estimation of each alternative data-generating model.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))
from mc_quadrants.calibration import (  # noqa: E402
    _align_probabilities_to_returns,
    calibrate_quadrant_model,
)
from mc_quadrants.simulation import (  # noqa: E402
    _latest_state_age_posterior,
    simulate_returns,
)
from scripts.compare_model_quality import scalar_score  # noqa: E402
from scripts.compare_qarch_forecasts import SETTINGS, WEIGHTS  # noqa: E402

NAMES = ('unconditional', 'current24', 'strong96')
HORIZONS = (1, 3, 12)


def training_slice(macro, returns, position):
    training = returns.iloc[:position].copy()
    return macro.loc[:training.index[-1]].copy(), training


def mean_variants(values, memberships):
    """Match soft-state mean estimation; strengths fixed before scoring."""
    values, memberships = np.asarray(values), np.asarray(memberships)
    global_mean = values.mean(axis=0)
    counts = memberships.sum(axis=0)
    deviations = memberships.T @ (values - global_mean)
    pooled = np.broadcast_to(global_mean, (memberships.shape[1], values.shape[1])).copy()
    return {
        'unconditional': pooled,
        'current24': pooled + deviations / (counts[:, None] + 24),
        'strong96': pooled + deviations / (counts[:, None] + 96),
    }


def shift_paths(samples, codes, old_means, new_means):
    return samples + (new_means - old_means)[codes]


def score_variant(samples, codes, state_means, first_probabilities, actual, tickers, training_variance):
    """Score exact 1m / Rao-Blackwellized longer-horizon expected log returns.

Future macro and MNTS innovations have mean zero. Integrating them out avoids
noisy sample-average means; only future regime occupancy needs Monte Carlo.
Portfolio simple-return means and CRPS retain simulation integration error.
"""
    predicted_monthly = state_means[codes].mean(axis=1)
    predicted_monthly[0] = first_probabilities @ state_means
    monthly_portfolio = np.expm1(samples) @ WEIGHTS
    realized_portfolio = np.expm1(actual) @ WEIGHTS
    row = {}
    for horizon in HORIZONS:
        if len(actual) < horizon:
            continue
        forecast = predicted_monthly[:horizon].sum(axis=0)
        observed = actual[:horizon].sum(axis=0)
        error = forecast - observed
        row[f'mse_{horizon}m'] = float(np.mean(error**2))
        row[f'scaled_mse_{horizon}m'] = float(np.mean(error**2 / (horizon * training_variance)))
        for j, ticker in enumerate(tickers):
            row[f'forecast_{ticker}_{horizon}m'] = float(forecast[j])
            row[f'actual_{ticker}_{horizon}m'] = float(observed[j])
            row[f'mse_{ticker}_{horizon}m'] = float(error[j]**2)
            row[f'bias_{ticker}_{horizon}m'] = float(error[j])
        draws = np.prod(1 + monthly_portfolio[:horizon], axis=0) - 1
        realized = float(np.prod(1 + realized_portfolio[:horizon]) - 1)
        row[f'portfolio_forecast_{horizon}m'] = float(draws.mean())
        row[f'portfolio_actual_{horizon}m'] = realized
        row[f'portfolio_mse_{horizon}m'] = float((draws.mean() - realized)**2)
        row[f'portfolio_crps_{horizon}m'] = scalar_score(draws, realized)
    return row


def run_origin(task):
    position, macro, returns, paths, seed, cache_dir, replay_check = task
    started = time.perf_counter()
    training_macro, training = training_slice(macro, returns, position)
    cache_path = Path(cache_dir) / f'{position}.pkl'
    if cache_path.exists():
        model = pickle.loads(cache_path.read_bytes())
    else:
        model = calibrate_quadrant_model(training, training_macro, volatility_model='garch', **SETTINGS)
        cache_path.write_bytes(pickle.dumps(model))
    probabilities = model.metadata['historical_regime_probabilities'].loc[:, model.states]
    aligned, weights = _align_probabilities_to_returns(probabilities, training, 0)
    variants = mean_variants(aligned.to_numpy(), weights.to_numpy())
    state_means = np.stack([model.moments[s].mean.to_numpy() for s in model.states])
    np.testing.assert_allclose(variants['current24'], state_means, atol=1e-14, rtol=0)
    posterior = _latest_state_age_posterior(model)
    assert posterior is not None
    initial = posterior.sum(axis=1)
    dep = model.metadata['dependence_fit']
    kwargs = dict(periods=12, paths=paths, random_seed=seed + position * 101,
                  duration_model='semi_markov', joint_macro=True, macro_parameter_uncertainty=False,
                  dynamic_correlation=True, garch=True, return_regime_codes=True,
                  **{key: dep[key] for key in ('garch_alpha', 'garch_beta', 'dcc_alpha', 'dcc_beta', 'dcc_asymmetry')})
    result = simulate_returns(model, **kwargs)
    replay_error = 0.0
    if replay_check:
        changed = copy.deepcopy(model)
        for j, state in enumerate(model.states):
            changed.moments[state] = replace(changed.moments[state], mean=pd.Series(
                variants['unconditional'][j], index=model.assets))
        direct = simulate_returns(changed, **kwargs)
        assert np.array_equal(result.regimes, direct.regimes)
        replay_error = float(np.max(np.abs(direct.returns - shift_paths(
            result.returns, result.regimes, state_means, variants['unconditional']))))
        assert replay_error < 1e-12, 'Mean ablation changed simulated residuals or dynamics'
    actual = returns.iloc[position:position + 12].to_numpy()
    rows = []
    for name, means in variants.items():
        samples = shift_paths(result.returns, result.regimes, state_means, means)
        rows.append(dict(model=name, date=str(returns.index[position].date()), training_months=position,
                         **score_variant(samples, result.regimes, means, initial, actual,
                                         model.assets, np.maximum(aligned.var().to_numpy(), 1e-12))))
    diagnostics = dict(date=str(returns.index[position].date()), training_months=position,
                       last_training_date=str(training.index[-1].date()), state_order=model.states,
                       state_weights=weights.sum().tolist(), first_probabilities=initial.tolist(),
                       means={key: value.tolist() for key, value in variants.items()},
                       replay_checked=replay_check, replay_max_error=replay_error,
                       seconds=time.perf_counter() - started)
    return rows, diagnostics


def paired_stats(candidate, baseline, family=1, length=12):
    pair = pd.concat([candidate, baseline], axis=1).dropna()
    difference = (pair.iloc[:, 0] - pair.iloc[:, 1]).to_numpy()
    n = len(difference)
    rng = np.random.default_rng(2601005)
    starts = rng.integers(n, size=(20000, int(np.ceil(n / length))))
    indexes = ((starts[:, :, None] + np.arange(length)) % n).reshape(20000, -1)[:, :n]
    bootstrap = difference[indexes].mean(axis=1)
    alpha = .05 / family
    return dict(n=n, candidate=float(pair.iloc[:, 0].mean()), baseline=float(pair.iloc[:, 1].mean()),
                delta=float(difference.mean()), improvement_percent=float(-100 * difference.mean() / pair.iloc[:, 1].mean()),
                ci95=np.quantile(bootstrap, [.025, .975]).tolist(),
                adjusted_ci95=np.quantile(bootstrap, [alpha / 2, 1 - alpha / 2]).tolist(),
                comparisons=family, block_length=length)


def summarize(rows, diagnostics, tickers):
    frame = pd.DataFrame(rows).set_index(['model', 'date']).sort_index()
    assert frame.index.is_unique
    result = dict(primary={}, horizons={}, assets={}, secondary={}, periods={}, stability={})
    baseline = frame.loc['unconditional']
    for name in NAMES[1:]:
        candidate = frame.loc[name]
        result['primary'][name] = paired_stats(candidate.mse_1m, baseline.mse_1m, 2)
        result['primary'][name]['block6'] = paired_stats(candidate.mse_1m, baseline.mse_1m, 2, 6)
        result['primary'][name]['block24'] = paired_stats(candidate.mse_1m, baseline.mse_1m, 2, 24)
        result['horizons'][name] = {str(h): paired_stats(candidate[f'mse_{h}m'], baseline[f'mse_{h}m'], 6)
                                    for h in HORIZONS}
        result['assets'][name] = {t: {str(h): paired_stats(candidate[f'mse_{t}_{h}m'], baseline[f'mse_{t}_{h}m'], 24)
                                     for h in HORIZONS} for t in tickers}
        result['secondary'][name] = {f'{metric}_{h}m': paired_stats(candidate[f'{metric}_{h}m'], baseline[f'{metric}_{h}m'], 18)
                                     for metric in ('scaled_mse', 'portfolio_mse', 'portfolio_crps') for h in HORIZONS}
        result['periods'][name] = {}
        for lo, hi in [('2011', '2015'), ('2016', '2019'), ('2020', '2022'), ('2023', '2025')]:
            mask = (baseline.index.str[:4] >= lo) & (baseline.index.str[:4] <= hi)
            b, c = baseline.loc[mask, 'mse_1m'], candidate.loc[mask, 'mse_1m']
            if len(b):
                result['periods'][name][f'{lo}-{hi}'] = dict(n=len(b), improvement_percent=float(100 * (1 - c.mean()/b.mean())))
    result['scores'] = {name: frame.loc[name].mean(numeric_only=True).to_dict() for name in NAMES}
    contrasts = np.array([np.array(d['means']['current24']) - np.array(d['means']['unconditional']) for d in diagnostics])
    # Subtract the ETF's unconditional mean; adjacent-origin flips are highly dependent.
    for j, ticker in enumerate(tickers):
        result['stability'][ticker] = {s: dict(
            fraction_positive=float(np.mean(contrasts[:, k, j] > 0)),
            adjacent_sign_changes=int(np.sum(np.sign(contrasts[1:, k, j]) != np.sign(contrasts[:-1, k, j]))),
            median_annualized_log_mean_difference=float(12 * np.median(contrasts[:, k, j])),
            latest_annualized_log_mean_difference=float(12 * contrasts[-1, k, j]))
            for k, s in enumerate(diagnostics[0]['state_order'])}
    return result


def write_report(output, manifest, summary):
    lines = ['# Regime-dependent expected returns: controlled ablation', '',
             'Only the mean specification changes. Residual volatility, correlation, tails, macro dynamics, regime paths and innovations are fixed to the current calibration at each origin. Production is unchanged.', '',
             f"{manifest['origins']} expanding-window origins; minimum 60 training months; {manifest['paths']:,} common simulated paths per origin. The primary target is expected monthly **log return**, not simple return. All strengths (24 and 96) were fixed before scoring; there is no holdout tuning.", '',
             'Macro-only and unconditional expected log returns coincide: the macro link uses zero-mean innovations. Keeping that link preserves macro risk and dependence; it supplies no separate conditional log-return drift.', '',
             '**Limitations:** the frozen macro data are revised history with an availability lag, not initial-release vintages. The FRED key was unavailable. This repeatedly examined sample is development evidence, not a fresh confirmatory holdout. This isolates mean effects and does not refit residual models separately for each mean specification.', '',
             '| Specification vs unconditional | 1m MSE improvement | Paired Δ MSE | Adjusted 95% interval |',
             '|---|---:|---:|---|']
    for name, entry in summary['primary'].items():
        lines.append(f"| {name} | {entry['improvement_percent']:+.2f}% | {entry['delta']:+.8f} | {entry['adjusted_ci95']} |")
    lines += ['', 'Positive improvement means lower error. Primary intervals use paired circular blocks of 12 months, 20,000 replicates and Bonferroni adjustment for two comparisons. Six- and 24-month block sensitivities are saved.', '',
              '## Horizon sensitivity', '', '| Specification | 1m | 3m | 12m |', '|---|---:|---:|---:|']
    for name in NAMES[1:]:
        lines.append('| ' + name + ' | ' + ' | '.join(f"{summary['horizons'][name][str(h)]['improvement_percent']:+.2f}%" for h in HORIZONS) + ' |')
    lines += ['', 'Longer horizons target cumulative log returns. First-month regime probabilities are integrated exactly from the engine starting posterior; future occupancy is simulated. Horizons have 178/176/167 outcomes in the full run. Horizon-family intervals adjust six comparisons.', '',
              '## Per-ETF monthly MSE improvement', '', '| Specification | ' + ' | '.join(manifest['assets']) + ' |', '|---|' + '---:|' * len(manifest['assets'])]
    for name in NAMES[1:]:
        lines.append('| ' + name + ' | ' + ' | '.join(f"{summary['assets'][name][t]['1']['improvement_percent']:+.2f}%" for t in manifest['assets']) + ' |')
    lines += ['', 'Per-ETF inference adjusts 24 comparisons (two alternatives × four ETFs × three horizons). Raw aggregate MSE weights the four asset losses equally; a training-variance-scaled metric is saved as a sensitivity check.', '',
              '## Fixed portfolio', '', '40% SPY, 30% IEF, 15% GLD, 15% DBC, monthly rebalanced, without leverage, costs, taxes or flows. Forecasts here are cumulative simple portfolio returns. Monte Carlo means and distribution scores use the same shocks for all specifications.', '',
              '| Specification | 1m portfolio MSE improvement | 12m portfolio MSE improvement | 12m CRPS improvement |', '|---|---:|---:|---:|']
    for name in NAMES[1:]:
        entries = summary['secondary'][name]
        lines.append(f"| {name} | {entries['portfolio_mse_1m']['improvement_percent']:+.2f}% | {entries['portfolio_mse_12m']['improvement_percent']:+.2f}% | {entries['portfolio_crps_12m']['improvement_percent']:+.2f}% |")
    lines += ['', '## Audit and reproducibility', '',
              'The original 24-observation means are reproduced algebraically and asserted equal at every origin. Selected origins also compare the mean-shift replay with a direct engine simulation under changed means and identical random seeds. Forecasts are generated before the next realized return is read for scoring.', '',
              'Files: `manifest.json` (data/source hashes and protocol), `forecasts.csv`, `diagnostics.json` (state means, weights and probabilities), `summary.json` (bias, time-period sensitivity, sign stability and adjusted intervals). Cached fitted models support independent-seed replay. Adjustment families are separate and do not cover previous research rounds.', '']
    (output / 'report.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='research/regime_means_2026-10-05/main')
    parser.add_argument('--cache', default='research/regime_means_2026-10-05/models')
    parser.add_argument('--paths', type=int, default=32768)
    parser.add_argument('--seed', type=int, default=20261005)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    output, cache = ROOT / args.output, ROOT / args.cache
    output.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    data = (ROOT / 'research/model_quality_2025/data.pkl').read_bytes()
    macro, returns = pickle.loads(data)
    returns = returns.loc['2006-03-01':].dropna()
    assert returns.columns.tolist() == ['SPY', 'IEF', 'GLD', 'DBC']
    positions = list(range(60, len(returns)))[:args.limit]
    sources = [Path(__file__), ROOT / 'scripts/compare_qarch_forecasts.py', ROOT / 'scripts/compare_model_quality.py']
    sources += list((ROOT / 'src/mc_quadrants').glob('*.py')) + list((ROOT / 'src/mc_quadrants').glob('*.cpp'))
    sources += list((ROOT / 'src/mc_quadrants').glob('*.hpp'))
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    cache_signature = dict(data_sha256=hashlib.sha256(data).hexdigest(), settings=SETTINGS,
                           production_hashes={k: v for k, v in hashes.items() if k.startswith('src/')})
    signature_path = cache / 'signature.json'
    if signature_path.exists():
        assert json.loads(signature_path.read_text()) == cache_signature, 'Stale model cache'
    else:
        signature_path.write_text(json.dumps(cache_signature, indent=2))
    manifest = dict(**cache_signature, source_hashes=hashes, assets=returns.columns.tolist(),
                    origins=len(positions), paths=args.paths, seed=args.seed, strengths=[24, 96],
                    forecast_dates=[str(returns.index[p].date()) for p in positions],
                    macro_vintage=macro.attrs, primary='equal-weight average ETF 1m log-return MSE',
                    baseline='unconditional mean with macro innovation risk retained',
                    no_tuning=True, holdout_status='repeatedly inspected exploratory sample',
                    model_cache=str(cache), weights=WEIGHTS.tolist())
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    rows, diagnostics = [], []
    started = time.perf_counter()
    tasks = [(p, macro, returns, args.paths, args.seed, str(cache), p in (60, 120, 180, 237)) for p in positions]
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn')) as pool:
        for i, (batch, diagnostic) in enumerate(pool.map(run_origin, tasks), 1):
            rows.extend(batch)
            diagnostics.append(diagnostic)
            if i % 10 == 0 or i == len(tasks):
                pd.DataFrame(rows).to_csv(output / 'forecasts.csv', index=False)
                (output / 'diagnostics.json').write_text(json.dumps(diagnostics, indent=2, allow_nan=False))
                print(f'{i}/{len(tasks)} origins, {time.perf_counter() - started:.1f}s', flush=True)
    summary = summarize(rows, diagnostics, returns.columns.tolist())
    manifest['elapsed_seconds'] = time.perf_counter() - started
    assert all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == digest for p, digest in hashes.items())
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    write_report(output, manifest, summary)
    print(json.dumps(summary['primary'], indent=2))


if __name__ == '__main__':
    main()
