#!/usr/bin/env python3
"""Compare independently fitted ETF volatility forecasts with frozen pooled fits."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.compare_model_quality import paired_ci  # noqa: E402

NAMES = ('garch', 'gjr', 'egarch', 'qarch')
TICKERS = ('SPY', 'IEF', 'GLD', 'DBC')
LABELS = {'garch': 'GARCH', 'gjr': 'GJR-GARCH', 'egarch': 'EGARCH', 'qarch': 'QARCH', 'figarch': 'FIGARCH'}


def family_interval(values, comparisons):
    """Bonferroni-adjusted percentile interval using the same paired blocks."""
    values = np.asarray(values)
    count = len(values)
    rng = np.random.default_rng(481)
    starts = rng.integers(count, size=(20000, int(np.ceil(count/12))))
    indexes = ((starts[:, :, None]+np.arange(12)) % count).reshape(20000, -1)[:, :count]
    alpha = .05/comparisons
    return np.quantile(values[indexes].mean(axis=1), [alpha/2, 1-alpha/2]).tolist()


def paired_comparison(independent, reference, comparisons):
    pair = pd.concat([independent, reference], axis=1).dropna()
    delta = pair.iloc[:, 0]-pair.iloc[:, 1]
    return dict(independent=float(pair.iloc[:, 0].mean()), reference=float(pair.iloc[:, 1].mean()),
                delta=float(delta.mean()), ci95=paired_ci(delta), ci95_block6=paired_ci(delta, length=6),
                family_ci95=family_interval(delta, comparisons), family_comparisons=comparisons, n=len(delta))


def analyze(independent_path, pooled_path):
    im = json.loads((independent_path/'manifest.json').read_text())
    pm = json.loads((pooled_path/'manifest.json').read_text())
    assert im['pooled_parameters'] is False and pm['pooled_parameters'] is True
    for key in ['data_sha256', 'forecast_start', 'forecast_end', 'origins', 'paths', 'seed', 'settings', 'weights']:
        assert im[key] == pm[key], f'Non-comparable {key}'
    for path, digest in pm['source_hashes'].items():
        if path.startswith('src/'):
            assert im['source_hashes'][path] == digest, f'Production source changed: {path}'
    independent = pd.read_csv(independent_path/'forecasts.csv').set_index(['model', 'date']).sort_index()
    pooled = pd.read_csv(pooled_path/'forecasts.csv').set_index(['model', 'date']).sort_index()
    names = tuple(im['models'])
    assert independent.loc[list(NAMES)].sort_index().index.equals(pooled.index)
    assert independent.index.is_unique
    assert len(independent) == im['origins']*len(names)
    result = dict(scope='ETF-specific volatility coefficients; shared across regimes; other engine components fixed',
                  independent_source=str(independent_path), pooled_source=str(pooled_path),
                  manifest=im, per_asset={}, aggregate={}, diagnostics={})
    for ticker in TICKERS:
        metric = f'qlike_{ticker}'
        entries = {}
        for name in names:
            current = independent.loc[name, metric]
            entries[name] = dict(
                vs_same_pooled_model=paired_comparison(current, pooled.loc[name, metric], 16) if name in NAMES else None,
                vs_independent_garch=paired_comparison(current, independent.loc['garch', metric], 4*(len(names)-1)),
                crps_vs_same_pooled_model=paired_comparison(independent.loc[name, f'crps_{ticker}'], pooled.loc[name, f'crps_{ticker}'], 16) if name in NAMES else None)
        pairwise = {}
        for i, a in enumerate(names):
            for b in names[i+1:]:
                pairwise[f'{b}_minus_{a}'] = paired_comparison(independent.loc[b, metric], independent.loc[a, metric], 2*len(names)*(len(names)-1))
        result['per_asset'][ticker] = dict(models=entries, pairwise=pairwise,
            lowest_independent_qlike=min(names, key=lambda name: entries[name]['vs_independent_garch']['independent']),
            lowest_pooled_qlike=min(NAMES, key=lambda name: entries[name]['vs_same_pooled_model']['reference']))
    for metric in ['qlike_assets', 'crps_1m', 'crps_3m', 'crps_12m', 'drawdown_crps_12m']:
        result['aggregate'][metric] = {name: paired_comparison(independent.loc[name, metric], pooled.loc[name, metric], 4) for name in NAMES}
    if 'figarch' in names:
        result['figarch_vs_independent_garch'] = {metric: paired_comparison(independent.loc['figarch', metric], independent.loc['garch', metric], 1)
            for metric in ['qlike_assets', 'crps_1m', 'crps_3m', 'crps_12m', 'drawdown_crps_12m']}
    result['diagnostics']['max_selected_search_step'] = independent.groupby(level=0).search_step.max().to_dict()
    result['diagnostics']['origin_model_rows'] = len(independent)
    raw_rows = json.loads((independent_path/'forecasts.json').read_text())
    if 'figarch' in names:
        fractional = np.array([row['parameters'][2] for row in raw_rows if row['model'] == 'figarch'])
        result['diagnostics']['figarch_fractional_memory'] = {ticker: dict(
            median_d=float(np.median(fractional[:, j])), fraction_d_below_001=float(np.mean(fractional[:, j] < .01)))
            for j, ticker in enumerate(TICKERS)}
    selected_fits = [fit for row in raw_rows for fit in row['asset_fits']]
    assert all(fit.get('optimizer_success', True) for fit in selected_fits)
    assert all(fit['search_step'] < 1e-6 for fit in selected_fits)
    seed_file = independent_path/'seed-check/forecasts.csv'
    if seed_file.exists():
        seed_manifest = json.loads((seed_file.parent/'manifest.json').read_text())
        for key in ['data_sha256', 'models', 'forecast_start', 'forecast_end', 'origins', 'paths', 'settings', 'source_hashes']:
            assert seed_manifest[key] == im[key], f'Seed check has different {key}'
        seed = pd.read_csv(seed_file).set_index(['model', 'date']).sort_index()
        assert seed.index.equals(independent.index)
        for metric in ['qlike_assets', *[f'qlike_{ticker}' for ticker in TICKERS]]:
            np.testing.assert_array_equal(seed[metric], independent[metric])
        result['seed_check'] = dict(seed=seed_manifest['seed'], rows=len(seed), analytic_scores_identical=True,
            mean_scores={name: {metric: float(seed.loc[name, metric].mean()) for metric in ['crps_1m', 'crps_12m']} for name in names})
    result['diagnostics']['input_hashes'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [independent_path/'forecasts.csv', pooled_path/'forecasts.csv', Path(__file__)]}
    return result


def write_report(result, out):
    names = tuple(result['manifest']['models'])
    lines = ['# Independent ETF volatility fits versus pooled fits', '',
        'Each model was refitted separately for each ETF at every expanding-window forecast origin. The original four models change volatility coefficient pooling only; regime baselines, macro forecasts, tail laws, correlation dynamics and simulation shocks are shared with the previous comparison.'
        + (' FIGARCH is an additional independent-fit model; there is no pooled FIGARCH baseline.' if 'figarch' in names else ''), '',
        '## One-month volatility forecast scores', '',
        'Lower QLIKE is better. Bold marks the lowest independent-fit sample average in each row; it does not establish a statistically superior model.', '',
        '| ETF | '+' | '.join(LABELS[n] for n in names)+' | Best pooled model |', '|---|'+'---:|'*len(names)+'---|']
    for ticker, asset in result['per_asset'].items():
        values = []
        for name in names:
            v = asset['models'][name]['vs_independent_garch']['independent']
            values.append(f'**{v:.5f}**' if name == asset['lowest_independent_qlike'] else f'{v:.5f}')
        lines.append('| '+ticker+' | '+' | '.join(values)+' | '+LABELS[asset['lowest_pooled_qlike']]+' |')
    lines += ['', '## Does fitting separately improve forecasting?', '',
        'Negative differences mean independent fitting improves on the same model with pooled coefficients. The adjusted intervals account for the 16 ETF/model comparisons in this table.', '',
        '| ETF | Model | Independent minus pooled QLIKE | Pointwise 95% interval | Family-adjusted 95% interval |', '|---|---|---:|---|---|']
    for ticker, asset in result['per_asset'].items():
        for name in NAMES:
            v = asset['models'][name]['vs_same_pooled_model']
            lo, hi = v['ci95']
            flo, fhi = v['family_ci95']
            lines.append(f'| {ticker} | {LABELS[name]} | {v["delta"]:+.5f} | [{lo:+.5f}, {hi:+.5f}] | [{flo:+.5f}, {fhi:+.5f}] |')
    lines += ['', '## Portfolio distribution forecast scores', '',
        'CRPS reduction relative to the same model with pooled coefficients. Positive percentages indicate improvement in forecast error, not investment returns.', '',
        '| Model | 1 month | 3 months | 12 months |', '|---|---:|---:|---:|']
    for name in NAMES:
        improvements = [-100*result['aggregate'][m][name]['delta']/result['aggregate'][m][name]['reference'] for m in ['crps_1m', 'crps_3m', 'crps_12m']]
        lines.append(f'| {LABELS[name]} | {improvements[0]:+.3f}% | {improvements[1]:+.3f}% | {improvements[2]:+.3f}% |')
    if 'figarch' in names:
        lines += ['', '## FIGARCH', '',
            'FIGARCH(1,d,1) is estimated through its ARCH-infinity expansion, truncated at 1,000 monthly lags. It has a freely fitted positive intercept; it is not forced to have unit long-run variance. Missing presample squared normalized residuals are set to one, and historical memory is carried into forecasts.', '',
            'The untruncated specification with positive intercept and 0<d<1 does not have a finite unconditional variance. The finite truncation is a numerical forecasting approximation, not a cure for the theoretical long-horizon limitation. With only 60–237 training months, the fractional-memory parameter and presample assumptions are weakly identified.', '',
            'The recurrence and constraints follow the [arch FIGARCH documentation](https://arch.readthedocs.io/en/latest/univariate/generated/arch.univariate.FIGARCH.html). Coefficients use multistart constrained SLSQP with the same training QML loss and 12-month scoring warmup as the other models.', '',
            '| ETF | FIGARCH minus independent GARCH QLIKE | Pointwise 95% interval | Family-adjusted 95% interval |', '|---|---:|---|---|']
        for ticker, asset in result['per_asset'].items():
            v = asset['models']['figarch']['vs_independent_garch']
            lo, hi = v['ci95']
            flo, fhi = v['family_ci95']
            lines.append(f'| {ticker} | {v["delta"]:+.5f} | [{lo:+.5f}, {hi:+.5f}] | [{flo:+.5f}, {fhi:+.5f}] |')
        gains = [-100*result['figarch_vs_independent_garch'][m]['delta']/result['figarch_vs_independent_garch'][m]['reference'] for m in ['crps_1m', 'crps_3m', 'crps_12m']]
        lines += ['', f'FIGARCH portfolio CRPS improvement versus independently fitted GARCH: 1 month {gains[0]:+.3f}%; 3 months {gains[1]:+.3f}%; 12 months {gains[2]:+.3f}%.', '']
        lines += ['Fitted fractional-memory parameter d:', '', '| ETF | Median d | Fraction of fits with d < 0.01 |', '|---|---:|---:|']
        for ticker, values in result['diagnostics']['figarch_fractional_memory'].items():
            lines.append(f'| {ticker} | {values["median_d"]:.3f} | {values["fraction_d_below_001"]:.1%} |')
    rankings = '; '.join(ticker+': '+LABELS[asset['lowest_independent_qlike']] for ticker, asset in result['per_asset'].items())
    pointwise, adjusted = [], []
    for ticker, asset in result['per_asset'].items():
        for name in NAMES:
            contrast = asset['models'][name]['vs_same_pooled_model']
            if contrast['ci95'][1] < 0:
                pointwise.append(ticker+' '+LABELS[name])
            if contrast['family_ci95'][1] < 0:
                adjusted.append(ticker+' '+LABELS[name])
    lines += ['', '## Interpretation', '',
        'Lowest independent-fit average QLIKE: '+rankings+'. These are sample rankings rather than evidence of a unique winner.', '',
        'Independent-fitting improvements over the same pooled model with pointwise intervals excluding zero: '+(', '.join(pointwise) or 'none')+'. After adjustment for all 16 comparisons: '+(', '.join(adjusted) or 'none')+'.', '']
    if 'figarch' in names:
        worse = [ticker for ticker, asset in result['per_asset'].items() if asset['models']['figarch']['vs_independent_garch']['delta'] > 0]
        lines += ['FIGARCH has higher average QLIKE than independent GARCH for: '+(', '.join(worse) or 'none')+'.', '']
    if 'seed_check' in result:
        lines += [f'A second complete forecast seed ({result["seed_check"]["seed"]}) produced {result["seed_check"]["rows"]} model/origin rows. All analytic one-month QLIKE scores are identical to the first seed. Simulated CRPS results for the second seed are in `comparison.json` and `seed-check/`.', '']
    lines += ['', '## Method and limits', '',
        '- 178 monthly forecast origins, March 2011–December 2025; initial training history starts March 2006. SPY/IEF/GLD/DBC, 8,192 paths per origin. 167 complete twelve-month outcomes.',
        '- Portfolio weights 40%/30%/15%/15%, monthly rebalance, no leverage, borrowing costs, taxes or cash flows.',
        '- Each ETF receives its own GARCH, GJR, EGARCH and QARCH coefficients. Coefficients remain shared across regimes for that ETF; this is not a separate volatility fit for every ETF/regime combination.',
        '- Every forecast uses only earlier data. The same unpenalized Gaussian QML objective, 12-month warmup, parameter bounds and multistart optimizer are used. Model families are evaluated across all dates; sample winners are descriptive, not a hindsight-selected trading strategy.',
        '- All models retain normalized volatility memory across regime changes and use filtered initialization. MNTS moment centering and EGARCH moment-domain bounds are asset-specific.',
        '- QLIKE uses exact first-step marginal variance, including macro innovation variance and regime-mixture uncertainty. The squared monthly return error proxy is noisy.',
        '- Paired circular block bootstrap: 12-month blocks, 4,000 draws for pointwise intervals; six-month sensitivity in JSON. Bonferroni-adjusted intervals use 20,000 bootstrap draws. They are approximate and do not remove uncertainty caused by repeated model development on this sample.',
        f'- `comparison.json` includes all independent pairwise ETF/model contrasts (family size {2*len(names)*(len(names)-1)}), contrasts to independently fitted GARCH (family size {4*(len(names)-1)}), per-asset CRPS, and independent-versus-pooled results.',
        '- Latest-revised macro data are availability-shifted one month, not historical release vintages. This is not a point-in-time backtest. A single monthly ETF history is limited evidence for generalizing or choosing permanent per-ETF model defaults.',
        '- Difficult independent EGARCH fits receive constrained SLSQP refinement based solely on training convergence. All selected coordinate-search fits have final steps below 0.000001; all selected SLSQP fits report convergence. Local convergence does not guarantee a global optimum.',
        '- GJR, EGARCH and FIGARCH remain research implementations. No production simulation defaults or UI selectors are changed.', '',
        '## Reproduce', '', '```bash',
        '.venv/bin/python scripts/compare_volatility_models.py --independent '+('--include-figarch ' if 'figarch' in names else '')+'--workers 4',
        '.venv/bin/python scripts/summarize_independent_volatility.py', '```', '',
        'Outputs: `manifest.json`, `forecasts.csv`, `forecasts.json`, `summary.json`, and `comparison.json`. The manifests preserve dataset and source hashes. The original pooled results remain in `research/four_volatility_models_2026-10-04`.', '']
    (out/'report.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--independent', default='research/independent_volatility_models_2026-10-04')
    parser.add_argument('--pooled', default='research/four_volatility_models_2026-10-04')
    args = parser.parse_args()
    independent, pooled = ROOT/args.independent, ROOT/args.pooled
    result = analyze(independent, pooled)
    (independent/'comparison.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    write_report(result, independent)
    print(json.dumps({ticker: {key: value for key, value in asset.items() if key.startswith('lowest_')}
                      for ticker, asset in result['per_asset'].items()}, indent=2))
    print(independent/'report.md')


if __name__ == '__main__':
    main()
