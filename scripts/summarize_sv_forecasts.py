#!/usr/bin/env python3
"""Report the SV forecast experiment and its numerical/seed checks."""
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
from scripts.summarize_independent_volatility import paired_comparison  # noqa: E402

NAMES = ['garch', 'gjr', 'egarch', 'qarch', 'figarch']
LABELS = dict(garch='GARCH', gjr='GJR-GARCH', egarch='EGARCH', qarch='QARCH', figarch='FIGARCH', sv='SV–MNTS')
TICKERS = ['SPY', 'IEF', 'GLD', 'DBC']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='research/stochastic_volatility_2026-10-04')
    args = parser.parse_args()
    out = ROOT/args.output
    manifest = json.loads((out/'manifest.json').read_text())
    rows = json.loads((out/'forecasts.json').read_text())
    sv = pd.DataFrame(rows).set_index('date').sort_index()
    baseline = pd.read_csv(Path(manifest['baseline'])/'forecasts.csv').set_index(['model', 'date']).sort_index()
    assert sv.index.is_unique and len(sv) == manifest['origins']
    assert sv.index.tolist() == sorted(manifest['forecast_dates'])
    results = {'manifest': manifest, 'aggregate': {}, 'per_asset': {}, 'checks': {}}
    metrics = ['qlike_assets', 'crps_1m', 'crps_3m', 'crps_12m', 'drawdown_crps_12m', 'loss_brier']
    for metric in metrics:
        results['aggregate'][metric] = {name: paired_comparison(sv[metric], baseline.loc[name, metric].loc[sv.index], 5) for name in NAMES}
    for ticker in TICKERS:
        results['per_asset'][ticker] = {
            name: paired_comparison(sv[f'qlike_{ticker}'], baseline.loc[name, f'qlike_{ticker}'].loc[sv.index], 20) for name in NAMES}
    fits = [fit for row in rows for fit in row['asset_fits']]
    moments = [d for row in rows for d in row['density_diagnostics']]
    results['checks'].update(maximum_posterior_edge_mass=max(f['maximum_posterior_edge_mass'] for f in fits),
        max_garch_replay_error=max(r['baseline_replay_max_error'] for r in rows),
        optimizer_success=all(f['optimizer_success'] for f in fits),
        minimum_converged_starts=min(f['converged_starts'] for f in fits),
        approximate_innovation_second_moment_range=[min(d['second_moment'] for d in moments), max(d['second_moment'] for d in moments)],
        max_abs_approximate_innovation_mean=max(abs(d['mean']) for d in moments))
    results['parameters'] = {}
    for j, ticker in enumerate(TICKERS):
        values = np.array([r['asset_fits'][j]['parameters'] for r in rows])
        results['parameters'][ticker] = dict(median_mu=float(np.median(values[:, 0])), median_phi=float(np.median(values[:, 1])),
            median_stationary_sd=float(np.median(values[:, 2])), fraction_sd_below_005=float(np.mean(values[:, 2]<.05)),
            fraction_phi_at_upper_bound=float(np.mean(values[:, 1]>.9799)))
    for check in ['numerical-check', 'seed-check']:
        path = out/check/'forecasts.csv'
        if path.exists():
            other_manifest = json.loads((path.parent/'manifest.json').read_text())
            for key in ['data_sha256', 'assets', 'settings', 'paths', 'latent_parameters', 'source_hashes']:
                assert other_manifest[key] == manifest[key], f'{check} changed {key}'
            extra = pd.read_csv(path).set_index('date').sort_index()
            match = sv.loc[extra.index]
            results['checks'][check] = {'origins': len(extra), 'metrics': {}}
            for metric in ['qlike_assets', *[f'qlike_{t}' for t in TICKERS], 'crps_1m', 'crps_12m']:
                delta = extra[metric]-match[metric]
                results['checks'][check]['metrics'][metric] = dict(mean_difference=float(delta.mean()),
                    max_absolute_difference=float(delta.abs().max()), main_mean=float(match[metric].mean()), check_mean=float(extra[metric].mean()))
            if check == 'seed-check':
                assert extra.index.equals(sv.index)
                np.testing.assert_allclose(extra['qlike_assets'], sv['qlike_assets'], rtol=0, atol=1e-12)
    results['coverage'] = {name: {metric: float((sv if name == 'sv' else baseline.loc[name].loc[sv.index])[metric].mean())
        for metric in ['coverage90', 'breach05', 'width90']} for name in [*NAMES, 'sv']}
    results['source_hashes'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__), out/'forecasts.csv', Path(manifest['baseline'])/'forecasts.csv']}
    (out/'comparison.json').write_text(json.dumps(results, indent=2, allow_nan=False))
    lines = ['# Stochastic-volatility forecasting evaluation', '',
        'Mean-reverting log-SV with MNTS return innovations was fitted independently for each ETF and evaluated against the five previously fitted models. The same frozen data, macro model, correlations, return shocks and forecast dates are retained.', '',
        f'SV minus independent GARCH mean volatility loss: {results["aggregate"]["qlike_assets"]["garch"]["delta"]:+.5f} (negative favors SV). Portfolio CRPS changes versus GARCH: one month {100*results["aggregate"]["crps_1m"]["garch"]["delta"]/results["aggregate"]["crps_1m"]["garch"]["reference"]:+.2f}%; twelve months {100*results["aggregate"]["crps_12m"]["garch"]["delta"]/results["aggregate"]["crps_12m"]["garch"]["reference"]:+.2f}% (positive means higher forecast error).', '',
        '## Volatility forecast scores', '',
        'One-month QLIKE, lower is better. Bold identifies the lowest sample average per ETF, not a statistically established winner.', '',
        '| ETF | '+' | '.join(LABELS[n] for n in [*NAMES, 'sv'])+' |', '|---|'+'---:|'*6]
    for ticker in TICKERS:
        values = [float(baseline.loc[name, f'qlike_{ticker}'].loc[sv.index].mean()) for name in NAMES]+[float(sv[f'qlike_{ticker}'].mean())]
        best = int(np.argmin(values))
        lines.append('| '+ticker+' | '+' | '.join(f'**{v:.5f}**' if i == best else f'{v:.5f}' for i, v in enumerate(values))+' |')
    lines += ['', '| Reference | Mean SV minus reference QLIKE | Pointwise 95% interval | Family-adjusted 95% interval |', '|---|---:|---|---|']
    for name in NAMES:
        r = results['aggregate']['qlike_assets'][name]
        lo, hi = r['ci95']
        flo, fhi = r['family_ci95']
        lines.append(f'| {LABELS[name]} | {r["delta"]:+.5f} | [{lo:+.5f}, {hi:+.5f}] | [{flo:+.5f}, {fhi:+.5f}] |')
    lines += ['', 'Negative differences favor SV. Primary intervals adjust for five model comparisons; per-ETF intervals in `comparison.json` adjust for 20 comparisons. Paired circular block bootstrap: 12-month blocks, 4,000 replicates for pointwise intervals and 20,000 for Bonferroni-adjusted intervals. Six-month block sensitivity is also saved.', '',
        '## Portfolio forecasts', '',
        '| Model | 1-month CRPS | 3-month CRPS | 12-month CRPS | 90% coverage | Below 5th percentile |', '|---|---:|---:|---:|---:|---:|']
    for name in [*NAMES, 'sv']:
        frame = sv if name == 'sv' else baseline.loc[name].loc[sv.index]
        lines.append(f'| {LABELS[name]} | {frame.crps_1m.mean():.6f} | {frame.crps_3m.mean():.6f} | {frame.crps_12m.mean():.6f} | {frame.coverage90.mean():.2%} | {frame.breach05.mean():.2%} |')
    lines += ['', 'Portfolio: 40% SPY, 30% IEF, 15% GLD, 15% DBC; monthly rebalance, no leverage/costs/taxes/cash flows. CRPS measures predictive distribution error, not returns. The complete sample has 178 one-month, 176 three-month and 167 twelve-month outcomes.', '',
        '## Specification and inference', '',
        'For regime-normalized residual x: x_t = exp(h_t/2) z_t, and h_(t+1) = mu + phi(h_t−mu) + tau eta_(t+1). The return innovation z follows the existing fitted MNTS marginal; eta is standard Gaussian, independent of return shocks and other ETFs’ volatility shocks. tau = stationary_sd * sqrt(1−phi²).', '',
        '- One set of (mu, phi, stationary_sd) per ETF, shared across macro regimes. Fixed bounds: mu in [−2,2], phi in [0,0.98], stationary_sd in [0,1.5]. No volatility leverage effect or common volatility factor.',
        '- Four-start L-BFGS-B maximizes the approximate marginal MNTS likelihood in each training window. The first 12 months initialize the filter but do not enter the scored training likelihood. Other benchmark models were fitted with Gaussian QML, so inference methods differ as well as volatility dynamics.',
        f'- The filtering grid has {manifest["grid_points"]} points over [−7,7] stationary standard deviations. Gaussian transition probabilities integrate over grid cells. Filtering uses only returns available at the origin; no future-data smoothing.',
        f'- MNTS emission densities integrate conditional normal densities over {manifest["density_nodes"]} empirical subordinator quantiles from {manifest["density_sample_size"]:,} fixed-seed exact draws per regime. Log densities are interpolated on an asinh grid. This is numerical likelihood integration, not an exact analytic MNTS likelihood.',
        '- Forecasts sample the final filtered volatility state and then continuous AR(1) volatility innovations. Analytic one-month variance integrates exp(h), including the lognormal correction; it does not exponentiate mean log variance.',
        '- Latent-state and future shock uncertainty are included. Parameters remain point estimates. Marginal ETF filtering does not infer a joint posterior over latent volatilities using cross-asset returns.',
        '- Shared return innovations preserve the pre-existing MNTS/ADCC dependence model. A separate RNG supplies volatility shocks. Saved GARCH forecasts are replayed on every origin to verify common return draws and scoring.', '',
        'Numerical-filtering reference: [Langrock, MacDonald and Zucchini, 2012](https://www.sciencedirect.com/science/article/abs/pii/S0927539811000661). SV inference and extensions: [Hosszejni and Kastner, 2021](https://www.jstatsoft.org/article/view/v100i12).', '',
        '## Diagnostics', '',
        f'- All selected fits report convergence: {results["checks"]["optimizer_success"]}. Minimum converged starts per fit: {results["checks"]["minimum_converged_starts"]}/4. Convergence is local, not proof of a global optimum.',
        f'- Largest filtered probability on the two boundary grid points: {results["checks"]["maximum_posterior_edge_mass"]:.3g}.',
        f'- Largest saved-GARCH replay CRPS difference: {results["checks"]["max_garch_replay_error"]:.3g}.',
        f'- Approximate emission-law second moments: {results["checks"]["approximate_innovation_second_moment_range"]}. Forecasts use the original unit-variance MNTS sampler.', '',
        '| ETF | Median persistence phi | Median stationary log-variance SD | Fits with SD < 0.05 |', '|---|---:|---:|---:|']
    for ticker, r in results['parameters'].items():
        lines.append(f'| {ticker} | {r["median_phi"]:.3f} | {r["median_stationary_sd"]:.3f} | {r["fraction_sd_below_005"]:.1%} |')
    lines += ['', 'Persistence is weakly identified when the volatility-state SD is near zero; its median should not be interpreted as evidence of persistent volatility in those fits.']
    for check in ['numerical-check', 'seed-check']:
        if check in results['checks']:
            r = results['checks'][check]
            lines += ['', f'{check}: {r["origins"]} origins. Mean change in primary QLIKE: {r["metrics"]["qlike_assets"]["mean_difference"]:+.6f}; largest monthly change: {r["metrics"]["qlike_assets"]["max_absolute_difference"]:.6f}.']
            if check == 'numerical-check':
                lines += ['This check uses every sixth origin, a 121-point state grid, 2,048 mixture nodes and 1,048,576 subordinator draws per regime. It tests both state discretization and emission-density integration. Individual monthly differences remain possible; it is not a proof of exact likelihood evaluation.']
    lines += ['', '## Limits and reproduction', '',
        'Latest-revised macro observations are shifted for availability, not historical release vintages. The dataset is a short monthly sample and has already been used for model development. These results concern this mean-reverting SV specification and its numerical marginal estimator; they do not settle the merits of leverage-SV, factor-SV, other priors, or daily-data estimation. Production defaults and UI options are unchanged.', '',
        '```bash', 'python scripts/compare_sv_forecasts.py --workers 4',
        'python scripts/compare_sv_forecasts.py --workers 2 --stride 6 --grid-points 121 --density-nodes 2048 --density-sample-size 1048576 --output research/stochastic_volatility_2026-10-04/numerical-check',
        'python scripts/compare_sv_forecasts.py --workers 4 --baseline research/independent_volatility_models_2026-10-04/seed-check --output research/stochastic_volatility_2026-10-04/seed-check',
        'python scripts/summarize_sv_forecasts.py', '```', '',
        'Requires the optional research dependencies and the frozen previous benchmark. Outputs: `manifest.json`, `forecasts.json`, `forecasts.csv`, `summary.json`, `comparison.json`, and numerical/seed-check directories.', '']
    (out/'report.md').write_text('\n'.join(lines))
    print(json.dumps(results['aggregate']['qlike_assets'], indent=2))
    print(out/'report.md')


if __name__ == '__main__':
    main()
