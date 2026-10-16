#!/usr/bin/env python3
"""Audit seed sensitivity and quantify long-horizon mean-specification sensitivity."""
from __future__ import annotations

import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))
from mc_quadrants.calibration import _align_probabilities_to_returns, calibrate_quadrant_model  # noqa: E402
from mc_quadrants.simulation import simulate_returns  # noqa: E402
from scripts.compare_regime_means import (  # noqa: E402
    HORIZONS,
    NAMES,
    SETTINGS,
    WEIGHTS,
    mean_variants,
    paired_stats,
    shift_paths,
)

OUT = ROOT / 'research/regime_means_2026-10-05'


def main():
    manifests = [json.loads((OUT / folder / 'manifest.json').read_text()) for folder in ('main', 'seed-check')]
    main_manifest, check_manifest = manifests
    for key in ('data_sha256', 'source_hashes', 'forecast_dates', 'settings', 'assets'):
        assert main_manifest[key] == check_manifest[key], key
    assert main_manifest['seed'] != check_manifest['seed']
    for path, digest in main_manifest['source_hashes'].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest, path
    frames = [pd.read_csv(OUT / folder / 'forecasts.csv').set_index(['model', 'date']).sort_index()
              for folder in ('main', 'seed-check')]
    a, b = frames
    assert a.index.equals(b.index) and len(a) == 178 * 3 and a.index.is_unique
    for h, count in zip(HORIZONS, (178, 176, 167)):
        assert all(a.loc[name, f'mse_{h}m'].notna().sum() == count for name in NAMES)
    np.testing.assert_array_equal(a.mse_1m, b.mse_1m)
    summaries = [json.loads((OUT / folder / 'summary.json').read_text()) for folder in ('main', 'seed-check')]
    checks = dict(exact_1m_mse_identical_across_seeds=True, rows=len(a), horizon_seed_check={}, block24={},
                  source_hashes_verified=True, summary_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    for name in NAMES[1:]:
        checks['horizon_seed_check'][name] = {str(h): {
            'main_improvement_percent': summaries[0]['horizons'][name][str(h)]['improvement_percent'],
            'second_seed_improvement_percent': summaries[1]['horizons'][name][str(h)]['improvement_percent'],
            'second_seed_adjusted_ci95': summaries[1]['horizons'][name][str(h)]['adjusted_ci95'],
        } for h in HORIZONS}
        checks['block24'][name] = {str(h): paired_stats(a.loc[name, f'mse_{h}m'],
                                                        a.loc['unconditional', f'mse_{h}m'], 6, 24)
                                   for h in HORIZONS}
    diagnostics = json.loads((OUT / 'main/diagnostics.json').read_text())
    checks['engine_replay_origins'] = [d['date'] for d in diagnostics if d['replay_checked']]
    checks['engine_replay_max_error'] = max(d['replay_max_error'] for d in diagnostics)
    assert len(checks['engine_replay_origins']) == 4 and checks['engine_replay_max_error'] < 1e-12

    # Long-horizon scenario sensitivity, explicitly not an observed forecast test.
    data = (ROOT / 'research/model_quality_2025/data.pkl').read_bytes()
    assert hashlib.sha256(data).hexdigest() == main_manifest['data_sha256']
    macro, returns = pickle.loads(data)
    returns = returns.loc['2006-03-01':].dropna()
    model = calibrate_quadrant_model(returns, macro.loc[:returns.index[-1]], volatility_model='garch', **SETTINGS)
    aligned, memberships = _align_probabilities_to_returns(
        model.metadata['historical_regime_probabilities'].loc[:, model.states], returns, 0)
    variants = mean_variants(aligned.to_numpy(), memberships.to_numpy())
    old_means = np.stack([model.moments[s].mean.to_numpy() for s in model.states])
    np.testing.assert_allclose(variants['current24'], old_means, atol=1e-14)
    dep = model.metadata['dependence_fit']
    result = simulate_returns(model, periods=360, paths=8192, random_seed=20261007,
                              duration_model='semi_markov', joint_macro=True, macro_parameter_uncertainty=False,
                              dynamic_correlation=True, garch=True, return_regime_codes=True,
                              **{key: dep[key] for key in ('garch_alpha', 'garch_beta', 'dcc_alpha', 'dcc_beta', 'dcc_asymmetry')})
    sensitivity = dict(calibration_end=str(returns.index[-1].date()), paths=8192, seed=20261007,
                       scope='Nominal wealth sensitivity, not forecasting evidence; fixed calibration, no parameter uncertainty',
                       weights=WEIGHTS.tolist(), initial_wealth=1, models={})
    for name in NAMES:
        samples = shift_paths(result.returns, result.regimes, old_means, variants[name])
        portfolio = np.expm1(samples) @ WEIGHTS
        log_wealth = np.log1p(portfolio).cumsum(axis=0)
        sensitivity['models'][name] = {}
        for years in (10, 30):
            quantiles = np.exp(np.quantile(log_wealth[years * 12 - 1], [.05, .5, .95]))
            assert np.isfinite(quantiles).all()
            sensitivity['models'][name][str(years)] = dict(zip(('q05', 'median', 'q95'), quantiles.tolist()))
    (OUT / 'checks.json').write_text(json.dumps(checks, indent=2, allow_nan=False))
    (OUT / 'wealth_sensitivity.json').write_text(json.dumps(sensitivity, indent=2, allow_nan=False))
    report = (OUT / 'main/report.md').read_text()
    report += '\n## Independent-seed and longer-block checks\n\n'
    report += 'Repeated all 178 origins with seed 20261006 and 65,536 paths, versus seed 20261005 and 32,768 paths. Monthly ETF point scores are exactly identical because those forecasts are analytic. The direct-engine replay matched the mean-shift method within ' + f"{checks['engine_replay_max_error']:.2g}.\n\n"
    report += '| Specification | 12m MSE improvement, main | Second seed | Δ MSE adjusted interval, 24-month blocks |\n|---|---:|---:|---|\n'
    for name in NAMES[1:]:
        entry = checks['horizon_seed_check'][name]['12']
        report += f"| {name} | {entry['main_improvement_percent']:+.2f}% | {entry['second_seed_improvement_percent']:+.2f}% | {checks['block24'][name]['12']['adjusted_ci95']} |\n"
    report += '\n## Bias and stability\n\nMean one-month forecast error in basis points of log return; positive means overprediction.\n\n'
    report += '| Specification | ' + ' | '.join(main_manifest['assets']) + ' |\n|---|' + '---:|' * 4 + '\n'
    for name in NAMES:
        report += '| ' + name + ' | ' + ' | '.join(f"{a.loc[name, f'bias_{ticker}_1m'].mean() * 10000:+.1f}" for ticker in main_manifest['assets']) + ' |\n'
    report += '\nSome regime contrasts retain their sign, others reverse as estimates update; sign persistence alone does not establish predictive value. Full state-level contrasts, adjacent-origin sign changes and subperiod scores are saved in `main/summary.json`. No per-ETF model was selected after seeing these scores.\n'
    report += '\n## Long-term wealth sensitivity\n\n'
    report += 'Fit through December 2025; 8,192 common 30-year paths; same fixed portfolio and risk dynamics. Nominal terminal wealth per unit invested. This is a sensitivity exercise without realized future outcomes, not evidence that any scenario is more accurate. It omits parameter uncertainty and real purchasing-power adjustment.\n\n'
    report += '| Specification | Median 10-year wealth | Median 30-year wealth | 30-year 5th–95th percentile |\n|---|---:|---:|---|\n'
    for name in NAMES:
        s = sensitivity['models'][name]
        report += f"| {name} | {s['10']['median']:.3f} | {s['30']['median']:.3f} | {s['30']['q05']:.3f}–{s['30']['q95']:.3f} |\n"
    report += '\n## Interpretation\n\n'
    report += 'The current regime-dependent means do not establish a monthly forecasting advantage; their point error is higher for all four ETFs. Stronger shrinkage moves forecasts toward the unconditional baseline and reduces the gap. Current-regime means increase aggregate 12-month MSE by about 8.05%; the six-comparison-adjusted interval excludes zero using both 12- and 24-month blocks and the result survives the second seed. Stronger shrinkage does not establish a significant difference from unconditional means. This supports considering simpler means, while keeping regimes for risk dynamics. The revised macro history and repeatedly examined sample limit the strength of this finding. It does not establish that regimes are universally useless or authorize a production default change.\n'
    (OUT / 'report.md').write_text(report)
    print(json.dumps(dict(seed_checks=checks['horizon_seed_check'], block24=checks['block24'], wealth=sensitivity['models']), indent=2))


if __name__ == '__main__':
    main()
