# Economic return model and reality fixes

Follow-up (2026-10-16): [persistent macro news and conditional volatility](macro-stress-model.md) are implemented as optional candidates, with a new chronological comparison and parameter-bank checks.

Updated 2026-10-13. Economic return assumptions remain **opt-in and experimental**. The return changes improve important historical scores, but do not yet reproduce every stress episode. CPI accounting and explicit monthly-close drawdown labels apply independently of that switch.

## Use

Enable **Economic return model (experimental)** in the quadrant settings, with joint macro paths. Expand **Economic return assumptions** for persistent-yield and equity-response choices. The advanced JSON field accepts the asset-specific mappings below. Saved configurations and API requests retain these settings.

```python
economic_returns = {}  # anchors + adaptive persistent yields + fitted equity macro responses
# Alternatives:
economic_returns = {"rate_stress": False}  # anchors alone
economic_returns = {"yield_model": "random_walk"}  # no-change yield forecast
# Previous assumptions, for comparison:
economic_returns = {"yield_model": "legacy", "equity_rate_mode": "inflation_gate"}
# Disable economic return assumptions:
economic_returns = None
```

Rebuild the native library with `./scripts/build_native.sh`. Its ABI is now 11; an older binary is rejected rather than silently using the old inflation calculations. Economic repricing uses the ordinary portfolio ledger and Python macro generator for persistent yields, with native asset-return sampling when available. Default chunks are 2,000 paths for this mode; an explicit chunk size is respected. It can consume more time and memory than the compact baseline.

## CPI and purchasing power

Trailing year-over-year inflation is not a monthly price change. With observed `cpi_monthly_log`, the joint process now fits monthly log CPI changes with a regularized AR(1), regime-conditioned means, and innovations correlated with the other macro shocks. It derives the public year-over-year series from a rolling twelve-month price history:

```
log(1 + YoY[t]) = sum(monthly_log_CPI[t-11:t+1])
```

All real-wealth, spending-indexation, and validation calculations recover monthly price changes from this identity. Their legacy annual-rate interface receives `expm1(12 * monthly_log_CPI)`, so the monthly deflator is exactly `exp(-monthly_log_CPI)`. Financing and regime classification still consume trailing annual inflation. Both native compact reporting and ordinary accounting implement the same identity.

Missing monthly CPI observations are excluded from innovation fitting; no interpolation is presented as observed data. If twelve consecutive initial monthly changes are unavailable, the initial annual change is allocated evenly over twelve months and a warning is emitted. A missing latest monthly change uses that same initialization assumption. With only a YoY input column, the older YoY forecasting process remains, with exact accounting conditional on this assumed initial history; monthly CPI inputs are preferred. Input histories whose complete monthly changes contradict their YoY series are rejected.

CPI data must be aligned to their information-availability dates. The FRED loader preserves observed CPI levels separately as validation outcomes. Latest revised data are still not equivalent to historical publication vintages.

## Yield forecasts and physical returns

The loader now uses last available business-day observations within each month from [DGS2](https://fred.stlouisfed.org/series/DGS2), [DGS7](https://fred.stlouisfed.org/series/DGS7), [DGS20](https://fred.stlouisfed.org/series/DGS20), and [DFII10](https://fred.stlouisfed.org/series/DFII10). Market yields remain on their observation calendar while macro releases move to their availability calendar. They are excluded from synthetic backfill.

| Asset channel | Default reference |
|---|---|
| Government bond, duration ≤3.5 years | `bond_yield_2y` |
| Government bond, duration ≥12 years | `bond_yield_20y` |
| Other government bonds | `bond_yield` (seven-year reference) |
| Cash | `interest_rate` |
| Inflation-linked bonds | `real_bond_yield` (ten-year reference) |
| Corporate bonds | Duration-matched Treasury reference plus explicit spread |

SHY is a short government bond with duration risk, not cash. These are coarse reference tenors, not each fund's actual yield curve. Override them with `asset_yield_columns`, for example `{"TLT": "my_long_yield"}`. CSV yields/spreads default to annual percentage points (`yield_scale=0.01`); decimal inputs require `yield_scale=1`.

Persistent yield forecasts combine a no-change forecast with a damped estimate of recent changes. Weights use past one-step squared forecast errors, measured before updating the trend, with a 24-observation equal-risk prior. Bootstrap block joins cannot train the trend. Neither component has a fixed historical yield-level attractor. Joint macro innovation covariance still supplies correlated rate, growth, inflation, and spread shocks. The adaptive forecast only marginally outperforms the simpler random walk in this sample; this is not evidence of reliable rate predictability.

For government bonds, monthly physical log return is:

```
log(1 + previous_yield) / 12 - duration * yield_change
```

Cash has carry without duration repricing. Higher yields raise future carry after the initial price loss. Historical physical factors are removed before fitting residual returns and reinserted along simulated paths; displayed current carry is subtracted from the path adjustment to avoid double counting.

The default equity response uses fitted macro forecast innovations through the existing return-factor regression. It no longer imposes a loss whenever inflation passes an arbitrary threshold. These are statistical innovations, not identified monetary-policy or supply shocks. The previous inflation-gated sensitivity remains available for comparisons.

## Return anchors and optional inputs

Mean shrinkage preserves fitted regime differences:

```
weight = prior_months / (observations + prior_months)
mean_shift = weight * (prior_target - sample_mean)
```

Defaults are 120 prior months, a 5% real geometric equity return, and 2% inflation. These are assumptions, not guaranteed portfolio rankings. With explicit physical carry, fixed-income residual targets are zero. Without it, anchors use current matched yields, expected credit losses, and inflation for TIPS.

An optional valuation multiple changes the equity annual log-return prior by:

```
valuation_weight * log(2) / half_life_years * log(reference_multiple / current_multiple)
```

The default weight is 0.25 and half-life ten years. Supply `equity_valuation_columns={"SPY": "cape"}` and optionally `valuation_reference={"SPY": 20}`; otherwise the reference is the training-sample median. Multiples must be positive and available at the forecast date. This is a soft prior, not a valuation forecast or market-timing rule. No valuation series is downloaded or fabricated automatically.

Corporate bonds require `asset_spread_columns={"LQD": "credit_spread"}` and an explicit annual expected-loss fraction such as `credit_loss_rates={"LQD": 0.01}`. This number is an example, not an estimate. Spreads evolve in log space to remain positive. Physical returns include total yield carry, spread and Treasury repricing, and expected loss. Discrete defaults, recoveries, and credit migrations remain unmodeled.

TIPS use real-yield repricing and lagged CPI accrual. They require `cpi_monthly_log` and `tips_inflation_log` (the realized lagged monthly indexation). The forecast applies a three-month observation-calendar lag, accounting for the loader's known CPI publication lag. CSVs without availability metadata are assumed to use the observation calendar. This monthly approximation omits daily index interpolation, exact coupon timing, and the maturity deflation floor. The loader's seasonally adjusted CPI is also a proxy for the contractual non-seasonally-adjusted index.

Bootstrap models preserve physical yield pairs, exclude artificial transitions, vary fitted parameters, and restore observed-origin yields, valuation multiples, CPI history, and latest trend state. The existing independent equity-prior uncertainty and parameter-bank convergence diagnostics remain available.

## Historical simulation checks

Run:

```bash
python scripts/benchmark_reality_fixes.py
```

Results and input hashes are under `research/reality_fixes_2026-10-13/`. The comparison uses SPY/IEF from March 2006 to December 2025, 30 one-year origins, 21 five-year origins, and 10 ten-year origins. Each fit generates 4,096 production paths across six allocations. All main comparison variants use the monthly CPI model; “previous” means the prior GS7/yield-VAR/equity-gate specification within this controlled comparison. `yield_only_ablation/` retains the earlier yield-only comparison.

CRPS scores the whole forecast distribution; lower is better. Coverage is the fraction of observed outcomes inside the predicted central 90% interval.

| One-year metric | Baseline | Previous economic specification | Revised economic specification |
|---|---:|---:|---:|
| Bond return CRPS | 0.0484 | 0.0608 | 0.0420 |
| Bond drawdown CRPS | 0.0198 | 0.0319 | 0.0191 |
| Bond drawdown 90% coverage | 90.0% | 53.3% | 86.7% |
| 60/40 return CRPS | 0.0509 | 0.0529 | 0.0505 |
| Stock return CRPS | 0.0817 | 0.0754 | 0.0792 |

Relative to the previous economic specification, bond return CRPS falls about 31% and drawdown CRPS about 40%. The stock return score worsens slightly. Twelve-month yield CRPS falls from 0.958 to 0.581 percentage points. Merely replacing monthly-average GS7 with endpoint DGS7 while retaining the historical-mean attractor performs worse; fixing the forecast dynamics matters.

**The 2022 tail remains a failure case.** At December 2021, revised one-year bond drawdown p95 is 11.83%, versus an observed 16.88%. The previous specification gives 15.21% for that event but systematically predicts too much bond drawdown at other origins. Improving broad scores does not resolve inflation-regime tail calibration.

The windows overlap, macro observations are latest revised data, and these choices followed inspection of the sample. This is diagnostic historical evidence, not an untouched holdout or causal attribution. Valuation, TIPS, and credit channels have behavior/identity tests but have not passed separate chronological real-market benchmarks.

## Remaining limits

Drawdowns are **monthly-close drawdowns**, which omit losses and recoveries within a month. No daily bridge has been added. Fixed duration still omits exact coupon cashflows, changing maturity, curve roll-down, and convexity. There is no identified supply-shock process, full credit-default model, or worldwide disaster calibration. Correlation can vary with regimes and common innovations; diversified portfolios are not guaranteed to avoid simultaneous losses.

These limits, especially the remaining 2022 miss and the small number of independent long-horizon outcomes, keep the economic mode experimental.

## Parameter-ensemble and implementation verification

The revised mode was also run from December 2025 with 128 recalibrated models per bank, 512 paths per model, two independent parameter banks, and two path seeds per bank (65,536 paths per run). GARCH/ADCC and semi-Markov regimes were enabled; per-path coefficient uncertainty was disabled because the bootstrap already varies fitted parameters.

```bash
python scripts/benchmark_parameter_convergence.py --economic both --reality-fixes \
  --origins 2025-12-31 --bank-seeds 101 202 \
  --output research/reality_fixes_2026-10-13/ensemble
```

| Allocation | Median nominal ten-year CAGR, across four runs | Monthly-close drawdown p95 |
|---|---:|---:|
| 100% stocks | 9.16–9.31% | 60.67–62.98% |
| 60% stocks / 40% bonds | 7.58–7.67% | 39.19–41.05% |
| 100% bonds | 4.28–4.30% | 22.59–23.08% |

The stock/bond ordering and diversification are present in simulated outcomes, not enforced by ranking constraints. Equity tail estimates still vary by about 2.3 percentage points across runs; convergence should be assessed at the user's required precision. The December 2025 CPI history contains a missing monthly observation, so this check uses the disclosed flat initial twelve-month CPI allocation. It is not a fully observed CPI-history benchmark.

The full Python run completed with **540 passes and one existing backfill byte-hash snapshot failure**. Three subsequently added edge-case tests also pass; the complete new reality-fix test file has 18 passing tests. The snapshot mismatch was reproduced against its recorded original source commit in an isolated tree during the preceding implementation; its fixture has not been rewritten. JavaScript tests pass (7), `node --check web/app.js` passes, and Ruff checks pass on the changed engine/research/test files. No browser visual check was performed for the settings controls.

Tests cover CPI price identities and base effects, native and Python monthly CPI generation, compact-versus-standard real reporting, missing CPI observations, yield persistence and bootstrap joins, maturity routing, valuation origin restoration, credit spread positivity/loss carry, TIPS rate sensitivity, and anchor-only real/credit adjustments. TIPS/credit/valuation tests are behavioral checks, not historical calibration evidence.
