# Persistent macro stress: implementation and simulation evidence

Implemented 2026-10-16. The engine now supports persistent statistical macro news and conditional macro volatility. **These are optional research candidates. The default remains `macro_stress="none"`**, because the historical improvements are modest, uncertain, and uneven across metrics.

## Enable and reproduce

In **Economic return assumptions**, choose **Macro stress dynamics**. The setting is saved with the other controls. Python/API callers use:

```python
economic_returns = {"macro_stress": "volatility"}  # conditional macro volatility
# Also available for research:
economic_returns = {"macro_stress": "persistent"}  # filtered forecast-mean correction
economic_returns = {"macro_stress": "combined"}    # both components
```

Joint macro paths and economic rate stress must be enabled. Effective assumptions and fitted state parameters are included in API methodology. Existing settings without this key retain the current model. The macro generator uses the Python path for these candidates, while native asset-return sampling remains available; no native ABI change is required beyond the existing version 11.

```bash
python scripts/benchmark_macro_stress.py
python scripts/benchmark_parameter_convergence.py --economic both \
  --macro-stress volatility --reality-fixes --origins 2025-12-31 \
  --bank-seeds 101 202 \
  --output research/macro_stress_2026-10-16/ensemble_volatility
```

Research artifacts are under `research/macro_stress_2026-10-16/`: candidate plan, input/source hashes, fitted states, portfolio and macro scores, paired block intervals, and parameter-bank samples.

## Model

The input is the joint vector of forecast errors from the existing growth, monthly CPI, policy-rate, and maturity-matched yield forecasts. Existing innovations retain their fitted correlations. CPI still accumulates monthly log changes into its twelve-month price history; market yields retain their endpoint calendar.

**Persistent news:** a one-sided exponential filter stores recent base-forecast errors in standardized units:

```
correction[t] = news[t] @ response
innovation[t] = base_forecast_error[t] - correction[t]
news[t+1] = 0.8 * news[t] + 0.2 * base_forecast_error[t]
```

The response matrix has a zero-response ridge prior equivalent to 24 observations at the filter's iid variance. Its actual state transition `0.8 I + 0.2 response` is bounded to spectral radius 0.97. There is no fixed yield-level attractor. Corrections enter the same conditional forecasts used to construct historical residuals; asset exposures are refitted to the resulting innovations. The mean correction is not counted a second time as a new return shock.

**Conditional volatility:** a common scalar multiplies the regime-conditioned macro innovation covariance. Its GARCH update is:

```
h[t+1] = 1 - alpha - beta + beta * h[t] + alpha * energy[t]
```

Energy measures squared new innovations relative to the covariance of the current regime. This accounts for risk already explained by regimes. Covariances retain their estimated historical targets, with the usual shrinkage. State-specific inverse covariance metrics and their effective ranks normalize the updates; constant CPI and collinear directions cannot amplify numerical covariance-floor noise into volatility explosions.

The candidate pairs `(alpha, beta)` are `(0,0)`, `(.05,.8)`, `(.1,.8)`, `(.1,.88)`, and `(.2,.75)`. Training-history sequential Gaussian quasi-likelihood, with a two-parameter BIC penalty, selects among them. Constant variance is a supported result. All pairs have `alpha + beta < 1`. Large shocks affect the next variance, rather than the variance that generated them. Common scaling makes uncertainty in multiple macro drivers rise together while preserving the fitted innovation correlation within a regime.

Missing transitions and bootstrap block joins reset the filters and contribute no forecast score. Bootstrap models refit response coefficients, variance choices, and covariance metrics. Their raw-unit news state and conditional variance at the forecast origin are restored from the observed reference model. This uses a shared estimated origin state, rather than a separate posterior distribution of hidden states for each parameter draw.

These are **statistical forecast errors**, not identified causal supply, demand, or monetary-policy shocks. The design is motivated by evidence that inflation persistence and uncertainty change over time, including [Stock and Watson's inflation research](https://www.princeton.edu/~mwatson/papers/Stock_Watson_JMCB_2007.pdf). It does not reproduce that paper's unobserved-components stochastic-volatility model.

## Historical simulation results

The fixed-fit comparison uses 4,096 production simulation paths for each of 30 chronological origins and five candidates: current, random-walk yields, persistent news, conditional volatility, and both components. It retains semi-Markov regimes, fitted GARCH/ADCC asset dynamics, monthly CPI, and endpoint DGS7 yields. Six stock/bond allocations are evaluated at one, five, and ten years where observations permit. The current candidate uses the same seeds and updated inputs as the alternatives.

CRPS scores whole forecast distributions; lower is better. Coverage refers to the central predicted 90% interval.

| One-year metric | Current | Conditional volatility | Persistent news | Combined |
|---|---:|---:|---:|---:|
| Bond return CRPS | 0.04192 | 0.04181 | 0.04418 | 0.04418 |
| Bond drawdown CRPS | 0.01908 | 0.01783 | 0.01982 | 0.01845 |
| Bond drawdown coverage | 86.7% | 90.0% | 86.7% | 90.0% |
| 60/40 return CRPS | 0.05053 | 0.05057 | 0.05222 | 0.05218 |
| Stock return CRPS | 0.07923 | 0.07898 | 0.07926 | 0.07880 |
| Joint stock–bond annual-loss Brier score | 0.08596 | 0.08665 | 0.09650 | 0.09630 |

Conditional volatility reduces bond drawdown CRPS about **6.6%**. Its one-year inflation CRPS improves from 1.0653 to 1.0295 percentage points, but inflation interval coverage declines from 86.7% to 80.0%. Yield CRPS improves slightly, from 0.5797 to 0.5756 percentage points. These are trade-offs, not uniform gains.

The paired stationary-block interval for the bond drawdown CRPS difference is approximately **[-0.00293, +0.00034]**, spanning zero. Intervals use paired origin losses and expected block lengths corresponding to overlapping forecast horizons. Horizons with fewer than three effective blocks receive no interval. They do not establish statistical superiority or remove the effects of model selection on inspected data.

The persistent-news candidate worsens the joint-loss Brier score by approximately 0.01055, with a paired interval of [0.00172, 0.02301]. This is evidence against promoting that component. The combined model has a similar regression.

At December 2021, conditional volatility raises the next-year bond drawdown p95 from **11.45% to 12.26%**. Actual 2022 monthly-close drawdown was **16.88%**. The inflation stress failure is reduced but remains unresolved. The persistent-mean candidate reduces this p95 to 10.09%, illustrating why adding persistence is not automatically an improvement.

For forecast origins in 2020 onward, bond drawdown coverage improves from 72.7% to 81.8%, still below the nominal 90%. Earlier-origin coverage is 94.7% for both models. This is a descriptive split of already inspected data, not a newly untouched evaluation set.

## Parameter-bank checks

The volatility candidate was also simulated from December 2025 using two independently fitted banks of 128 models, 512 paths per model, and two path seeds per bank: 65,536 paths per run. Within-path coefficient uncertainty is disabled to avoid counting parameter uncertainty twice.

| Allocation | Median nominal ten-year CAGR across four runs | Monthly-close drawdown p95 |
|---|---:|---:|
| Stocks | 9.14–9.38% | 60.59–62.76% |
| 60/40 | 7.56–7.72% | 39.10–40.82% |
| Bonds | 4.28–4.29% | 24.29–24.80% |

The allocation trade-off remains present without a forced ordering rule. Equity tail estimates vary by about 2.2 percentage points across runs. Agreement across four runs is a stability check, not proof of convergence at every desired precision. The December 2025 dataset has an incomplete CPI initialization history, so this check uses the disclosed flat allocation of initial monthly CPI changes.

## Limits and verification

The data remain post-2006 US ETFs and latest-revised macro observations. The windows overlap and have already informed model development. There is no independent final holdout, international inflation-regime validation, daily drawdown process, or causal shock identification. Conditional volatility also remains Gaussian conditional on its variance; it does not fully describe rare structural breaks. These limitations and the scoring trade-offs keep all new candidates optional.

Nine new tests cover one-sided filters, join resets, recovery of a known persistent mean in synthetic data, conditional variance behavior, unit invariance, degenerate CPI, regime risk accounting, observed-origin bootstrap restoration, API validation, and end-to-end purchasing-power accounting. The final focused engine/API/pipeline run passed 168 tests; JavaScript tests passed seven and syntax/lint checks passed. The final full Python suite completed with **552 passes and one existing backfill byte-hash snapshot failure**. The failing hashes are the same previously reproduced mismatch; its fixture is unchanged. No browser visual check was performed for the new selector.
