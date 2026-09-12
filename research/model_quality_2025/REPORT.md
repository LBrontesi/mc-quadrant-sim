# Before/after retrospective forecast comparison

The updated model improved 0 of 7 return/loss/drawdown score point estimates. Negative improvement percentages below mean deterioration, not improvement.
This bundled comparison does not identify which individual change caused a gain or regression. Macro RMSE and interval coverage are separate diagnostics, not evidence that all return forecasts improved.

72 monthly forecast origins, January 2020–December 2025; expanding training history.
SPY/IEF/GLD/DBC, weights 40/30/15/15; 4,096 paths, 8 bootstrap models per origin.
Both use joint macro, GARCH/ADCC, semi-Markov durations and monthly rebalancing; no taxes or trading costs.
Training starts in March 2006, excluding DBC's partial inception month. Growth is industrial-production YoY; inflation is CPI YoY; rates are FEDFUNDS.
Baseline commit: `3d5cbc72d01d13362268622def3de9e343474971`. Raw data SHA-256: `c5feef78d6fb8e314ab03b372f621e7f81f46a0acf1ea724c1988ba2501c68ba`.

| Score (lower is better) | Baseline | Updated | Improvement | 95% CI of updated − baseline |
|---|---:|---:|---:|---|
| energy | 0.055313 | 0.055911 | -1.1% | [0.000131960439904723, 0.001023094283448158] |
| crps_1m | 0.015986 | 0.016205 | -1.4% | [-9.147014943091136e-05, 0.0005713453846863447] |
| loss_brier | 0.218881 | 0.226817 | -3.6% | [0.0016423113644123077, 0.014108614602850544] |
| crps_3m | 0.024806 | 0.025405 | -2.4% | [-0.0002357402390894387, 0.0013825778644838997] |
| crps_12m | 0.058204 | 0.058271 | -0.1% | [-0.002766952212515572, 0.002899412563485212] |
| drawdown_crps_3m | 0.013522 | 0.014141 | -4.6% | [-0.0004229848433980502, 0.0015527152698577892] |
| drawdown_crps_12m | 0.023616 | 0.024776 | -4.9% | [-0.0012788620003457835, 0.0035402188059051485] |

## Calibration checks

| Diagnostic | Baseline | Updated |
|---|---:|---:|
| breach_05 | 0.069444 | 0.069444 |
| coverage_90 | 0.819444 | 0.861111 |
| interval_width | 0.078958 | 0.090372 |
| pit | 0.544963 | 0.540134 |
| expected_duration | 23.553104 | 24.215156 |
| macro_growth_rmse | 5.184758 | 2.893451 |
| macro_inflation_rmse | 2.886089 | 0.636578 |
| macro_interest_rate_rmse | 2.585987 | 0.397458 |

Target: 5% lower-tail breaches and 90% interval coverage. Wider intervals are not automatically better.
Macro RMSE is in the series' percentage-point units, not investment returns.

## One-month portfolio score by calendar year

| Year | Baseline CRPS | Updated CRPS |
|---|---:|---:|
| 2020 | 0.024982 | 0.024615 |
| 2021 | 0.011487 | 0.011623 |
| 2022 | 0.023756 | 0.023717 |
| 2023 | 0.017197 | 0.017125 |
| 2024 | 0.009782 | 0.010554 |
| 2025 | 0.008712 | 0.009594 |

Selected pre-2020 settings: `{'mean_prior_strength': 48.0}`.

## Limits

This is a paired retrospective ablation, not a point-in-time investment backtest. FRED history is revised; availability uses a fixed one-month approximation. No synthetic pre-inception returns are used.
The updated settings were selected using only pre-2020 data. Both versions are refitted at each origin using past data only. Baseline is the committed package; updated is the working package. The original baseline bootstrap starting-state and macro-uncertainty behavior are preserved deliberately.
Confidence intervals use paired circular 12-month block resampling (4,000 replicates), accommodating overlapping annual forecasts. They do not adjust for multiple comparisons or eliminate Monte Carlo error. Only one seed schedule and one portfolio are tested.
An interval excluding zero suggests a paired difference under this resampling scheme, not universal superiority or inferiority. This six-year sample is particularly limited for tail-risk validation.
Three-/12-month observations are limited to horizons that end by December 2025. There is no 30-year realized holdout and no direct observation of the true latent economic regime.
The fixed asset universe creates selection/survivorship limitations. Results cannot establish future forecasting or investment performance.

Detailed calibration/coverage diagnostics and data/code fingerprints are in the adjacent JSON files.
