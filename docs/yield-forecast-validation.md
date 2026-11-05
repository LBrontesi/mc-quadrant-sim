# Yield forecast validation — 4 November 2026

## Findings

The expanded comparison does **not justify promoting a Nelson–Siegel candidate
to the default**. Adaptive forecasts with conditional macro volatility are the
most promising stress candidate, but broad forecast superiority remains unproven.

- Original Nelson–Siegel worsens average yield CRPS by **14.0%**, IEF return
  CRPS by **10.4%**, and joint-loss Brier score by **39.3%** versus adaptive.
- Removing inflation repairs much of the original error, but worsens 60/40
  return CRPS by **6.3%** and joint-loss Brier score by **14.6%**. Both degradations
  have positive adjusted intervals under the primary 12-month block analysis.
  Its nominal 90% bond-drawdown interval covers only **74.6%** of outcomes.
- Random-walk curve factors perform approximately like adaptive. Curve geometry
  alone provides no demonstrated broad forecasting gain.
- Adaptive plus conditional volatility improves yield CRPS **1.4%**, bond-return
  CRPS **0.9%**, and bond-drawdown CRPS **10.6%**. The drawdown difference has a
  pointwise 95% interval of **[-0.00354, -0.00020]**, but its five-comparison
  adjusted interval **[-0.00414, +0.00030]** spans zero. Adjusted intervals also
  span zero with 6-month blocks; the 24-month-block interval excludes zero.
  The conclusion therefore depends on the inference specification. The yield
  and bond-return improvements are not established by the adjusted intervals.
- For 2022, adaptive plus volatility raises bond-drawdown P95 from **11.77%** to
  **12.46%**, still below the realized **16.88%**. All candidates forecast a
  positive median IEF return, while the realized return was **-15.16%**. A single
  P95 breach alone does not prove miscalibration, but this stress case remains
  poorly represented.
- The separate September 2026 yield check favors adaptive and random-walk
  factors over directional Nelson–Siegel. NS without inflation scores worse
  than the original NS there. The favorable 2025 bond result for NS without
  inflation does not generalize to this later yield observation.

Both simulation seeds reproduce the aggregate ordering of the primary metrics.
Next evidence needed is fresh point-in-time validation of adaptive plus
volatility, including parameter uncertainty and additional stress episodes.
No candidate was retuned from these results and no production default changed.

## Scope and evidence limits

Six candidates were fixed before aggregate scoring. Each model was refitted using only the history available at its forecast origin. The run covers **167 monthly origins, February 2011–December 2024**, with **8,192 paths per seed and two seeds** at each origin. There are 30 previously scored origins and 137 additional origins. All annual outcomes end by December 2025. Six stock/bond allocations use monthly rebalancing, before fees and taxes.

All candidates enable the experimental economic-return channel and joint macro
paths. This is a controlled fixed-fit comparison, not a reproduction of every
out-of-the-box UI setting, which can also include parameter-bootstrap banks.

The dataset has already informed model development, and adjacent annual outcomes overlap. Additional origins are robustness checks, not an untouched holdout. Macro inputs use revised, availability-aligned data, not historical vintages. No parameter-uncertainty banks were used. The separate December 2025 fit forecasts January–September 2026 yields from cached source data; those nine observations are one correlated trajectory, and frozen 2026 portfolio returns are unavailable. October is excluded because the cached daily input stops mid-month.

Candidates and acceptance criteria are recorded in [the fixed plan](../research/yield_validation_2026-11-04/PLAN.md). Production defaults and model code were not changed.

## Aggregate forecast scores

Lower is better. Yield CRPS is in annual yield percentage points; return and drawdown CRPS use decimal fractions. Seeds are averaged, not treated as extra historical observations.

| Model | Yield CRPS | Bond return CRPS | Bond drawdown CRPS | 60/40 return CRPS | Joint-loss Brier |
|---|---:|---:|---:|---:|---:|
| Adaptive | 0.52686 | 0.03796 | 0.01724 | 0.04719 | 0.05810 |
| Original Nelson–Siegel | 0.60060 | 0.04191 | 0.01772 | 0.05211 | 0.08095 |
| NS without inflation | 0.52369 | 0.03716 | 0.01676 | 0.05015 | 0.06658 |
| Random-walk curve | 0.52637 | 0.03800 | 0.01717 | 0.04745 | 0.05836 |
| Adaptive + volatility | 0.51962 | 0.03760 | 0.01542 | 0.04708 | 0.05825 |
| Random-walk curve + volatility | 0.52355 | 0.03782 | 0.01578 | 0.04737 | 0.05813 |

Relative change versus adaptive (%; negative is better):

| Model | Yield CRPS | Bond return CRPS | Bond drawdown CRPS | 60/40 return CRPS | Joint-loss Brier |
|---|---:|---:|---:|---:|---:|
| Adaptive | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Original Nelson–Siegel | 14.00 | 10.41 | 2.77 | 10.43 | 39.34 |
| NS without inflation | -0.60 | -2.09 | -2.78 | 6.27 | 14.60 |
| Random-walk curve | -0.09 | 0.11 | -0.40 | 0.56 | 0.45 |
| Adaptive + volatility | -1.37 | -0.93 | -10.55 | -0.22 | 0.26 |
| Random-walk curve + volatility | -0.63 | -0.35 | -8.51 | 0.39 | 0.05 |

## Uncertainty and trade-offs

Empirical coverage of nominal central 90% intervals (%):

| Model | Yield | Bond return | Bond drawdown |
|---|---:|---:|---:|
| Adaptive | 90.42 | 88.92 | 90.42 |
| Original Nelson–Siegel | 79.64 | 76.95 | 79.64 |
| NS without inflation | 79.04 | 76.05 | 74.55 |
| Random-walk curve | 88.32 | 86.23 | 90.12 |
| Adaptive + volatility | 91.02 | 88.32 | 94.01 |
| Random-walk curve + volatility | 88.02 | 85.63 | 93.11 |

Coverage above 90% is not automatically better; it can indicate conservative
intervals. Distribution scores and calibration must be assessed together.

Paired circular bootstrap uses 20,000 replicates after averaging seeds within each origin. The primary block spans 12 monthly origins; 6- and 24-month sensitivities are retained. Intervals below adjust across the five candidate comparisons within each metric, not across all metrics or earlier research choices. These approximate intervals do not establish external validity.

**Yield**

| Model | Difference | Adjusted 95% interval |
|---|---:|---:|
| Original Nelson–Siegel | 0.07374 | [-0.06612, 0.27115] |
| NS without inflation | -0.00317 | [-0.10350, 0.14420] |
| Random-walk curve | -0.00049 | [-0.01554, 0.01732] |
| Adaptive + volatility | -0.00724 | [-0.02439, 0.01004] |
| Random-walk curve + volatility | -0.00331 | [-0.02587, 0.02089] |

**Bond return**

| Model | Difference | Adjusted 95% interval |
|---|---:|---:|
| Original Nelson–Siegel | 0.00395 | [-0.00610, 0.01734] |
| NS without inflation | -0.00079 | [-0.00848, 0.00995] |
| Random-walk curve | 0.00004 | [-0.00090, 0.00110] |
| Adaptive + volatility | -0.00035 | [-0.00133, 0.00069] |
| Random-walk curve + volatility | -0.00013 | [-0.00151, 0.00152] |

**Bond drawdown**

| Model | Difference | Adjusted 95% interval |
|---|---:|---:|
| Original Nelson–Siegel | 0.00048 | [-0.00273, 0.00453] |
| NS without inflation | -0.00048 | [-0.00269, 0.00194] |
| Random-walk curve | -0.00007 | [-0.00062, 0.00069] |
| Adaptive + volatility | -0.00182 | [-0.00414, 0.00030] |
| Random-walk curve + volatility | -0.00147 | [-0.00329, 0.00056] |

**Balanced return**

| Model | Difference | Adjusted 95% interval |
|---|---:|---:|
| Original Nelson–Siegel | 0.00492 | [0.00109, 0.00906] |
| NS without inflation | 0.00296 | [0.00028, 0.00574] |
| Random-walk curve | 0.00026 | [-0.00004, 0.00056] |
| Adaptive + volatility | -0.00010 | [-0.00042, 0.00013] |
| Random-walk curve + volatility | 0.00019 | [-0.00018, 0.00058] |

**Joint loss**

| Model | Difference | Adjusted 95% interval |
|---|---:|---:|
| Original Nelson–Siegel | 0.02285 | [0.00426, 0.05004] |
| NS without inflation | 0.00848 | [0.00159, 0.01756] |
| Random-walk curve | 0.00026 | [-0.00029, 0.00084] |
| Adaptive + volatility | 0.00015 | [-0.00075, 0.00167] |
| Random-walk curve + volatility | 0.00003 | [-0.00079, 0.00092] |

## Additional forecast origins

137 origins; relative change versus adaptive (%). Calendar-year outcomes still come from the previously inspected dataset.

| Model | Yield CRPS | Bond return CRPS | Bond drawdown CRPS |
|---|---:|---:|---:|
| Adaptive | 0.00 | 0.00 | 0.00 |
| Original Nelson–Siegel | 12.13 | 8.54 | 1.64 |
| NS without inflation | -1.21 | -2.63 | -3.21 |
| Random-walk curve | -0.17 | 0.02 | -0.48 |
| Adaptive + volatility | -1.56 | -1.11 | -11.48 |
| Random-walk curve + volatility | -0.80 | -0.56 | -9.56 |

## Nonoverlapping calendar-year outcomes

14 origins; relative change versus adaptive (%). Calendar-year outcomes still come from the previously inspected dataset.

| Model | Yield CRPS | Bond return CRPS | Bond drawdown CRPS |
|---|---:|---:|---:|
| Adaptive | 0.00 | 0.00 | 0.00 |
| Original Nelson–Siegel | 19.16 | 14.58 | 2.28 |
| NS without inflation | 0.22 | -2.99 | -0.54 |
| Random-walk curve | -1.25 | -1.32 | -0.92 |
| Adaptive + volatility | -2.48 | -1.71 | -10.19 |
| Random-walk curve + volatility | -2.84 | -2.64 | -6.69 |

## 2022 bond forecast from 2021-12-31

Actual IEF return: **-15.16%**. Actual monthly-close maximum drawdown: **16.88%**. Forecast quantiles below are averaged across the two seeds.

| Model | Median bond return (%) | Bond drawdown P95 (%) | Return CRPS | Drawdown CRPS |
|---|---:|---:|---:|---:|
| Adaptive | 2.1490 | 11.7692 | 0.1352 | 0.0978 |
| Original Nelson–Siegel | 8.7453 | 7.7521 | 0.2085 | 0.1215 |
| NS without inflation | 2.6201 | 9.7635 | 0.1481 | 0.1065 |
| Random-walk curve | 2.1814 | 11.1237 | 0.1380 | 0.1007 |
| Adaptive + volatility | 2.0855 | 12.4558 | 0.1346 | 0.0957 |
| Random-walk curve + volatility | 2.0991 | 11.5476 | 0.1377 | 0.0998 |

## 2025 bond forecast from 2024-12-31

Actual IEF return: **8.03%**. Actual monthly-close maximum drawdown: **1.24%**. Forecast quantiles below are averaged across the two seeds.

| Model | Median bond return (%) | Bond drawdown P95 (%) | Return CRPS | Drawdown CRPS |
|---|---:|---:|---:|---:|
| Adaptive | 4.5376 | 9.6753 | 0.0229 | 0.0188 |
| Original Nelson–Siegel | 6.9789 | 7.4568 | 0.0140 | 0.0139 |
| NS without inflation | 8.2932 | 7.0192 | 0.0124 | 0.0125 |
| Random-walk curve | 4.6422 | 8.9315 | 0.0219 | 0.0178 |
| Adaptive + volatility | 4.5985 | 8.6168 | 0.0218 | 0.0138 |
| Random-walk curve + volatility | 4.6326 | 8.5616 | 0.0212 | 0.0144 |

## Separate 2026 yield check

The model is fit once at December 2025. September 2026 observed seven-year Treasury yield in the cached source: **5.19%**. This is a nine-month forecast, not a one-year result. Forecast quantiles are averaged across seeds.

| Model | P05 yield (%) | Median yield (%) | P95 yield (%) | Yield CRPS |
|---|---:|---:|---:|---:|
| Adaptive | 2.6376 | 3.9406 | 5.2247 | 0.8461 |
| Original Nelson–Siegel | 2.6812 | 3.7514 | 4.8172 | 1.0852 |
| NS without inflation | 2.6656 | 3.6513 | 4.6819 | 1.1942 |
| Random-walk curve | 2.7671 | 3.9774 | 5.1586 | 0.8398 |
| Adaptive + volatility | 2.6485 | 3.9331 | 5.2286 | 0.8536 |
| Random-walk curve + volatility | 2.7704 | 3.9740 | 5.1603 | 0.8450 |

## Reproduction and verification

```sh
.venv/bin/python scripts/validate_yield_candidates.py --output research/yield_validation_repeat
```

The runner refuses to overwrite a directory with an existing manifest. Raw per-origin scores, paired comparisons, input/source hashes and runtime provenance are saved under `research/yield_validation_2026-11-04/`. That directory is gitignored by the repository. The runner is `scripts/validate_yield_candidates.py`.

Artifact checks verified expected row counts, unique candidate/seed/origin keys, finite scores, and unchanged hashes for every source/input captured in the manifest. A synthetic paired-inference check verified difference direction, seed averaging, historical sample size and constant-difference intervals.

The full Python suite passed **561 tests in 124.49 seconds**. All **7 JavaScript
tests** passed and repository-wide Ruff checks passed. The formerly documented
backfill snapshot failure did not reproduce in this run. These checks validate
software behavior; they do not establish forecast accuracy.
