# Allocation simulation realism investigation

Completed 12 October 2026. This is an investigation of generated paths, not a change to the production model.

**Finding:** the engine produces the expected broad stock/bond/diversification pattern with the full sample, but that does not establish realistic calibration. Historical refits expose weak bond-stress forecasts, training-window-dependent return rankings, and substantial seed sensitivity with eight parameter recalibrations.

## What was actually simulated

- SPY and IEF monthly returns, March 2006–December 2025. IEF is one Treasury-bond exposure; results do not cover every kind of bond or stock market.
- Six allocations, from 0% to 100% stocks in 20-point steps. Every allocation uses the **same simulated asset paths** within each run.
- Monthly rebalancing, nominal total returns, no fees, taxes or contributions/withdrawals. Drawdowns are measured at month ends, including the initial portfolio value; intramonth losses are not measured.
- Production return generator: probabilistic quadrants, semi-Markov durations, MNTS innovations, fitted GARCH and dynamic correlation, joint macro BVAR ensemble, mean prior 24 and duration prior 8.
- Controlled baseline: structural links disabled, no outer parameter bootstrap, per-path macro coefficient uncertainty enabled. This isolates the fitted model; it is **not the complete UI default configuration**.
- Walk-forward baseline: 4,096 paths per model per origin, chronological prefix refits, 59 usable origins. Each origin also has independent-MNTS and paired stationary-block benchmarks. Horizons have 59/51/42/20 forecasts for 1/3/5/10 years respectively.
- Full-history sensitivities: 20,000 ten-year paths for each of seven variants, also examined at 1/3/5 years.
- App-setting checks: structural links enabled, with and without eight anchored parameter-bootstrap models, at February 2011, December 2021 and December 2025. Each configuration uses 20,000 paths and two seeds. Bootstrap allocation, state anchoring, seed offsets and disabling duplicate macro coefficient uncertainty follow the production pipeline. These checks cover the relevant return-generator settings, not a browser/API end-to-end run or every UI setting.

## 1. Full-history allocation behavior is broadly sensible

Controlled baseline, calibrated through December 2025; ten-year outcomes:

| Stocks / bonds | Median annualized return | Median maximum drawdown | 95th-percentile maximum drawdown |
|---|---:|---:|---:|
| 0 / 100 | 3.3% | 13.9% | 30.0% |
| 20 / 80 | 5.1% | 11.3% | 27.6% |
| 40 / 60 | 6.7% | 13.3% | 32.2% |
| 60 / 40 | 8.1% | 17.6% | 39.7% |
| 80 / 20 | 9.5% | 23.0% | 48.4% |
| 100 / 0 | 10.7% | 28.6% | 57.2% |

Stocks outperform bonds in 86.5% of paired ten-year paths and have deeper drawdowns in 91.7%. The 60/40 portfolio has a smaller drawdown than stocks in 99.8%; 20/80 beats pure bonds on drawdown in 73.8%. Thus mixed-portfolio drawdown need not sit between the two endpoints.

With structural links plus eight parameter recalibrations, the same broad full-history pattern remains in both checked seeds: median stock return 9.9–11.9%, bond return 2.8–3.3%, and 60/40 return 7.5–8.6%. These are model outputs conditional on data ending in 2025, not forecasts calibrated to October 2026 market conditions.

## 2. Return rankings depend heavily on the calibration window

With information through February 2011, the controlled simulation gives median ten-year CAGR of **3.2% for stocks versus 6.5% for bonds**, while stocks retain greater drawdown risk. Two independent 20,000-path checks reproduce approximately 3.3% versus 6.5%. Enabling structural links with fixed parameters also leaves approximately 3.4% versus 6.5%.

The stock median falls below the bond median at 11 of the 20 ten-year forecast origins. Those origins overlap heavily and are concentrated early in the available history; this is not an estimated population frequency.

Code explanation: `estimate_probabilistic_regime_moments` shrinks regime means toward the **sample-wide historical mean**. It does not provide an independent long-term expected-return anchor. Structural links adjust macro sensitivities, rather than replacing that mean target. A crisis-heavy initial training sample can therefore propagate low stock returns and high bond returns far into the future.

This does not justify forcing stocks to outperform on every path. It does justify checking how long-horizon expectations respond to sample choice and independently estimated economic return assumptions.

## 3. The 2022 joint selloff is underrepresented

Forecast origin December 2021; realized outcomes over calendar 2022:

| Allocation | Realized maximum drawdown | Baseline forecast 95th percentile | Structural + bootstrap8 95th percentile, two seeds |
|---|---:|---:|---:|
| Bonds | 16.9% | 9.7% | 8.6–9.2% |
| 60/40 | 20.5% | 19.5% | 17.0–19.5% |
| Stocks | 23.9% | 33.8% | 29.9–34.0% |

The problem is not uniformly insufficient equity tails. The equity loss fits inside these forecast ranges, while the bond loss lands around the **99.6th percentile** in both structural/bootstrap runs.

Actual monthly stock/bond log-return correlation was +0.61. The structural/bootstrap forecast median was about −0.34, with 95th percentiles +0.43 and +0.48. Both assets lost money in 7 of 12 months; those forecasts expected joint losses in approximately 10–12% of months. This episode is a warning signal, not proof by itself that a particular tail probability is wrong.

The wider walk-forward sample also shows a bond mismatch. For five-year horizons, actual maximum drawdown averages 13.1%, versus an average forecast median of 8.2%. Actual longest underwater spells average 32 months versus a forecast-median average of 16 months. Stock spells go the other way: 12 versus 23 months. Underwater spells include unfinished spells at the horizon; these numbers are not completed recovery-time estimates.

Inflation-related changes in stock/bond correlation have an economic basis: the BIS documents positive correlation during the inflation/tightening period and reduced bond hedging benefits. [BIS, December 2023](https://www.bis.org/publications/qr-202312/markets-adjust-to-higher-longer). Interest-rate sensitivity also depends on bond maturity and coupon characteristics. [SEC interest-rate risk bulletin](https://www.investor.gov/sites/default/files/ib_interestraterisk.pdf).

## 4. Eight parameter recalibrations produce unstable reported summaries

Same full-history data, structural links enabled, 20,000 paths in each run:

| Metric | Fixed fit, seed A | Fixed fit, seed B | Bootstrap8, seed A | Bootstrap8, seed B |
|---|---:|---:|---:|---:|
| Stock median ten-year CAGR | 10.83% | 10.87% | 9.89% | 11.92% |
| Stock 95th-percentile drawdown | 56.8% | 57.1% | 71.1% | 55.3% |
| 60/40 median ten-year CAGR | 8.20% | 8.24% | 7.45% | 8.62% |

At the February 2011 origin, bootstrap8 stock medians move from 7.0% to 1.6% between seeds, changing the stock/bond ranking. Both bootstrap model sampling and inner path sampling change with the seed, so this does not completely isolate the source of the variation. The much smaller variation in fixed-fit runs makes convergence of the outer parameter ensemble a priority. More paths within only eight fitted models need not solve that problem.

## 5. Adding heavier tails alone is unlikely to address the main findings

Full-history controlled sensitivities, ten-year 60/40 maximum-drawdown 95th percentile:

| Variant | Drawdown |
|---|---:|
| Baseline | 39.7% |
| GARCH disabled | 38.9% |
| Dynamic correlation disabled | 40.5% |
| Near-Gaussian innovations | 39.4% |
| Mean prior 0 | 48.1% |
| Mean prior 48 | 37.0% |

The mean prior has a much larger effect here than those component switches. These are sensitivity experiments, not out-of-sample model-selection results. The near-Gaussian variant retains target moments but changes the innovation law without refitting other coefficients; switches need not preserve identical random draws across variants.

## Recommended next experiments

1. **Establish parameter-ensemble convergence.** Separately vary the bootstrap bank seed and inner path seed; compare 8/32/64 or more fitted models. Set tolerances for median returns, tail drawdowns and allocation rankings before choosing a default count.
2. **Test economic return anchors.** Compare current sample-mean shrinkage with uncertain, horizon-aware priors based on bond yield/carry and equity cash flows or risk premia. Preserve the possibility of stock underperformance. Validate across calibration windows rather than tuning the full-sample ranking.
3. **Improve persistent bond/inflation stress.** Test an explicit yield-curve/duration/carry channel and inflation-conditioned stock/bond dependence. Existing structural priors are already present; merely switching them on did not fix the checked episode. The return-factor link currently uses forecast innovations, with explained variance capped at calibration, so persistent repricing needs a separate diagnostic.
4. **Use portfolio behavior as an acceptance criterion.** Keep drawdown depth, underwater duration, joint losses, return calibration and seed stability alongside distribution scores. Retain 2022 as a known diagnostic, then evaluate candidate changes on other withheld periods to avoid optimizing for one event.

## Evidence limits and reproducibility

The macro data are release-lag aligned but **latest revised**, not true historical vintages. No future rows enter a prefix fit, but revision leakage remains possible. Horizon windows overlap: the available forecasts contain only 14/4/2/1 non-overlapping windows at 1/3/5/10 years, and even these are not necessarily statistically independent. The financial crisis occurs before the first forecast origin and therefore contributes training/descriptive data rather than an out-of-sample crisis test. No claim of validated ten-year tail frequencies is warranted.

Run from the repository root:

```sh
.venv/bin/python scripts/investigate_allocation_realism.py
.venv/bin/python scripts/check_allocation_realism_settings.py
.venv/bin/python -m pytest tests/test_allocation_realism.py -q
```

Outputs are in `research/allocation_realism_2026-10-11/` (git-ignored): `manifest.json`, `scores.csv`, `summary.csv`, `dependence.csv`, `sensitivities.csv`, `ordering.csv`, `historical_windows.csv`, and the `settings_*.csv`/`settings_manifest.json` files. The baseline manifest records the frozen data hash and source hashes. The directory name reflects the investigation's start date.

Verification: all three investigation tests pass; both scripts pass Ruff; the baseline run also checks its portfolio-return reducer against production portfolio accounting. No production calibration or simulation behavior was changed by this investigation.
