# Engine reality audit — 12 October 2026

Implementation update (2026-10-13): the follow-up fixes and measured simulation results are documented in [economic-return-model.md](economic-return-model.md). The audit below records the earlier model state.

This audit combines source inspection, primary-source research, and additional simulations. It makes no production engine changes. Proposed remedies remain candidates until they pass chronological tests.

## 1. The current bond problem is primarily a yield-forecast problem

The experimental model applies sensible first-order duration mechanics to yields that can move too aggressively toward historical levels. Calibration limits the autoregressive matrix's spectral radius to 0.98 and simulation pulls macro levels toward regime-conditioned historical centers (`calibration.py`, `_fit_joint_macro_dynamics`; `simulation.py`, `simulate_joint_regime_macro_paths`). The exact contribution of these choices needs an ablation; the resulting yield drift itself is directly observable.

New diagnostic: 8,192 paths, a 12-month horizon, seed 20261012, frozen SPY/IEF history, GS7, baseline FIT settings plus structural links and the full economic mode. No outer parameter bootstrap; macro coefficient uncertainty follows the production simulator default.

| Forecast origin | Initial 7-year yield | Predicted median yield in 12 months | Observed yield in 12 months |
|---|---:|---:|---:|
| August 2011 | 1.63% | 3.46% | 1.14% |
| February 2012 | 1.37% | 3.33% | 1.35% |
| August 2020 | 0.46% | 2.28% | 1.06% |
| December 2021 | 1.40% | 2.50% | 3.72% |
| August 2023 | 4.26% | 2.72% | 3.77% |

For February 2012, the median cumulative physical bond component is approximately -10.7% over a year, before the fitted residual return. This is a forecast whose central tendency is wrong, not simply a missing extreme tail. The selected origins are diagnostic examples, not an unbiased performance sample; the earlier 30-origin score comparison establishes the broader regression.

[Federal Reserve research on yield forecasting](https://www.federalreserve.gov/pubs/ifdp/2010/993/ifdp993.htm) finds substantial variation in model performance across periods and motivates combining forecasts. Highly persistent yields make a no-change forecast a demanding benchmark.

**Candidate remedy:** separate slow-moving yield levels from temporary macro deviations. Compare a no-change/random-walk benchmark, a slowly evolving level model, and the current macro model. Estimate shrinkage or combination weights using past forecast scores only. Replace the universal stationarity cap for the yield-level component with an explicit persistent trend plus stable deviations; simply removing numerical stability checks would be unsafe modeling. Keep rate shocks and duration sensitivity, but do not presume rapid normalization from every low-yield origin.

Evaluate the yield forecasts first: bias, CRPS, 50/80/90/95% interval coverage at 1/3/12 months, stratified by low-rate, tightening, easing and inflation periods. Then evaluate bond returns and drawdowns. Avoid calibrating duration to compensate for a biased yield forecast.

## 2. Yield timing and maturity are part of the return identity

GS7 is a [monthly average of business-day yields](https://fred.stlouisfed.org/series/GS7), whereas monthly ETF returns span endpoints. [DGS7 supplies daily yields](https://fred.stlouisfed.org/series/DGS7). Matching the last available market observations at both endpoints is the first measurement correction to test. It is not yet demonstrated to solve the forecast regression.

A single seven-year yield with different duration multipliers cannot reproduce all short/intermediate/long bond behavior. Long yields incorporate expected future short rates and term premiums, which move differently across time and maturities. [New York Fed research](https://www.newyorkfed.org/research/staff_reports/sr775.html) documents this decomposition.

**Candidate remedy:** use a small yield curve (for example 2-, 7-, and 20/30-year points), with a level/slope representation and maturity-matched exposures. Add carry and roll-down, then convexity or cash-flow repricing. Forward rates alone are not unbiased future-yield forecasts because they include risk premiums. Explicitly separate a held-to-maturity bond from an ETF that maintains a maturity range. [AQR's bond-return decomposition](https://www.aqr.com/-/media/AQR/Documents/Insights/Alternative-Thinking/Alternative-Thinking-Style-Premia-Bond-Returns.pdf) discusses carry, duration and the holding-period qualifications to yield-based return expectations.

## 3. Year-over-year CPI is not this month's inflation

The data loader computes year-over-year inflation. The pipeline passes those simulated observations to inflation deflators, which use `(1 + annual_rate) ** (1/12)` each month. That interpretation is exact for an annualized current monthly rate, but not for a changing trailing twelve-month CPI rate. [BLS distinguishes one-month and twelve-month changes](https://www.bls.gov/opub/hom/cpi/concepts.htm).

A deterministic counterexample: CPI rises once from 100 to 110 and remains 110. Start the forecast immediately after the jump. Over the next twelve months actual cumulative inflation is zero. The current conversion of the eleven remaining 10% year-over-year observations, followed by zero, produces approximately **9.13% additional inflation**. This can distort purchasing power, indexed withdrawals and inflation stress persistence.

**Candidate remedy:** simulate monthly log CPI changes and accumulate a price index. Derive twelve-month inflation from that index for regime classification and reporting. Initialize the trailing eleven monthly changes from the information available at the forecast origin, with explicit treatment of publication lag. A fixed annual inflation assumption can retain its existing conversion. This is a mathematical-consistency correction independent of return-model selection.

## 4. Monthly drawdown and daily drawdown measure different risk

Using the repository's frozen Yahoo-adjusted daily prices:

| Calendar year / asset | Daily-close maximum drawdown | Monthly-close maximum drawdown |
|---|---:|---:|
| 2020 SPY | 33.72% | 19.45% |
| 2020 IEF | 4.68% | 2.02% |
| 2022 SPY | 24.50% | 23.93% |
| 2022 IEF | 18.18% | 16.88% |

Both calculations include the previous year-end starting value and use the same adjusted price series. Daily-close drawdown still excludes intraday extremes. More frequent sampling cannot reduce the maximum drawdown when the sampled monthly endpoints are retained.

**Candidate remedy:** label current risk metrics as monthly-close drawdowns. For daily drawdowns, margin breaches and execution/liquidity risk, add a separately calibrated daily process nested within the monthly macro state. A bridge that merely interpolates monthly returns does not automatically reproduce jumps, volatility bursts or market microstructure. Validate at the matching frequency before increasing volatility to match a daily crash statistic.

## 5. The type of surprise matters more than an inflation threshold alone

The current shared equity rate effect turns on between 3% and 5% lagged inflation. This permits joint stock/bond losses but does not identify whether a yield change reflects stronger growth, inflation news, a monetary-policy surprise or a term-premium shock. Generic fitted macro betas and dynamic correlations can absorb some of these relationships, but do not explicitly identify the causes.

[BIS analysis](https://www.bis.org/publications/qr-202312/markets-adjust-to-higher-longer) explains why growth news can produce negative stock/bond correlation in low-inflation conditions while inflation and expected policy responses can produce positive correlation in high-inflation conditions.

**Candidate remedy:** compare a small set of growth, inflation and discount-rate/risk-premium innovations with separately estimated asset responses. Prefer actual expectation/surprise measures when available. VAR forecast errors are a useful statistical approximation, not automatically identified economic shocks. Estimate sensitivities with shrinkage and allow uncertainty; do not force the correlation sign from a high/low inflation label. Test joint downside probabilities, correlation distributions and both recession and inflation episodes.

## 6. Starting valuation and expected cash-flow growth matter for equities

The new 5% real equity-return prior is transparent but constant. The engine has no explicit earnings/dividend valuation state. A historical return average can reflect a period of rising valuation multiples or unusually favorable earnings growth, rather than a sustainable future cash-flow return. [Federal Reserve analysis of corporate profit growth and stock returns](https://www.federalreserve.gov/econres/notes/feds-notes/the-coming-long-run-slowdown-in-corporate-profit-growth-and-stock-returns-20220906.html) illustrates why extrapolating past returns can be problematic.

**Candidate remedy:** compare the constant prior with a cautiously valuation-conditioned long-horizon prior combining payout yield, real per-share cash-flow growth and uncertain valuation change. Do not treat CAPE or an earnings-yield spread as a reliable short-term timing rule. Use strong shrinkage, data available at each historical origin, wider prior uncertainty, and a constant-premium control.

## 7. Different bond types need different factors

The experimental physical channel applies only to nominal government bonds and equities. Other classes still have historical return distributions and some structural macro priors, but lack the corresponding explicit cash-flow/risk-factor mechanics.

- **Short Treasuries versus cash:** SHY is currently classified as `cash` in `structural.py`, although it is a [1–3-year Treasury ETF](https://www.ishares.com/us/products/239452/ishares-13-year-treasury-bond-etf). Its 1.8-year duration is retained in older priors, so it is not simulated as completely riskless; nevertheless, it misses the new government-bond carry/repricing channel. Correct the classification together with a short-maturity yield reference, not by applying seven-year yields indiscriminately. True cash should earn a short-rate-linked return with its own reinvestment/fee assumptions.
- **TIPS:** inflation accrual and real-yield repricing must be separate. Positive inflation does not guarantee a positive mark-to-market return. [New York Fed work on real and nominal curves](https://www.newyorkfed.org/medialibrary/media/research/staff_reports/sr570.pdf) separates real rates, inflation expectations and risk/liquidity premiums.
- **Corporate bonds:** Treasury duration is only one exposure; spreads, expected credit losses and changing risk appetite matter. [Federal Reserve research on the excess bond premium](https://www.federalreserve.gov/econresdata/notes/feds-notes/2016/recession-risk-and-the-excess-bond-premium-20160408.html) shows the connection between credit conditions and recession risk. Add spread-duration and credit-loss dynamics before treating corporate bonds as equivalent portfolio protection to Treasuries.

There should be no universal rule that every bond portfolio has lower drawdown than every equity portfolio. Maturity, credit, inflation, currency exposure and horizon determine the comparison. Nor should a mixed portfolio be forced to outperform both components: diversification benefits depend on the joint distribution.

## Recommended sequence and acceptance criteria

1. Correct CPI accumulation and explicitly label drawdown sampling frequency; add deterministic identity tests.
2. Align endpoint yields with returns and distinguish maturity/cash exposures; validate historical return decomposition.
3. Compare persistent yield-level forecasts with no-change and existing macro benchmarks using chronological scores. Keep the full stress model experimental until broad bond coverage improves without losing inflation-tail behavior.
4. Add separately tested surprise-dependent cross-asset responses, then valuation-conditioned equity priors and additional bond-class factors.

Retain existing GARCH/ADCC/MNTS and parameter-ensemble convergence controls while isolating these changes. They already address volatility clustering, asymmetric dependence and heavy tails; changing them simultaneously would obscure which correction helps.

Predefine evaluation folds and candidates before fitting combinations. Report nominal and real returns, matched-frequency drawdowns, recovery time, joint losses, return CRPS, yield CRPS and interval coverage. Use block-aware uncertainty estimates for overlapping forecast windows. Extend the evidence beyond the current post-2006 U.S. ETF sample with suitably matched historical/global data and point-in-time macro vintages. A better fit to 2022 alone is insufficient, and the same history used to select improvements cannot serve as an untouched final holdout.

## Reproduction

Diagnostic script and outputs are saved under `research/economic_returns_2026-10-12/`:

- `diagnose_reality.py`
- `yield_diagnosis.csv`
- `drawdown_sampling_diagnosis.csv`

The prior benchmark manifest identifies the monthly macro/return and GS7 inputs. Daily-price provenance and SHA256 are in `research/sv_extensions_2026-10-04/daily_manifest.json` (Yahoo Finance adjusted closes, February 2006–December 2025). No new return-model implementation or claim of a validated fix is made by this audit.
