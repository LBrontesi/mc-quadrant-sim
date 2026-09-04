# MC Quadrant Simulator

A research-oriented, monthly Monte Carlo engine for long-horizon portfolio
planning. The production model combines four macroeconomic states with
regime-specific multivariate normal tempered-stable returns, GARCH volatility,
asymmetric dynamic correlation, and a native C++17 simulation backend. New web
sessions default to a 30-year horizon and 100,000 paths.

The four macro quadrants are:

| Regime | Growth | Inflation | Typical interpretation |
| --- | --- | --- | --- |
| `high_growth_low_inflation` | High | Low | Goldilocks / disinflationary expansion |
| `high_growth_high_inflation` | High | High | Overheating expansion |
| `low_growth_high_inflation` | Low | High | Stagflation |
| `low_growth_low_inflation` | Low | Low | Recession / deflationary slowdown |

The model is designed to be calibrated from real historical data:

1. Release-aware macro data initializes semantically identified growth/inflation quadrants.
2. An explicit-duration hidden semi-Markov model jointly estimates constrained macro emissions, latent states, state-age hazards, and exit destinations.
3. Parametric return means are shrunk and covariance matrices use Ledoit-Wolf shrinkage.
4. Optional stationary-bootstrap recalibrations measure parameter uncertainty.
5. Growth, inflation, the short rate, regimes, returns, dynamic correlations, and portfolio accounting are simulated together.
6. A fused multithreaded C++ kernel runs the production MNTS-GARCH paths and portfolio ledgers without retaining an unnecessary full return cube.
7. Walk-forward validation compares regime-switching MNTS forecasts with an unconditional MNTS benchmark.

## Methodology

### 1. Data And Alignment

Asset prices are converted to log returns and, for the dashboard market-data
path, aggregated to monthly frequency. FRED industrial production and CPI are
converted to year-over-year percentage changes. The monthly effective federal
funds rate (`FEDFUNDS`) is retained as an annual percentage-rate level rather
than transformed into a year-over-year change. Uploaded macro CSVs are assumed
to already contain the growth and inflation measures selected by the user and
can optionally provide an annual short-rate column.
When the requested market-data end date falls inside an unfinished month, that
partial month is excluded so the final observation is never labeled as a future
month-end. CSV asset inputs explicitly distinguish price levels, log returns,
and simple returns; simple returns compound within a month before conversion to
the log-return representation used by calibration.

Yahoo Finance inputs can optionally use historical proxy tickers to extend an
asset before its inception. A proxy is level-scaled at the last overlapping
observation and only fills missing primary prices; it never overwrites primary
history. Proxy returns are therefore an explicit approximation, not a claim
that the ETF existed earlier.

The Yahoo dashboards also support source-labeled synthetic backfills inspired
by the `IEFSIM`/`DBMFSIM` convention used by portfolio backtesters. Entering
`IEF, DBMF` as synthetic assets (with both tickers included in the Yahoo input)
keeps `IEF` and `DBMF` as observed-only series,
creates `IEF_SIM` and `DBMF_SIM` for modeled pre-inception segments, and
creates `IEFSIM` and `DBMFSIM` as stitched series for calibration. The generated
segment is reproducible from its seed and calibrated from observed monthly
returns; it is not an official index history.

#### Regime-conditioned synthetic backfill

The default backfill method (`synthetic_method="regime"`) reconstructs a
pre-inception history from the actual historical economy rather than from the
asset's full-sample moments alone:

1. Every historical month is classified into the four growth/inflation
   quadrants using the real FRED macro history for that date. Data-driven
   thresholds (median/mean) use a causal expanding window; fixed numeric
   thresholds classify every month without look-ahead.
2. A factor model `r_asset = alpha + sum(beta_j * r_anchor,j) + epsilon` is
   estimated on the asset's observed overlap against a default anchor universe
   (`SPY, IEF, GLD, DBC, EFA, VNQ, TIP, SHY`), which is fetched automatically.
   The target asset is removed from this anchor matrix, preventing an identity
   regressor from leaking the value being reconstructed into its own model.
3. For each pre-inception month the synthetic return combines the real anchor
   returns that month with a regime-specific residual when the asset has
   enough observed history in that regime, falling back to regime moments or
   the observed sample otherwise.
4. Reconstructed backward price levels anchor to the first observed price:
   `P_t = P_first * exp(-sum of synthetic log returns after t)`.

Each synthetic asset gets a feasibility grade describing how much of its
history rests on observed behavior versus projection:

| Grade | Meaning |
| --- | --- |
| `A` | All regimes observed, stable factor model (R² ≥ 0.5) |
| `B` | Partial regime coverage or moderate factor fit |
| `C` | Weak factor fit, short history, or proxy needed |
| `X` | Not enough observed history to backfill |

The asset-category registry (`EQUITY`, `LONG_TERM_BOND`, `MANAGED_FUTURES`,
and so on) provides default labels and can be overridden per asset with values
such as `DBMF:MANAGED_FUTURES`. A `full_sample` method is also available for
backward compatibility; it uses the asset's observed moments only.

The critical assumption is that behavior estimated on the post-inception window
applies to earlier anchor-based months. Synthetic pre-inception values are an
explicit approximation for deeper backtests, not the fund's actual historical
NAV.

Portfolio currency conversion uses historical FX levels aligned to the return
frequency. For an asset in EUR and a USD portfolio, the EUR local-currency log
return is combined with the USD-per-EUR log FX return. A current spot quote is
appropriate for converting a displayed value, but not for simulating future FX
risk, so the simulator requires historical FX data for foreign assets.

Macro observations are classified before they are joined to asset returns. In
default FRED mode, a one-period release lag is a conservative approximation;
the values are still the latest revised vintage. Strict point-in-time mode uses
the official FRED API's `output_type=4` initial releases, retains each
`realtime_start` availability date, and aligns the macro row to the month when
all inputs were public. Set `FRED_API_KEY` on the server and select **ALFRED
initial releases**. Custom macro CSVs can provide an `AvailableDate` column for
the same release-aware alignment. When exact availability dates are present,
the approximate macro-release lag is automatically set to zero instead of
delaying the information a second time.

### 2. Quadrant Classification

For each macro observation, growth and inflation are compared with their chosen
thresholds. The default threshold is the historical median; mean and fixed
numeric thresholds are also supported. Values greater than or equal to the
threshold are considered high.

| Growth | Inflation | Regime |
| --- | --- | --- |
| High | Low | High growth / low inflation |
| High | High | High growth / high inflation |
| Low | High | Low growth / high inflation |
| Low | Low | Low growth / low inflation |

A full-sample median silently uses future data to classify the past, which
leaks look-ahead information into calibration. Setting a **causal threshold
window** re-estimates each cutoff on an expanding window of prior observations
only (the earliest rows stay unclassified until enough history exists). The
UIs default to a 12-period window. Direct calibration keeps the full-sample
behavior unless `threshold_window` is supplied; walk-forward validation always
uses a causal 12-period window when none is supplied.

HSMM initialization uses a persistence-aware hard path: a causal three-month
trailing macro average, threshold hysteresis (default `0.15` historical standard
deviations), and two-month confirmation before accepting a new quadrant. Soft
logistic probabilities remain a compatibility fallback for models without
fitted HSMM emissions. The calibrated joint macro simulator uses the fitted
bivariate emission likelihood instead. Neither form is multiplied as
independent adjacent membership to estimate historical transitions.

### 3. Explicit-Duration Hidden Semi-Markov Model

The default quadrant model is a Gaussian-emission hidden semi-Markov model
(HSMM). Growth and inflation are the observed emissions, while the hidden state
is expanded to `(quadrant, months in quadrant)`. The persistent classifier fixes
the economic identity of the four emission distributions, but it does not
provide the final transition counts.

A scaled forward-backward pass returns filtered state probabilities, smoothed
state-age posteriors, and joint expected transitions. EM updates three distinct
objects from those joint posteriors:

- the state-specific joint growth/inflation means and covariances, shrunk toward
  the full sample while preserving each quadrant's high/low economic identity;

- a zero-diagonal exit-destination matrix describing where the economy moves
  after leaving each quadrant;
- state- and age-specific discrete exit hazards describing when it leaves.

The Viterbi path provides a duration-consistent hard history for diagnostics and
historical bootstrap pools. When probabilistic return moments are enabled, the
causal filtered HSMM probabilities weight observations instead. Transition
estimation never uses products of independent marginal memberships.

The transition heatmap shown by the API and UI is a one-month summary of the
HSMM: each state's expected exit rate is combined with its exit-destination
probabilities. The underlying destination matrix, filtered/smoothed
probabilities, convergence status, log likelihood, and state-age posterior are
retained in model metadata.

When transition uncertainty is non-zero, each transition row is sampled from a
Dirichlet distribution. The dashboard maps uncertainty `u` in `[0, 1]` to a
row concentration of `max(1, 1 / u^2)`. Higher uncertainty therefore produces
more variation around the calibrated transition probabilities.

When parameter recalibration is enabled, the empirical Dirichlet control is
not applied again. Each outer draw instead resamples paired macro/return months
with geometrically sized stationary-bootstrap blocks and recalibrates
thresholds, transition probabilities, durations, means, covariances, and joint
macro dynamics. This separates uncertainty in fitted parameters from ordinary
market-path randomness.

A first-order Markov chain implies a constant exit hazard and geometrically
distributed run lengths. The HSMM instead regularizes sparse state-age expected
exit counts toward the pooled age-dependent hazard. The minimum duration is five
months by default; after that floor, exit risk can change with regime age. No
short historical episode is discarded before fitting. The dashboard and
scenario API default to explicit HSMM durations, while
`duration_model="markov"` remains a sensitivity benchmark using the one-month
summary matrix. The age-dependent hazard design follows the
motivation in the Federal Reserve Bank of Minneapolis discussion paper
[A Markov-Switching Model of GNP Growth with Duration Dependence](https://www.minneapolisfed.org/research/discussion-papers/a-markov-switching-model-of-gnp-growth-with-duration-dependence).

With joint macro paths enabled, simulated growth and inflation update the
destination probabilities at each eligible exit through the fitted bivariate
Gaussian emission likelihood. This retains growth/inflation dependence instead
of multiplying two independent threshold probabilities. Consequently,
simulated transitions depend on both regime age and current macro conditions
rather than on age alone. The Gaussian likelihood is stored as six quadratic
coefficients per state, and semi-Markov simulations evaluate it only for paths
whose duration clock permits an exit.

Unless a state is selected explicitly, simulations begin from the latest
filtered HSMM posterior over both quadrant and current quadrant age. Semi-Markov
paths therefore sample the remaining duration of the current episode rather
than restarting its duration clock. Models without that posterior retain the
stationary-distribution fallback.

An alternative **HMM regime model** fits a Gaussian-emission hidden Markov
model directly on asset returns with expectation-maximization (states learned
from the return distribution rather than macro thresholds, with k-means
initialization and restarts). The fitted means, covariances, transition
matrix, and most-likely (Viterbi) state path are wrapped in the same
`ScenarioModel`, so simulation, reporting, and diagnostics work unchanged.
State labels are `state_0..state_{n-1}`; their economic meaning follows from
the fitted moments (a low-mean, high-covariance state is a stress regime).

### 4. Regime-Specific Return Moments

Returns aligned to each regime provide a state-specific mean and covariance. If
a regime has fewer observations than `min_observations`, its estimates are
blended toward the full-sample estimates. Covariance matrices are additionally
shrunk toward the full-sample covariance, projected to the nearest positive
semidefinite matrix, and converted to correlations. Optional pairwise
correlation views are blended into each regime and projected back to a valid
correlation matrix.

Expected returns can be shrunk toward the full-history mean with
`mean_prior_strength`. This is deliberately separate from covariance shrinkage:
regime means are typically the least stable long-horizon inputs, especially in
sparsely observed stagflation and recession states.

By default the shrinkage intensity is no longer a fixed `0.25` blend: the
Ledoit-Wolf optimal intensity is computed from the data, so sparse regimes are
shrunk aggressively while long histories keep their empirical covariance. The
PSD projection uses the Higham (2002) alternating projection, which preserves
the diagonal and minimizes the Frobenius distance to the PSD cone instead of
clipping eigenvalues. A fixed shrinkage value can still be forced.

The diagnostics panel reports observations per regime and covariance condition
numbers so sparse or unstable calibrations are visible rather than hidden.

### 5. Return Sampling

Each simulated month first draws one of the four growth/inflation states, then
draws returns from that state's **multivariate normal tempered-stable (MNTS)**
law. The model calibrates a common state-level tail index and tempering rate,
asset-specific skew parameters, and a latent dependence matrix. A shared
tempered-stable subordinator produces fat tails and joint extremes while the
latent factor preserves cross-asset dependence. Parameters are standardized so
the fitted state means, volatilities, and correlations remain the target first
two moments.

The native engine samples the tempered-stable subordinator exactly with the
two-dimensional single-rejection algorithm of Qu, Dassios, and Zhao (2021).
For every state parameter pair it selects the lowest-cost of four gamma-based
proposal envelopes, using either a uniform or truncated-normal auxiliary angle.
Acceptance tests are evaluated in log space for stability. Devroye's exact
double-rejection algorithm and the former hybrid are retained as validation and
benchmark references; they generate the same target law. The product exposes
no alternative return-path distribution: MNTS-GARCH remains the single
production law.

**GARCH(1,1) volatility clustering** adds conditional-variance dynamics within
each regime: every asset's variance follows
`h_t = omega + alpha * eps_{t-1}^2 + beta * h_{t-1}` anchored so the
unconditional level matches the regime covariance, and the variance re-anchors
when a path enters a new regime. Shocks therefore cluster in time without
drifting away from the calibrated regime covariance. `garch_alpha` governs
responsiveness to new shocks (default 0.10) and `garch_beta` the persistence
of past variance (default 0.85).

**Asymmetric dynamic correlation (ADCC)** evolves the correlation matrix after
each standardized shock. `dcc_alpha` controls shock response, `dcc_beta`
controls persistence, and `dcc_asymmetry` increases the response to joint
negative MNTS shocks. It re-anchors to the relevant state-level latent
correlation after a regime change.

**Joint macro-financial paths** default to a shrinkage Bayesian VAR(1) ensemble
that blends full-history and rolling-window coefficients for growth, inflation,
and the effective federal funds rate when available. Stability constraints,
posterior coefficient draws, an instability score, and regime-conditioned
innovation covariances propagate macro-parameter uncertainty into scenarios.
Simulated macro values influence time-varying transition probabilities, while
the same macro innovations—including short-rate changes—affect asset returns.
Only growth and inflation define quadrant membership; the rate is an additional
state variable. Optional structural asset profiles place weak economically
signed priors on return links: bond duration anchors rate sensitivity, while
growth and inflation exposures depend on the asset class. Inflation creates a different purchasing-power deflator for
every path, and the simulated short rate supplies the path-specific risk-free
benchmark and leveraged financing base. This compact model improves internal
consistency but is not a structural macroeconomic forecast, a Taylor-rule
model, or a full yield-curve model.

**Walk-forward validation** (quadrant model only, enabled by default in the
dashboard) checks the regime model and selected portfolio strictly out of sample: each split fits on
data up to period `t` and scores the next observation under the one-step
regime-switching MNTS predictive draws versus an unconditional MNTS model fitted
on the same history. A multivariate energy score provides a proper scoring rule
without approximating the MNTS density. Validation also reports transition Brier and log scores, observed
versus predicted switches per decade, completed-duration log scores and errors,
stability of expected durations across rolling calibration vintages,
actual-state probabilities, portfolio probability-integral-transform
diagnostics, 95% VaR breach frequency and clustering, and a Newey-West/HAC
score comparison. Multi-horizon 3/12/60-month errors and a mean-variance
certainty-equivalent advantage test whether complexity adds decision value.
The UI shows expected months in every state and expected
switches per decade, and warns when the implied persistence is unusually low.
Weak or miscalibrated results are surfaced as warnings rather than hidden.

#### Legacy-methodology comparison

The removed Student-t and Gaussian-GARCH return laws are retained only in Git
history for controlled research comparisons; they are not selectable product
options. A fixed-seed benchmark used the same 2007-2025 monthly SPY/IEF/GLD/DBC
and FRED history, 40/30/15/15 portfolio, macro/regime paths, monthly
rebalancing, and 10 basis-point transaction cost for every method.

In 20,000 simulated 30-year paths, MNTS-GARCH left median terminal wealth
essentially unchanged versus the legacy Student-t model (833.9 from an initial
100 in both cases), but increased the median maximum drawdown from 21.5% to
23.8% and the severe 5% drawdown from 36.0% to 42.3%. Its simulated monthly
skewness was -0.76 versus +0.15 for Student-t and 0.00 for Gaussian-GARCH; the
historical portfolio estimate was -1.17. This is the intended effect: preserve
the central long-term scenario while representing asymmetric downside and
volatility clustering more realistically.

A stricter expanding-window test trained on at least 120 months and forecast
January 2017 through December 2025. Each of 108 monthly origins used 32,768
paths; 97 origins also had a realized rolling 12-month outcome. Regime and
macro-path hashes matched exactly across methods.

| Out-of-sample score | Student-t | Gaussian-GARCH | MNTS-GARCH |
| --- | ---: | ---: | ---: |
| One-month portfolio CRPS (lower is better) | 0.01482 | 0.01498 | **0.01449** |
| One-month kernel log score (higher is better) | 2.169 | 2.158 | **2.219** |
| Twelve-month portfolio CRPS | 0.06226 | 0.06224 | **0.06045** |
| Twelve-month kernel log score | 0.781 | 0.779 | **0.834** |
| Observed one-month 95% VaR breach rate | 4.63% | 4.63% | 3.70% |

Against Student-t, the MNTS-GARCH CRPS improvement was 2.2% at one month
(`p=0.0012`) and 2.9% at 12 months (`p=0.0015`) using Newey-West/HAC score
comparisons. MNTS-GARCH nevertheless forecast a conservative average 1% monthly
VaR of -8.51%, compared with a -6.61% worst realized portfolio month in this
short validation window. Only about one 1% breach is expected in 108 months, so
this is evidence for further tail-parameter regularization, not a conclusive
99% coverage test. The benchmark supports MNTS-GARCH as the production law
while making clear that tail calibration and macro/mean calibration still
require ongoing out-of-sample monitoring.

### 6. Portfolio Accounting

Weights are normalized to sum to one. The legacy mode combines simulated log
returns with the weighted-log approximation. Buy-and-hold and rebalancing modes
track asset holdings directly: buy-and-hold lets weights drift, while scheduled
mode rebalances monthly, quarterly, or annually. Transaction costs are charged as:

```text
cost = transaction_cost_bps / 10,000 * sum(abs(target_holdings - current_holdings))
```

The dashboard defaults to monthly rebalancing with a 10 basis-point cost; the
core API retains the legacy mode unless a rebalancing frequency is supplied.
An optional liquidity stress model multiplies the base cost by the simulated
regime (0.75x Goldilocks, 1.25x overheating, 2.0x stagflation, and 1.5x
recession by default), so adverse-state rebalancing is more expensive.

Periodic cash flows are supported in both accounting modes. A **contribution**
is invested at the target allocation at the start of every period. Retirement
spending is requested at the end of the month and may contain non-overlapping
recurring phases plus one-time expenses. Each amount is real purchasing power
net of tax: the simulated inflation path converts it to nominal cash and the
Italian ledger sells enough holdings to fund both the requested net spending
and disposal tax. If the portfolio is exhausted, the funded amount records only
the cash that actually reaches the investor after immediate tax.

The `decumulation` API object supports `manual` and `safe_rate` modes, monthly,
quarterly, or annual phase frequency, and `fixed` or `guyton_klinger` policy.
Guyton–Klinger reviews each phase annually, cuts spending by 10% above the 120%
withdrawal-rate guardrail, raises it by 10% below 80%, and defaults to real
70%/130% floor and ceiling. Inflation indexing is skipped after a negative real
return year, and each new phase resets the reference rate and bounds. One-time
expenses do not affect guardrails.

```json
{
  "decumulation": {
    "enabled": true,
    "mode": "manual",
    "phases": [
      {"start_month": 37, "end_month": 180, "frequency": "monthly", "annual_real_amount": 24000},
      {"start_month": 181, "end_month": 360, "frequency": "monthly", "annual_real_amount": 18000}
    ],
    "one_time_expenses": [{"month": 60, "real_amount": 15000}],
    "policy": {
      "type": "guyton_klinger",
      "upper_guardrail": 1.20,
      "lower_guardrail": 0.80,
      "adjustment": 0.10,
      "floor": 0.70,
      "ceiling": 1.30,
      "skip_inflation_after_negative_real_return": true
    },
    "safe_rate": {
      "objective": "survival",
      "target_probability": 0.90,
      "minimum_bequest": 0
    },
    "annual_inflation_fallback": 0.02
  }
}
```

Legacy `withdrawal` and `withdrawal_start_period` remain supported and are
normalized to one monthly nominal phase, preserving prior results. Existing
browser settings, saved scenarios, and share links are migrated automatically
to the real-spending phase format. Wealth is floored at zero; contributions and
decumulation may coexist.

`POST /api/safe-rate` calibrates and simulates the market paths once, then
replays both fixed and guardrail policies on those identical paths. It searches
0–25% at 0.05 percentage-point precision for survival, preservation of initial
real capital, or a minimum real bequest. The response includes point success,
a Wilson 95% Monte Carlo interval, the evaluated safe-rate curve, and an
explicit `≥25%` warning if the upper bound still meets the target. A path counts
as successful only when every expense is fully funded. Normal `POST
/api/simulate` does not run this solver.

ETF expense ratios are supplied per asset as annual percentage points, for
example `SPY:0.03, IEF:0.15`. They are applied as a forward monthly fee drag
to simulated asset growth. Historical ETF price returns may already include
fund expenses, so applying an expense ratio is an explicit additional forward
assumption rather than a historical fee reconstruction.

Leverage is modeled with explicit borrowed balance and path-specific financing
cost when joint short-rate paths are available, with a deterministic fallback.
A leverage multiple above `1.0` requires an explicit rebalancing
frequency. Optional maintenance margin liquidates paths when equity divided by
asset value falls below the configured threshold. The model is monthly and
does not represent intramonth margin calls.

With joint macro paths, each monthly financing charge uses `simulated short
rate + financing spread + sensitivity * simulated inflation`. The financing
input therefore acts as a spread above the policy rate; without a stochastic
short rate it remains the fixed financing rate. For backward compatibility,
models without joint macro paths can still apply inflation sensitivity to each
regime's historical average inflation. The reported
`effective_financing_rate` is the average rate actually applied across the
simulated paths.

#### Country-neutral accounting and optional tax policies

Every run first produces a country-neutral gross ledger. If taxation is
enabled, the selected country policy consumes the exact same simulated market
paths and produces a second, path-dependent after-tax ledger. The API and UI
therefore report both gross and net terminal wealth plus their tax drag; with
taxation disabled, the gross and active ledgers are the same object and no
duplicate accounting pass is performed.

Country policies are registered through `TaxPolicy` in
`src/mc_quadrants/tax_policy.py`. A new country can supply validation,
path-dependent accounting, and metadata without modifying return simulation or
calibration. The request contract uses `tax_country` and `tax_regime`; legacy
requests that send only `tax_regime=italy_administered` remain compatible.
Italy is the only registered country for now.

Selecting Italy enables **administered**, **declarative**, or **managed** tax
timing. Market returns and regime calibration are unchanged. The administered
account settles tax on disposals, the declarative account accrues disposal tax
to the tax-year boundary, and the managed account taxes the annual portfolio
result. The account tracks average cost
per asset, realizes gains and losses on withdrawals and rebalancing sales,
funds the requested withdrawal plus its disposal tax, and can tax all remaining
unrealized gains through a final liquidation. Unused eligible losses are kept
for the current tax year and the next four tax years, oldest first. A simulation
start date controls real calendar boundaries and produces taxes-by-year output.
Modeled purchase and sale costs and optional Italian financial-transaction tax
adjust tax basis and disposal proceeds.

The default `FUND` category applies 26%. `GOVERNMENT_BOND` applies the
12.5%-equivalent taxable fraction. Funds can specify the actual eligible
government-security share instead of an all-or-nothing category. `FUND` models positive proceeds as
non-offsettable fund income while negative results enter the eligible loss
ledger. `GOVERNMENT_BOND_FUND` applies both approximations. Categories are
entered as `ASSET:CATEGORY`, for example
`SPY:FUND, BTP:GOVERNMENT_BOND`; unspecified assets default to `FUND`.
The current UI assumes accumulating ETFs / total-return series and therefore
does not expose dividend or coupon fields. The backward-compatible API can
still accept income and withholding metadata. Per-asset UI metadata covers
category, eligible government-security share, account location, and an
advanced financial-transaction-tax rate. The 0.20% levy applies as stamp duty to domestic accounts or
IVAFE to foreign accounts in automatic mode, avoiding double taxation.
Financial-transaction tax is deliberately instrument-specific rather than a
portfolio default: from 1 January 2026 the statutory share-transfer rate is
0.40%, normally reduced by half on regulated markets, while exemptions and the
separate derivatives schedule prevent a safe ticker-only inference. The UI
therefore asks for the applicable rate and otherwise assumes zero. The 2026
change is set by
[Law 199/2025, Article 1(29-31)](https://www.normattiva.it/atto/caricaDettaglioAtto?atto.codiceRedazionale=25G00212&atto.dataPubblicazioneGazzetta=2025-12-30&tipoDettaglio=originario).

These defaults follow the current statutory 26% rate in
[Decree-Law 66/2014, Article 3](https://www.normattiva.it/uri-res/N2Ls?urn%3Anir%3Astato%3Adecreto.legge%3A2014-04-24%3B66~art3=),
the 12.5% government-security treatment retained by
[Legislative Decree 461/1997](https://www.normattiva.it/atto/caricaDettaglioAtto?atto.articolo.numero=0&atto.codiceRedazionale=097G0497&atto.dataPubblicazioneGazzetta=1998-01-03&tabID=0.6902947756616171),
the four-subsequent-year loss rule in
[TUIR Article 68(6)](https://www.normattiva.it/uri-res/N2Ls?urn%3Anir%3Apresidente.repubblica%3Adecreto%3A1986-12-22%3B917~art68-com6=),
and the two-per-thousand financial-product levy in
[Decree-Law 201/2011, Article 19](https://www.normattiva.it/uri-res/N2Ls?urn%3Anir%3Astato%3Adecreto.legge%3A2011-12-06%3B201~art19-com15=).
The acquisition-cost basis is consistent with the current
[Agenzia delle Entrate capital-gains instructions](https://infoprecompilata.agenziaentrate.gov.it/portale/semplificata-mod-plusvalenze-natura-finanziaria).

This is intentionally not a tax-return engine. Declarative lot identification remains an average-cost
planning proxy; treaty eligibility, foreign-tax-credit limits, exemptions,
cash-account rules, and taxpayer-specific facts require professional review.
Leverage is disabled with Italian tax accounting, and the legacy weighted-return
mode is unavailable because neither exposes asset-level disposals or cost basis.
Rules are versioned as the `IT-2026` planning snapshot; simulated future years
hold that snapshot constant and the API displays an explicit warning.
Verify metadata and current rules with a qualified Italian tax adviser.

### 7. Reported Risk Metrics

Terminal wealth includes the mean, standard deviation, 5th/50th/95th
percentiles, and probability of finishing below the initial value. At 95%
confidence, VaR is `initial value - 5th percentile`, while expected shortfall
is `initial value - average wealth in the worst 5% tail`. Maximum drawdown is
calculated path-by-path from the initial value and each subsequent running
peak.

Annualized return and volatility are derived from the simulated periodic,
time-weighted portfolio returns: arithmetic mean is multiplied by periods per
year and periodic standard deviation by its square root. Sharpe, Sortino, and
Omega use the path-specific simulated short rate when available and otherwise
use the configured fallback risk-free rate. In real-wealth reports the nominal
short rate is converted into a path-consistent real rate using that path's
inflation. Contributions
enter at the start of a period and withdrawals leave at its end, and real cash
flows are inflation-adjusted exactly once. Terminal wealth percentiles, VaR,
expected shortfall, and probability of loss remain terminal-distribution
statistics.

Downside-focused metrics are also reported. The Ulcer Index is the square
root of the mean squared path drawdown, penalizing both depth and duration of
declines. The Sortino ratio divides excess return by annualized downside
deviation instead of total volatility. The Calmar ratio divides geometric
annualized return by the mean maximum drawdown. Geometric annualized return
compounds the mean logarithmic periodic return. Terminal skewness and excess
kurtosis describe the shape of the terminal distribution.

Planning metrics connect that risk distribution to an investor objective. A
configurable wealth target drives the probability of success and the conditional
expected shortfall among paths that miss it. The report also includes risk of
ruin, the Omega ratio relative to the configured risk-free rate, worst rolling
12-month returns, maximum time underwater, completed recovery time, and the
share of paths still underwater at the horizon.

Interactive risk charts complement the headline metrics. The goal-probability
curve maps possible wealth targets to terminal success rates. Rolling-horizon
bands show P05, median, and P95 annualized returns over the available holding
periods. The Drawdowns view includes an underwater probability fan, a bounded
depth-versus-duration episode scatter, and the gain required to recover the
previous peak. Portfolio comparison adds terminal quantile curves so dominance
can be inspected across the full distribution rather than inferred from one
average or percentile.

### 8. Important Assumptions

- Latest-revised FRED mode still contains revision look-ahead; select ALFRED initial releases for point-in-time analysis.
- The Bayesian macro ensemble and structural priors remain statistical and do not model the complete yield curve or monetary-policy reaction function.
- Parameter bootstrap quantifies historical estimation instability, not every possible future structural break.
- MNTS tail, GARCH, and ADCC specifications remain model assumptions and should be stress-tested.
- Transaction costs are charged only at modeled rebalancing events.
- Italian taxes are an optional simplified planning approximation, not tax advice or a filing calculation.
- Results are scenario estimates, not forecasts or investment advice.

### 9. Coherence With Market Reality

The model is designed to stay consistent with how long-term portfolios behave
in practice:

**Strengths**

- Regime conditioning captures the well-documented tendency for asset
  correlations and volatilities to change with the growth/inflation cycle,
  which a single full-sample covariance matrix misses.
- State-specific MNTS sampling produces skewed fat tails and common-subordinator
  joint extremes while preserving calibrated first and second moments.
- Rebalancing with transaction costs models the friction investors actually
  pay. Release alignment reduces timing bias, while ALFRED initial-release
  mode also avoids using later macro revisions.
- Correlation overrides allow investment views to be blended with empirical
  estimates when history is short or regimes are structurally different.

**Limitations and honest approximations**

- Monthly MNTS paths do not resolve intramonth jumps or liquidity spirals.
- Annualized volatility scales periodic dispersion by the square root of time;
  it does not claim that monthly returns are independent or normally distributed.
- Bond returns still come from historical price behavior rather than explicit
  duration and yield-curve factors.

**Long-term analysis features**

- Set **Contribution / period** to model dollar-cost averaging, then enable
  **Decumulation** to schedule retirement phases and one-time expenses. Cash
  flows are invested at (or funded pro-rata from) the target allocation and
  cannot drive wealth below zero. Use **Calculate safe rate** only when the
  solver is needed; ordinary manual analysis remains a single simulation.
- Set **Inflation assumption** above zero to report inflation-adjusted
  (purchasing power) wealth, VaR, drawdowns, and annualized metrics.
- Set **Fallback risk-free rate** for scenarios without joint stochastic rate
  paths. Joint macro scenarios use their simulated short rate automatically.
- The **Portfolio preset** picker applies established model allocations
  (60/40, Three-Fund, Permanent Portfolio, Golden Butterfly, All Seasons,
  Core Four, Risk Parity) mapped onto the loaded tickers. Approximations are
  labeled; for example IEF stands in for long-term treasuries and SHY for
  short-term/cash holdings.

The dashboard's **Methodology integrity** panel shows data-vintage quality,
release alignment, regime assignment, parameter recalibrations, joint macro
paths, dynamic dependence, and unconditional-MNTS benchmark performance separately.
It never treats a larger number of Monte Carlo paths as evidence that the
calibrated model itself is more certain.

## Install

```bash
cd mc-quadrant-sim
python -m pip install -e ".[dev]"
```

Optional data download helpers need:

```bash
python -m pip install -e ".[data]"
```

With `uv`, the checked-in lockfile provides a reproducible environment:

```bash
uv sync --extra dev --extra data
```

Build the C++17 production backend on macOS (including Apple Silicon) or Linux:

```bash
./scripts/build_native.sh
```

The native backend uses `std::thread`; the scenario `workers` setting controls
that thread pool. A fused MNTS-GARCH kernel generates one path at a time and
immediately updates gross, DIY, and optional wrapper ledgers, so no full asset-
return cube is retained. Large eligible tax simulations also generate the
four-state Markov or explicit-duration semi-Markov process inside the same C++
kernel. Joint macro simulations use that compact path as well: native workers
generate macro innovations, likelihood-conditioned regime transitions,
MNTS-GARCH returns, and portfolio ledgers one path at a time. Macro shocks are
streamed into returns instead of retained for every path. The kernel keeps every
terminal outcome, tax aggregate, regime count, and maximum drawdown, but keeps
complete monthly histories for only a deterministic 25,000-path reporting
sample. Detailed mode remains available when every path history is required. If the shared
library is missing or its ABI version does not match, execution automatically
falls back to the Python reference implementation. Set
`MC_DISABLE_NATIVE_SIM=1` to force that reference path for verification.

The production MNTS subordinator uses the exact two-dimensional
single-rejection Algorithm 3.1 of
[Qu, Dassios, and Zhao (2021)](https://doi.org/10.1145/3449357). The C++ engine
precomputes the four envelope constants for each quadrant and chooses the
smallest one. The exact [Devroye (2009)](https://doi.org/10.1145/1596519.1596523)
sampler and the former simple-rejection/Devroye hybrid remain callable for
distributional regression tests and reproducible performance comparisons.
The Python fallback continues to use Devroye's exact reference algorithm.

Automatic Italian-tax runs without active decumulation use one fused native
batch and up to eight threads by default. Tax totals, terminal statistics,
final-horizon wealth percentiles, goal success, shortfall, maximum drawdowns,
and ruin remain exact across every requested path. Intermediate chart bands
and advanced path diagnostics use the retained 25,000-path sample, preventing
reporting memory from dominating the simulation.

On the development machine, Qu's sampler won all 45 points in the calibrated
grid (`alpha=0.55..0.95`, `tempering=0.04..20`, 100,000 draws and three repeats
per point). Its median speedup was 5.86x versus Devroye and 3.13x versus the
former hybrid; the respective ranges were 3.13x-7.68x and 1.13x-7.68x. The
fused 360-month x 100,000-path x four-asset x four-quadrant benchmark fell from
3.14 seconds with the former hybrid to 2.62 seconds with Qu, a 16.7% end-to-end
reduction using eight native threads. After adding native regime generation and
compact reporting, the equivalent 500,000-path benchmark takes 10.89 seconds,
versus 13.97 seconds in detailed native mode: 22.0% less time. Estimated retained
history memory falls from 3.06 GB to 165 MB, a 94.6% reduction, while all-path
terminal estimates remain exact. These are hardware-specific engineering
benchmarks rather than runtime guarantees. Run
`scripts/benchmark_nts_samplers.py` for the sampler grid and
`scripts/benchmark_native.py` for the complete scenario. The constrained HSMM,
joint macro-membership, and latest-posterior comparison is reproducible with
`scripts/benchmark_regime_estimation.py`; use
`scripts/benchmark_joint_macro.py` for the Python optimizer, streamed native
path, distribution check, and retained-memory comparison. Pass `--repeats 1`
for a single large-scale measurement instead of the default median.

Then you can adapt:

```bash
python examples/calibrate_real_data.py
```

## Run The Web Backend

```bash
python web_app.py
```

The backend serves the web UI and the shared simulation API from real market
data. By default it loads live Yahoo Finance / FRED history; asset and macro
CSV uploads are supported as an optional override from the UI.

## Run The Gradio UI

```bash
python gradio_app.py
```

Open `http://127.0.0.1:7860` after the server starts. Set `PORT` to use a
different port. The optional data helpers need `.[data]`.

## Run The Streamlit UI

```bash
streamlit run streamlit_app.py
```

## Run The Web UI

The `web-ui-prod` branch provides the same simulation methodology through a plain
HTML/CSS/JavaScript interface and a small Python HTTP backend:

```bash
python web_app.py
```

Open `http://127.0.0.1:7860`. Set `PORT` to use a different port. The browser
client calls the shared `/api/load`, `/api/simulate`, and `/api/wealth`
contracts used by the other frontends.

The server uses adaptive path chunking, limits concurrent heavy jobs, caps request bodies, and sends a
deterministic reporting sample of at most 5,000 paths to the browser instead of serializing every path.
All requested paths still contribute to summary statistics, percentiles, and risk metrics.
Production limits can be adjusted with
`MAX_CONCURRENT_JOBS` and `MAX_REQUEST_BYTES`.

During longer jobs the web interface shows the active stage and elapsed time;
after completion it collapses the
settings, moves focus to the results, and provides an **Edit scenario** shortcut.
The allocation/status areas, metric cards, result tabs, and charts adapt to
mobile widths without causing page-level horizontal overflow.

For a faster first run, the web UI starts with the settings collapsed and a
single **Run analysis** action that loads market data and runs the model. New
sessions default to 100,000 paths over a 30-year, 360-month horizon.

Frontend network and resource-planning logic live in `web/api-client.js` and
`web/resource-planner.js`; `web/app.js` is responsible for application state,
controls, and rendering. Static asset references are relative, including the
logo, so they resolve when `web/index.html` is inspected directly with a
`file://` URL. The simulation APIs still require running `web_app.py`.

Open the URL printed by Streamlit (default `http://localhost:8501`). It
supports the same Yahoo Finance/FRED and CSV sources and the same
methodology controls, rendered with Altair charts. Charts are layout to the
browser width.

Both frontends delegate all data loading, scenario building, and result
shaping to the shared `mc_quadrants.api` layer, so the simulation methodology
is identical regardless of the interface. The **Model methodology** section
in each sidebar selects the regime model (quadrant or HMM), the regime
duration model (Markov benchmark or explicit-duration HSMM), the causal threshold window,
three-month smoothing, hysteresis, transition confirmation, probabilistic
moment weights, expected-return and duration-hazard shrinkage, parameter
recalibrations, joint macro paths, GARCH/ADCC dynamics, and walk-forward
validation.

The UIs load real data from Yahoo Finance/FRED by default and optionally
accept uploaded asset and macro CSVs. Yahoo mode starts at 1990 by default and
accepts optional proxy pairs such as `SPY:^GSPC, GLD:GC=F`. Select `IEF` or
`DBMF` in the synthetic asset picker, then choose the resulting `IEFSIM` or
`DBMFSIM` series for a backtest. Select a portfolio currency such as `EUR`
and optionally map assets with values such as `EFA:EUR`; Yahoo FX pairs are
loaded automatically. The correlation overrides section blends per-regime
targets for the first two selected tickers. Portfolio presets (60/40,
Three-Fund, Permanent, Golden Butterfly, All Seasons, Core Four, Risk Parity)
apply established model allocations to the loaded tickers, and the
inflation/risk-free inputs report real terms and a proper Sharpe ratio.
Periodic contributions and advanced decumulation model accumulation and
retirement in one horizon. The decumulation switch opens a phase/event editor,
safe-rate criterion, target probability, and guardrail controls; all dates are
bounded by the active simulation horizon.
The web UI uses an editorial research-studio layout:
an interactive four-regime hero, warm light and charcoal dark themes, scroll
progress, section reveals, responsive motion, and a compact allocation editor.
Tabbed result views cover Growth, Returns, Drawdowns, Correlations, Monthly
returns, Retirement, paired research, and diagnostics. The
Retirement tab reports the safe-rate curve, spending-survival curve, funded and
cumulative real-spending fans, cuts/increases, exhaustion, terminal capital,
paired fixed-versus-guardrail outcomes, and—when Italy is selected—annual tax,
gross-sale, and net-spending totals. Results also include metric cards,
wealth percentile curves, terminal wealth histograms, regime mix, transition
and correlation heatmaps, a monthly-return calendar, macro scatter,
calibration diagnostics, scenario comparison, and bounded CSV path samples.

The decision-reporting layer also provides survival, capital-preservation,
profit, and target-success probabilities through time; representative worst/P05/median/P95/best
paths with their regime histories; selectable metric distributions; and a
sequence-risk view comparing CAGR with money-weighted returns when contributions
are active. The **Research Lab** runs Portfolio B with the exact random seed and
market-path assumptions used for Portfolio A, reports paired differences and
path win rates, a Monte Carlo confidence interval for the mean paired terminal
difference, conditional regret, and an empirical quantile-dominance score. It
also compares goal, ruin, Omega, and drawdown-duration metrics; sweeps
monthly/quarterly/annual/buy-and-hold rebalancing; and
stores up to 20 named scenarios locally. Share links serialize the controls,
portfolio selection, weights, and seed so a run can be reconstructed without
embedding uploaded CSV contents.

Gradio
charts are rendered with Plotly; Streamlit charts use Altair.

## Testing And CI

Run the local checks with:

```bash
uv run ruff check .
uv run pytest -q
uv run python -m compileall -q src tests web_app.py
```

GitHub Actions runs these checks on Python 3.10, 3.11, and 3.12 for pull
requests and pushes to the production web branches, including `web-ui-prod`.
It also runs a Chromium/Playwright smoke test against the real Python web
server using uploaded CSV fixtures. That test completes a simulation, verifies
the results UI, checks for browser console errors, and asserts that the 390 px
mobile layout has no page-level horizontal overflow. A separate container job
builds the Docker image, starts it, and checks `/api/health`.

## Run With Docker

```bash
docker build -t mc-quadrant-sim .
docker run --rm -p 7860:7860 mc-quadrant-sim
```

Open `http://127.0.0.1:7860` after the container starts.

## Calibrate From Your Own CSVs

```python
from mc_quadrants.calibration import calibrate_quadrant_model
from mc_quadrants.data import prices_to_returns, read_macro_csv, read_prices_csv
from mc_quadrants.simulation import simulate_returns, simulate_portfolio_paths, summarize_terminal_wealth

prices = read_prices_csv("prices.csv")      # Date, SPY, IEF, GLD, DBC...
macro = read_macro_csv("macro.csv")         # Date, growth, inflation
returns = prices_to_returns(prices, method="log").resample("ME").sum()

model = calibrate_quadrant_model(
    returns=returns,
    macro=macro,
    growth_col="growth",
    inflation_col="inflation",
    macro_lag_periods=1,
    growth_threshold="median",
    inflation_threshold="median",
    threshold_window=12,
    probabilistic_regimes=True,
    regime_temperature=0.35,
    regime_smoothing_window=3,
    regime_hysteresis=0.15,
    regime_confirmation_periods=2,
    min_regime_duration=5,
    mean_prior_strength=24,
    joint_macro=True,
    correlation_overrides={
        "high_growth_high_inflation": {("SPY", "IEF"): 0.35},
        "low_growth_high_inflation": {("SPY", "IEF"): 0.25},
        "low_growth_low_inflation": {("SPY", "IEF"): -0.30},
    },
    override_weight=0.50,
)

result = simulate_returns(
    model,
    periods=360,
    paths=100000,
    random_seed=7,
    distribution="mnts",
    duration_model="semi_markov",
    min_regime_duration=5,
    garch=True,
    joint_macro=True,
    dynamic_correlation=True,
)
wealth = simulate_portfolio_paths(
    result,
    weights={"SPY": 0.55, "IEF": 0.30, "GLD": 0.10, "DBC": 0.05},
    rebalance_frequency=1,
    transaction_cost_bps=10,
    contribution=100.0,
    decumulation={
        "enabled": True,
        "mode": "manual",
        "phases": [{
            "start_month": 37,
            "end_month": 120,
            "frequency": "monthly",
            "annual_real_amount": 480.0,
        }],
        "policy": "guyton_klinger",
    },
    initial_value=100.0,
)
print(summarize_terminal_wealth(wealth))
```

For a reusable application workflow, `mc_quadrants.pipeline.run_scenario()`
returns the calibrated model, simulated paths, wealth, risk summary, and
calibration diagnostics together. Set `parameter_draws=8` and
`parameter_block_size=12` there to distribute paths across eight complete
stationary-bootstrap recalibrations; the returned `parameter_uncertainty`
table reports the resulting model-risk bands.

## Suggested Real Data Inputs

Asset prices can come from Yahoo Finance, Bloomberg, Refinitiv, your broker, or
flat CSVs. Macro inputs can come from FRED, OECD, World Bank, or internal data.
When an ETF does not have enough history, use a clearly labeled asset-class
proxy or upload a total-return history from a data vendor. Proxy backfills
should not be interpreted as the ETF's actual pre-inception performance.

Reasonable monthly macro choices:

- Growth: industrial production year-over-year, real GDP nowcast, PMI diffusion index, or unemployment gap.
- Inflation: CPI year-over-year, core CPI year-over-year, or inflation surprise.
- Short rate: effective federal funds rate, SOFR, or a three-month Treasury rate
  expressed as an annual level rather than a year-over-year change.

Using medians as thresholds gives balanced historical states. Using fixed
thresholds gives a more economic definition, for example growth above 0 and
inflation above 3 percent.

## Notes

- Correlations are estimated separately by quadrant.
- A covariance shrinkage parameter blends each quadrant estimate with the full-sample covariance. This helps when one quadrant has few observations.
- Correlation overrides are optional. They are useful when history is sparse or when you want to blend empirical estimates with an investment view.
- Every quadrant uses a calibrated MNTS return law with state-specific tails, skewness, volatility, and dependence.
- Parameter recalibrations use paired stationary-bootstrap blocks and refit the complete MNTS model; this measures parameter uncertainty rather than selecting another return law.
- A non-zero transition uncertainty setting samples the Markov matrix row-by-row only when parameter recalibration is disabled.
- Portfolio paths can model periodic rebalancing and transaction costs charged on traded notional. The default `rebalance_frequency=None` preserves the original weighted-log behavior.
- ALFRED initial-release mode and custom `AvailableDate` values provide point-in-time macro alignment; the release lag remains the fallback for latest-revised FRED history.
- Unit/integration, browser smoke, and Docker health checks run automatically
  through GitHub Actions.
