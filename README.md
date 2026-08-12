# MC Quadrant Simulator

A research-oriented Monte Carlo scenario engine built around the classic four
macro quadrants:

| Regime | Growth | Inflation | Typical interpretation |
| --- | --- | --- | --- |
| `high_growth_low_inflation` | High | Low | Goldilocks / disinflationary expansion |
| `high_growth_high_inflation` | High | High | Overheating expansion |
| `low_growth_high_inflation` | Low | High | Stagflation |
| `low_growth_low_inflation` | Low | Low | Recession / deflationary slowdown |

The model is designed to be calibrated from real historical data:

1. Release-aware macro data initializes semantically identified growth/inflation quadrants.
2. An explicit-duration hidden semi-Markov model jointly estimates latent states, state-age hazards, and exit destinations.
3. Parametric return means are shrunk and covariance matrices use Ledoit-Wolf shrinkage.
4. Optional stationary-bootstrap recalibrations measure parameter uncertainty.
5. Growth, inflation, the short rate, regimes, returns, dynamic correlations, and portfolio accounting are simulated together.
6. Walk-forward validation compares the model with Gaussian and Student-t benchmarks.

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
logistic quadrant probabilities remain part of the joint macro simulator, with
width controlled by `regime_temperature`. They are deliberately not multiplied
as independent adjacent memberships to estimate historical transitions.

### 3. Explicit-Duration Hidden Semi-Markov Model

The default quadrant model is a Gaussian-emission hidden semi-Markov model
(HSMM). Growth and inflation are the observed emissions, while the hidden state
is expanded to `(quadrant, months in quadrant)`. The persistent classifier fixes
the economic identity of the four emission distributions, but it does not
provide the final transition counts.

A scaled forward-backward pass returns filtered state probabilities, smoothed
state-age posteriors, and joint expected transitions. EM updates two distinct
objects from those joint posteriors:

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
destination probabilities at each eligible exit. Consequently, simulated
transitions depend on both regime age and current macro conditions rather than
on age alone.

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

Each simulated period first draws a regime, then draws asset returns using that
regime's parameters:

- `normal`: multivariate Gaussian sampling.
- `student_t`: a scale-mixture multivariate Student-t draw. The covariance is
  rescaled so finite degrees of freedom preserve the calibrated covariance.
- `bootstrap`: samples historical returns observed in the selected regime.
- `block_bootstrap`: samples short consecutive blocks from regime history,
  preserving some local return structure while a path remains in a regime.

Student-t degrees of freedom must be greater than `2` so the variance is
finite. Bootstrap methods require historical returns saved during calibration.

**GARCH(1,1) volatility clustering** adds conditional-variance dynamics within
each regime (Gaussian draws only): every asset's variance follows
`h_t = omega + alpha * eps_{t-1}^2 + beta * h_{t-1}` anchored so the
unconditional level matches the regime covariance, and the variance re-anchors
when a path enters a new regime. Shocks therefore cluster in time without
drifting away from the calibrated regime covariance. `garch_alpha` governs
responsiveness to new shocks (default 0.10) and `garch_beta` the persistence
of past variance (default 0.85).

**Asymmetric dynamic correlation (ADCC)** evolves the correlation matrix after
each standardized shock. `dcc_alpha` controls shock response, `dcc_beta`
controls persistence, and `dcc_asymmetry` increases the response to joint
negative shocks. ADCC works with Normal or Student-t returns and re-anchors to
the relevant regime correlation after a state change.

**Joint macro-financial paths** fit a regularized VAR(1) to growth, inflation,
and the effective federal funds rate when it is available, together with
regime-conditioned innovation covariances and a ridge return-factor link.
Simulated macro values influence time-varying transition probabilities, while
the same macro innovations—including short-rate changes—affect asset returns.
Only growth and inflation define quadrant membership; the rate is an additional
state variable. Inflation creates a different purchasing-power deflator for
every path, and the simulated short rate supplies the path-specific risk-free
benchmark and leveraged financing base. This compact model improves internal
consistency but is not a structural macroeconomic forecast, a Taylor-rule
model, or a full yield-curve model.

**Walk-forward validation** (quadrant model only, enabled by default in the
dashboard) checks the regime model and selected portfolio strictly out of sample: each split fits on
data up to period `t` and scores the next observation under the one-step
regime mixture density versus unconditional Gaussian and Student-t models fitted
on the same history. It also reports transition Brier and log scores, observed
versus predicted switches per decade, completed-duration log scores and errors,
stability of expected durations across rolling calibration vintages,
actual-state probabilities, portfolio probability-integral-transform
diagnostics, 95% VaR breach frequency and clustering, and a Newey-West/HAC
log-score comparison. The UI shows expected months in every state and expected
switches per decade, and warns when the implied persistence is unusually low.
Weak or miscalibrated results are surfaced as warnings rather than hidden.

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

Periodic cash flows are supported in both accounting modes. A **contribution**
is invested at the target allocation at the start of every period, which is
dollar-cost averaging: new money buys the same diversified weights regardless
of recent performance. A **withdrawal** is funded at the end of every period by
selling a pro-rata slice of current holdings, so funding never concentrates in
one asset class between rebalances. In legacy mode the same schedule is applied
to the blended portfolio return. Wealth paths are floored at zero, so a
withdrawal larger than the remaining balance simply exhausts the portfolio.
Cash-flow scenarios suit retirement-style analysis: accumulation phases use a
positive contribution, drawdown phases a positive withdrawal, and the
difference between terminal wealth and total contributed shows how much of the
outcome came from returns rather than savings.

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

Selecting **Italy — simplified administered regime** applies an after-tax
planning approximation. Market returns and regime calibration are unchanged.
The account tracks average cost
per asset, realizes gains and losses on withdrawals and rebalancing sales,
funds the requested withdrawal plus its disposal tax, and can tax all remaining
unrealized gains through a final liquidation. Unused eligible losses are kept
for the current tax year and the next four tax years, oldest first. Because a
scenario has no calendar start date, each consecutive 12-month block is treated
as a tax year. Modeled purchase and sale transaction costs adjust tax basis and
disposal proceeds.

The default `STANDARD` category applies 26%. `GOVERNMENT_BOND` applies the
12.5%-equivalent taxable fraction. `FUND` models positive proceeds as
non-offsettable fund income while negative results enter the eligible loss
ledger. `GOVERNMENT_BOND_FUND` applies both approximations. Categories are
entered as `ASSET:CATEGORY`, for example
`SPY:FUND, BTP:GOVERNMENT_BOND`; unspecified assets default to `STANDARD`.
The annual stamp-duty/IVAFE proxy defaults to 0.20% and is applied monthly.

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

This is intentionally not a tax-return engine. It does not distinguish every
ETF domicile, UCITS status, government-bond percentage, intermediary regime,
dividend/coupon component, account-level exemption, or individual taxpayer
circumstance. Historical adjusted returns combine price and distributions, so
the simulator does not separately tax dividends or coupons. The 0.20% input is
one configurable account-level proxy—not a claim that stamp duty and IVAFE are
both due. Leverage is disabled with Italian tax accounting, and the legacy
weighted-return mode is unavailable because neither exposes asset-level
disposals or cost basis. Verify classifications and current rules with a
qualified Italian tax adviser.

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
- The joint macro VAR is statistical rather than structural and does not model the complete yield curve or monetary-policy reaction function.
- Parameter bootstrap quantifies historical estimation instability, not every possible future structural break.
- Parametric tail and ADCC specifications remain model assumptions; empirical bootstrap remains a useful benchmark.
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
- Student-t and bootstrap/block-bootstrap sampling produce fat tails and
  extreme outcomes instead of assuming Gaussian returns.
- Rebalancing with transaction costs models the friction investors actually
  pay. Release alignment reduces timing bias, while ALFRED initial-release
  mode also avoids using later macro revisions.
- Correlation overrides allow investment views to be blended with empirical
  estimates when history is short or regimes are structurally different.

**Limitations and honest approximations**

- Student-t tails are symmetric and monthly paths do not model intramonth jumps.
- Annualized volatility scales periodic dispersion by the square root of time;
  it does not claim that monthly returns are independent or normally distributed.
- Bond returns still come from historical price behavior rather than explicit
  duration and yield-curve factors.

**Long-term analysis features**

- Set **Contribution / period** to model dollar-cost averaging into the
  portfolio, and **Withdrawal / period** to model retirement-style drawdowns.
  Cash flows are invested at (or funded pro-rata from) the target allocation
  every period and cannot drive wealth below zero.
- Set **Inflation assumption** above zero to report inflation-adjusted
  (purchasing power) wealth, VaR, drawdowns, and annualized metrics.
- Set **Fallback risk-free rate** for scenarios without joint stochastic rate
  paths. Joint macro scenarios use their simulated short rate automatically.
- The **Portfolio preset** picker applies PortfolioCharts-style allocations
  (60/40, Three-Fund, Permanent Portfolio, Golden Butterfly, All Seasons,
  Core Four, Risk Parity) mapped onto the loaded tickers. Approximations are
  labeled; for example IEF stands in for long-term treasuries and SHY for
  short-term/cash holdings.

The dashboard's **Methodology integrity** panel shows data-vintage quality,
release alignment, regime assignment, parameter recalibrations, joint macro
paths, dynamic dependence, and Student-t benchmark performance separately.
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
client calls the same `/api/load`, `/api/simulate`, `/api/compare`, and
`/api/wealth` payload contracts used by the other frontends.

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
sessions default to 100,000 paths over a ten-year, 120-month horizon.

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
apply PortfolioCharts-style allocations to the loaded tickers, and the
inflation/risk-free inputs report real terms and a proper Sharpe ratio.
Periodic contributions and withdrawals model dollar-cost averaging and
retirement drawdowns. The web UI uses an editorial research-studio layout:
an interactive four-regime hero, warm light and charcoal dark themes, scroll
progress, section reveals, responsive motion, and a compact allocation editor.
Tabbed result views cover Growth, Returns, Drawdowns, Correlations, Monthly
returns, paired research, and distribution comparison. Results include metric cards,
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
    periods=120,
    paths=5000,
    random_seed=7,
    distribution="student_t",
    degrees_of_freedom=5,
    duration_model="semi_markov",
    min_regime_duration=5,
    joint_macro=True,
    dynamic_correlation=True,
)
wealth = simulate_portfolio_paths(
    result,
    weights={"SPY": 0.55, "IEF": 0.30, "GLD": 0.10, "DBC": 0.05},
    rebalance_frequency=1,
    transaction_cost_bps=10,
    contribution=100.0,
    withdrawal=0.0,
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
- Returns can be sampled from either a Gaussian or finite-variance Student-t distribution within each quadrant. Lower Student-t degrees of freedom create heavier tails.
- Historical and block bootstrap sampling preserve observed regime-specific return shapes and unusual outcomes.
- Parameter recalibrations use paired stationary-bootstrap blocks and refit the complete parametric model; this is distinct from empirical return-path bootstrap sampling.
- A non-zero transition uncertainty setting samples the Markov matrix row-by-row only when parameter recalibration is disabled.
- Portfolio paths can model periodic rebalancing and transaction costs charged on traded notional. The default `rebalance_frequency=None` preserves the original weighted-log behavior.
- ALFRED initial-release mode and custom `AvailableDate` values provide point-in-time macro alignment; the release lag remains the fallback for latest-revised FRED history.
- Unit/integration, browser smoke, and Docker health checks run automatically
  through GitHub Actions.
