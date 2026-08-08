---
title: Four-Quadrant Monte Carlo Simulator
emoji: 📈
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# MC Quadrant Simulator

A starter Python project for a Monte Carlo simulator built around the classic
four macro quadrants:

| Regime | Growth | Inflation | Typical interpretation |
| --- | --- | --- | --- |
| `high_growth_low_inflation` | High | Low | Goldilocks / disinflationary expansion |
| `high_growth_high_inflation` | High | High | Overheating expansion |
| `low_growth_high_inflation` | Low | High | Stagflation |
| `low_growth_low_inflation` | Low | Low | Recession / deflationary slowdown |

The model is designed to be calibrated from real historical data:

1. Macro data is classified into one of the four quadrants.
2. A Markov transition matrix is estimated from observed quadrant changes.
3. Asset returns are grouped by quadrant.
4. Each quadrant receives its own expected returns, volatility, covariance, and correlation matrix.
5. Monte Carlo paths draw the next quadrant from the transition matrix and sample asset returns from that quadrant's distribution.

## Methodology

### 1. Data And Alignment

Asset prices are converted to log returns and, for the dashboard market-data
path, aggregated to monthly frequency. FRED industrial production and CPI are
converted to year-over-year percentage changes. Uploaded macro CSVs are assumed
to already contain the growth and inflation measures selected by the user.

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

Macro observations are classified before they are joined to asset returns. The
dashboard defaults to a one-period macro release lag: a macro regime observed
at period `t` is used for asset returns beginning at `t + 1`. Remaining dates
are aligned with forward-fill. This is a conservative approximation, not a
full real-time vintage or release-calendar database.

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

### 3. Markov Regime Model

The transition matrix counts adjacent historical regime changes and adds the
configured smoothing value to every cell before normalizing each row. The
default smoothing value is `1.0`, which prevents zero-probability transitions
when history is sparse.

When transition uncertainty is non-zero, each transition row is sampled from a
Dirichlet distribution. The dashboard maps uncertainty `u` in `[0, 1]` to a
row concentration of `max(1, 1 / u^2)`. Higher uncertainty therefore produces
more variation around the calibrated transition probabilities.

A first-order Markov chain implies geometrically distributed regime run
lengths, which understates how long real regimes persist. With **semi-Markov
durations** the simulator draws each stay length from the empirical sojourn
distribution observed in history (stored in the calibrated model's metadata),
then transitions to a different state via the matrix with self-transitions
renormalized away. The dashboard defaults to semi-Markov; the core API uses
the plain chain unless `duration_model="semi_markov"` is requested.

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

**Walk-forward validation** (quadrant model only, enabled by default in the
dashboard) checks the regime model strictly out of sample: each split fits on
data up to period `t` and scores the next observation under the one-step
regime mixture density versus an unconditional Gaussian fitted on the same
history. The reported advantage (log-likelihood units per period) and the
one-step regime hit rate show whether regime conditioning actually predicts
returns; a non-positive advantage is surfaced as a warning rather than hidden.

### 6. Portfolio Accounting

Weights are normalized to sum to one. The legacy mode combines simulated log
returns with the weighted-log approximation. Rebalancing mode instead tracks
asset holdings, applies each asset's gross return, and rebalances monthly,
quarterly, or annually. Transaction costs are charged as:

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

Leverage is modeled with explicit borrowed balance and deterministic financing
cost. A leverage multiple above `1.0` requires an explicit rebalancing
frequency. Optional maintenance margin liquidates paths when equity divided by
asset value falls below the configured threshold. The model is monthly and
does not represent intramonth margin calls.

### 7. Reported Risk Metrics

Terminal wealth includes the mean, standard deviation, 5th/50th/95th
percentiles, and probability of finishing below the initial value. At 95%
confidence, VaR is `initial value - 5th percentile`, while expected shortfall
is `initial value - average wealth in the worst 5% tail`. Maximum drawdown is
calculated path-by-path from the initial value and each subsequent running
peak.

Annualized metrics are derived from the terminal distribution: the annualized
return scales the mean terminal growth to one year, the annualized volatility
scales the terminal standard deviation by the square root of time, and the
Sharpe ratio uses a zero risk-free rate. These are approximations that assume
independent, identically distributed monthly returns, not a full time-series
return decomposition.

Downside-focused metrics are also reported. The Ulcer Index is the square
root of the mean squared path drawdown, penalizing both depth and duration of
declines. The Sortino ratio divides excess return by annualized downside
deviation instead of total volatility. The Calmar ratio divides annualized
return by the mean maximum drawdown. The geometric annualized return
compounds the mean logarithmic terminal growth and is always lower than or
equal to the arithmetic annualized return. Terminal skewness and excess
kurtosis describe the shape of the terminal distribution.

### 8. Important Assumptions

- Macro release lag is period-based and does not model data revisions or exact publication dates.
- Regime transitions are Markovian and depend only on the current regime.
- Parametric draws do not model volatility clustering unless optional GARCH is enabled; bootstrap methods are the better choice when preserving historical shape matters.
- Transaction costs are charged only at modeled rebalancing events.
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
  pay, and the macro release lag removes obvious look-ahead bias.
- Correlation overrides allow investment views to be blended with empirical
  estimates when history is short or regimes are structurally different.

**Limitations and honest approximations**

- Returns are drawn from static regime distributions; volatility clustering,
  skewness, and regime-switching within a quarter are not modeled.
- The annualized volatility estimate scales terminal dispersion by the square
  root of time and assumes independent monthly returns.
- Markov probabilities and regime moments are estimated from the available
  history; sparse regimes are blended toward the full sample.
- Deterministic inflation and risk-free assumptions are constant, not
  stochastic. A positive inflation assumption expresses results in real terms;
  the default is nominal.

**Long-term analysis features**

- Set **Contribution / period** to model dollar-cost averaging into the
  portfolio, and **Withdrawal / period** to model retirement-style drawdowns.
  Cash flows are invested at (or funded pro-rata from) the target allocation
  every period and cannot drive wealth below zero.
- Set **Inflation assumption** above zero to report inflation-adjusted
  (purchasing power) wealth, VaR, drawdowns, and annualized metrics.
- Set **Risk-free rate** to compute a proper Sharpe ratio instead of a zero
  risk-free baseline.
- The **Portfolio preset** picker applies PortfolioCharts-style allocations
  (60/40, Three-Fund, Permanent Portfolio, Golden Butterfly, All Seasons,
  Core Four, Risk Parity) mapped onto the loaded tickers. Approximations are
  labeled; for example IEF stands in for long-term treasuries and SHY for
  short-term/cash holdings.

**Future directions**

- Regime-dependent Student-t degrees of freedom (per-regime tail shapes).
- Bond duration and yield-curve simulation instead of price-only histories.
- Inflation-indexed cash flows, where contributions and withdrawals grow with
  the inflation assumption.

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

The `web-ui` branch provides the same simulation methodology through a plain
HTML/CSS/JavaScript interface and a small Python HTTP backend:

```bash
python web_app.py
```

Open `http://127.0.0.1:7860`. Set `PORT` to use a different port. The browser
client calls the same `/api/load`, `/api/simulate`, `/api/compare`, and
`/api/wealth` payload contracts used by the other frontends.

Open the URL printed by Streamlit (default `http://localhost:8501`). It
supports the same Yahoo Finance/FRED and CSV sources and the same
methodology controls, rendered with Altair charts. Charts are layout to the
browser width.

Both frontends delegate all data loading, scenario building, and result
shaping to the shared `mc_quadrants.api` layer, so the simulation methodology
is identical regardless of the interface. The **Model methodology** section
in each sidebar selects the regime model (quadrant or HMM), the regime
duration model (Markov chain or semi-Markov), the causal threshold window,
GARCH volatility clustering, and walk-forward validation.

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
retirement drawdowns. Results include metric cards,
wealth percentile curves, terminal wealth histograms, regime mix, transition
and correlation heatmaps, macro scatter, calibration diagnostics, scenario
comparison, and CSV downloads. Gradio charts are rendered with Plotly;
Streamlit charts use Altair.

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
calibration diagnostics together.

## Suggested Real Data Inputs

Asset prices can come from Yahoo Finance, Bloomberg, Refinitiv, your broker, or
flat CSVs. Macro inputs can come from FRED, OECD, World Bank, or internal data.
When an ETF does not have enough history, use a clearly labeled asset-class
proxy or upload a total-return history from a data vendor. Proxy backfills
should not be interpreted as the ETF's actual pre-inception performance.

Reasonable monthly macro choices:

- Growth: industrial production year-over-year, real GDP nowcast, PMI diffusion index, or unemployment gap.
- Inflation: CPI year-over-year, core CPI year-over-year, or inflation surprise.

Using medians as thresholds gives balanced historical states. Using fixed
thresholds gives a more economic definition, for example growth above 0 and
inflation above 3 percent.

## Notes

- Correlations are estimated separately by quadrant.
- A covariance shrinkage parameter blends each quadrant estimate with the full-sample covariance. This helps when one quadrant has few observations.
- Correlation overrides are optional. They are useful when history is sparse or when you want to blend empirical estimates with an investment view.
- Returns can be sampled from either a Gaussian or finite-variance Student-t distribution within each quadrant. Lower Student-t degrees of freedom create heavier tails.
- Historical and block bootstrap sampling preserve observed regime-specific return shapes and unusual outcomes.
- A non-zero transition uncertainty setting samples the Markov matrix row-by-row from Dirichlet distributions.
- Portfolio paths can model periodic rebalancing and transaction costs charged on traded notional. The default `rebalance_frequency=None` preserves the original weighted-log behavior.
- Macro release lags shift regime labels before calibrating asset moments, reducing same-period look-ahead bias.
- `pytest` runs automatically through GitHub Actions on supported Python versions.
