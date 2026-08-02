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

### 3. Markov Regime Model

The transition matrix counts adjacent historical regime changes and adds the
configured smoothing value to every cell before normalizing each row. The
default smoothing value is `1.0`, which prevents zero-probability transitions
when history is sparse.

When transition uncertainty is non-zero, each transition row is sampled from a
Dirichlet distribution. The dashboard maps uncertainty `u` in `[0, 1]` to a
row concentration of `max(1, 1 / u^2)`. Higher uncertainty therefore produces
more variation around the calibrated transition probabilities.

### 4. Regime-Specific Return Moments

Returns aligned to each regime provide a state-specific mean and covariance. If
a regime has fewer observations than `min_observations`, its estimates are
blended toward the full-sample estimates. Covariance matrices are additionally
shrunk toward the full-sample covariance, projected to the nearest positive
semidefinite matrix, and converted to correlations. Optional pairwise
correlation views are blended into each regime and projected back to a valid
correlation matrix.

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

### 8. Important Assumptions

- Macro release lag is period-based and does not model data revisions or exact publication dates.
- Regime transitions are Markovian and depend only on the current regime.
- Parametric draws do not model volatility clustering; bootstrap methods are the better choice when preserving historical shape matters.
- Transaction costs are charged only at modeled rebalancing events.
- Results are scenario estimates, not forecasts or investment advice.

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

## Run The Demo

```bash
mcq-demo
```

The demo uses synthetic history so it runs offline, but it exercises the same
calibration and simulation pipeline used for real data. It includes eight asset
classes and 420 monthly observations from January 1990 through December 2024.

## Run The Web UI

The UI is written in plain HTML, CSS, and JavaScript (no frontend framework,
no CDN dependencies) and is served by a small Python backend that wraps the
same simulation core:

```bash
python web_app.py
```

Open `http://127.0.0.1:7860` after the server starts. Set `PORT` to use a
different port. The optional data helpers need `.[data]`.

The web UI supports the offline demo, Yahoo Finance/FRED downloads, and
uploaded asset and macro CSVs. Yahoo mode starts at 1990 by default and
accepts optional proxy pairs such as `SPY:^GSPC, GLD:GC=F`. Select `IEF` or
`DBMF` in the synthetic asset picker, then choose the resulting `IEFSIM` or
`DBMFSIM` series for a backtest. Select a portfolio currency such as `EUR`
and optionally map assets with values such as `EFA:EUR`; Yahoo FX pairs are
loaded automatically. Results include metric cards, wealth percentile curves,
terminal wealth histograms, regime mix, transition and correlation heatmaps,
macro scatter, calibration diagnostics, scenario comparison, and CSV
downloads. Charts are rendered client-side as SVG.

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
