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
uv sync --extra dev --extra gradio --extra data
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
calibration and simulation pipeline used for real data.

## Run The Streamlit Dashboard

```bash
python -m pip install -e ".[app]"
streamlit run streamlit_app.py
```

The dashboard starts with the same offline demo data and can also calibrate from
uploaded CSV files. Asset selection is ticker-based: choose the tickers in the
sidebar, then enter weights only for those selected tickers before running the
simulation.

## Run The Gradio App

```bash
python -m pip install -e ".[gradio,data]"
python gradio_app.py
```

The Gradio app supports the offline demo, Yahoo Finance/FRED downloads, and
uploaded asset and macro CSVs. Uploading a macro CSV populates the growth and
inflation column selectors from its headers. It also exposes release-lag
handling, Gaussian/Student-t/bootstrap sampling, transition uncertainty,
rebalancing costs, risk metrics, diagnostics, scenario comparison, and CSV
downloads.

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
