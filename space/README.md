---
title: Four-Quadrant Monte Carlo Simulator
emoji: 📈
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.22.0
app_file: gradio_app.py
pinned: false
---

# Four-Quadrant Monte Carlo Simulator

A Monte Carlo simulator built around the classic four macro quadrants:

| Regime | Growth | Inflation | Typical interpretation |
| --- | --- | --- | --- |
| `high_growth_low_inflation` | High | Low | Goldilocks / disinflationary expansion |
| `high_growth_high_inflation` | High | High | Overheating expansion |
| `low_growth_high_inflation` | Low | High | Stagflation |
| `low_growth_low_inflation` | Low | Low | Recession / deflationary slowdown |

## Features

- **Demo mode** - runs offline with synthetic data
- **Yahoo Finance** - fetch real asset prices and FRED macro data
- **Historical proxies** - extend pre-inception history with explicit, scaled Yahoo series
- **Synthetic series** - source-labeled `ASSETSIM` backtests with reproducible pre-inception segments
- **Currency conversion** - historical FX returns aligned to the selected portfolio currency
- **CSV upload** - calibrate from your own data
- **Markov transition matrix** - estimated from historical regime changes
- **Regime-specific asset moments** - expected returns, volatility, covariance, and correlation per quadrant
- **Monte Carlo simulation** - draws regime paths and samples asset returns
- **Return models** - Gaussian, Student-t, historical bootstrap, and block bootstrap
- **Portfolio analysis** - wealth percentiles, downside risk, drawdown, regime mix, and costs
- **Diagnostics and exports** - calibration warnings, scenario comparisons, and CSV downloads

## How to use

1. Choose a data source (Demo, Yahoo Finance, or CSV upload)
2. Select portfolio tickers and set weights
3. Configure calibration thresholds and simulation parameters
4. Click **Run Simulation** to see results
5. Use the download controls to export wealth paths, risk summaries, and diagnostics

## Local development

```bash
cd mc-quadrant-sim
python -m pip install -e ".[gradio,data]"
python gradio_app.py
