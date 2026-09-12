#!/usr/bin/env python3
"""Offline default-like API benchmark, including calibration and reporting.

Run each worker/path configuration in a fresh process for comparable peak RSS.
Repeated runs in one process also measure bootstrap calibration cache reuse.
Data downloads and walk-forward validation are excluded unless requested.
"""

from __future__ import annotations

import argparse
import cProfile
import gc
import json
import pstats
import resource
import sys
import time

import numpy as np
import pandas as pd

from mc_quadrants import api


def benchmark_payload(paths: int, periods: int, draws: int, workers: int) -> dict:
    # Match the API regression fixture used for the before/after measurements.
    rng = np.random.default_rng(7)
    dates = pd.Index(pd.date_range("2010-01-31", periods=120, freq="ME"), name="Date")
    prices = pd.DataFrame(
        {
            ticker: 100 * np.exp(np.cumsum(rng.normal(0.004, 0.02, len(dates))))
            for ticker in ["SPY", "IEF", "GLD", "DBC", "EFA", "VNQ", "TIP", "SHY"]
        },
        index=dates,
    )
    macro = pd.DataFrame(
        {
            "growth": rng.normal(0.0, 0.6, len(dates)),
            "inflation": rng.normal(0.0, 0.4, len(dates)),
            "interest_rate": np.clip(rng.normal(3.0, 1.0, len(dates)), 0.0, None),
        },
        index=dates,
    )
    return {
        "source": "csv",
        "csv_prices": prices.reset_index().to_csv(index=False),
        "csv_macro": macro.reset_index().to_csv(index=False),
        "asset_input": "Price levels", "monthly": True,
        "growth_col": "growth", "inflation_col": "inflation", "rate_col": "interest_rate",
        "selected_tickers": ["SPY", "IEF", "GLD", "DBC"],
        "weights": {"SPY": 40, "IEF": 20, "GLD": 10, "DBC": 10},
        "paths": paths, "periods": periods, "workers": workers,
        "random_seed": 7, "initial_value": 100.0, "target_wealth": 200.0,
        "base_currency": "USD", "currency_map": "",
        "growth_threshold": "median", "inflation_threshold": "median",
        "macro_lag": 1, "transition_uncertainty": 0, "distribution": "mnts",
        "rebalance": "monthly", "cost_bps": 10, "start_state": "Stationary",
        "parameter_draws": draws, "parameter_block_size": 12,
        "joint_macro": True, "dynamic_correlation": True, "structural_returns": True,
        "macro_parameter_uncertainty": True, "macro_model": "bvar_ensemble",
        "threshold_window": 12, "probabilistic_regimes": True, "mean_prior_strength": 24,
        "duration_model": "semi_markov", "walk_forward": False,
        "tax_country": "none", "tax_regime": "none",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=int, default=500_000)
    parser.add_argument("--periods", type=int, default=360)
    parser.add_argument("--draws", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--walk-forward", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    payload = benchmark_payload(args.paths, args.periods, args.draws, args.workers)
    payload["walk_forward"] = args.walk_forward
    for repeat in range(args.repeats):
        gc.collect()
        profiler = cProfile.Profile() if args.profile else None
        started = time.perf_counter()
        response = (
            profiler.runcall(api.build_simulate_response, payload)
            if profiler else api.build_simulate_response(payload)
        )
        elapsed = time.perf_counter() - started
        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_bytes = peak_rss if sys.platform == "darwin" else peak_rss * 1024
        print(json.dumps({
            "run": repeat + 1, "seconds": elapsed, "peak_parent_rss_gb": peak_bytes / 1e9,
            "resources": response["resources"], "parameter_draws": args.draws,
            "walk_forward": args.walk_forward, "downloads": False,
        }), flush=True)
        if profiler:
            pstats.Stats(profiler).strip_dirs().sort_stats("cumulative").print_stats(20)
        del response


if __name__ == "__main__":
    main()
