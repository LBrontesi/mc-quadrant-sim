#!/usr/bin/env python3
"""Benchmark Python and streamed-native joint macro simulation paths."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time

import numpy as np
import pandas as pd

from mc_quadrants.calibration import calibrate_quadrant_model
from mc_quadrants.simulation import simulate_joint_regime_macro_paths, simulate_returns
from mc_quadrants.taxes import prepare_italian_native_configuration


def _model_fixture():
    rng = np.random.default_rng(81)
    dates = pd.date_range("1980-01-31", periods=360, freq="ME")
    phases = np.resize(np.repeat(np.arange(4), 15), len(dates))
    centers = np.array([[2.4, 1.6], [2.1, 4.1], [-1.0, 4.3], [-0.8, 1.4]])
    macro_values = centers[phases] + rng.multivariate_normal(
        [0.0, 0.0],
        [[0.28, 0.08], [0.08, 0.35]],
        size=len(dates),
    )
    macro = pd.DataFrame(macro_values, index=dates, columns=["growth", "inflation"])
    asset_noise = rng.multivariate_normal(
        np.zeros(4),
        [
            [0.0016, -0.00012, 0.00010, 0.00015],
            [-0.00012, 0.00045, 0.00005, -0.00003],
            [0.00010, 0.00005, 0.00090, 0.00012],
            [0.00015, -0.00003, 0.00012, 0.00120],
        ],
        size=len(dates),
    )
    returns = pd.DataFrame(
        asset_noise
        + np.column_stack(
            (
                0.004 + 0.002 * macro_values[:, 0],
                0.002 - 0.001 * macro_values[:, 1],
                0.002 - 0.0005 * macro_values[:, 0] + 0.0015 * macro_values[:, 1],
                0.001 + 0.001 * macro_values[:, 0] + 0.001 * macro_values[:, 1],
            )
        ),
        index=dates,
        columns=["Stocks", "Bonds", "Gold", "Commodities"],
    )
    return calibrate_quadrant_model(
        returns,
        macro,
        growth_threshold=0.5,
        inflation_threshold=2.7,
        min_observations=8,
        joint_macro=True,
    )


def _portfolio_configuration(model, periods: int) -> dict[str, object]:
    configuration = prepare_italian_native_configuration(
        periods=periods,
        assets=model.assets,
        target_weights=np.full(len(model.assets), 1.0 / len(model.assets)),
        initial_value=100.0,
        rebalance_frequency=3,
        transaction_cost_bps=5.0,
        contribution=0.0,
        contribution_allocation="pro_rata",
        withdrawal=0.0,
        withdrawal_start_period=1,
        asset_tax_categories=None,
        asset_tax_metadata=None,
        annual_wealth_tax=0.002,
        terminal_liquidation=True,
        tax_regime="italy_administered",
        wealth_tax_mode="stamp_duty",
        start_date=None,
        wrapper_benchmark=True,
    )
    configuration["expense_ratios"] = np.zeros(len(model.assets))
    configuration["return_kind"] = "log"
    return configuration


def _macro_summary(result) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    regimes = np.asarray(result.regimes, dtype=int)
    frequencies = np.bincount(regimes.ravel(), minlength=4) / regimes.size
    macro = np.asarray(result.macro_paths, dtype=float)
    return frequencies, macro.mean(axis=(0, 1)), macro.std(axis=(0, 1))


def benchmark(
    periods: int,
    paths: int,
    reporting_paths: int,
    workers: int,
    repeats: int,
) -> dict[str, object]:
    if min(periods, paths, reporting_paths, workers, repeats) < 1:
        raise ValueError("benchmark sizes, workers, and repeats must be positive")
    model = _model_fixture()
    base_configuration = _portfolio_configuration(model, periods)

    python_macro_timings = []
    for repeat in range(repeats + 1):
        gc.collect()
        started = time.perf_counter()
        macro_result = simulate_joint_regime_macro_paths(
            model,
            periods=periods,
            paths=paths,
            random_seed=300 + repeat,
            duration_model="semi_markov",
            macro_parameter_uncertainty=True,
            return_codes=True,
        )
        python_macro_timings.append(time.perf_counter() - started)
        del macro_result

    execution_results = {}
    final_results = {}
    for name, compact in (
        ("python_macro_detailed_native", False),
        ("streamed_compact_native", True),
    ):
        timings = []
        for repeat in range(repeats + 1):
            gc.collect()
            configuration = {
                **base_configuration,
                "compact_reporting": compact,
                "compact_reporting_paths": reporting_paths,
            }
            started = time.perf_counter()
            result = simulate_returns(
                model,
                periods=periods,
                paths=paths,
                random_seed=500 + repeat,
                duration_model="semi_markov",
                joint_macro=True,
                macro_parameter_uncertainty=True,
                native_threads=workers,
                return_regime_codes=True,
                native_portfolio_config=configuration,
            )
            timings.append(time.perf_counter() - started)
            final_results[name] = result
        execution_results[name] = {
            "runs_seconds": timings,
            "median_seconds_excluding_warmup": float(statistics.median(timings[1:])),
        }

    detailed_frequency, detailed_mean, detailed_std = _macro_summary(
        final_results["python_macro_detailed_native"]
    )
    compact_frequency, compact_mean, compact_std = _macro_summary(
        final_results["streamed_compact_native"]
    )
    dimensions = len(model.metadata["macro_dynamics"]["columns"])
    retained_paths = min(paths, reporting_paths)
    previous_macro_bytes = periods * paths * (2 * dimensions * 8 + 1)
    streamed_macro_bytes = periods * retained_paths * (dimensions * 8 + 1)
    detailed_seconds = execution_results["python_macro_detailed_native"][
        "median_seconds_excluding_warmup"
    ]
    compact_seconds = execution_results["streamed_compact_native"][
        "median_seconds_excluding_warmup"
    ]
    return {
        "configuration": {
            "periods": periods,
            "paths": paths,
            "reporting_paths": retained_paths,
            "workers": workers,
            "repeats": repeats,
        },
        "optimized_python_macro": {
            "runs_seconds": python_macro_timings,
            "median_seconds_excluding_warmup": float(
                statistics.median(python_macro_timings[1:])
            ),
        },
        "end_to_end": execution_results,
        "streamed_speedup": float(detailed_seconds / compact_seconds),
        "distribution_check": {
            "max_state_frequency_difference": float(
                np.max(np.abs(detailed_frequency - compact_frequency))
            ),
            "macro_mean_absolute_differences": np.abs(
                detailed_mean - compact_mean
            ).tolist(),
            "macro_std_absolute_differences": np.abs(detailed_std - compact_std).tolist(),
        },
        "macro_regime_storage_estimate": {
            "previous_bytes": previous_macro_bytes,
            "streamed_bytes": streamed_macro_bytes,
            "reduction_fraction": 1.0 - streamed_macro_bytes / previous_macro_bytes,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--periods", type=int, default=120)
    parser.add_argument("--paths", type=int, default=25_000)
    parser.add_argument("--reporting-paths", type=int, default=10_000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    print(
        json.dumps(
            benchmark(
                args.periods,
                args.paths,
                args.reporting_paths,
                args.workers,
                args.repeats,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
