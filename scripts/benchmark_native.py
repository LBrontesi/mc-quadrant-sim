#!/usr/bin/env python3
"""Reproducible performance check for the production native tax scenario."""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from mc_quadrants.matrix import nearest_correlation
from mc_quadrants.native import native_available
from mc_quadrants.simulation import simulate_portfolio_paths, simulate_returns
from mc_quadrants.taxes import (
    italian_native_result_frame,
    prepare_italian_native_configuration,
)
from mc_quadrants.types import MNTSParameters, RegimeMoments, ScenarioModel


def benchmark(
    periods: int,
    paths: int,
    workers: int,
    compare_python: bool = False,
    repeats: int = 3,
) -> dict[str, object]:
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    assets = ["Equity", "Bonds", "Gold", "Commodities"]
    states = [
        "high_growth_low_inflation",
        "high_growth_high_inflation",
        "low_growth_high_inflation",
        "low_growth_low_inflation",
    ]
    base_mean = np.array([0.006, 0.002, 0.003, 0.001])
    base_covariance = np.array(
        [
            [0.0016, 0.0002, 0.0001, 0.0],
            [0.0002, 0.0004, 0.0001, 0.0],
            [0.0001, 0.0001, 0.0009, 0.00005],
            [0.0, 0.0, 0.00005, 0.0002],
        ],
    )
    mean_offsets = np.array([0.002, 0.001, -0.004, -0.002])
    volatility_multipliers = np.array([0.80, 1.05, 1.50, 1.25])
    tail_indexes = np.array([1.70, 1.55, 1.30, 1.40])
    temperings = np.array([0.90, 0.60, 0.35, 0.45])
    skewness_values = np.array(
        [
            [-0.20, -0.05, 0.10, 0.05],
            [-0.35, -0.10, 0.15, 0.10],
            [-0.65, -0.20, 0.30, 0.25],
            [-0.50, -0.15, 0.25, 0.15],
        ]
    )
    moments: dict[str, RegimeMoments] = {}
    for index, state in enumerate(states):
        covariance_values = base_covariance * volatility_multipliers[index] ** 2
        volatility = np.sqrt(np.diag(covariance_values))
        correlation_values = covariance_values / (volatility[:, None] * volatility[None, :])
        alpha = tail_indexes[index] / 2.0
        variance_t = (1.0 - alpha) / temperings[index]
        skewness = skewness_values[index]
        gaussian_scale = np.sqrt(1.0 - skewness * skewness * variance_t)
        latent = (
            correlation_values - variance_t * np.outer(skewness, skewness)
        ) / np.outer(gaussian_scale, gaussian_scale)
        latent_frame = nearest_correlation(
            pd.DataFrame(latent, index=assets, columns=assets)
        )
        covariance = pd.DataFrame(covariance_values, index=assets, columns=assets)
        correlation = pd.DataFrame(correlation_values, index=assets, columns=assets)
        moments[state] = RegimeMoments(
            mean=pd.Series(base_mean + mean_offsets[index], index=assets),
            covariance=covariance,
            correlation=correlation,
            observations=120,
            mnts=MNTSParameters(
                tail_index=float(tail_indexes[index]),
                tempering=float(temperings[index]),
                skewness=pd.Series(skewness, index=assets),
                gaussian_correlation=latent_frame,
            ),
        )
    transition = np.full((4, 4), 0.03)
    np.fill_diagonal(transition, 0.91)
    model = ScenarioModel(
        states=states,
        transition_matrix=pd.DataFrame(transition, index=states, columns=states),
        moments=moments,
    )
    weight_values = np.full(len(assets), 0.25)
    weights = dict(zip(assets, weight_values, strict=True))

    def run_fused(wrapper_benchmark: bool):
        native_config = prepare_italian_native_configuration(
            periods=periods,
            assets=assets,
            target_weights=weight_values,
            initial_value=100.0,
            rebalance_frequency=3,
            transaction_cost_bps=5.0,
            contribution=0.0,
            contribution_allocation="target",
            withdrawal=0.0,
            withdrawal_start_period=1,
            asset_tax_categories=None,
            asset_tax_metadata=None,
            annual_wealth_tax=0.002,
            terminal_liquidation=True,
            tax_regime="italy_administered",
            wealth_tax_mode="auto",
            start_date=None,
            wrapper_benchmark=wrapper_benchmark,
        )
        native_config.update(
            {
                "expense_ratios": np.zeros(len(assets)),
                "return_kind": "log",
                "state_transaction_cost_multipliers": {},
            }
        )
        result = simulate_returns(
            model,
            periods=periods,
            paths=paths,
            random_seed=42,
            distribution="mnts",
            return_regime_codes=True,
            native_threads=workers,
            native_portfolio_config=native_config,
        )
        if result.native_portfolio is None:
            raise RuntimeError("The fused native portfolio backend was not used.")
        return result, italian_native_result_frame(
            result.native_portfolio,
            result.native_portfolio["frame_metadata"],
            fused=True,
        )

    baseline_runs: list[float] = []
    wrapper_runs: list[float] = []
    baseline_result = None
    baseline = None
    wrapper = None
    for _ in range(repeats):
        baseline_started = time.perf_counter()
        baseline_result, baseline = run_fused(False)
        baseline_runs.append(time.perf_counter() - baseline_started)
        wrapper_started = time.perf_counter()
        _, wrapper = run_fused(True)
        wrapper_runs.append(time.perf_counter() - wrapper_started)
    assert baseline_result is not None and baseline is not None and wrapper is not None
    baseline_seconds = float(np.median(baseline_runs))
    wrapper_seconds = float(np.median(wrapper_runs))
    wrapper_overhead = wrapper_seconds / baseline_seconds - 1.0
    advanced_paths = min(paths, 5_000)
    advanced_started = time.perf_counter()
    advanced_result = simulate_returns(
        model,
        periods=periods,
        paths=advanced_paths,
        random_seed=42,
        distribution="mnts",
        return_regime_codes=True,
        native_threads=workers,
    )
    advanced = simulate_portfolio_paths(
        advanced_result,
        weights,
        rebalance_frequency=3,
        transaction_cost_bps=5.0,
        tax_country="IT",
        tax_regime="italy_administered",
        italy_annual_wealth_tax=0.002,
        tax_terminal_liquidation=True,
        tax_wrapper_benchmark=True,
        decumulation={
            "enabled": True,
            "mode": "manual",
            "phases": [
                {
                    "start_month": min(37, periods),
                    "end_month": periods,
                    "frequency": "monthly",
                    "annual_real_amount": 4.0,
                }
            ],
            "one_time_expenses": [
                {"month": min(60, periods), "real_amount": 5.0}
            ],
            "policy": "guyton_klinger",
            "annual_inflation_fallback": 0.02,
        },
    )
    advanced_seconds = time.perf_counter() - advanced_started
    requested = np.asarray(advanced.attrs["withdrawal_requested"], dtype=float)
    funded = np.asarray(advanced.attrs["withdrawal_funded"], dtype=float)
    funded_ratio = np.divide(
        funded.sum(axis=0),
        requested.sum(axis=0),
        out=np.ones(advanced_paths, dtype=float),
        where=requested.sum(axis=0) > 0,
    )
    terminal_values = baseline.iloc[-1].to_numpy(dtype=float)
    terminal_quantiles = np.quantile(terminal_values, (0.05, 0.50, 0.95))
    report: dict[str, object] = {
        "native_available": native_available(),
        "native_backend_used": bool(baseline.attrs.get("native_backend", False)),
        "periods": periods,
        "paths": paths,
        "assets": len(assets),
        "quadrants": len(states),
        "workers": workers,
        "repeats": repeats,
        "fused_total_seconds": round(baseline_seconds, 4),
        "return_cube_shape": list(baseline_result.returns.shape),
        "wrapper_fused_seconds": round(wrapper_seconds, 4),
        "wrapper_overhead_percent": round(wrapper_overhead * 100.0, 2),
        "target_under_15_seconds": baseline_seconds <= 15.0,
        "wrapper_overhead_under_10_percent": wrapper_overhead < 0.10,
        "advanced_decumulation_seconds": round(advanced_seconds, 4),
        "advanced_decumulation_paths": advanced_paths,
        "advanced_native_backend_used": bool(advanced.attrs.get("native_backend", False)),
        "advanced_funded_spending_ratio": float(np.mean(funded_ratio)),
        "advanced_guardrail_event_rate": float(
            np.mean(np.asarray(advanced.attrs["guardrail_events"]) != 0)
        ),
        "terminal_p05": float(terminal_quantiles[0]),
        "terminal_median": float(terminal_quantiles[1]),
        "terminal_p95": float(terminal_quantiles[2]),
        "terminal_mean": float(np.mean(terminal_values)),
        "terminal_std": float(np.std(terminal_values)),
        "wrapper_terminal_median": float(np.median(wrapper.attrs["wrapper_terminal_values"])),
    }
    if compare_python:
        previous_disable = os.environ.get("MC_DISABLE_NATIVE_SIM")
        os.environ["MC_DISABLE_NATIVE_SIM"] = "1"
        try:
            reference_started = time.perf_counter()
            reference_returns = simulate_returns(
                model,
                periods=periods,
                paths=paths,
                random_seed=42,
                distribution="mnts",
                return_regime_codes=True,
            )
            simulate_portfolio_paths(
                reference_returns,
                weights,
                rebalance_frequency=3,
                transaction_cost_bps=5.0,
                tax_country="IT",
                tax_regime="italy_administered",
                tax_terminal_liquidation=True,
            )
            reference_seconds = time.perf_counter() - reference_started
        finally:
            if previous_disable is None:
                os.environ.pop("MC_DISABLE_NATIVE_SIM", None)
            else:
                os.environ["MC_DISABLE_NATIVE_SIM"] = previous_disable
        report["python_reference_total_seconds"] = round(reference_seconds, 4)
        report["native_total_speedup"] = round(reference_seconds / baseline_seconds, 2)
        report["native_total_reduction_percent"] = round(
            (1.0 - baseline_seconds / reference_seconds) * 100.0,
            2,
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--periods", type=int, default=360)
    parser.add_argument("--paths", type=int, default=100_000)
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--compare-python", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            benchmark(
                args.periods,
                args.paths,
                args.workers,
                args.compare_python,
                args.repeats,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
