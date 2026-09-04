#!/usr/bin/env python3
"""Benchmark the HSMM state-estimation improvements on known synthetic truth."""

from __future__ import annotations

import argparse
import json
import statistics
import time

import numpy as np
import pandas as pd

from mc_quadrants.hsmm import fit_quadrant_hsmm
from mc_quadrants.regimes import REGIME_ORDER, classify_quadrants
from mc_quadrants.simulation import (
    _macro_quadrant_probabilities,
    _prepare_macro_emissions,
)


def _synthetic_macro(months: int) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20_260_903)
    exit_matrix = np.array(
        [
            [0.0, 0.52, 0.08, 0.40],
            [0.55, 0.0, 0.40, 0.05],
            [0.08, 0.42, 0.0, 0.50],
            [0.40, 0.05, 0.55, 0.0],
        ]
    )
    truth: list[int] = []
    state = 0
    while len(truth) < months:
        duration = int(np.clip(rng.negative_binomial(8, 0.42) + 6, 6, 48))
        truth.extend([state] * duration)
        state = int(rng.choice(len(REGIME_ORDER), p=exit_matrix[state]))
    truth_values = np.asarray(truth[:months], dtype=int)
    means = np.array(
        [
            [1.10, 1.45],
            [0.95, 3.25],
            [-0.75, 3.15],
            [-0.90, 1.55],
        ]
    )
    covariances = np.array(
        [
            [[0.50, -0.20], [-0.20, 0.42]],
            [[0.46, 0.25], [0.25, 0.55]],
            [[0.60, -0.28], [-0.28, 0.50]],
            [[0.52, 0.22], [0.22, 0.48]],
        ]
    )
    observations = np.vstack(
        [rng.multivariate_normal(means[state], covariances[state]) for state in truth_values]
    )
    index = pd.date_range("1965-01-31", periods=months, freq="ME")
    macro = pd.DataFrame(observations, index=index, columns=["growth", "inflation"])
    return macro, truth_values, means, covariances


def _probability_scores(probabilities: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    rows = np.arange(len(truth))
    one_hot = np.eye(len(REGIME_ORDER))[truth]
    return {
        "accuracy": float(np.mean(np.argmax(probabilities, axis=1) == truth)),
        "brier": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "log_loss": float(
            -np.mean(np.log(np.clip(probabilities[rows, truth], 1e-15, 1.0)))
        ),
    }


def benchmark(months: int, heldout_samples: int, repeats: int) -> dict[str, object]:
    if months < 120 or heldout_samples < 1 or repeats < 1:
        raise ValueError("months must be at least 120 and samples/repeats must be positive")
    macro, truth, true_means, true_covariances = _synthetic_macro(months)
    initial_labels = classify_quadrants(
        macro,
        growth_threshold=0.22,
        inflation_threshold=2.62,
    )
    one_hot = np.eye(len(REGIME_ORDER))[truth]
    fits = {}
    results: dict[str, object] = {}
    for name, update_emissions in (("fixed_emissions", False), ("updated_emissions", True)):
        timings = []
        for _ in range(repeats):
            started = time.perf_counter()
            fits[name] = fit_quadrant_hsmm(
                macro,
                initial_labels,
                min_duration=5,
                max_duration=60,
                max_iterations=30,
                update_emissions=update_emissions,
            )
            timings.append(time.perf_counter() - started)
        fit = fits[name]
        probabilities = fit.filtered_probabilities.to_numpy(dtype=float)
        decoded = fit.viterbi_path.to_numpy(dtype=str)
        labels = np.asarray(REGIME_ORDER, dtype=object)[truth].astype(str)
        results[name] = {
            "viterbi_accuracy": float(np.mean(decoded == labels)),
            "filtered_brier": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
            "filtered_log_loss": float(
                -np.mean(
                    np.log(np.clip(probabilities[np.arange(months), truth], 1e-15, 1.0))
                )
            ),
            "median_fit_seconds": float(statistics.median(timings)),
        }

    rng = np.random.default_rng(777)
    heldout_truth = rng.integers(0, len(REGIME_ORDER), size=heldout_samples)
    heldout = np.vstack(
        [
            rng.multivariate_normal(true_means[state], true_covariances[state])
            for state in heldout_truth
        ]
    )
    dynamics = {
        "thresholds": np.array([0.22, 2.62]),
        "probability_scales": macro.std(ddof=1).to_numpy() * 0.35,
    }
    prepared = _prepare_macro_emissions(
        REGIME_ORDER,
        fits["updated_emissions"].emission_means,
        fits["updated_emissions"].emission_covariances,
    )
    membership_results = {}
    for name, prepared_emissions in (("threshold_logistic", None), ("joint_emission", prepared)):
        timings = []
        for _ in range(repeats):
            started = time.perf_counter()
            probabilities = _macro_quadrant_probabilities(
                heldout,
                dynamics,
                prepared_emissions=prepared_emissions,
            )
            timings.append(time.perf_counter() - started)
        membership_results[name] = {
            **_probability_scores(probabilities, heldout_truth),
            "median_seconds": float(statistics.median(timings)),
        }
    results["macro_membership"] = membership_results

    improved = fits["updated_emissions"]
    latest_truth = int(truth[-1])
    latest_posterior = np.array(
        [sum(improved.latest_state_age_probabilities[state]) for state in REGIME_ORDER]
    )
    stationary = np.linalg.matrix_power(improved.transition_matrix.to_numpy(), 1_000)[0]
    results["simulation_start"] = {
        "truth": REGIME_ORDER[latest_truth],
        "stationary_probability_on_truth": float(stationary[latest_truth]),
        "latest_filtered_probability_on_truth": float(latest_posterior[latest_truth]),
    }
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--months", type=int, default=720)
    parser.add_argument("--heldout-samples", type=int, default=100_000)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    print(
        json.dumps(
            benchmark(args.months, args.heldout_samples, args.repeats),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
