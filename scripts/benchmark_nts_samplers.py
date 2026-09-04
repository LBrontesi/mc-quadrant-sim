#!/usr/bin/env python3
"""Compare exact native NTS subordinator samplers over the calibration range."""

from __future__ import annotations

import argparse
import json
import statistics
import time

import numpy as np

from mc_quadrants.native import sample_mnts_subordinators_native

ALGORITHMS = ("qu", "devroye", "legacy_hybrid")


def _nanoseconds_per_draw(
    samples: int,
    tail_index: float,
    tempering: float,
    algorithm: str,
    repeats: int,
) -> float:
    timings = []
    for repeat in range(repeats):
        started = time.perf_counter_ns()
        sample_mnts_subordinators_native(
            samples,
            tail_index,
            tempering,
            random_seed=1_000 + repeat,
            algorithm=algorithm,
        )
        timings.append((time.perf_counter_ns() - started) / samples)
    return float(statistics.median(timings))


def benchmark(samples: int, repeats: int) -> dict[str, object]:
    if samples < 1 or repeats < 1:
        raise ValueError("samples and repeats must be positive")
    alphas = (0.55, 0.65, 0.75, 0.85, 0.95)
    temperings = (0.04, 0.10, 0.25, 0.50, 1.00, 2.00, 5.00, 10.00, 20.00)
    for algorithm in ALGORITHMS:
        sample_mnts_subordinators_native(
            2_000,
            1.5,
            0.5,
            random_seed=42,
            algorithm=algorithm,
        )

    points: list[dict[str, float]] = []
    for alpha in alphas:
        for tempering in temperings:
            timings = {
                algorithm: _nanoseconds_per_draw(
                    samples,
                    2.0 * alpha,
                    tempering,
                    algorithm,
                    repeats,
                )
                for algorithm in ALGORITHMS
            }
            points.append(
                {
                    "alpha": alpha,
                    "tempering": tempering,
                    "lambda_alpha": tempering / alpha,
                    "qu_ns_per_draw": timings["qu"],
                    "devroye_ns_per_draw": timings["devroye"],
                    "legacy_hybrid_ns_per_draw": timings["legacy_hybrid"],
                    "qu_speedup_vs_devroye": timings["devroye"] / timings["qu"],
                    "qu_speedup_vs_legacy_hybrid": timings["legacy_hybrid"] / timings["qu"],
                }
            )

    devroye_speedups = [point["qu_speedup_vs_devroye"] for point in points]
    legacy_speedups = [point["qu_speedup_vs_legacy_hybrid"] for point in points]
    return {
        "samples_per_point": samples,
        "repeats": repeats,
        "grid_points": len(points),
        "alpha_range": [min(alphas), max(alphas)],
        "tempering_range": [min(temperings), max(temperings)],
        "qu_speedup_vs_devroye": {
            "minimum": min(devroye_speedups),
            "median": float(np.median(devroye_speedups)),
            "maximum": max(devroye_speedups),
        },
        "qu_speedup_vs_legacy_hybrid": {
            "minimum": min(legacy_speedups),
            "median": float(np.median(legacy_speedups)),
            "maximum": max(legacy_speedups),
        },
        "qu_wins_every_grid_point_vs_devroye": all(value > 1.0 for value in devroye_speedups),
        "qu_wins_every_grid_point_vs_legacy_hybrid": all(value > 1.0 for value in legacy_speedups),
        "points": points,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=100_000)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    print(json.dumps(benchmark(args.samples, args.repeats), indent=2))


if __name__ == "__main__":
    main()
