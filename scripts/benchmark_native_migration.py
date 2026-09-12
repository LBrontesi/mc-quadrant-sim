#!/usr/bin/env python3
"""Compare newly migrated ledgers against the reference on identical returns."""

from __future__ import annotations

import argparse
import gc
import json
import os
import time

import numpy as np

from mc_quadrants.native import native_available
from mc_quadrants.simulation import simulate_portfolio_paths
from mc_quadrants.types import SimulationResult


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=int, default=25_000)
    parser.add_argument("--periods", type=int, default=360)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if min(args.paths, args.periods, args.workers) <= 0:
        parser.error("paths, periods and workers must be positive")
    if not native_available():
        parser.error("build the native library first")
    rng = np.random.default_rng(47)
    result = SimulationResult(
        returns=rng.normal(.003, .035, (args.periods, args.paths, 4)),
        regimes=np.zeros((args.periods, args.paths), dtype=np.uint8),
        assets=["A", "B", "C", "D"], states=["state"], frequency="M",
    )
    previous_flag = os.environ.get("MC_DISABLE_NATIVE_SIM")
    try:
        for label, leverage in (("holdings_with_spending", 1), ("leveraged_with_spending", 2)):
            kwargs = dict(initial_value=1000, rebalance_frequency=3, transaction_cost_bps=10,
                          contribution=2, withdrawal=3, leverage_multiple=leverage,
                          financing_rate=.03 if leverage > 1 else 0, native_threads=args.workers)
            os.environ["MC_DISABLE_NATIVE_SIM"] = "1"
            started = time.perf_counter()
            reference = simulate_portfolio_paths(result, dict.fromkeys(result.assets, .25), **kwargs)
            reference_seconds = time.perf_counter() - started
            # Keep only the comparison matrix, not the reference's reporting arrays.
            reference_values = reference.to_numpy()
            del reference
            gc.collect()
            os.environ.pop("MC_DISABLE_NATIVE_SIM", None)
            started = time.perf_counter()
            native = simulate_portfolio_paths(result, dict.fromkeys(result.assets, .25), **kwargs)
            native_seconds = time.perf_counter() - started
            np.testing.assert_allclose(native, reference_values, rtol=2e-11, atol=1e-9)
            print(json.dumps({"case": label, "paths": args.paths, "periods": args.periods,
                              "reference_seconds": reference_seconds, "native_seconds": native_seconds,
                              "speedup": reference_seconds / native_seconds,
                              "max_absolute_wealth_difference": float(np.max(np.abs(native.to_numpy() - reference_values)))}), flush=True)
            del native, reference_values
            gc.collect()
    finally:
        if previous_flag is None:
            os.environ.pop("MC_DISABLE_NATIVE_SIM", None)
        else:
            os.environ["MC_DISABLE_NATIVE_SIM"] = previous_flag


if __name__ == "__main__":
    main()
