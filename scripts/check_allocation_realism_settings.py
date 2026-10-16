#!/usr/bin/env python3
"""Check allocation findings with structural links and UI-style parameter draws."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mc_quadrants.uncertainty import anchor_parameter_models, bootstrap_quadrant_models  # noqa: E402
from scripts.investigate_allocation_realism import (  # noqa: E402
    FIT,
    allocation_scores,
    calibrate_quadrant_model,
    dependence_scores,
    simulate,
    simulate_returns,
    summarize_paths,
)


def main():
    output = ROOT / "research/allocation_realism_2026-10-11"
    output.mkdir(parents=True, exist_ok=True)
    with (ROOT / "research/model_quality_2025/data.pkl").open("rb") as handle:
        macro, returns = pickle.load(handle)
    returns = returns.loc["2006-03-01":, ["SPY", "IEF"]]
    settings = {**FIT, "structural_returns": True}
    rows, scores, dependence, ordering = [], [], [], []
    seeds = (20261012, 20261013)
    paths = 20000
    for cutoff, horizon in [("2011-02-28", 120), ("2021-12-31", 12), ("2025-12-31", 120)]:
        training, training_macro = returns.loc[:cutoff], macro.loc[:cutoff]
        reference = calibrate_quadrant_model(training, training_macro, **settings)
        actual = returns.loc[returns.index > cutoff].iloc[:horizon].to_numpy()
        for seed in seeds:
            models = anchor_parameter_models(
                bootstrap_quadrant_models(
                    training,
                    training_macro,
                    draws=8,
                    block_size=12,
                    random_seed=seed + 10000,
                    **settings,
                ),
                reference,
            )
            chunks = []
            for draw, model in enumerate(models):
                fitted = model.metadata["dependence_fit"]
                coefficients = {
                    k: float(fitted[k])
                    for k in ("garch_alpha", "garch_beta", "dcc_alpha", "dcc_beta", "dcc_asymmetry")
                }
                chunks.append(
                    simulate_returns(
                        model,
                        periods=horizon,
                        paths=paths // 8,
                        random_seed=seed + draw * 100003,
                        duration_model="semi_markov",
                        garch=True,
                        dynamic_correlation=True,
                        joint_macro=True,
                        macro_parameter_uncertainty=False,
                        **coefficients,
                    ).returns
                )
            variants = [
                ("structural_fixed", simulate(reference, horizon, paths, seed).returns),
                ("structural_bootstrap8", np.concatenate(chunks, axis=1)),
            ]
            for label, forecast in variants:
                label = f"{label}_seed_{seed}"
                summary, order = summarize_paths(forecast, label, horizon)
                rows.extend(dict(origin=cutoff, **row) for row in summary)
                ordering.append(dict(origin=cutoff, **order))
                if len(actual) == horizon:
                    scores.extend(allocation_scores(actual, forecast, pd.Timestamp(cutoff), horizon, label))
                    dependence.append(
                        dependence_scores(actual, forecast, pd.Timestamp(cutoff), horizon, label)
                    )
            print(f"Settings check complete: {cutoff}, seed {seed}", flush=True)
    for name, data in [
        ("settings_sensitivities", rows),
        ("settings_scores", scores),
        ("settings_dependence", dependence),
        ("settings_ordering", ordering),
    ]:
        pd.DataFrame(data).to_csv(output / f"{name}.csv", index=False)
    (output / "settings_manifest.json").write_text(
        json.dumps(
            dict(
                settings=settings,
                seeds=seeds,
                paths=paths,
                bootstrap_draws=8,
                bootstrap_block_months=12,
                bootstrap_seed_offset=10000,
                simulation_seed_increment=100003,
                macro_uncertainty="per-path for fixed; outer bootstrap for ensemble",
                scope="UI structural/parameter settings only; SPY/IEF, no costs, monthly rebalance. Same data as manifest.json.",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
