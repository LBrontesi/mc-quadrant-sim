#!/usr/bin/env python3
"""Chronological comparisons for economic anchors and persistent yield stress."""

from __future__ import annotations

import hashlib
import json
import pickle
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mc_quadrants.calibration import calibrate_quadrant_model  # noqa: E402
from scripts.investigate_allocation_realism import (  # noqa: E402
    FIT,
    allocation_scores,
    dependence_scores,
    simulate,
    summarize_paths,
    summary_tables,
)


def main():
    output = ROOT / "research/economic_returns_2026-10-12"
    data = ROOT / "research/model_quality_2025/data.pkl"
    with data.open("rb") as handle:
        macro, returns = pickle.load(handle)
    returns = returns.loc["2006-03-01":, ["SPY", "IEF"]]
    yields = pd.read_csv(output / "GS7.csv", parse_dates=["Date"]).set_index("Date").GS7.astype(float)
    yields.index = yields.index + pd.offsets.MonthEnd(0)
    # Same-month yield observations are used for physical return decomposition;
    # all forecast origins are truncated, and no future yield enters a fit.
    macro["bond_yield"] = yields.reindex(macro.index)
    macro = macro.loc[: returns.index[-1]]
    settings = {**FIT, "structural_returns": True}
    variants = [
        ("baseline", None),
        ("anchors", {"rate_stress": False}),
        ("rate_stress", {"return_anchors": False}),
        ("both", {}),
    ]
    rows, dep, snapshots = [], [], []
    splits = sorted(
        set(range(60, len(returns) - 11, 6))
        | {
            returns.index.get_loc(pd.Timestamp("2020-01-31")),
            returns.index.get_loc(pd.Timestamp("2022-01-31")),
        }
    )
    for number, split in enumerate(splits):
        origin = returns.index[split - 1]
        horizons = [h for h in (12, 36, 60, 120) if split + h <= len(returns)]
        for label, options in variants:
            model = calibrate_quadrant_model(
                returns.iloc[:split], macro.loc[:origin], **settings, economic_returns=options
            )
            forecast = simulate(model, max(horizons), 4096, 20261012 + number)
            for h in horizons:
                actual = returns.iloc[split : split + h].to_numpy()
                rows.extend(allocation_scores(actual, forecast.returns[:h], origin, h, label))
                dep.append(dependence_scores(actual, forecast.returns[:h], origin, h, label))
        print(f"Origin {number + 1}/{len(splits)}: {origin.date()}", flush=True)
        pd.DataFrame(rows).to_csv(output / "scores.csv", index=False)
    summary, dependence = summary_tables(pd.DataFrame(rows), pd.DataFrame(dep))
    summary.to_csv(output / "summary.csv", index=False)
    pd.DataFrame(dep).to_csv(output / "dependence.csv", index=False)
    dependence.to_csv(output / "dependence_summary.csv", index=False)
    for cutoff, horizon in [("2011-02-28", 120), ("2021-12-31", 12), ("2025-12-31", 120)]:
        for label, options in variants:
            model = calibrate_quadrant_model(
                returns.loc[:cutoff], macro.loc[:cutoff], **settings, economic_returns=options
            )
            for seed in (20261012, 20261013):
                result = simulate(model, horizon, 20000, seed)
                values, _ = summarize_paths(result.returns, label, horizon)
                snapshots.extend(dict(origin=cutoff, seed=seed, **value) for value in values)
    pd.DataFrame(snapshots).to_csv(output / "snapshots.csv", index=False)
    (output / "manifest.json").write_text(
        json.dumps(
            dict(
                data_sha256=hashlib.sha256(data.read_bytes()).hexdigest(),
                yields_sha256=hashlib.sha256((output / "GS7.csv").read_bytes()).hexdigest(),
                yield_source="https://fred.stlouisfed.org/data/GS7",
                yield_units="annual percent; monthly business-day averages",
                settings=settings,
                paths=4096,
                snapshot_paths=20000,
                variants=variants,
                data_vintage="latest revised; not historical vintages",
                parameter_bootstrap=False,
                yield_calendar="same-month market observations; historical physical factor decomposition is contemporaneous",
                note="Default economic assumptions fixed before this comparison; no optimization on 2022. Windows overlap.",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
