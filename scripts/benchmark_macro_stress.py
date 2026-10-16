"""Chronological production-path comparison of persistent macro-news candidates."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mc_quadrants.calibration import calibrate_quadrant_model  # noqa: E402
from mc_quadrants.path_validation import scalar_scores  # noqa: E402
from scripts.benchmark_reality_fixes import load_data  # noqa: E402
from scripts.investigate_allocation_realism import (  # noqa: E402
    FIT,
    allocation_scores,
    dependence_scores,
    simulate,
    summary_tables,
)

OUTPUT = ROOT / "research/macro_stress_2026-10-16"
VARIANTS = {
    "current": {"macro_stress": "none"},
    "random_walk": {"yield_model": "random_walk", "macro_stress": "none"},
    "persistent": {"macro_stress": "persistent"},
    "volatility": {"macro_stress": "volatility"},
    "combined": {"macro_stress": "combined"},
}
PATHS = 4096
SEED = 20261016


def paired_intervals(scores, dependence, repetitions=4000):
    """Paired stationary-block intervals; preserve overlap and candidate pairing.

    Windows are already inspected research data, not a pristine final holdout.
    Long horizons with fewer than three effective blocks get no interval.
    """
    from mc_quadrants.uncertainty import stationary_bootstrap_indices

    output = []
    groups = [
        (scores, "cagr_crps", True),
        (scores, "max_drawdown_crps", True),
        (dependence, "joint_terminal_loss_brier", False),
    ]
    rng = np.random.default_rng(SEED)
    for frame, metric, allocations in groups:
        for (horizon, weight), part in frame.assign(stocks=frame.stocks if allocations else -1).groupby(
            ["horizon", "stocks"]
        ):
            pivot = part.pivot(index="origin", columns="model", values=metric).sort_index()
            block = max(2, math.ceil(horizon / 6))
            if len(pivot) < 3 * block:
                continue
            for model in VARIANTS:
                if model == "current":
                    continue
                delta = (pivot[model] - pivot.current).to_numpy()
                draws = np.array(
                    [
                        delta[stationary_bootstrap_indices(len(delta), block, rng)].mean()
                        for _ in range(repetitions)
                    ]
                )
                lo, hi = np.quantile(draws, [0.025, 0.975])
                output.append(
                    dict(
                        model=model,
                        horizon=horizon,
                        stocks=weight,
                        metric=metric,
                        difference=delta.mean(),
                        lower=lo,
                        upper=hi,
                        block=block,
                        origins=len(delta),
                    )
                )
    return pd.DataFrame(output)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    macro, returns, sources = load_data()
    macro["bond_yield"] = sources["DGS7"].reindex(macro.index)
    splits = sorted(
        set(range(60, len(returns) - 11, 6))
        | {returns.index.get_loc(pd.Timestamp(date)) for date in ("2020-01-31", "2022-01-31")}
    )
    plan = dict(
        variants=VARIANTS,
        paths=PATHS,
        seed=SEED,
        settings=FIT,
        horizons=[12, 60, 120],
        promotion="No automatic promotion: require broad proper-score improvement, joint-loss accuracy, and stable ensemble tails; inspect block-aware uncertainty.",
        limitations="Latest-revised macro; post-2006 US ETFs; previously inspected and overlapping windows.",
        sources="research/reality_fixes_2026-10-13/manifest.json",
        inputs={
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                ROOT / "research/model_quality_2025/data.pkl",
                ROOT / "research/reality_fixes_2026-10-13/DGS7.csv",
                ROOT / "research/reality_fixes_2026-10-13/CPIAUCSL.csv",
            ]
        },
    )
    (OUTPUT / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    scores, dependence, macro_scores, fits = [], [], [], []
    for number, split in enumerate(splits):
        origin = returns.index[split - 1]
        horizons = [h for h in (12, 60, 120) if split + h <= len(returns)]
        for label, options in VARIANTS.items():
            model = calibrate_quadrant_model(
                returns.iloc[:split],
                macro.loc[:origin].copy(),
                **FIT,
                structural_returns=True,
                economic_returns=options,
            )
            forecast = simulate(model, max(horizons), PATHS, SEED + number)
            stress = model.metadata["macro_dynamics"].get("stress_dynamics")
            if stress:
                fits.append(dict(origin=str(origin.date()), model=label, **stress))
            for h in horizons:
                actual = returns.iloc[split : split + h].to_numpy()
                scores.extend(allocation_scores(actual, forecast.returns[:h], origin, h, label))
                dependence.append(dependence_scores(actual, forecast.returns[:h], origin, h, label))
            for h in (1, 3, 12):
                date = origin + pd.offsets.MonthEnd(h)
                for column in ("bond_yield", "inflation"):
                    if date not in macro.index or not np.isfinite(macro.loc[date, column]):
                        continue
                    values = forecast.macro_paths[h - 1, :, forecast.macro_columns.index(column)]
                    macro_scores.append(
                        dict(
                            origin=str(origin.date()),
                            model=label,
                            horizon=h,
                            column=column,
                            **scalar_scores("forecast", macro.loc[date, column], values),
                        )
                    )
        print(f"Completed origin {number + 1}/{len(splits)}: {origin.date()}", flush=True)
        pd.DataFrame(scores).to_csv(OUTPUT / "scores.csv", index=False)
        pd.DataFrame(dependence).to_csv(OUTPUT / "dependence.csv", index=False)
    frame, dep = pd.DataFrame(scores), pd.DataFrame(dependence)
    summary, dep_summary = summary_tables(frame, dep)
    summary.to_csv(OUTPUT / "summary.csv", index=False)
    dep_summary.to_csv(OUTPUT / "dependence_summary.csv", index=False)
    pd.DataFrame(macro_scores).to_csv(OUTPUT / "macro_scores.csv", index=False)
    pd.DataFrame(macro_scores).groupby(["model", "horizon", "column"]).mean(numeric_only=True).to_csv(
        OUTPUT / "macro_summary.csv"
    )
    paired_intervals(frame, dep).to_csv(OUTPUT / "paired_intervals.csv", index=False)
    for label, period in (("pre2020", frame.origin < "2020"), ("2020onward", frame.origin >= "2020")):
        frame.loc[period].groupby(["model", "horizon", "stocks"]).mean(numeric_only=True).to_csv(
            OUTPUT / f"summary_{label}.csv"
        )
    (OUTPUT / "fitted_states.json").write_text(json.dumps(fits, indent=2) + "\n")
    plan["code"] = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [
            Path(__file__),
            ROOT / "src/mc_quadrants/macro_stress.py",
            ROOT / "src/mc_quadrants/calibration.py",
            ROOT / "src/mc_quadrants/simulation.py",
        ]
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(plan, indent=2) + "\n")


if __name__ == "__main__":
    main()
