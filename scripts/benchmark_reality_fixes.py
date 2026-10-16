"""Chronological ablations for endpoint yields and persistent yield forecasts."""

from __future__ import annotations

import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mc_quadrants.calibration import calibrate_quadrant_model  # noqa: E402
from mc_quadrants.path_validation import scalar_scores  # noqa: E402
from scripts.investigate_allocation_realism import (  # noqa: E402
    FIT,
    allocation_scores,
    dependence_scores,
    simulate,
    summary_tables,
)

OUTPUT = ROOT / "research/reality_fixes_2026-10-13"
VARIANTS = [
    ("baseline", None, "DGS7"),
    ("previous", {"yield_model": "legacy", "equity_rate_mode": "inflation_gate"}, "GS7"),
    ("endpoint_legacy", {"yield_model": "legacy", "equity_rate_mode": "inflation_gate"}, "DGS7"),
    ("endpoint_rw", {"yield_model": "random_walk", "equity_rate_mode": "inflation_gate"}, "DGS7"),
    ("endpoint_adaptive", {"yield_model": "adaptive", "equity_rate_mode": "inflation_gate"}, "DGS7"),
    ("revised", {"yield_model": "adaptive", "equity_rate_mode": "surprise"}, "DGS7"),
]


def load_data():
    with (ROOT / "research/model_quality_2025/data.pkl").open("rb") as handle:
        macro, returns = pickle.load(handle)
    returns = returns.loc["2006-03-01":, ["SPY", "IEF"]]
    sources = {}
    for code in ("GS7", "DGS7", "DGS2", "DGS20"):
        path = (
            ROOT / "research/economic_returns_2026-10-12/GS7.csv" if code == "GS7" else OUTPUT / f"{code}.csv"
        )
        raw = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")[code].astype(float)
        sources[code] = raw.resample("ME").last()
    cpi = pd.read_csv(OUTPUT / "CPIAUCSL.csv", parse_dates=["Date"]).set_index("Date").CPIAUCSL
    cpi = cpi.resample("ME").last()
    # Frozen macro is already one-month availability aligned. CPI changes use
    # that same information calendar; current revised levels are diagnostic.
    changes = np.log(cpi).diff().shift(1)
    macro["cpi_monthly_log"] = changes.reindex(macro.index)
    # Preserve the frozen macro series. Inconsistent revised CPI initialization
    # is rejected rather than silently replacing the original predictors.
    return macro.loc[: returns.index[-1]], returns, sources


def main():
    macro, returns, sources = load_data()
    rows, dep, yields = [], [], []
    splits = sorted(
        set(range(60, len(returns) - 11, 6))
        | {
            returns.index.get_loc(pd.Timestamp("2022-01-31")),
            returns.index.get_loc(pd.Timestamp("2020-01-31")),
        }
    )
    for number, split in enumerate(splits):
        origin = returns.index[split - 1]
        horizons = [h for h in (12, 60, 120) if split + h <= len(returns)]
        for label, options, source in VARIANTS:
            training_macro = macro.loc[:origin].copy()
            training_macro["bond_yield"] = sources[source].reindex(training_macro.index)
            model = calibrate_quadrant_model(
                returns.iloc[:split], training_macro, **FIT, structural_returns=True, economic_returns=options
            )
            forecast = simulate(model, max(horizons), 4096, 20261012 + number)
            for h in horizons:
                actual = returns.iloc[split : split + h].to_numpy()
                rows.extend(allocation_scores(actual, forecast.returns[:h], origin, h, label))
                dep.append(dependence_scores(actual, forecast.returns[:h], origin, h, label))
            if options is not None:
                index = forecast.macro_columns.index("bond_yield")
                for h in (1, 3, 12):
                    actual = sources[source].loc[origin + pd.offsets.MonthEnd(h)]
                    values = forecast.macro_paths[h - 1, :, index]
                    scores = scalar_scores("yield", actual, values)
                    scores["yield_bias"] = float(np.mean(values) - actual)
                    for coverage in (0.5, 0.8, 0.95):
                        low, high = np.quantile(values, [(1 - coverage) / 2, (1 + coverage) / 2])
                        scores[f"yield_coverage_{int(coverage * 100)}"] = float(low <= actual <= high)
                    yields.append(dict(origin=origin, horizon=h, model=label, **scores))
        print(f"Completed origin {number + 1}/{len(splits)}: {origin.date()}", flush=True)
        pd.DataFrame(rows).to_csv(OUTPUT / "scores.csv", index=False)
        pd.DataFrame(yields).to_csv(OUTPUT / "yield_scores.csv", index=False)
    summary, dependence = summary_tables(pd.DataFrame(rows), pd.DataFrame(dep))
    summary.to_csv(OUTPUT / "summary.csv", index=False)
    pd.DataFrame(dep).to_csv(OUTPUT / "dependence.csv", index=False)
    dependence.to_csv(OUTPUT / "dependence_summary.csv", index=False)
    pd.DataFrame(yields).groupby(["model", "horizon"]).mean(numeric_only=True).to_csv(
        OUTPUT / "yield_summary.csv"
    )
    inputs = [
        ROOT / "research/model_quality_2025/data.pkl",
        ROOT / "research/economic_returns_2026-10-12/GS7.csv",
        *OUTPUT.glob("*.csv"),
    ]
    (OUTPUT / "manifest.json").write_text(
        json.dumps(
            dict(
                variants=VARIANTS,
                paths=4096,
                settings=FIT,
                data="Frozen post-2006 US ETFs; latest-revised macro, not historical vintages; overlapping windows",
                inputs={
                    str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in inputs
                    if p.stem in {"data", "GS7", "DGS7", "DGS2", "DGS20", "CPIAUCSL"}
                },
                sources={
                    code: f"https://fred.stlouisfed.org/data/{code}"
                    for code in ("DGS7", "DGS2", "DGS20", "CPIAUCSL")
                },
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
