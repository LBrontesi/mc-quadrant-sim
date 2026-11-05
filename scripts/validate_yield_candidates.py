"""Frozen-candidate monthly validation; research only, with no production edits.

Candidates are fixed before scoring. Prefix-only refits prevent training on
future outcomes; repeated use of this dataset prevents independent-holdout claims.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from mc_quadrants import yield_curve  # noqa: E402
from mc_quadrants.calibration import calibrate_quadrant_model  # noqa: E402
from mc_quadrants.path_validation import scalar_scores  # noqa: E402
from scripts.benchmark_nelson_siegel import OUTPUT, read_fred_table  # noqa: E402
from scripts.benchmark_reality_fixes import load_data  # noqa: E402
from scripts.diagnose_nelson_siegel import fit_candidate  # noqa: E402
from scripts.investigate_allocation_realism import (  # noqa: E402
    FIT,
    allocation_scores,
    dependence_scores,
    simulate,
)

CANDIDATES = {
    "adaptive": ("adaptive", "none"),
    "original": ("original", "none"),
    "no_inflation": ("no_inflation", "none"),
    "factor_rw": ("factor_rw", "none"),
    "adaptive_volatility": ("adaptive", "volatility"),
    "factor_rw_volatility": ("factor_rw", "volatility"),
}


def comparisons(yields, portfolios, dependence):
    """Average seed replicates before paired circular calendar-block inference."""
    tables = [
        ("yield", yields[yields.horizon == 12], "yield_crps"),
        ("bond_return", portfolios[portfolios.stocks == 0], "cagr_crps"),
        ("bond_drawdown", portfolios[portfolios.stocks == 0], "max_drawdown_crps"),
        ("balanced_return", portfolios[portfolios.stocks == 0.6], "cagr_crps"),
        ("joint_loss", dependence, "joint_terminal_loss_brier"),
    ]
    rows = []
    rng = np.random.default_rng(20261104)
    for metric, frame, column in tables:
        paired = frame.groupby(["origin", "model"])[column].mean().unstack().sort_index()
        if paired.isna().any().any():
            raise ValueError("Incomplete paired candidate scores")
        n = len(paired)
        for block in (6, 12, 24):
            starts = rng.integers(n, size=(20000, int(np.ceil(n / block))))
            indexes = ((starts[..., None] + np.arange(block)) % n).reshape(20000, -1)[:, :n]
            for candidate in CANDIDATES:
                if candidate == "adaptive":
                    continue
                difference = (paired[candidate] - paired.adaptive).to_numpy()
                boot = difference[indexes].mean(axis=1)
                low, high = np.quantile(boot, [0.025, 0.975])
                # Five comparisons within each metric; no claim of correcting
                # for earlier research rounds or across all reported metrics.
                adjusted_low, adjusted_high = np.quantile(boot, [0.005, 0.995])
                rows.append(dict(metric=metric, model=candidate, block=block, origins=n,
                                 difference=difference.mean(), ci_low=low, ci_high=high,
                                 adjusted_low=adjusted_low, adjusted_high=adjusted_high))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=int, default=8192)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20261104, 20261105])
    parser.add_argument("--output", type=Path, default=ROOT / "research/yield_validation_2026-11-04")
    args = parser.parse_args()
    dest = args.output
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "manifest.json").exists():
        raise ValueError("Use a fresh output directory to preserve previous results")
    macro, returns, sources = load_data()
    for code in ("DGS5", "DGS10"):
        sources[code] = read_fred_table(OUTPUT / f"{code}.html", code)
    for column, code in zip(yield_curve.CURVE_TENORS, ("DGS2", "DGS5", "DGS7", "DGS10", "DGS20")):
        macro[column] = sources[code].reindex(macro.index)
    prior = set(pd.read_csv(OUTPUT / "yield_scores.csv").origin)
    splits = list(range(60, len(returns) - 11))
    files = [Path(__file__), ROOT / "scripts/diagnose_nelson_siegel.py",
             ROOT / "scripts/benchmark_nelson_siegel.py", ROOT / "scripts/benchmark_reality_fixes.py",
             ROOT / "scripts/investigate_allocation_realism.py",
             ROOT / "research/model_quality_2025/data.pkl",
             *[ROOT / "research/reality_fixes_2026-10-13" / f"{c}.csv"
               for c in ("DGS2", "DGS7", "DGS20", "CPIAUCSL")],
             OUTPUT / "DGS5.html", OUTPUT / "DGS10.html",
             *sorted((ROOT / "src/mc_quadrants").glob("*.py")),
             *sorted((ROOT / "src/mc_quadrants").glob("*.cpp"))]
    manifest = dict(paths=args.paths, seeds=args.seeds, candidates=CANDIDATES, settings=FIT,
                    origins=[str(returns.index[s - 1].date()) for s in splits],
                    previous_origins=sorted(prior),
                    extension_origin="2025-12-31", extension_months=9,
                    limits="Revised US data, overlapping outcomes, repeatedly inspected history; not an untouched holdout. Extension is one correlated yield trajectory, without realized portfolio returns. October 2026 excluded because frozen daily inputs stop mid-month. No parameter banks.",
                    inference="Seed-averaged paired circular 6/12/24-month blocks, 20000 replicates; Bonferroni across five candidates per metric only.",
                    hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    rows, yields, dependence, extension = [], [], [], []
    started = time.perf_counter()
    for number, split in enumerate(splits + [len(returns)]):
        origin = returns.index[split - 1]
        is_extension = split == len(returns)
        months = 9 if is_extension else 12
        for label, (variant, stress) in CANDIDATES.items():
            def fitter(*a, **kw):
                return fit_candidate(variant, *a, **kw)

            with patch.object(yield_curve, "fit_curve", fitter):
                model = calibrate_quadrant_model(
                    returns.iloc[:split], macro.loc[:origin], **FIT, structural_returns=True,
                    economic_returns={"yield_model": "adaptive" if variant == "adaptive" else "nelson_siegel",
                                      "macro_stress": stress},
                )
            for seed in args.seeds:
                forecast = simulate(model, months, args.paths, seed + split)
                if not np.isfinite(forecast.returns).all() or not np.isfinite(forecast.macro_paths).all():
                    raise ValueError(f"Nonfinite forecast: {origin} {label}")
                common = dict(origin=str(origin.date()), model=label, seed=seed,
                              previously_scored=str(origin.date()) in prior)
                index = forecast.macro_columns.index("bond_yield")
                for horizon in (range(1, 10) if is_extension else (1, 3, 12)):
                    draws = forecast.macro_paths[horizon - 1, :, index]
                    actual = sources["DGS7"].loc[origin + pd.offsets.MonthEnd(horizon)]
                    if not np.isfinite(actual):
                        raise ValueError("Missing realized yield")
                    row = dict(**common, horizon=horizon, **scalar_scores("yield", actual, draws),
                               squared_error=float((draws.mean() - actual) ** 2))
                    (extension if is_extension else yields).append(row)
                if not is_extension:
                    actual = returns.iloc[split:split + 12].to_numpy()
                    for row in allocation_scores(actual, forecast.returns, origin, 12, label):
                        rows.append(dict(row, seed=seed, previously_scored=common["previously_scored"]))
                    dependence.append(dict(dependence_scores(actual, forecast.returns, origin, 12, label),
                                           seed=seed, previously_scored=common["previously_scored"]))
        if number % 12 == 0 or is_extension:
            for name, data in (("portfolio_scores", rows), ("yield_scores", yields),
                               ("dependence_scores", dependence), ("extension_scores", extension)):
                if data:
                    pd.DataFrame(data).to_csv(dest / f"{name}.csv", index=False)
        print(f"{number + 1}/{len(splits) + 1} {origin.date()} ({time.perf_counter() - started:.1f}s)", flush=True)
    yf, pf, df = map(pd.DataFrame, (yields, rows, dependence))
    comparisons(yf, pf, df).to_csv(dest / "paired_comparisons.csv", index=False)
    for name, frame, keys in (("yield", yf, ["model", "horizon"]),
                              ("portfolio", pf, ["model", "stocks"]),
                              ("dependence", df, ["model"]),
                              ("extension", pd.DataFrame(extension), ["model", "horizon"])):
        frame.groupby(keys).mean(numeric_only=True).to_csv(dest / f"{name}_summary.csv")
    print(yf[yf.horizon == 12].groupby("model").yield_crps.mean().to_string(), flush=True)
    print(pf[pf.stocks == 0].groupby("model")[["cagr_crps", "max_drawdown_crps", "max_drawdown_coverage_90"]].mean().to_string(), flush=True)


if __name__ == "__main__":
    main()
