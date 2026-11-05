"""Chronological production-path comparison of optional yield forecasting models."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from mc_quadrants.calibration import calibrate_quadrant_model  # noqa: E402
from mc_quadrants.path_validation import scalar_scores  # noqa: E402
from mc_quadrants.yield_curve import CURVE_TENORS  # noqa: E402
from scripts.benchmark_reality_fixes import load_data  # noqa: E402
from scripts.investigate_allocation_realism import FIT, allocation_scores, simulate  # noqa: E402

OUTPUT = ROOT / "research/nelson_siegel_2026-10-27"
MODES = ("random_walk", "adaptive", "nelson_siegel")


def read_fred_table(path, code):
    pairs = re.findall(
        r'<th[^>]*scope="row"[^>]*>\s*(\d{4}-\d{2}-\d{2})\s*</th>\s*<td[^>]*>\s*([^<]+)</td>',
        path.read_text(),
    )
    pairs += re.findall(r"#(\d{4}-\d{2}-\d{2})\|\s*([\d.]+)", path.read_text())
    if not pairs:
        raise ValueError(f"No observations in {path}")
    series = pd.Series(
        pd.to_numeric([v.strip() for _, v in pairs], errors="coerce"),
        index=pd.to_datetime([d for d, _ in pairs]),
        name=code,
    )
    series.rename_axis("Date").to_csv(path.with_suffix(".csv"))
    return series.resample("ME").last()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=int, default=4096)
    parser.add_argument("--stride", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    macro, returns, sources = load_data()
    sources["DGS5"] = read_fred_table(OUTPUT / "DGS5.html", "DGS5")
    sources["DGS10"] = read_fred_table(OUTPUT / "DGS10.html", "DGS10")
    for column, code in zip(CURVE_TENORS, ("DGS2", "DGS5", "DGS7", "DGS10", "DGS20")):
        macro[column] = sources[code].reindex(macro.index)
    splits = sorted(
        set(range(60, len(returns) - 11, args.stride))
        | {returns.index.get_loc(pd.Timestamp(date)) for date in ("2020-01-31", "2022-01-31")}
    )
    rows, yields, timings, curve_scores = [], [], [], []
    start = time.perf_counter()
    for number, split in enumerate(splits):
        origin = returns.index[split - 1]
        for mode in MODES:
            before = time.perf_counter()
            model = calibrate_quadrant_model(
                returns.iloc[:split],
                macro.loc[:origin],
                **FIT,
                structural_returns=True,
                economic_returns={"yield_model": mode},
            )
            fitted = time.perf_counter()
            forecast = simulate(model, 12, args.paths, args.seed + number)
            timings.append(
                dict(
                    origin=origin,
                    model=mode,
                    fit_seconds=fitted - before,
                    simulation_seconds=time.perf_counter() - fitted,
                )
            )
            rows.extend(
                allocation_scores(
                    returns.iloc[split : split + 12].to_numpy(), forecast.returns, origin, 12, mode
                )
            )
            for horizon in (1, 3, 12):
                date = origin + pd.offsets.MonthEnd(horizon)
                index = forecast.macro_columns.index("bond_yield")
                draws = forecast.macro_paths[horizon - 1, :, index]
                actual = sources["DGS7"].loc[date]
                yields.append(
                    dict(
                        origin=origin,
                        horizon=horizon,
                        model=mode,
                        **scalar_scores("yield", actual, draws),
                        yield_bias=float(draws.mean() - actual),
                        yield_squared_error=float((draws.mean() - actual) ** 2),
                    )
                )
                if mode == "nelson_siegel":
                    for column, code in zip(CURVE_TENORS, ("DGS2", "DGS5", "DGS7", "DGS10", "DGS20")):
                        draws = forecast.macro_paths[horizon - 1, :, forecast.macro_columns.index(column)]
                        curve_scores.append(
                            dict(
                                origin=origin,
                                horizon=horizon,
                                tenor=column,
                                **scalar_scores("yield", sources[code].loc[date], draws),
                            )
                        )
        pd.DataFrame(rows).to_csv(output / "portfolio_scores.csv", index=False)
        pd.DataFrame(yields).to_csv(output / "yield_scores.csv", index=False)
        print(
            f"{number + 1}/{len(splits)} origins: {origin.date()} ({time.perf_counter() - start:.1f}s)",
            flush=True,
        )
    portfolio = pd.DataFrame(rows)
    portfolio.groupby(["model", "stocks"]).mean(numeric_only=True).to_csv(output / "portfolio_summary.csv")
    yield_summary = pd.DataFrame(yields).groupby(["model", "horizon"]).mean(numeric_only=True)
    yield_summary["yield_rmse"] = np.sqrt(yield_summary["yield_squared_error"])
    yield_summary.to_csv(output / "yield_summary.csv")
    pd.DataFrame(timings).to_csv(output / "timings.csv", index=False)
    pd.DataFrame(curve_scores).to_csv(output / "curve_scores.csv", index=False)
    files = [
        ROOT / "research/model_quality_2025/data.pkl",
        *[
            ROOT / "research/reality_fixes_2026-10-13" / f"{code}.csv"
            for code in ("DGS2", "DGS7", "DGS20", "CPIAUCSL")
        ],
        OUTPUT / "DGS5.csv",
        OUTPUT / "DGS10.csv",
        Path(__file__),
        *[
            ROOT / "src/mc_quadrants" / f"{name}.py"
            for name in ("yield_curve", "calibration", "simulation", "economic_returns")
        ],
    ]
    manifest = dict(
        paths=args.paths,
        seed=args.seed,
        origins=len(splits),
        settings=FIT,
        models=MODES,
        data="Frozen US SPY/IEF 2006–2025; revised macro and Treasury data; overlapping expanding-window forecasts; not real-time vintages or untouched holdout",
        comparison="Existing production RW/adaptive use asset-required seven-year yield; NS additionally uses the five-tenor panel. Measures complete model change, not isolated factor parameterization.",
        horizons=[1, 3, 12],
        portfolio_horizon=12,
        sources={code: f"https://fred.stlouisfed.org/data/{code}" for code in ("DGS5", "DGS10")},
        hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(yield_summary[["yield_crps", "yield_rmse", "yield_coverage_90"]].to_string(), flush=True)
    print(
        portfolio[portfolio.stocks.isin([0, 0.6, 1])]
        .groupby(["model", "stocks"])[["cagr_crps", "max_drawdown_crps", "max_drawdown_coverage_90"]]
        .mean()
        .to_string(),
        flush=True,
    )


if __name__ == "__main__":
    main()
