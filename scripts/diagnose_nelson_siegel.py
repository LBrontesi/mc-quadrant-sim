"""Research-only ablations of curve geometry, dynamics and macro conditioning.

Overrides exist only during calibration; each candidate refits its innovation
covariance and asset exposures against its own conditional forecast. Production
code and the original benchmark outputs are unchanged.
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
from mc_quadrants import economic_returns, yield_curve  # noqa: E402
from mc_quadrants.calibration import calibrate_quadrant_model  # noqa: E402
from mc_quadrants.path_validation import scalar_scores  # noqa: E402
from scripts.benchmark_nelson_siegel import OUTPUT, read_fred_table  # noqa: E402
from scripts.benchmark_reality_fixes import load_data  # noqa: E402
from scripts.investigate_allocation_realism import FIT, allocation_scores, simulate  # noqa: E402

DEST = OUTPUT / "diagnostics"
VARIANTS = (
    "adaptive",
    "adaptive_panel",
    "original",
    "no_cap",
    "no_macro",
    "no_regime",
    "diagonal_ar",
    "factor_rw",
    "no_inflation",
    "zero_inflation",
)
ORIGINAL_FIT = yield_curve.fit_curve
ORIGINAL_COLUMNS = economic_returns.economic_columns


def fit_candidate(variant, values, columns, valid, membership, states, macro_columns):
    spec, prediction = ORIGINAL_FIT(values, columns, valid, membership, states, macro_columns)
    if variant == "original":
        return spec, prediction
    factors = values[:, spec["indices"]] @ np.asarray(spec["inverse"]).T
    predictors = np.column_stack([factors, values[:, spec["macro_indices"]]])
    x = (predictors[:-1] - spec["mean"]) / spec["scale"]
    delta = np.diff(factors, axis=0)
    coefficient = np.asarray(spec["coefficient"]).copy()
    if variant in ("no_cap", "no_macro", "no_inflation"):
        size = 3 if variant == "no_macro" else x.shape[1]
        active = list(range(size))
        if variant == "no_inflation":
            active.remove(3 + macro_columns.index("inflation"))
        coefficient[:] = 0
        design = x[valid][:, active]
        coefficient[active] = np.linalg.solve(
            design.T @ design + 24 * np.eye(len(active)), design.T @ delta[valid]
        )
        if variant != "no_cap":
            transition = np.eye(3) + coefficient[:3] / np.asarray(spec["scale"])[:3, None]
            radius = float(np.max(np.abs(np.linalg.eigvals(transition))))
            if radius > 0.995:
                coefficient[:3] = (transition * (0.995 / radius) - np.eye(3)) * np.asarray(spec["scale"])[
                    :3, None
                ]
    elif variant == "zero_inflation":
        coefficient[3 + macro_columns.index("inflation")] = 0
    elif variant == "diagonal_ar":
        coefficient[:] = 0
        for j in range(3):
            slope = x[valid, j] @ delta[valid, j] / (x[valid, j] @ x[valid, j] + 24)
            persistence = np.clip(1 + slope / spec["scale"][j], -0.995, 0.995)
            coefficient[j, j] = (persistence - 1) * spec["scale"][j]
    elif variant == "factor_rw":
        coefficient[:] = 0
    residual = delta - x @ coefficient
    drift = residual[valid].mean(axis=0)
    drifts = []
    for i in range(len(states)):
        weights = membership[:-1, i] * valid
        local = (weights @ residual + 24 * drift) / (weights.sum() + 24)
        drifts.append(drift if variant == "no_regime" else local)
    if variant == "factor_rw":
        drifts = np.zeros((len(states), 3))
    spec.update(coefficient=coefficient.tolist(), state_drifts=np.asarray(drifts).tolist())
    return spec, yield_curve.curve_forecast(values[:-1], spec, membership[:-1])


def panel_columns(profiles, options):
    return sorted(set(ORIGINAL_COLUMNS(profiles, options)) | set(yield_curve.CURVE_TENORS))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--output", type=Path, default=DEST)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=VARIANTS)
    args = parser.parse_args()
    dest = args.output
    dest.mkdir(parents=True, exist_ok=True)
    macro, returns, sources = load_data()
    for code in ("DGS5", "DGS10"):
        sources[code] = read_fred_table(OUTPUT / f"{code}.html", code)
    for column, code in zip(yield_curve.CURVE_TENORS, ("DGS2", "DGS5", "DGS7", "DGS10", "DGS20")):
        macro[column] = sources[code].reindex(macro.index)
    splits = sorted(
        set(range(60, len(returns) - 11, 6))
        | {returns.index.get_loc(pd.Timestamp(d)) for d in ("2020-01-31", "2022-01-31")}
    )
    rows, yields, geometry, decomposition, trajectories = [], [], [], [], []
    started = time.perf_counter()
    for number, split in enumerate(splits):
        origin = returns.index[split - 1]
        for variant in args.variants:
            mode = "adaptive" if variant.startswith("adaptive") else "nelson_siegel"

            def fitter(*args, **kwargs):
                return fit_candidate(variant, *args, **kwargs)

            with (
                patch.object(yield_curve, "fit_curve", fitter),
                patch.object(
                    economic_returns,
                    "economic_columns",
                    panel_columns if variant == "adaptive_panel" else ORIGINAL_COLUMNS,
                ),
            ):
                model = calibrate_quadrant_model(
                    returns.iloc[:split],
                    macro.loc[:origin],
                    **FIT,
                    structural_returns=True,
                    economic_returns={"yield_model": mode},
                )
            result = simulate(model, 12, 4096, args.seed + number)
            rows.extend(
                allocation_scores(
                    returns.iloc[split : split + 12].to_numpy(), result.returns, origin, 12, variant
                )
            )
            index = result.macro_columns.index("bond_yield")
            for h in (1, 3, 12):
                actual = sources["DGS7"].loc[origin + pd.offsets.MonthEnd(h)]
                draws = result.macro_paths[h - 1, :, index]
                yields.append(
                    dict(
                        origin=origin,
                        model=variant,
                        horizon=h,
                        **scalar_scores("yield", actual, draws),
                        squared_error=float((draws.mean() - actual) ** 2),
                    )
                )
            curve = model.metadata["macro_dynamics"].get("curve_dynamics")
            if curve:
                dynamics = model.metadata["macro_dynamics"]
                latest = np.asarray(dynamics["latest"])
                basis = np.asarray(curve["loadings"])
                factors = latest[curve["indices"]] @ np.asarray(curve["inverse"]).T
                geometry.append(
                    dict(
                        origin=origin,
                        model=variant,
                        condition_number=float(np.linalg.cond(basis)),
                        seven_year_fit_error=float((factors @ basis.T)[2] - latest[curve["indices"][2]]),
                        panel_fit_rmse=float(
                            np.sqrt(np.mean((factors @ basis.T - latest[curve["indices"]]) ** 2))
                        ),
                    )
                )
                predictors = np.r_[factors, latest[curve["macro_indices"]]]
                standardized = (predictors - curve["mean"]) / curve["scale"]
                terms = standardized * (np.asarray(curve["coefficient"]) @ basis[2])
                labels = ["level", "slope", "curvature"] + [
                    dynamics["columns"][i] for i in curve["macro_indices"]
                ]
                for label, term in zip(labels, terms):
                    decomposition.append(
                        dict(
                            origin=origin,
                            model=variant,
                            predictor=label,
                            seven_year_monthly_change=float(term),
                        )
                    )
                for state, drift in zip(curve["states"], np.asarray(curve["state_drifts"]) @ basis[2]):
                    decomposition.append(
                        dict(
                            origin=origin,
                            model=variant,
                            predictor="drift_" + state,
                            seven_year_monthly_change=float(drift),
                        )
                    )
            if origin == pd.Timestamp("2021-12-31"):
                for month in range(12):
                    low, median, high = np.quantile(result.macro_paths[month, :, index], [0.05, 0.5, 0.95])
                    trajectories.append(
                        dict(
                            model=variant,
                            month=month + 1,
                            mean=float(result.macro_paths[month, :, index].mean()),
                            p05=low,
                            p50=median,
                            p95=high,
                            actual=sources["DGS7"].loc[origin + pd.offsets.MonthEnd(month + 1)],
                        )
                    )
        print(
            f"{number + 1}/{len(splits)} {origin.date()} ({time.perf_counter() - started:.1f}s)", flush=True
        )
    for filename, data in [
        ("portfolio_scores", rows),
        ("yield_scores", yields),
        ("geometry", geometry),
        ("decomposition", decomposition),
        ("2022_trajectories", trajectories),
    ]:
        pd.DataFrame(data).to_csv(dest / f"{filename}.csv", index=False)
    pd.DataFrame(rows).groupby(["model", "stocks"]).mean(numeric_only=True).to_csv(
        dest / "portfolio_summary.csv"
    )
    summary = pd.DataFrame(yields).groupby(["model", "horizon"]).mean(numeric_only=True)
    summary["rmse"] = np.sqrt(summary["squared_error"])
    summary.to_csv(dest / "yield_summary.csv")
    files = [
        Path(__file__),
        ROOT / "scripts/benchmark_nelson_siegel.py",
        *[
            ROOT / "src/mc_quadrants" / f"{name}.py"
            for name in ("yield_curve", "calibration", "economic_returns", "simulation")
        ],
    ]
    (dest / "manifest.json").write_text(
        json.dumps(
            dict(
                origins=len(splits),
                paths=4096,
                seed=args.seed,
                variants=args.variants,
                settings=FIT,
                data_manifest="../manifest.json",
                hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
            ),
            indent=2,
        )
        + "\n"
    )
    print(summary[["yield_crps", "rmse", "yield_coverage_90"]].to_string(), flush=True)


if __name__ == "__main__":
    main()
