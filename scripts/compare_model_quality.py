#!/usr/bin/env python3
"""Reproducible retrospective deployment comparison; never edits production code.

Downloads adjusted ETF/FRED history, freezes it, and compares the committed
pre-upgrade package with the current working package in separate processes.
Macro data are revised history with a fixed one-month availability approximation.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import pickle
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
import pandas as pd


def scalar_score(draws, actual):
    values = np.sort(np.asarray(draws))
    n = len(values)
    return float(np.mean(np.abs(values - actual)) - np.sum((2 * np.arange(1, n + 1) - n - 1) * values) / n**2)


def vector_score(draws, actual):
    """Exact empirical energy score, independent of mixture ordering."""
    values = np.asarray(draws)
    distance = 0.0
    for start in range(0, len(values), 128):
        distance += np.linalg.norm(values[start : start + 128, None] - values[None], axis=2).sum()
    return float(np.linalg.norm(values - actual, axis=1).mean() - 0.5 * distance / len(values) ** 2)


def worker(args):
    sys.path.insert(0, args.package)
    from mc_quadrants.calibration import calibrate_quadrant_model
    from mc_quadrants.simulation import simulate_returns
    from mc_quadrants.uncertainty import bootstrap_quadrant_models

    out = Path(args.output)
    with (out / "data.pkl").open("rb") as handle:
        macro, returns = pickle.load(handle)
    # Discard the inception month's partial DBC return.
    returns = returns.loc["2006-03-01":]
    kwargs = dict(
        growth_threshold="median",
        inflation_threshold="median",
        threshold_window=12,
        macro_lag_periods=0,
        probabilistic_regimes=True,
        mean_prior_strength=24.0,
        duration_prior_strength=8.0,
        joint_macro=True,
        macro_model="bvar_ensemble",
    )
    selection = None
    if args.variant == "updated":
        from mc_quadrants.model_selection import select_calibration_hyperparameters
        from mc_quadrants.uncertainty import anchor_parameter_models

        development = returns.loc[returns.index < "2020-01-01"]
        selection = select_calibration_hyperparameters(
            development,
            macro.loc[macro.index <= development.index[-1]],
            validation_kwargs=dict(
                growth_col="growth",
                inflation_col="inflation",
                growth_threshold="median",
                inflation_threshold="median",
                threshold_window=12,
                macro_lag_periods=0,
                probabilistic_regimes=True,
                mean_prior_strength=24.0,
                duration_prior_strength=8.0,
                weights=dict(zip(returns.columns, [0.4, 0.3, 0.15, 0.15])),
            ),
        )
        kwargs.update(selection["selected_parameters"])
    weights = np.array([0.4, 0.3, 0.15, 0.15])
    rows = []
    started = time.perf_counter()
    for position in np.flatnonzero(returns.index >= "2020-01-01"):
        target = returns.index[position]
        training = returns.iloc[:position]
        training_macro = macro.loc[macro.index <= training.index[-1]]
        reference = calibrate_quadrant_model(training, training_macro, **kwargs)
        models = bootstrap_quadrant_models(
            training, training_macro, draws=8, block_size=12, random_seed=91000 + int(position), **kwargs
        )
        if args.variant == "updated":
            models = anchor_parameter_models(models, reference)
        cubes, macro_cubes = [], []
        for draw, model in enumerate(models):
            coefficients = (
                {
                    key: model.metadata["dependence_fit"][key]
                    for key in ("garch_alpha", "garch_beta", "dcc_alpha", "dcc_beta", "dcc_asymmetry")
                }
                if args.variant == "updated"
                else {}
            )
            result = simulate_returns(
                model,
                periods=12,
                paths=args.paths // 8,
                random_seed=202600 + int(position) * 101 + draw,
                duration_model="semi_markov",
                garch=True,
                dynamic_correlation=True,
                joint_macro=True,
                macro_parameter_uncertainty=args.variant == "baseline",
                return_regime_codes=True,
                **coefficients,
            )
            cubes.append(result.returns)
            if result.macro_paths is not None:
                macro_cubes.append(result.macro_paths)
        samples = np.concatenate(cubes, axis=1)
        actual = returns.iloc[position].to_numpy()
        first = samples[0]
        energy = vector_score(first, actual)
        portfolio = np.expm1(samples) @ weights
        observed_portfolio = np.expm1(returns.iloc[position : position + 12].to_numpy()) @ weights
        actual_1m = float(observed_portfolio[0])
        q05, q50, q95 = np.quantile(portfolio[0], [0.05, 0.5, 0.95])
        row = dict(
            date=str(target.date()),
            energy=float(energy),
            crps_1m=scalar_score(portfolio[0], actual_1m),
            actual_1m=actual_1m,
            median_1m=float(q50),
            breach_05=int(actual_1m < q05),
            coverage_90=int(q05 <= actual_1m <= q95),
            interval_width=float(q95 - q05),
            pit=float(np.mean(portfolio[0] <= actual_1m)),
            loss_probability=float(np.mean(portfolio[0] < 0)),
            loss_actual=int(actual_1m < 0),
            expected_duration=float(np.mean(list(reference.metadata["expected_duration_months"].values()))),
        )
        row["loss_brier"] = (row["loss_probability"] - row["loss_actual"]) ** 2
        for horizon in (3, 12):
            if position + horizon > len(returns):
                continue
            wealth = np.cumprod(1 + portfolio[:horizon], axis=0)
            realized = np.cumprod(1 + observed_portfolio[:horizon])
            row[f"crps_{horizon}m"] = scalar_score(wealth[-1] - 1, realized[-1] - 1)
            row[f"median_{horizon}m"] = float(np.median(wealth[-1]) - 1)
            row[f"actual_{horizon}m"] = float(realized[-1] - 1)
            nav = np.vstack([np.ones(wealth.shape[1]), wealth])
            drawdown = np.max(1 - nav / np.maximum.accumulate(nav, axis=0), axis=0)
            actual_nav = np.r_[1, realized]
            actual_dd = float(np.max(1 - actual_nav / np.maximum.accumulate(actual_nav)))
            row[f"drawdown_crps_{horizon}m"] = scalar_score(drawdown, actual_dd)
        if macro_cubes:
            predicted = np.concatenate(macro_cubes, axis=1)[0].mean(axis=0)
            if target in macro.index:
                for i, column in enumerate(macro.columns):
                    row[f"macro_{column}_squared_error"] = float(
                        (predicted[i] - macro.loc[target, column]) ** 2
                    )
        rows.append(row)
        if len(rows) % 6 == 0:
            (out / f"{args.variant}_partial.json").write_text(json.dumps(rows, allow_nan=False))
            print(
                f"{args.variant}: {len(rows)}/72 monthly origins; {time.perf_counter() - started:.1f}s",
                flush=True,
            )
    result = dict(
        variant=args.variant,
        seconds=time.perf_counter() - started,
        settings=kwargs,
        selection=selection,
        rows=rows,
    )
    (out / f"{args.variant}.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print(f"Completed {args.variant}", flush=True)


def paired_ci(values, length=12, repeats=4000):
    rng = np.random.default_rng(481)
    values = np.asarray(values)
    count = len(values)
    starts = rng.integers(count, size=(repeats, int(np.ceil(count / length))))
    indexes = ((starts[:, :, None] + np.arange(length)) % count).reshape(repeats, -1)[:, :count]
    return np.quantile(values[indexes].mean(axis=1), [0.025, 0.975]).tolist()


def report(out):
    manifest = json.loads((out / "manifest.json").read_text())
    assert hashlib.sha256((out / "data.pkl").read_bytes()).hexdigest() == manifest["data_sha256"]
    old = json.loads((out / "baseline.json").read_text())
    new = json.loads((out / "updated.json").read_text())
    a, b = (pd.DataFrame(result["rows"]).set_index("date") for result in (old, new))
    assert a.index.equals(b.index)
    metrics = [
        "energy",
        "crps_1m",
        "loss_brier",
        "crps_3m",
        "crps_12m",
        "drawdown_crps_3m",
        "drawdown_crps_12m",
    ]
    results = {}
    for metric in metrics:
        paired = pd.concat([a[metric], b[metric]], axis=1).dropna()
        difference = paired.iloc[:, 1] - paired.iloc[:, 0]
        results[metric] = dict(
            baseline=float(paired.iloc[:, 0].mean()),
            updated=float(paired.iloc[:, 1].mean()),
            improvement_percent=float(-difference.mean() / paired.iloc[:, 0].mean() * 100),
            delta_ci_95=paired_ci(difference),
            observations=len(paired),
        )
    diagnostics = {
        variant: {
            key: float(frame[key].mean())
            for key in ["breach_05", "coverage_90", "interval_width", "pit", "expected_duration"]
        }
        for variant, frame in (("baseline", a), ("updated", b))
    }
    for variant, frame in (("baseline", a), ("updated", b)):
        for key in frame.columns:
            if key.startswith("macro_"):
                diagnostics[variant][key.replace("squared_error", "rmse")] = float(np.sqrt(frame[key].mean()))
    result = dict(
        metrics=results, diagnostics=diagnostics, selected_parameters=new["selection"]["selected_parameters"]
    )
    (out / "comparison.json").write_text(json.dumps(result, indent=2))
    lines = [
        "# Before/after retrospective forecast comparison",
        "",
        f"The updated model improved {sum(value['improvement_percent'] > 0 for value in results.values())} of {len(results)} return/loss/drawdown score point estimates. Negative improvement percentages below mean deterioration, not improvement.",
        "This bundled comparison does not identify which individual change caused a gain or regression. Macro RMSE and interval coverage are separate diagnostics, not evidence that all return forecasts improved.",
        "",
        "72 monthly forecast origins, January 2020–December 2025; expanding training history.",
        "SPY/IEF/GLD/DBC, weights 40/30/15/15; 4,096 paths, 8 bootstrap models per origin.",
        "Both use joint macro, GARCH/ADCC, semi-Markov durations and monthly rebalancing; no taxes or trading costs.",
        "Training starts in March 2006, excluding DBC's partial inception month. Growth is industrial-production YoY; inflation is CPI YoY; rates are FEDFUNDS.",
        f"Baseline commit: `{manifest['baseline_revision']}`. Raw data SHA-256: `{manifest['data_sha256']}`.",
        "",
        "| Score (lower is better) | Baseline | Updated | Improvement | 95% CI of updated − baseline |",
        "|---|---:|---:|---:|---|",
    ]
    for key, value in results.items():
        lines.append(
            f"| {key} | {value['baseline']:.6f} | {value['updated']:.6f} | {value['improvement_percent']:+.1f}% | {value['delta_ci_95']} |"
        )
    lines += ["", "## Calibration checks", "", "| Diagnostic | Baseline | Updated |", "|---|---:|---:|"]
    for key in diagnostics["baseline"]:
        lines.append(f"| {key} | {diagnostics['baseline'][key]:.6f} | {diagnostics['updated'][key]:.6f} |")
    lines += [
        "",
        "Target: 5% lower-tail breaches and 90% interval coverage. Wider intervals are not automatically better.",
        "Macro RMSE is in the series' percentage-point units, not investment returns.",
        "",
        "## One-month portfolio score by calendar year",
        "",
        "| Year | Baseline CRPS | Updated CRPS |",
        "|---|---:|---:|",
    ]
    for year in range(2020, 2026):
        mask = a.index.str.startswith(str(year))
        lines.append(
            f"| {year} | {a.loc[mask, 'crps_1m'].mean():.6f} | {b.loc[mask, 'crps_1m'].mean():.6f} |"
        )
    lines += ["", f"Selected pre-2020 settings: `{new['selection']['selected_parameters']}`."]
    lines += [
        "",
        "## Limits",
        "",
        "This is a paired retrospective ablation, not a point-in-time investment backtest. FRED history is revised; availability uses a fixed one-month approximation. No synthetic pre-inception returns are used.",
        "The updated settings were selected using only pre-2020 data. Both versions are refitted at each origin using past data only. Baseline is the committed package; updated is the working package. The original baseline bootstrap starting-state and macro-uncertainty behavior are preserved deliberately.",
        "Confidence intervals use paired circular 12-month block resampling (4,000 replicates), accommodating overlapping annual forecasts. They do not adjust for multiple comparisons or eliminate Monte Carlo error. Only one seed schedule and one portfolio are tested.",
        "An interval excluding zero suggests a paired difference under this resampling scheme, not universal superiority or inferiority. This six-year sample is particularly limited for tail-risk validation.",
        "Three-/12-month observations are limited to horizons that end by December 2025. There is no 30-year realized holdout and no direct observation of the true latent economic regime.",
        "The fixed asset universe creates selection/survivorship limitations. Results cannot establish future forecasting or investment performance.",
        "",
        "Detailed calibration/coverage diagnostics and data/code fingerprints are in the adjacent JSON files.",
    ]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="research/model_quality_2025")
    parser.add_argument("--package")
    parser.add_argument("--variant", choices=["baseline", "updated"])
    parser.add_argument("--paths", type=int, default=4096)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    if args.paths < 8 or args.paths % 8:
        parser.error("paths must be a positive multiple of eight")
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.variant:
        worker(args)
    elif args.report:
        report(out)
    elif args.prepare:
        from mc_quadrants.data import load_market_data

        macro, returns, _, _ = load_market_data(
            ["SPY", "IEF", "GLD", "DBC"], "2005-01-01", "2025-12-31", synthetic_assets=()
        )
        returns = returns.dropna()
        macro.index = macro.index + pd.offsets.MonthEnd(1)
        macro.attrs.update(availability_aligned=True, point_in_time=False)
        data = pickle.dumps((macro, returns))
        (out / "data.pkl").write_bytes(data)
        root = Path(__file__).resolve().parents[1]
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        archive = subprocess.check_output(["git", "archive", revision, "src"], cwd=root)
        baseline = out / "baseline_package"
        baseline.mkdir(exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(archive)) as handle:
            handle.extractall(baseline, filter="data")
        hashes = {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (root / "src/mc_quadrants").glob("*")
            if path.is_file()
        }
        (out / "manifest.json").write_text(
            json.dumps(
                dict(
                    baseline_revision=revision,
                    data_sha256=hashlib.sha256(data).hexdigest(),
                    working_files_sha256=hashes,
                    return_start=str(returns.index.min()),
                    return_end=str(returns.index.max()),
                    macro_vintage="latest revised",
                    sources=["Yahoo adjusted ETF prices", "FRED INDPRO/CPIAUCSL/FEDFUNDS"],
                ),
                indent=2,
            )
        )
        print(f"Prepared {len(returns)} monthly observations in {out}")
    else:
        parser.error("Choose --prepare, --variant with --package, or --report")


if __name__ == "__main__":
    main()
