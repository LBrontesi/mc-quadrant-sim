#!/usr/bin/env python3
"""Reproducible allocation behavior study. Does not alter production models.

Frozen revised data, chronological prefix refits, shared paths across weights,
paired iid/block benchmarks, and explicitly labeled full-history sensitivities.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mc_quadrants.calibration import calibrate_quadrant_model  # noqa: E402
from mc_quadrants.path_validation import crps  # noqa: E402
from mc_quadrants.simulation import simulate_portfolio_paths, simulate_returns  # noqa: E402
from mc_quadrants.types import MNTSParameters  # noqa: E402
from mc_quadrants.validation import _sample_mnts_observations, _unconditional_mnts_moments  # noqa: E402

WEIGHTS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
HORIZONS = (12, 36, 60, 120)
FIT = dict(
    threshold_window=12,
    probabilistic_regimes=True,
    mean_prior_strength=24.0,
    duration_prior_strength=8.0,
    joint_macro=True,
    macro_model="bvar_ensemble",
)


def portfolio_statistics(log_returns, weight):
    """Exact monthly-rebalanced returns; underwater spells include censoring."""
    growth = 1 + np.expm1(log_returns) @ np.array([weight, 1 - weight])
    if not np.isfinite(growth).all() or (growth <= 0).any():
        raise ValueError("Nonfinite/nonpositive portfolio growth; no clipping permitted.")
    wealth = np.vstack([np.ones((1, growth.shape[1])), np.cumprod(growth, axis=0)])
    peaks = np.maximum.accumulate(wealth, axis=0)
    drawdowns = 1 - wealth / peaks
    underwater = wealth[1:] < peaks[1:] * (1 - 1e-12)
    length = np.zeros(growth.shape[1], dtype=int)
    longest = length.copy()
    for month in underwater:
        length = np.where(month, length + 1, 0)
        longest = np.maximum(longest, length)
    return dict(
        cagr=wealth[-1] ** (12 / len(growth)) - 1,
        max_drawdown=drawdowns.max(axis=0),
        max_underwater_months=longest,
        underwater_at_end=underwater[-1].astype(float),
        terminal_loss=(wealth[-1] < 1).astype(float),
    )


def paired_block_returns(training, months, paths, seed, block_size=12):
    """Stationary bootstrap of paired observed asset returns, not parameters."""
    values = np.asarray(training)
    rng = np.random.default_rng(seed)
    indexes = rng.integers(len(values), size=paths)
    samples = np.empty((months, paths, values.shape[1]))
    for t in range(months):
        if t:
            restart = rng.random(paths) < 1 / block_size
            indexes = np.where(restart, rng.integers(len(values), size=paths), (indexes + 1) % len(values))
        samples[t] = values[indexes]
    return samples


def simulate(model, months, paths, seed, *, garch=True, dynamic=True, macro=True):
    fitted = model.metadata["dependence_fit"]
    coefficients = {
        k: float(fitted[k]) for k in ("garch_alpha", "garch_beta", "dcc_alpha", "dcc_beta", "dcc_asymmetry")
    }
    return simulate_returns(
        model,
        periods=months,
        paths=paths,
        random_seed=seed,
        duration_model="semi_markov",
        garch=garch,
        dynamic_correlation=dynamic,
        joint_macro=macro,
        **coefficients,
    )


def outcome_score(actual, draws):
    lower, median, upper = np.quantile(draws, [0.05, 0.5, 0.95])
    return dict(
        actual=float(actual),
        mean=float(np.mean(draws)),
        p05=float(lower),
        p50=float(median),
        p95=float(upper),
        crps=crps(actual, draws),
        coverage_90=float(lower <= actual <= upper),
        pit=float((np.mean(draws < actual) + np.mean(draws <= actual)) / 2),
    )


def allocation_scores(actual_returns, forecast, origin, horizon, label):
    rows = []
    for weight in WEIGHTS:
        actual = portfolio_statistics(actual_returns[:, None], weight)
        predicted = portfolio_statistics(forecast, weight)
        row = dict(origin=str(origin.date()), horizon=horizon, stocks=weight, model=label)
        for metric in ("cagr", "max_drawdown", "max_underwater_months"):
            row.update(
                {
                    f"{metric}_{key}": value
                    for key, value in outcome_score(actual[metric][0], predicted[metric]).items()
                }
            )
        for metric in ("underwater_at_end", "terminal_loss"):
            probability = float(predicted[metric].mean())
            row.update(
                {
                    f"{metric}_probability": probability,
                    f"{metric}_actual": float(actual[metric][0]),
                    f"{metric}_brier": float((probability - actual[metric][0]) ** 2),
                }
            )
        for threshold in (0.2, 0.3, 0.5):
            prefix = f"drawdown_over_{int(threshold * 100)}"
            probability = float(np.mean(predicted["max_drawdown"] > threshold))
            event = float(actual["max_drawdown"][0] > threshold)
            row.update(
                {
                    f"{prefix}_probability": probability,
                    f"{prefix}_actual": event,
                    f"{prefix}_brier": (probability - event) ** 2,
                }
            )
        rows.append(row)
    return rows


def dependence_scores(actual, forecast, origin, horizon, label):
    x, y = forecast[:, :, 0], forecast[:, :, 1]
    xc, yc = x - x.mean(axis=0), y - y.mean(axis=0)
    corr = (xc * yc).sum(axis=0) / np.sqrt((xc**2).sum(axis=0) * (yc**2).sum(axis=0))
    actual_corr = float(np.corrcoef(actual.T)[0, 1])
    joint_probability = float(np.mean((forecast.sum(axis=0) < 0).all(axis=1)))
    joint_event = float((actual.sum(axis=0) < 0).all())
    return dict(
        origin=str(origin.date()),
        horizon=horizon,
        model=label,
        **{f"correlation_{k}": v for k, v in outcome_score(actual_corr, corr).items()},
        joint_terminal_loss_probability=joint_probability,
        joint_terminal_loss_actual=joint_event,
        joint_terminal_loss_brier=(joint_probability - joint_event) ** 2,
        joint_monthly_loss_actual=float(np.mean((actual < 0).all(axis=1))),
        joint_monthly_loss_predicted=float(np.mean((forecast < 0).all(axis=2))),
    )


def nonoverlapping_count(origins, horizon):
    end = None
    count = 0
    for origin in sorted(pd.to_datetime(origins)):
        if end is None or origin >= end:
            count += 1
            end = origin + pd.offsets.MonthEnd(horizon)
    return count


def summary_tables(scores, dependence):
    summary = []
    for (label, horizon, weight), group in scores.groupby(["model", "horizon", "stocks"]):
        row = dict(
            model=label,
            horizon=int(horizon),
            stocks=float(weight),
            origins=len(group),
            nonoverlapping_origins=nonoverlapping_count(group.origin, horizon),
        )
        for column in group:
            if column not in ("origin", "model", "horizon", "stocks"):
                row[column] = float(group[column].mean())
        summary.append(row)
    dep = dependence.groupby(["model", "horizon"]).mean(numeric_only=True).reset_index()
    return pd.DataFrame(summary), dep


def near_gaussian(model):
    """Diagnostic limit of MNTS, same target moments and dependence coefficients."""
    from copy import deepcopy

    from mc_quadrants.matrix import covariance_to_correlation

    metadata = deepcopy(model.metadata)

    def law(correlation):
        return MNTSParameters(1.5, 1e6, pd.Series(0.0, index=model.assets), correlation)

    moments = {s: replace(m, mnts=law(m.correlation)) for s, m in model.moments.items()}
    if "macro_dynamics" in metadata:
        dynamics = metadata["macro_dynamics"]
        dynamics["residual_mnts"] = {
            s: law(covariance_to_correlation(pd.DataFrame(c, index=model.assets, columns=model.assets)))
            for s, c in dynamics["return_residual_covariances"].items()
        }
    return replace(model, moments=moments, metadata=metadata)


def summarize_paths(returns, label, horizon):
    rows = []
    stats = {w: portfolio_statistics(returns, w) for w in WEIGHTS}
    for weight, result in stats.items():
        row = dict(variant=label, horizon=horizon, stocks=weight)
        for key, values in result.items():
            for suffix, value in zip(("p05", "p50", "p95"), np.quantile(values, [0.05, 0.5, 0.95])):
                row[f"{key}_{suffix}"] = float(value)
            row[f"{key}_mean"] = float(np.mean(values))
        for threshold in (0.2, 0.3, 0.5):
            row[f"drawdown_over_{int(threshold * 100)}"] = float(np.mean(result["max_drawdown"] > threshold))
        rows.append(row)
    ordering = dict(
        variant=label,
        horizon=horizon,
        stocks_return_above_bonds=float(np.mean(stats[1.0]["cagr"] > stats[0.0]["cagr"])),
        stocks_drawdown_above_bonds=float(np.mean(stats[1.0]["max_drawdown"] > stats[0.0]["max_drawdown"])),
        balanced_drawdown_below_stocks=float(
            np.mean(stats[0.6]["max_drawdown"] < stats[1.0]["max_drawdown"])
        ),
        stock20_drawdown_below_bonds=float(np.mean(stats[0.2]["max_drawdown"] < stats[0.0]["max_drawdown"])),
    )
    return rows, ordering


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "research/model_quality_2025/data.pkl")
    parser.add_argument("--output", type=Path, default=ROOT / "research/allocation_realism_2026-10-11")
    parser.add_argument("--paths", type=int, default=4096)
    parser.add_argument("--long-paths", type=int, default=20000)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261011)
    args = parser.parse_args()
    if min(args.paths, args.long_paths, args.stride) < 1:
        parser.error("paths, long-paths and stride must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    with args.data.open("rb") as handle:
        macro, returns = pickle.load(handle)
    returns = returns.loc["2006-03-01":, ["SPY", "IEF"]].dropna()
    if not returns.index.to_period("M").equals(
        pd.period_range(returns.index[0], returns.index[-1], freq="M")
    ):
        raise ValueError("Investigation requires a complete monthly panel.")
    if not macro.attrs.get("availability_aligned"):
        raise ValueError("Use explicitly availability-aligned macro inputs.")
    macro = macro.loc[: returns.index[-1]]
    files = (
        list((ROOT / "src/mc_quadrants").glob("*.py"))
        + list((ROOT / "src/mc_quadrants").glob("*.cpp"))
        + [Path(__file__)]
    )
    manifest = dict(
        data=str(args.data),
        data_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),
        source_hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        history=[str(returns.index[0].date()), str(returns.index[-1].date())],
        settings=FIT,
        paths=args.paths,
        long_paths=args.long_paths,
        stride=args.stride,
        seed=args.seed,
        data_vintage=macro.attrs.get("data_vintage"),
        point_in_time=False,
        horizons=list(HORIZONS),
        stock_weights=list(WEIGHTS),
        monthly_rebalancing=True,
        nominal=True,
        taxes_fees_cashflows=False,
        parameter_bootstrap=False,
        macro_parameter_uncertainty=True,
        monthly_drawdowns=True,
        uncertainty_note="Shared simulated returns across weights; full-history sensitivities are not out-of-sample comparisons.",
    )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    scores, dep, fit_rows = [], [], []
    positions = set(range(60, len(returns), args.stride))
    for target in ("2018-01-31", "2020-01-31", "2022-01-31"):
        positions.add(returns.index.get_loc(pd.Timestamp(target)))
    for number, split in enumerate(sorted(positions)):
        horizons = [h for h in HORIZONS if split + h <= len(returns)]
        if not horizons:
            continue
        origin = returns.index[split - 1]
        training = returns.iloc[:split]
        training_macro = macro.loc[:origin]
        model = calibrate_quadrant_model(training, training_macro, **FIT)
        seed = args.seed + origin.year * 12 + origin.month
        forecast = simulate(model, max(horizons), args.paths, seed)
        unconditional = _unconditional_mnts_moments(training)
        iid = _sample_mnts_observations(unconditional, max(horizons) * args.paths, seed + 1).reshape(
            max(horizons), args.paths, 2
        )
        bootstrap = paired_block_returns(training, max(horizons), args.paths, seed + 2)
        fit_rows.append(
            dict(
                origin=str(origin.date()),
                **{
                    k: v
                    for k, v in model.metadata["dependence_fit"].items()
                    if isinstance(v, (str, float, int))
                },
            )
        )
        for h in horizons:
            actual = returns.iloc[split : split + h].to_numpy()
            for label, draws in [
                ("production", forecast.returns),
                ("iid_mnts", iid),
                ("historical_blocks", bootstrap),
            ]:
                scores.extend(allocation_scores(actual, draws[:h], origin, h, label))
                dep.append(dependence_scores(actual, draws[:h], origin, h, label))
        if number % 5 == 0:
            print(f"Walk-forward {number + 1}/{len(positions)}: {origin.date()}", flush=True)
            pd.DataFrame(scores).to_csv(args.output / "scores.csv", index=False)
    score_frame, dep_frame = pd.DataFrame(scores), pd.DataFrame(dep)
    score_frame.to_csv(args.output / "scores.csv", index=False)
    dep_frame.to_csv(args.output / "dependence.csv", index=False)
    pd.DataFrame(fit_rows).to_csv(args.output / "fits.csv", index=False)
    summary, dep_summary = summary_tables(score_frame, dep_frame)
    summary.to_csv(args.output / "summary.csv", index=False)
    dep_summary.to_csv(args.output / "dependence_summary.csv", index=False)
    print("Walk-forward complete. Running full-history simulation sensitivities.", flush=True)
    full = calibrate_quadrant_model(returns, macro, **FIT)
    variants = [
        ("production", full, {}),
        ("no_garch", full, {"garch": False}),
        ("constant_correlation", full, {"dynamic": False}),
        ("near_gaussian_innovations", near_gaussian(full), {}),
        (
            "no_joint_macro",
            calibrate_quadrant_model(returns, macro, **{**FIT, "joint_macro": False}),
            {"macro": False},
        ),
        ("mean_prior_0", calibrate_quadrant_model(returns, macro, **{**FIT, "mean_prior_strength": 0.0}), {}),
        (
            "mean_prior_48",
            calibrate_quadrant_model(returns, macro, **{**FIT, "mean_prior_strength": 48.0}),
            {},
        ),
    ]
    long_rows, ordering = [], []
    for label, model, options in variants:
        result = simulate(model, 120, args.long_paths, args.seed, **options)
        # Verify the research reducer against actual production accounting.
        if label == "production":
            ledger = simulate_portfolio_paths(
                replace(
                    result,
                    returns=result.returns[:, :8],
                    regimes=result.regimes[:, :8],
                    macro_paths=result.macro_paths[:, :8],
                ),
                {"SPY": 0.6, "IEF": 0.4},
                initial_value=1,
                rebalance_frequency=1,
            )
            stats = portfolio_statistics(result.returns[:, :8], 0.6)
            np.testing.assert_allclose(
                ledger.iloc[-1].to_numpy() ** 0.1 - 1, stats["cagr"], rtol=1e-10, atol=1e-12
            )
        for horizon in HORIZONS:
            rows, order = summarize_paths(result.returns[:horizon], label, horizon)
            long_rows.extend(rows)
            ordering.append(order)
        print(f"Sensitivity complete: {label}", flush=True)
    pd.DataFrame(long_rows).to_csv(args.output / "sensitivities.csv", index=False)
    pd.DataFrame(ordering).to_csv(args.output / "ordering.csv", index=False)
    # Descriptive historical windows, including GFC: not a forecast backtest.
    historical = []
    for horizon in HORIZONS:
        for start in range(len(returns) - horizon + 1):
            for weight in WEIGHTS:
                values = portfolio_statistics(
                    returns.iloc[start : start + horizon].to_numpy()[:, None], weight
                )
                historical.append(
                    dict(
                        start=str(returns.index[start].date()),
                        horizon=horizon,
                        stocks=weight,
                        **{k: float(v[0]) for k, v in values.items()},
                    )
                )
    pd.DataFrame(historical).to_csv(args.output / "historical_windows.csv", index=False)
    print(f"Finished: {args.output}", flush=True)


if __name__ == "__main__":
    main()
