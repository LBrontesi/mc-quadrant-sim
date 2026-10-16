#!/usr/bin/env python3
"""Evaluate six richer SV specifications on the frozen monthly ETF benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
import mc_quadrants.dependence as dependence  # noqa: E402
import mc_quadrants.simulation as simulation  # noqa: E402
from mc_quadrants.calibration import calibrate_quadrant_model  # noqa: E402
from mc_quadrants.native import sample_mnts_subordinators_native  # noqa: E402
from scripts.compare_qarch_forecasts import SETTINGS, first_variance, score_forecast  # noqa: E402
from scripts.compare_volatility_models import update  # noqa: E402
from scripts.complex_sv_research import (  # noqa: E402
    fit_complex,
    fit_lifted,
    fit_rich_factor,
    native,
    simulate_complex,
    simulate_lifted,
    simulate_rich_factor,
)

ORIGINAL_FIT = dependence.fit_dependence
ORIGINAL_MACRO = simulation.simulate_joint_regime_macro_paths


def prepare_context(task):
    p, macro, returns, rv, baseline, paths, seed, grid_points, quadrature = task
    captured = {}
    subordinators = []

    def capture_fit(r, regimes, moments, **kwargs):
        states = list(moments)
        codes = np.array([states.index(s) for s in regimes])
        means = np.stack([moments[s].mean.to_numpy() for s in states])
        levels = np.stack([np.diag(moments[s].covariance) for s in states])
        captured.update(
            x=(r.to_numpy() - means[codes]) / np.sqrt(levels[codes]),
            codes=codes,
            states=states,
            dates=r.index,
            levels=levels,
        )
        return ORIGINAL_FIT(r, regimes, moments, **kwargs)

    def capture_macro(*args, **kwargs):
        result = ORIGINAL_MACRO(*args, **kwargs)
        captured["macro_shocks"] = result[2]
        return result

    def capture_subordinator(rng, n, a, t):
        result = sample_mnts_subordinators_native(n, a, t, int(rng.integers(0, 2**62)))
        subordinators.append(result)
        return result

    dependence.fit_dependence = capture_fit
    simulation.simulate_joint_regime_macro_paths = capture_macro
    simulation.simulate_parametric_native = lambda **kwargs: None
    simulation.sample_mnts_subordinators = capture_subordinator
    training = returns.iloc[:p]
    model = calibrate_quadrant_model(
        training, macro.loc[: training.index[-1]], volatility_model="garch", **SETTINGS
    )
    assert captured["states"] == model.states
    x, codes = captured["x"], captured["codes"]
    dep = model.metadata["dependence_fit"]
    dynamics = model.metadata["macro_dynamics"]
    beta = np.asarray(dynamics["return_betas"])
    levels = captured["levels"]
    macro_var = np.stack(
        [
            np.diag(beta.T @ np.asarray(dynamics["state_innovation_covariances"][s]) @ beta)
            for s in model.states
        ]
    )
    observed_rv = rv.reindex(captured["dates"]).to_numpy()
    assert np.isfinite(observed_rv).all() and np.all(observed_rv > 0)
    result = simulation.simulate_returns(
        model,
        periods=12,
        paths=paths,
        random_seed=seed + p * 101,
        duration_model="semi_markov",
        joint_macro=True,
        macro_parameter_uncertainty=False,
        dynamic_correlation=True,
        garch=False,
        return_regime_codes=True,
        **{key: dep[key] for key in ("dcc_alpha", "dcc_beta", "dcc_asymmetry")},
    )
    means = np.stack([model.moments[s].mean.to_numpy() for s in model.states])
    shared_mean = means[result.regimes] + captured["macro_shocks"] @ beta
    vols = np.sqrt(levels[result.regimes])
    z = (result.returns - shared_mean) / vols
    gaussian = np.empty_like(z)
    counter = 0
    for t in range(12):
        for i, state in enumerate(model.states):
            mask = result.regimes[t] == i
            if not mask.any():
                continue
            sub = subordinators[counter]
            counter += 1
            assert len(sub) == mask.sum()
            law = dynamics["residual_mnts"][state]
            nu = law.skewness.reindex(model.assets).to_numpy()
            scale2 = np.maximum(1 - nu * nu * (1 - law.tail_index / 2) / law.tempering, 1e-10)
            gaussian[t, mask] = (z[t, mask] - nu * (sub[:, None] - 1)) / np.sqrt(sub[:, None] * scale2)
    assert counter == len(subordinators)
    mean, baseline_var, _ = first_variance(model, "garch")
    posterior = simulation._latest_state_age_posterior(model)
    probabilities = (
        posterior.sum(axis=1)
        if posterior is not None
        else simulation.stationary_distribution(model.transition_matrix).to_numpy()
    )
    actual = returns.iloc[p : p + 12].to_numpy()
    h = np.broadcast_to(baseline["next_variance_ratio"], z[0].shape).copy()
    params = np.asarray(baseline["parameters"])
    replay = np.empty_like(z)
    for t in range(12):
        residual = np.sqrt(h) * z[t]
        replay[t] = shared_mean[t] + vols[t] * residual
        h = update("garch", params, h, residual, 0.5, 0.8)
    audit = score_forecast(replay, actual, mean, model.assets)
    audit_error = max(abs(audit[k] - baseline[k]) for k in ["crps_1m", "crps_3m", "crps_12m"] if k in audit)
    assert audit_error < 1e-12
    return dict(
        x=x,
        rv=observed_rv / levels[codes],
        macro=macro_var[codes] / levels[codes],
        gaussian=gaussian,
        shared_mean=shared_mean,
        vols=vols,
        mean=mean,
        baseline_var=baseline_var,
        probabilities=probabilities,
        levels=levels,
        actual=actual,
        assets=model.assets,
        date=str(returns.index[p].date()),
        audit_error=audit_error,
    )


MODELS = ["realized_skewt", "two_component", "joint_jumps", "switching", "rich_factor", "lifted_heston"]


def run_origin(task):
    p, macro, returns, rv, baseline, paths, seed, options, output = task
    target = Path(output) / "origins" / f"{returns.index[p].date()}.json"
    if target.exists():
        return json.loads(target.read_text())
    context = prepare_context((p, macro, returns, rv, baseline, paths, seed, 0, 0))
    x = context["x"]
    rows = []
    for name in MODELS:
        started = time.perf_counter()
        if name in ["realized_skewt", "two_component", "joint_jumps", "switching"]:
            fits = [
                fit_complex(
                    name,
                    x[:, j],
                    context["rv"][:, j],
                    context["macro"][:, j],
                    options["grid_points"],
                    options["component_grid"],
                )
                for j in range(4)
            ]
            normalized = simulate_complex(fits, context["gaussian"], seed + p * 101 + 23101)
            variance = np.array([f["next_variance_ratio"] for f in fits])
        elif name == "rich_factor":
            fit = fit_rich_factor(x, options["mcmc_scale"])
            fits = [fit]
            normalized = simulate_rich_factor(fit, paths, 12, seed + p * 101 + 24101)
            variance = np.asarray(fit["marginal_next_variance"])
        else:
            fits = [fit_lifted(x[:, j], options["particles"], options["steps"]) for j in range(4)]
            # Lifted forecasts are assessed marginally: separate filters supply no
            # joint asset driver specification. Do not report a fabricated portfolio score.
            normalized = np.stack(
                [
                    simulate_lifted(fit, paths, 12, seed + p * 101 + 25101 + j)[0]
                    for j, fit in enumerate(fits)
                ],
                axis=2,
            )
            variance = np.array([f["next_variance_ratio"] for f in fits])
        row = score_model(name, normalized, variance, fits, context, p)
        rows.append(row)
        print(f"{context['date']} {name}: {time.perf_counter() - started:.1f}s", flush=True)
        if name == "rich_factor":
            b, d = np.asarray(fit["loadings"]), np.asarray(fit["static_idiosyncratic_variance"])
            static = simulate_rich_factor(fit, paths, 12, seed + p * 101 + 24101, constant=True)
            rows.append(
                score_model("two_factor_t_control", static, (b * b).sum(axis=1) + d, [fit], context, p)
            )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(rows, indent=2, allow_nan=False))
    temporary.replace(target)
    return rows


def score_model(name, normalized, expected, fits, context, p):
    samples = context["shared_mean"] + context["vols"] * normalized
    variance = context["baseline_var"] + (context["probabilities"] @ context["levels"]) * (expected - 1)
    qlike = np.log(variance) + (context["actual"][0] - context["mean"]) ** 2 / variance
    score = score_forecast(samples, context["actual"], context["mean"], context["assets"])
    if name == "lifted_heston":
        # Marginal predictions are valid, but cross-asset correlations are unspecified.
        score = {k: v for k, v in score.items() if any(k.endswith("_" + t) for t in context["assets"])}
    row = dict(
        model=name,
        date=context["date"],
        training_months=p,
        **score,
        qlike_assets_mc=score.get("qlike_assets"),
        asset_fits=fits,
        baseline_replay_max_error=context["audit_error"],
    )
    row["qlike_assets"] = float(qlike.mean())
    for j, ticker in enumerate(context["assets"]):
        row[f"qlike_{ticker}"] = float(qlike[j])
        row[f"analytic_variance_{ticker}"] = float(variance[j])
    row["crps_assets"] = float(np.mean([row[f"crps_{t}"] for t in context["assets"]]))
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="research/complex_sv_2026-10-05/main")
    parser.add_argument("--baseline", default="research/independent_volatility_models_2026-10-04")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--grid-points", type=int, default=41)
    parser.add_argument("--component-grid", type=int, default=21)
    parser.add_argument("--particles", type=int, default=128)
    parser.add_argument("--steps", type=int, default=21)
    parser.add_argument("--mcmc-scale", type=int, default=1)
    args = parser.parse_args()
    baseline = ROOT / args.baseline
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    bm = json.loads((baseline / "manifest.json").read_text())
    data = (ROOT / "research/model_quality_2025/data.pkl").read_bytes()
    assert hashlib.sha256(data).hexdigest() == bm["data_sha256"]
    for path, digest in bm["source_hashes"].items():
        if path.startswith("src/"):
            assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest, path
    macro, returns = pickle.loads(data)
    returns = returns.loc["2006-03-01":].dropna()
    rv_path = ROOT / "research/sv_extensions_2026-10-04/monthly_realized_variance.csv"
    rv = pd.read_csv(rv_path, index_col=0, parse_dates=True).reindex(columns=returns.columns)
    rows = json.loads((baseline / "forecasts.json").read_text())
    garch = {r["date"]: r for r in rows if r["model"] == "garch"}
    positions = list(range(60, len(returns)))[:: args.stride]
    if args.limit:
        positions = positions[: args.limit]
    options = {
        key: getattr(args, key)
        for key in ["grid_points", "component_grid", "particles", "steps", "mcmc_scale"]
    }
    manifest = dict(
        origins=len(positions),
        dates=[str(returns.index[p].date()) for p in positions],
        models=MODELS + ["two_factor_t_control"],
        options=options,
        seed=bm["seed"],
        paths=bm["paths"],
        data_sha256=bm["data_sha256"],
        rv_sha256=hashlib.sha256(rv_path.read_bytes()).hexdigest(),
        settings=SETTINGS,
        baseline=str(baseline),
        macro_vintage=bm["macro_vintage"],
        source_hashes={
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in list((ROOT / "src/mc_quadrants").glob("*.py"))
            + [
                Path(__file__),
                ROOT / "scripts/complex_sv_research.py",
                ROOT / "scripts/complex_sv_native.cpp",
            ]
        },
    )
    path = output / "manifest.json"
    if path.exists():
        saved = json.loads(path.read_text())
        for key in manifest:
            assert manifest[key] == saved[key], f"Cannot resume changed configuration: {key}"
    path.write_text(json.dumps(manifest, indent=2))
    native()
    tasks = [
        (
            p,
            macro,
            returns,
            rv,
            garch[str(returns.index[p].date())],
            bm["paths"],
            bm["seed"],
            options,
            str(output),
        )
        for p in positions
    ]
    results = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as executor:
        for n, batch in enumerate(executor.map(run_origin, tasks), 1):
            results.extend(batch)
            if n % 10 == 0 or n == len(tasks):
                (output / "forecasts.json").write_text(json.dumps(results, indent=2, allow_nan=False))
            print(f"{n}/{len(tasks)} origins; {time.perf_counter() - started:.1f}s", flush=True)
    pd.DataFrame(results).to_csv(output / "forecasts.csv", index=False)
    manifest["elapsed_seconds"] = time.perf_counter() - started
    path.write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
