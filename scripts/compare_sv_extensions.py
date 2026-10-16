#!/usr/bin/env python3
"""Evaluate five SV extensions and Gaussian/factor controls on frozen ETF data."""

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
from scripts.compare_volatility_models import summarize, update  # noqa: E402
from scripts.stochastic_volatility_research import simulate_variance  # noqa: E402
from scripts.sv_extension_models import (  # noqa: E402
    fit_dynamic,
    fit_factor,
    fit_realized,
    fit_rough,
    native_filter,
    simulate_dynamic,
)

ORIGINAL_FIT = dependence.fit_dependence
ORIGINAL_MACRO = simulation.simulate_joint_regime_macro_paths


def run_origin(task):
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
    gaussian_fits = [
        fit_dynamic("leverage", x[:, j], grid_points=grid_points, quadrature=quadrature, allow_leverage=False)
        for j in range(4)
    ]
    leverage_fits = [
        fit_dynamic("leverage", x[:, j], gaussian_fits[j], grid_points=grid_points, quadrature=quadrature)
        for j in range(4)
    ]
    heston_fits = [
        fit_dynamic("heston", x[:, j], grid_points=grid_points, quadrature=quadrature) for j in range(4)
    ]
    realized_fits = [
        fit_realized(
            x[:, j], observed_rv[:, j] / levels[codes, j], macro_var[codes, j] / levels[codes, j], grid_points
        )
        for j in range(4)
    ]
    rough_fits = [fit_rough(x[:, j]) for j in range(4)]
    factor_fit = fit_factor(x, grid_points)
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
    sample_specs = {}
    for name, fits in [
        ("gaussian_sv_control", gaussian_fits),
        ("leverage_sv", leverage_fits),
        ("heston_euler", heston_fits),
    ]:
        variances = simulate_dynamic(fits, gaussian, seed + p * 101 + 13001)
        sample_specs[name] = (
            np.sqrt(variances) * gaussian,
            np.array([f["next_variance_ratio"] for f in fits]),
            fits,
        )
    variances = simulate_variance(realized_fits, 12, paths, seed + p * 101 + 14001)
    sample_specs["realized_sv"] = (
        np.sqrt(variances) * gaussian,
        np.array([f["next_variance_ratio"] for f in realized_fits]),
        realized_fits,
    )
    rng = np.random.default_rng(seed + p * 101 + 15001)
    rough_variances = np.empty_like(gaussian)
    for j, fit in enumerate(rough_fits):
        cov = np.asarray(fit["forecast_log_variance_covariance"])
        chol = np.linalg.cholesky(cov + np.eye(12) * 1e-10)
        logs = np.asarray(fit["forecast_log_variance_mean"])[:, None] + chol @ rng.standard_normal(
            (12, paths)
        )
        rough_variances[:, :, j] = np.exp(logs)
    sample_specs["rough_sv"] = (
        np.sqrt(rough_variances) * gaussian,
        np.array([f["next_variance_ratio"] for f in rough_fits]),
        rough_fits,
    )
    factor_variance = simulate_variance([factor_fit], 12, paths, seed + p * 101 + 16001)
    rng = np.random.default_rng(seed + p * 101 + 17001)
    common = rng.standard_normal((12, paths, 1))
    idio = rng.standard_normal((12, paths, 4))
    b = np.asarray(factor_fit["loadings"])
    d = np.asarray(factor_fit["idiosyncratic_variance"])
    sample_specs["factor_sv"] = (
        np.sqrt(factor_variance) * common * b + idio * np.sqrt(d),
        np.asarray(factor_fit["marginal_next_variance"]),
        [factor_fit],
    )
    sample_specs["factor_constant_control"] = (common * b + idio * np.sqrt(d), d + b * b, [factor_fit])
    # A simple realized-variance control helps distinguish new data from model complexity.
    rv_normalized = np.maximum(observed_rv / levels[codes] - macro_var[codes] / levels[codes], 0.02)
    lambdas = np.linspace(0, 0.98, 50)
    losses = np.zeros((len(lambdas), 4))
    h = np.ones((len(lambdas), 4))
    for t in range(len(x)):
        if t >= 12:
            losses += np.log(h) + x[t] ** 2 / h
        h = lambdas[:, None] * h + (1 - lambdas[:, None]) * rv_normalized[t]
    best = losses.argmin(axis=0)
    initial = h[best, np.arange(4)]
    # Forecast RV EWMA's conditional variance as constant over the horizon.
    sample_specs["rv_ewma_control"] = (
        np.sqrt(initial) * gaussian,
        initial,
        [{"lambdas": lambdas[best].tolist(), "next_variance_ratio": initial.tolist()}],
    )
    rows = []
    for name, (normalized, expected, fits) in sample_specs.items():
        samples = shared_mean + vols * normalized
        variance = baseline_var + (probabilities @ levels) * (expected - 1)
        qlike = np.log(variance) + (actual[0] - mean) ** 2 / variance
        row = dict(
            model=name,
            date=str(returns.index[p].date()),
            training_months=p,
            **score_forecast(samples, actual, mean, model.assets),
            asset_fits=fits,
            baseline_replay_max_error=audit_error,
        )
        row["qlike_assets_mc"] = row["qlike_assets"]
        row["qlike_assets"] = float(qlike.mean())
        for j, ticker in enumerate(model.assets):
            row[f"qlike_{ticker}"] = float(qlike[j])
            row[f"analytic_variance_{ticker}"] = float(variance[j])
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="research/sv_extensions_2026-10-04/main")
    parser.add_argument("--baseline", default="research/independent_volatility_models_2026-10-04")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--grid-points", type=int, default=81)
    parser.add_argument("--quadrature", type=int, default=15)
    args = parser.parse_args()
    baseline = ROOT / args.baseline
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    bm = json.loads((baseline / "manifest.json").read_text())
    data = (ROOT / "research/model_quality_2025/data.pkl").read_bytes()
    assert hashlib.sha256(data).hexdigest() == bm["data_sha256"]
    assert bm["settings"] == SETTINGS
    for path, digest in bm["source_hashes"].items():
        if path.startswith("src/"):
            assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    macro, returns = pickle.loads(data)
    returns = returns.loc["2006-03-01":].dropna()
    rv_path = ROOT / "research/sv_extensions_2026-10-04/monthly_realized_variance.csv"
    rv = pd.read_csv(rv_path, index_col=0, parse_dates=True).reindex(columns=returns.columns)
    rows = json.loads((baseline / "forecasts.json").read_text())
    garch = {r["date"]: r for r in rows if r["model"] == "garch"}
    positions = list(range(60, len(returns)))[:: args.stride]
    if args.limit:
        positions = positions[: args.limit]
    manifest = dict(
        origins=len(positions),
        dates=[str(returns.index[p].date()) for p in positions],
        seed=bm["seed"],
        paths=bm["paths"],
        data_sha256=bm["data_sha256"],
        rv_sha256=hashlib.sha256(rv_path.read_bytes()).hexdigest(),
        baseline=str(baseline),
        grid_points=args.grid_points,
        quadrature=args.quadrature,
        settings=SETTINGS,
        scope="research prototypes: Gaussian SV-leverage, one-factor Gaussian return SV, monthly realized SV, monthly full-truncation Heston variance approximation, fBm rough log-SV QML",
        macro_vintage=bm["macro_vintage"],
        source_hashes={
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in list((ROOT / "src/mc_quadrants").glob("*.py"))
            + [
                Path(__file__),
                ROOT / "scripts/sv_extension_models.py",
                ROOT / "scripts/sv_dynamic_filter.cpp",
                ROOT / "scripts/stochastic_volatility_research.py",
            ]
        },
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    native_filter()
    started = time.perf_counter()
    results = []
    tasks = [
        (
            p,
            macro,
            returns,
            rv,
            garch[str(returns.index[p].date())],
            bm["paths"],
            bm["seed"],
            args.grid_points,
            args.quadrature,
        )
        for p in positions
    ]
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as executor:
        for n, batch in enumerate(executor.map(run_origin, tasks), 1):
            results.extend(batch)
            (out / "forecasts.json").write_text(json.dumps(results, indent=2, allow_nan=False))
            print(f"{n}/{len(positions)} origins; {time.perf_counter() - started:.1f}s", flush=True)
    manifest["elapsed_seconds"] = time.perf_counter() - started
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    pd.DataFrame(results).to_csv(out / "forecasts.csv", index=False)
    selected = set(manifest["dates"])
    combined = [r for r in rows if r["date"] in selected] + results
    (out / "summary.json").write_text(json.dumps(summarize(combined), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
