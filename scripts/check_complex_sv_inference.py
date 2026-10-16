#!/usr/bin/env python3
"""Extend the five worst factor-MCMC diagnostics without selecting on forecast losses."""

from __future__ import annotations

import hashlib
import json
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.compare_complex_sv import prepare_context, score_model  # noqa: E402
from scripts.complex_sv_research import fit_rich_factor, simulate_rich_factor, split_rhat  # noqa: E402


def check(task):
    p, macro, returns, rv, baseline, original, manifest = task
    context = prepare_context((p, macro, returns, rv, baseline, manifest["paths"], manifest["seed"], 0, 0))
    fit = fit_rich_factor(context["x"], mcmc_scale=8)
    samples = simulate_rich_factor(fit, manifest["paths"], 12, manifest["seed"] + p * 101 + 24101)
    row = score_model(
        "rich_factor_long_chain", samples, np.array(fit["marginal_next_variance"]), [fit], context, p
    )
    posterior = np.array(fit["posterior_draws"]).reshape(2, fit["draws_per_chain"], 6, 4)
    h, mu, phi, tau = [posterior[..., k] for k in range(4)]
    component = np.exp(mu + phi * (h - mu) + tau * tau / 2)
    b = np.array(fit["loadings"])
    marginal = component[..., :2] @ (b * b).T + component[..., 2:]
    row["maximum_marginal_prediction_rhat"] = float(split_rhat(marginal).max())
    row["original_maximum_parameter_rhat"] = original["asset_fits"][0]["maximum_split_rhat"]
    row["relative_variance_changes"] = (
        np.array(fit["marginal_next_variance"])
        / np.array(original["asset_fits"][0]["marginal_next_variance"])
        - 1
    ).tolist()
    row["qlike_change"] = row["qlike_assets"] - original["qlike_assets"]
    return row


def main():
    directory = ROOT / "research/complex_sv_2026-10-05"
    manifest = json.loads((directory / "main/manifest.json").read_text())
    for path, digest in manifest["source_hashes"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest, path
    rows = json.loads((directory / "main/forecasts.json").read_text())
    candidates = sorted(
        [r for r in rows if r["model"] == "rich_factor"],
        key=lambda r: -r["asset_fits"][0]["maximum_split_rhat"],
    )[:5]
    macro, returns = pickle.loads((ROOT / "research/model_quality_2025/data.pkl").read_bytes())
    returns = returns.loc["2006-03-01":].dropna()
    rv = pd.read_csv(
        ROOT / "research/sv_extensions_2026-10-04/monthly_realized_variance.csv",
        index_col=0,
        parse_dates=True,
    ).reindex(columns=returns.columns)
    baseline = json.loads((Path(manifest["baseline"]) / "forecasts.json").read_text())
    garch = {r["date"]: r for r in baseline if r["model"] == "garch"}
    tasks = [(r["training_months"], macro, returns, rv, garch[r["date"]], r, manifest) for r in candidates]
    output = directory / "factor-convergence-check"
    output.mkdir(parents=True, exist_ok=True)
    results = []
    with ProcessPoolExecutor(max_workers=2, mp_context=get_context("spawn")) as executor:
        for row in executor.map(check, tasks):
            (output / f"{row['date']}.json").write_text(json.dumps(row, indent=2, allow_nan=False))
            compact = {k: v for k, v in row.items() if k != "asset_fits"}
            fit = row["asset_fits"][0]
            compact.update(
                maximum_parameter_split_rhat=fit["maximum_split_rhat"],
                burn=fit["burn"],
                draws_per_chain=fit["draws_per_chain"],
                thin=fit["thin"],
            )
            results.append(compact)
            print(row["date"], fit["maximum_split_rhat"], row["relative_variance_changes"], flush=True)
    (output / "summary.json").write_text(
        json.dumps(
            dict(
                selection="five highest primary parameter split-Rhat values; no forecast-loss selection",
                results=results,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                main_source_hashes=manifest["source_hashes"],
            ),
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
