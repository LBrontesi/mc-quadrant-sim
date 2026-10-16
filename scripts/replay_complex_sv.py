#!/usr/bin/env python3
"""Repeat forecast simulations with frozen complex-SV fits and a second seed."""

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
sys.path.insert(0, str(ROOT))
from scripts.compare_complex_sv import MODELS, prepare_context, score_model  # noqa: E402
from scripts.complex_sv_research import simulate_complex, simulate_lifted, simulate_rich_factor  # noqa: E402


def replay(task):
    source, target, p, macro, returns, rv, baseline, paths, seed = task
    path = Path(target) / "origins" / f"{returns.index[p].date()}.json"
    if path.exists():
        return json.loads(path.read_text())
    rows = json.loads((Path(source) / "origins" / f"{returns.index[p].date()}.json").read_text())
    context = prepare_context((p, macro, returns, rv, baseline, paths, seed, 0, 0))
    output = []
    for row in rows:
        name, fits = row["model"], row["asset_fits"]
        if name in MODELS[:4]:
            normalized = simulate_complex(fits, context["gaussian"], seed + p * 101 + 23101)
            variance = np.array([f["next_variance_ratio"] for f in fits])
        elif name in ["rich_factor", "two_factor_t_control"]:
            fit = fits[0]
            normalized = simulate_rich_factor(
                fit, paths, 12, seed + p * 101 + 24101, constant=name == "two_factor_t_control"
            )
            if name == "rich_factor":
                variance = np.array(fit["marginal_next_variance"])
            else:
                variance = (np.array(fit["loadings"]) ** 2).sum(axis=1) + np.array(
                    fit["static_idiosyncratic_variance"]
                )
        else:
            normalized = np.stack(
                [
                    simulate_lifted(fit, paths, 12, seed + p * 101 + 25101 + j)[0]
                    for j, fit in enumerate(fits)
                ],
                axis=2,
            )
            variance = np.array([f["next_variance_ratio"] for f in fits])
        output.append(score_model(name, normalized, variance, fits, context, p))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(output, indent=2, allow_nan=False))
    temporary.replace(path)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="research/complex_sv_2026-10-05/main")
    parser.add_argument("--output", default="research/complex_sv_2026-10-05/seed-check")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    source, output = ROOT / args.source, ROOT / args.output
    m = json.loads((source / "manifest.json").read_text())
    for path, digest in m["source_hashes"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest, path
    baseline = ROOT / "research/independent_volatility_models_2026-10-04/seed-check"
    bm = json.loads((baseline / "manifest.json").read_text())
    assert bm["data_sha256"] == m["data_sha256"] and bm["paths"] == m["paths"]
    macro, returns = pickle.loads((ROOT / "research/model_quality_2025/data.pkl").read_bytes())
    returns = returns.loc["2006-03-01":].dropna()
    rv = pd.read_csv(
        ROOT / "research/sv_extensions_2026-10-04/monthly_realized_variance.csv",
        index_col=0,
        parse_dates=True,
    ).reindex(columns=returns.columns)
    rows = json.loads((baseline / "forecasts.json").read_text())
    garch = {r["date"]: r for r in rows if r["model"] == "garch"}
    positions = [p for p in range(60, len(returns)) if str(returns.index[p].date()) in m["dates"]]
    for p in positions:
        assert (source / "origins" / f"{returns.index[p].date()}.json").exists()
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(
        m,
        baseline=str(baseline),
        seed=bm["seed"],
        fixed_fit_source=str(source),
        replay_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    manifest.pop("elapsed_seconds", None)
    target = output / "manifest.json"
    if target.exists():
        previous = json.loads(target.read_text())
        previous.pop("elapsed_seconds", None)
        assert previous == manifest
    target.write_text(json.dumps(manifest, indent=2))
    tasks = [
        (
            str(source),
            str(output),
            p,
            macro,
            returns,
            rv,
            garch[str(returns.index[p].date())],
            m["paths"],
            bm["seed"],
        )
        for p in positions
    ]
    results = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as executor:
        for n, batch in enumerate(executor.map(replay, tasks), 1):
            results.extend(batch)
            print(f"{n}/{len(tasks)} origins; {time.perf_counter() - started:.1f}s", flush=True)
    (output / "forecasts.json").write_text(json.dumps(results, indent=2, allow_nan=False))
    pd.DataFrame(results).to_csv(output / "forecasts.csv", index=False)
    manifest["elapsed_seconds"] = time.perf_counter() - started
    target.write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
