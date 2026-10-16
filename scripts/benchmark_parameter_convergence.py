#!/usr/bin/env python3
"""Separate outer parameter-bank convergence from conditional path noise."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mc_quadrants.calibration import calibrate_quadrant_model  # noqa: E402
from mc_quadrants.simulation import simulate_returns  # noqa: E402
from mc_quadrants.uncertainty import anchor_parameter_models, bootstrap_quadrant_models  # noqa: E402
from scripts.investigate_allocation_realism import FIT, portfolio_statistics  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, default=128)
    parser.add_argument("--paths-per-model", type=int, default=512)
    parser.add_argument("--origins", nargs="+", default=["2011-02-28", "2025-12-31"])
    parser.add_argument("--bank-seeds", nargs="+", type=int, default=[101, 202, 303])
    parser.add_argument("--output", type=Path, default=ROOT / "research/parameter_convergence_2026-10-12")
    parser.add_argument("--economic", choices=("anchors", "both"))
    parser.add_argument("--macro-stress", choices=("none", "persistent", "volatility", "combined"), default="none")
    parser.add_argument("--reality-fixes", action="store_true", help="Use endpoint yields and monthly CPI inputs.")
    args = parser.parse_args()
    if args.macro_stress != "none" and args.economic != "both":
        parser.error("--macro-stress requires --economic both")
    args.output.mkdir(parents=True, exist_ok=True)
    with (ROOT / "research/model_quality_2025/data.pkl").open("rb") as handle:
        macro, returns = pickle.load(handle)
    returns = returns.loc["2006-03-01":, ["SPY", "IEF"]]
    settings = {**FIT, "structural_returns": True}
    if args.economic:
        yields = pd.read_csv(ROOT / "research/economic_returns_2026-10-12/GS7.csv", parse_dates=["Date"]).set_index("Date").GS7.astype(float)
        yields.index += pd.offsets.MonthEnd(0)
        macro["bond_yield"] = yields.reindex(macro.index)
        settings["economic_returns"] = {"rate_stress": args.economic == "both", "macro_stress": args.macro_stress}
    if args.reality_fixes:
        from scripts.benchmark_reality_fixes import load_data
        macro, returns, sources = load_data()
        macro["bond_yield"] = sources["DGS7"].reindex(macro.index)
    rows = []
    for cutoff in args.origins:
        training, training_macro = returns.loc[:cutoff], macro.loc[:cutoff]
        reference = calibrate_quadrant_model(training, training_macro, **settings)
        for bank_seed in args.bank_seeds:
            models = anchor_parameter_models(
                bootstrap_quadrant_models(
                    training,
                    training_macro,
                    draws=args.draws,
                    block_size=12,
                    random_seed=bank_seed,
                    **settings,
                ),
                reference,
            )
            print(f"Fitted {cutoff}, bank {bank_seed}, {len(models)} models", flush=True)
            for path_seed in (1001, 2002):
                metrics = {
                    f"{weight}_{metric}": []
                    for weight in (0.0, 0.6, 1.0)
                    for metric in ("cagr", "max_drawdown")
                }
                for draw, model in enumerate(models):
                    fitted = model.metadata["dependence_fit"]
                    coefficients = {
                        k: float(fitted[k])
                        for k in ("garch_alpha", "garch_beta", "dcc_alpha", "dcc_beta", "dcc_asymmetry")
                    }
                    result = simulate_returns(
                        model,
                        periods=120,
                        paths=args.paths_per_model,
                        random_seed=path_seed + draw * 100003,
                        duration_model="semi_markov",
                        garch=True,
                        dynamic_correlation=True,
                        joint_macro=True,
                        macro_parameter_uncertainty=False,
                        **coefficients,
                    )
                    for weight in (0.0, 0.6, 1.0):
                        stats = portfolio_statistics(result.returns, weight)
                        for metric in ("cagr", "max_drawdown"):
                            metrics[f"{weight}_{metric}"].append(stats[metric])
                arrays = {key: np.asarray(value) for key, value in metrics.items()}
                np.savez_compressed(args.output / f"metrics_{cutoff}_{bank_seed}_{path_seed}.npz", **arrays)
                for count in (8, 16, 32, 64, 128, 256):
                    if count > args.draws:
                        continue
                    for weight in (0.0, 0.6, 1.0):
                        rows.append(
                            dict(
                                origin=cutoff,
                                bank_seed=bank_seed,
                                path_seed=path_seed,
                                models=count,
                                paths_per_model=args.paths_per_model,
                                stocks=weight,
                                median_cagr=float(np.median(arrays[f"{weight}_cagr"][:count])),
                                drawdown_p95=float(
                                    np.quantile(arrays[f"{weight}_max_drawdown"][:count], 0.95)
                                ),
                            )
                        )
                pd.DataFrame(rows).to_csv(args.output / "convergence.csv", index=False)
                print(f"Simulated {cutoff}, bank {bank_seed}, path seed {path_seed}", flush=True)
    (args.output / "manifest.json").write_text(
        json.dumps(
            {
                **vars(args),
                "output": str(args.output),
                "settings": settings,
                "path_seeds": [1001, 2002],
                "periods": 120,
                "note": "Equal paths per model; nested model-bank prefixes; inner seed changes keep bank fixed.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
