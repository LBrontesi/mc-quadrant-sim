#!/usr/bin/env python3
"""Summarize SV prototypes, paired forecast comparisons, and robustness checks."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.summarize_independent_volatility import paired_comparison  # noqa: E402

OUT = ROOT / "research/sv_extensions_2026-10-04"
NEW = ["leverage_sv", "factor_sv", "realized_sv", "heston_euler", "rough_sv"]
CONTROLS = ["gaussian_sv_control", "factor_constant_control", "rv_ewma_control"]
OLD = ["garch", "gjr", "egarch", "qarch", "figarch", "sv_mnts"]
TICKERS = ["SPY", "IEF", "GLD", "DBC"]
LABELS = dict(
    garch="GARCH",
    gjr="GJR-GARCH",
    egarch="EGARCH",
    qarch="QARCH",
    figarch="FIGARCH",
    sv_mnts="Basic SV–MNTS",
    leverage_sv="Leverage SV",
    factor_sv="One-factor SV",
    realized_sv="Realized SV (daily-close proxy)",
    heston_euler="Heston variance (monthly Euler)",
    rough_sv="Rough log-SV (fBm QML)",
    gaussian_sv_control="Gaussian SV control",
    factor_constant_control="Constant factor control",
    rv_ewma_control="RV-EWMA control",
)
METRICS = ["qlike_assets", "crps_1m", "crps_3m", "crps_12m", "drawdown_crps_12m", "loss_brier"]


def read_frame(directory):
    return pd.read_csv(directory / "forecasts.csv").set_index(["model", "date"]).sort_index()


def main():
    manifest = json.loads((OUT / "main/manifest.json").read_text())
    baseline = Path(manifest["baseline"])
    frame = read_frame(OUT / "main")
    basic = pd.read_csv(ROOT / "research/stochastic_volatility_2026-10-04/forecasts.csv")
    basic["model"] = "sv_mnts"
    all_scores = pd.concat([read_frame(baseline), basic.set_index(["model", "date"]), frame]).sort_index()
    assert frame.index.is_unique and len(frame) == 8 * manifest["origins"]
    for name in OLD + NEW + CONTROLS:
        assert all_scores.loc[name].index.tolist() == sorted(manifest["dates"])
    result = dict(manifest=manifest, scores={}, vs_garch={}, incremental={}, per_asset={}, checks={})
    for name in OLD + NEW + CONTROLS:
        result["scores"][name] = all_scores.loc[name, METRICS + ["coverage90", "breach05"]].mean().to_dict()
    for name in NEW:
        result["vs_garch"][name] = {
            metric: paired_comparison(all_scores.loc[name, metric], all_scores.loc["garch", metric], 5)
            for metric in METRICS
        }
        result["per_asset"][name] = {
            ticker: paired_comparison(
                all_scores.loc[name, f"qlike_{ticker}"], all_scores.loc["garch", f"qlike_{ticker}"], 20
            )
            for ticker in TICKERS
        }
    pairs = [(n, "gaussian_sv_control") for n in ["leverage_sv", "heston_euler", "rough_sv", "realized_sv"]]
    pairs += [("realized_sv", "rv_ewma_control"), ("factor_sv", "factor_constant_control")]
    for name, reference in pairs:
        result["incremental"][f"{name}_minus_{reference}"] = {
            metric: paired_comparison(
                all_scores.loc[name, metric], all_scores.loc[reference, metric], len(pairs)
            )
            for metric in METRICS
        }
    raw = json.loads((OUT / "main/forecasts.json").read_text())
    fits = [f for row in raw for f in row["asset_fits"]]
    result["checks"].update(
        max_garch_replay_error=max(r["baseline_replay_max_error"] for r in raw),
        optimizer_success=all(f.get("optimizer_success", True) for f in fits),
        max_posterior_edge_mass=max(f.get("maximum_posterior_edge_mass", 0) for f in fits),
    )
    result["checks"]["posterior_edge_mass_by_model"] = {
        name: max(
            f.get("maximum_posterior_edge_mass", 0)
            for row in raw
            if row["model"] == name
            for f in row["asset_fits"]
        )
        for name in NEW + CONTROLS
    }
    lookup = {(r["date"], r["model"]): r for r in raw}
    result["checks"]["leverage_nested_objective_max_excess"] = max(
        a["objective"] - b["objective"]
        for row in raw
        if row["model"] == "leverage_sv"
        for a, b in zip(row["asset_fits"], lookup[row["date"], "gaussian_sv_control"]["asset_fits"])
    )
    result["parameters"] = {}
    for name in NEW:
        rows = [r for r in raw if r["model"] == name]
        values = np.array([[f["parameters"] for f in r["asset_fits"]] for r in rows])
        result["parameters"][name] = dict(medians=np.median(values, axis=0).tolist())
        if name in ["leverage_sv", "heston_euler"]:
            result["parameters"][name]["rho_at_bound_fraction"] = np.mean(
                abs(values[:, :, 3]) > 0.8999, axis=0
            ).tolist()
        if name == "rough_sv":
            result["parameters"][name]["H_lower_bound_fraction"] = np.mean(
                values[:, :, 2] < 0.0301, axis=0
            ).tolist()
            result["parameters"][name]["scale_lower_bound_fraction"] = np.mean(
                values[:, :, 1] < 0.0051, axis=0
            ).tolist()
    for check in ["numerical-check", "seed-check"]:
        directory = OUT / check
        extra = read_frame(directory)
        other = json.loads((directory / "manifest.json").read_text())
        for key in ["data_sha256", "rv_sha256", "paths", "settings", "source_hashes"]:
            assert manifest[key] == other[key], (check, key)
        matching = frame.loc[extra.index]
        result["checks"][check] = {"origins": other["origins"], "models": {}}
        for name in NEW + CONTROLS:
            details = {}
            for metric in ["qlike_assets", "crps_1m", "crps_12m"]:
                difference = extra.loc[name, metric] - matching.loc[name, metric]
                details[metric] = dict(
                    mean_difference=float(difference.mean()),
                    max_absolute_difference=float(difference.abs().max()),
                    main_mean=float(matching.loc[name, metric].mean()),
                    check_mean=float(extra.loc[name, metric].mean()),
                )
            result["checks"][check]["models"][name] = details
        if check == "seed-check":
            assert extra.index.equals(frame.index)
            stable = extra.index.get_level_values("model") != "rough_sv"
            np.testing.assert_allclose(
                extra.loc[stable, "qlike_assets"], frame.loc[stable, "qlike_assets"], rtol=0, atol=1e-12
            )
            # Dense GP QML refits can differ slightly with floating-point BLAS execution.
            np.testing.assert_allclose(
                extra.loc["rough_sv", "qlike_assets"],
                frame.loc["rough_sv", "qlike_assets"],
                rtol=0,
                atol=1e-4,
            )
            seed_baseline = read_frame(Path(other["baseline"]))
            result["checks"][check]["vs_garch"] = {
                name: {
                    metric: float(extra.loc[name, metric].mean() - seed_baseline.loc["garch", metric].mean())
                    for metric in ["crps_1m", "crps_12m"]
                }
                for name in NEW
            }
    result["artifact_hashes"] = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [OUT / "main/forecasts.csv", baseline / "forecasts.csv", Path(__file__)]
    }
    (OUT / "comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    lines = [
        "# Five stochastic-volatility extensions: forecast evaluation",
        "",
        "All five requested families were evaluated as explicit research prototypes, alongside three controls, the five existing independently fitted volatility models, and basic SV–MNTS. These results assess the implementations below; they do not establish a ranking of entire model families.",
        "",
        "**Finding:** none of the five new prototypes establishes an improvement over GARCH in average volatility forecast loss. Realized SV has the best sample average, with a 0.51% improvement in one-month portfolio CRPS, but its volatility-loss confidence interval includes zero and twelve-month CRPS is 1.19% worse. Its per-ETF QLIKE improves on GARCH for SPY, GLD and DBC, but not IEF; no per-ETF improvement survives the adjusted intervals. The other four have worse average QLIKE. Heston's near-tie is provisional because numerical refinement materially changes individual scores. These results do not justify changing the engine default.",
        "",
        f"Frozen monthly sample: March 2006–December 2025. Expanding estimation begins with 60 months; {manifest['origins']} one-month forecast origins span March 2011–December 2025. Each origin uses 8,192 paths for 12 months, with 176 three-month and 167 twelve-month realized outcomes.",
        "",
        "## Main comparison",
        "",
        "Lower QLIKE and CRPS are better. QLIKE is averaged equally across ETFs; its differences, not percentage changes, are meaningful. Portfolio CRPS changes below are relative to independent GARCH.",
        "",
        "| Model | Mean QLIKE | Δ QLIKE vs GARCH | Adjusted 95% interval | 1m CRPS change | 12m CRPS change |",
        "|---|---:|---:|---|---:|---:|",
    ]
    for name in NEW:
        r = result["vs_garch"][name]
        q = r["qlike_assets"]
        lo, hi = q["family_ci95"]
        c1, c12 = r["crps_1m"], r["crps_12m"]
        lines.append(
            f"| {LABELS[name]} | {q['independent']:.5f} | {q['delta']:+.5f} | [{lo:+.5f}, {hi:+.5f}] | {100 * c1['delta'] / c1['reference']:+.2f}% | {100 * c12['delta'] / c12['reference']:+.2f}% |"
        )
    lines += [
        "",
        "Negative differences favor the new model. Intervals use paired circular 12-month blocks: 4,000 pointwise bootstrap replicates and 20,000 for Bonferroni adjustment across the five new models. Per-ETF comparisons adjust across 20 tests. Six-month block sensitivity, pointwise intervals and secondary metric intervals are in `comparison.json`. Adjustments are within each metric/family, not across the entire history of model research.",
        "",
        "## Per-ETF volatility forecasts",
        "",
        "| Model | SPY | IEF | GLD | DBC |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in OLD + NEW + CONTROLS:
        lines.append(
            "| "
            + LABELS[name]
            + " | "
            + " | ".join(f"{all_scores.loc[name, f'qlike_{t}'].mean():.5f}" for t in TICKERS)
            + " |"
        )
    lines += [
        "",
        "The lowest sample average is not necessarily a statistically established winner. Every univariate model is refitted independently for each ETF at every origin; factor SV instead fits the four ETFs jointly.",
        "",
        "## Incremental controls",
        "",
        "| Candidate minus control | Δ mean QLIKE | Adjusted 95% interval |",
        "|---|---:|---|",
    ]
    for name, reference in pairs:
        r = result["incremental"][f"{name}_minus_{reference}"]["qlike_assets"]
        lo, hi = r["family_ci95"]
        lines.append(
            f"| {LABELS[name]} minus {LABELS[reference]} | {r['delta']:+.5f} | [{lo:+.5f}, {hi:+.5f}] |"
        )
    lines += [
        "",
        "These six comparisons receive their own multiplicity adjustment. Gaussian SV controls for changing the return innovation law from MNTS to Gaussian. Constant factor controls for replacing the residual dependence structure. RV-EWMA helps separate the benefit of additional daily data from the realized-SV specification.",
        "",
        "## Portfolio forecast quality",
        "",
        "| Model | 1m CRPS | 3m CRPS | 12m CRPS | 90% coverage | Below 5th percentile |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in OLD + NEW + CONTROLS:
        r = result["scores"][name]
        lines.append(
            f"| {LABELS[name]} | {r['crps_1m']:.6f} | {r['crps_3m']:.6f} | {r['crps_12m']:.6f} | {r['coverage90']:.2%} | {r['breach05']:.2%} |"
        )
    lines += [
        "",
        "Portfolio weights: SPY 40%, IEF 30%, GLD 15%, DBC 15%, monthly rebalancing, without leverage, costs, taxes or cash flows. CRPS measures predictive-distribution error, not investment performance. Drawdown CRPS and loss-probability Brier scores are also saved.",
        "",
        "## What was fitted",
        "",
        "- **Leverage SV:** Gaussian returns with latent AR(1) log variance; correlation rho between the return shock and the next volatility shock. Negative rho lets negative returns increase subsequent volatility. This is a volatility leverage effect, separate from borrowing to lever a portfolio. Three-start approximate maximum likelihood; causal filtered states.",
        "- **Factor SV:** one Gaussian return factor with stochastic log variance and constant ETF-specific idiosyncratic variance. Static factor loadings and idiosyncratic variances are estimated first, then common volatility dynamics. The mean log variance enforces unit long-run factor variance. It replaces the MNTS/ADCC residual dependence model; the matching constant-factor control uses the same loadings and idiosyncratic variances.",
        "- **Realized SV:** Gaussian return likelihood combined with a noisy log realized-variance measurement. Daily adjusted closes provide monthly n/(n−1) × [sum(r²) − sum(r)²/n], requiring 15 daily returns. This is a demeaned daily-close proxy, not intraday realized variance. Measurement bias and noise are fitted. The measurement links regime-scaled residual variance plus estimated macro innovation variance to observed total-return RV; that mapping is approximate.",
        "- **Heston variance approximation:** a monthly full-truncation Euler square-root variance process, v_next=max(1e−8, v+kappa(theta−v)+xi sqrt(v) eta), with correlated Gaussian return/variance shocks. It is not an exact continuous-time Heston transition or integrated-variance return likelihood. A CIR stationary density initializes the filter, with 12 warmup observations.",
        "- **Rough log-SV approximation:** log variance follows fractional Brownian motion with H in [0.03,0.49]. Gaussian quasi-likelihood uses bias-corrected log squared residuals and fixed log-chi-square noise variance π²/2. Forecasts condition the Gaussian latent process jointly over the 12 future months. This is a nonstationary fBm log-SV proxy, not rough Heston, an exact return likelihood, or a high-frequency roughness estimate.",
        "- **Controls:** Gaussian AR(1) SV sets rho=0 in the same dynamic filter; RV-EWMA fits its decay on training data and holds forecast variance constant across future horizons.",
        "",
        "The model families draw on [Hosszejni and Kastner](https://www.jstatsoft.org/article/view/v100i12), [Heston](https://www.ma.imperial.ac.uk/~ajacquie/IC_Num_Methods/IC_Num_Methods_Docs/Literature/Heston.pdf), [Takahashi, Omori and Watanabe](https://www.sciencedirect.com/science/article/pii/S0167947308003824), and [Gatheral, Jaisson and Rosenbaum](https://arxiv.org/abs/1410.3394). The prototypes are not full replications of those papers.",
        "",
        "## Controls, diagnostics and limitations",
        "",
        "- Training windows end before the forecast month. Daily RV is sliced to the same training dates. Latent-state and future-shock uncertainty are simulated; fitted-parameter uncertainty is not.",
        "- Macro paths, conditional means and regime variance scales are shared. For univariate Gaussian variants, Gaussian shocks are recovered from the existing MNTS mixture draws. The benchmark ADCC correlation paths remain those driven by the original MNTS recursion; they are not refitted jointly with each Gaussian SV model. Factor variants use their own common-factor dependence and matched control draws.",
        "- Analytic first-month variance includes latent-state uncertainty and the lognormal correction, avoiding Monte Carlo noise in primary QLIKE.",
        f"- All selected fits report convergence: {result['checks']['optimizer_success']}. Largest posterior edge mass: {result['checks']['max_posterior_edge_mass']:.3g}. Maximum saved-GARCH replay error: {result['checks']['max_garch_replay_error']:.3g}. Maximum leverage-model training objective excess over its nested rho=0 control: {result['checks']['leverage_nested_objective_max_excess']:.3g}. Convergence is local, not a guarantee of the global optimum.",
        "",
        "| Model | SPY rho at bound | IEF rho at bound | GLD rho at bound | DBC rho at bound |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ["leverage_sv", "heston_euler"]:
        lines.append(
            "| "
            + LABELS[name]
            + " | "
            + " | ".join(f"{x:.1%}" for x in result["parameters"][name]["rho_at_bound_fraction"])
            + " |"
        )
    lines += [
        "",
        "Boundary fits indicate weak identification or constrained optima; parameter medians and rough-process boundary rates are saved in `comparison.json`.",
        "",
        "The 0.997 maximum edge probability belongs to Heston: the monthly truncated Euler process has an artificial atom at the variance floor, which can dominate the posterior for nearly zero normalized residuals. This is a material approximation limitation, not evidence of stable estimation. The other filtered models have maximum edge probability below 3e−8. Rough-process H estimates reach the lower bound in 72–98% of origins across ETFs, and scale reaches its lower bound in 49–72%; these data weakly identify rough dynamics.",
        "",
        "| Numerical refinement | Mean Δ QLIKE | Max origin Δ QLIKE (absolute) |",
        "|---|---:|---:|",
    ]
    for name in NEW + CONTROLS:
        r = result["checks"]["numerical-check"]["models"][name]["qlike_assets"]
        lines.append(f"| {LABELS[name]} | {r['mean_difference']:+.6f} | {r['max_absolute_difference']:.6f} |")
    lines += [
        "",
        "Numerical refinement evaluates every sixth origin (30 origins): state grids increase from 81 to 121 points and dynamic transition Gauss–Hermite quadrature from 15 to 25 nodes. Rough QML and RV-EWMA do not use those grids. Differences combine integration and local optimization sensitivity. This is a sensitivity check, not numerical-convergence proof.",
        "",
        "Heston's mean QLIKE shifts by −0.00670 on the refinement subset, with a maximum origin shift of 0.32227. This is material relative to its full-sample +0.00591 gap to GARCH, although the subset and full sample differ. Its small ranking differences are therefore unresolved. A better transition/integrated-return estimator is needed before drawing a Heston conclusion. Leverage, factor and realized SV show much smaller refinement shifts.",
        "",
        "| Second simulation seed | 1m CRPS Δ vs GARCH | 12m CRPS Δ vs GARCH |",
        "|---|---:|---:|",
    ]
    for name in NEW:
        r = result["checks"]["seed-check"]["vs_garch"][name]
        lines.append(f"| {LABELS[name]} | {r['crps_1m']:+.7f} | {r['crps_12m']:+.7f} |")
    lines += [
        "",
        "All 178 origins were repeated with a second simulation seed. Analytic QLIKE agrees to 1e−12 except for tiny rough-QML refit differences, measured in `comparison.json`; the dense numerical optimizer is sensitive to floating-point execution. Simulation seeds affect the distribution scores.",
        "",
        "Latest-revised macro data are shifted for release availability, not true historical vintages. The same short monthly sample has supported several rounds of model research, so it is not a fresh holdout. Daily-close RV adds information unavailable to the monthly-only models. Weak identification, local optimization and monthly Heston/rough approximations limit family-wide conclusions. Production engine defaults and UI options are unchanged.",
        "",
        "## Reproduction",
        "",
        "Requires the optional research dependencies, a C++17 compiler, and the saved frozen input/baseline artifacts.",
        "",
        "```bash",
        "python scripts/prepare_sv_realized_data.py",
        "python scripts/compare_sv_extensions.py --workers 4",
        "python scripts/compare_sv_extensions.py --workers 2 --stride 6 --grid-points 121 --quadrature 25 --output research/sv_extensions_2026-10-04/numerical-check",
        "python scripts/compare_sv_extensions.py --workers 4 --baseline research/independent_volatility_models_2026-10-04/seed-check --output research/sv_extensions_2026-10-04/seed-check",
        "python scripts/summarize_sv_extensions.py",
        "```",
        "",
        "Artifacts include per-origin fits and scores, checksums, manifests, comparison intervals, the daily price snapshot and reproducible RV proxy. `main/summary.json` excludes the older basic SV benchmark; the combined 14-model analysis is in `comparison.json` and this report.",
        "",
    ]
    (OUT / "report.md").write_text("\n".join(lines))
    print(json.dumps({name: result["vs_garch"][name]["qlike_assets"] for name in NEW}, indent=2))
    print(OUT / "report.md")


if __name__ == "__main__":
    main()
