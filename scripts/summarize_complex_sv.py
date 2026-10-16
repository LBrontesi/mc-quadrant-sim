#!/usr/bin/env python3
"""Compare six richer SV models and report estimation/numerical limitations."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.complex_sv_research import split_rhat  # noqa: E402
from scripts.summarize_independent_volatility import paired_comparison  # noqa: E402

OUT = ROOT / "research/complex_sv_2026-10-05"
NEW = ["realized_skewt", "two_component", "joint_jumps", "switching", "rich_factor", "lifted_heston"]
TICKERS = ["SPY", "IEF", "GLD", "DBC"]
LABELS = dict(
    realized_skewt="Realized skew-t leverage SV",
    two_component="Fast/slow two-component SV",
    joint_jumps="Joint return/volatility jump SV",
    switching="Two-regime SV",
    rich_factor="Two-factor Student-t SV",
    lifted_heston="Three-factor lifted Heston",
    two_factor_t_control="Static two-factor t control",
    garch="GARCH",
    gjr="GJR-GARCH",
    egarch="EGARCH",
    qarch="QARCH",
    figarch="FIGARCH",
    sv_mnts="Basic SV–MNTS",
    gaussian_sv_control="Gaussian SV",
    leverage_sv="Leverage SV",
    factor_sv="One-factor SV",
    realized_sv="Basic realized SV",
    heston_euler="Monthly Heston Euler",
    rough_sv="Rough log-SV QML",
    factor_constant_control="Static one-factor control",
    rv_ewma_control="RV-EWMA",
)


def frame(directory):
    data = pd.read_csv(directory / "forecasts.csv").set_index(["model", "date"]).sort_index()
    data["crps_assets"] = data[[f"crps_{t}" for t in TICKERS]].mean(axis=1)
    return data


def main():
    m = json.loads((OUT / "main/manifest.json").read_text())
    scores = frame(OUT / "main")
    old = frame(Path(m["baseline"]))
    prior = frame(ROOT / "research/sv_extensions_2026-10-04/main")
    basic = pd.read_csv(ROOT / "research/stochastic_volatility_2026-10-04/forecasts.csv")
    basic["model"] = "sv_mnts"
    basic = basic.set_index(["model", "date"])
    basic["crps_assets"] = basic[[f"crps_{t}" for t in TICKERS]].mean(axis=1)
    combined = pd.concat([old, prior, basic, scores]).sort_index()
    for name in combined.index.get_level_values(0).unique():
        assert combined.loc[name].index.tolist() == sorted(m["dates"])
    assert scores.index.is_unique and len(scores) == 7 * m["origins"]
    metrics = [
        "qlike_assets",
        "crps_assets",
        "crps_1m",
        "crps_3m",
        "crps_12m",
        "drawdown_crps_12m",
        "loss_brier",
    ]
    r = dict(manifest=m, vs_garch={}, incremental={}, per_asset={}, scores={}, diagnostics={}, checks={})
    for name in combined.index.get_level_values(0).unique():
        r["scores"][name] = combined.loc[name, metrics + ["coverage90", "breach05"]].mean().dropna().to_dict()
    for name in NEW:
        selected = metrics[:2] if name == "lifted_heston" else metrics
        r["vs_garch"][name] = {
            key: paired_comparison(scores.loc[name, key], old.loc["garch", key], 6) for key in selected
        }
        r["per_asset"][name] = {
            t: paired_comparison(scores.loc[name, f"qlike_{t}"], old.loc["garch", f"qlike_{t}"], 24)
            for t in TICKERS
        }
    pairs = [
        ("realized_skewt", "realized_sv"),
        ("two_component", "gaussian_sv_control"),
        ("joint_jumps", "gaussian_sv_control"),
        ("switching", "gaussian_sv_control"),
        ("rich_factor", "factor_sv"),
        ("rich_factor", "two_factor_t_control"),
        ("lifted_heston", "heston_euler"),
    ]
    for candidate, reference in pairs:
        r["incremental"][f"{candidate}_minus_{reference}"] = {
            key: paired_comparison(combined.loc[candidate, key], combined.loc[reference, key], 7)
            for key in metrics[:2]
        }
    raw = json.loads((OUT / "main/forecasts.json").read_text())
    r["diagnostics"]["maximum_baseline_replay_error"] = max(row["baseline_replay_max_error"] for row in raw)
    parameter_names = dict(
        realized_skewt=[
            "mu",
            "phi",
            "stationary_sd",
            "rho_copula",
            "nu",
            "skew_gamma",
            "rv_bias",
            "rv_noise",
        ],
        two_component=["mu", "phi_fast", "phi_slow", "sd_fast", "sd_slow"],
        joint_jumps=[
            "mu",
            "phi",
            "diffusion_stationary_sd",
            "jump_probability",
            "return_jump_mean",
            "return_jump_sd",
            "vol_jump_size",
        ],
        switching=["mu_low", "mean_gap", "phi_low", "phi_high", "sd_low", "sd_high", "stay_low", "stay_high"],
    )
    for name in NEW[:4]:
        fits = [fit for row in raw if row["model"] == name for fit in row["asset_fits"]]
        p = np.array([f["parameters"] for f in fits])
        bounds = np.array(fits[0]["parameter_bounds"])
        r["diagnostics"][name] = dict(
            maximum_edge_mass=max(f["maximum_posterior_edge_mass"] for f in fits),
            optimizer_success=all(f["optimizer_success"] for f in fits),
            minimum_converged_starts=min(f["converged_starts"] for f in fits),
            parameter_names=parameter_names[name],
            parameter_medians=np.median(p, axis=0).tolist(),
            bound_fractions=np.mean(
                (abs(p - bounds[:, 0]) < 1e-4) | (abs(p - bounds[:, 1]) < 1e-4), axis=0
            ).tolist(),
        )
    factors = [row["asset_fits"][0] for row in raw if row["model"] == "rich_factor"]
    marginal_rhats = []
    for fit in factors:
        samples = np.array(fit["posterior_draws"]).reshape(2, fit["draws_per_chain"], 6, 4)
        h, mu, phi, tau = [samples[..., k] for k in range(4)]
        v = np.exp(mu + phi * (h - mu) + tau * tau / 2)
        b = np.array(fit["loadings"])
        marginal = v[..., :2] @ (b * b).T + v[..., 2:]
        marginal_rhats.append(float(split_rhat(marginal).max()))
    r["diagnostics"]["rich_factor"] = dict(
        maximum_parameter_split_rhat=max(f["maximum_split_rhat"] for f in factors),
        fraction_parameter_rhat_above_11=float(np.mean([f["maximum_split_rhat"] > 1.1 for f in factors])),
        maximum_component_prediction_rhat=max(f["maximum_predictive_split_rhat"] for f in factors),
        maximum_marginal_prediction_rhat=max(marginal_rhats),
        fraction_marginal_prediction_rhat_above_11=float(np.mean(np.array(marginal_rhats) > 1.1)),
        maximum_slice_failures=max(info[1] for f in factors for info in f["chain_diagnostics"]),
        fraction_extended=float(np.mean([f["mcmc_extended"] for f in factors])),
    )
    lifted = [f for row in raw if row["model"] == "lifted_heston" for f in row["asset_fits"]]
    r["diagnostics"]["lifted_heston"] = dict(
        maximum_training_floor_fraction=max(f["training_floor_fraction"] for f in lifted),
        maximum_forecast_floor_fraction=max(f["forecast_floor_fraction"] for f in lifted),
        minimum_particle_ess_fraction=min(f["minimum_effective_particle_fraction"] for f in lifted),
        maximum_relative_variance_mc_se=max(
            f["variance_monte_carlo_se"] / f["next_variance_ratio"] for f in lifted
        ),
    )
    for check in ["numerical-check", "seed-check"]:
        directory = OUT / check
        cm = json.loads((directory / "manifest.json").read_text())
        for key in ["data_sha256", "rv_sha256", "paths", "settings", "source_hashes"]:
            assert cm[key] == m[key], (check, key)
        extra = frame(directory)
        matching = scores.loc[extra.index]
        r["checks"][check] = dict(origins=cm["origins"], models={})
        for name in NEW + ["two_factor_t_control"]:
            r["checks"][check]["models"][name] = {}
            for metric in metrics[:2] + ([] if name == "lifted_heston" else ["crps_1m", "crps_12m"]):
                delta = extra.loc[name, metric] - matching.loc[name, metric]
                r["checks"][check]["models"][name][metric] = dict(
                    mean_difference=float(delta.mean()),
                    maximum_absolute_difference=float(delta.abs().max()),
                    main_mean=float(matching.loc[name, metric].mean()),
                    check_mean=float(extra.loc[name, metric].mean()),
                )
        if check == "seed-check":
            assert extra.index.equals(scores.index)
            np.testing.assert_allclose(extra.qlike_assets, scores.qlike_assets, rtol=0, atol=1e-12)
            second = frame(Path(cm["baseline"]))
            r["checks"][check]["vs_garch"] = {
                name: {
                    metric: float(extra.loc[name, metric].mean() - second.loc["garch", metric].mean())
                    for metric in metrics[:2] + ([] if name == "lifted_heston" else ["crps_1m", "crps_12m"])
                }
                for name in NEW
            }
        else:
            r["checks"][check]["vs_garch_on_matching_dates"] = {
                name: dict(
                    primary_difference=float(
                        (
                            matching.loc[name, "qlike_assets"]
                            - old.loc["garch", "qlike_assets"].loc[matching.loc[name].index]
                        ).mean()
                    ),
                    refined_difference=float(
                        (
                            extra.loc[name, "qlike_assets"]
                            - old.loc["garch", "qlike_assets"].loc[extra.loc[name].index]
                        ).mean()
                    ),
                )
                for name in NEW
            }
    convergence = json.loads((OUT / "factor-convergence-check/summary.json").read_text())
    assert convergence["main_source_hashes"] == m["source_hashes"]
    r["checks"]["factor-convergence-check"] = convergence
    r["artifact_hashes"] = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [
            Path(__file__),
            OUT / "main/forecasts.csv",
            OUT / "seed-check/forecasts.csv",
            OUT / "numerical-check/forecasts.csv",
        ]
    }
    (OUT / "comparison.json").write_text(json.dumps(r, indent=2, allow_nan=False))
    marginal = r["vs_garch"]["realized_skewt"]["crps_assets"]
    portfolio = r["vs_garch"]["realized_skewt"]["crps_1m"]
    variance_winners = [name for name in NEW if r["vs_garch"][name]["qlike_assets"]["family_ci95"][1] < 0]
    lines = [
        "# Richer stochastic-volatility models: six-model evaluation",
        "",
        "Six research specifications were fitted and evaluated against the saved independent GARCH family and earlier SV benchmarks. All 178 monthly forecast origins (March 2011–December 2025) use expanding training windows, four ETFs and 8,192 simulated paths. This isolates richer monthly specifications; it is not a daily-frequency calibration experiment. Frozen daily adjusted closes supply the monthly realized-variance proxy.",
        "",
        f"**Outcome:** {len(variance_winners)} of the six candidates establish lower average volatility forecast loss than GARCH under the adjusted intervals. Realized skew-t SV has the lowest sample-average volatility loss among the new candidates, but basic realized SV from the previous experiment remains slightly better on that metric. The additional complexity does not justify replacing the engine default.",
        "",
        f"There is a narrower positive result: realized skew-t SV improves average one-month ETF return-distribution CRPS by {-100 * marginal['delta'] / marginal['reference']:.2f}% relative to GARCH. Its six-model-adjusted interval for the CRPS change is [{marginal['family_ci95'][0]:+.7f}, {marginal['family_ci95'][1]:+.7f}], which excludes zero. The corresponding portfolio improvement of {-100 * portfolio['delta'] / portfolio['reference']:.2f}% is not conclusive after adjustment: [{portfolio['family_ci95'][0]:+.7f}, {portfolio['family_ci95'][1]:+.7f}]. This is evidence about distribution shape in this experiment, not an established variance-forecast improvement or a fresh-holdout result.",
        "",
        "Interpret jump, switching-regime and lifted-Heston near-ties cautiously because numerical refinement materially changes some scores. The richer factor model also has widespread parameter-chain convergence failures; its results are exploratory. The realized skew-t and two-component specifications show smaller numerical shifts.",
        "",
        "## Main comparison",
        "",
        "Negative changes favor the candidate. QLIKE is average one-month variance forecast loss across SPY, IEF, GLD and DBC; percentage changes are inappropriate for this loss. CRPS evaluates a predictive distribution, not investment returns. Marginal CRPS averages the four ETF scores.",
        "",
        "| Model | Δ QLIKE vs GARCH | Adjusted 95% interval | Marginal CRPS change | Portfolio 1m CRPS change | Portfolio 12m CRPS change |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for name in NEW:
        q = r["vs_garch"][name]["qlike_assets"]
        lo, hi = q["family_ci95"]
        c = r["vs_garch"][name]["crps_assets"]
        changes = []
        for metric in ["crps_1m", "crps_12m"]:
            v = r["vs_garch"][name].get(metric)
            changes.append(f"{100 * v['delta'] / v['reference']:+.2f}%" if v else "Not evaluated")
        lines.append(
            f"| {LABELS[name]} | {q['delta']:+.5f} | [{lo:+.5f}, {hi:+.5f}] | {100 * c['delta'] / c['reference']:+.2f}% | "
            + " | ".join(changes)
            + " |"
        )
    lines += [
        "",
        "Paired circular block bootstrap: 12-month blocks, 4,000 pointwise replicates, 20,000 Bonferroni-adjusted replicates across six new candidates for each metric. ETF-specific comparisons adjust across 24 tests. Six-month block sensitivity is saved. These adjustments do not cover all previous rounds of model development or all metrics jointly. The same dataset has been inspected repeatedly, so this is research evaluation rather than a fresh confirmatory holdout.",
        "",
        "## Per-ETF QLIKE",
        "",
        "| Model | SPY | IEF | GLD | DBC |",
        "|---|---:|---:|---:|---:|",
    ]
    display = (
        [
            "garch",
            "gjr",
            "egarch",
            "qarch",
            "figarch",
            "sv_mnts",
            "realized_sv",
            "gaussian_sv_control",
            "factor_sv",
            "heston_euler",
        ]
        + NEW
        + ["two_factor_t_control"]
    )
    for name in display:
        lines.append(
            "| "
            + LABELS[name]
            + " | "
            + " | ".join(f"{combined.loc[name, f'qlike_{t}'].mean():.5f}" for t in TICKERS)
            + " |"
        )
    lines += [
        "",
        "## Incremental comparisons",
        "",
        "| Candidate minus reference | Δ QLIKE | Adjusted 95% interval |",
        "|---|---:|---|",
    ]
    for candidate, reference in pairs:
        q = r["incremental"][f"{candidate}_minus_{reference}"]["qlike_assets"]
        lo, hi = q["family_ci95"]
        lines.append(
            f"| {LABELS[candidate]} minus {LABELS[reference]} | {q['delta']:+.5f} | [{lo:+.5f}, {hi:+.5f}] |"
        )
    lines += [
        "",
        "These seven comparisons receive their own family adjustment. The richer factor model changes both the dependence model and inference method; its static two-factor Student-t control shares fitted loadings and tail thickness. Lifted Heston differs from the earlier Heston experiment in time discretization, latent dynamics and integrated-return likelihood.",
        "",
        "## What was implemented",
        "",
        "1. **Realized skew-t leverage SV:** latent AR(1) log variance; standardized Fernandez–Steel two-piece Student-t returns; a Gaussian copula correlates the Gaussianized return innovation with next-period log-volatility noise. This is copula leverage, not a linear correlation between a Student-t return and a Gaussian shock. A noisy log-RV measurement includes estimated macro innovation variance, measurement bias and measurement noise. Eight parameters are jointly estimated per ETF. Daily-close RV remains a proxy, not intraday realized variance.",
        "2. **Two-component SV:** log variance is the sum of a fast and a slow Gaussian AR(1) component. Persistence ranges are [0,0.75] and [0.8,0.99], respectively, to identify their roles. A joint two-dimensional filter preserves the posterior dependence induced by observed returns; forecasts evolve both components continuously.",
        "3. **Joint jump SV:** monthly Gaussian log-SV plus a centered Bernoulli return jump with normally distributed size. The same indicator raises next-month log variance. Jump probability, return-jump mean/SD and volatility-jump size are fitted. A joint volatility/jump filter integrates over unobserved jumps. This is a discrete monthly SVCJ-style model, not an exact continuous-time jump-diffusion likelihood. Initialization uses a Gaussian approximation with the jump-adjusted stationary log-variance variance, followed by 12 warmup months.",
        "4. **Regime-switching SV:** two latent volatility regimes have different log-variance means, persistence and scale, with fitted transition probabilities. A standardized latent volatility state persists through regime changes. These volatility regimes are additional to the engine’s macro regimes; they are not identified with them.",
        "5. **Richer factor SV:** two common return factors and four asset-specific components each have their own latent AR(1) volatility. Standardized Student-t innovations have fixed eight degrees of freedom. Loadings and initial idiosyncratic levels are estimated by Gaussian covariance ML and then held fixed. Joint latent inference uses Gaussian-mixture Gibbs updates and elliptical slice sampling. Volatility parameters are sampled too: log-variance means have N(initial level,1.5²) priors, innovation variances IG(2,0.08), and persistence a Beta(8,2) prior truncated to [0.01,0.98]. Forecasts include posterior parameter uncertainty. This model does not add cross-asset leverage, stochastic loadings or uncertainty in loadings; it is not a full replication of the richer factor-SV paper.",
        "6. **Lifted Heston:** three exponential-kernel factors, driven by one common Brownian motion per ETF, approximate a mean-reverting square-root Volterra variance process. Mean-reversion rates are fixed at [0.05,0.5,5] per month, with normalized weights proportional to rate^(0.5−H). Returns integrate variance and correlated shocks over 21 substeps per month. A bootstrap particle filter estimates latent states. Training chooses among 24 predeclared combinations of kappa in {0.1,0.5}, vol-of-vol in {0.15,0.35}, rho in {−0.7,0}, H in {0.1,0.3,0.49}; theta is estimated from training second moments. The top three training candidates are reevaluated with four times as many particles. This is a coarse finite-grid, projected-Euler research approximation, not exact rough Heston or a published-scheme replication.",
        "",
        "References: [realized skew-t SV](https://arxiv.org/abs/2401.13179), [Fernandez–Steel skew-t](https://www.tandfonline.com/doi/abs/10.1080/01621459.1998.10474117), [SV and factor-SV inference](https://www.jstatsoft.org/article/view/v100i12), [elliptical slice sampling](https://proceedings.mlr.press/v9/murray10a.html), [jumps in returns and volatility](https://onlinelibrary.wiley.com/doi/abs/10.1111/1540-6261.00566), [regime-switching SV](https://researchportal.hkust.edu.hk/en/publications/a-stochastic-volatility-model-with-markov-switching/), [lifted Heston](https://arxiv.org/abs/1810.04868).",
        "",
        "## Estimation diagnostics",
        "",
        "| Model | Maximum posterior boundary mass | Minimum converged optimization starts |",
        "|---|---:|---:|",
    ]
    for name in NEW[:4]:
        d = r["diagnostics"][name]
        lines.append(f"| {LABELS[name]} | {d['maximum_edge_mass']:.3g} | {d['minimum_converged_starts']}/2 |")
    f = r["diagnostics"]["rich_factor"]
    h = r["diagnostics"]["lifted_heston"]
    lines += [
        "",
        f"Factor SV uses two MCMC chains. The initial schedule is 1,200 burn-in iterations followed by 200 retained draws per chain, thinning every five iterations. If any monitored split-Rhat exceeds 1.1, both chains rerun with 2,500 burn-in iterations and 400 draws, thinning every ten iterations. This extension was triggered for {f['fraction_extended']:.1%} of origins.",
        "",
        f"After extension, the maximum parameter split-Rhat is {f['maximum_parameter_split_rhat']:.3f}; {f['fraction_parameter_rhat_above_11']:.1%} of origins still exceed 1.1. Maximum split-Rhat for ETF predictive variance is {f['maximum_marginal_prediction_rhat']:.3f}, with {f['fraction_marginal_prediction_rhat_above_11']:.1%} of origins exceeding 1.1. Slice-update failures: {f['maximum_slice_failures']:.0f}. These are basic split-Rhat checks, not rank-normalized Rhat or effective-sample-size certification. Nonconverged factor results must be treated as exploratory; a forecast score cannot cure an inference failure.",
        "",
        "The five worst primary parameter-Rhat origins receive a separate convergence check with eight times the original burn-in/retained-draw schedules. Selection uses convergence diagnostics alone, not forecast losses. These checks do not replace the primary scores or establish convergence at every origin.",
        "",
        "| Forecast date | Original parameter Rhat | Longer-chain parameter Rhat | Longer-chain predictive Rhat | Largest ETF variance change |",
        "|---|---:|---:|---:|---:|",
        *[
            f"| {row['date']} | {row['original_maximum_parameter_rhat']:.3f} | {row['maximum_parameter_split_rhat']:.3f} | {row['maximum_marginal_prediction_rhat']:.3f} | {max(abs(v) for v in row['relative_variance_changes']):.2%} |"
            for row in convergence["results"]
        ],
        "",
        f"Lifted Heston: maximum training floor frequency {h['maximum_training_floor_fraction']:.3%}; maximum first-month forecast floor frequency {h['maximum_forecast_floor_fraction']:.3%}. Minimum effective-particle fraction before resampling {h['minimum_particle_ess_fraction']:.3%}. Maximum estimated relative Monte Carlo SE of first-month variance {h['maximum_relative_variance_mc_se']:.3%}. The variance estimate averages 32,768 integrated-variance draws; it is not analytic. This SE excludes filtering, parameter selection and discretization error.",
        "",
        "## Numerical and simulation sensitivity",
        "",
        "| Model | Refinement mean Δ QLIKE | Maximum origin absolute Δ QLIKE |",
        "|---|---:|---:|",
    ]
    for name in NEW:
        d = r["checks"]["numerical-check"]["models"][name]["qlike_assets"]
        lines.append(
            f"| {LABELS[name]} | {d['mean_difference']:+.6f} | {d['maximum_absolute_difference']:.6f} |"
        )
    lines += [
        "",
        "The refinement run evaluates every twelfth origin (15 March forecasts, not a representative random sample): univariate grids increase 41→61, each two-component grid 21→33, lifted search particles 128→256 (finalists 512→1,024), substeps 21→42, and MCMC burn-in/retained draws double. It checks combined filtering, local optimization, particle selection and MCMC sensitivity. It does not prove convergence or establish rankings for untested discretizations.",
        "",
        "| Model | Second-seed marginal CRPS Δ vs GARCH | Second-seed portfolio 1m CRPS Δ vs GARCH |",
        "|---|---:|---:|",
    ]
    for name in NEW:
        d = r["checks"]["seed-check"]["vs_garch"][name]
        portfolio = f"{d['crps_1m']:+.7f}" if "crps_1m" in d else "Not evaluated"
        lines.append(f"| {LABELS[name]} | {d['crps_assets']:+.7f} | {portfolio} |")
    lines += [
        "",
        "The second-seed run retains all fitted parameters and filtered/posterior states and repeats forecast shocks only. One-month QLIKE is identical; lifted Heston also reuses its fixed-seed integrated-variance estimate. Particle and MCMC inference sensitivity is assessed separately in the refinement run.",
        "",
        "## Fairness and scope",
        "",
        "- Each training window ends before its forecast month; all realized-variance inputs are restricted to that window. The same frozen macro data, conditional means and regime variance scales are used. Saved GARCH forecasts are replayed at every origin; maximum replay difference is "
        + f"{r['diagnostics']['maximum_baseline_replay_error']:.3g}.",
        "- The first four specifications fit approximate marginal likelihoods with two optimization starts and 12-month filtering warmup, excluding that warmup from parameter likelihood. Factor covariance fitting excludes the same 12 months, while its Bayesian latent/volatility-parameter posterior uses the full training history. Local optimization success is not proof of a global optimum.",
        "- Gaussian/skew-t univariate variants retain the prior benchmark’s Gaussian return primitives and its MNTS-driven ADCC correlation paths. Jumps have independent ETF-specific indicators. Factor SV supplies its own joint dependence model and fixed-loading Student-t control. Thus portfolio changes can reflect dependence and tail assumptions as well as volatility dynamics.",
        "- Lifted Heston is fitted marginally and has no implemented joint cross-asset Brownian-driver model. Portfolio CRPS, coverage and drawdown results are intentionally omitted for it. Per-ETF variance and return-distribution scores remain defined.",
        "- Portfolio forecasts use weights 40% SPY, 30% IEF, 15% GLD, 15% DBC with monthly rebalancing and no leverage/costs/taxes/cash flows. There are 176 complete three-month and 167 complete twelve-month outcomes.",
        "- Portfolio CRPS is computed on the finite simulation ensemble. Student-t log-return innovations do not have finite exponential moments, so tail behavior would need a deliberate production specification before using these research laws for expected wealth or unbounded payoff calculations.",
        "- Latest-revised macro observations are lagged for availability, not historical release vintages. Monthly sample size is limited, RV uses daily closes, parameter boundaries and inference limitations matter. Results assess these particular specifications and estimators, not the theoretical limits of each family. Production defaults remain unchanged.",
        "",
        "## Reproduction",
        "",
        "```bash",
        "python scripts/compare_complex_sv.py --workers 4",
        "python scripts/compare_complex_sv.py --workers 2 --stride 12 --grid-points 61 --component-grid 33 --particles 256 --steps 42 --mcmc-scale 2 --output research/complex_sv_2026-10-05/numerical-check",
        "python scripts/replay_complex_sv.py --workers 4",
        "python scripts/check_complex_sv_inference.py",
        "python scripts/summarize_complex_sv.py",
        "```",
        "",
        "Requires optional research dependencies, a C++17 compiler, frozen data and saved baseline artifacts. Runs checkpoint each complete origin and resume only with matching configuration/source hashes. Outputs include per-origin fits, raw scores, manifests, bootstrap comparisons and this report.",
        "",
    ]
    (OUT / "report.md").write_text("\n".join(lines))
    print(json.dumps({name: r["vs_garch"][name]["qlike_assets"] for name in NEW}, indent=2))
    print(OUT / "report.md")


if __name__ == "__main__":
    main()
