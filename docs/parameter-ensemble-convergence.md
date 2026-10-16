# Parameter-ensemble convergence: first implementation

12 October 2026. This implements the first priority from the allocation-realism investigation. Economic return anchors and persistent inflation/rate stress remain separate model changes; this work makes subsequent comparisons less sensitive to numerical sampling noise.

## Production changes

- Separate `parameter_seed` from the market-path `random_seed` in the pipeline, API, UI and chronological validation. Holding the parameter seed fixed reuses the same fitted bank when the path seed changes. Existing API calls without this option retain their previous seed behavior.
- Increase the new UI default from 8 to 128 fitted models, allow up to 512, and preserve existing saved choices. This increases cold calibration time. Changing only the market-path seed can reuse the cached parameter bank.
- Report outer-ensemble precision for the terminal-wealth median and maximum-drawdown P95. The diagnostic resamples **whole parameter models**, using their conditional empirical distributions with equal model weights. It does not treat all market paths as independent calibration draws.
- Use exact native drawdown arrays for compact reporting, rather than the smaller displayed path sample. Without native reductions, use the complete portfolio wealth paths.
- Show `insufficient_models`, `insufficient_paths`, `unstable`, or `within_tolerance`, plus Monte Carlo intervals, in metadata/API and the parameter-uncertainty card. Unresolved precision also appears in diagnostics warnings.

The defaults are a more stable starting point, not a claim that 128 is always enough. No return means, volatility parameters, correlations, or tail laws were changed.

## Measured path behavior

Two historical calibration dates, three independently sampled parameter banks per date, two independent market-path seeds for each bank, 128 models per bank and 512 ten-year paths per model. Smaller banks use nested prefixes of 8/16/32/64 models. Each run compares stocks, bonds and 60/40 on shared market paths. Structural links and joint macro dynamics are enabled; macro-parameter uncertainty is sampled through the outer bootstrap only, matching production behavior.

For 100% stocks, holding the market-path seed fixed at 1001:

| Calibration through | Parameter models | Median ten-year CAGR across three banks | Drawdown P95 across three banks |
|---|---:|---:|---:|
| February 2011 | 8 | −0.22% to 5.89% | 75.4% to 90.6% |
| February 2011 | 128 | 2.97% to 3.51% | 85.9% to 87.4% |
| December 2025 | 8 | 9.48% to 11.25% | 52.9% to 65.5% |
| December 2025 | 128 | 10.47% to 10.85% | 58.8% to 61.4% |

Thus the return spread fell from 6.11 to 0.54 percentage points for the earlier calibration and from 1.77 to 0.38 points for the full-history calibration. Improvement is not monotonic at every intermediate bank size: one small prefix can appear stable by chance.

Across the checked banks and all three allocations, changing only the inner path seed at 128 models moved median CAGR by at most approximately 0.10 percentage points at the earlier origin and 0.12 points at the later origin. The corresponding largest drawdown-P95 changes were 0.46 and 0.30 percentage points. Outer-bank sampling dominates several of the larger discrepancies.

These are numerical-convergence experiments, not out-of-sample validation. The underlying frozen revised data and its limitations are documented in [the allocation-realism investigation](allocation-realism-investigation.md). The enlarged bank does not solve low historical equity-return estimates or the 2022 bond-stress mismatch.

## Diagnostic interpretation

The diagnostic uses 256 multinomial resamples of the fitted model bank. A resampled mixture quantile is obtained by inverting its weighted empirical CDF, including probability masses such as zero drawdown. Path arrays are not expanded for every resample. Unequal path counts do not give one fitted model more statistical weight than another.

For each metric, the 2.5th and 97.5th percentiles across resampled-bank estimates form an approximate 95% Monte Carlo interval. The largest endpoint deviation from the original estimate is compared with a precision target:

- Terminal median: 2% of the median's absolute value.
- Drawdown P95: 0.02, i.e. two percentage points.

At least 32 fitted models and 128 paths per model are required to report `within_tolerance`. These are pragmatic minimum checks, not mathematically sufficient conditions for convergence. The intervals are conditional on the sampled per-model paths and do not separately estimate inner-path error. Small banks can miss rare parameter regions entirely, so the diagnostic must be complemented by independently seeded banks. It does not certify economic realism.

Terminal wealth is used instead of incorrectly labeling cash-flow-affected wealth growth as investment CAGR. Drawdown uses the same portfolio accounting as the reported simulation, including configured cash flows and taxes. The UI explicitly distinguishes these precision intervals from predictive investment-outcome bands.

## Reproduction and tests

```sh
.venv/bin/python scripts/benchmark_parameter_convergence.py
.venv/bin/python -m pytest tests/test_ensemble_convergence.py tests/test_pipeline.py tests/test_api.py tests/test_validation.py tests/test_path_validation.py tests/test_web_app.py -q
node --check web/app.js
```

The benchmark writes `convergence.csv`, per-bank/path-seed metric arrays, and a manifest to `research/parameter_convergence_2026-10-12/`. Tests cover weighted mixture quantiles and point masses, unequal path counts, whole-model resampling, reproducibility, invalid samples, insufficient banks, and seed separation through real pipeline simulations. Repeating conditional paths does not spuriously shrink the reported outer error.

Verification completed: all 513 Python tests pass, all seven portfolio-preset JavaScript tests pass, changed Python production modules pass Ruff, and `web/app.js` passes the JavaScript syntax check. A live browser smoke run was not performed in this environment.

The next economic-model comparisons should hold a sufficiently large parameter bank fixed while changing the model, then repeat across independent banks. Return anchors require explicit, dated return/yield assumptions; persistent bond stress requires a yield/duration channel and inflation-sensitive joint behavior, not an arbitrary increase in unconditional volatility.
