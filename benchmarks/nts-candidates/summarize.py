"""Summarize saved candidate timings; ratios compare medians at each point."""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read(name):
    return list(csv.DictReader((HERE / name).open()))


def median(rows, engine, mode):
    return statistics.median(float(r["ns_per_vector"]) for r in rows if r["engine"] == engine and r["mode"] == mode)


def main():
    rows = read("confirmation.csv")
    points = sorted({(r["tail"], r["tempering"]) for r in rows})
    candidates = [("plus", "scalar"), ("plusplus", "scalar"), ("mcfarland", "scalar"),
                  ("current", "batch64"), ("current", "batch256"), ("current", "vforce64"),
                  ("current", "vforce256"), ("plus", "vforce256"), ("plusplus", "vforce256"),
                  ("mcfarland", "vforce256")]
    summaries = []
    for assets in ("1", "4"):
        for engine, mode in candidates:
            ratios, detailed = [], []
            for tail, tempering in points:
                group = [r for r in rows if r["tail"] == tail and r["tempering"] == tempering and r["assets"] == assets]
                baseline, candidate = median(group, "current", "scalar"), median(group, engine, mode)
                ratios.append(baseline / candidate)
                detailed.append(dict(tail=float(tail), tempering=float(tempering), baseline_ns=baseline,
                                     candidate_ns=candidate, speedup=baseline / candidate))
            summaries.append(dict(engine=engine, mode=mode, assets=int(assets),
                                  median_speedup=statistics.median(ratios), minimum_speedup=min(ratios),
                                  maximum_speedup=max(ratios), points=detailed))
    short = []
    path_rows = read("short_paths.csv")
    for periods in sorted({r["periods"] for r in path_rows}):
        group = [r for r in path_rows if r["periods"] == periods]
        baseline = median(group, "current", "scalar")
        for engine, mode in sorted({(r["engine"], r["mode"]) for r in group}):
            candidate = median(group, engine, mode)
            short.append(dict(periods=int(periods), engine=engine, mode=mode, baseline_ns=baseline,
                              candidate_ns=candidate, speedup=baseline / candidate))
    validation = read("validation.csv")
    report = dict(confirmation_runs=len(rows), validation_runs=len(validation),
                  validation_points=len({(r["tail"], r["tempering"]) for r in validation}),
                  largest_distribution_error_standard_errors=max(float(r["max_z"]) for r in validation),
                  bulk=summaries, short_paths=short)
    (HERE / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    for entry in summaries:
        if entry["assets"] == 4:
            print(f"{entry['engine']:12s} {entry['mode']:12s} {entry['median_speedup']:.3f}x median "
                  f"({entry['minimum_speedup']:.3f}–{entry['maximum_speedup']:.3f}x)")
    print(f"Validated {len(validation)} runs across {report['validation_points']} points; "
          f"maximum error {report['largest_distribution_error_standard_errors']:.3f} standard errors")


if __name__ == "__main__":
    main()
