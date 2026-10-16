#!/usr/bin/env python3
"""Reproduce the monthly realized-variance proxy from frozen adjusted daily closes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def monthly_realized_variance(prices):
    daily = np.log(prices / prices.shift(1))
    grouped = daily.resample("ME")
    count = grouped.count()
    rv = (grouped.apply(lambda x: (x * x).sum()) - grouped.sum() ** 2 / count) * count / (count - 1)
    return rv.where(count >= 15)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", default="research/sv_extensions_2026-10-04")
    args = parser.parse_args()
    directory = ROOT / args.directory
    path = directory / "daily_adjusted_prices.csv"
    manifest = json.loads((directory / "daily_manifest.json").read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["sha256"]
    prices = pd.read_csv(path, index_col=0, parse_dates=True)
    rv = monthly_realized_variance(prices)
    target = directory / "monthly_realized_variance.csv"
    if target.exists():
        existing = pd.read_csv(target, index_col=0, parse_dates=True)
        pd.testing.assert_frame_equal(existing, rv, check_freq=False, rtol=1e-11, atol=1e-15)
        print("Verified existing frozen realized-variance proxy; preserved its bytes.")
    else:
        rv.to_csv(target)
    metadata = dict(
        source=manifest,
        formula="n/(n-1) * (sum(daily_log_return^2) - sum(daily_log_return)^2/n)",
        minimum_daily_returns=15,
        months=len(rv),
        limitation="Demeaned daily-close proxy, not intraday realized variance.",
        monthly_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
    )
    (directory / "realized_manifest.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
