import numpy as np
import pandas as pd
import pytest

from mc_quadrants.backfill import (
    backward_price_levels,
    categorize_asset,
    simulate_regime_conditioned_pre_inception_returns,
)


def _month_ends(start: str, periods: int) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=periods, freq="ME")


def _anchor_from_growth(dates: pd.DatetimeIndex, growth: np.ndarray) -> pd.Series:
    rng = np.random.default_rng(3)
    return pd.Series(
        np.where(growth >= 0, 0.02, -0.03) + rng.normal(0, 0.004, len(dates)),
        index=dates,
    )


def _dataset():
    dates = _month_ends("2018-01-31", 84)
    growth = np.where(
        dates < pd.Timestamp("2020-01-31"),
        4.0,
        np.where(dates < pd.Timestamp("2022-01-31"), -2.0, 4.0),
    )
    inflation = np.where(growth >= 0, 2.0, 4.0)
    macro = pd.DataFrame({"growth": growth, "inflation": inflation}, index=dates)
    anchor_series = _anchor_from_growth(dates, growth)
    anchor_returns = anchor_series.to_frame("SPY")
    observed_dates = dates[dates >= pd.Timestamp("2022-01-31")]
    observed_anchor = anchor_series.loc[observed_dates].to_numpy(dtype=float)
    rng = np.random.default_rng(11)
    dbmf = pd.Series(
        0.8 * observed_anchor + rng.normal(0, 0.004, len(observed_dates)),
        index=observed_dates,
        name="DBMF",
    )
    returns = pd.DataFrame({"SPY": anchor_series, "DBMF": dbmf})
    return macro, returns, anchor_returns


def _run(dataset, **overrides):
    macro, returns, anchor_returns = dataset
    kwargs = {
        "returns": returns,
        "macro": macro,
        "assets": ["DBMF"],
        "growth_threshold": 0.0,
        "inflation_threshold": 3.0,
        "macro_lag_periods": 0,
        "anchor_returns": anchor_returns,
        "random_seed": 7,
        "min_observations": 12,
        "degrees_of_freedom": 30,
    }
    kwargs.update(overrides)
    return simulate_regime_conditioned_pre_inception_returns(**kwargs)


def test_regime_conditioned_generates_stitched_history():
    simulated, report = _run(_dataset())

    assert "DBMF_SIM" in simulated.columns
    assert simulated.index.min() < pd.Timestamp("2022-01-31")
    assert report["DBMF"]["history_months"] == 36
    assert report["DBMF"]["factor_r2"] > 0.5
    assert report["DBMF"]["grade"] in {"A", "B", "C"}


def test_regime_conditioned_respects_macro_states():
    simulated, _ = _run(_dataset())

    pre = simulated["DBMF_SIM"]
    goldilocks = pre[pre.index < pd.Timestamp("2020-01-31")]
    stagflation = pre[
        (pre.index >= pd.Timestamp("2020-01-31")) & (pre.index < pd.Timestamp("2022-01-31"))
    ]
    assert goldilocks.mean() > stagflation.mean()


def test_regime_conditioned_is_reproducible_with_seed():
    first, _ = _run(_dataset())
    second, _ = _run(_dataset())

    assert first["DBMF_SIM"].equals(second["DBMF_SIM"])


def test_regime_conditioned_rejects_short_history():
    macro, returns, anchor_returns = _dataset()
    short_returns = returns.loc[returns.index >= pd.Timestamp("2024-07-31")]

    simulated, report = _run((macro, short_returns, anchor_returns))

    assert report["DBMF"]["grade"] == "X"
    assert simulated.empty


def test_numeric_thresholds_classify_from_the_start():
    simulated, _ = _run(_dataset())

    assert simulated.index.min() == pd.Timestamp("2018-01-31")


def test_categorize_asset_applies_overrides():
    assert categorize_asset("DBMF") == "MANAGED_FUTURES"
    assert categorize_asset("DBMF_SIM", {"DBMF": "COMMODITIES"}) == "COMMODITIES"
    assert categorize_asset("UNKNOWN") == "UNCATEGORIZED"


def test_backward_price_levels_reconstructs_anchor():
    returns = pd.Series(
        [0.10, -0.05, 0.02],
        index=_month_ends("2018-01-31", 3),
    )

    prices = backward_price_levels(returns, 100.0)

    assert prices.iloc[0] == pytest.approx(100.0 * np.exp(-0.07))
    assert prices.iloc[1] == pytest.approx(100.0 * np.exp(0.03))
    assert prices.iloc[2] == pytest.approx(100.0 * np.exp(-0.02))
