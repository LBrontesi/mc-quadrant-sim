import numpy as np
import pandas as pd
import pytest

from mc_quadrants.data import (
    backfill_prices,
    combine_observed_and_simulated_returns,
    prices_to_returns,
    simulate_pre_inception_returns,
)


def test_prices_to_returns_sorts_dates_and_preserves_return_method():
    prices = pd.DataFrame(
        {"Stocks": [110.0, 100.0]},
        index=pd.to_datetime(["2020-02-29", "2020-01-31"]),
    )

    returns = prices_to_returns(prices, method="SIMPLE")

    assert returns.index.tolist() == list(pd.to_datetime(["2020-01-31", "2020-02-29"]))
    assert np.isnan(returns.iloc[0, 0])
    assert returns.iloc[1, 0] == pytest.approx(0.10)


def test_prices_to_returns_rejects_invalid_price_levels():
    prices = pd.DataFrame({"Stocks": [100.0, 0.0]})

    with pytest.raises(ValueError, match="greater than zero"):
        prices_to_returns(prices)


def test_prices_to_returns_allows_missing_observations():
    prices = pd.DataFrame({"Stocks": [np.nan, 100.0, 110.0]})

    returns = prices_to_returns(prices)

    assert np.isnan(returns.iloc[0, 0])
    assert returns.iloc[2, 0] == pytest.approx(np.log(1.1))


def test_backfill_prices_scales_proxy_and_preserves_primary_values():
    primary = pd.DataFrame(
        {"SPY": [200.0, 220.0]},
        index=pd.to_datetime(["2020-02-29", "2020-03-31"]),
    )
    proxy = pd.Series(
        [100.0, 110.0],
        index=pd.to_datetime(["2019-12-31", "2020-02-29"]),
    )

    extended = backfill_prices(primary, {"SPY": proxy})

    assert extended.loc[pd.Timestamp("2019-12-31"), "SPY"] == pytest.approx(200.0 / 110.0 * 100.0)
    assert extended.loc[pd.Timestamp("2020-02-29"), "SPY"] == pytest.approx(200.0)
    assert extended.loc[pd.Timestamp("2020-03-31"), "SPY"] == pytest.approx(220.0)


def test_simulated_source_is_separate_from_stitched_series():
    dates = pd.date_range("2020-02-29", periods=12, freq="ME")
    observed = pd.DataFrame({"IEF": np.linspace(-0.02, 0.02, len(dates))}, index=dates)

    simulated = simulate_pre_inception_returns(
        observed,
        assets=["IEF"],
        start="2019-01-01",
        random_seed=7,
    )
    combined = combine_observed_and_simulated_returns(observed, simulated)

    assert "IEF_SIM" in combined.columns
    assert "IEFSIM" in combined.columns
    assert combined.loc[: "2020-01-31", "IEF_SIM"].notna().any()
    assert combined.loc["2020-02-29":, "IEF_SIM"].isna().all()
    assert combined.loc[:, "IEFSIM"].notna().all()
