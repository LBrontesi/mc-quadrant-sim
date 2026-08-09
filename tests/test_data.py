import numpy as np
import pandas as pd
import pytest

import mc_quadrants.data as data_module
from mc_quadrants.data import (
    align_macro_to_availability,
    backfill_prices,
    combine_observed_and_simulated_returns,
    convert_returns_to_base_currency,
    prices_to_returns,
    simulate_pre_inception_returns,
    yoy_change,
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


def test_currency_conversion_adds_historical_fx_return():
    dates = pd.date_range("2020-01-31", periods=3, freq="ME")
    local_returns = pd.DataFrame({"EFA": [0.01, 0.02, -0.01]}, index=dates)
    fx_rates = pd.DataFrame({"EUR": [1.10, 1.20, 1.10]}, index=dates)

    converted = convert_returns_to_base_currency(
        local_returns,
        asset_currencies={"EFA": "EUR"},
        base_currency="USD",
        fx_rates=fx_rates,
    )

    assert converted.loc[dates[0], "EFA"] == pytest.approx(0.01)
    assert converted.loc[dates[1], "EFA"] == pytest.approx(0.02 + np.log(1.20 / 1.10))
    assert converted.loc[dates[2], "EFA"] == pytest.approx(-0.01 + np.log(1.10 / 1.20))


def test_fetch_yahoo_fx_rates_normalizes_direct_pair(monkeypatch):
    dates = pd.date_range("2020-01-01", periods=2, freq="D")

    def fake_fetch(tickers, start, end):
        assert tickers == ["EURUSD=X"]
        return pd.DataFrame({"EURUSD=X": [1.10, 1.20]}, index=dates)

    monkeypatch.setattr(data_module, "fetch_yahoo_prices", fake_fetch)

    rates = data_module.fetch_yahoo_fx_rates(["eur"], "usd", "2020-01-01", "2020-01-03")

    assert rates.columns.tolist() == ["EUR"]
    assert rates["EUR"].tolist() == [1.10, 1.20]


def test_market_data_drops_an_incomplete_final_month(monkeypatch):
    dates = pd.date_range("2020-01-01", "2020-03-15", freq="D")
    prices = pd.DataFrame({"SPY": np.linspace(100.0, 120.0, len(dates))}, index=dates)
    macro_dates = pd.date_range("2018-01-01", "2020-03-01", freq="MS")
    macro_levels = pd.DataFrame(
        {
            "growth": np.linspace(100.0, 130.0, len(macro_dates)),
            "inflation": np.linspace(100.0, 125.0, len(macro_dates)),
        },
        index=macro_dates,
    )

    monkeypatch.setattr(data_module, "fetch_yahoo_prices", lambda *args, **kwargs: prices)
    monkeypatch.setattr(data_module, "fetch_fred_macro", lambda *args, **kwargs: macro_levels)
    data_module._load_market_data_cached.cache_clear()

    _, returns, _, _ = data_module.load_market_data(["SPY"], "2018-01-01", "2020-03-15")

    assert returns.index.max() == pd.Timestamp("2020-02-29")


def test_yoy_macro_preserves_and_aligns_initial_release_dates():
    dates = pd.date_range("2019-01-01", periods=15, freq="MS")
    levels = pd.DataFrame(
        {
            "growth": np.linspace(100.0, 115.0, len(dates)),
            "inflation": np.linspace(100.0, 107.0, len(dates)),
        },
        index=dates,
    )
    levels.attrs.update(
        {
            "release_dates": pd.DataFrame(
                {
                    "growth": dates + pd.Timedelta(days=45),
                    "inflation": dates + pd.Timedelta(days=40),
                },
                index=dates,
            ),
            "data_vintage": "initial_release",
            "point_in_time": True,
        }
    )

    transformed = yoy_change(levels).dropna()
    available = align_macro_to_availability(transformed)

    assert available.attrs["point_in_time"] is True
    assert available.attrs["availability_aligned"] is True
    assert available.index[0] == pd.Timestamp("2020-02-29")
