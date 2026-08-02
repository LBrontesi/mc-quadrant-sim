from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


def read_prices_csv(path: str, date_col: str = "Date") -> pd.DataFrame:
    """Read a wide price CSV with one date column and one column per asset."""

    prices = pd.read_csv(path, parse_dates=[date_col])
    return prices.set_index(date_col).sort_index()


def read_macro_csv(path: str, date_col: str = "Date") -> pd.DataFrame:
    """Read a macro CSV with one date column and macro indicator columns."""

    macro = pd.read_csv(path, parse_dates=[date_col])
    return macro.set_index(date_col).sort_index()


def prices_to_returns(prices: pd.DataFrame, method: str = "log") -> pd.DataFrame:
    """Convert prices to log or simple returns."""

    numeric_prices = prices.astype(float)
    if method == "log":
        return np.log(numeric_prices / numeric_prices.shift(1))
    if method == "simple":
        return numeric_prices.pct_change()
    raise ValueError("method must be 'log' or 'simple'.")


def fetch_yahoo_prices(
    tickers: Sequence[str] | str,
    start: str,
    end: str | None = None,
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """Fetch adjusted prices with yfinance when the optional dependency is installed."""

    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("Install optional dependency with: pip install -e '.[data]'") from exc

    ticker_list = [tickers] if isinstance(tickers, str) else list(tickers)
    data = yf.download(
        ticker_list,
        start=start,
        end=end,
        auto_adjust=auto_adjust,
        progress=False,
    )

    if isinstance(data.columns, pd.MultiIndex):
        price_field = "Close" if auto_adjust else "Adj Close"
        if price_field not in data.columns.get_level_values(0):
            price_field = "Close"
        prices = data[price_field]
    else:
        price_field = "Close" if auto_adjust else "Adj Close"
        if isinstance(data, pd.Series):
            prices = data.to_frame(name=ticker_list[0])
        elif price_field in data.columns:
            prices = data[[price_field]].rename(columns={price_field: ticker_list[0]})
        elif "Close" in data.columns:
            prices = data[["Close"]].rename(columns={"Close": ticker_list[0]})
        else:
            prices = data

    return prices.dropna(how="all").sort_index()


def fetch_fred_macro(series: Mapping[str, str], start: str, end: str | None = None) -> pd.DataFrame:
    """Fetch FRED series with pandas-datareader when the optional dependency is installed."""

    try:
        from pandas_datareader import data as pdr
    except ImportError as exc:
        raise ImportError("Install optional dependency with: pip install -e '.[data]'") from exc

    frames = []
    for output_name, fred_code in series.items():
        raw = pdr.DataReader(fred_code, "fred", start, end)
        frames.append(raw.rename(columns={fred_code: output_name}))

    return pd.concat(frames, axis=1).sort_index()


def yoy_change(data: pd.DataFrame, periods: int = 12, scale: float = 100.0) -> pd.DataFrame:
    """Compute year-over-year percent changes for monthly-style macro data."""

    return data.pct_change(periods=periods) * scale
