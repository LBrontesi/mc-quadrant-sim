from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import timedelta
from functools import lru_cache

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

    method = str(method).lower()
    if method not in {"log", "simple"}:
        raise ValueError("method must be 'log' or 'simple'.")
    if prices.empty:
        raise ValueError("prices must contain at least one row.")
    if prices.index.has_duplicates:
        raise ValueError("prices must not contain duplicate dates.")
    try:
        numeric_prices = prices.sort_index().apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("prices must contain only numeric values.") from exc
    values = numeric_prices.to_numpy(dtype=float)
    invalid = np.isinf(values) | (np.isfinite(values) & (values <= 0))
    if invalid.any():
        raise ValueError("Price levels must be finite and greater than zero when present.")
    if method == "log":
        return np.log(numeric_prices / numeric_prices.shift(1))
    if method == "simple":
        return numeric_prices.pct_change()


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


@lru_cache(maxsize=8)
def _load_market_data_cached(
    tickers: tuple[str, ...],
    start: str,
    end: str,
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[str, ...]]:
    start_date = pd.Timestamp(start)
    end_date = pd.Timestamp(end)
    if end_date <= start_date:
        raise ValueError("History end must be after history start.")

    prices = fetch_yahoo_prices(
        list(tickers),
        start=start_date.strftime("%Y-%m-%d"),
        end=(end_date + timedelta(days=1)).strftime("%Y-%m-%d"),
    )
    prices = prices.copy()
    prices.columns = [str(column).strip().upper() for column in prices.columns]
    available = tuple(
        ticker
        for ticker in tickers
        if ticker in prices.columns and prices[ticker].notna().sum() >= 2
    )
    if not available:
        raise ValueError("Yahoo Finance did not return usable price history for these tickers.")

    returns = prices_to_returns(prices.loc[:, list(available)], method="log")
    returns = returns.resample("ME").sum(min_count=1).dropna(how="all")
    macro_levels = fetch_fred_macro(
        {"growth": "INDPRO", "inflation": "CPIAUCSL"},
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
    ).resample("ME").last()
    macro = yoy_change(macro_levels).apply(pd.to_numeric, errors="coerce").dropna(how="any")
    if macro.empty:
        raise ValueError("Not enough FRED history to calculate year-over-year growth and inflation.")
    return macro, returns, available


def load_market_data(
    tickers: Sequence[str],
    start: str,
    end: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Load cached Yahoo prices and FRED macro inputs for dashboard use."""

    raw_tickers = (tickers,) if isinstance(tickers, str) else tickers
    normalized_tickers = tuple(dict.fromkeys(str(ticker).strip().upper() for ticker in raw_tickers))
    macro, returns, available = _load_market_data_cached(
        normalized_tickers,
        pd.Timestamp(start).strftime("%Y-%m-%d"),
        pd.Timestamp(end).strftime("%Y-%m-%d"),
    )
    return macro.copy(), returns.copy(), list(available)


def yoy_change(data: pd.DataFrame, periods: int = 12, scale: float = 100.0) -> pd.DataFrame:
    """Compute year-over-year percent changes for monthly-style macro data."""

    return data.pct_change(periods=periods) * scale
