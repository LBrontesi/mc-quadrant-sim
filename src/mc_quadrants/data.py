from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import timedelta
from functools import lru_cache

import numpy as np
import pandas as pd

from mc_quadrants.backfill import (
    DEFAULT_ANCHOR_UNIVERSE,
    simulate_regime_conditioned_pre_inception_returns,
)


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


def backfill_prices(
    prices: pd.DataFrame,
    historical_proxies: Mapping[str, pd.Series | pd.DataFrame],
) -> pd.DataFrame:
    """Fill missing asset history with level-scaled proxy series.

    Primary observations always win. Each proxy must overlap its asset at
    least once so its level can be scaled to the primary series; the proxy is
    then used only where the primary value is missing.
    """

    if prices.index.has_duplicates:
        raise ValueError("prices must not contain duplicate dates.")

    extended = prices.sort_index().copy()
    for asset, proxy in historical_proxies.items():
        asset = str(asset).strip()
        if not asset:
            raise ValueError("Historical proxy asset names must not be empty.")

        if isinstance(proxy, pd.DataFrame):
            if proxy.shape[1] != 1:
                raise ValueError(f"Historical proxy for {asset} must contain exactly one series.")
            proxy_series = proxy.iloc[:, 0]
        else:
            proxy_series = proxy
        if not isinstance(proxy_series, pd.Series):
            raise TypeError(f"Historical proxy for {asset} must be a Series or one-column DataFrame.")
        if proxy_series.index.has_duplicates:
            raise ValueError(f"Historical proxy for {asset} has duplicate dates.")

        try:
            proxy_series = proxy_series.sort_index().apply(pd.to_numeric, errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Historical proxy for {asset} must be numeric.") from exc
        proxy_values = proxy_series.to_numpy(dtype=float)
        invalid = np.isinf(proxy_values) | (np.isfinite(proxy_values) & (proxy_values <= 0))
        if invalid.any():
            raise ValueError(f"Historical proxy for {asset} contains invalid price levels.")

        if asset not in extended.columns:
            extended[asset] = np.nan
        extended = extended.reindex(extended.index.union(proxy_series.index)).sort_index()
        overlap = pd.concat(
            [extended[asset].rename("primary"), proxy_series.rename("proxy")],
            axis=1,
            join="inner",
        ).dropna()
        if overlap.empty:
            raise ValueError(f"Historical proxy for {asset} must overlap the primary price history.")

        primary_anchor, proxy_anchor = overlap.iloc[-1].to_numpy(dtype=float)
        if primary_anchor <= 0 or proxy_anchor <= 0:
            raise ValueError(f"Historical proxy for {asset} has a non-positive overlap value.")
        scaled_proxy = proxy_series * (primary_anchor / proxy_anchor)
        extended[asset] = extended[asset].combine_first(scaled_proxy)

    return extended.sort_index()


def simulate_pre_inception_returns(
    returns: pd.DataFrame,
    assets: Sequence[str] | str,
    start: str | pd.Timestamp,
    random_seed: int = 42,
    distribution: str = "student_t",
    degrees_of_freedom: float = 5.0,
    frequency: str = "ME",
) -> pd.DataFrame:
    """Generate source-labeled monthly returns before each asset's inception.

    The observed return mean and covariance are used as calibration inputs. A
    generated series ends immediately before the first observed return for its
    asset, so it cannot overwrite or overlap the observed source.
    """

    if returns.empty:
        raise ValueError("returns must contain at least one row.")
    if returns.index.has_duplicates:
        raise ValueError("returns must not contain duplicate dates.")

    raw_assets = (assets,) if isinstance(assets, str) else assets
    asset_list = list(dict.fromkeys(str(asset).strip() for asset in raw_assets))
    if not asset_list or any(not asset for asset in asset_list):
        raise ValueError("At least one non-empty asset is required.")
    missing_assets = [asset for asset in asset_list if asset not in returns.columns]
    if missing_assets:
        raise KeyError(f"Returns are missing synthetic assets: {', '.join(missing_assets)}")

    distribution = str(distribution).lower().replace("-", "_")
    if distribution not in {"normal", "student_t", "t"}:
        raise ValueError("distribution must be 'normal' or 'student_t'.")
    if distribution == "t":
        distribution = "student_t"
    if distribution == "student_t" and (
        not np.isfinite(degrees_of_freedom) or degrees_of_freedom <= 2
    ):
        raise ValueError("degrees_of_freedom must be finite and greater than 2 for Student-t returns.")

    try:
        observed = returns.sort_index().loc[:, asset_list].apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("returns must contain only numeric values.") from exc
    if np.isinf(observed.to_numpy(dtype=float)).any():
        raise ValueError("returns must contain only finite values.")

    first_observations: dict[str, pd.Timestamp] = {}
    for asset in asset_list:
        valid = observed[asset].dropna()
        if len(valid) < 2:
            raise ValueError(f"At least two observed returns are required for synthetic asset: {asset}")
        first_observations[asset] = pd.Timestamp(valid.index[0])

    first_observation = max(first_observations.values())
    start_period = pd.Timestamp(start).to_period("M").to_timestamp("M")
    end_period = first_observation - pd.offsets.MonthEnd(1)
    columns = [f"{asset}_SIM" for asset in asset_list]
    if start_period > end_period:
        return pd.DataFrame(columns=columns, index=pd.DatetimeIndex([], name=observed.index.name))
    simulation_index = pd.date_range(start_period, end_period, freq=frequency)

    means = observed.mean().to_numpy(dtype=float)
    covariance = observed.cov().reindex(index=asset_list, columns=asset_list)
    variances = observed.var(ddof=1).to_numpy(dtype=float)
    covariance_values = covariance.to_numpy(dtype=float)
    for index, variance in enumerate(variances):
        if not np.isfinite(covariance_values[index, index]):
            covariance_values[index, index] = variance if np.isfinite(variance) else 0.0
    covariance_values = np.nan_to_num(covariance_values, nan=0.0, posinf=0.0, neginf=0.0)
    covariance_values = (covariance_values + covariance_values.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_values)
    covariance_values = (eigenvectors * np.clip(eigenvalues, 1e-12, None)) @ eigenvectors.T
    covariance_values = (covariance_values + covariance_values.T) / 2.0

    rng = np.random.default_rng(random_seed)
    if distribution == "normal":
        draws = rng.multivariate_normal(means, covariance_values, size=len(simulation_index))
    else:
        scale = covariance_values * (degrees_of_freedom - 2.0) / degrees_of_freedom
        normal_draws = rng.multivariate_normal(
            np.zeros(len(asset_list)),
            scale,
            size=len(simulation_index),
        )
        chi_squared = rng.chisquare(degrees_of_freedom, size=len(simulation_index))
        draws = means + normal_draws / np.sqrt(chi_squared / degrees_of_freedom)[:, None]

    simulated = pd.DataFrame(np.nan, index=simulation_index, columns=columns)
    for index, asset in enumerate(asset_list):
        asset_dates = simulation_index < first_observations[asset]
        simulated.loc[asset_dates, f"{asset}_SIM"] = draws[asset_dates, index]
    simulated.index.name = observed.index.name
    return simulated


def combine_observed_and_simulated_returns(
    observed_returns: pd.DataFrame,
    simulated_returns: pd.DataFrame,
    simulation_suffix: str = "_SIM",
    stitched_suffix: str = "SIM",
) -> pd.DataFrame:
    """Keep source columns and add a full stitched ``ASSETSIM`` series."""

    if observed_returns.index.has_duplicates or simulated_returns.index.has_duplicates:
        raise ValueError("Observed and simulated returns must not contain duplicate dates.")
    if not simulation_suffix or not stitched_suffix:
        raise ValueError("Source suffixes must not be empty.")

    simulated = simulated_returns.sort_index()
    combined_index = observed_returns.index.union(simulated.index)
    combined = observed_returns.sort_index().reindex(combined_index)
    for simulation_column in simulated.columns:
        if not str(simulation_column).endswith(simulation_suffix):
            raise ValueError(f"Simulated return column must end with {simulation_suffix}: {simulation_column}")
        asset = str(simulation_column)[: -len(simulation_suffix)]
        if asset not in combined.columns:
            raise KeyError(f"Observed returns are missing simulated asset: {asset}")
        stitched_column = f"{asset}{stitched_suffix}"
        if stitched_column in combined.columns:
            raise ValueError(f"Stitched return column already exists: {stitched_column}")
        simulated_series = simulated[simulation_column].reindex(combined_index)
        combined[simulation_column] = simulated_series
        combined[stitched_column] = combined[asset].combine_first(simulated_series)

    return combined.sort_index()


def convert_returns_to_base_currency(
    returns: pd.DataFrame,
    asset_currencies: Mapping[str, str] | None = None,
    base_currency: str = "USD",
    fx_rates: pd.DataFrame | None = None,
    fx_quote: str = "base_per_foreign",
    default_asset_currency: str = "USD",
    fx_frequency: str = "ME",
) -> pd.DataFrame:
    """Convert log returns into a base currency using historical FX levels.

    ``fx_rates`` must contain positive levels quoted as base-currency units per
    unit of foreign currency, such as ``EURUSD=X`` for EUR assets in a USD
    portfolio. A static spot quote changes the value level but cannot model FX
    risk, so historical levels are required whenever a foreign asset is used.
    """

    if returns.empty:
        raise ValueError("returns must contain at least one row.")
    if returns.index.has_duplicates:
        raise ValueError("returns must not contain duplicate dates.")
    if fx_quote not in {"base_per_foreign", "foreign_per_base"}:
        raise ValueError("fx_quote must be 'base_per_foreign' or 'foreign_per_base'.")

    base = str(base_currency).strip().upper()
    default = str(default_asset_currency).strip().upper()
    if len(base) != 3 or len(default) != 3:
        raise ValueError("Currency codes must be three-letter ISO codes.")

    currencies = {str(asset).strip().upper(): str(currency).strip().upper() for asset, currency in (asset_currencies or {}).items()}
    normalized_returns = returns.sort_index().copy()
    foreign_assets: dict[str, str] = {}
    for asset in normalized_returns.columns:
        asset_name = str(asset).strip().upper()
        base_asset = asset_name.removesuffix("_SIM").removesuffix("SIM")
        currency = currencies.get(asset_name, currencies.get(base_asset, default))
        if len(currency) != 3:
            raise ValueError(f"Invalid currency for asset {asset}: {currency}")
        if currency != base:
            foreign_assets[asset] = currency

    if not foreign_assets:
        return normalized_returns
    if fx_rates is None:
        raise ValueError(
            f"Historical FX rates are required to convert {base} portfolio returns from: "
            + ", ".join(sorted(set(foreign_assets.values())))
        )
    if not isinstance(fx_rates.index, pd.DatetimeIndex) or fx_rates.index.has_duplicates:
        raise ValueError("fx_rates must use a unique DatetimeIndex.")
    try:
        levels = fx_rates.sort_index().apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("fx_rates must contain only numeric values.") from exc
    levels.columns = [str(column).strip().upper() for column in levels.columns]
    level_values = levels.to_numpy(dtype=float)
    invalid = np.isinf(level_values) | (np.isfinite(level_values) & (level_values <= 0))
    if invalid.any():
        raise ValueError("fx_rates must contain positive finite levels when present.")

    missing_currencies = sorted(set(foreign_assets.values()).difference(levels.columns))
    if missing_currencies:
        raise KeyError(f"FX rates are missing currencies: {', '.join(missing_currencies)}")
    period_levels = levels.ffill().resample(fx_frequency).last()
    fx_returns = np.log(period_levels / period_levels.shift(1))
    aligned_fx_returns = fx_returns.reindex(normalized_returns.index, method="ffill").fillna(0.0)

    converted = normalized_returns.copy()
    for asset, currency in foreign_assets.items():
        adjustment = aligned_fx_returns[currency]
        if fx_quote == "base_per_foreign":
            converted[asset] = converted[asset] + adjustment
        else:
            converted[asset] = converted[asset] - adjustment
    return converted


def fetch_yahoo_fx_rates(
    currencies: Sequence[str],
    base_currency: str,
    start: str,
    end: str | None = None,
) -> pd.DataFrame:
    """Fetch Yahoo FX levels quoted as base currency per foreign currency."""

    base = str(base_currency).strip().upper()
    requested = list(dict.fromkeys(str(currency).strip().upper() for currency in currencies))
    requested = [currency for currency in requested if currency and currency != base]
    if len(base) != 3 or any(len(currency) != 3 for currency in requested):
        raise ValueError("Currency codes must be three-letter ISO codes.")
    if not requested:
        return pd.DataFrame()

    frames: dict[str, pd.Series] = {}
    missing: list[str] = []
    for currency in requested:
        direct_ticker = f"{currency}{base}=X"
        inverse_ticker = f"{base}{currency}=X"
        direct = fetch_yahoo_prices([direct_ticker], start=start, end=end)
        direct_column = direct_ticker.upper()
        if direct_column in direct.columns and direct[direct_column].notna().sum() >= 2:
            frames[currency] = direct[direct_column].rename(currency)
            continue

        inverse = fetch_yahoo_prices([inverse_ticker], start=start, end=end)
        inverse_column = inverse_ticker.upper()
        if inverse_column in inverse.columns and inverse[inverse_column].notna().sum() >= 2:
            frames[currency] = (1.0 / inverse[inverse_column]).rename(currency)
        else:
            missing.append(currency)

    if missing:
        raise ValueError(f"Yahoo Finance returned no usable FX history for: {', '.join(missing)}")
    return pd.concat(frames.values(), axis=1).sort_index()


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
    ticker_list = list(dict.fromkeys(str(ticker).strip().upper() for ticker in ticker_list if str(ticker).strip()))
    if not ticker_list:
        raise ValueError("At least one Yahoo Finance ticker is required.")
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
    historical_proxies: tuple[tuple[str, str], ...] = (),
    synthetic_assets: tuple[str, ...] = (),
    synthetic_seed: int = 42,
    synthetic_method: str = "regime",
    synthetic_categories: tuple[tuple[str, str], ...] = (),
    growth_threshold: str | float = "median",
    inflation_threshold: str | float = "median",
    threshold_window: int | None = None,
    macro_lag_periods: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[str, ...], dict[str, dict[str, object]]]:
    start_date = pd.Timestamp(start)
    end_date = pd.Timestamp(end)
    if end_date <= start_date:
        raise ValueError("History end must be after history start.")

    fetch_tickers = list(tickers)
    for asset in synthetic_assets:
        if asset not in fetch_tickers:
            fetch_tickers.append(asset)
    needs_anchors = bool(synthetic_assets) and synthetic_method == "regime"
    if needs_anchors:
        for anchor in DEFAULT_ANCHOR_UNIVERSE:
            if anchor not in fetch_tickers:
                fetch_tickers.append(anchor)

    prices = fetch_yahoo_prices(
        list(fetch_tickers),
        start=start_date.strftime("%Y-%m-%d"),
        end=(end_date + timedelta(days=1)).strftime("%Y-%m-%d"),
    )
    prices = prices.copy()
    prices.columns = [str(column).strip().upper() for column in prices.columns]
    prices = prices.reindex(columns=list(fetch_tickers))

    if historical_proxies:
        proxy_tickers = tuple(proxy for _, proxy in historical_proxies)
        proxy_prices = fetch_yahoo_prices(
            list(proxy_tickers),
            start=start_date.strftime("%Y-%m-%d"),
            end=(end_date + timedelta(days=1)).strftime("%Y-%m-%d"),
        )
        proxy_prices = proxy_prices.copy()
        proxy_prices.columns = [str(column).strip().upper() for column in proxy_prices.columns]
        proxy_series = {}
        for asset, proxy_ticker in historical_proxies:
            if proxy_ticker not in proxy_prices.columns:
                raise ValueError(f"Yahoo Finance returned no history for proxy ticker: {proxy_ticker}")
            proxy_series[asset] = proxy_prices[proxy_ticker]
        prices = backfill_prices(prices, proxy_series)

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

    synthetic_report: dict[str, dict[str, object]] = {}
    if synthetic_assets:
        if synthetic_method == "regime":
            anchor_columns = [anchor for anchor in DEFAULT_ANCHOR_UNIVERSE if anchor in prices.columns]
            anchor_returns = (
                prices_to_returns(prices.loc[:, anchor_columns], method="log")
                .resample("ME")
                .sum(min_count=1)
                .dropna(how="all")
                if anchor_columns
                else None
            )
            category_map = {asset: category for asset, category in synthetic_categories}
            simulated_returns, synthetic_report = simulate_regime_conditioned_pre_inception_returns(
                returns,
                macro,
                assets=list(synthetic_assets),
                growth_threshold=growth_threshold,
                inflation_threshold=inflation_threshold,
                threshold_window=threshold_window,
                macro_lag_periods=macro_lag_periods,
                anchor_returns=anchor_returns,
                random_seed=synthetic_seed,
                categories=category_map,
            )
        else:
            simulated_returns = simulate_pre_inception_returns(
                returns,
                assets=synthetic_assets,
                start=start_date,
                random_seed=synthetic_seed,
            )
        if not simulated_returns.empty:
            returns = combine_observed_and_simulated_returns(returns, simulated_returns)
        available = tuple(returns.columns)
    return macro, returns, available, synthetic_report


def load_market_data(
    tickers: Sequence[str],
    start: str,
    end: str,
    historical_proxies: Mapping[str, str] | None = None,
    synthetic_assets: Sequence[str] | str = (),
    synthetic_seed: int = 42,
    synthetic_method: str = "regime",
    synthetic_categories: Mapping[str, str] | None = None,
    growth_threshold: str | float = "median",
    inflation_threshold: str | float = "median",
    threshold_window: int | None = None,
    macro_lag_periods: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, dict[str, object]]]:
    """Load Yahoo prices/FRED macro data, optionally backfilling with proxies.

    ``synthetic_method="regime"`` generates pre-inception history conditioned on
    the actual historical macro regime at each date plus a factor model; the
    ``"full_sample"`` method uses the asset's observed moments only. The last
    return value is a per-asset synthetic feasibility report.
    """

    raw_tickers = (tickers,) if isinstance(tickers, str) else tickers
    normalized_tickers = tuple(dict.fromkeys(str(ticker).strip().upper() for ticker in raw_tickers))
    normalized_proxies: list[tuple[str, str]] = []
    for asset, proxy in (historical_proxies or {}).items():
        normalized_asset = str(asset).strip().upper()
        normalized_proxy = str(proxy).strip().upper()
        if normalized_asset not in normalized_tickers:
            raise ValueError(f"Historical proxy asset is not in the selected tickers: {normalized_asset}")
        if not normalized_proxy:
            raise ValueError(f"Historical proxy ticker is empty for asset: {normalized_asset}")
        normalized_proxies.append((normalized_asset, normalized_proxy))
    raw_synthetic_assets = (synthetic_assets,) if isinstance(synthetic_assets, str) else synthetic_assets
    normalized_synthetic_assets = tuple(
        dict.fromkeys(str(asset).strip().upper() for asset in raw_synthetic_assets if str(asset).strip())
    )
    missing_synthetic = [asset for asset in normalized_synthetic_assets if asset not in normalized_tickers]
    if missing_synthetic:
        raise ValueError(
            "Synthetic assets must be included in the selected tickers: "
            + ", ".join(missing_synthetic)
        )
    normalized_categories = tuple(
        (str(asset).strip().upper(), str(category).strip().upper())
        for asset, category in (synthetic_categories or {}).items()
    )

    macro, returns, available, synthetic_report = _load_market_data_cached(
        normalized_tickers,
        pd.Timestamp(start).strftime("%Y-%m-%d"),
        pd.Timestamp(end).strftime("%Y-%m-%d"),
        tuple(dict.fromkeys(normalized_proxies)),
        normalized_synthetic_assets,
        int(synthetic_seed),
        str(synthetic_method).lower(),
        normalized_categories,
        growth_threshold,
        inflation_threshold,
        threshold_window,
        int(macro_lag_periods),
    )
    return macro.copy(), returns.copy(), list(available), synthetic_report


def yoy_change(data: pd.DataFrame, periods: int = 12, scale: float = 100.0) -> pd.DataFrame:
    """Compute year-over-year percent changes for monthly-style macro data."""

    return data.pct_change(periods=periods) * scale
