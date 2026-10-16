"""Exact CPI accounting from trailing inflation and an observed price history."""

from __future__ import annotations

import numpy as np
import pandas as pd


def calibrate_cpi_history(macro, inflation_col):
    """Keep only information supplied on/before the forecast information date.

    cpi_monthly_log is a monthly log price change, aligned to the same information
    calendar as inflation. Without it we disclose a flat historical allocation,
    rather than treating future trailing inflation as a monthly rate.
    """
    values = pd.to_numeric(macro[inflation_col], errors="coerce").dropna()
    scale = 0.01 if values.abs().quantile(0.9) >= 0.5 else 1.0
    latest = float(values.iloc[-1]) * scale
    if latest <= -1 or not np.isfinite(latest):
        raise ValueError("CPI inflation must be finite and greater than -100%.")
    monthly = macro.get("cpi_monthly_log")
    history = None
    if monthly is not None:
        monthly = pd.to_numeric(monthly, errors="coerce").loc[: values.index[-1]]
        tail = monthly.iloc[-12:]
        contiguous = (
            isinstance(tail.index, pd.DatetimeIndex)
            and tail.index.to_period("M").equals(
                pd.period_range(tail.index[0].to_period("M"), periods=12, freq="M")
            )
            if len(tail) == 12
            else False
        )
        if contiguous and np.isfinite(tail).all():
            history = tail.to_numpy(dtype=float)
            if (
                not np.isclose(history.sum(), np.log1p(latest), atol=0.002)
                and "valid_macro_transitions" in macro.attrs
            ):
                history = None
            elif not np.isclose(history.sum(), np.log1p(latest), atol=0.002):
                raise ValueError(
                    "CPI monthly changes and latest year-over-year inflation use inconsistent calendars/units."
                )
    source = "observed_monthly_changes" if history is not None else "flat_history_assumption"
    if history is None:
        history = np.full(12, np.log1p(latest) / 12)
    observation_dates = macro.attrs.get("observation_dates")
    lag = 0
    if isinstance(observation_dates, pd.Series):
        known = observation_dates.loc[: values.index[-1]].dropna()
        if not known.empty:
            lag = (
                values.index[-1].to_period("M").ordinal - pd.Timestamp(known.iloc[-1]).to_period("M").ordinal
            )
    return dict(
        monthly_log_history=history.tolist(),
        latest_yoy=latest,
        scale=scale,
        history_source=source,
        observation_lag_months=lag,
        information_date=str(values.index[-1]),
        accounting="exact_trailing_twelve_month_identity",
    )


def monthly_cpi_from_yoy(yoy, history):
    """Invert log(1+YoY[t]) = sum(monthly_log[t-11:t+1]) exactly."""
    annual = np.asarray(yoy, dtype=float)
    if annual.ndim != 2 or not np.isfinite(annual).all() or (annual <= -1).any():
        raise ValueError("Year-over-year CPI paths must be finite, two-dimensional and exceed -100%.")
    initial = np.asarray(history, dtype=float)
    if initial.shape != (12,) or not np.isfinite(initial).all():
        raise ValueError("CPI initialization requires twelve finite monthly log changes.")
    ring = np.broadcast_to(initial[:, None], (12, annual.shape[1])).copy()
    rolling = ring.sum(axis=0)
    result = np.empty_like(annual)
    for t, target in enumerate(np.log1p(annual)):
        slot = t % 12
        change = target - rolling + ring[slot]
        result[t] = change
        ring[slot] = change
        rolling = target
    return result


def accounting_inflation_paths(model, result):
    """Return annualized monthly CPI rates for the existing accounting interface.

    Financing and macro regimes still consume trailing inflation separately.
    """
    column = model.metadata.get("inflation_col", "inflation")
    if result.macro_paths is None or column not in result.macro_columns:
        return None
    values = result.macro_paths[..., result.macro_columns.index(column)]
    cpi = model.metadata.get("cpi_process")
    if cpi is None:
        scale = 0.01 if model.metadata.get("macro_dynamics", {}).get("inflation_is_percent") else 1.0
        return values * scale
    changes = monthly_cpi_from_yoy(values * cpi["scale"], cpi["monthly_log_history"])
    with np.errstate(over="raise", invalid="raise"):
        rates = np.expm1(12 * changes)
    if not np.isfinite(rates).all() or (rates <= -1).any():
        raise ValueError("Simulated CPI changes exceed numerical accounting bounds; no clipping is applied.")
    return rates
