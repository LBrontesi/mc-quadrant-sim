"""Regime-conditioned synthetic backfill for pre-inception ETF history.

Reconstructs a plausible pre-inception return history for an asset using the
actual historical macro regime at each date (growth/inflation quadrant) plus a
factor model estimated on the asset's observed history, in the spirit of
backtesters that provide simulated series such as ``DBMFSIM`` before a fund's
inception.

The generated series is an explicit approximation, not a claim that the ETF
existed earlier. Each asset receives a feasibility grade (A/B/C/X) describing
how much of its synthetic history rests on observed behavior versus projection.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Real

import numpy as np
import pandas as pd

from mc_quadrants.mnts import (
    DEFAULT_TAIL_INDEX,
    DEFAULT_TEMPERING,
    sample_mnts_subordinators,
)
from mc_quadrants.regimes import (
    REGIME_ORDER,
    ThresholdSpec,
    classify_persistent_quadrants,
)

ASSET_CATEGORIES: dict[str, str] = {
    "EQUITY": "Equity",
    "LONG_TERM_BOND": "Long-term bond",
    "SHORT_TERM_BOND": "Short-term bond",
    "INTERNATIONAL_EQUITY": "International equity",
    "GOLD": "Gold",
    "COMMODITIES": "Commodities",
    "REAL_ESTATE": "Real estate",
    "TIPS": "TIPS",
    "MANAGED_FUTURES": "Managed futures",
    "UNCATEGORIZED": "Uncategorized",
}

DEFAULT_TICKER_CATEGORIES: dict[str, str] = {
    "SPY": "EQUITY",
    "QQQ": "EQUITY",
    "IEF": "LONG_TERM_BOND",
    "TLT": "LONG_TERM_BOND",
    "SHY": "SHORT_TERM_BOND",
    "EFA": "INTERNATIONAL_EQUITY",
    "GLD": "GOLD",
    "DBC": "COMMODITIES",
    "VNQ": "REAL_ESTATE",
    "TIP": "TIPS",
    "DBMF": "MANAGED_FUTURES",
    "KMLM": "MANAGED_FUTURES",
}

DEFAULT_ANCHOR_UNIVERSE: list[str] = ["SPY", "IEF", "GLD", "DBC", "EFA", "VNQ", "TIP", "SHY"]

_MIN_OBSERVATIONS_DEFAULT = 12
_CAUSAL_WINDOW_DEFAULT = 36


def categorize_asset(ticker: str, overrides: Mapping[str, str] | None = None) -> str:
    """Return the category code for an asset, applying user overrides first."""

    key = str(ticker).strip().upper().removesuffix("_SIM").removesuffix("SIM")
    if overrides and key in overrides:
        return str(overrides[key]).strip().upper()
    return DEFAULT_TICKER_CATEGORIES.get(key, "UNCATEGORIZED")


def category_label(category: str) -> str:
    """Human-readable label for a category code."""

    return ASSET_CATEGORIES.get(str(category).upper(), str(category))


def backward_price_levels(
    synthetic_log_returns: pd.Series,
    first_observed_price: float,
) -> pd.Series:
    """Reconstruct backward price levels from synthetic log returns.

    Each pre-inception price is anchored to the first observed price:

        P_t = P_first * exp(-sum of synthetic log returns between t and inception)
    """

    if synthetic_log_returns.empty:
        return pd.Series(dtype=float, name="price")
    if not np.isfinite(first_observed_price) or first_observed_price <= 0:
        raise ValueError("first_observed_price must be positive and finite.")
    values = synthetic_log_returns.sort_index().to_numpy(dtype=float)
    suffix_sums = np.flip(np.cumsum(np.flip(values)))
    levels = first_observed_price * np.exp(-suffix_sums)
    return pd.Series(levels, index=synthetic_log_returns.sort_index().index, name="price")


def _mnts_noise(rng: np.random.Generator, mean: float, std: float) -> float:
    """Draw one standardized skewed, fat-tailed MNTS residual."""

    if not np.isfinite(std) or std <= 0:
        return float(mean)
    subordinator = sample_mnts_subordinators(
        rng,
        1,
        DEFAULT_TAIL_INDEX,
        DEFAULT_TEMPERING,
    )[0]
    skewness = -0.35
    variance_t = (2.0 - DEFAULT_TAIL_INDEX) / (2.0 * DEFAULT_TEMPERING)
    gaussian_scale = np.sqrt(1.0 - skewness * skewness * variance_t)
    innovation = skewness * (subordinator - 1.0)
    innovation += np.sqrt(subordinator) * gaussian_scale * rng.standard_normal()
    return float(mean + std * innovation)


def _fit_factor_model(
    asset_returns: pd.Series,
    anchor_returns: pd.DataFrame | None,
    min_observations: int,
) -> tuple[np.ndarray | None, float | None, float | None, int]:
    """Fit r_asset = alpha + beta @ r_anchors + epsilon on the overlap window."""

    if anchor_returns is None or anchor_returns.empty:
        return None, None, None, 0
    aligned = pd.concat(
        [asset_returns.rename("y"), anchor_returns.loc[asset_returns.index]],
        axis=1,
    ).dropna()
    if len(aligned) < min_observations:
        return None, None, None, int(len(aligned))
    y = aligned["y"].to_numpy(dtype=float)
    x = aligned.drop(columns=["y"])
    if x.shape[1] == 0:
        return None, None, None, int(len(aligned))
    design = np.column_stack([np.ones(len(y)), x.to_numpy(dtype=float)])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ beta
    ss_total = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(residual**2)) / ss_total if ss_total > 0 else 0.0
    residual_vol = float(residual.std(ddof=1)) if len(residual) > 1 else float(np.abs(residual).mean() or 0.0)
    return beta, r2, residual_vol, int(len(aligned))


def _grade(history_months: int, r2: float | None, covered_regimes: int) -> str:
    if history_months < 12:
        return "X"
    if history_months < 24:
        return "C"
    if r2 is None:
        return "C"
    if r2 >= 0.5 and covered_regimes == len(REGIME_ORDER):
        return "A"
    if r2 >= 0.25 and covered_regimes >= 1:
        return "B"
    return "C"


def _report_entry(
    asset: str,
    category: str,
    history_months: int,
    counts: Mapping[str, int],
    r2: float | None,
    residual_vol: float | None,
    grade: str,
    warnings: Sequence[str],
) -> dict[str, object]:
    return {
        "asset": asset,
        "grade": grade,
        "category": category,
        "history_months": int(history_months),
        "observations_by_regime": {str(key): int(value) for key, value in counts.items()},
        "factor_r2": None if r2 is None else float(r2),
        "factor_residual_vol": None if residual_vol is None else float(residual_vol),
        "warnings": list(warnings),
    }


def simulate_regime_conditioned_pre_inception_returns(
    returns: pd.DataFrame,
    macro: pd.DataFrame,
    assets: Sequence[str],
    growth_col: str = "growth",
    inflation_col: str = "inflation",
    growth_threshold: ThresholdSpec = "median",
    inflation_threshold: ThresholdSpec = "median",
    macro_lag_periods: int = 1,
    threshold_window: int | None = None,
    anchor_returns: pd.DataFrame | None = None,
    min_observations: int = _MIN_OBSERVATIONS_DEFAULT,
    random_seed: int = 42,
    categories: Mapping[str, str] | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    """Generate regime-conditioned pre-inception returns for each synthetic asset.

    For every pre-inception month the actual historical growth/inflation regime
    is determined (with a causal threshold window for data-driven thresholds).
    The asset's behavior in that month is a factor-model projection using real
    historical anchor returns plus a regime-specific residual when the asset
    has enough observed history in that regime, falling back to regime moments
    or the observed sample otherwise.

    Returns a frame of ``{ASSET}_SIM`` log returns and a feasibility report per
    asset. The series is synthetic and not the asset's actual historical NAV.
    """

    numeric_both = isinstance(growth_threshold, Real) and isinstance(inflation_threshold, Real)
    classification_window = None if numeric_both else int(threshold_window or _CAUSAL_WINDOW_DEFAULT)
    regimes = classify_persistent_quadrants(
        macro,
        growth_col=growth_col,
        inflation_col=inflation_col,
        growth_threshold=growth_threshold,
        inflation_threshold=inflation_threshold,
        threshold_window=classification_window,
        smoothing_window=3,
        hysteresis=0.15,
        confirmation_periods=2,
    )
    governing = regimes.sort_index().shift(macro_lag_periods)

    generation_index = (
        anchor_returns.index if anchor_returns is not None and not anchor_returns.empty else returns.index
    )
    governing_full = governing.reindex(generation_index, method="ffill")

    rng = np.random.default_rng(random_seed)
    simulated_frames: list[pd.DataFrame] = []
    report: dict[str, dict[str, object]] = {}

    for asset in assets:
        observed = returns[asset].dropna().sort_index()
        model_anchors = (
            anchor_returns.drop(columns=[asset], errors="ignore")
            if anchor_returns is not None
            else None
        )
        category = category_label(categorize_asset(asset, overrides=categories))
        if observed.empty or len(observed) < min_observations:
            report[asset] = _report_entry(
                asset, category, int(len(observed)), {}, None, None, "X",
                ["Not enough observed history to backfill this asset."],
            )
            continue
        first_observed = observed.index[0]
        asset_regimes = governing.reindex(observed.index, method="ffill").dropna()
        regime_counts = {
            state: int((asset_regimes == state).sum()) for state in REGIME_ORDER
        }

        regime_mean: dict[str, float] = {}
        regime_std: dict[str, float] = {}
        for state in REGIME_ORDER:
            mask = asset_regimes == state
            values = observed.loc[mask]
            if int(mask.sum()) >= min_observations:
                regime_mean[state] = float(values.mean())
                regime_std[state] = float(values.std(ddof=1)) if len(values) > 1 else 0.0

        beta, r2, residual_vol, _ = _fit_factor_model(observed, model_anchors, min_observations)
        residual_by_regime: dict[str, tuple[float, float]] = {}
        full_residual_vol = 0.0
        if beta is not None:
            aligned = model_anchors.loc[observed.index].dropna()
            if len(aligned) >= min_observations:
                y = observed.loc[aligned.index].to_numpy(dtype=float)
                design = np.column_stack([np.ones(len(y)), aligned.to_numpy(dtype=float)])
                residuals = y - design @ beta
                full_residual_vol = float(residuals.std(ddof=1)) if len(residuals) > 1 else 0.0
                aligned_regimes = asset_regimes.reindex(aligned.index).dropna()
                for state in REGIME_ORDER:
                    mask = aligned_regimes == state
                    indices = np.flatnonzero(mask.to_numpy())
                    if len(indices) >= min_observations:
                        residual_by_regime[state] = (
                            float(residuals[indices].mean()),
                            float(residuals[indices].std(ddof=1)) if len(indices) > 1 else 0.0,
                        )

        pre_inception_dates = [date for date in generation_index if date < first_observed]
        generated: dict[pd.Timestamp, float] = {}
        for date in pre_inception_dates:
            state = governing_full.get(date)
            if state is None or pd.isna(state):
                continue
            state = str(state)
            if (
                beta is not None
                and model_anchors is not None
                and date in model_anchors.index
                and model_anchors.loc[date].notna().all()
            ):
                factor_part = float(beta[0] + beta[1:] @ model_anchors.loc[date].to_numpy(dtype=float))
                if state in residual_by_regime:
                    mean, std = residual_by_regime[state]
                else:
                    mean, std = 0.0, full_residual_vol
                generated[date] = _mnts_noise(rng, factor_part + mean, std)
            elif state in regime_mean:
                generated[date] = _mnts_noise(rng, regime_mean[state], regime_std[state])
            else:
                generated[date] = _mnts_noise(
                    rng,
                    float(observed.mean()),
                    float(observed.std(ddof=1) or 0.0),
                )

        simulated_series = pd.Series(generated, dtype=float).sort_index()
        if not simulated_series.empty:
            simulated_frames.append(simulated_series.to_frame(f"{asset}_SIM"))

        covered_regimes = sum(
            1 for count in regime_counts.values() if count >= min_observations
        )
        grade = _grade(int(len(observed)), r2, covered_regimes)
        warnings: list[str] = []
        if beta is None:
            warnings.append("Factor model could not be estimated; regime moments and the observed sample were used.")
        if r2 is not None and r2 < 0.25:
            warnings.append("Factor model has low explanatory power; synthetic history is a weak approximation.")
        if covered_regimes < len(REGIME_ORDER):
            warnings.append("Not all regimes were observed in the asset's own history; missing regimes are projected.")
        warnings.append("Factor behavior is estimated on the observed window and applied to pre-inception anchor returns.")
        report[asset] = _report_entry(
            asset, category, int(len(observed)), regime_counts, r2, residual_vol, grade, warnings
        )

    if not simulated_frames:
        return pd.DataFrame(), report
    simulated = pd.concat(simulated_frames, axis=1)
    return simulated, report
