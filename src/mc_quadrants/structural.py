from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

ASSET_CLASSES = {
    "equity",
    "government_bond",
    "corporate_bond",
    "inflation_linked_bond",
    "cash",
    "commodity",
    "gold",
    "real_estate",
    "managed_futures",
    "other",
}

_KNOWN_CLASSES = {
    "SPY": "equity",
    "QQQ": "equity",
    "EFA": "equity",
    "EEM": "equity",
    "VNQ": "real_estate",
    "IEF": "government_bond",
    "TLT": "government_bond",
    "SHY": "cash",
    "TIP": "inflation_linked_bond",
    "GLD": "gold",
    "DBC": "commodity",
    "DBMF": "managed_futures",
    "KMLM": "managed_futures",
}

_DEFAULT_DURATION = {
    "government_bond": 7.0,
    "corporate_bond": 5.0,
    "inflation_linked_bond": 6.5,
    "cash": 0.25,
}

_KNOWN_DURATION = {"IEF": 7.3, "TLT": 16.0, "SHY": 1.8, "TIP": 6.5}

_DEFAULT_INCOME_YIELD = {
    "equity": 0.018,
    "government_bond": 0.025,
    "corporate_bond": 0.035,
    "inflation_linked_bond": 0.020,
    "cash": 0.020,
    "real_estate": 0.035,
    "commodity": 0.0,
    "gold": 0.0,
    "managed_futures": 0.0,
    "other": 0.0,
}


def _base_symbol(asset: str) -> str:
    return str(asset).strip().upper().removesuffix("_SIM").removesuffix("SIM")


def infer_asset_class(asset: str) -> str:
    """Infer a conservative structural class from common symbols and names."""

    symbol = _base_symbol(asset)
    if symbol in _KNOWN_CLASSES:
        return _KNOWN_CLASSES[symbol]
    name = symbol.replace("-", "_")
    if any(token in name for token in ("BOND", "TREASURY", "GOVT")):
        return "government_bond"
    if any(token in name for token in ("CREDIT", "CORPORATE")):
        return "corporate_bond"
    if any(token in name for token in ("CASH", "BILL", "MONEY")):
        return "cash"
    if any(token in name for token in ("GOLD", "PRECIOUS")):
        return "gold"
    if any(token in name for token in ("COMMOD", "ENERGY")):
        return "commodity"
    if any(token in name for token in ("REIT", "REAL_ESTATE")):
        return "real_estate"
    return "other"


def build_asset_profiles(
    assets: list[str],
    asset_classes: Mapping[str, str] | None = None,
    durations: Mapping[str, float] | None = None,
    income_yields: Mapping[str, float] | None = None,
) -> dict[str, dict[str, Any]]:
    """Build validated structural profiles without requiring a security master."""

    class_map = {str(k).strip().upper(): str(v).strip().lower() for k, v in (asset_classes or {}).items()}
    duration_map = {str(k).strip().upper(): float(v) for k, v in (durations or {}).items()}
    yield_map = {str(k).strip().upper(): float(v) for k, v in (income_yields or {}).items()}
    profiles: dict[str, dict[str, Any]] = {}
    for asset in assets:
        key = str(asset).strip().upper()
        base = _base_symbol(asset)
        asset_class = class_map.get(key, class_map.get(base, infer_asset_class(asset)))
        if asset_class not in ASSET_CLASSES:
            allowed = ", ".join(sorted(ASSET_CLASSES))
            raise ValueError(f"Unknown asset class '{asset_class}' for {asset}. Expected one of: {allowed}.")
        duration = duration_map.get(key, duration_map.get(base, _KNOWN_DURATION.get(base, _DEFAULT_DURATION.get(asset_class, 0.0))))
        income_yield = yield_map.get(key, yield_map.get(base, _DEFAULT_INCOME_YIELD[asset_class]))
        if not np.isfinite(duration) or duration < 0 or duration > 50:
            raise ValueError(f"Duration for {asset} must be between 0 and 50 years.")
        if not np.isfinite(income_yield) or not -0.20 < income_yield < 1:
            raise ValueError(f"Income yield for {asset} must be a decimal between -20% and 100%.")
        profiles[asset] = {
            "asset_class": asset_class,
            "duration_years": float(duration),
            "income_yield": float(income_yield),
            "source": "override" if key in class_map or base in class_map else "inferred",
        }
    return profiles


def structural_beta_prior(
    assets: list[str],
    macro_columns: list[str],
    profiles: Mapping[str, Mapping[str, Any]],
    *,
    growth_col: str,
    inflation_col: str,
    rate_col: str | None,
    macro_is_percent: Mapping[str, bool],
) -> np.ndarray:
    """Return economically signed priors for macro-change return sensitivities.

    Coefficients map changes in the original macro units to monthly log returns.
    They are deliberately weak anchors; calibration data can move away from them.
    """

    prior = np.zeros((len(macro_columns), len(assets)), dtype=float)
    for asset_index, asset in enumerate(assets):
        profile = profiles[asset]
        asset_class = str(profile["asset_class"])
        duration = float(profile.get("duration_years", 0.0))
        for macro_index, column in enumerate(macro_columns):
            unit = 0.01 if macro_is_percent.get(column, False) else 1.0
            if column == rate_col:
                if asset_class in {"government_bond", "corporate_bond", "inflation_linked_bond", "cash"}:
                    prior[macro_index, asset_index] = -duration * unit
                elif asset_class in {"equity", "real_estate"}:
                    prior[macro_index, asset_index] = -0.75 * unit
            elif column == growth_col:
                if asset_class in {"equity", "real_estate", "corporate_bond"}:
                    prior[macro_index, asset_index] = 0.50 * unit
                elif asset_class in {"government_bond", "cash"}:
                    prior[macro_index, asset_index] = -0.10 * unit
            elif column == inflation_col:
                if asset_class in {"commodity", "gold", "inflation_linked_bond"}:
                    prior[macro_index, asset_index] = 0.45 * unit
                elif asset_class in {"government_bond", "real_estate"}:
                    prior[macro_index, asset_index] = -0.30 * unit
    return prior
