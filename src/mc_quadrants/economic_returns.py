"""Explicit economic return assumptions and a shared, persistent yield channel.

The yield input is an annual decimal after applying yield_scale. Duration is a
constant log-price sensitivity, not a full bond-pricing or yield-curve model.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import numpy as np
import pandas as pd

DEFAULTS = dict(
    return_anchors=True,
    rate_stress=True,
    anchor_strength=120.0,
    equity_real_return=0.05,
    inflation_anchor=0.02,
    anchor_std=0.02,
    equity_rate_duration=4.0,
    stress_inflation_threshold=0.03,
    stress_inflation_width=0.02,
    bond_yield_col="bond_yield",
    yield_scale=0.01,
    yield_model="adaptive",
    equity_rate_mode="surprise",
    macro_stress="none",
    asset_yield_columns={},
    asset_spread_columns={},
    credit_loss_rates={},
    equity_valuation_columns={},
    valuation_reference={},
    valuation_weight=0.25,
    valuation_half_life_years=10.0,
)
PREVIOUS_YIELD = "__economic_previous_yield"
PREVIOUS_INFLATION = "__economic_previous_inflation"


def economic_assumptions(options: Mapping | None) -> dict | None:
    if options is None:
        return None
    if not isinstance(options, Mapping) or set(options) - set(DEFAULTS):
        raise ValueError("economic_returns contains unsupported assumptions.")
    result = {**DEFAULTS, **options}
    for key in ("return_anchors", "rate_stress"):
        if not isinstance(result[key], bool):
            raise ValueError(f"{key} must be a boolean.")
    map_fields = {
        "asset_yield_columns",
        "asset_spread_columns",
        "credit_loss_rates",
        "equity_valuation_columns",
        "valuation_reference",
    }
    string_fields = {"bond_yield_col", "yield_model", "equity_rate_mode", "macro_stress"}
    for key in map_fields:
        if not isinstance(result[key], Mapping):
            raise ValueError(f"{key} must be an asset-to-value mapping.")
        result[key] = {str(asset).upper(): value for asset, value in result[key].items()}
        for asset, value in result[key].items():
            if key in {"credit_loss_rates", "valuation_reference"}:
                value = float(value)
                if not np.isfinite(value) or (value < 0 if key == "credit_loss_rates" else value <= 0):
                    raise ValueError(f"Invalid {key} for {asset}.")
                if key == "credit_loss_rates" and value >= 1:
                    raise ValueError("Annual credit losses must be below 100%.")
                result[key][asset] = value
            elif not isinstance(value, str) or not value or value.startswith("__"):
                raise ValueError(f"Invalid macro column for {asset}.")
    if result["yield_model"] not in {"adaptive", "random_walk", "legacy"}:
        raise ValueError("yield_model must be adaptive, random_walk, or legacy.")
    if result["equity_rate_mode"] not in {"surprise", "inflation_gate"}:
        raise ValueError("equity_rate_mode must be surprise or inflation_gate.")
    if not isinstance(result["macro_stress"], str) or result["macro_stress"] not in {"none", "persistent", "volatility", "combined"}:
        raise ValueError("macro_stress must be none, persistent, volatility, or combined.")
    if result["macro_stress"] != "none" and not result["rate_stress"]:
        raise ValueError("macro_stress requires rate_stress and joint macro paths.")
    for key in set(DEFAULTS) - {"return_anchors", "rate_stress"} - map_fields - string_fields:
        if isinstance(result[key], bool):
            raise ValueError(f"{key} must be numeric.")
        result[key] = float(result[key])
        if not np.isfinite(result[key]):
            raise ValueError(f"{key} must be finite.")
    for key in ("anchor_strength", "anchor_std", "equity_rate_duration"):
        if result[key] < 0:
            raise ValueError(f"{key} must be non-negative.")
    if result["anchor_std"] > 1:
        raise ValueError("anchor_std must not exceed one annual log-return unit.")
    if not -0.5 < result["equity_real_return"] < 0.5 or not -0.1 < result["inflation_anchor"] < 0.5:
        raise ValueError("Economic return/inflation anchors are outside supported bounds.")
    if result["stress_inflation_width"] <= 0 or result["yield_scale"] not in (0.01, 1.0):
        raise ValueError("Stress width must be positive; yield_scale must be 0.01 (percent) or 1 (decimal).")
    if not 0 <= result["valuation_weight"] <= 1 or result["valuation_half_life_years"] <= 0:
        raise ValueError("Valuation weight must be in [0, 1] and half-life positive.")
    column = result["bond_yield_col"]
    if not isinstance(column, str) or not column.strip() or column.startswith("__"):
        raise ValueError("bond_yield_col must name a public macro column.")
    return result if result["return_anchors"] or result["rate_stress"] else None


def structural_returns(previous_yield, current_yield, previous_inflation, profiles, options):
    """Carry plus duration repricing; inflation gates the equity rate exposure.

    All arrays use decimal annual rates and share a shape. The final axis of
    the result enumerates profiles in insertion order. The gate uses lagged
    inflation, avoiding a look-ahead trigger in historical decomposition.
    """
    previous_yield, current_yield, previous_inflation = np.broadcast_arrays(
        previous_yield,
        current_yield,
        previous_inflation,
    )
    if not all(np.isfinite(v).all() for v in (previous_yield, current_yield, previous_inflation)):
        raise ValueError("Economic factors must be finite.")
    if (previous_yield <= -1).any() or (current_yield <= -1).any():
        raise ValueError("Yield must exceed -100%; no silent clipping is applied.")
    change = current_yield - previous_yield
    gate = np.clip(
        (previous_inflation - options["stress_inflation_threshold"]) / options["stress_inflation_width"], 0, 1
    )
    result = np.zeros((*change.shape, len(profiles)))
    for index, profile in enumerate(profiles.values()):
        kind = profile["asset_class"]
        if kind in {"government_bond", "corporate_bond", "inflation_linked_bond"}:
            result[..., index] = np.log1p(previous_yield) / 12 - profile["duration_years"] * change
        elif kind == "cash":
            result[..., index] = np.log1p(previous_yield) / 12
        elif kind == "equity" and options["equity_rate_mode"] == "inflation_gate":
            result[..., index] = -options["equity_rate_duration"] * gate * change
    return result


def asset_channels(profiles, options):
    channels = {}
    for asset, profile in profiles.items():
        kind, duration = profile["asset_class"], profile["duration_years"]
        default = (
            "interest_rate"
            if kind == "cash"
            else "real_bond_yield"
            if kind == "inflation_linked_bond"
            else "bond_yield_2y"
            if duration <= 3.5
            else "bond_yield_20y"
            if duration >= 12
            else options["bond_yield_col"]
        )
        if kind == "equity":
            default = options["bond_yield_col"]
        if kind not in {"equity", "government_bond", "cash", "corporate_bond", "inflation_linked_bond"}:
            continue
        column = options["asset_yield_columns"].get(asset.upper(), default)
        spread = options["asset_spread_columns"].get(asset.upper())
        if kind == "corporate_bond":
            if not spread or asset.upper() not in options["credit_loss_rates"]:
                raise ValueError(
                    f"{asset} requires an explicit spread column and annual credit-loss assumption."
                )
        channels[asset] = dict(
            yield_column=column,
            spread_column=spread,
            credit_loss=options["credit_loss_rates"].get(asset.upper(), 0.0),
        )
    return channels


def economic_columns(profiles, options):
    channels = asset_channels(profiles, options)
    columns = set()
    for asset, channel in channels.items():
        if profiles[asset]["asset_class"] == "equity" and not options["rate_stress"]:
            continue
        columns.add(channel["yield_column"])
        if channel["spread_column"]:
            columns.add(channel["spread_column"])
    return sorted(columns)


def previous_column(column):
    return "__economic_previous_" + column


def prepare_economic_history(returns, macro, profiles, options, inflation_col):
    """Fit residual returns on contemporaneous, maturity-matched physical factors."""
    channels = asset_channels(profiles, options)
    columns = economic_columns(profiles, options)
    missing = set(columns) - set(macro)
    if missing:
        raise ValueError(
            f"Economic returns require observed {sorted(missing)} yields/spreads; policy rates are not substituted."
        )
    scale = options["yield_scale"]
    observed = {column: pd.to_numeric(macro[column], errors="coerce") * scale for column in columns}
    lagged = {
        column: pd.to_numeric(macro[previous_column(column)], errors="coerce") * scale
        if previous_column(column) in macro
        else observed[column].shift(1)
        for column in columns
    }
    # Compatibility with previously serialized/bootstrap seven-year inputs.
    if PREVIOUS_YIELD in macro and options["bond_yield_col"] in observed:
        lagged[options["bond_yield_col"]] = pd.to_numeric(macro[PREVIOUS_YIELD], errors="coerce") * scale
    inflation = pd.to_numeric(macro[inflation_col], errors="coerce")
    inflation_scale = 0.01 if inflation.abs().quantile(0.9) >= 0.5 else 1.0
    previous_inflation = (
        macro[PREVIOUS_INFLATION] if PREVIOUS_INFLATION in macro else inflation.shift(1)
    ) * inflation_scale
    latest = {}
    for column, values in observed.items():
        if values.dropna().empty or not np.isfinite(values.dropna()).all():
            raise ValueError(f"{column} must contain finite yield/spread observations.")
        latest[column] = float(values.dropna().iloc[-1])
    adjusted = returns.copy()
    carry = np.zeros(len(profiles))
    for index, (asset, profile) in enumerate(profiles.items()):
        if asset not in channels or (profile["asset_class"] == "equity" and not options["rate_stress"]):
            continue
        channel = channels[asset]
        column, spread = channel["yield_column"], channel["spread_column"]
        current = observed[column] + (observed[spread] if spread else 0)
        previous = lagged[column] + (lagged[spread] if spread else 0)
        total = latest[column] + (latest[spread] if spread else 0)
        if total <= -1:
            raise ValueError("Yield must exceed -100%.")
        if options["rate_stress"]:
            panel = (
                pd.concat(
                    [
                        current.rename("current"),
                        previous.rename("previous"),
                        previous_inflation.rename("inflation"),
                    ],
                    axis=1,
                )
                .reindex(returns.index)
                .dropna()
            )
            factor = structural_returns(
                panel.previous, panel.current, panel.inflation, {asset: profile}, options
            )[:, 0]
            if profile["asset_class"] == "corporate_bond":
                factor += np.log1p(-channel["credit_loss"]) / 12
            if profile["asset_class"] == "inflation_linked_bond":
                if "tips_inflation_log" not in macro or "cpi_monthly_log" not in macro:
                    raise ValueError(
                        "TIPS require cpi_monthly_log and tips_inflation_log: observed CPI changes and lagged indexation."
                    )
                factor += macro["tips_inflation_log"].reindex(panel.index).to_numpy()
            adjusted[asset] -= pd.Series(factor, index=panel.index)
            if profile["asset_class"] != "equity":
                carry[index] = np.log1p(total) / 12 + np.log1p(-channel["credit_loss"]) / 12
    adjusted = adjusted.dropna()
    if len(adjusted) < 24:
        raise ValueError("Economic returns require at least 24 complete paired observations.")
    valuation = {}
    for asset, column in options["equity_valuation_columns"].items():
        if asset not in profiles or profiles[asset]["asset_class"] != "equity":
            raise ValueError(f"Valuation override {asset} must identify an equity asset.")
        if column not in macro:
            raise ValueError(f"Missing observed valuation column {column}.")
        values = pd.to_numeric(macro[column], errors="coerce").dropna()
        if values.empty or not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError("Valuation multiples must be finite and positive.")
        reference = options["valuation_reference"].get(asset, float(values.median()))
        correction = (
            options["valuation_weight"]
            * np.log(2)
            / options["valuation_half_life_years"]
            * np.log(reference / values.iloc[-1])
        )
        valuation[asset] = dict(
            column=column,
            latest=float(values.iloc[-1]),
            reference=reference,
            annual_log_adjustment=float(correction),
        )
    return adjusted, dict(
        assumptions=options,
        latest_yield=latest.get(options["bond_yield_col"], 0.0),
        latest_yields=latest,
        channels=channels,
        valuation=valuation,
        latest_inflation=float(inflation.dropna().iloc[-1]) * inflation_scale,
        inflation_scale=inflation_scale,
        current_carry=carry.tolist(),
        yield_source=columns,
        yield_is_common_reference=len(columns) == 1,
        basis="residual_fitted_after_physical_factors" if options["rate_stress"] else "anchored_total_return",
    )


def anchor_moments(moments, returns, profiles, metadata):
    """Shrink the level of the mean; preserve estimated regime differences."""
    options = metadata["assumptions"]
    means = returns.mean()
    targets = means.copy()
    for asset, profile in profiles.items():
        if profile["asset_class"] == "equity":
            targets[asset] = (
                np.log1p(options["equity_real_return"]) + np.log1p(options["inflation_anchor"])
            ) / 12 + metadata.get("valuation", {}).get(asset, {}).get("annual_log_adjustment", 0.0) / 12
        elif profile["asset_class"] in {"government_bond", "corporate_bond", "inflation_linked_bond", "cash"}:
            channel = metadata.get("channels", {}).get(asset)
            level = (
                metadata["latest_yield"]
                if channel is None
                else metadata["latest_yields"][channel["yield_column"]]
                + (metadata["latest_yields"][channel["spread_column"]] if channel["spread_column"] else 0)
            )
            target = np.log1p(level) / 12
            if channel:
                target += np.log1p(-channel["credit_loss"]) / 12
            if profile["asset_class"] == "inflation_linked_bond":
                target += np.log1p(options["inflation_anchor"]) / 12
            targets[asset] = 0.0 if options["rate_stress"] else target
    weight = options["anchor_strength"] / (len(returns) + options["anchor_strength"])
    shift = weight * (targets - means) if options["return_anchors"] else means * 0
    metadata.update(
        anchor_weight=weight if options["return_anchors"] else 0.0,
        monthly_prior_targets=targets.to_dict(),
        monthly_mean_shift=shift.to_dict(),
    )
    return {state: replace(moment, mean=moment.mean + shift) for state, moment in moments.items()}


def economic_path_adjustment(model, macro_paths):
    """Reinsert the fitted structural channel using the actual simulated levels."""
    metadata = model.metadata["economic_returns"]
    options = metadata["assumptions"]
    dynamics = model.metadata["macro_dynamics"]
    columns = dynamics["columns"]
    inflation = macro_paths[..., columns.index(model.metadata["inflation_col"])] * metadata["inflation_scale"]
    previous_inflation = np.vstack(
        [np.full((1, inflation.shape[1]), metadata["latest_inflation"]), inflation[:-1]]
    )
    result = np.zeros((*inflation.shape, len(model.assets)))
    for index, asset in enumerate(model.assets):
        profile = model.metadata["asset_profiles"][asset]
        if asset not in metadata["channels"]:
            continue
        channel = metadata["channels"][asset]
        column, spread = channel["yield_column"], channel["spread_column"]
        yields = macro_paths[..., columns.index(column)] * options["yield_scale"]
        latest = metadata["latest_yields"][column]
        if spread:
            yields = yields + macro_paths[..., columns.index(spread)] * options["yield_scale"]
            latest += metadata["latest_yields"][spread]
        previous = np.vstack([np.full((1, yields.shape[1]), latest), yields[:-1]])
        result[..., index] = structural_returns(
            previous, yields, previous_inflation, {asset: profile}, options
        )[..., 0]
        if profile["asset_class"] == "corporate_bond":
            result[..., index] += np.log1p(-channel["credit_loss"]) / 12
        elif profile["asset_class"] == "inflation_linked_bond":
            from mc_quadrants.cpi import monthly_cpi_from_yoy

            cpi = model.metadata["cpi_process"]
            changes = monthly_cpi_from_yoy(inflation, cpi["monthly_log_history"])
            lag = 3 - cpi.get("observation_lag_months", 0)
            if not 0 <= lag <= 3:
                raise ValueError("TIPS CPI observation lag must be between zero and three months.")
            historical = np.broadcast_to(
                np.asarray(cpi["monthly_log_history"])[12 - lag :, None], (lag, changes.shape[1])
            )
            result[..., index] += np.concatenate([historical, changes], axis=0)[: len(changes)]
    return result - np.asarray(metadata["current_carry"])
