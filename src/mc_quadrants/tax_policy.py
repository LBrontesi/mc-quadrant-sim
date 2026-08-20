from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import pandas as pd

from mc_quadrants.taxes import (
    ITALY_DEFAULT_WEALTH_TAX_RATE,
    ITALY_GOVERNMENT_BOND_RATE,
    ITALY_LOSS_CARRY_YEARS,
    ITALY_STANDARD_TAX_RATE,
    ITALY_TAX_CATEGORIES,
    ITALY_TAX_REGIMES,
    ITALY_WEALTH_TAX_MODES,
    simulate_italian_portfolio_tax,
)


@dataclass(frozen=True)
class TaxSimulationContext:
    """Country-neutral inputs supplied to an optional portfolio tax policy."""

    asset_growth: np.ndarray
    assets: list[str]
    target_weights: np.ndarray
    initial_value: float
    rebalance_frequency: int
    transaction_cost_bps: float
    transaction_cost_rate_paths: np.ndarray | None
    contribution: float
    contribution_allocation: str
    withdrawal: float
    withdrawal_start_period: int
    decumulation: Any
    withdrawal_inflation_paths: np.ndarray | None
    annual_inflation: float
    safe_withdrawal_rate: float
    asset_categories: Mapping[str, str] | None
    asset_metadata: Mapping[str, Mapping[str, object]] | None
    annual_wealth_tax: float
    wealth_tax_mode: str
    terminal_liquidation: bool
    start_date: str | None
    wrapper_benchmark: bool
    native_threads: int


class TaxPolicy(Protocol):
    """Extension point for path-dependent country tax accounting."""

    country_code: str
    regime: str
    country_label: str
    regime_label: str

    def validate(
        self,
        *,
        rebalance_frequency: int | None,
        leverage_multiple: float,
        target_weights: np.ndarray,
    ) -> None: ...

    def simulate(self, context: TaxSimulationContext) -> pd.DataFrame: ...

    def metadata(self, context: TaxSimulationContext | None = None) -> dict[str, Any]: ...


@dataclass(frozen=True)
class TaxSelection:
    country: str
    regime: str
    policy: TaxPolicy | None

    @property
    def enabled(self) -> bool:
        return self.policy is not None


@dataclass(frozen=True)
class ItalyAdministeredTaxPolicy:
    country_code: str = "IT"
    regime: str = "italy_administered"
    country_label: str = "Italy"
    regime_label: str = "Simplified administered regime"

    def validate(
        self,
        *,
        rebalance_frequency: int | None,
        leverage_multiple: float,
        target_weights: np.ndarray,
    ) -> None:
        if rebalance_frequency is None:
            raise ValueError(
                "Italian tax accounting requires holdings-based accounting, not legacy weighted returns."
            )
        if not np.isclose(leverage_multiple, 1.0):
            raise ValueError("Italian tax accounting is not available with leveraged portfolios.")
        if (np.asarray(target_weights, dtype=float) < 0).any():
            raise ValueError("Italian tax accounting requires non-negative portfolio weights.")

    def simulate(self, context: TaxSimulationContext) -> pd.DataFrame:
        return simulate_italian_portfolio_tax(
            context.asset_growth,
            assets=context.assets,
            target_weights=context.target_weights,
            initial_value=context.initial_value,
            rebalance_frequency=context.rebalance_frequency,
            transaction_cost_bps=context.transaction_cost_bps,
            transaction_cost_rate_paths=context.transaction_cost_rate_paths,
            contribution=context.contribution,
            contribution_allocation=context.contribution_allocation,
            withdrawal=context.withdrawal,
            withdrawal_start_period=context.withdrawal_start_period,
            decumulation=context.decumulation,
            withdrawal_inflation_paths=context.withdrawal_inflation_paths,
            annual_inflation=context.annual_inflation,
            safe_withdrawal_rate=context.safe_withdrawal_rate,
            asset_tax_categories=context.asset_categories,
            asset_tax_metadata=context.asset_metadata,
            annual_wealth_tax=context.annual_wealth_tax,
            terminal_liquidation=context.terminal_liquidation,
            tax_regime=self.regime,
            wealth_tax_mode=context.wealth_tax_mode,
            start_date=context.start_date,
            wrapper_benchmark=context.wrapper_benchmark,
            native_threads=context.native_threads,
        )

    def metadata(self, context: TaxSimulationContext | None = None) -> dict[str, Any]:
        return {
            "country": self.country_code,
            "country_label": self.country_label,
            "regime": self.regime,
            "label": f"{self.country_label} — {self.regime_label.lower()}",
            "standard_rate": ITALY_STANDARD_TAX_RATE,
            "government_bond_rate": ITALY_GOVERNMENT_BOND_RATE,
            "annual_wealth_tax_rate": (
                context.annual_wealth_tax
                if context is not None
                else ITALY_DEFAULT_WEALTH_TAX_RATE
            ),
            "terminal_liquidation": (
                context.terminal_liquidation if context is not None else False
            ),
            "loss_carry_years": ITALY_LOSS_CARRY_YEARS,
            "rule_snapshot": "IT-2026",
            "supported_asset_categories": sorted(ITALY_TAX_CATEGORIES),
            "supported_regimes": sorted(ITALY_TAX_REGIMES),
            "supported_wealth_tax_modes": sorted(ITALY_WEALTH_TAX_MODES),
        }


ITALY_ADMINISTERED_POLICY = ItalyAdministeredTaxPolicy()
ITALY_DECLARATIVE_POLICY = ItalyAdministeredTaxPolicy(
    regime="italy_declarative",
    regime_label="Declarative regime",
)
ITALY_MANAGED_POLICY = ItalyAdministeredTaxPolicy(
    regime="italy_managed",
    regime_label="Managed regime",
)
TAX_POLICY_REGISTRY: dict[tuple[str, str], TaxPolicy] = {
    (ITALY_ADMINISTERED_POLICY.country_code, ITALY_ADMINISTERED_POLICY.regime): (
        ITALY_ADMINISTERED_POLICY
    ),
    (ITALY_DECLARATIVE_POLICY.country_code, ITALY_DECLARATIVE_POLICY.regime): ITALY_DECLARATIVE_POLICY,
    (ITALY_MANAGED_POLICY.country_code, ITALY_MANAGED_POLICY.regime): ITALY_MANAGED_POLICY,
}
TAX_COUNTRY_ALIASES = {
    "IT": "IT",
    "ITALY": "IT",
    "ITALIA": "IT",
}
TAX_REGIME_ALIASES = {
    "ITALY": "italy_administered",
    "ITALY_ADMINISTERED": "italy_administered",
    "ADMINISTERED": "italy_administered",
    "ITALY_DECLARATIVE": "italy_declarative",
    "DECLARATIVE": "italy_declarative",
    "ITALY_MANAGED": "italy_managed",
    "MANAGED": "italy_managed",
}


def register_tax_policy(policy: TaxPolicy, *, replace: bool = False) -> None:
    """Register a country policy without modifying the simulation engine."""

    country = str(policy.country_code).strip().upper()
    regime = str(policy.regime).strip().lower()
    if not country or country == "NONE" or not regime or regime == "none":
        raise ValueError("Tax policies require a country code and regime name.")
    key = (country, regime)
    if key in TAX_POLICY_REGISTRY and not replace:
        raise ValueError(f"Tax policy {country}/{regime} is already registered.")
    TAX_POLICY_REGISTRY[key] = policy


def available_tax_countries() -> list[dict[str, Any]]:
    """Describe registered country policies for API and UI discovery."""

    countries: dict[str, dict[str, Any]] = {}
    for policy in TAX_POLICY_REGISTRY.values():
        entry = countries.setdefault(
            policy.country_code,
            {
                "code": policy.country_code,
                "label": policy.country_label,
                "regimes": [],
            },
        )
        entry["regimes"].append(
            {"value": policy.regime, "label": policy.regime_label}
        )
    return sorted(countries.values(), key=lambda entry: str(entry["code"]))


def resolve_tax_selection(
    country: str | None = None,
    regime: str | None = None,
) -> TaxSelection:
    """Resolve new country settings and legacy Italian regime values."""

    raw_country = str(country or "none").strip().upper()
    raw_regime = str(regime or "none").strip().upper()
    normalized_regime = TAX_REGIME_ALIASES.get(raw_regime, raw_regime.lower())
    normalized_country = TAX_COUNTRY_ALIASES.get(raw_country, raw_country)

    # Legacy clients selected Italy directly through tax_regime.
    if normalized_country in {"", "NONE"} and normalized_regime == "italy_administered":
        normalized_country = "IT"
    if normalized_country in {"", "NONE"}:
        return TaxSelection(country="none", regime="none", policy=None)
    if normalized_country not in {entry[0] for entry in TAX_POLICY_REGISTRY}:
        available = ", ".join(sorted({entry[0] for entry in TAX_POLICY_REGISTRY}))
        raise ValueError(f"Unknown tax country '{country}'. Available countries: {available}.")
    if normalized_regime in {"", "none"}:
        default_policy = TAX_POLICY_REGISTRY.get((normalized_country, "italy_administered"))
        if default_policy is not None:
            return TaxSelection(default_policy.country_code, default_policy.regime, default_policy)
        country_policies = [
            policy
            for (code, _), policy in TAX_POLICY_REGISTRY.items()
            if code == normalized_country
        ]
        if len(country_policies) != 1:
            raise ValueError(f"Select a tax regime for country {normalized_country}.")
        policy = country_policies[0]
        return TaxSelection(policy.country_code, policy.regime, policy)
    policy = TAX_POLICY_REGISTRY.get((normalized_country, normalized_regime))
    if policy is None:
        raise ValueError(
            f"Unknown tax regime '{regime}' for country {normalized_country}."
        )
    return TaxSelection(policy.country_code, policy.regime, policy)
