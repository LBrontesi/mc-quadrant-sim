from dataclasses import dataclass

import pandas as pd
import pytest

from mc_quadrants.tax_policy import (
    TAX_POLICY_REGISTRY,
    TaxSimulationContext,
    available_tax_countries,
    register_tax_policy,
    resolve_tax_selection,
)


def test_builtin_registry_exposes_italian_tax_regimes():
    assert available_tax_countries() == [
        {
            "code": "IT",
            "label": "Italy",
                "regimes": [
                    {"value": "italy_administered", "label": "Simplified administered regime"},
                    {"value": "italy_declarative", "label": "Declarative regime"},
                    {"value": "italy_managed", "label": "Managed regime"},
                ],
        }
    ]
    assert resolve_tax_selection().enabled is False
    assert resolve_tax_selection("IT").regime == "italy_administered"
    assert resolve_tax_selection(None, "italy_administered").country == "IT"


def test_unknown_country_and_country_regime_are_rejected():
    with pytest.raises(ValueError, match="Unknown tax country"):
        resolve_tax_selection("FR")
    with pytest.raises(ValueError, match="Unknown tax regime"):
        resolve_tax_selection("IT", "ordinary")


def test_country_policy_can_be_registered_without_changing_the_engine():
    @dataclass(frozen=True)
    class TestPolicy:
        country_code: str = "ZZ"
        regime: str = "test"
        country_label: str = "Test country"
        regime_label: str = "Test regime"

        def validate(self, **_kwargs) -> None:
            return None

        def simulate(self, context: TaxSimulationContext) -> pd.DataFrame:
            return pd.DataFrame(context.asset_growth[:, :, 0])

        def metadata(self, context=None):
            return {"country": self.country_code, "regime": self.regime}

    policy = TestPolicy()
    try:
        register_tax_policy(policy)
        selection = resolve_tax_selection("ZZ", "test")
        assert selection.enabled
        assert selection.policy is policy
        with pytest.raises(ValueError, match="already registered"):
            register_tax_policy(policy)
    finally:
        TAX_POLICY_REGISTRY.pop(("ZZ", "test"), None)
