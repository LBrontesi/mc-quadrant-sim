"""Parity and routing coverage for the remaining simulation kernels moved to C++."""

import numpy as np
import pandas as pd
import pytest

from mc_quadrants.decumulation import inflation_index
from mc_quadrants.native import native_available
from mc_quadrants.native_metrics import money_weighted_returns_native
from mc_quadrants.simulation import (
    inflation_deflators,
    simulate_portfolio_paths,
    summarize_wealth_risk,
)
from mc_quadrants.taxes import simulate_italian_portfolio_tax
from mc_quadrants.types import SimulationResult

pytestmark = pytest.mark.skipif(not native_available(), reason="native simulator is not compiled")


@pytest.mark.parametrize("periods", [1, 12, 360, 1200])
def test_native_money_weighted_returns_match_discounted_cashflows(periods):
    terminal = np.array([0, .01, 100, 1000 + 3 * periods, 5000, 1e8])
    native = money_weighted_returns_native(terminal, periods, 1000, 3)
    low = np.full(len(terminal), -.99)
    high = np.full(len(terminal), 10.0)
    time = np.arange(1, periods + 1)[:, None]
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        for _ in range(64):
            rate = (low + high) / 2
            discount = np.power(1 + rate[None, :], time)
            npv = -1000 - np.sum(3 / discount, axis=0) + terminal / discount[-1]
            low = np.where(npv > 0, rate, low)
            high = np.where(npv > 0, high, rate)
    reference = np.clip((1 + (low + high) / 2)**12 - 1, -1, 100)
    np.testing.assert_allclose(native, reference, rtol=1e-12, atol=5e-14)


@pytest.mark.parametrize("return_kind", ["log", "simple"])
@pytest.mark.parametrize("rebalance,leverage", [(None, 1), (0, 1), (1, 1), (3, 1), (3, 2)])
@pytest.mark.parametrize("spending", ["none", "legacy", "guardrails"])
def test_neutral_ledger_matches_reference(monkeypatch, return_kind, rebalance, leverage, spending):
    rng = np.random.default_rng(142)
    returns = rng.normal(0.004, 0.11, (38, 17, 3))
    # Include insolvency and subsequent contributions/re-entry.
    returns[8, 0] = -3.0
    if return_kind == "simple":
        returns = np.expm1(returns)
    result = SimulationResult(returns=returns, regimes=np.zeros((38, 17), dtype=int),
                              assets=["A", "B", "C"], states=["state"], frequency="M")
    kwargs = dict(initial_value=1000, return_kind=return_kind, rebalance_frequency=rebalance,
                  contribution=3.0, contribution_allocation="target" if leverage > 1 or rebalance is None else "underweight_first",
                  asset_expense_ratios={"A": .002, "B": .004}, native_threads=4)
    if rebalance:
        kwargs.update(transaction_cost_bps=12)
        if leverage == 1:
            kwargs.update(state_transaction_cost_multipliers={"state": 1.5})
    if leverage > 1:
        kwargs.update(leverage_multiple=leverage, financing_rate=.02, maintenance_margin=.1,
                      financing_rate_paths=rng.uniform(0, .05, (38, 17)))
    if spending == "legacy":
        kwargs.update(withdrawal=6, withdrawal_start_period=4)
    elif spending == "guardrails":
        kwargs.update(annual_inflation=.03, withdrawal_inflation_paths=rng.uniform(-.03, .1, (38, 17)),
                      decumulation={"mode": "manual", "policy": "guyton_klinger", "phases": [
                          {"start_month": 2, "end_month": 25, "annual_real_amount": 100, "frequency": "quarterly"},
                          {"start_month": 27, "end_month": 38, "annual_real_amount": 160},
                      ], "one_time_expenses": [{"month": 16, "real_amount": 40}]})
    native = simulate_portfolio_paths(result, {"A": .5, "B": .3, "C": .2}, **kwargs)
    assert native.attrs.get("native_portfolio_backend") is True
    monkeypatch.setenv("MC_DISABLE_NATIVE_SIM", "1")
    reference = simulate_portfolio_paths(result, {"A": .5, "B": .3, "C": .2}, **kwargs)
    np.testing.assert_allclose(native, reference, rtol=2e-12, atol=1e-10)
    for key in ("withdrawal_requested", "withdrawal_funded", "guardrail_events"):
        np.testing.assert_allclose(native.attrs[key], reference.attrs[key], rtol=2e-12, atol=1e-10)
    for key in ("transaction_cost_total", "margin_calls"):
        if key in reference.attrs:
            assert native.attrs[key] == pytest.approx(reference.attrs[key], rel=2e-12, abs=1e-10)


@pytest.mark.parametrize("regime", ["italy_administered", "italy_declarative", "italy_managed"])
@pytest.mark.parametrize("wrapper", [False, True])
def test_distribution_income_ledger_matches_reference(monkeypatch, regime, wrapper):
    rng = np.random.default_rng(29)
    growth = np.exp(rng.normal(.004, .08, (61, 13, 3)))
    kwargs = dict(assets=["F", "S", "G"], target_weights=np.array([.5, .3, .2]),
                  initial_value=1000, rebalance_frequency=3, transaction_cost_bps=10,
                  contribution=2, withdrawal=5, withdrawal_start_period=9,
                  tax_regime=regime, start_date="2025-07-31", wrapper_benchmark=wrapper,
                  asset_tax_metadata={
                      "F": {"category": "fund", "annual_income_yield": .06, "government_bond_fraction": .4,
                            "foreign_withholding_rate": .15, "foreign_tax_credit_rate": .1},
                      "S": {"category": "standard", "annual_income_yield": .04, "account_location": "foreign"},
                      "G": {"category": "government_bond", "annual_income_yield": .03},
                  })
    native = simulate_italian_portfolio_tax(growth, **kwargs)
    assert native.attrs["native_backend"] is True
    monkeypatch.setenv("MC_DISABLE_NATIVE_SIM", "1")
    reference = simulate_italian_portfolio_tax(growth, **kwargs)
    np.testing.assert_allclose(native, reference, rtol=2e-11, atol=1e-9)
    for key, value in reference.attrs.items():
        if key.endswith("_total") and isinstance(value, (float, int)):
            assert native.attrs[key] == pytest.approx(value, rel=2e-11, abs=1e-9), key
    for year, stats in reference.attrs["tax_by_year"].items():
        for key, value in stats.items():
            assert native.attrs["tax_by_year"][year][key] == pytest.approx(value, rel=2e-11, abs=1e-9), key


@pytest.mark.parametrize("stochastic", [False, True])
def test_inflation_and_risk_match_reference(monkeypatch, stochastic):
    rng = np.random.default_rng(42)
    rates = rng.uniform(-.05, .1, (48, 19)) if stochastic else None
    wealth = pd.DataFrame(1000 * np.exp(np.cumsum(rng.normal(0, .06, (48, 19)), axis=0)))
    kwargs = dict(initial_value=1000, contribution=3, withdrawal=2, annual_inflation=.02,
                  inflation_paths=rates)
    native_cpi = inflation_index(48, 19, annual_inflation=.02, inflation_paths=rates)
    native_deflator = inflation_deflators(48, 19, annual_inflation=.02, inflation_paths=rates)
    native_risk = summarize_wealth_risk(wealth, **kwargs)
    monkeypatch.setenv("MC_DISABLE_NATIVE_SIM", "1")
    np.testing.assert_allclose(native_cpi, inflation_index(48, 19, annual_inflation=.02, inflation_paths=rates), rtol=2e-13)
    np.testing.assert_allclose(native_deflator, inflation_deflators(48, 19, annual_inflation=.02, inflation_paths=rates), rtol=2e-13)
    pd.testing.assert_series_equal(native_risk, summarize_wealth_risk(wealth, **kwargs), rtol=2e-11, atol=1e-10)
