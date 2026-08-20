import numpy as np
import pytest

from mc_quadrants.decumulation import (
    SpendingController,
    inflation_index,
    normalize_decumulation,
    success_mask,
    wilson_interval,
)
from mc_quadrants.simulation import simulate_portfolio_paths
from mc_quadrants.types import SimulationResult


def _plan(raw, periods=36):
    return normalize_decumulation(raw, periods=periods)


def test_phases_use_inclusive_boundaries_and_frequency():
    plan = _plan(
        {
            "mode": "manual",
            "phases": [
                {
                    "start_month": 2,
                    "end_month": 8,
                    "frequency": "quarterly",
                    "annual_real_amount": 1200,
                }
            ],
        },
        periods=8,
    )
    cpi = np.ones((8, 2))
    controller = SpendingController(plan, paths=2, initial_value=100_000, cpi=cpi)
    amounts = [controller.request(month, np.full(2, 100_000.0))[0][0] for month in range(1, 9)]
    assert amounts == [0, 300, 0, 0, 300, 0, 0, 300]


def test_one_time_expenses_in_same_month_are_summed_and_inflated():
    plan = _plan(
        {
            "mode": "manual",
            "one_time_expenses": [
                {"month": 2, "real_amount": 100},
                {"month": 2, "real_amount": 50},
            ],
        },
        periods=2,
    )
    cpi = np.array([[1.0], [1.1]])
    controller = SpendingController(plan, paths=1, initial_value=1_000, cpi=cpi)
    assert controller.request(1, np.array([1_000.0]))[0][0] == 0
    assert controller.request(2, np.array([1_000.0]))[0][0] == pytest.approx(165.0)


def test_overlapping_phases_are_rejected():
    with pytest.raises(ValueError, match="cannot overlap"):
        _plan(
            {
                "phases": [
                    {"start_month": 1, "end_month": 12, "annual_real_amount": 12},
                    {"start_month": 12, "end_month": 24, "annual_real_amount": 12},
                ]
            },
            periods=24,
        )


def test_legacy_withdrawal_remains_nominal_and_monthly():
    plan = normalize_decumulation(
        None,
        periods=4,
        legacy_withdrawal=10,
        legacy_start_period=3,
        annual_inflation_fallback=0.50,
    )
    controller = SpendingController(
        plan,
        paths=1,
        initial_value=100,
        cpi=inflation_index(4, 1, annual_inflation=0.50),
    )
    assert [controller.request(month, np.array([100.0]))[0][0] for month in range(1, 5)] == [
        0,
        0,
        10,
        10,
    ]


def test_guyton_klinger_cuts_and_resets_at_a_new_phase():
    plan = _plan(
        {
            "mode": "manual",
            "policy": "guyton_klinger",
            "phases": [
                {"start_month": 1, "end_month": 13, "annual_real_amount": 1200},
                {"start_month": 14, "end_month": 24, "annual_real_amount": 2400},
            ],
        },
        periods=24,
    )
    controller = SpendingController(plan, paths=1, initial_value=10_000, cpi=np.ones((24, 1)))
    first, first_event = controller.request(1, np.array([10_000.0]))
    cut, cut_event = controller.request(13, np.array([5_000.0]))
    reset, reset_event = controller.request(14, np.array([5_000.0]))
    assert first[0] == pytest.approx(100.0)
    assert first_event[0] == 0
    assert cut[0] == pytest.approx(90.0)
    assert cut_event[0] == -1
    assert reset[0] == pytest.approx(200.0)
    assert reset_event[0] == 0


def test_negative_real_year_suspends_inflation_indexing():
    plan = _plan(
        {
            "mode": "manual",
            "policy": {
                "type": "guyton_klinger",
                "upper_guardrail": 10,
                "lower_guardrail": 0.01,
            },
            "phases": [
                {"start_month": 1, "end_month": 13, "annual_real_amount": 1200}
            ],
        },
        periods=13,
    )
    cpi = np.ones((13, 1))
    cpi[12] = 1.10
    controller = SpendingController(plan, paths=1, initial_value=10_000, cpi=cpi)
    controller.request(1, np.array([10_000.0]))
    amount, _ = controller.request(13, np.array([9_000.0]))
    assert amount[0] == pytest.approx(100.0)


def test_safe_rate_success_requires_every_expense_to_be_funded():
    wealth = np.array([[100.0, 100.0], [0.0, 50.0]])
    requested = np.array([[10.0, 10.0], [20.0, 20.0]])
    funded = np.array([[10.0, 10.0], [19.0, 20.0]])
    survival = success_mask(
        wealth,
        requested,
        funded,
        objective="survival",
        initial_value=100,
    )
    preservation = success_mask(
        wealth,
        requested,
        funded,
        objective="preserve_initial",
        initial_value=40,
    )
    assert survival.tolist() == [False, True]
    assert preservation.tolist() == [False, True]


def test_wilson_interval_contains_observed_probability():
    lower, upper = wilson_interval(90, 100)
    assert lower < 0.9 < upper


def test_italian_ledger_funds_spending_net_of_disposal_tax():
    result = SimulationResult(
        returns=np.array([[[1.0]]]),
        regimes=np.zeros((1, 1), dtype=int),
        assets=["ETF"],
        states=[],
        frequency="M",
    )
    wealth = simulate_portfolio_paths(
        result,
        {"ETF": 1.0},
        return_kind="simple",
        rebalance_frequency=0,
        tax_country="IT",
        italy_annual_wealth_tax=0.0,
        tax_terminal_liquidation=False,
        decumulation={
            "mode": "manual",
            "phases": [
                {
                    "start_month": 1,
                    "end_month": 1,
                    "frequency": "monthly",
                    "annual_real_amount": 600,
                }
            ],
        },
    )
    assert wealth.attrs["withdrawal_requested"][0, 0] == pytest.approx(50.0)
    assert wealth.attrs["withdrawal_funded"][0, 0] == pytest.approx(50.0)
    assert wealth.attrs["taxes_paid_total"] > 0
    assert wealth.iloc[0, 0] < 150.0


def test_italian_administered_shortfall_is_measured_after_disposal_tax():
    result = SimulationResult(
        returns=np.array([[[1.0]]]),
        regimes=np.zeros((1, 1), dtype=int),
        assets=["ETF"],
        states=[],
        frequency="M",
    )
    wealth = simulate_portfolio_paths(
        result,
        {"ETF": 1.0},
        return_kind="simple",
        rebalance_frequency=0,
        tax_country="IT",
        italy_annual_wealth_tax=0.0,
        tax_terminal_liquidation=False,
        decumulation={
            "mode": "manual",
            "phases": [
                {
                    "start_month": 1,
                    "end_month": 1,
                    "frequency": "monthly",
                    "annual_real_amount": 3600,
                }
            ],
        },
    )
    assert wealth.attrs["withdrawal_requested"][0, 0] == pytest.approx(300.0)
    assert wealth.attrs["withdrawal_funded"][0, 0] == pytest.approx(174.0)
    assert wealth.attrs["taxes_paid_total"] == pytest.approx(26.0)
    assert wealth.iloc[0, 0] == pytest.approx(0.0)
