import numpy as np
import pytest

from mc_quadrants.taxes import (
    ITALY_LOSS_CARRY_YEARS,
    _advance_tax_year,
    simulate_italian_portfolio_tax,
)


def test_italian_losses_expire_after_four_subsequent_tax_years():
    buckets = np.zeros((ITALY_LOSS_CARRY_YEARS + 1, 1), dtype=float)
    buckets[-1, 0] = 10.0

    for _ in range(ITALY_LOSS_CARRY_YEARS):
        _advance_tax_year(buckets)
        assert buckets.sum() == 10.0

    _advance_tax_year(buckets)

    assert buckets.sum() == 0.0


def test_income_is_separated_from_price_return_and_taxed_when_distributed():
    wealth = simulate_italian_portfolio_tax(
        np.ones((1, 1, 1)),
        assets=["ETF"],
        target_weights=np.array([1.0]),
        initial_value=100.0,
        rebalance_frequency=0,
        annual_wealth_tax=0.0,
        terminal_liquidation=True,
        asset_tax_metadata={"ETF": {"category": "fund", "annual_income_yield": 0.12}},
    )

    assert wealth.iloc[-1, 0] == pytest.approx(99.74)
    assert wealth.attrs["investment_income_tax_total"] == pytest.approx(0.26)
    assert wealth.attrs["terminal_liquidation_tax_total"] == pytest.approx(0.0)


def test_government_fraction_and_foreign_tax_credit_are_instrument_specific():
    wealth = simulate_italian_portfolio_tax(
        np.ones((1, 1, 1)),
        assets=["FUND"],
        target_weights=np.array([1.0]),
        initial_value=100.0,
        rebalance_frequency=0,
        annual_wealth_tax=0.0,
        terminal_liquidation=False,
        asset_tax_metadata={
            "FUND": {
                "category": "fund",
                "annual_income_yield": 0.12,
                "government_bond_fraction": 0.5,
                "foreign_withholding_rate": 0.10,
                "foreign_tax_credit_rate": 0.10,
            }
        },
    )

    # Half of the distribution receives the 12.5%-equivalent base. A 10%
    # foreign withholding is credited against the Italian liability.
    assert wealth.attrs["foreign_withholding_tax_total"] == pytest.approx(0.10)
    assert wealth.attrs["investment_income_tax_total"] == pytest.approx(0.0925)
    assert wealth.iloc[-1, 0] == pytest.approx(99.8075)


@pytest.mark.parametrize("regime", ["italy_administered", "italy_declarative", "italy_managed"])
def test_all_italian_regimes_produce_calendar_year_tax_reporting(regime):
    wealth = simulate_italian_portfolio_tax(
        np.full((13, 2, 1), 1.01),
        assets=["Asset"],
        target_weights=np.array([1.0]),
        initial_value=100.0,
        rebalance_frequency=1,
        annual_wealth_tax=0.0,
        terminal_liquidation=True,
        tax_regime=regime,
        start_date="2025-07-31",
    )

    assert wealth.attrs["tax_regime"] == regime
    assert set(wealth.attrs["tax_by_year"]) == {"2025", "2026"}
    assert np.isfinite(wealth.to_numpy()).all()


def test_stamp_duty_and_ivafe_follow_account_location_without_double_charge():
    wealth = simulate_italian_portfolio_tax(
        np.ones((12, 1, 2)),
        assets=["Domestic", "Foreign"],
        target_weights=np.array([0.5, 0.5]),
        initial_value=100.0,
        rebalance_frequency=0,
        annual_wealth_tax=0.002,
        terminal_liquidation=False,
        wealth_tax_mode="auto",
        asset_tax_metadata={
            "Domestic": {"account_location": "domestic"},
            "Foreign": {"account_location": "foreign"},
        },
    )

    assert wealth.attrs["stamp_duty_total"] == pytest.approx(wealth.attrs["ivafe_total"])
    assert wealth.attrs["wealth_tax_total"] == pytest.approx(
        wealth.attrs["stamp_duty_total"] + wealth.attrs["ivafe_total"]
    )


def test_italian_tax_ledger_respects_delayed_decumulation():
    wealth = simulate_italian_portfolio_tax(
        np.ones((4, 1, 1)),
        assets=["Asset"],
        target_weights=np.array([1.0]),
        initial_value=100.0,
        rebalance_frequency=0,
        withdrawal=10.0,
        withdrawal_start_period=3,
        annual_wealth_tax=0.0,
        terminal_liquidation=False,
    )

    assert wealth.iloc[:, 0].tolist() == pytest.approx([100.0, 100.0, 90.0, 80.0])
    assert wealth.attrs["withdrawal_start_period"] == 3
