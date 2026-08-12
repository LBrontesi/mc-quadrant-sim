from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

ITALY_STANDARD_TAX_RATE = 0.26
ITALY_GOVERNMENT_BOND_RATE = 0.125
ITALY_DEFAULT_WEALTH_TAX_RATE = 0.002
ITALY_LOSS_CARRY_YEARS = 4

ITALY_TAX_CATEGORIES = {
    "standard",
    "government_bond",
    "fund",
    "government_bond_fund",
}


@dataclass(frozen=True)
class _TaxProfile:
    taxable_fraction: np.ndarray
    gains_offsettable: np.ndarray


def normalize_italy_tax_categories(
    assets: list[str],
    categories: Mapping[str, str] | None,
) -> dict[str, str]:
    """Return a validated category for each asset, defaulting to standard."""

    supplied = {str(asset).strip().upper(): str(value).strip().lower() for asset, value in (categories or {}).items()}
    unknown = sorted(set(supplied.values()) - ITALY_TAX_CATEGORIES)
    if unknown:
        allowed = ", ".join(sorted(ITALY_TAX_CATEGORIES))
        raise ValueError(f"Unknown Italian tax category '{unknown[0]}'. Expected one of: {allowed}.")
    return {asset: supplied.get(str(asset).strip().upper(), "standard") for asset in assets}


def _tax_profile(assets: list[str], categories: Mapping[str, str] | None) -> _TaxProfile:
    normalized = normalize_italy_tax_categories(assets, categories)
    # Article 68 TUIR expresses government-security gains and losses at the
    # fraction that makes the standard 26% rate equivalent to 12.5%.
    government_fraction = ITALY_GOVERNMENT_BOND_RATE / ITALY_STANDARD_TAX_RATE
    taxable_fraction = np.array(
        [
            government_fraction if normalized[asset] in {"government_bond", "government_bond_fund"} else 1.0
            for asset in assets
        ],
        dtype=float,
    )
    gains_offsettable = np.array(
        [normalized[asset] in {"standard", "government_bond"} for asset in assets],
        dtype=bool,
    )
    return _TaxProfile(taxable_fraction=taxable_fraction, gains_offsettable=gains_offsettable)


def _consume_losses(loss_buckets: np.ndarray, amount: np.ndarray) -> np.ndarray:
    """Consume the oldest available losses first and return the uncovered base."""

    remaining = np.maximum(np.asarray(amount, dtype=float), 0.0).copy()
    for bucket in range(loss_buckets.shape[0]):
        used = np.minimum(loss_buckets[bucket], remaining)
        loss_buckets[bucket] -= used
        remaining -= used
    return remaining


def _advance_tax_year(loss_buckets: np.ndarray) -> None:
    """Age losses and discard amounts older than four subsequent tax years."""

    loss_buckets[:-1] = loss_buckets[1:]
    loss_buckets[-1] = 0.0


def _taxable_components(
    sales: np.ndarray,
    holdings: np.ndarray,
    basis: np.ndarray,
    profile: _TaxProfile,
    transaction_cost_rate: float,
) -> tuple[np.ndarray, np.ndarray]:
    safe_holdings = np.maximum(holdings, 1e-300)
    basis_sold = np.where(holdings > 0, basis * sales / safe_holdings, 0.0)
    realized = sales * (1.0 - transaction_cost_rate) - basis_sold
    return realized, realized * profile.taxable_fraction[None, :]


def _preview_sales_tax(
    sales: np.ndarray,
    holdings: np.ndarray,
    basis: np.ndarray,
    profile: _TaxProfile,
    loss_buckets: np.ndarray,
    transaction_cost_rate: float,
) -> np.ndarray:
    """Calculate disposal tax without copying or mutating the loss ledger."""

    _, taxable = _taxable_components(
        sales,
        holdings,
        basis,
        profile,
        transaction_cost_rate,
    )
    taxable_gains = np.maximum(taxable, 0.0)
    non_offsettable = taxable_gains[:, ~profile.gains_offsettable].sum(axis=1)
    offsettable = taxable_gains[:, profile.gains_offsettable].sum(axis=1)
    available_losses = loss_buckets.sum(axis=0) + np.maximum(-taxable, 0.0).sum(axis=1)
    return (non_offsettable + np.maximum(offsettable - available_losses, 0.0)) * ITALY_STANDARD_TAX_RATE


def _settle_sales(
    sales: np.ndarray,
    holdings: np.ndarray,
    basis: np.ndarray,
    profile: _TaxProfile,
    loss_buckets: np.ndarray,
    transaction_cost_rate: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Settle one disposal event and mutate the path-level loss ledger."""

    realized, taxable = _taxable_components(
        sales,
        holdings,
        basis,
        profile,
        transaction_cost_rate,
    )

    realized_losses = np.maximum(-realized, 0.0).sum(axis=1)
    new_loss_base = np.maximum(-taxable, 0.0).sum(axis=1)
    loss_buckets[-1] += new_loss_base

    taxable_gains = np.maximum(taxable, 0.0)
    non_offsettable = taxable_gains[:, ~profile.gains_offsettable].sum(axis=1)
    offsettable = taxable_gains[:, profile.gains_offsettable].sum(axis=1)
    uncovered = _consume_losses(loss_buckets, offsettable)
    tax = (non_offsettable + uncovered) * ITALY_STANDARD_TAX_RATE
    realized_gains = np.maximum(realized, 0.0).sum(axis=1)
    return tax, realized_gains, realized_losses


def _apply_sales(
    sales: np.ndarray,
    holdings: np.ndarray,
    basis: np.ndarray,
    profile: _TaxProfile,
    loss_buckets: np.ndarray,
    transaction_cost_rate: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tax, gains, losses = _settle_sales(
        sales,
        holdings,
        basis,
        profile,
        loss_buckets,
        transaction_cost_rate,
    )
    basis_fraction = np.divide(
        sales,
        holdings,
        out=np.zeros_like(sales),
        where=holdings > 0,
    )
    basis *= np.maximum(1.0 - basis_fraction, 0.0)
    holdings -= sales
    np.maximum(holdings, 0.0, out=holdings)
    np.maximum(basis, 0.0, out=basis)
    return tax, gains, losses


def _raise_cash_with_tax(
    requested: float,
    holdings: np.ndarray,
    basis: np.ndarray,
    profile: _TaxProfile,
    loss_buckets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fund a net withdrawal and the tax created by the required disposals."""

    paths = holdings.shape[0]
    tax_total = np.zeros(paths, dtype=float)
    gain_total = np.zeros(paths, dtype=float)
    loss_total = np.zeros(paths, dtype=float)
    cash_required = np.full(paths, max(float(requested), 0.0), dtype=float)
    for _ in range(12):
        values = holdings.sum(axis=1)
        gross_sale = np.minimum(cash_required, values)
        if not np.any(gross_sale > 1e-10):
            break
        sales = np.divide(
            holdings * gross_sale[:, None],
            values[:, None],
            out=np.zeros_like(holdings),
            where=values[:, None] > 0,
        )
        tax, gains, losses = _apply_sales(sales, holdings, basis, profile, loss_buckets)
        tax_total += tax
        gain_total += gains
        loss_total += losses
        cash_required = tax
    return tax_total, gain_total, loss_total


def _rebalance_after_tax(
    holdings: np.ndarray,
    basis: np.ndarray,
    target_weights: np.ndarray,
    cost_rate: float,
    profile: _TaxProfile,
    loss_buckets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rebalance to weights after solving for tax and transaction-cost drag."""

    values = holdings.sum(axis=1)
    post_cost_value = values.copy()
    for _ in range(8):
        target = post_cost_value[:, None] * target_weights
        sales = np.maximum(holdings - target, 0.0)
        purchases = np.maximum(target - holdings, 0.0)
        preview_tax = _preview_sales_tax(
            sales,
            holdings,
            basis,
            profile,
            loss_buckets,
            cost_rate,
        )
        costs = (sales + purchases).sum(axis=1) * cost_rate
        updated = np.maximum(values - preview_tax - costs, 0.0)
        if np.allclose(updated, post_cost_value, rtol=1e-11, atol=1e-10):
            post_cost_value = updated
            break
        post_cost_value = updated

    target = post_cost_value[:, None] * target_weights
    sales = np.maximum(holdings - target, 0.0)
    tax, gains, losses = _apply_sales(
        sales,
        holdings,
        basis,
        profile,
        loss_buckets,
        cost_rate,
    )
    purchases = np.maximum(target - holdings, 0.0)
    basis += purchases * (1.0 + cost_rate)
    holdings += purchases
    return tax, gains, losses


def simulate_italian_portfolio_tax(
    asset_growth: np.ndarray,
    assets: list[str],
    target_weights: np.ndarray,
    initial_value: float,
    rebalance_frequency: int,
    transaction_cost_bps: float = 0.0,
    contribution: float = 0.0,
    withdrawal: float = 0.0,
    asset_tax_categories: Mapping[str, str] | None = None,
    annual_wealth_tax: float = ITALY_DEFAULT_WEALTH_TAX_RATE,
    terminal_liquidation: bool = True,
) -> pd.DataFrame:
    """Simulate a simplified Italian administered-regime tax account.

    The model uses average cost, a four-subsequent-tax-year loss ledger,
    26% standard taxation, the 12.5% government-security equivalent base,
    and a configurable annual stamp-duty/IVAFE proxy. Fund gains are treated
    as non-offsettable income while fund losses enter the capital-loss ledger.
    """

    growth = np.asarray(asset_growth, dtype=float)
    weights = np.asarray(target_weights, dtype=float)
    if growth.ndim != 3 or growth.shape[2] != len(assets):
        raise ValueError("asset_growth must have shape (periods, paths, assets).")
    if not np.isfinite(growth).all() or (growth <= 0).any():
        raise ValueError("asset_growth must contain positive, finite values.")
    if weights.shape != (len(assets),) or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Italian tax accounting requires finite, non-negative portfolio weights.")
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError("Portfolio weights must sum to 1 for Italian tax accounting.")
    if rebalance_frequency < 0:
        raise ValueError("rebalance_frequency must be non-negative.")
    if not np.isfinite(annual_wealth_tax) or not 0 <= annual_wealth_tax < 1:
        raise ValueError("annual_wealth_tax must be a decimal between 0 and 1.")

    profile = _tax_profile(assets, asset_tax_categories)
    periods, paths, asset_count = growth.shape
    holdings = np.broadcast_to(initial_value * weights, (paths, asset_count)).copy()
    basis = holdings.copy()
    wealth = np.empty((periods, paths), dtype=float)
    loss_buckets = np.zeros((ITALY_LOSS_CARRY_YEARS + 1, paths), dtype=float)
    cost_rate = float(transaction_cost_bps) / 10_000.0
    wealth_tax_rate = float(annual_wealth_tax) / 12.0

    capital_gains_tax = np.zeros(paths, dtype=float)
    wealth_tax = np.zeros(paths, dtype=float)
    terminal_tax = np.zeros(paths, dtype=float)
    realized_gains = np.zeros(paths, dtype=float)
    realized_losses = np.zeros(paths, dtype=float)

    for period in range(periods):
        if period and period % 12 == 0:
            _advance_tax_year(loss_buckets)
        if contribution:
            purchases = float(contribution) * weights
            holdings += purchases
            basis += purchases
        holdings *= growth[period]

        if withdrawal:
            taxes, gains, losses = _raise_cash_with_tax(
                withdrawal,
                holdings,
                basis,
                profile,
                loss_buckets,
            )
            capital_gains_tax += taxes
            realized_gains += gains
            realized_losses += losses

        if rebalance_frequency > 0 and (period + 1) % rebalance_frequency == 0:
            taxes, gains, losses = _rebalance_after_tax(
                holdings,
                basis,
                weights,
                cost_rate,
                profile,
                loss_buckets,
            )
            capital_gains_tax += taxes
            realized_gains += gains
            realized_losses += losses

        if wealth_tax_rate:
            charge = holdings.sum(axis=1) * wealth_tax_rate
            scale = np.maximum(1.0 - wealth_tax_rate, 0.0)
            holdings *= scale
            basis *= scale
            wealth_tax += charge
        wealth[period] = holdings.sum(axis=1)

    if terminal_liquidation and periods:
        sales = holdings.copy()
        taxes, gains, losses = _settle_sales(sales, holdings, basis, profile, loss_buckets)
        terminal_tax += taxes
        realized_gains += gains
        realized_losses += losses
        wealth[-1] = np.maximum(wealth[-1] - taxes, 0.0)

    frame = pd.DataFrame(wealth, columns=[f"path_{index}" for index in range(paths)])
    categories = normalize_italy_tax_categories(assets, asset_tax_categories)
    frame.attrs.update(
        {
            "margin_calls": 0,
            "tax_regime": "italy_administered",
            "asset_tax_categories": categories,
            "capital_gains_tax_total": float(capital_gains_tax.sum()),
            "wealth_tax_total": float(wealth_tax.sum()),
            "terminal_liquidation_tax_total": float(terminal_tax.sum()),
            "taxes_paid_total": float((capital_gains_tax + wealth_tax + terminal_tax).sum()),
            "realized_gains_total": float(realized_gains.sum()),
            "realized_losses_total": float(realized_losses.sum()),
            "loss_carryforward_total": float(loss_buckets.sum()),
            "annual_wealth_tax": float(annual_wealth_tax),
            "tax_terminal_liquidation": bool(terminal_liquidation),
        }
    )
    return frame
