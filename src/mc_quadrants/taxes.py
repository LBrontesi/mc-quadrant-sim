from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from mc_quadrants.decumulation import (
    DecumulationPlan,
    SpendingController,
    funded_amount,
    inflation_index,
    normalize_decumulation,
)
from mc_quadrants.native import simulate_italian_portfolios_native

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
ITALY_TAX_REGIMES = {"italy_administered", "italy_declarative", "italy_managed"}
ITALY_WEALTH_TAX_MODES = {"auto", "stamp_duty", "ivafe", "both", "none"}
CONTRIBUTION_ALLOCATION_MODES = {"target", "underweight_first"}


@dataclass(frozen=True)
class _TaxProfile:
    taxable_fraction: np.ndarray
    gains_offsettable: np.ndarray
    annual_income_yield: np.ndarray
    foreign_withholding_rate: np.ndarray
    foreign_tax_credit_rate: np.ndarray
    financial_transaction_tax_rate: np.ndarray
    wealth_tax_eligible: np.ndarray
    foreign_account: np.ndarray
    metadata: dict[str, dict[str, object]]


def normalize_italy_tax_categories(
    assets: list[str],
    categories: Mapping[str, str] | None,
) -> dict[str, str]:
    """Return a validated category for each asset, defaulting to a fund/ETF."""

    supplied = {str(asset).strip().upper(): str(value).strip().lower() for asset, value in (categories or {}).items()}
    unknown = sorted(set(supplied.values()) - ITALY_TAX_CATEGORIES)
    if unknown:
        allowed = ", ".join(sorted(ITALY_TAX_CATEGORIES))
        raise ValueError(f"Unknown Italian tax category '{unknown[0]}'. Expected one of: {allowed}.")
    return {asset: supplied.get(str(asset).strip().upper(), "fund") for asset in assets}


def contribution_allocations(
    holdings: np.ndarray,
    target_weights: np.ndarray,
    contribution: float,
    mode: str,
) -> np.ndarray:
    """Allocate gross contribution cash without selling existing holdings."""

    cash = max(float(contribution), 0.0)
    if cash == 0.0:
        return np.zeros_like(holdings)
    if mode == "target":
        return np.broadcast_to(cash * target_weights, holdings.shape).copy()

    target_after_cash = (holdings.sum(axis=1) + cash)[:, None] * target_weights
    deficits = np.maximum(target_after_cash - holdings, 0.0)
    deficit_total = deficits.sum(axis=1)
    applied = np.minimum(deficit_total, cash)
    allocations = np.divide(
        deficits * applied[:, None],
        deficit_total[:, None],
        out=np.zeros_like(holdings),
        where=deficit_total[:, None] > 0,
    )
    allocations += (cash - applied)[:, None] * target_weights
    return allocations


def normalize_italy_tax_metadata(
    assets: list[str],
    categories: Mapping[str, str] | None,
    metadata: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, dict[str, object]]:
    """Validate per-instrument tax facts used by the planning engine."""

    normalized_categories = normalize_italy_tax_categories(assets, categories)
    supplied = {str(asset).strip().upper(): dict(values) for asset, values in (metadata or {}).items()}
    result: dict[str, dict[str, object]] = {}
    for asset in assets:
        values = supplied.get(str(asset).strip().upper(), {})
        category = str(values.get("category", normalized_categories[asset])).strip().lower()
        if category not in ITALY_TAX_CATEGORIES:
            allowed = ", ".join(sorted(ITALY_TAX_CATEGORIES))
            raise ValueError(f"Unknown Italian tax category '{category}'. Expected one of: {allowed}.")
        default_share = 1.0 if category in {"government_bond", "government_bond_fund"} else 0.0
        government_share = float(values.get("government_bond_fraction", default_share))
        income_yield = float(values.get("annual_income_yield", 0.0))
        foreign_withholding = float(values.get("foreign_withholding_rate", 0.0))
        foreign_credit = float(values.get("foreign_tax_credit_rate", 0.0))
        ftt_rate = float(values.get("financial_transaction_tax_rate", 0.0))
        location = str(values.get("account_location", "domestic")).strip().lower()
        for name, value in {
            "government_bond_fraction": government_share,
            "foreign_withholding_rate": foreign_withholding,
            "foreign_tax_credit_rate": foreign_credit,
            "financial_transaction_tax_rate": ftt_rate,
        }.items():
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} for {asset} must be a decimal between 0 and 1.")
        if not np.isfinite(income_yield) or not 0 <= income_yield < 1:
            raise ValueError(f"annual_income_yield for {asset} must be a decimal between 0 and 1.")
        if location not in {"domestic", "foreign"}:
            raise ValueError(f"account_location for {asset} must be 'domestic' or 'foreign'.")
        result[asset] = {
            "category": category,
            "government_bond_fraction": government_share,
            "annual_income_yield": income_yield,
            "foreign_withholding_rate": foreign_withholding,
            "foreign_tax_credit_rate": foreign_credit,
            "financial_transaction_tax_rate": ftt_rate,
            "wealth_tax_eligible": bool(values.get("wealth_tax_eligible", True)),
            "account_location": location,
            "domicile": str(values.get("domicile", "")).strip().upper(),
            "instrument_type": str(values.get("instrument_type", category)).strip().lower(),
        }
    return result


def _tax_profile(
    assets: list[str],
    categories: Mapping[str, str] | None,
    metadata: Mapping[str, Mapping[str, object]] | None = None,
) -> _TaxProfile:
    normalized = normalize_italy_tax_metadata(assets, categories, metadata)
    # Article 68 TUIR expresses government-security gains and losses at the
    # fraction that makes the standard 26% rate equivalent to 12.5%.
    government_fraction = ITALY_GOVERNMENT_BOND_RATE / ITALY_STANDARD_TAX_RATE
    taxable_fraction = np.array(
        [
            1.0 - float(normalized[asset]["government_bond_fraction"]) * (1.0 - government_fraction)
            for asset in assets
        ],
        dtype=float,
    )
    gains_offsettable = np.array(
        [normalized[asset]["category"] in {"standard", "government_bond"} for asset in assets],
        dtype=bool,
    )
    return _TaxProfile(
        taxable_fraction=taxable_fraction,
        gains_offsettable=gains_offsettable,
        annual_income_yield=np.array([normalized[asset]["annual_income_yield"] for asset in assets], dtype=float),
        foreign_withholding_rate=np.array([normalized[asset]["foreign_withholding_rate"] for asset in assets], dtype=float),
        foreign_tax_credit_rate=np.array([normalized[asset]["foreign_tax_credit_rate"] for asset in assets], dtype=float),
        financial_transaction_tax_rate=np.array(
            [normalized[asset]["financial_transaction_tax_rate"] for asset in assets], dtype=float
        ),
        wealth_tax_eligible=np.array([normalized[asset]["wealth_tax_eligible"] for asset in assets], dtype=bool),
        foreign_account=np.array([normalized[asset]["account_location"] == "foreign" for asset in assets], dtype=bool),
        metadata=normalized,
    )


def _consume_losses(loss_buckets: np.ndarray, amount: np.ndarray) -> np.ndarray:
    """Consume the oldest available losses first and return the uncovered base."""

    remaining = np.maximum(np.asarray(amount, dtype=float), 0.0).copy()
    for bucket in range(loss_buckets.shape[0]):
        used = np.minimum(loss_buckets[bucket], remaining)
        loss_buckets[bucket] -= used
        remaining -= used
    return remaining


def _advance_tax_year(loss_buckets: np.ndarray) -> np.ndarray:
    """Age losses and discard amounts older than four subsequent tax years."""

    expired = loss_buckets[0].copy()
    loss_buckets[:-1] = loss_buckets[1:]
    loss_buckets[-1] = 0.0
    return expired


def _taxable_components(
    sales: np.ndarray,
    holdings: np.ndarray,
    basis: np.ndarray,
    profile: _TaxProfile,
    transaction_cost_rate: float,
) -> tuple[np.ndarray, np.ndarray]:
    safe_holdings = np.maximum(holdings, 1e-300)
    basis_sold = np.where(holdings > 0, basis * sales / safe_holdings, 0.0)
    cost_rate = np.asarray(transaction_cost_rate, dtype=float)
    if cost_rate.ndim == 1:
        cost_rate = cost_rate[:, None]
    realized = sales * (1.0 - cost_rate) - basis_sold
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
    requested: float | np.ndarray,
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
    requested_values = np.asarray(requested, dtype=float)
    if requested_values.ndim == 0:
        requested_values = np.full(paths, max(float(requested_values), 0.0), dtype=float)
    if requested_values.shape != (paths,):
        raise ValueError("requested withdrawal must match the number of paths.")
    cash_required = np.maximum(requested_values, 0.0).copy()
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


def _raise_cash_deferred(
    requested: float | np.ndarray,
    holdings: np.ndarray,
    basis: np.ndarray,
    profile: _TaxProfile,
    loss_buckets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Raise a withdrawal while accruing, rather than funding, disposal tax."""

    values = holdings.sum(axis=1)
    requested_values = np.asarray(requested, dtype=float)
    if requested_values.ndim == 0:
        requested_values = np.full(len(values), max(float(requested_values), 0.0))
    if requested_values.shape != values.shape:
        raise ValueError("requested withdrawal must match the number of paths.")
    gross_sale = np.minimum(np.maximum(requested_values, 0.0), values)
    sales = np.divide(
        holdings * gross_sale[:, None],
        values[:, None],
        out=np.zeros_like(holdings),
        where=values[:, None] > 0,
    )
    return _apply_sales(sales, holdings, basis, profile, loss_buckets)


def _rebalance_after_tax_detailed(
    holdings: np.ndarray,
    basis: np.ndarray,
    target_weights: np.ndarray,
    cost_rate: float | np.ndarray,
    profile: _TaxProfile,
    loss_buckets: np.ndarray,
    *,
    settle_tax_immediately: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Rebalance with disposal tax, transaction costs, and per-asset FTT."""

    values = holdings.sum(axis=1)
    path_cost_rate = np.asarray(cost_rate, dtype=float)
    if path_cost_rate.ndim == 0:
        path_cost_rate = np.full(len(holdings), float(path_cost_rate), dtype=float)
    if path_cost_rate.shape != (len(holdings),):
        raise ValueError("Path transaction cost rates must match the number of paths.")
    post_cost_value = values.copy()
    for _ in range(8):
        target = post_cost_value[:, None] * target_weights
        sales = np.maximum(holdings - target, 0.0)
        purchases = np.maximum(target - holdings, 0.0)
        preview_tax = (
            _preview_sales_tax(sales, holdings, basis, profile, loss_buckets, cost_rate)
            if settle_tax_immediately
            else np.zeros(len(holdings), dtype=float)
        )
        costs = (sales + purchases).sum(axis=1) * path_cost_rate
        ftt = (purchases * profile.financial_transaction_tax_rate[None, :]).sum(axis=1)
        updated = np.maximum(values - preview_tax - costs - ftt, 0.0)
        if np.allclose(updated, post_cost_value, rtol=1e-11, atol=1e-10):
            post_cost_value = updated
            break
        post_cost_value = updated

    target = post_cost_value[:, None] * target_weights
    sales = np.maximum(holdings - target, 0.0)
    tax, gains, losses = _apply_sales(sales, holdings, basis, profile, loss_buckets, cost_rate)
    purchases = np.maximum(target - holdings, 0.0)
    costs = (sales + purchases).sum(axis=1) * path_cost_rate
    ftt = (purchases * profile.financial_transaction_tax_rate[None, :]).sum(axis=1)
    basis += purchases * (
        1.0 + path_cost_rate[:, None] + profile.financial_transaction_tax_rate[None, :]
    )
    holdings += purchases
    return tax, gains, losses, ftt, costs


def _deduct_charge(
    holdings: np.ndarray,
    basis: np.ndarray,
    charge: np.ndarray,
) -> None:
    values = holdings.sum(axis=1)
    scale = np.divide(
        np.maximum(values - np.maximum(charge, 0.0), 0.0),
        values,
        out=np.zeros_like(values),
        where=values > 0,
    )
    holdings *= scale[:, None]
    basis *= scale[:, None]


def _simulation_tax_years(periods: int, start_date: str | pd.Timestamp | None) -> np.ndarray:
    if start_date is None:
        return 2000 + np.arange(periods, dtype=int) // 12
    first = pd.Timestamp(start_date).to_period("M")
    return np.array([(first + period).year for period in range(periods)], dtype=int)


def _wealth_tax_masks(profile: _TaxProfile, mode: str) -> tuple[np.ndarray, np.ndarray]:
    eligible = profile.wealth_tax_eligible
    domestic = eligible & ~profile.foreign_account
    foreign = eligible & profile.foreign_account
    if mode == "none":
        return np.zeros_like(eligible), np.zeros_like(eligible)
    if mode == "stamp_duty":
        return domestic, np.zeros_like(eligible)
    if mode == "ivafe":
        return np.zeros_like(eligible), foreign
    if mode in {"auto", "both"}:
        return domestic, foreign
    raise ValueError(f"Unknown Italian wealth tax mode '{mode}'.")


def _wrapper_taxable_fraction(holdings: np.ndarray, profile: _TaxProfile) -> np.ndarray:
    values = holdings.sum(axis=1)
    return np.divide(
        holdings @ profile.taxable_fraction,
        values,
        out=np.ones_like(values),
        where=values > 0,
    )


def _sell_wrapper_units(
    requested: np.ndarray,
    holdings: np.ndarray,
    basis: np.ndarray,
    profile: _TaxProfile,
) -> tuple[np.ndarray, np.ndarray]:
    """Sell wrapper units and return the realized tax and gross proceeds."""

    values = holdings.sum(axis=1)
    gross_sale = np.minimum(np.maximum(requested, 0.0), values)
    fraction = np.divide(gross_sale, values, out=np.zeros_like(values), where=values > 0)
    basis_sold = basis * fraction
    gain = np.maximum(gross_sale - basis_sold, 0.0)
    tax = gain * _wrapper_taxable_fraction(holdings, profile) * ITALY_STANDARD_TAX_RATE
    holdings *= (1.0 - fraction)[:, None]
    basis -= basis_sold
    np.maximum(basis, 0.0, out=basis)
    return tax, gross_sale


def _simulate_wrapper_benchmark(
    growth: np.ndarray,
    target_weights: np.ndarray,
    initial_value: float,
    rebalance_frequency: int,
    transaction_cost_rate_paths: np.ndarray | None,
    cost_rate: float,
    contribution: float,
    withdrawal: float,
    withdrawal_start_period: int,
    decumulation: DecumulationPlan,
    cpi: np.ndarray,
    safe_withdrawal_rate: float,
    contribution_allocation: str,
    profile: _TaxProfile,
    annual_wealth_tax: float,
    wealth_tax_mode: str,
    tax_regime: str,
    years: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate a single accumulating fund whose internal trades are tax-free."""

    periods, paths, asset_count = growth.shape
    holdings = np.broadcast_to(initial_value * target_weights, (paths, asset_count)).copy()
    basis = np.full(paths, float(initial_value), dtype=float)
    previous_value = np.full(paths, float(initial_value), dtype=float)
    pending_tax = np.zeros(paths, dtype=float)
    log_return_sum = np.zeros(paths, dtype=float)
    log_return_count = np.zeros(paths, dtype=np.int32)
    wealth_tax_rate = float(annual_wealth_tax) / 12.0
    stamp_mask, ivafe_mask = _wealth_tax_masks(profile, wealth_tax_mode)
    taxable_mask = stamp_mask | ivafe_mask
    declarative = tax_regime == "italy_declarative"
    spending = SpendingController(
        decumulation,
        paths=paths,
        initial_value=initial_value,
        cpi=cpi,
        safe_rate=safe_withdrawal_rate,
    )

    for period in range(periods):
        if period and years[period] != years[period - 1] and declarative:
            _deduct_charge(holdings, basis[:, None], pending_tax)
            pending_tax.fill(0.0)

        if contribution:
            allocated = contribution_allocations(
                holdings,
                target_weights,
                contribution,
                contribution_allocation,
            )
            purchases = allocated / (1.0 + profile.financial_transaction_tax_rate[None, :])
            holdings += purchases
            basis += float(contribution)

        holdings *= growth[period]

        requested, _ = spending.request(period + 1, holdings.sum(axis=1))
        funded = np.zeros(paths, dtype=float)
        if np.any(requested > 0):
            before_sale = holdings.sum(axis=1).copy()
            if declarative:
                tax, _ = _sell_wrapper_units(requested, holdings, basis, profile)
                pending_tax += tax
                funded = np.minimum(
                    requested,
                    np.maximum(before_sale - holdings.sum(axis=1), 0.0),
                )
            else:
                cash_required = requested
                tax_total = np.zeros(paths, dtype=float)
                for _ in range(12):
                    tax, gross_sale = _sell_wrapper_units(cash_required, holdings, basis, profile)
                    if not np.any(gross_sale > 1e-10):
                        break
                    tax_total += tax
                    cash_required = tax
                gross_sales = np.maximum(before_sale - holdings.sum(axis=1), 0.0)
                funded = np.minimum(requested, np.maximum(gross_sales - tax_total, 0.0))

        if rebalance_frequency > 0 and (period + 1) % rebalance_frequency == 0:
            values = holdings.sum(axis=1)
            target = values[:, None] * target_weights
            sales = np.maximum(holdings - target, 0.0)
            purchases = np.maximum(target - holdings, 0.0)
            active_cost_rate = (
                transaction_cost_rate_paths[period]
                if transaction_cost_rate_paths is not None
                else np.full(paths, cost_rate, dtype=float)
            )
            charges = (sales + purchases).sum(axis=1) * active_cost_rate
            charges += (purchases * profile.financial_transaction_tax_rate[None, :]).sum(axis=1)
            remaining = np.maximum(values - charges, 0.0)
            holdings = remaining[:, None] * target_weights

        if wealth_tax_rate:
            charge = holdings[:, taxable_mask].sum(axis=1) * wealth_tax_rate
            _deduct_charge(holdings, basis[:, None], charge)

        value = holdings.sum(axis=1)
        denominator = previous_value + float(contribution)
        numerator = value + funded
        valid = (denominator > 0) & (numerator > 0)
        log_return_sum[valid] += np.log(numerator[valid] / denominator[valid])
        log_return_count[valid] += 1
        previous_value = value

    terminal = holdings.sum(axis=1)
    if periods:
        terminal_before_tax = terminal.copy()
        liquidation_gain = np.maximum(terminal - basis, 0.0)
        terminal_tax = (
            liquidation_gain
            * _wrapper_taxable_fraction(holdings, profile)
            * ITALY_STANDARD_TAX_RATE
        )
        terminal = np.maximum(terminal - terminal_tax - pending_tax, 0.0)
        valid_terminal = (terminal_before_tax > 0) & (terminal > 0)
        log_return_sum[valid_terminal] += np.log(
            terminal[valid_terminal] / terminal_before_tax[valid_terminal]
        )

    annualized = np.zeros(paths, dtype=float)
    valid_count = log_return_count > 0
    annualized[valid_count] = (
        np.exp(log_return_sum[valid_count] / log_return_count[valid_count] * 12.0) - 1.0
    )
    return terminal, annualized


def prepare_italian_native_configuration(
    *,
    periods: int,
    assets: list[str],
    target_weights: np.ndarray,
    initial_value: float,
    rebalance_frequency: int,
    transaction_cost_bps: float,
    contribution: float,
    contribution_allocation: str,
    withdrawal: float,
    withdrawal_start_period: int,
    asset_tax_categories: Mapping[str, str] | None,
    asset_tax_metadata: Mapping[str, Mapping[str, object]] | None,
    annual_wealth_tax: float,
    terminal_liquidation: bool,
    tax_regime: str,
    wealth_tax_mode: str,
    start_date: str | pd.Timestamp | None,
    wrapper_benchmark: bool,
) -> dict[str, object]:
    """Prepare immutable inputs and reporting metadata for the fused kernel."""

    profile = _tax_profile(assets, asset_tax_categories, asset_tax_metadata)
    if not np.allclose(profile.annual_income_yield, 0.0):
        raise ValueError("The fused native kernel requires accumulating/total-return instruments.")
    years = _simulation_tax_years(periods, start_date)
    ordered_years = list(dict.fromkeys(int(year) for year in years))
    year_index = {year: index for index, year in enumerate(ordered_years)}
    year_slots = np.array([year_index[int(year)] for year in years], dtype=np.int32)
    stamp_mask, ivafe_mask = _wealth_tax_masks(profile, wealth_tax_mode)
    managed = tax_regime == "italy_managed"
    declarative = tax_regime == "italy_declarative"
    wrapper_available = bool(wrapper_benchmark and terminal_liquidation and not managed)
    categories = {asset: str(profile.metadata[asset]["category"]) for asset in assets}
    return {
        "native_kwargs": {
            "target_weights": np.asarray(target_weights, dtype=float),
            "initial_value": float(initial_value),
            "rebalance_frequency": int(rebalance_frequency),
            "transaction_cost_bps": float(transaction_cost_bps),
            "transaction_cost_rate_paths": None,
            "contribution": float(contribution),
            "contribution_allocation": contribution_allocation,
            "withdrawal": float(withdrawal),
            "withdrawal_start_period": int(withdrawal_start_period),
            "tax_regime": tax_regime,
            "taxable_fraction": profile.taxable_fraction,
            "gains_offsettable": profile.gains_offsettable,
            "financial_transaction_tax_rate": profile.financial_transaction_tax_rate,
            "stamp_mask": stamp_mask,
            "ivafe_mask": ivafe_mask,
            "annual_wealth_tax": float(annual_wealth_tax),
            "terminal_liquidation": bool(terminal_liquidation),
            "wrapper_benchmark": wrapper_available,
            "year_slots": year_slots,
        },
        "frame_metadata": {
            "assets": list(assets),
            "asset_tax_categories": categories,
            "asset_tax_metadata": profile.metadata,
            "tax_regime": tax_regime,
            "annual_wealth_tax": float(annual_wealth_tax),
            "wealth_tax_mode": wealth_tax_mode,
            "terminal_liquidation": bool(terminal_liquidation),
            "ordered_years": ordered_years,
            "declarative": declarative,
            "managed": managed,
            "start_date": str(pd.Timestamp(start_date).date()) if start_date is not None else None,
            "contribution_allocation": contribution_allocation,
            "withdrawal_start_period": int(withdrawal_start_period),
            "wrapper_requested": bool(wrapper_benchmark),
            "wrapper_available": wrapper_available,
        },
    }


def italian_native_result_frame(
    native_result: Mapping[str, object],
    frame_metadata: Mapping[str, object],
    *,
    fused: bool,
) -> pd.DataFrame:
    """Convert native ledger arrays into the reference DataFrame contract."""

    wealth = np.asarray(native_result["wealth"], dtype=float)
    paths = wealth.shape[1]
    year_metric_names = (
        "capital_gains_tax",
        "managed_result_tax",
        "deferred_tax_payment",
        "expired_losses",
        "financial_transaction_tax",
        "stamp_duty",
        "ivafe",
        "terminal_liquidation_tax",
        "gross_sales_for_spending",
        "net_spending",
    )
    ordered_years = list(frame_metadata["ordered_years"])
    year_values = np.asarray(native_result["year_stats"], dtype=float)
    tax_by_year = {
        str(year): {
            name: float(year_values[index, metric])
            for metric, name in enumerate(year_metric_names)
        }
        for index, year in enumerate(ordered_years)
    }
    stats = native_result["tax_stats"]
    if not isinstance(stats, Mapping):
        raise RuntimeError("Native tax ledger returned invalid statistics.")
    wrapper_requested = bool(frame_metadata["wrapper_requested"])
    wrapper_available = bool(frame_metadata["wrapper_available"])
    managed = bool(frame_metadata["managed"])
    declarative = bool(frame_metadata["declarative"])
    frame = pd.DataFrame(wealth, columns=[f"path_{index}" for index in range(paths)])
    frame.attrs.update(
        {
            "margin_calls": 0,
            "tax_regime": str(frame_metadata["tax_regime"]),
            "asset_tax_categories": dict(frame_metadata["asset_tax_categories"]),
            "asset_tax_metadata": dict(frame_metadata["asset_tax_metadata"]),
            **{
                str(name): float(np.asarray(values, dtype=float).sum())
                for name, values in stats.items()
            },
            "annual_wealth_tax": float(frame_metadata["annual_wealth_tax"]),
            "wealth_tax_mode": str(frame_metadata["wealth_tax_mode"]),
            "tax_terminal_liquidation": bool(frame_metadata["terminal_liquidation"]),
            "tax_by_year": tax_by_year,
            "tax_basis_method": (
                "average_cost" if not declarative else "average_cost_planning_proxy"
            ),
            "tax_timing": "annual" if declarative or managed else "transaction",
            "start_date": frame_metadata["start_date"],
            "contribution_allocation": str(frame_metadata["contribution_allocation"]),
            "withdrawal_start_period": int(frame_metadata["withdrawal_start_period"]),
            "tax_wrapper_benchmark_requested": wrapper_requested,
            "tax_wrapper_benchmark_available": wrapper_available,
            "tax_wrapper_unavailable_reason": (
                "managed_regime"
                if wrapper_requested and managed
                else "terminal_liquidation_required"
                if wrapper_requested and not bool(frame_metadata["terminal_liquidation"])
                else None
            ),
            "wrapper_terminal_values": native_result["wrapper_terminal_values"],
            "wrapper_annualized_returns": native_result["wrapper_annualized_returns"],
            "native_backend": True,
            "native_fused_backend": bool(fused),
            "native_gross_wealth": np.asarray(native_result["gross_wealth"], dtype=float),
            "native_gross_transaction_cost_total": float(
                native_result["gross_transaction_cost_total"]
            ),
        }
    )
    return frame


def simulate_italian_portfolio_tax(
    asset_growth: np.ndarray,
    assets: list[str],
    target_weights: np.ndarray,
    initial_value: float,
    rebalance_frequency: int,
    transaction_cost_bps: float = 0.0,
    transaction_cost_rate_paths: np.ndarray | None = None,
    contribution: float = 0.0,
    contribution_allocation: str = "target",
    withdrawal: float = 0.0,
    withdrawal_start_period: int = 1,
    decumulation: Mapping[str, object] | DecumulationPlan | None = None,
    withdrawal_inflation_paths: np.ndarray | None = None,
    annual_inflation: float = 0.0,
    safe_withdrawal_rate: float = 0.0,
    asset_tax_categories: Mapping[str, str] | None = None,
    asset_tax_metadata: Mapping[str, Mapping[str, object]] | None = None,
    annual_wealth_tax: float = ITALY_DEFAULT_WEALTH_TAX_RATE,
    terminal_liquidation: bool = True,
    tax_regime: str = "italy_administered",
    wealth_tax_mode: str = "auto",
    start_date: str | pd.Timestamp | None = None,
    wrapper_benchmark: bool = False,
    native_threads: int = 1,
) -> pd.DataFrame:
    """Simulate Italian investment taxation as an event-driven planning model.

    The engine separates price appreciation from configured distributions,
    handles government-security fractions, foreign withholding, Italian FTT,
    calendar tax years, stamp duty/IVAFE location, and the administered,
    declarative, and managed timing conventions. It remains a planning model,
    not a filing calculation.
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
    contribution_allocation = str(contribution_allocation).strip().lower()
    if contribution_allocation not in CONTRIBUTION_ALLOCATION_MODES:
        allowed = ", ".join(sorted(CONTRIBUTION_ALLOCATION_MODES))
        raise ValueError(f"Unknown contribution allocation '{contribution_allocation}'. Expected: {allowed}.")
    tax_regime = str(tax_regime).strip().lower()
    if tax_regime not in ITALY_TAX_REGIMES:
        raise ValueError(f"Unknown Italian tax regime '{tax_regime}'.")
    wealth_tax_mode = str(wealth_tax_mode).strip().lower()
    if wealth_tax_mode not in ITALY_WEALTH_TAX_MODES:
        raise ValueError(f"Unknown Italian wealth tax mode '{wealth_tax_mode}'.")

    profile = _tax_profile(assets, asset_tax_categories, asset_tax_metadata)
    periods, paths, asset_count = growth.shape
    try:
        withdrawal_start_value = float(withdrawal_start_period)
        withdrawal_start_period = int(withdrawal_start_value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("withdrawal_start_period must be an integer.") from exc
    if not np.isfinite(withdrawal_start_value) or not np.isclose(
        withdrawal_start_value, withdrawal_start_period
    ):
        raise ValueError("withdrawal_start_period must be an integer.")
    if not 1 <= withdrawal_start_period <= periods:
        raise ValueError(
            "withdrawal_start_period must be between 1 and the simulation periods."
        )
    decumulation_plan = normalize_decumulation(
        decumulation,
        periods=periods,
        legacy_withdrawal=withdrawal,
        legacy_start_period=withdrawal_start_period,
        annual_inflation_fallback=annual_inflation,
    )
    cpi = inflation_index(
        periods,
        paths,
        annual_inflation=decumulation_plan.annual_inflation_fallback,
        inflation_paths=withdrawal_inflation_paths,
    )
    holdings = np.broadcast_to(initial_value * weights, (paths, asset_count)).copy()
    basis = holdings.copy()
    wealth = np.empty((periods, paths), dtype=float)
    loss_buckets = np.zeros((ITALY_LOSS_CARRY_YEARS + 1, paths), dtype=float)
    cost_rate = float(transaction_cost_bps) / 10_000.0
    if transaction_cost_rate_paths is not None:
        path_cost_rates = np.asarray(transaction_cost_rate_paths, dtype=float)
        if path_cost_rates.shape != (periods, paths):
            raise ValueError("transaction_cost_rate_paths must have shape (periods, paths).")
        if not np.isfinite(path_cost_rates).all() or (path_cost_rates < 0).any():
            raise ValueError("transaction_cost_rate_paths must be finite and non-negative.")
    else:
        path_cost_rates = None
    wealth_tax_rate = float(annual_wealth_tax) / 12.0
    years = _simulation_tax_years(periods, start_date)
    stamp_mask, ivafe_mask = _wealth_tax_masks(profile, wealth_tax_mode)
    managed = tax_regime == "italy_managed"
    declarative = tax_regime == "italy_declarative"

    # Accumulating/total-return instruments need no distribution ledger, so
    # the complete gross + DIY + optional wrapper accounting can run in C++.
    # Any unsupported profile or unavailable/incompatible library falls back
    # to the reference implementation below without changing the API.
    native_result: dict[str, object] | None = None
    native_wrapper = bool(wrapper_benchmark and terminal_liquidation and not managed)
    if np.allclose(profile.annual_income_yield, 0.0):
        ordered_years = list(dict.fromkeys(int(year) for year in years))
        year_index = {year: index for index, year in enumerate(ordered_years)}
        year_slots = np.array([year_index[int(year)] for year in years], dtype=np.int32)
        try:
            native_result = simulate_italian_portfolios_native(
                growth,
                weights,
                initial_value=initial_value,
                rebalance_frequency=rebalance_frequency,
                transaction_cost_bps=transaction_cost_bps,
                transaction_cost_rate_paths=path_cost_rates,
                contribution=contribution,
                contribution_allocation=contribution_allocation,
                withdrawal=withdrawal,
                withdrawal_start_period=withdrawal_start_period,
                tax_regime=tax_regime,
                taxable_fraction=profile.taxable_fraction,
                gains_offsettable=profile.gains_offsettable,
                financial_transaction_tax_rate=profile.financial_transaction_tax_rate,
                stamp_mask=stamp_mask,
                ivafe_mask=ivafe_mask,
                annual_wealth_tax=annual_wealth_tax,
                terminal_liquidation=terminal_liquidation,
                wrapper_benchmark=native_wrapper,
                year_slots=year_slots,
                decumulation=decumulation_plan,
                withdrawal_cpi=cpi,
                safe_withdrawal_rate=safe_withdrawal_rate,
                workers=native_threads,
            )
        except (AttributeError, OSError, RuntimeError):
            native_result = None
        if native_result is not None:
            year_metric_names = (
                "capital_gains_tax",
                "managed_result_tax",
                "deferred_tax_payment",
                "expired_losses",
                "financial_transaction_tax",
                "stamp_duty",
                "ivafe",
                "terminal_liquidation_tax",
                "gross_sales_for_spending",
                "net_spending",
            )
            year_values = np.asarray(native_result["year_stats"], dtype=float)
            tax_by_year = {
                str(year): {
                    name: float(year_values[index, metric])
                    for metric, name in enumerate(year_metric_names)
                }
                for index, year in enumerate(ordered_years)
            }
            stats = native_result["tax_stats"]
            if not isinstance(stats, Mapping):
                raise RuntimeError("Native tax ledger returned invalid statistics.")
            wrapper_terminal = native_result["wrapper_terminal_values"]
            wrapper_annualized = native_result["wrapper_annualized_returns"]
            frame = pd.DataFrame(
                np.asarray(native_result["wealth"], dtype=float),
                columns=[f"path_{index}" for index in range(paths)],
            )
            categories = {asset: str(profile.metadata[asset]["category"]) for asset in assets}
            frame.attrs.update(
                {
                    "margin_calls": 0,
                    "tax_regime": tax_regime,
                    "asset_tax_categories": categories,
                    "asset_tax_metadata": profile.metadata,
                    **{
                        str(name): float(np.asarray(values, dtype=float).sum())
                        for name, values in stats.items()
                    },
                    "annual_wealth_tax": float(annual_wealth_tax),
                    "wealth_tax_mode": wealth_tax_mode,
                    "tax_terminal_liquidation": bool(terminal_liquidation),
                    "tax_by_year": tax_by_year,
                    "tax_basis_method": (
                        "average_cost" if not declarative else "average_cost_planning_proxy"
                    ),
                    "tax_timing": "annual" if declarative or managed else "transaction",
                    "start_date": (
                        str(pd.Timestamp(start_date).date()) if start_date is not None else None
                    ),
                    "contribution_allocation": contribution_allocation,
                    "withdrawal_start_period": withdrawal_start_period,
                    "tax_wrapper_benchmark_requested": bool(wrapper_benchmark),
                    "tax_wrapper_benchmark_available": native_wrapper,
                    "tax_wrapper_unavailable_reason": (
                        "managed_regime"
                        if wrapper_benchmark and managed
                        else "terminal_liquidation_required"
                        if wrapper_benchmark and not terminal_liquidation
                        else None
                    ),
                    "wrapper_terminal_values": wrapper_terminal,
                    "wrapper_annualized_returns": wrapper_annualized,
                    "native_backend": True,
                    "native_gross_wealth": np.asarray(
                        native_result["gross_wealth"], dtype=float
                    ),
                    "native_gross_transaction_cost_total": float(
                        native_result["gross_transaction_cost_total"]
                    ),
                    "decumulation": decumulation_plan.to_dict(),
                    "withdrawal_requested": (
                        np.asarray(native_result["withdrawal_requested"], dtype=float)
                        if native_result.get("withdrawal_requested") is not None
                        else np.broadcast_to(
                            (
                                float(withdrawal)
                                * (np.arange(1, periods + 1) >= withdrawal_start_period)
                            )[:, None],
                            (periods, paths),
                        ).copy()
                    ),
                    "withdrawal_funded": (
                        np.asarray(native_result["withdrawal_funded"], dtype=float)
                        if native_result.get("withdrawal_funded") is not None
                        else np.minimum(
                            np.broadcast_to(
                                (
                                    float(withdrawal)
                                    * (np.arange(1, periods + 1) >= withdrawal_start_period)
                                )[:, None],
                                (periods, paths),
                            ),
                            np.vstack(
                                [
                                    np.full((1, paths), initial_value),
                                    np.asarray(native_result["wealth"], dtype=float)[:-1],
                                ]
                            ),
                        )
                    ),
                    "guardrail_events": (
                        np.asarray(native_result["guardrail_events"], dtype=np.int8)
                        if native_result.get("guardrail_events") is not None
                        else np.zeros((periods, paths), dtype=np.int8)
                    ),
                    "withdrawal_cpi": cpi,
                }
            )
            return frame

    capital_gains_tax = np.zeros(paths, dtype=float)
    investment_income_tax = np.zeros(paths, dtype=float)
    foreign_withholding_tax = np.zeros(paths, dtype=float)
    financial_transaction_tax = np.zeros(paths, dtype=float)
    transaction_cost_total = np.zeros(paths, dtype=float)
    wealth_tax = np.zeros(paths, dtype=float)
    stamp_duty = np.zeros(paths, dtype=float)
    ivafe = np.zeros(paths, dtype=float)
    terminal_tax = np.zeros(paths, dtype=float)
    realized_gains = np.zeros(paths, dtype=float)
    realized_losses = np.zeros(paths, dtype=float)
    expired_losses = np.zeros(paths, dtype=float)
    pending_tax = np.zeros(paths, dtype=float)
    tax_by_year: dict[str, dict[str, float]] = {}
    year_start_value = np.full(paths, initial_value, dtype=float)
    year_contributions = np.zeros(paths, dtype=float)
    year_withdrawals = np.zeros(paths, dtype=float)
    spending = SpendingController(
        decumulation_plan,
        paths=paths,
        initial_value=initial_value,
        cpi=cpi,
        safe_rate=safe_withdrawal_rate,
    )
    requested_spending = np.zeros((periods, paths), dtype=float)
    funded_spending = np.zeros((periods, paths), dtype=float)
    guardrail_events = np.zeros((periods, paths), dtype=np.int8)

    def record(year: int, name: str, amount: np.ndarray) -> None:
        bucket = tax_by_year.setdefault(str(int(year)), {})
        bucket[name] = bucket.get(name, 0.0) + float(np.asarray(amount, dtype=float).sum())

    def settle_managed_year(year: int) -> None:
        nonlocal year_start_value, year_contributions, year_withdrawals
        values = holdings.sum(axis=1)
        result = values + year_withdrawals - year_contributions - year_start_value
        weighted_fraction = float(target_weights @ profile.taxable_fraction)
        tax_base = result * weighted_fraction
        loss_buckets[-1] += np.maximum(-tax_base, 0.0)
        uncovered = _consume_losses(loss_buckets, np.maximum(tax_base, 0.0))
        tax = uncovered * ITALY_STANDARD_TAX_RATE
        _deduct_charge(holdings, basis, tax)
        capital_gains_tax[:] += tax
        record(year, "managed_result_tax", tax)
        year_start_value = holdings.sum(axis=1).copy()
        year_contributions.fill(0.0)
        year_withdrawals.fill(0.0)

    for period in range(periods):
        if period and years[period] != years[period - 1]:
            if managed:
                settle_managed_year(int(years[period - 1]))
            elif declarative and np.any(pending_tax > 0):
                _deduct_charge(holdings, basis, pending_tax)
                record(int(years[period - 1]), "deferred_tax_payment", pending_tax)
                pending_tax.fill(0.0)
            expired = _advance_tax_year(loss_buckets)
            expired_losses += expired
            record(int(years[period - 1]), "expired_losses", expired)
        if contribution:
            cash_allocated = contribution_allocations(
                holdings,
                weights,
                contribution,
                contribution_allocation,
            )
            purchases = cash_allocated / (1.0 + profile.financial_transaction_tax_rate)
            ftt = cash_allocated - purchases
            holdings += purchases
            basis += cash_allocated
            financial_transaction_tax += ftt.sum(axis=1)
            year_contributions += float(contribution)
            record(int(years[period]), "financial_transaction_tax", ftt.sum(axis=1))

        opening = holdings.copy()
        monthly_income_yield = profile.annual_income_yield / 12.0
        distributions = opening * monthly_income_yield[None, :]
        price_growth = growth[period] - monthly_income_yield[None, :]
        if (price_growth <= 0).any():
            raise ValueError("Configured income yield is incompatible with a simulated return below -100%.")
        holdings = opening * price_growth
        source_tax_components = distributions * profile.foreign_withholding_rate[None, :]
        source_tax = source_tax_components.sum(axis=1)
        italian_income_liability = (
            distributions * profile.taxable_fraction[None, :] * ITALY_STANDARD_TAX_RATE
        )
        credits = np.minimum(
            source_tax_components,
            distributions * profile.foreign_tax_credit_rate[None, :],
        )
        italian_income_tax_components = np.maximum(italian_income_liability - credits, 0.0)
        italian_income = italian_income_tax_components.sum(axis=1)
        if managed:
            italian_income_tax_components.fill(0.0)
            italian_income.fill(0.0)
        net_distributions = distributions - source_tax_components - italian_income_tax_components
        holdings += net_distributions
        basis += net_distributions
        investment_income_tax += italian_income
        foreign_withholding_tax += source_tax
        record(int(years[period]), "investment_income_tax", italian_income)
        record(int(years[period]), "foreign_withholding_tax", source_tax)

        available_before_withdrawal = holdings.sum(axis=1)
        requested, policy_events = spending.request(
            period + 1, available_before_withdrawal
        )
        requested_spending[period] = requested
        guardrail_events[period] = policy_events
        if np.any(requested > 0):
            before_sale = holdings.sum(axis=1).copy()
            immediate_withdrawal_tax = np.zeros(paths, dtype=float)
            if managed:
                values = holdings.sum(axis=1)
                gross_sale = np.minimum(requested, values)
                fraction = np.divide(gross_sale, values, out=np.zeros_like(values), where=values > 0)
                holdings *= (1.0 - fraction)[:, None]
                basis *= (1.0 - fraction)[:, None]
            else:
                settlement = _raise_cash_deferred if declarative else _raise_cash_with_tax
                taxes, gains, losses = settlement(
                    requested,
                    holdings,
                    basis,
                    profile,
                    loss_buckets,
                )
                capital_gains_tax += taxes
                pending_tax += taxes if declarative else 0.0
                if not declarative:
                    immediate_withdrawal_tax = taxes
                realized_gains += gains
                realized_losses += losses
                record(int(years[period]), "capital_gains_tax", taxes)
            gross_sales = np.maximum(before_sale - holdings.sum(axis=1), 0.0)
            # Administered-regime tax is withheld from the disposal proceeds.  If
            # the portfolio is exhausted, only the residual cash reaches the
            # investor; with sufficient assets gross sales already include the
            # extra disposal needed to fund the tax.  Declarative and managed
            # tax is settled later, so the current-period net cash is the sale.
            spendable_cash = (
                gross_sales
                if managed or declarative
                else np.maximum(gross_sales - immediate_withdrawal_tax, 0.0)
            )
            funded = np.minimum(requested, spendable_cash)
            funded_spending[period] = funded
            year_withdrawals += funded
            record(int(years[period]), "gross_sales_for_spending", gross_sales)
            record(int(years[period]), "net_spending", funded)

        if rebalance_frequency > 0 and (period + 1) % rebalance_frequency == 0:
            active_cost_rate = path_cost_rates[period] if path_cost_rates is not None else cost_rate
            if managed:
                values = holdings.sum(axis=1)
                target = values[:, None] * weights
                turnover = np.abs(target - holdings)
                purchases = np.maximum(target - holdings, 0.0)
                charges = turnover.sum(axis=1) * active_cost_rate + (
                    purchases * profile.financial_transaction_tax_rate[None, :]
                ).sum(axis=1)
                transaction_cost_total += turnover.sum(axis=1) * active_cost_rate
                ftt = (purchases * profile.financial_transaction_tax_rate[None, :]).sum(axis=1)
                _deduct_charge(holdings, basis, charges)
                holdings[:] = holdings.sum(axis=1)[:, None] * weights
                basis[:] = holdings
            else:
                taxes, gains, losses, ftt, costs = _rebalance_after_tax_detailed(
                    holdings,
                    basis,
                    weights,
                    active_cost_rate,
                    profile,
                    loss_buckets,
                    settle_tax_immediately=not declarative,
                )
                capital_gains_tax += taxes
                pending_tax += taxes if declarative else 0.0
                realized_gains += gains
                realized_losses += losses
                record(int(years[period]), "capital_gains_tax", taxes)
                transaction_cost_total += costs
            financial_transaction_tax += ftt
            record(int(years[period]), "financial_transaction_tax", ftt)

        if wealth_tax_rate:
            stamp_charge = holdings[:, stamp_mask].sum(axis=1) * wealth_tax_rate
            ivafe_charge = holdings[:, ivafe_mask].sum(axis=1) * wealth_tax_rate
            charge = stamp_charge + ivafe_charge
            _deduct_charge(holdings, basis, charge)
            wealth_tax += charge
            stamp_duty += stamp_charge
            ivafe += ivafe_charge
            record(int(years[period]), "stamp_duty", stamp_charge)
            record(int(years[period]), "ivafe", ivafe_charge)
        wealth[period] = holdings.sum(axis=1)

    if periods:
        final_year = int(years[-1])
        if managed:
            settle_managed_year(final_year)
            wealth[-1] = holdings.sum(axis=1)
        elif terminal_liquidation:
            sales = holdings.copy()
            taxes, gains, losses = _settle_sales(sales, holdings, basis, profile, loss_buckets)
            terminal_tax += taxes
            realized_gains += gains
            realized_losses += losses
            payment = taxes + (pending_tax if declarative else 0.0)
            record(final_year, "terminal_liquidation_tax", taxes)
            wealth[-1] = np.maximum(wealth[-1] - payment, 0.0)
            pending_tax.fill(0.0)
        elif declarative and np.any(pending_tax > 0):
            _deduct_charge(holdings, basis, pending_tax)
            record(final_year, "deferred_tax_payment", pending_tax)
            pending_tax.fill(0.0)
            wealth[-1] = holdings.sum(axis=1)

    wrapper_available = bool(
        wrapper_benchmark and terminal_liquidation and tax_regime != "italy_managed"
    )
    wrapper_terminal: np.ndarray | None = None
    wrapper_annualized: np.ndarray | None = None
    if wrapper_available:
        wrapper_terminal, wrapper_annualized = _simulate_wrapper_benchmark(
            growth,
            weights,
            initial_value,
            rebalance_frequency,
            path_cost_rates,
            cost_rate,
            contribution,
            withdrawal,
            withdrawal_start_period,
            decumulation_plan,
            cpi,
            safe_withdrawal_rate,
            contribution_allocation,
            profile,
            annual_wealth_tax,
            wealth_tax_mode,
            tax_regime,
            years,
        )

    frame = pd.DataFrame(wealth, columns=[f"path_{index}" for index in range(paths)])
    categories = {asset: str(profile.metadata[asset]["category"]) for asset in assets}
    total_taxes = (
        capital_gains_tax
        + investment_income_tax
        + foreign_withholding_tax
        + financial_transaction_tax
        + wealth_tax
        + terminal_tax
    )
    frame.attrs.update(
        {
            "margin_calls": 0,
            "tax_regime": tax_regime,
            "asset_tax_categories": categories,
            "asset_tax_metadata": profile.metadata,
            "capital_gains_tax_total": float(capital_gains_tax.sum()),
            "investment_income_tax_total": float(investment_income_tax.sum()),
            "foreign_withholding_tax_total": float(foreign_withholding_tax.sum()),
            "financial_transaction_tax_total": float(financial_transaction_tax.sum()),
            "wealth_tax_total": float(wealth_tax.sum()),
            "stamp_duty_total": float(stamp_duty.sum()),
            "ivafe_total": float(ivafe.sum()),
            "terminal_liquidation_tax_total": float(terminal_tax.sum()),
            "taxes_paid_total": float(total_taxes.sum()),
            "realized_gains_total": float(realized_gains.sum()),
            "realized_losses_total": float(realized_losses.sum()),
            "loss_carryforward_total": float(loss_buckets.sum()),
            "expired_losses_total": float(expired_losses.sum()),
            "transaction_cost_total": float(transaction_cost_total.sum()),
            "annual_wealth_tax": float(annual_wealth_tax),
            "wealth_tax_mode": wealth_tax_mode,
            "tax_terminal_liquidation": bool(terminal_liquidation),
            "tax_by_year": tax_by_year,
            "tax_basis_method": "average_cost" if not declarative else "average_cost_planning_proxy",
            "tax_timing": "annual" if declarative or managed else "transaction",
            "start_date": str(pd.Timestamp(start_date).date()) if start_date is not None else None,
            "contribution_allocation": contribution_allocation,
            "withdrawal_start_period": withdrawal_start_period,
            "tax_wrapper_benchmark_requested": bool(wrapper_benchmark),
            "tax_wrapper_benchmark_available": wrapper_available,
            "tax_wrapper_unavailable_reason": (
                "managed_regime"
                if wrapper_benchmark and tax_regime == "italy_managed"
                else "terminal_liquidation_required"
                if wrapper_benchmark and not terminal_liquidation
                else None
            ),
            "wrapper_terminal_values": wrapper_terminal,
            "wrapper_annualized_returns": wrapper_annualized,
            "native_backend": False,
            "decumulation": decumulation_plan.to_dict(),
            "withdrawal_requested": requested_spending,
            "withdrawal_funded": funded_spending,
            "guardrail_events": guardrail_events,
            "withdrawal_cpi": cpi,
        }
    )
    return frame
