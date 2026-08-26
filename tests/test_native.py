import numpy as np
import pytest

from mc_quadrants.decumulation import inflation_index, normalize_decumulation
from mc_quadrants.native import (
    native_available,
    sample_mnts_subordinators_native,
    simulate_italian_portfolios_native,
    simulate_parametric_italian_portfolios_native,
    simulate_parametric_native,
)
from mc_quadrants.taxes import simulate_italian_portfolio_tax

pytestmark = pytest.mark.skipif(not native_available(), reason="native simulator is not compiled")


def _inputs(periods: int = 8, paths: int = 20_000):
    regimes = np.zeros((periods, paths), dtype=np.uint8)
    means = np.array([[0.01, -0.005]])
    covariance = np.array([[[0.04, 0.018], [0.018, 0.09]]])
    volatility = np.sqrt(np.diagonal(covariance, axis1=1, axis2=2))
    correlation = covariance / (
        volatility[:, :, None] * volatility[:, None, :]
    )
    return {
        "regime_codes": regimes,
        "means": means,
        "gaussian_correlation_cholesky": np.linalg.cholesky(correlation),
        "gaussian_correlations": correlation,
        "volatilities": volatility,
        "tail_indexes": np.array([1.5]),
        "temperings": np.array([0.5]),
        "skewness": np.zeros((1, 2)),
        "gaussian_scales": np.ones((1, 2)),
        "random_seed": 42,
        "garch": False,
        "garch_alpha": 0.10,
        "garch_beta": 0.85,
        "dynamic_correlation": False,
        "dcc_alpha": 0.04,
        "dcc_beta": 0.94,
        "dcc_asymmetry": 0.01,
    }


def test_native_mnts_is_reproducible_and_preserves_moments():
    inputs = _inputs()

    first = simulate_parametric_native(**inputs)
    second = simulate_parametric_native(**inputs)

    assert np.array_equal(first, second)
    assert np.allclose(first.mean(axis=(0, 1)), inputs["means"][0], atol=0.002)
    assert np.allclose(
        np.cov(first.reshape(-1, 2), rowvar=False),
        np.diag(inputs["volatilities"][0])
        @ inputs["gaussian_correlations"][0]
        @ np.diag(inputs["volatilities"][0]),
        atol=0.002,
    )


def test_native_tempered_stable_subordinator_matches_theoretical_moments():
    draws = sample_mnts_subordinators_native(300_000, 1.5, 0.5, 123)
    centered = draws - draws.mean()

    assert draws.mean() == pytest.approx(1.0, abs=0.01)
    assert draws.var() == pytest.approx(0.5, abs=0.025)
    assert np.mean(centered**3) == pytest.approx(1.25, abs=0.15)


def test_native_tempered_stable_subordinator_fallback_matches_moments():
    # tail_index=1 gives alpha=0.5 and lambda**alpha=2.0, above the
    # simple-rejection cutoff, so this covers the Devroye fallback.
    first = sample_mnts_subordinators_native(300_000, 1.0, 1.0, 321)
    second = sample_mnts_subordinators_native(300_000, 1.0, 1.0, 321)

    assert np.array_equal(first, second)
    assert first.mean() == pytest.approx(1.0, abs=0.01)
    assert first.var() == pytest.approx(0.5, abs=0.025)


def test_native_mnts_innovations_have_asymmetric_fat_tails():
    inputs = _inputs(periods=1, paths=250_000)
    inputs["skewness"] = np.array([[-0.65, 0.30]])
    variance_t = 0.5
    inputs["gaussian_scales"] = np.sqrt(
        1.0 - inputs["skewness"] ** 2 * variance_t
    )

    draws = simulate_parametric_native(**inputs)[0]
    standardized = (draws - inputs["means"][0]) / inputs["volatilities"][0]
    centered = standardized - standardized.mean(axis=0)
    variance = np.mean(centered**2, axis=0)
    skewness = np.mean(centered**3, axis=0) / variance**1.5
    excess_kurtosis = np.mean(centered**4, axis=0) / variance**2 - 3.0

    assert skewness[0] < -0.8
    assert skewness[1] > 0.25
    assert np.all(excess_kurtosis > 1.0)


def test_native_parametric_paths_are_identical_with_one_or_many_threads():
    inputs = _inputs(periods=24, paths=2_000)

    single = simulate_parametric_native(**inputs, workers=1)
    parallel = simulate_parametric_native(**inputs, workers=4)

    assert np.array_equal(single, parallel)


@pytest.mark.parametrize(
    ("garch", "dynamic_correlation"),
    [(False, True), (True, False), (True, True)],
)
def test_native_advanced_models_are_finite(garch, dynamic_correlation):
    inputs = _inputs(periods=48, paths=2_000)
    inputs.update(
        garch=garch,
        dynamic_correlation=dynamic_correlation,
    )

    result = simulate_parametric_native(**inputs)

    assert result.shape == (48, 2_000, 2)
    assert np.isfinite(result).all()


def test_native_joint_macro_effect_is_applied():
    inputs = _inputs(periods=4, paths=100)
    baseline = simulate_parametric_native(**inputs)
    shocks = np.ones((4, 100, 1))
    betas = np.array([[0.03, -0.02]])

    shifted = simulate_parametric_native(
        **inputs,
        macro_shocks=shocks,
        macro_betas=betas,
    )

    assert np.allclose(shifted - baseline, np.broadcast_to(betas, shifted.shape))


@pytest.mark.parametrize(
    "regime",
    ["italy_administered", "italy_declarative", "italy_managed"],
)
def test_native_italian_ledger_matches_python_reference(monkeypatch, regime):
    rng = np.random.default_rng(17)
    growth = np.exp(rng.normal(0.004, 0.04, size=(25, 11, 3)))
    kwargs = {
        "asset_growth": growth,
        "assets": ["Equity", "Mixed", "Bond"],
        "target_weights": np.array([0.5, 0.3, 0.2]),
        "initial_value": 1_234.0,
        "rebalance_frequency": 3,
        "transaction_cost_bps": 7.0,
        "transaction_cost_rate_paths": rng.uniform(0.0002, 0.0015, size=(25, 11)),
        "contribution": 13.0,
        "contribution_allocation": "underweight_first",
        "withdrawal": 4.0,
        "asset_tax_metadata": {
            "Equity": {
                "category": "fund",
                "financial_transaction_tax_rate": 0.001,
                "account_location": "domestic",
            },
            "Mixed": {
                "category": "standard",
                "government_bond_fraction": 0.5,
                "account_location": "foreign",
            },
            "Bond": {
                "category": "government_bond",
                "government_bond_fraction": 1.0,
                "financial_transaction_tax_rate": 0.0005,
                "account_location": "domestic",
            },
        },
        "annual_wealth_tax": 0.002,
        "terminal_liquidation": True,
        "wealth_tax_mode": "auto",
        "start_date": "2025-07-01",
        "wrapper_benchmark": True,
        "tax_regime": regime,
    }

    native = simulate_italian_portfolio_tax(**kwargs, native_threads=4)
    monkeypatch.setenv("MC_DISABLE_NATIVE_SIM", "1")
    reference = simulate_italian_portfolio_tax(**kwargs)

    assert native.attrs["native_backend"] is True
    assert reference.attrs["native_backend"] is False
    assert np.allclose(native, reference, rtol=1e-9, atol=1e-9)
    for name in (
        "capital_gains_tax_total",
        "financial_transaction_tax_total",
        "wealth_tax_total",
        "terminal_liquidation_tax_total",
        "taxes_paid_total",
        "realized_gains_total",
        "realized_losses_total",
        "loss_carryforward_total",
        "expired_losses_total",
        "transaction_cost_total",
    ):
        assert native.attrs[name] == pytest.approx(reference.attrs[name], rel=1e-9, abs=1e-9)
    if regime != "italy_managed":
        assert native.attrs["wrapper_terminal_values"] == pytest.approx(
            reference.attrs["wrapper_terminal_values"], rel=1e-9, abs=1e-9
        )


def test_native_italian_ledger_is_path_identical_across_thread_counts():
    growth = np.exp(np.random.default_rng(23).normal(0.003, 0.05, size=(18, 40, 2)))
    kwargs = {
        "asset_growth": growth,
        "assets": ["A", "B"],
        "target_weights": np.array([0.6, 0.4]),
        "initial_value": 500.0,
        "rebalance_frequency": 3,
        "transaction_cost_bps": 5.0,
        "contribution": 8.0,
        "withdrawal": 2.0,
        "annual_wealth_tax": 0.002,
        "wrapper_benchmark": True,
    }

    single = simulate_italian_portfolio_tax(**kwargs, native_threads=1)
    parallel = simulate_italian_portfolio_tax(**kwargs, native_threads=4)

    assert np.array_equal(single.to_numpy(), parallel.to_numpy())
    assert np.array_equal(
        single.attrs["wrapper_terminal_values"],
        parallel.attrs["wrapper_terminal_values"],
    )


@pytest.mark.parametrize(
    "regime",
    ["italy_administered", "italy_declarative", "italy_managed"],
)
def test_fused_parametric_kernel_matches_separate_generation_and_ledger(regime):
    inputs = _inputs(periods=24, paths=500)
    weights = np.array([0.6, 0.4])
    monthly_fee_log = np.log1p(-np.array([0.001, 0.002])) / 12.0
    year_slots = np.arange(24, dtype=np.int32) // 12
    ledger = {
        "initial_value": 100.0,
        "rebalance_frequency": 3,
        "transaction_cost_bps": 5.0,
        "transaction_cost_rate_paths": None,
        "contribution": 2.0,
        "contribution_allocation": "underweight_first",
        "withdrawal": 1.0,
        "withdrawal_start_period": 5,
        "tax_regime": regime,
        "taxable_fraction": np.array([1.0, 0.125 / 0.26]),
        "gains_offsettable": np.array([False, True]),
        "financial_transaction_tax_rate": np.array([0.001, 0.0]),
        "stamp_mask": np.array([True, False]),
        "ivafe_mask": np.array([False, True]),
        "annual_wealth_tax": 0.002,
        "terminal_liquidation": True,
        "wrapper_benchmark": regime != "italy_managed",
        "year_slots": year_slots,
        "workers": 4,
    }
    returns = simulate_parametric_native(**inputs, workers=4)
    separate = simulate_italian_portfolios_native(
        np.exp(returns + monthly_fee_log),
        weights,
        **ledger,
    )
    fused = simulate_parametric_italian_portfolios_native(
        **inputs,
        monthly_fee_log=monthly_fee_log,
        return_kind="log",
        target_weights=weights,
        **ledger,
    )

    for name in (
        "gross_wealth",
        "wealth",
        "wrapper_terminal_values",
        "wrapper_annualized_returns",
    ):
        assert np.array_equal(fused[name], separate[name])
    assert np.allclose(fused["year_stats"], separate["year_stats"], rtol=1e-14, atol=1e-10)
    for name in fused["tax_stats"]:
        assert np.array_equal(fused["tax_stats"][name], separate["tax_stats"][name])


@pytest.mark.parametrize("policy", ["fixed", "guyton_klinger"])
@pytest.mark.parametrize(
    "regime",
    ["italy_administered", "italy_declarative", "italy_managed"],
)
def test_advanced_decumulation_matches_python_tax_reference(monkeypatch, policy, regime):
    periods = 24
    paths = 40
    rng = np.random.default_rng(812)
    growth = np.exp(rng.normal(0.004, 0.025, size=(periods, paths, 2)))
    plan = normalize_decumulation(
        {
            "mode": "manual",
            "policy": policy,
            "phases": [
                {
                    "start_month": 1,
                    "end_month": 12,
                    "frequency": "monthly",
                    "annual_real_amount": 12,
                },
                {
                    "start_month": 13,
                    "end_month": 24,
                    "frequency": "quarterly",
                    "annual_real_amount": 18,
                },
            ],
            "one_time_expenses": [{"month": 10, "real_amount": 4}],
        },
        periods=periods,
    )
    inflation_paths = np.full((periods, paths), 0.02)
    cpi = inflation_index(periods, paths, inflation_paths=inflation_paths)
    kwargs = {
        "asset_growth": growth,
        "assets": ["ETF", "BTP"],
        "target_weights": np.array([0.65, 0.35]),
        "initial_value": 100.0,
        "rebalance_frequency": 3,
        "transaction_cost_bps": 3.0,
        "decumulation": plan,
        "withdrawal_inflation_paths": inflation_paths,
        "annual_inflation": 0.02,
        "asset_tax_categories": {"ETF": "fund", "BTP": "government_bond"},
        "annual_wealth_tax": 0.002,
        "terminal_liquidation": True,
        "tax_regime": regime,
        "wrapper_benchmark": regime != "italy_managed",
    }
    monkeypatch.delenv("MC_DISABLE_NATIVE_SIM", raising=False)
    native = simulate_italian_portfolio_tax(**kwargs)
    monkeypatch.setenv("MC_DISABLE_NATIVE_SIM", "1")
    reference = simulate_italian_portfolio_tax(**kwargs)

    assert native.attrs["native_backend"] is True
    assert reference.attrs["native_backend"] is False
    assert np.allclose(native.to_numpy(), reference.to_numpy(), rtol=2e-10, atol=2e-8)
    assert np.allclose(native.attrs["withdrawal_cpi"], cpi)
    assert np.allclose(
        native.attrs["withdrawal_requested"],
        reference.attrs["withdrawal_requested"],
        rtol=2e-12,
        atol=2e-10,
    )
    assert np.array_equal(
        native.attrs["guardrail_events"], reference.attrs["guardrail_events"]
    )
    if regime != "italy_managed":
        assert np.allclose(
            native.attrs["wrapper_terminal_values"],
            reference.attrs["wrapper_terminal_values"],
            rtol=2e-10,
            atol=2e-8,
        )
