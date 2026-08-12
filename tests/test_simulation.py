import numpy as np
import pandas as pd
import pytest

from mc_quadrants.calibration import calibrate_quadrant_model
from mc_quadrants.regimes import Regime
from mc_quadrants.simulation import (
    _batched_cholesky,
    simulate_portfolio_paths,
    simulate_regime_paths,
    simulate_returns,
    stationary_distribution,
    summarize_wealth_risk,
)
from mc_quadrants.types import ScenarioModel, SimulationResult


def _calibrated_model():
    dates = pd.date_range("2020-01-31", periods=48, freq="ME")
    macro = pd.DataFrame(
        {
            "growth": np.tile([2.0, 2.5, -1.0, -1.5], 12),
            "inflation": np.tile([1.0, 4.0, 4.5, 1.2], 12),
        },
        index=dates,
    )
    returns = pd.DataFrame(
        {
            "Stocks": np.linspace(-0.03, 0.04, len(dates)),
            "Bonds": np.linspace(0.02, -0.01, len(dates)),
        },
        index=dates,
    )
    return calibrate_quadrant_model(
        returns,
        macro,
        growth_threshold=0.0,
        inflation_threshold=3.0,
        min_observations=3,
        correlation_overrides={
            Regime.HIGH_GROWTH_HIGH_INFLATION.value: {("Stocks", "Bonds"): 0.30},
            Regime.LOW_GROWTH_LOW_INFLATION.value: {("Stocks", "Bonds"): -0.30},
        },
        override_weight=0.50,
    )


def test_calibration_and_simulation_shapes():
    model = _calibrated_model()

    result = simulate_returns(model, periods=6, paths=10, random_seed=1)
    wealth = simulate_portfolio_paths(result, {"Stocks": 0.6, "Bonds": 0.4})

    assert result.returns.shape == (6, 10, 2)
    assert result.regimes.shape == (6, 10)
    assert wealth.shape == (6, 10)


def test_compact_regime_codes_match_public_labels():
    model = _calibrated_model()

    labels = simulate_regime_paths(model, periods=12, paths=20, random_seed=9)
    codes = simulate_regime_paths(
        model,
        periods=12,
        paths=20,
        random_seed=9,
        return_codes=True,
    )

    assert codes.dtype.kind in "iu"
    assert np.array_equal(labels, np.asarray(model.states, dtype=object)[codes])


def test_vectorized_cholesky_matches_numpy():
    rng = np.random.default_rng(12)
    samples = rng.standard_normal((40, 6, 6))
    positive_definite = samples @ np.swapaxes(samples, 1, 2) + np.eye(6)[None] * 0.1

    actual = _batched_cholesky(positive_definite)
    expected = np.linalg.cholesky(positive_definite)

    assert np.allclose(actual, expected, rtol=1e-10, atol=1e-12)


def test_student_t_sampling_is_reproducible():
    model = _calibrated_model()

    first = simulate_returns(
        model,
        periods=6,
        paths=10,
        random_seed=1,
        distribution="student_t",
        degrees_of_freedom=5,
    )
    second = simulate_returns(
        model,
        periods=6,
        paths=10,
        random_seed=1,
        distribution="student_t",
        degrees_of_freedom=5,
    )

    assert np.array_equal(first.returns, second.returns)
    assert first.distribution == "student_t"
    assert first.degrees_of_freedom == 5.0

    with pytest.raises(ValueError, match="greater than 2"):
        simulate_returns(model, periods=1, paths=1, distribution="student_t", degrees_of_freedom=2)

    bootstrap = simulate_returns(
        model,
        periods=6,
        paths=10,
        random_seed=1,
        distribution="block_bootstrap",
        block_size=3,
    )
    assert bootstrap.distribution == "block_bootstrap"
    assert np.isfinite(bootstrap.returns).all()


def test_rebalancing_transaction_costs_reduce_wealth():
    result = SimulationResult(
        returns=np.array(
            [
                [[0.10, 0.00]],
                [[0.00, 0.10]],
            ]
        ),
        regimes=np.empty((2, 1), dtype=object),
        assets=["Stocks", "Bonds"],
        states=[],
        frequency="M",
    )

    without_costs = simulate_portfolio_paths(
        result,
        {"Stocks": 0.5, "Bonds": 0.5},
        rebalance_frequency=1,
    )
    with_costs = simulate_portfolio_paths(
        result,
        {"Stocks": 0.5, "Bonds": 0.5},
        rebalance_frequency=1,
        transaction_cost_bps=100,
    )

    assert with_costs.iloc[-1, 0] < without_costs.iloc[-1, 0]


def test_buy_and_hold_tracks_drifting_asset_holdings():
    result = SimulationResult(
        returns=np.array(
            [
                [[0.10, 0.00]],
                [[0.10, 0.00]],
            ]
        ),
        regimes=np.empty((2, 1), dtype=object),
        assets=["Stocks", "Bonds"],
        states=[],
        frequency="M",
    )

    buy_and_hold = simulate_portfolio_paths(
        result,
        {"Stocks": 0.5, "Bonds": 0.5},
        rebalance_frequency=0,
    )
    legacy = simulate_portfolio_paths(result, {"Stocks": 0.5, "Bonds": 0.5})

    assert buy_and_hold.iloc[-1, 0] == pytest.approx(50.0 * np.exp(0.20) + 50.0)
    assert buy_and_hold.iloc[-1, 0] != pytest.approx(legacy.iloc[-1, 0])


def test_italian_tax_applies_standard_and_government_bond_rates_at_liquidation():
    result = SimulationResult(
        returns=np.array([[[0.10]]]),
        regimes=np.empty((1, 1), dtype=object),
        assets=["Asset"],
        states=[],
        frequency="M",
    )

    standard = simulate_portfolio_paths(
        result,
        {"Asset": 1.0},
        return_kind="simple",
        rebalance_frequency=0,
        tax_regime="italy_administered",
        italy_annual_wealth_tax=0.0,
    )
    government_bond = simulate_portfolio_paths(
        result,
        {"Asset": 1.0},
        return_kind="simple",
        rebalance_frequency=0,
        tax_regime="italy_administered",
        asset_tax_categories={"Asset": "government_bond"},
        italy_annual_wealth_tax=0.0,
    )

    assert standard.iloc[-1, 0] == pytest.approx(107.40)
    assert standard.attrs["terminal_liquidation_tax_total"] == pytest.approx(2.60)
    assert government_bond.iloc[-1, 0] == pytest.approx(108.75)
    assert government_bond.attrs["terminal_liquidation_tax_total"] == pytest.approx(1.25)


def test_italian_tax_loss_ledger_offsets_standard_but_not_fund_income():
    result = SimulationResult(
        returns=np.array([[[-0.20]], [[0.50]]]),
        regimes=np.empty((2, 1), dtype=object),
        assets=["Asset"],
        states=[],
        frequency="M",
    )

    standard = simulate_portfolio_paths(
        result,
        {"Asset": 1.0},
        return_kind="simple",
        rebalance_frequency=0,
        withdrawal=10.0,
        tax_regime="italy_administered",
        italy_annual_wealth_tax=0.0,
    )
    fund = simulate_portfolio_paths(
        result,
        {"Asset": 1.0},
        return_kind="simple",
        rebalance_frequency=0,
        withdrawal=10.0,
        tax_regime="italy_administered",
        asset_tax_categories={"Asset": "fund"},
        italy_annual_wealth_tax=0.0,
    )

    assert standard.iloc[-1, 0] == pytest.approx(91.10)
    assert standard.attrs["terminal_liquidation_tax_total"] == pytest.approx(3.90)
    assert standard.attrs["loss_carryforward_total"] == pytest.approx(0.0)
    assert fund.iloc[-1, 0] == pytest.approx(90.45)
    assert fund.attrs["taxes_paid_total"] == pytest.approx(4.55)
    assert fund.attrs["loss_carryforward_total"] == pytest.approx(2.50)


def test_italian_wealth_tax_is_monthly_and_terminal_liquidation_is_optional():
    result = SimulationResult(
        returns=np.zeros((12, 1, 1)),
        regimes=np.empty((12, 1), dtype=object),
        assets=["Asset"],
        states=[],
        frequency="M",
    )

    wealth = simulate_portfolio_paths(
        result,
        {"Asset": 1.0},
        return_kind="simple",
        rebalance_frequency=0,
        tax_regime="italy_administered",
        italy_annual_wealth_tax=0.002,
        tax_terminal_liquidation=False,
    )

    expected = 100.0 * (1.0 - 0.002 / 12.0) ** 12
    assert wealth.iloc[-1, 0] == pytest.approx(expected)
    assert wealth.attrs["wealth_tax_total"] == pytest.approx(100.0 - expected)
    assert wealth.attrs["terminal_liquidation_tax_total"] == pytest.approx(0.0)


def test_italian_tax_rejects_legacy_accounting_and_leverage():
    result = SimulationResult(
        returns=np.zeros((1, 1, 1)),
        regimes=np.empty((1, 1), dtype=object),
        assets=["Asset"],
        states=[],
        frequency="M",
    )

    with pytest.raises(ValueError, match="holdings-based"):
        simulate_portfolio_paths(result, {"Asset": 1.0}, tax_regime="italy_administered")
    with pytest.raises(ValueError, match="leveraged"):
        simulate_portfolio_paths(
            result,
            {"Asset": 1.0},
            rebalance_frequency=1,
            leverage_multiple=2.0,
            tax_regime="italy_administered",
        )


def test_buy_and_hold_rejects_rebalancing_costs_and_leverage():
    result = _two_period_result()

    with pytest.raises(ValueError, match="Transaction costs require"):
        simulate_portfolio_paths(
            result,
            {"Stocks": 0.5, "Bonds": 0.5},
            rebalance_frequency=0,
            transaction_cost_bps=10,
        )
    with pytest.raises(ValueError, match="Leverage and financing require"):
        simulate_portfolio_paths(
            result,
            {"Stocks": 0.5, "Bonds": 0.5},
            rebalance_frequency=0,
            leverage_multiple=2.0,
        )


def _two_period_result() -> SimulationResult:
    return SimulationResult(
        returns=np.array(
            [
                [[0.10, 0.00]],
                [[0.10, 0.00]],
            ]
        ),
        regimes=np.empty((2, 1), dtype=object),
        assets=["Stocks", "Bonds"],
        states=[],
        frequency="M",
    )


def test_periodic_contributions_increase_wealth():
    result = _two_period_result()
    plain = simulate_portfolio_paths(result, {"Stocks": 1.0})
    with_dca = simulate_portfolio_paths(result, {"Stocks": 1.0}, contribution=10.0)
    assert with_dca.iloc[-1, 0] > plain.iloc[-1, 0]
    assert with_dca.iloc[-1, 0] == pytest.approx(((100.0 + 10.0) * np.exp(0.10) + 10.0) * np.exp(0.10))


def test_withdrawals_reduce_wealth_and_floor_at_zero():
    result = _two_period_result()
    plain = simulate_portfolio_paths(result, {"Stocks": 1.0})
    with_small = simulate_portfolio_paths(result, {"Stocks": 1.0}, withdrawal=20.0)
    assert with_small.iloc[-1, 0] < plain.iloc[-1, 0]
    exhausted = simulate_portfolio_paths(result, {"Stocks": 1.0}, withdrawal=500.0)
    assert exhausted.iloc[-1, 0] == 0.0
    assert (exhausted.to_numpy() >= 0).all()


def test_cash_flows_work_in_rebalancing_mode():
    result = _two_period_result()
    with_flows = simulate_portfolio_paths(
        result,
        {"Stocks": 0.5, "Bonds": 0.5},
        rebalance_frequency=1,
        contribution=10.0,
        withdrawal=5.0,
        transaction_cost_bps=10,
    )
    without_flows = simulate_portfolio_paths(
        result,
        {"Stocks": 0.5, "Bonds": 0.5},
        rebalance_frequency=1,
        transaction_cost_bps=10,
    )
    assert with_flows.iloc[-1, 0] > without_flows.iloc[-1, 0]
    assert np.isfinite(with_flows.to_numpy()).all()
    assert (with_flows.to_numpy() >= 0).all()


def test_cash_flows_validate_inputs():
    result = _two_period_result()
    with pytest.raises(ValueError, match="contribution"):
        simulate_portfolio_paths(result, {"Stocks": 1.0}, contribution=-1.0)
    with pytest.raises(ValueError, match="withdrawal"):
        simulate_portfolio_paths(result, {"Stocks": 1.0}, withdrawal=-1.0)


def test_cash_flow_adjusted_metrics_ignore_regular_contributions():
    result = SimulationResult(
        returns=np.zeros((2, 1, 1)),
        regimes=np.empty((2, 1), dtype=object),
        assets=["Stocks"],
        states=[],
        frequency="M",
    )
    wealth = simulate_portfolio_paths(result, {"Stocks": 1.0}, contribution=10.0)

    summary = summarize_wealth_risk(wealth, periods_per_year=12, contribution=10.0)

    assert summary["cash_flow_adjusted_annualized_return"] == pytest.approx(0.0)
    assert summary["cash_flow_adjusted_volatility"] == pytest.approx(0.0)
    assert summary["total_contributed"] == pytest.approx(20.0)


def test_real_cash_flow_metrics_deflate_wealth_and_flows_once():
    wealth = pd.DataFrame({"path_0": [110.0]})

    summary = summarize_wealth_risk(
        wealth,
        periods_per_year=1,
        annual_inflation=0.10,
        contribution=10.0,
    )

    expected_real_return = 100.0 / 110.0 - 1.0
    assert summary["cash_flow_adjusted_annualized_return"] == pytest.approx(expected_real_return)
    assert summary["cash_flow_adjusted_volatility"] == pytest.approx(0.0)


def test_expense_ratio_reduces_simulated_wealth():
    result = SimulationResult(
        returns=np.zeros((2, 1, 1)),
        regimes=np.empty((2, 1), dtype=object),
        assets=["Stocks"],
        states=[],
        frequency="M",
    )

    wealth = simulate_portfolio_paths(
        result,
        {"Stocks": 1.0},
        asset_expense_ratios={"Stocks": 0.12},
    )

    assert wealth.iloc[-1, 0] == pytest.approx(100.0 * np.exp(2 * np.log(0.88) / 12))


def test_leverage_charges_financing_and_preserves_equity_accounting():
    result = SimulationResult(
        returns=np.zeros((1, 1, 1)),
        regimes=np.empty((1, 1), dtype=object),
        assets=["Stocks"],
        states=[],
        frequency="M",
    )

    wealth = simulate_portfolio_paths(
        result,
        {"Stocks": 1.0},
        rebalance_frequency=1,
        leverage_multiple=2.0,
        financing_rate=0.12,
    )

    expected = 200.0 - 100.0 * (1.12 ** (1 / 12))
    assert wealth.iloc[-1, 0] == pytest.approx(expected)
    assert wealth.attrs["margin_calls"] == 0


def test_state_dependent_financing_rate_charges_by_regime():
    result = SimulationResult(
        returns=np.zeros((2, 2, 1)),
        regimes=np.array([["hi", "lo"], ["hi", "lo"]], dtype=object),
        assets=["Stocks"],
        states=["hi", "lo"],
        frequency="M",
    )

    wealth = simulate_portfolio_paths(
        result,
        {"Stocks": 1.0},
        rebalance_frequency=1,
        leverage_multiple=2.0,
        financing_rate=0.0,
        financing_inflation_sensitivity=1.0,
        state_inflation={"hi": 0.12, "lo": 0.0},
    )

    hi_growth = (1.0 + 0.12) ** (1 / 12)
    expected_hi = 200.0 - 100.0 * hi_growth
    expected_hi = 2.0 * expected_hi - expected_hi * hi_growth
    expected_lo = 200.0 - 100.0 * 1.0
    assert wealth.iloc[-1, 0] == pytest.approx(expected_hi)
    assert wealth.iloc[-1, 1] == pytest.approx(expected_lo)
    assert expected_hi < expected_lo


def test_stochastic_short_rate_and_inflation_drive_financing_by_path():
    result = SimulationResult(
        returns=np.zeros((1, 2, 1)),
        regimes=np.empty((1, 2), dtype=object),
        assets=["Stocks"],
        states=[],
        frequency="M",
    )

    wealth = simulate_portfolio_paths(
        result,
        {"Stocks": 1.0},
        rebalance_frequency=1,
        leverage_multiple=2.0,
        financing_rate=0.01,
        financing_inflation_sensitivity=0.5,
        financing_rate_paths=np.array([[0.02, 0.08]]),
        financing_inflation_paths=np.array([[0.02, 0.06]]),
    )

    low_rate = 0.02 + 0.01 + 0.5 * 0.02
    high_rate = 0.08 + 0.01 + 0.5 * 0.06
    assert wealth.iloc[0, 0] == pytest.approx(200.0 - 100.0 * (1.0 + low_rate) ** (1 / 12))
    assert wealth.iloc[0, 1] == pytest.approx(200.0 - 100.0 * (1.0 + high_rate) ** (1 / 12))
    assert wealth.iloc[0, 1] < wealth.iloc[0, 0]
    assert wealth.attrs["effective_financing_rate"] == pytest.approx((low_rate + high_rate) / 2)


def test_leverage_liquidates_when_maintenance_margin_is_breached():
    result = SimulationResult(
        returns=np.array([[[-0.60]]]),
        regimes=np.empty((1, 1), dtype=object),
        assets=["Stocks"],
        states=[],
        frequency="M",
    )

    wealth = simulate_portfolio_paths(
        result,
        {"Stocks": 1.0},
        rebalance_frequency=1,
        leverage_multiple=2.0,
        maintenance_margin=0.25,
    )

    assert wealth.iloc[-1, 0] == 0.0
    assert wealth.attrs["margin_calls"] == 1


def test_wealth_risk_summary_includes_downside_metrics():
    wealth = pd.DataFrame(
        {
            "path_0": [105.0, 90.0, 95.0],
            "path_1": [110.0, 120.0, 130.0],
        }
    )

    summary = summarize_wealth_risk(wealth)

    assert summary["probability_of_loss"] == 0.5
    assert summary["var_95"] > 0
    assert summary["expected_shortfall_95"] >= summary["var_95"]
    assert 0 < summary["max_drawdown_mean"] < 1
    assert {
        "ulcer_index_mean",
        "sortino_ratio",
        "calmar_ratio",
        "geometric_annualized_return",
        "terminal_skewness",
        "terminal_kurtosis",
    }.issubset(summary.index)


def test_ulcer_index_matches_known_drawdown_series():
    wealth = pd.DataFrame({"path_0": [90.0]})

    summary = summarize_wealth_risk(wealth)

    assert summary["ulcer_index_mean"] == pytest.approx(np.sqrt(0.005))


def test_sortino_uses_downside_deviation():
    wealth = pd.DataFrame({"path_0": [100.0, 95.0]})

    summary = summarize_wealth_risk(wealth, periods_per_year=12)

    annualized_return = -0.025 * 12
    annualized_downside = np.sqrt((0.05**2) / 2) * np.sqrt(12)
    assert summary["sortino_ratio"] == pytest.approx(annualized_return / annualized_downside)


def test_geometric_annualized_return_matches_single_path():
    wealth = pd.DataFrame({"path_0": [110.0]})

    summary = summarize_wealth_risk(wealth, periods_per_year=1)

    assert summary["geometric_annualized_return"] == pytest.approx(0.10)
    assert summary["calmar_ratio"] == 0.0


def test_stationary_distribution_solves_markov_balance_equation():
    transition = pd.DataFrame(
        [[0.9, 0.1], [0.2, 0.8]],
        index=["growth", "recession"],
        columns=["growth", "recession"],
    )

    distribution = stationary_distribution(transition)

    assert np.allclose(distribution.to_numpy(), [2 / 3, 1 / 3])
    assert np.allclose(distribution.to_numpy() @ transition.to_numpy(), distribution.to_numpy())


def test_single_path_risk_summary_has_zero_standard_deviation():
    wealth = pd.DataFrame({"path_0": [101.0, 103.0]})

    summary = summarize_wealth_risk(wealth)

    assert summary["std"] == 0.0


def test_annualized_metrics_match_known_terminal_returns():
    wealth = pd.DataFrame(
        {
            "path_0": [110.0],
            "path_1": [130.0],
        }
    )

    summary = summarize_wealth_risk(wealth, periods_per_year=12)

    assert summary["annualized_return"] == pytest.approx(0.20 * 12)
    assert summary["annualized_volatility"] == pytest.approx((10.0 / 100.0) * np.sqrt(12))
    assert summary["sharpe_ratio"] == pytest.approx(
        summary["annualized_return"] / summary["annualized_volatility"]
    )


def test_annualized_metrics_default_to_single_period_scaling():
    wealth = pd.DataFrame({"path_0": [110.0]})

    summary = summarize_wealth_risk(wealth, periods_per_year=1)

    assert summary["annualized_return"] == pytest.approx(0.10)
    assert summary["sharpe_ratio"] == 0.0

    with pytest.raises(ValueError, match="periods_per_year"):
        summarize_wealth_risk(wealth, periods_per_year=0)


def test_sharpe_uses_risk_free_rate():
    wealth = pd.DataFrame({"path_0": [120.0]})

    summary = summarize_wealth_risk(wealth, periods_per_year=1, risk_free_rate=0.02)

    assert summary["sharpe_ratio"] == pytest.approx(0.0)
    assert summary["annualized_return"] == pytest.approx(0.20)

    with pytest.raises(ValueError, match="risk_free_rate"):
        summarize_wealth_risk(wealth, risk_free_rate=np.nan)


def test_risk_metrics_use_stochastic_real_short_rate_paths():
    wealth = pd.DataFrame({"path_0": [110.0], "path_1": [120.0]})

    summary = summarize_wealth_risk(
        wealth,
        periods_per_year=1,
        inflation_paths=np.array([[0.02, 0.04]]),
        risk_free_paths=np.array([[0.05, 0.08]]),
    )

    expected = np.mean([(1.05 / 1.02) - 1.0, (1.08 / 1.04) - 1.0])
    assert summary["effective_risk_free_rate"] == pytest.approx(expected)
    assert np.isfinite(summary["sharpe_ratio"])


def test_inflation_adjusts_wealth_to_real_terms():
    wealth = pd.DataFrame({"path_0": [110.0]})

    summary = summarize_wealth_risk(wealth, periods_per_year=1, annual_inflation=0.10)

    assert summary["annualized_return"] == pytest.approx(0.0)
    assert summary["mean"] == pytest.approx(100.0)

    with pytest.raises(ValueError, match="annual_inflation"):
        summarize_wealth_risk(wealth, annual_inflation=-0.1)


def test_inflation_compounds_per_year_not_per_period():
    wealth = pd.DataFrame(
        {"path_0": 100.0 * np.power(1.10, np.arange(1, 13, dtype=float) / 12.0)}
    )

    summary = summarize_wealth_risk(wealth, periods_per_year=12, annual_inflation=0.10)

    assert summary["mean"] == pytest.approx(100.0)
    assert summary["annualized_return"] == pytest.approx(0.0)


def test_portfolio_rejects_non_finite_weights():
    result = SimulationResult(
        returns=np.zeros((1, 1, 1)),
        regimes=np.empty((1, 1), dtype=object),
        assets=["Stocks"],
        states=[],
        frequency="M",
    )

    with pytest.raises(ValueError, match="finite"):
        simulate_portfolio_paths(result, {"Stocks": np.nan})


def _persistent_model() -> ScenarioModel:
    from mc_quadrants.types import RegimeMoments

    transition = pd.DataFrame(
        [[0.95, 0.05], [0.05, 0.95]],
        index=["state_a", "state_b"],
        columns=["state_a", "state_b"],
    )
    moments = {
        "state_a": RegimeMoments(
            mean=pd.Series([0.01], index=["Stocks"]),
            covariance=pd.DataFrame([[0.01]], index=["Stocks"], columns=["Stocks"]),
            correlation=pd.DataFrame([[1.0]], index=["Stocks"], columns=["Stocks"]),
            observations=100,
        ),
        "state_b": RegimeMoments(
            mean=pd.Series([-0.01], index=["Stocks"]),
            covariance=pd.DataFrame([[0.04]], index=["Stocks"], columns=["Stocks"]),
            correlation=pd.DataFrame([[1.0]], index=["Stocks"], columns=["Stocks"]),
            observations=100,
        ),
    }
    return ScenarioModel(
        states=["state_a", "state_b"],
        transition_matrix=transition,
        moments=moments,
        metadata={
            "duration_hazards": {
                "state_a": np.full(240, 1.0 / 50.0),
                "state_b": np.full(240, 1.0 / 50.0),
            }
        },
    )


def _switches_per_path(regimes: np.ndarray) -> float:
    changes = regimes[1:, :] != regimes[:-1, :]
    return float(changes.sum(axis=0).mean())


def test_semi_markov_sojourns_switch_less_than_markov_chain():
    model = _persistent_model()

    markov = simulate_regime_paths(model, periods=60, paths=40, random_seed=4)
    semi = simulate_regime_paths(model, periods=60, paths=40, random_seed=4, duration_model="semi_markov")

    assert markov.shape == (60, 40)
    assert semi.shape == (60, 40)
    assert _switches_per_path(semi) < _switches_per_path(markov)


def test_semi_markov_honors_exact_initial_and_following_sojourn_lengths():
    model = _persistent_model()
    model.metadata["duration_hazards"] = {
        "state_a": np.array([0.0, 0.0, 1.0]),
        "state_b": np.array([0.0, 0.0, 1.0]),
    }

    regimes = simulate_regime_paths(
        model,
        periods=8,
        paths=1,
        start_state="state_a",
        random_seed=4,
        duration_model="semi_markov",
        min_regime_duration=3,
    ).ravel()

    assert regimes.tolist() == [
        "state_a",
        "state_a",
        "state_a",
        "state_b",
        "state_b",
        "state_b",
        "state_a",
        "state_a",
    ]


def test_semi_markov_requires_duration_hazard_metadata():
    model = _persistent_model()
    model.metadata.pop("duration_hazards")

    with pytest.raises(ValueError, match="duration_hazards"):
        simulate_regime_paths(model, periods=6, paths=2, duration_model="semi_markov")


def test_semi_markov_rejects_unknown_duration_model():
    model = _persistent_model()

    with pytest.raises(ValueError, match="duration_model"):
        simulate_regime_paths(model, periods=6, paths=2, duration_model="weibull")


def test_garch_creates_volatility_clustering():
    from mc_quadrants.types import RegimeMoments

    moments = {
        "only": RegimeMoments(
            mean=pd.Series([0.0], index=["Stocks"]),
            covariance=pd.DataFrame([[0.01]], index=["Stocks"], columns=["Stocks"]),
            correlation=pd.DataFrame([[1.0]], index=["Stocks"], columns=["Stocks"]),
            observations=1000,
        )
    }
    model = ScenarioModel(
        states=["only"],
        transition_matrix=pd.DataFrame([[1.0]], index=["only"], columns=["only"]),
        moments=moments,
    )

    def squared_autocorrelation(result: SimulationResult) -> float:
        path = result.returns[:, 0, 0]
        squared = path**2
        centered = squared - squared.mean()
        return float((centered[1:] * centered[:-1]).mean() / (centered**2).mean())

    plain = simulate_returns(model, periods=600, paths=1, random_seed=11)
    garch_result = simulate_returns(
        model,
        periods=600,
        paths=1,
        random_seed=11,
        garch=True,
        garch_alpha=0.12,
        garch_beta=0.85,
    )

    assert squared_autocorrelation(garch_result) > squared_autocorrelation(plain)
    assert np.isfinite(garch_result.returns).all()


def test_garch_is_reproducible_and_validated():
    model = _persistent_model()
    first = simulate_returns(
        model,
        periods=12,
        paths=5,
        random_seed=2,
        garch=True,
    )
    second = simulate_returns(
        model,
        periods=12,
        paths=5,
        random_seed=2,
        garch=True,
    )
    assert np.array_equal(first.returns, second.returns)

    with pytest.raises(ValueError, match="distribution='normal'"):
        simulate_returns(model, periods=4, paths=2, distribution="student_t", garch=True)
    with pytest.raises(ValueError, match="garch_alpha"):
        simulate_returns(model, periods=4, paths=2, garch=True, garch_alpha=1.5)
    with pytest.raises(ValueError, match="less than 1"):
        simulate_returns(model, periods=4, paths=2, garch=True, garch_alpha=0.3, garch_beta=0.8)


def test_joint_macro_and_dynamic_dependence_produce_consistent_paths():
    rng = np.random.default_rng(12)
    dates = pd.date_range("1995-01-31", periods=120, freq="ME")
    macro = pd.DataFrame(
        {
            "growth": np.tile([2.0, 2.0, -2.0, -2.0], 30) + rng.normal(0, 0.2, 120),
            "inflation": np.tile([1.0, 4.0, 4.0, 1.0], 30) + rng.normal(0, 0.2, 120),
        },
        index=dates,
    )
    returns = pd.DataFrame(
        rng.normal(0.004, 0.025, size=(120, 3)),
        index=dates,
        columns=["Stocks", "Bonds", "Gold"],
    )
    model = calibrate_quadrant_model(
        returns,
        macro,
        growth_threshold=0.0,
        inflation_threshold=3.0,
        min_observations=6,
        probabilistic_regimes=True,
        mean_prior_strength=24.0,
        joint_macro=True,
    )

    result = simulate_returns(
        model,
        periods=24,
        paths=40,
        random_seed=5,
        distribution="student_t",
        joint_macro=True,
        dynamic_correlation=True,
    )

    assert result.returns.shape == (24, 40, 3)
    assert result.macro_paths.shape == (24, 40, 2)
    assert result.macro_columns == ["growth", "inflation"]
    assert np.isfinite(result.returns).all()
    assert np.isfinite(result.macro_paths).all()
