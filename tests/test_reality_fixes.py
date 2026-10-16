import numpy as np
import pytest

from mc_quadrants.calibration import calibrate_quadrant_model
from mc_quadrants.cpi import accounting_inflation_paths, calibrate_cpi_history, monthly_cpi_from_yoy
from mc_quadrants.economic_returns import (
    economic_assumptions,
    economic_path_adjustment,
    prepare_economic_history,
)
from mc_quadrants.native import native_available
from mc_quadrants.simulation import inflation_deflators, simulate_joint_regime_macro_paths
from mc_quadrants.structural import build_asset_profiles
from mc_quadrants.types import SimulationResult
from mc_quadrants.uncertainty import anchor_parameter_models, bootstrap_quadrant_models
from mc_quadrants.yield_forecasts import fit_yield_forecasts
from tests.test_economic_returns import history


def test_one_time_price_jump_is_not_charged_again_in_future_inflation():
    past = np.r_[np.zeros(11), np.log(1.1)]
    yoy = np.r_[np.full(11, 0.1), 0.0][:, None]
    monthly = monthly_cpi_from_yoy(yoy, past)
    np.testing.assert_allclose(monthly, 0, atol=1e-14)
    deflator = inflation_deflators(12, 1, inflation_paths=np.expm1(12 * monthly))
    np.testing.assert_allclose(deflator, 1, atol=1e-14)


def test_three_year_cpi_reconstruction_matches_actual_price_levels():
    rng = np.random.default_rng(12)
    changes = rng.normal(0.002, 0.003, (48, 5))
    changes[:12] = changes[:12, :1]
    prices = np.exp(np.vstack([np.zeros((1, 5)), changes.cumsum(axis=0)]))
    future_yoy = prices[13:] / prices[1:-12] - 1
    reconstructed = monthly_cpi_from_yoy(future_yoy, changes[:12, 0])
    np.testing.assert_allclose(reconstructed, changes[12:], atol=1e-14)
    np.testing.assert_allclose(np.exp(reconstructed.cumsum(axis=0)), prices[13:] / prices[12], atol=1e-13)


def test_cpi_initialization_rejects_calendar_inconsistency_and_discloses_missing_history():
    _, macro = history()
    result = calibrate_cpi_history(macro, "inflation")
    assert result["history_source"] == "flat_history_assumption"
    macro["cpi_monthly_log"] = 0.0
    with pytest.raises(ValueError, match="inconsistent"):
        calibrate_cpi_history(macro, "inflation")


def test_accounting_uses_observed_cpi_history_and_bootstrap_restores_it():
    returns, macro = history()
    macro["inflation"] = 100 * np.expm1(np.log(1.02))
    macro["cpi_monthly_log"] = np.log(1.02) / 12
    model = calibrate_quadrant_model(returns, macro, joint_macro=True)
    assert model.metadata["cpi_process"]["history_source"] == "observed_monthly_changes"
    models = bootstrap_quadrant_models(returns, macro, draws=1, random_seed=2, joint_macro=True)
    anchored = anchor_parameter_models(models, model)[0]
    assert anchored.metadata["cpi_process"] == model.metadata["cpi_process"]
    columns = model.metadata["macro_dynamics"]["columns"]
    levels = np.broadcast_to(model.metadata["macro_dynamics"]["latest"], (24, 2, len(columns))).copy()
    result = SimulationResult(
        np.zeros((24, 2, 2)),
        np.zeros((24, 2)),
        ["SPY", "IEF"],
        [],
        "ME",
        macro_paths=levels,
        macro_columns=columns,
    )
    rates = accounting_inflation_paths(model, result)
    np.testing.assert_allclose(rates, 0.02, atol=1e-13)


def test_no_change_yield_does_not_drift_to_historical_regime_centers():
    returns, macro = history()
    macro.loc[macro.index[-1], "bond_yield"] = 0.05
    model = calibrate_quadrant_model(
        returns, macro, joint_macro=True, economic_returns={"yield_model": "random_walk"}
    )
    dynamics = model.metadata["macro_dynamics"]
    dimension = len(dynamics["columns"])
    for state in model.states:
        dynamics["state_innovation_covariances"][state] = (np.eye(dimension) * 1e-14).tolist()
    _, paths, _ = simulate_joint_regime_macro_paths(
        model, 36, 20, random_seed=2, macro_parameter_uncertainty=True
    )
    np.testing.assert_allclose(paths[..., dynamics["columns"].index("bond_yield")], 0.05, atol=0.001)


def test_bootstrap_join_cannot_train_a_yield_trend():
    values = np.array([1.0, 1.1, 1.2, 80.0, 80.1, 80.2])[:, None]
    valid = np.array([True, True, False, True, True])
    specs, _ = fit_yield_forecasts(values, ["yield"], ["yield"], valid)
    assert abs(specs["yield"]["latest_drift"]) < 0.1
    assert specs["yield"]["observations"] == 4


def test_short_and_long_bonds_require_their_own_yield_reference():
    returns, macro = history()
    returns = returns.rename(columns={"SPY": "SHY", "IEF": "TLT"})
    profiles = build_asset_profiles(list(returns))
    assert profiles["SHY"]["asset_class"] == "government_bond"
    options = economic_assumptions({})
    with pytest.raises(ValueError, match="bond_yield_2y"):
        prepare_economic_history(returns, macro, profiles, options, "inflation")
    macro["bond_yield_2y"] = 2.0
    macro["bond_yield_20y"] = 5.0
    _, meta = prepare_economic_history(returns, macro, profiles, options, "inflation")
    assert meta["channels"]["SHY"]["yield_column"] == "bond_yield_2y"
    assert meta["channels"]["TLT"]["yield_column"] == "bond_yield_20y"
    assert meta["current_carry"][0] == pytest.approx(np.log1p(0.02) / 12)
    assert meta["current_carry"][1] == pytest.approx(np.log1p(0.05) / 12)


def test_valuation_prior_is_causal_and_current_multiple_is_preserved_in_bootstrap():
    returns, macro = history()
    macro["cape"] = np.linspace(15.0, 30.0, len(macro))
    options = {
        "rate_stress": False,
        "equity_valuation_columns": {"SPY": "cape"},
        "valuation_reference": {"SPY": 20.0},
    }
    high = calibrate_quadrant_model(returns, macro, economic_returns=options)
    low_macro = macro.copy()
    low_macro.loc[low_macro.index[-1], "cape"] = 10
    low = calibrate_quadrant_model(returns, low_macro, economic_returns=options)
    for state in high.states:
        assert low.moments[state].mean.SPY > high.moments[state].mean.SPY
    sampled = bootstrap_quadrant_models(returns, macro, 2, random_seed=3, economic_returns=options)
    for model in anchor_parameter_models(sampled, high):
        assert model.metadata["economic_returns"]["valuation"]["SPY"]["latest"] == 30


def test_credit_channel_requires_assumptions_and_losses_reduce_physical_carry():
    returns, macro = history()
    returns = returns.rename(columns={"IEF": "LQD"})
    macro["spread"] = 1.0
    with pytest.raises(ValueError, match="credit-loss"):
        calibrate_quadrant_model(returns, macro, joint_macro=True, economic_returns={})
    options = {"asset_spread_columns": {"LQD": "spread"}, "credit_loss_rates": {"LQD": 0.01}}
    model = calibrate_quadrant_model(returns, macro, joint_macro=True, economic_returns=options)
    meta = model.metadata["economic_returns"]
    assert meta["current_carry"][1] == pytest.approx(
        (np.log1p(macro.bond_yield.iloc[-1] / 100 + 0.01) + np.log1p(-0.01)) / 12
    )
    columns = model.metadata["macro_dynamics"]["columns"]
    levels = np.broadcast_to(model.metadata["macro_dynamics"]["latest"], (2, 3, len(columns))).copy()
    levels[..., columns.index("spread")] = 1.0
    np.testing.assert_allclose(economic_path_adjustment(model, levels)[..., 1], 0, atol=1e-14)


def test_tips_can_lose_value_when_real_yields_rise_despite_inflation():
    returns, macro = history()
    returns = returns.rename(columns={"IEF": "TIP"})
    macro["real_bond_yield"] = 1.0
    macro["inflation"] = 100 * np.expm1(12 * 0.002)
    macro["cpi_monthly_log"] = 0.002
    macro["tips_inflation_log"] = 0.002
    model = calibrate_quadrant_model(returns, macro, joint_macro=True, economic_returns={})
    columns = model.metadata["macro_dynamics"]["columns"]
    levels = np.broadcast_to(model.metadata["macro_dynamics"]["latest"], (1, 3, len(columns))).copy()
    levels[..., columns.index("real_bond_yield")] += 1.0
    physical = economic_path_adjustment(model, levels) + model.metadata["economic_returns"]["current_carry"]
    assert (physical[..., 1] < -0.04).all()


def test_surprise_mode_does_not_impose_equity_loss_from_inflation_level_alone():
    returns, macro = history()
    model = calibrate_quadrant_model(returns, macro, joint_macro=True, economic_returns={})
    columns = model.metadata["macro_dynamics"]["columns"]
    levels = np.broadcast_to(model.metadata["macro_dynamics"]["latest"], (1, 3, len(columns))).copy()
    levels[..., columns.index("bond_yield")] += 1.0
    assert (economic_path_adjustment(model, levels)[..., 0] == 0).all()
    assert model.metadata["macro_dynamics"]["return_exposure_basis"] == "state_centered_forecast_innovations"


@pytest.mark.parametrize("disable_native", [False, True])
def test_monthly_cpi_generation_preserves_price_identity_without_annual_echo(monkeypatch, disable_native):
    if disable_native:
        monkeypatch.setenv("MC_DISABLE_NATIVE_SIM", "1")
    returns, macro = history()
    macro["inflation"] = 100 * np.expm1(12 * 0.002)
    macro["cpi_monthly_log"] = 0.002
    model = calibrate_quadrant_model(returns, macro, joint_macro=True)
    dynamics = model.metadata["macro_dynamics"]
    dimensions = len(dynamics["columns"])
    dynamics["cpi_dynamics"]["persistence"] = 0
    dynamics["cpi_dynamics"]["state_centers"] = {state: 0.3 for state in model.states}
    for state in model.states:
        dynamics["state_innovation_covariances"][state] = (np.eye(dimensions) * 1e-14).tolist()
    _, paths, _ = simulate_joint_regime_macro_paths(
        model, 36, 40, random_seed=7, macro_parameter_uncertainty=False
    )
    changes = monthly_cpi_from_yoy(paths[..., 1] * 0.01, model.metadata["cpi_process"]["monthly_log_history"])
    np.testing.assert_allclose(changes, 0.003, atol=1e-6)
    np.testing.assert_allclose(paths[11:, :, 1] * 0.01, np.expm1(12 * 0.003), atol=3e-6)


def test_credit_spread_paths_remain_positive():
    returns, macro = history()
    returns = returns.rename(columns={"IEF": "LQD"})
    macro["spread"] = np.exp(np.random.default_rng(4).normal(0, 0.4, len(macro)))
    model = calibrate_quadrant_model(
        returns,
        macro,
        joint_macro=True,
        economic_returns={"asset_spread_columns": {"LQD": "spread"}, "credit_loss_rates": {"LQD": 0.01}},
    )
    _, paths, _ = simulate_joint_regime_macro_paths(model, 120, 100, random_seed=2)
    index = model.metadata["macro_dynamics"]["columns"].index("spread")
    assert np.isfinite(paths[..., index]).all()
    assert (paths[..., index] > 0).all()


@pytest.mark.skipif(not native_available(), reason="native backend unavailable")
def test_compact_monthly_cpi_matches_standard_reporting():
    from mc_quadrants.pipeline import run_scenario
    from tests.test_pipeline import _scenario_kwargs

    options = _scenario_kwargs(
        dict(
            periods=24,
            paths=30000,
            workers=2,
            walk_forward=False,
            joint_macro=True,
            tax_country="none",
            rebalance_frequency=3,
        )
    )
    macro = options["macro"]
    macro["cpi_monthly_log"] = 0.002 + 0.001 * np.sin(np.arange(len(macro)))
    macro["inflation"] = 100 * np.expm1(macro.cpi_monthly_log.rolling(12, min_periods=1).sum())
    scenario = run_scenario(**options)
    indices = np.asarray(scenario.wealth.attrs["sample_indices"])
    np.testing.assert_allclose(
        scenario.reporting_wealth.attrs["full_terminal_values"][indices],
        scenario.reporting_wealth.iloc[-1],
        rtol=1e-12,
    )


def test_missing_monthly_cpi_is_excluded_from_fit_and_initialization_disclosed():
    returns, macro = history()
    macro["inflation"] = 100 * np.expm1(0.024)
    macro["cpi_monthly_log"] = 0.002
    macro.loc[macro.index[-1], "cpi_monthly_log"] = np.nan
    model = calibrate_quadrant_model(returns, macro, joint_macro=True)
    assert model.metadata["cpi_process"]["history_source"] == "flat_history_assumption"
    _, paths, _ = simulate_joint_regime_macro_paths(model, 24, 10, random_seed=7)
    assert np.isfinite(paths).all()


@pytest.mark.parametrize(
    "asset,expected",
    [("TIP", (np.log1p(0.01) + np.log1p(0.02)) / 12), ("LQD", (np.log1p(0.04) + np.log1p(-0.01)) / 12)],
)
def test_anchor_only_targets_include_inflation_or_expected_credit_losses(asset, expected):
    returns, macro = history()
    returns = returns.rename(columns={"IEF": asset})
    macro["bond_yield"] = 3.0
    macro["real_bond_yield"] = 1.0
    macro["spread"] = 1.0
    options = {"rate_stress": False}
    if asset == "LQD":
        options.update(asset_spread_columns={"LQD": "spread"}, credit_loss_rates={"LQD": 0.01})
    model = calibrate_quadrant_model(returns, macro, economic_returns=options)
    assert model.metadata["economic_returns"]["monthly_prior_targets"][asset] == pytest.approx(expected)
