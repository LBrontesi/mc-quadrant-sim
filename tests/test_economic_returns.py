from copy import deepcopy
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from mc_quadrants.api import scenario_kwargs
from mc_quadrants.calibration import calibrate_quadrant_model
from mc_quadrants.economic_returns import (
    anchor_moments,
    economic_assumptions,
    economic_path_adjustment,
    prepare_economic_history,
    structural_returns,
)
from mc_quadrants.pipeline import run_scenario
from mc_quadrants.structural import build_asset_profiles
from mc_quadrants.types import RegimeMoments
from mc_quadrants.uncertainty import anchor_parameter_models, bootstrap_quadrant_models


def history():
    dates = pd.date_range("2000-01-31", periods=84, freq="ME")
    t = np.arange(len(dates))
    macro = pd.DataFrame(
        dict(
            growth=2 * np.sin(t / 6),
            inflation=3 + 2 * np.cos(t / 9),
            interest_rate=2 + np.sin(t / 8),
            bond_yield=3 + 0.4 * np.sin(t / 7),
        ),
        index=dates,
    )
    returns = pd.DataFrame(
        np.random.default_rng(14).normal([0.005, 0.002], [0.03, 0.008], (len(dates), 2)),
        index=dates,
        columns=["SPY", "IEF"],
    )
    return returns, macro


def test_yield_channel_accumulates_repricing_and_carry_without_repeating_shocks():
    options = economic_assumptions({"equity_rate_mode": "inflation_gate"})
    profiles = build_asset_profiles(["SPY", "IEF"])
    rising = np.linspace(0.03, 0.05, 13)
    factors = structural_returns(rising[:-1], rising[1:], 0.06, profiles, options)
    assert factors[:, 0].sum() == pytest.approx(-4 * 0.02)
    assert factors[:, 1].sum() == pytest.approx(np.log1p(rising[:-1]).sum() / 12 - 7.3 * 0.02)
    flat = structural_returns(0.05, 0.05, 0.06, profiles, options)
    assert flat[0] == 0
    assert flat[1] == pytest.approx(np.log1p(0.05) / 12)
    low_inflation = structural_returns(0.03, 0.04, 0.02, profiles, options)
    assert low_inflation[0] == 0
    assert low_inflation[1] < 0
    falling = structural_returns(0.04, 0.03, 0.06, profiles, options)
    assert (falling > 0).all()


def test_history_removes_exactly_the_factors_that_simulation_reinserts():
    returns, macro = history()
    options = economic_assumptions({})
    profiles = build_asset_profiles(list(returns))
    physical = structural_returns(
        macro.bond_yield.shift(1).iloc[1:].to_numpy() / 100,
        macro.bond_yield.iloc[1:].to_numpy() / 100,
        macro.inflation.shift(1).iloc[1:].to_numpy() / 100,
        profiles,
        options,
    )
    synthetic = returns.copy()
    synthetic.iloc[1:] = physical + np.array([0.005, 0.001])
    residual, _ = prepare_economic_history(synthetic, macro, profiles, options, "inflation")
    np.testing.assert_allclose(residual, np.tile([0.005, 0.001], (83, 1)), atol=1e-12)


def test_anchors_change_mean_level_preserving_regime_contrasts_and_other_assets():
    assets = ["SPY", "IEF", "GLD"]
    profiles = build_asset_profiles(assets)
    returns = pd.DataFrame(np.tile([-0.002, 0.006, 0.004], (60, 1)), columns=assets)
    covariance = pd.DataFrame(np.eye(3) * 0.001, index=assets, columns=assets)
    moments = {
        s: RegimeMoments(returns.mean() + d, covariance, covariance, 30)
        for s, d in [("a", 0.001), ("b", -0.001)]
    }
    metadata = dict(assumptions=economic_assumptions({"rate_stress": False}), latest_yield=0.02)
    anchored = anchor_moments(moments, returns, profiles, metadata)
    assert anchored["a"].mean.SPY > moments["a"].mean.SPY
    assert anchored["a"].mean.IEF < moments["a"].mean.IEF
    assert anchored["a"].mean.GLD == moments["a"].mean.GLD
    np.testing.assert_allclose(anchored["a"].mean - anchored["b"].mean, moments["a"].mean - moments["b"].mean)
    pd.testing.assert_frame_equal(anchored["a"].covariance, moments["a"].covariance)


@pytest.mark.parametrize("disable_native", [False, True])
def test_native_and_reference_samplers_apply_the_same_economic_channel(monkeypatch, disable_native):
    import mc_quadrants.simulation as simulation

    if disable_native:
        monkeypatch.setenv("MC_DISABLE_NATIVE_SIM", "1")
    returns, macro = history()
    model = calibrate_quadrant_model(
        returns, macro, probabilistic_regimes=True, joint_macro=True, economic_returns={}
    )
    columns = model.metadata["macro_dynamics"]["columns"]
    values = np.broadcast_to(
        np.array(model.metadata["macro_dynamics"]["latest"]), (6, 20, len(columns))
    ).copy()
    values[:, :, columns.index("bond_yield")] += np.arange(1, 7)[:, None] * 0.1
    shocks = np.zeros_like(values)
    codes = np.zeros((6, 20), dtype=np.uint8)
    monkeypatch.setattr(
        simulation, "simulate_joint_regime_macro_paths", lambda *a, **kw: (codes, values, shocks)
    )
    metadata = deepcopy(model.metadata)
    metadata.pop("economic_returns")
    baseline = replace(model, metadata=metadata)
    first = simulation.simulate_returns(model, 6, 20, random_seed=3, joint_macro=True)
    second = simulation.simulate_returns(baseline, 6, 20, random_seed=3, joint_macro=True)
    np.testing.assert_allclose(
        first.returns - second.returns, economic_path_adjustment(model, values), atol=1e-12
    )


def test_bootstrap_preserves_observed_origin_and_has_prior_uncertainty():
    returns, macro = history()
    reference = calibrate_quadrant_model(returns, macro, joint_macro=True, economic_returns={})
    models = bootstrap_quadrant_models(
        returns, macro, draws=3, random_seed=12, joint_macro=True, economic_returns={}
    )
    anchored = anchor_parameter_models(models, reference)
    priors = [m.metadata["economic_returns"]["assumptions"]["equity_real_return"] for m in anchored]
    assert len(set(priors)) == 3
    for before, after in zip(models, anchored):
        assert (
            after.metadata["economic_returns"]["latest_yield"]
            == reference.metadata["economic_returns"]["latest_yield"]
        )
        for state in before.states:
            old = before.moments[state].mean - before.metadata["economic_returns"]["current_carry"]
            new = after.moments[state].mean - after.metadata["economic_returns"]["current_carry"]
            np.testing.assert_allclose(old, new)


def test_pipeline_preserves_economic_mode_through_parameter_draws_and_accounting():
    returns, macro = history()
    scenario = run_scenario(
        returns,
        macro,
        list(returns),
        "growth",
        "inflation",
        "median",
        "median",
        periods=12,
        paths=64,
        random_seed=12,
        start_state=None,
        weights={"SPY": 0.6, "IEF": 0.4},
        joint_macro=True,
        economic_returns={},
        parameter_draws=2,
        parameter_seed=7,
        rebalance_frequency=1,
        walk_forward=False,
    )
    assert scenario.model.metadata["economic_returns"]["assumptions"]["rate_stress"]
    assert "bond_yield" in scenario.result.macro_columns
    assert scenario.wealth.shape[1] == 64
    assert np.isfinite(scenario.wealth).all().all()


def test_missing_yields_and_incompatible_modes_fail_instead_of_using_policy_rate():
    returns, macro = history()
    with pytest.raises(ValueError, match="observed.*bond_yield"):
        calibrate_quadrant_model(
            returns, macro.drop(columns="bond_yield"), joint_macro=True, economic_returns={}
        )
    with pytest.raises(ValueError, match="joint_macro"):
        calibrate_quadrant_model(returns, macro, economic_returns={})
    with pytest.raises(ValueError, match="joint macro"):
        scenario_kwargs({"weights": {"SPY": 1}, "economic_returns": {}})
    result = scenario_kwargs({"weights": {"SPY": 1}, "joint_macro": True, "economic_returns": {}})
    assert result["economic_returns"]["anchor_strength"] == 120


def test_bootstrap_retains_physical_pairs_and_masks_block_joins(monkeypatch):
    import mc_quadrants.uncertainty as uncertainty
    from mc_quadrants.economic_returns import PREVIOUS_INFLATION, previous_column

    returns, macro = history()
    options = economic_assumptions({})
    profiles = build_asset_profiles(list(returns))
    expected, _ = prepare_economic_history(returns, macro, profiles, options, "inflation")
    original_calibrate = uncertainty.calibrate_quadrant_model
    captured = []

    def capture(sampled_returns, sampled_macro, **kwargs):
        residual, _ = prepare_economic_history(sampled_returns, sampled_macro, profiles, options, "inflation")
        original_positions = [
            np.flatnonzero(returns.SPY.to_numpy() == value)[0] for value in sampled_returns.SPY
        ]
        np.testing.assert_allclose(residual, expected.loc[returns.index[original_positions]], atol=1e-12)
        np.testing.assert_array_equal(
            sampled_macro.attrs["valid_macro_transitions"], np.diff(original_positions) == 1
        )
        assert sampled_macro[[previous_column("bond_yield"), PREVIOUS_INFLATION]].notna().all().all()
        captured.append(True)
        return original_calibrate(sampled_returns, sampled_macro, **kwargs)

    monkeypatch.setattr(uncertainty, "calibrate_quadrant_model", capture)
    uncertainty.bootstrap_quadrant_models(
        returns, macro, draws=1, random_seed=9, macro_lag_periods=1, joint_macro=True, economic_returns={}
    )
    assert captured


def test_economic_walk_forward_uses_production_paths_and_parameter_models():
    from mc_quadrants.validation import walk_forward_validation

    returns, macro = history()
    result = walk_forward_validation(
        returns,
        macro,
        "growth",
        "inflation",
        "median",
        "median",
        min_train_periods=60,
        max_splits=1,
        horizons=(1, 12),
        validation_paths=64,
        weights={"SPY": 0.6, "IEF": 0.4},
        parameter_draws=2,
        parameter_seed=3,
        calibration_options={"joint_macro": True, "economic_returns": {}},
    )
    assert result.summary["splits"] == 1
    assert np.isfinite(result.summary["regime_energy_score_mean"])


def test_disabled_options_do_not_require_yields_and_invalid_prior_is_rejected():
    assert economic_assumptions({"return_anchors": False, "rate_stress": False}) is None
    with pytest.raises(ValueError, match="anchor_std"):
        economic_assumptions({"anchor_std": 10})


def test_anchor_only_bootstrap_uses_current_bond_yield():
    returns, macro = history()
    options = {"rate_stress": False}
    reference = calibrate_quadrant_model(returns, macro, economic_returns=options)
    models = bootstrap_quadrant_models(returns, macro, draws=2, random_seed=4, economic_returns=options)
    anchored = anchor_parameter_models(models, reference)
    target = np.log1p(reference.metadata["economic_returns"]["latest_yield"]) / 12
    for before, after in zip(models, anchored):
        old = before.metadata["economic_returns"]
        new = after.metadata["economic_returns"]
        correction = old["anchor_weight"] * (target - old["monthly_prior_targets"]["IEF"])
        assert new["monthly_prior_targets"]["IEF"] == pytest.approx(target)
        for state in before.states:
            assert after.moments[state].mean.IEF - before.moments[state].mean.IEF == pytest.approx(correction)


def test_market_yields_keep_observation_calendar_with_release_aligned_macro(monkeypatch):
    import mc_quadrants.data as data

    dates = pd.date_range("2018-01-31", periods=36, freq="ME")
    levels = pd.DataFrame(
        {
            "growth": np.linspace(100, 120, 36),
            "inflation": np.linspace(100, 110, 36),
            "interest_rate": 2.0,
            "bond_yield": np.arange(36) / 10,
        },
        index=dates,
    )
    levels.attrs.update(
        point_in_time=True,
        data_vintage="initial_release",
        release_dates=pd.DataFrame(
            {column: dates + pd.offsets.MonthEnd(1) for column in levels}, index=dates
        ),
    )
    prices = pd.DataFrame({"SPY": np.linspace(100, 130, 36)}, index=dates)
    monkeypatch.setattr(data, "fetch_yahoo_prices", lambda *a, **kw: prices)
    monkeypatch.setattr(data, "fetch_fred_macro", lambda *a, **kw: levels)
    data._load_market_data_cached.cache_clear()
    macro, _, _, _ = data.load_market_data(
        ["SPY"], "2018-01-01", "2020-12-31", macro_vintage="initial_release"
    )
    date = pd.Timestamp("2020-06-30")
    assert macro.attrs["availability_aligned"]
    assert macro.loc[date, "bond_yield"] == levels.loc[date, "bond_yield"]
    # Availability alignment still delays the macro predictor by one month.
    prior = date - pd.offsets.MonthEnd(1)
    assert macro.loc[date, "growth"] == pytest.approx(100 * levels.growth.pct_change(12).loc[prior])
    data._load_market_data_cached.cache_clear()
