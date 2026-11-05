"""Curve geometry, masked calibration, and production path integration."""

import json

import numpy as np
import pandas as pd
import pytest

from mc_quadrants.calibration import calibrate_quadrant_model
from mc_quadrants.economic_returns import economic_assumptions
from mc_quadrants.simulation import simulate_joint_regime_macro_paths, simulate_returns
from mc_quadrants.uncertainty import anchor_parameter_models, bootstrap_quadrant_models
from mc_quadrants.yield_curve import CURVE_TENORS, curve_forecast, fit_curve, loadings


def history():
    rng = np.random.default_rng(781)
    n = 144
    factors = np.empty((n, 3))
    factors[0] = [4, -2, 1]
    for t in range(1, n):
        factors[t] = [4, -2, 1] + 0.95 * (factors[t - 1] - [4, -2, 1]) + rng.normal(0, 0.08, 3)
    macro = pd.DataFrame(
        factors @ loadings(list(CURVE_TENORS.values())).T,
        columns=list(CURVE_TENORS),
        index=pd.date_range("2000-01-31", periods=n, freq="ME"),
    )
    macro["growth"] = 2 + np.sin(np.arange(n) / 6)
    macro["inflation"] = 2 + np.cos(np.arange(n) / 9)
    macro["interest_rate"] = 2 + 0.4 * np.sin(np.arange(n) / 7)
    returns = pd.DataFrame(
        rng.normal([0.005, 0.002], [0.03, 0.009], (n, 2)), index=macro.index, columns=["SPY", "IEF"]
    )
    return returns, macro


def test_curve_geometry_and_factor_recovery():
    basis = loadings([0.01, 2, 5, 7, 10, 20, 1000])
    factors = np.array([[4, -2, 1], [5, 1, -2]])
    np.testing.assert_allclose(factors @ basis.T @ np.linalg.pinv(basis).T, factors, atol=1e-12)
    assert abs((factors @ basis.T)[0, -1] - 4) < 0.01
    with pytest.raises(ValueError):
        loadings([0, 2])


@pytest.mark.parametrize("stress", ["none", "combined"])
def test_production_paths_have_a_curve_at_every_time_and_path(stress):
    returns, macro = history()
    model = calibrate_quadrant_model(
        returns,
        macro,
        joint_macro=True,
        economic_returns={"yield_model": "nelson_siegel", "macro_stress": stress},
    )
    curve = model.metadata["macro_dynamics"]["curve_dynamics"]
    json.dumps(curve)  # Model metadata survives persistence/API serialization.
    first = simulate_joint_regime_macro_paths(model, 60, 64, random_seed=7)
    second = simulate_joint_regime_macro_paths(model, 60, 64, random_seed=7)
    np.testing.assert_array_equal(first[1], second[1])
    yields = first[1][..., curve["indices"]]
    basis, inverse = np.asarray(curve["loadings"]), np.asarray(curve["inverse"])
    np.testing.assert_allclose(yields @ inverse.T @ basis.T, yields, atol=1e-10)
    assert np.isfinite(yields).all()
    assert np.std(yields[-1, :, 0]) > 0.01
    assert np.std(np.diff(yields, axis=0)) > 0.01
    # Even paths occupying one regime retain distinct curve factors.
    state = first[0][-1, 0]
    assert np.std(yields[-1, first[0][-1] == state, 0]) > 0.001
    result = simulate_returns(model, 12, 16, random_seed=4, joint_macro=True)
    assert np.isfinite(result.returns).all()


def test_regime_and_macro_predictors_affect_conditional_curve():
    _, macro = history()
    columns = list(macro)
    membership = np.eye(4)[np.arange(len(macro)) % 4]
    spec, predicted = fit_curve(
        macro.to_numpy(),
        columns,
        np.ones(len(macro) - 1, bool),
        membership,
        ["a", "b", "c", "d"],
        ["growth", "inflation", "interest_rate"],
    )
    np.testing.assert_allclose(predicted, curve_forecast(macro.to_numpy()[:-1], spec, membership[:-1]))
    # Controlled coefficients verify the runtime uses both conditioning inputs.
    spec["coefficient"] = np.zeros_like(spec["coefficient"]).tolist()
    spec["coefficient"][3][0] = 0.1
    spec["state_drifts"] = [[0, 0, 0], [0.2, 0, 0], [0, 0, 0], [0, 0, 0]]
    current = np.repeat(macro.to_numpy()[-1:], 2, axis=0)
    current[1, columns.index("growth")] += spec["scale"][3]
    forecast = curve_forecast(current, spec, np.eye(4)[:2])
    np.testing.assert_allclose(forecast[1] - forecast[0], 0.3)


def test_bootstrap_restores_observed_origin_and_retains_curve_parameters():
    returns, macro = history()
    options = dict(joint_macro=True, economic_returns={"yield_model": "nelson_siegel"})
    reference = calibrate_quadrant_model(returns, macro, **options)
    models = bootstrap_quadrant_models(returns, macro, draws=2, random_seed=2, **options)
    anchored = anchor_parameter_models(models, reference)
    for before, after in zip(models, anchored):
        np.testing.assert_allclose(
            after.metadata["macro_dynamics"]["latest"], reference.metadata["macro_dynamics"]["latest"]
        )
        assert (
            before.metadata["macro_dynamics"]["curve_dynamics"]
            == after.metadata["macro_dynamics"]["curve_dynamics"]
        )
        _, paths, _ = simulate_joint_regime_macro_paths(after, 6, 8, random_seed=3)
        assert np.isfinite(paths).all()


def test_missing_curve_inputs_fail_explicitly():
    returns, macro = history()
    with pytest.raises(ValueError, match="bond_yield_5y"):
        calibrate_quadrant_model(
            returns,
            macro.drop(columns="bond_yield_5y"),
            joint_macro=True,
            economic_returns={"yield_model": "nelson_siegel"},
        )
    assert economic_assumptions({"yield_model": "nelson_siegel"})["yield_model"] == "nelson_siegel"


def test_invalid_bootstrap_joins_cannot_train_curve_dynamics():
    _, macro = history()
    values = macro.to_numpy()
    valid = np.ones(len(values) - 1, bool)
    valid[39:41] = False
    membership = np.eye(4)[np.arange(len(values)) % 4]
    args = (list(macro), valid, membership, ["a", "b", "c", "d"], ["growth", "inflation", "interest_rate"])
    original, _ = fit_curve(values, *args)
    contaminated = values.copy()
    contaminated[40] += 100
    changed, _ = fit_curve(contaminated, *args)
    assert original == changed


def test_curve_calibration_is_invariant_to_yield_units():
    _, macro = history()
    values = macro.to_numpy()
    membership = np.eye(4)[np.arange(len(values)) % 4]
    args = (
        list(macro),
        np.ones(len(values) - 1, bool),
        membership,
        ["a", "b", "c", "d"],
        ["growth", "inflation", "interest_rate"],
    )
    _, percent = fit_curve(values, *args)
    decimal_values = values.copy()
    decimal_values[:, : len(CURVE_TENORS)] /= 100
    _, decimal = fit_curve(decimal_values, *args)
    np.testing.assert_allclose(decimal, percent / 100, atol=1e-12)
