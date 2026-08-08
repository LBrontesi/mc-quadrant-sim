"""Four-quadrant macro regime Monte Carlo simulator."""

from mc_quadrants.calibration import calibrate_quadrant_model, estimate_regime_moments
from mc_quadrants.data import (
    backfill_prices,
    combine_observed_and_simulated_returns,
    convert_returns_to_base_currency,
    fetch_yahoo_fx_rates,
    simulate_pre_inception_returns,
)
from mc_quadrants.diagnostics import CalibrationDiagnostics, build_calibration_diagnostics
from mc_quadrants.hmm import HmmFit, fit_hmm_model
from mc_quadrants.matrix import nearest_psd_higham
from mc_quadrants.pipeline import SimulationRun, compare_distributions, run_scenario
from mc_quadrants.regimes import (
    REGIME_ORDER,
    Regime,
    classify_quadrants,
    estimate_transition_matrix,
    sojourn_durations,
)
from mc_quadrants.simulation import (
    simulate_portfolio_paths,
    simulate_regime_paths,
    simulate_returns,
    summarize_wealth_risk,
)
from mc_quadrants.types import RegimeMoments, ScenarioModel, SimulationResult
from mc_quadrants.validation import WalkForwardResult, walk_forward_validation

__all__ = [
    "REGIME_ORDER",
    "Regime",
    "RegimeMoments",
    "ScenarioModel",
    "SimulationResult",
    "CalibrationDiagnostics",
    "SimulationRun",
    "WalkForwardResult",
    "HmmFit",
    "compare_distributions",
    "build_calibration_diagnostics",
    "backfill_prices",
    "combine_observed_and_simulated_returns",
    "convert_returns_to_base_currency",
    "calibrate_quadrant_model",
    "classify_quadrants",
    "estimate_regime_moments",
    "estimate_transition_matrix",
    "fetch_yahoo_fx_rates",
    "simulate_portfolio_paths",
    "simulate_regime_paths",
    "simulate_returns",
    "simulate_pre_inception_returns",
    "summarize_wealth_risk",
    "run_scenario",
    "fit_hmm_model",
    "walk_forward_validation",
    "sojourn_durations",
    "nearest_psd_higham",
]
