"""Four-quadrant macro regime Monte Carlo simulator."""

from mc_quadrants.calibration import calibrate_quadrant_model, estimate_regime_moments
from mc_quadrants.data import backfill_prices
from mc_quadrants.diagnostics import CalibrationDiagnostics, build_calibration_diagnostics
from mc_quadrants.pipeline import SimulationRun, compare_distributions, run_scenario
from mc_quadrants.regimes import REGIME_ORDER, Regime, classify_quadrants, estimate_transition_matrix
from mc_quadrants.simulation import simulate_portfolio_paths, simulate_returns, summarize_wealth_risk
from mc_quadrants.types import RegimeMoments, ScenarioModel, SimulationResult

__all__ = [
    "REGIME_ORDER",
    "Regime",
    "RegimeMoments",
    "ScenarioModel",
    "SimulationResult",
    "CalibrationDiagnostics",
    "SimulationRun",
    "compare_distributions",
    "build_calibration_diagnostics",
    "backfill_prices",
    "calibrate_quadrant_model",
    "classify_quadrants",
    "estimate_regime_moments",
    "estimate_transition_matrix",
    "simulate_portfolio_paths",
    "simulate_returns",
    "summarize_wealth_risk",
    "run_scenario",
]
