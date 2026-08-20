"""Four-quadrant macro regime Monte Carlo simulator."""

from mc_quadrants.calibration import (
    calibrate_quadrant_model,
    estimate_regime_moments,
    estimate_weighted_regime_moments,
)
from mc_quadrants.data import (
    align_macro_to_availability,
    backfill_prices,
    combine_observed_and_simulated_returns,
    convert_returns_to_base_currency,
    fetch_fred_initial_release,
    fetch_yahoo_fx_rates,
    simulate_pre_inception_returns,
)
from mc_quadrants.diagnostics import CalibrationDiagnostics, build_calibration_diagnostics
from mc_quadrants.hmm import HmmFit, fit_hmm_model
from mc_quadrants.matrix import nearest_psd_higham
from mc_quadrants.pipeline import SimulationRun, run_scenario
from mc_quadrants.regimes import (
    REGIME_ORDER,
    Regime,
    classify_persistent_quadrants,
    classify_quadrants,
    estimate_duration_hazards,
    estimate_probabilistic_transition_matrix,
    estimate_transition_matrix,
    quadrant_probabilities,
    smooth_macro_for_regimes,
    sojourn_durations,
)
from mc_quadrants.simulation import (
    inflation_adjust_wealth,
    simulate_joint_regime_macro_paths,
    simulate_portfolio_paths,
    simulate_regime_paths,
    simulate_returns,
    summarize_wealth_risk,
)
from mc_quadrants.tax_policy import (
    TaxPolicy,
    TaxSelection,
    TaxSimulationContext,
    available_tax_countries,
    register_tax_policy,
    resolve_tax_selection,
)
from mc_quadrants.types import MNTSParameters, RegimeMoments, ScenarioModel, SimulationResult
from mc_quadrants.uncertainty import bootstrap_quadrant_models, stationary_bootstrap_indices
from mc_quadrants.validation import WalkForwardResult, walk_forward_validation

__all__ = [
    "REGIME_ORDER",
    "Regime",
    "MNTSParameters",
    "RegimeMoments",
    "ScenarioModel",
    "SimulationResult",
    "CalibrationDiagnostics",
    "SimulationRun",
    "WalkForwardResult",
    "HmmFit",
    "TaxPolicy",
    "TaxSelection",
    "TaxSimulationContext",
    "available_tax_countries",
    "register_tax_policy",
    "resolve_tax_selection",
    "build_calibration_diagnostics",
    "backfill_prices",
    "align_macro_to_availability",
    "fetch_fred_initial_release",
    "bootstrap_quadrant_models",
    "stationary_bootstrap_indices",
    "combine_observed_and_simulated_returns",
    "convert_returns_to_base_currency",
    "calibrate_quadrant_model",
    "classify_quadrants",
    "classify_persistent_quadrants",
    "estimate_duration_hazards",
    "estimate_regime_moments",
    "estimate_weighted_regime_moments",
    "estimate_probabilistic_transition_matrix",
    "estimate_transition_matrix",
    "fetch_yahoo_fx_rates",
    "simulate_portfolio_paths",
    "simulate_joint_regime_macro_paths",
    "simulate_regime_paths",
    "simulate_returns",
    "simulate_pre_inception_returns",
    "summarize_wealth_risk",
    "inflation_adjust_wealth",
    "quadrant_probabilities",
    "smooth_macro_for_regimes",
    "run_scenario",
    "fit_hmm_model",
    "walk_forward_validation",
    "sojourn_durations",
    "nearest_psd_higham",
]
