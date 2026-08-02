from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from mc_quadrants.matrix import nearest_psd
from mc_quadrants.types import ScenarioModel, SimulationResult


def stationary_distribution(transition_matrix: pd.DataFrame) -> pd.Series:
    """Compute the long-run state distribution implied by a transition matrix."""

    matrix = transition_matrix.to_numpy(dtype=float).T
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    closest = np.argmin(np.abs(eigenvalues - 1.0))
    vector = np.real(eigenvectors[:, closest])
    if vector.sum() < 0:
        vector = -vector
    vector = np.maximum(vector, 0.0)
    if vector.sum() == 0:
        vector = np.ones(len(vector))
    vector = vector / vector.sum()
    return pd.Series(vector, index=transition_matrix.index)


def _rng(random_seed: int | None = None) -> np.random.Generator:
    return np.random.default_rng(random_seed)


def _sample_multivariate_t(
    rng: np.random.Generator,
    mean: np.ndarray,
    covariance: np.ndarray,
    size: int,
    degrees_of_freedom: float,
) -> np.ndarray:
    """Draw Student-t returns with the requested covariance matrix."""

    # Scale the normal mixture so the resulting t distribution retains the
    # calibrated covariance when degrees_of_freedom is finite.
    scale = covariance * (degrees_of_freedom - 2.0) / degrees_of_freedom
    normal_draws = rng.multivariate_normal(np.zeros(len(mean)), scale, size=size)
    chi_squared = rng.chisquare(degrees_of_freedom, size=size)
    return mean + normal_draws / np.sqrt(chi_squared / degrees_of_freedom)[:, None]


def _sample_transition_matrix(
    rng: np.random.Generator,
    transition: np.ndarray,
    concentration: float,
) -> np.ndarray:
    """Draw a transition matrix from row-wise Dirichlet distributions."""

    if not np.isfinite(concentration) or concentration <= 0:
        raise ValueError("transition_concentration must be positive and finite.")
    return np.vstack(
        [rng.dirichlet(np.maximum(row, 1e-12) * concentration) for row in transition]
    )


def simulate_regime_paths(
    model: ScenarioModel,
    periods: int,
    paths: int,
    start_state: str | None = None,
    random_seed: int | None = None,
    transition_concentration: float | None = None,
) -> np.ndarray:
    """Simulate Markov regime paths."""

    model.validate()
    if periods <= 0 or paths <= 0:
        raise ValueError("periods and paths must be positive.")

    rng = _rng(random_seed)
    states = model.states
    transition = model.transition_matrix.loc[states, states].to_numpy(dtype=float)
    if transition_concentration is not None:
        transition = _sample_transition_matrix(rng, transition, transition_concentration)

    if start_state is None:
        sampled_transition = pd.DataFrame(transition, index=states, columns=states)
        start_probabilities = stationary_distribution(sampled_transition).to_numpy()
        current = rng.choice(len(states), size=paths, p=start_probabilities)
    else:
        if start_state not in states:
            raise ValueError(f"Unknown start_state: {start_state}")
        current = np.full(paths, states.index(start_state), dtype=int)

    simulated = np.empty((periods, paths), dtype=object)
    for period in range(periods):
        simulated[period] = [states[index] for index in current]
        next_state = np.empty(paths, dtype=int)
        for state_index in range(len(states)):
            mask = current == state_index
            if mask.any():
                next_state[mask] = rng.choice(len(states), size=mask.sum(), p=transition[state_index])
        current = next_state

    return simulated


def simulate_returns(
    model: ScenarioModel,
    periods: int,
    paths: int,
    start_state: str | None = None,
    random_seed: int | None = None,
    distribution: str = "normal",
    degrees_of_freedom: float = 5.0,
    block_size: int = 3,
    transition_concentration: float | None = None,
) -> SimulationResult:
    """Simulate regime-dependent multivariate asset returns.

    ``distribution="student_t"`` preserves each regime's calibrated mean and
    covariance while allowing more extreme outcomes than a Gaussian draw.
    ``distribution="bootstrap"`` samples historical observations for each
    regime, while ``distribution="block_bootstrap"`` keeps short blocks of
    observations together. Finite-variance Student-t sampling requires more
    than two degrees of freedom.
    """

    distribution = distribution.lower().replace("-", "_")
    if distribution not in {"normal", "student_t", "t", "bootstrap", "block_bootstrap"}:
        raise ValueError("distribution must be 'normal', 'student_t', 'bootstrap', or 'block_bootstrap'.")
    if distribution == "t":
        distribution = "student_t"
    if distribution == "student_t" and degrees_of_freedom <= 2:
        raise ValueError("degrees_of_freedom must be greater than 2 for Student-t returns.")
    if block_size <= 0:
        raise ValueError("block_size must be positive.")

    regime_paths = simulate_regime_paths(
        model,
        periods=periods,
        paths=paths,
        start_state=start_state,
        random_seed=random_seed,
        transition_concentration=transition_concentration,
    )
    rng = _rng(None if random_seed is None else random_seed + 1)
    assets = model.assets
    returns = np.empty((periods, paths, len(assets)), dtype=float)
    bootstrap_starts = np.full((len(model.states), paths), -1, dtype=int)
    bootstrap_offsets = np.full((len(model.states), paths), block_size, dtype=int)
    previous_state_indices = np.full(paths, -1, dtype=int)

    for period in range(periods):
        for state_index, state in enumerate(model.states):
            mask = regime_paths[period] == state
            if not mask.any():
                continue
            path_indices = np.flatnonzero(mask)
            if distribution in {"bootstrap", "block_bootstrap"}:
                historical = model.historical_returns.get(state)
                if historical is None or historical.empty:
                    raise ValueError(
                        f"No historical returns are available for bootstrap regime: {state}"
                    )
                historical_values = historical.loc[:, assets].to_numpy(dtype=float)
                if distribution == "bootstrap":
                    row_indices = rng.integers(len(historical_values), size=len(path_indices))
                else:
                    state_starts = bootstrap_starts[state_index, path_indices]
                    state_offsets = bootstrap_offsets[state_index, path_indices]
                    new_state = previous_state_indices[path_indices] != state_index
                    reset_block = new_state | (state_offsets >= block_size)
                    state_starts[reset_block] = rng.integers(
                        len(historical_values),
                        size=reset_block.sum(),
                    )
                    state_offsets[reset_block] = 0
                    row_indices = (state_starts + state_offsets) % len(historical_values)
                    state_offsets += 1
                    bootstrap_starts[state_index, path_indices] = state_starts
                    bootstrap_offsets[state_index, path_indices] = state_offsets
                draws = historical_values[row_indices]
            else:
                moments = model.moments[state]
                mean = moments.mean.reindex(assets).to_numpy(dtype=float)
                covariance = moments.covariance.reindex(index=assets, columns=assets).to_numpy(dtype=float)
                covariance = nearest_psd(covariance)
                if distribution == "normal":
                    draws = rng.multivariate_normal(mean, covariance, size=mask.sum())
                else:
                    draws = _sample_multivariate_t(
                        rng,
                        mean,
                        covariance,
                        size=mask.sum(),
                        degrees_of_freedom=float(degrees_of_freedom),
                    )
            returns[period, mask, :] = draws
            previous_state_indices[path_indices] = state_index

    return SimulationResult(
        returns=returns,
        regimes=regime_paths,
        assets=assets,
        states=model.states.copy(),
        frequency=model.frequency,
        distribution=distribution,
        degrees_of_freedom=(float(degrees_of_freedom) if distribution == "student_t" else None),
        transition_concentration=transition_concentration,
    )


def simulate_portfolio_paths(
    result: SimulationResult,
    weights: Mapping[str, float],
    initial_value: float = 100.0,
    return_kind: str = "log",
    rebalance_frequency: int | None = None,
    transaction_cost_bps: float = 0.0,
) -> pd.DataFrame:
    """Convert simulated asset returns into portfolio wealth paths.

    With ``rebalance_frequency=None`` the original weighted-return behavior is
    retained. Set a positive frequency to model holdings, periodic rebalancing,
    and transaction costs in basis points charged on traded notional.
    """

    weight_vector = pd.Series(weights, dtype=float).reindex(result.assets).fillna(0.0)
    if np.isclose(weight_vector.sum(), 0.0):
        raise ValueError("Portfolio weights must have a non-zero sum.")
    if not np.isclose(weight_vector.sum(), 1.0):
        weight_vector = weight_vector / weight_vector.sum()

    if return_kind not in {"log", "simple"}:
        raise ValueError("return_kind must be 'log' or 'simple'.")
    if rebalance_frequency is not None:
        rebalance_frequency = int(rebalance_frequency)
        if rebalance_frequency <= 0:
            raise ValueError("rebalance_frequency must be positive or None.")
    if not np.isfinite(transaction_cost_bps) or transaction_cost_bps < 0:
        raise ValueError("transaction_cost_bps must be a non-negative number.")
    if rebalance_frequency is None and not np.isclose(transaction_cost_bps, 0.0):
        raise ValueError("Transaction costs require a rebalancing frequency.")

    if rebalance_frequency is None:
        portfolio_returns = result.returns @ weight_vector.to_numpy(dtype=float)
        if return_kind == "log":
            wealth = initial_value * np.exp(np.cumsum(portfolio_returns, axis=0))
        else:
            wealth = initial_value * np.cumprod(1.0 + portfolio_returns, axis=0)
        return pd.DataFrame(wealth, columns=[f"path_{i}" for i in range(result.returns.shape[1])])

    if return_kind == "log":
        asset_growth = np.exp(result.returns)
    else:
        asset_growth = 1.0 + result.returns
        if (asset_growth <= 0).any():
            raise ValueError("Simple returns must be greater than -100% for rebalancing.")

    periods, paths, assets = result.returns.shape
    holdings = np.broadcast_to(
        initial_value * weight_vector.to_numpy(dtype=float),
        (paths, assets),
    ).copy()
    wealth = np.empty((periods, paths), dtype=float)
    cost_rate = float(transaction_cost_bps) / 10_000.0

    for period in range(periods):
        holdings *= asset_growth[period]
        value_before_rebalance = holdings.sum(axis=1)
        if (period + 1) % rebalance_frequency == 0:
            target_holdings = value_before_rebalance[:, None] * weight_vector.to_numpy(dtype=float)
            turnover = np.abs(target_holdings - holdings).sum(axis=1)
            costs = turnover * cost_rate
            value_after_costs = value_before_rebalance - costs
            holdings = value_after_costs[:, None] * weight_vector.to_numpy(dtype=float)
            wealth[period] = value_after_costs
        else:
            wealth[period] = value_before_rebalance

    return pd.DataFrame(wealth, columns=[f"path_{i}" for i in range(result.returns.shape[1])])


def summarize_terminal_wealth(wealth: pd.DataFrame) -> pd.Series:
    """Summarize terminal wealth and downside risk across Monte Carlo paths."""

    return summarize_wealth_risk(wealth)


def summarize_wealth_risk(
    wealth: pd.DataFrame,
    initial_value: float = 100.0,
    confidence: float = 0.95,
) -> pd.Series:
    """Calculate terminal, loss-tail, and drawdown metrics."""

    if wealth.empty or wealth.shape[1] == 0:
        raise ValueError("wealth must contain at least one simulated path.")
    if not np.isfinite(initial_value) or initial_value <= 0:
        raise ValueError("initial_value must be positive and finite.")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1.")

    terminal = wealth.iloc[-1]
    tail_probability = 1.0 - confidence
    lower_tail = float(terminal.quantile(tail_probability))
    tail = terminal[terminal <= lower_tail]
    wealth_with_initial = pd.concat(
        [pd.DataFrame([np.full(wealth.shape[1], initial_value)], columns=wealth.columns), wealth],
        ignore_index=True,
    )
    running_max = wealth_with_initial.cummax(axis=0)
    drawdown = wealth_with_initial / running_max - 1.0
    max_drawdown = -drawdown.min(axis=0)
    return pd.Series(
        {
            "mean": terminal.mean(),
            "std": terminal.std(),
            "p05": terminal.quantile(0.05),
            "p50": terminal.quantile(0.50),
            "p95": terminal.quantile(0.95),
            "probability_of_loss": float((terminal < initial_value).mean()),
            "var_95": initial_value - lower_tail,
            "expected_shortfall_95": initial_value - float(tail.mean()),
            "max_drawdown_mean": float(max_drawdown.mean()),
            "max_drawdown_p95": float(max_drawdown.quantile(0.95)),
            "max_drawdown_worst": float(max_drawdown.max()),
        }
    )
