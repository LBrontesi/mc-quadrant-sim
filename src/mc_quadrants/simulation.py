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


def simulate_regime_paths(
    model: ScenarioModel,
    periods: int,
    paths: int,
    start_state: str | None = None,
    random_seed: int | None = None,
) -> np.ndarray:
    """Simulate Markov regime paths."""

    model.validate()
    if periods <= 0 or paths <= 0:
        raise ValueError("periods and paths must be positive.")

    rng = _rng(random_seed)
    states = model.states
    transition = model.transition_matrix.loc[states, states].to_numpy(dtype=float)

    if start_state is None:
        start_probabilities = stationary_distribution(model.transition_matrix.loc[states, states]).to_numpy()
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
) -> SimulationResult:
    """Simulate regime-dependent multivariate asset returns."""

    regime_paths = simulate_regime_paths(
        model,
        periods=periods,
        paths=paths,
        start_state=start_state,
        random_seed=random_seed,
    )
    rng = _rng(None if random_seed is None else random_seed + 1)
    assets = model.assets
    returns = np.empty((periods, paths, len(assets)), dtype=float)

    for period in range(periods):
        for state in model.states:
            mask = regime_paths[period] == state
            if not mask.any():
                continue
            moments = model.moments[state]
            mean = moments.mean.reindex(assets).to_numpy(dtype=float)
            covariance = moments.covariance.reindex(index=assets, columns=assets).to_numpy(dtype=float)
            covariance = nearest_psd(covariance)
            returns[period, mask, :] = rng.multivariate_normal(mean, covariance, size=mask.sum())

    return SimulationResult(
        returns=returns,
        regimes=regime_paths,
        assets=assets,
        states=model.states.copy(),
        frequency=model.frequency,
    )


def simulate_portfolio_paths(
    result: SimulationResult,
    weights: Mapping[str, float],
    initial_value: float = 100.0,
    return_kind: str = "log",
) -> pd.DataFrame:
    """Convert simulated asset returns into portfolio wealth paths."""

    weight_vector = pd.Series(weights, dtype=float).reindex(result.assets).fillna(0.0)
    if np.isclose(weight_vector.sum(), 0.0):
        raise ValueError("Portfolio weights must have a non-zero sum.")
    if not np.isclose(weight_vector.sum(), 1.0):
        weight_vector = weight_vector / weight_vector.sum()

    portfolio_returns = result.returns @ weight_vector.to_numpy(dtype=float)
    if return_kind == "log":
        wealth = initial_value * np.exp(np.cumsum(portfolio_returns, axis=0))
    elif return_kind == "simple":
        wealth = initial_value * np.cumprod(1.0 + portfolio_returns, axis=0)
    else:
        raise ValueError("return_kind must be 'log' or 'simple'.")

    return pd.DataFrame(wealth, columns=[f"path_{i}" for i in range(result.returns.shape[1])])


def summarize_terminal_wealth(wealth: pd.DataFrame) -> pd.Series:
    """Summarize terminal wealth across Monte Carlo paths."""

    terminal = wealth.iloc[-1]
    return pd.Series(
        {
            "mean": terminal.mean(),
            "std": terminal.std(),
            "p05": terminal.quantile(0.05),
            "p50": terminal.quantile(0.50),
            "p95": terminal.quantile(0.95),
        }
    )
