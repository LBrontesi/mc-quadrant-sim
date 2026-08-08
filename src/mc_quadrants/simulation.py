from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from mc_quadrants.matrix import nearest_psd
from mc_quadrants.types import ScenarioModel, SimulationResult


def stationary_distribution(transition_matrix: pd.DataFrame) -> pd.Series:
    """Compute the long-run state distribution implied by a transition matrix."""

    if transition_matrix.empty:
        raise ValueError("transition_matrix must contain at least one state.")
    if transition_matrix.index.has_duplicates or transition_matrix.columns.has_duplicates:
        raise ValueError("Transition matrix labels must be unique.")
    if set(transition_matrix.index) != set(transition_matrix.columns):
        raise ValueError("Transition matrix rows and columns must contain the same states.")

    states = list(transition_matrix.index)
    matrix = transition_matrix.loc[states, states].to_numpy(dtype=float)
    if not np.isfinite(matrix).all() or (matrix < 0).any():
        raise ValueError("Transition matrix must contain finite, non-negative probabilities.")
    if not np.allclose(matrix.sum(axis=1), 1.0):
        raise ValueError("Transition matrix rows must sum to 1.")

    # Solve pi P = pi with a normalization row instead of selecting an
    # eigenvector, which is unstable for repeated or nearly repeated eigenvalues.
    system = matrix.T - np.eye(len(states))
    system[-1] = 1.0
    target = np.zeros(len(states))
    target[-1] = 1.0
    vector, *_ = np.linalg.lstsq(system, target, rcond=None)
    vector = np.clip(vector, 0.0, None)
    if not np.isfinite(vector).all() or np.isclose(vector.sum(), 0.0):
        vector = np.ones(len(states), dtype=float)
    vector /= vector.sum()
    return pd.Series(vector, index=states)


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
    return np.vstack([rng.dirichlet(np.maximum(row, 1e-12) * concentration) for row in transition])


def _sample_sojourn(
    rng: np.random.Generator,
    state: int,
    duration_map: dict[str, np.ndarray],
    states: list[str],
    transition: np.ndarray,
    min_duration: int = 3,
) -> int:
    """Draw a sojourn length for a state from its empirical run distribution.

    States never observed in history fall back to the geometric duration
    implied by the transition matrix, capped to stay conservative. A minimum
    duration floor prevents unrealistic single-period regime flips.
    """

    observed = duration_map.get(states[state], np.array([], dtype=int))
    if len(observed):
        valid = observed[observed >= min_duration]
        if len(valid):
            return int(rng.choice(valid))
        return int(np.maximum(int(rng.choice(observed)), min_duration))
    persistence = float(transition[state, state])
    if persistence < 1.0:
        mean_duration = 1.0 / (1.0 - persistence)
        return int(np.clip(round(mean_duration), min_duration, 240))
    return max(min_duration, 1)


def simulate_regime_paths(
    model: ScenarioModel,
    periods: int,
    paths: int,
    start_state: str | None = None,
    random_seed: int | None = None,
    transition_concentration: float | None = None,
    duration_model: str = "markov",
    min_regime_duration: int = 3,
) -> np.ndarray:
    """Simulate Markov (or semi-Markov) regime paths.

    ``duration_model="semi_markov"`` replaces the geometric sojourn times of a
    first-order Markov chain with the empirical run-length distribution of
    each state observed in history (stored in ``model.metadata`` under
    ``sojourn_durations``). Regimes then persist for realistic lengths instead
    of flipping with the constant per-period probability implied by the
    transition matrix. Transitions between states still follow the calibrated
    matrix with the self-transition probabilities renormalized away.
    """

    model.validate()
    if periods <= 0 or paths <= 0:
        raise ValueError("periods and paths must be positive.")

    rng = _rng(random_seed)
    states = model.states
    transition = model.transition_matrix.loc[states, states].to_numpy(dtype=float)
    if transition_concentration is not None:
        transition = _sample_transition_matrix(rng, transition, transition_concentration)

    duration_model = str(duration_model).lower()
    if duration_model not in {"markov", "semi_markov"}:
        raise ValueError("duration_model must be 'markov' or 'semi_markov'.")
    sojourns: dict[str, np.ndarray] | None = None
    if duration_model == "semi_markov":
        sojourns = model.metadata.get("sojourn_durations")
        if not isinstance(sojourns, dict):
            raise ValueError(
                "semi_markov requires the model to expose 'sojourn_durations' "
                "in its metadata; calibrate the model with sojourn support."
            )

    if start_state is None:
        sampled_transition = pd.DataFrame(transition, index=states, columns=states)
        start_probabilities = stationary_distribution(sampled_transition).to_numpy()
        current = rng.choice(len(states), size=paths, p=start_probabilities)
    else:
        if start_state not in states:
            raise ValueError(f"Unknown start_state: {start_state}")
        current = np.full(paths, states.index(start_state), dtype=int)

    if duration_model == "markov":
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

    simulated = np.empty((periods, paths), dtype=object)
    remaining = np.zeros(paths, dtype=int)
    for period in range(periods):
        simulated[period] = [states[index] for index in current]
        remaining -= 1
        for state_index in range(len(states)):
            mask = (current == state_index) & (remaining <= 0)
            if not mask.any():
                continue
            other_states = [index for index in range(len(states)) if index != state_index]
            probabilities = transition[state_index, other_states]
            probabilities = probabilities / max(float(probabilities.sum()), 1e-300)
            following = rng.choice(other_states, size=mask.sum(), p=probabilities)
            next_sojourns = np.array(
                [_sample_sojourn(rng, int(index), sojourns, states, transition, min_regime_duration) for index in following],
                dtype=int,
            )
            current[mask] = following
            remaining[mask] = next_sojourns - 1
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
    duration_model: str = "markov",
    min_regime_duration: int = 3,
    garch: bool = False,
    garch_alpha: float = 0.10,
    garch_beta: float = 0.85,
) -> SimulationResult:
    """Simulate regime-dependent multivariate asset returns.

    ``distribution="student_t"`` preserves each regime's calibrated mean and
    covariance while allowing more extreme outcomes than a Gaussian draw.
    ``distribution="bootstrap"`` samples historical observations for each
    regime, while ``distribution="block_bootstrap"`` keeps short blocks of
    observations together. Finite-variance Student-t sampling requires more
    than two degrees of freedom.

    ``duration_model="semi_markov"`` draws regime run lengths from the
    empirical sojourn distribution instead of the geometric lengths implied by
    a first-order chain.

    ``garch=True`` (Gaussian draws only) adds GARCH(1,1) conditional variance
    dynamics within each regime: each asset's unconditional variance anchors
    the long-run level, ``garch_alpha`` governs the response to new shocks,
    and ``garch_beta`` the persistence of past variance. Variance is re-anchored
    to the new regime's level when a path switches states, so volatility
    clusters without drifting away from the calibrated regime covariance.
    """

    distribution = str(distribution).lower().replace("-", "_")
    if distribution not in {"normal", "student_t", "t", "bootstrap", "block_bootstrap"}:
        raise ValueError("distribution must be 'normal', 'student_t', 'bootstrap', or 'block_bootstrap'.")
    if distribution == "t":
        distribution = "student_t"
    if distribution == "student_t" and (not np.isfinite(degrees_of_freedom) or degrees_of_freedom <= 2):
        raise ValueError("degrees_of_freedom must be finite and greater than 2 for Student-t returns.")
    if block_size <= 0:
        raise ValueError("block_size must be positive.")
    if not np.isfinite(garch_alpha) or not 0 <= garch_alpha < 1:
        raise ValueError("garch_alpha must be between 0 and 1.")
    if not np.isfinite(garch_beta) or not 0 <= garch_beta < 1:
        raise ValueError("garch_beta must be between 0 and 1.")
    if garch_alpha + garch_beta >= 1:
        raise ValueError("garch_alpha + garch_beta must be less than 1.")
    if garch and distribution != "normal":
        raise ValueError("GARCH volatility clustering requires distribution='normal'.")

    regime_paths = simulate_regime_paths(
        model,
        periods=periods,
        paths=paths,
        start_state=start_state,
        random_seed=random_seed,
        transition_concentration=transition_concentration,
        duration_model=duration_model,
        min_regime_duration=min_regime_duration,
    )
    rng = _rng(None if random_seed is None else random_seed + 1)
    assets = model.assets
    returns = np.empty((periods, paths, len(assets)), dtype=float)
    bootstrap_starts = np.full((len(model.states), paths), -1, dtype=int)
    bootstrap_offsets = np.full((len(model.states), paths), block_size, dtype=int)
    previous_state_indices = np.full(paths, -1, dtype=int)

    garch_levels: dict[str, np.ndarray] | None = None
    garch_omega: dict[str, np.ndarray] | None = None
    conditional_variance: np.ndarray | None = None
    if garch:
        garch_levels = {
            state: np.diag(model.moments[state].covariance.to_numpy(dtype=float)) for state in model.states
        }
        garch_omega = {
            state: (1.0 - garch_alpha - garch_beta) * level for state, level in garch_levels.items()
        }
        conditional_variance = np.empty((paths, len(assets)), dtype=float)

    for period in range(periods):
        for state_index, state in enumerate(model.states):
            mask = regime_paths[period] == state
            if not mask.any():
                continue
            path_indices = np.flatnonzero(mask)
            if distribution in {"bootstrap", "block_bootstrap"}:
                historical = model.historical_returns.get(state)
                if historical is None or historical.empty:
                    raise ValueError(f"No historical returns are available for bootstrap regime: {state}")
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
                if garch:
                    levels = garch_levels[state]
                    omega = garch_omega[state]
                    reanchored = (period == 0) | (previous_state_indices[path_indices] != state_index)
                    if reanchored.any():
                        conditional_variance[path_indices[reanchored]] = levels
                    correlation = moments.correlation.reindex(index=assets, columns=assets).to_numpy(
                        dtype=float
                    )
                    correlation = nearest_psd(correlation)
                    innovations = rng.multivariate_normal(
                        np.zeros(len(assets)),
                        correlation,
                        size=mask.sum(),
                    )
                    scale = np.sqrt(conditional_variance[path_indices])
                    draws = mean + innovations * scale
                    conditional_variance[path_indices] = (
                        omega
                        + garch_alpha * (innovations * scale) ** 2
                        + garch_beta * conditional_variance[path_indices]
                    )
                elif distribution == "normal":
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


def _simulate_leveraged_portfolio_paths(
    asset_growth: np.ndarray,
    target_weights: np.ndarray,
    initial_value: float,
    rebalance_frequency: int,
    transaction_cost_bps: float,
    leverage_multiple: float,
    financing_rate: float,
    maintenance_margin: float,
    contribution: float,
    withdrawal: float,
) -> pd.DataFrame:
    """Simulate leveraged holdings with explicit debt and financing costs."""

    periods, paths, assets = asset_growth.shape
    holdings = np.broadcast_to(
        initial_value * leverage_multiple * target_weights,
        (paths, assets),
    ).copy()
    debt = np.full(paths, initial_value * (leverage_multiple - 1.0), dtype=float)
    wealth = np.empty((periods, paths), dtype=float)
    margin_calls = np.zeros(paths, dtype=bool)
    financing_growth = (1.0 + financing_rate) ** (1.0 / 12.0)
    cost_rate = float(transaction_cost_bps) / 10_000.0

    for period in range(periods):
        if contribution:
            holdings += contribution * leverage_multiple * target_weights
            debt += contribution * (leverage_multiple - 1.0)
        holdings *= asset_growth[period]
        debt *= financing_growth

        asset_value = holdings.sum(axis=1)
        equity = asset_value - debt
        if withdrawal:
            available = np.maximum(equity, 0.0)
            funded = np.minimum(withdrawal, available)
            fraction = np.divide(
                funded,
                asset_value,
                out=np.zeros_like(funded),
                where=asset_value > 0,
            )
            holdings -= holdings * fraction[:, None]
            exhausted = withdrawal > equity
            if exhausted.any():
                holdings[exhausted] = 0.0
                debt[exhausted] = 0.0
                margin_calls[exhausted] = True

        asset_value = holdings.sum(axis=1)
        equity = asset_value - debt
        margin_ratio = np.divide(
            equity,
            asset_value,
            out=np.zeros_like(equity),
            where=asset_value > 0,
        )
        breached = (equity <= 0) | ((asset_value > 0) & (margin_ratio < maintenance_margin))
        if breached.any():
            holdings[breached] = 0.0
            debt[breached] = 0.0
            equity[breached] = 0.0
            margin_calls[breached] = True

        if (period + 1) % rebalance_frequency == 0:
            active = ~margin_calls
            target_holdings = equity[:, None] * leverage_multiple * target_weights
            turnover = np.abs(target_holdings - holdings).sum(axis=1)
            costs = turnover * cost_rate
            equity_after_costs = equity - costs
            liquidate = active & (equity_after_costs <= 0)
            if liquidate.any():
                holdings[liquidate] = 0.0
                debt[liquidate] = 0.0
                equity_after_costs[liquidate] = 0.0
                margin_calls[liquidate] = True
            active = ~margin_calls
            holdings[active] = equity_after_costs[active, None] * leverage_multiple * target_weights
            debt[active] = equity_after_costs[active] * (leverage_multiple - 1.0)
            equity = equity_after_costs

        wealth[period] = np.maximum(equity, 0.0)

    frame = pd.DataFrame(wealth, columns=[f"path_{i}" for i in range(paths)])
    frame.attrs.update({"margin_calls": int(margin_calls.sum())})
    return frame


def simulate_portfolio_paths(
    result: SimulationResult,
    weights: Mapping[str, float],
    initial_value: float = 100.0,
    return_kind: str = "log",
    rebalance_frequency: int | None = None,
    transaction_cost_bps: float = 0.0,
    asset_expense_ratios: Mapping[str, float] | None = None,
    leverage_multiple: float = 1.0,
    financing_rate: float = 0.0,
    maintenance_margin: float = 0.0,
    contribution: float = 0.0,
    withdrawal: float = 0.0,
) -> pd.DataFrame:
    """Convert simulated asset returns into portfolio wealth paths.

    With ``rebalance_frequency=None`` the original weighted-return behavior is
    retained. Set a positive frequency to model holdings, periodic rebalancing,
    and transaction costs in basis points charged on traded notional.

    ``contribution`` and ``withdrawal`` are periodic cash flows in the same
    currency as ``initial_value``. A contribution is invested at the target
    allocation at the start of every period (dollar-cost averaging); a
    withdrawal is funded by selling a pro-rata slice of current holdings at
    the end of every period. Wealth is floored at zero, so a path cannot be
    driven negative by withdrawals.
    """

    if not np.isfinite(initial_value) or initial_value <= 0:
        raise ValueError("initial_value must be positive and finite.")
    if result.returns.ndim != 3 or result.returns.shape[2] != len(result.assets):
        raise ValueError("result.returns must have shape (periods, paths, assets).")
    if not np.isfinite(result.returns).all():
        raise ValueError("Simulated returns must contain only finite values.")
    if not np.isfinite(contribution) or contribution < 0:
        raise ValueError("contribution must be a finite, non-negative number.")
    if not np.isfinite(withdrawal) or withdrawal < 0:
        raise ValueError("withdrawal must be a finite, non-negative number.")
    if not np.isfinite(leverage_multiple) or leverage_multiple < 1:
        raise ValueError("leverage_multiple must be at least 1.0.")
    if not np.isfinite(financing_rate) or financing_rate < 0:
        raise ValueError("financing_rate must be a finite, non-negative number.")
    if not np.isfinite(maintenance_margin) or not 0 <= maintenance_margin < 1:
        raise ValueError("maintenance_margin must be between 0 and 1.")
    if np.isclose(leverage_multiple, 1.0) and not np.isclose(maintenance_margin, 0.0):
        raise ValueError("maintenance_margin only applies when leverage_multiple is greater than 1.")
    if leverage_multiple > 1.0 and maintenance_margin >= 1.0 / leverage_multiple:
        raise ValueError("maintenance_margin must be below the initial equity margin for the selected leverage.")

    provided_weights = pd.Series(weights, dtype=float)
    weight_vector = provided_weights.reindex(result.assets)
    missing_assets = ~pd.Index(result.assets).isin(provided_weights.index)
    weight_vector.loc[missing_assets] = 0.0
    if not np.isfinite(weight_vector.to_numpy(dtype=float)).all():
        raise ValueError("Portfolio weights must be finite numbers.")
    weight_total = float(weight_vector.sum())
    if not np.isfinite(weight_total) or np.isclose(weight_total, 0.0):
        raise ValueError("Portfolio weights must have a non-zero sum.")
    if not np.isclose(weight_total, 1.0):
        weight_vector = weight_vector / weight_total

    return_kind = str(return_kind).lower()
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
    if rebalance_frequency is None and (
        not np.isclose(leverage_multiple, 1.0)
        or not np.isclose(financing_rate, 0.0)
        or not np.isclose(maintenance_margin, 0.0)
    ):
        raise ValueError("Leverage and financing require a rebalancing frequency.")

    provided_expense_ratios = pd.Series(asset_expense_ratios or {}, dtype=float)
    expense_ratios = provided_expense_ratios.reindex(result.assets).fillna(0.0)
    if not np.isfinite(expense_ratios.to_numpy(dtype=float)).all() or (expense_ratios < 0).any() or (expense_ratios >= 1).any():
        raise ValueError("Asset expense ratios must be finite decimals between 0 and 1.")
    monthly_fee_log = np.log1p(-expense_ratios.to_numpy(dtype=float)) / 12.0
    monthly_fee_growth = np.exp(monthly_fee_log)

    if rebalance_frequency is None:
        if return_kind == "log":
            portfolio_returns = (result.returns + monthly_fee_log) @ weight_vector.to_numpy(dtype=float)
            growth = np.exp(portfolio_returns)
            wealth = initial_value * np.exp(np.cumsum(portfolio_returns, axis=0))
        else:
            net_asset_returns = (1.0 + result.returns) * monthly_fee_growth - 1.0
            portfolio_returns = net_asset_returns @ weight_vector.to_numpy(dtype=float)
            if (1.0 + portfolio_returns <= 0).any():
                raise ValueError("Simple returns must be greater than -100% for positive wealth.")
            growth = 1.0 + portfolio_returns
            wealth = initial_value * np.cumprod(1.0 + portfolio_returns, axis=0)
        if contribution or withdrawal:
            periods, paths = result.returns.shape[:2]
            value = np.full(paths, initial_value, dtype=float)
            wealth = np.empty((periods, paths), dtype=float)
            for period in range(periods):
                value = np.maximum((value + contribution) * growth[period] - withdrawal, 0.0)
                wealth[period] = value
        if not np.isfinite(wealth).all():
            raise ValueError("Portfolio wealth contains non-finite values.")
        frame = pd.DataFrame(wealth, columns=[f"path_{i}" for i in range(result.returns.shape[1])])
        frame.attrs.update({"margin_calls": 0})
        return frame

    if return_kind == "log":
        asset_growth = np.exp(result.returns + monthly_fee_log)
    else:
        asset_growth = (1.0 + result.returns) * monthly_fee_growth
        if (asset_growth <= 0).any():
            raise ValueError("Simple returns must be greater than -100% for rebalancing.")
    if not np.isfinite(asset_growth).all():
        raise ValueError("Asset growth contains non-finite values.")

    periods, paths, assets = result.returns.shape
    target_weights = weight_vector.to_numpy(dtype=float)
    if (
        not np.isclose(leverage_multiple, 1.0)
        or not np.isclose(financing_rate, 0.0)
        or not np.isclose(maintenance_margin, 0.0)
    ):
        return _simulate_leveraged_portfolio_paths(
            asset_growth,
            target_weights,
            initial_value,
            rebalance_frequency,
            transaction_cost_bps,
            leverage_multiple,
            financing_rate,
            maintenance_margin,
            contribution,
            withdrawal,
        )
    holdings = np.broadcast_to(
        initial_value * target_weights,
        (paths, assets),
    ).copy()
    wealth = np.empty((periods, paths), dtype=float)
    cost_rate = float(transaction_cost_bps) / 10_000.0

    for period in range(periods):
        if contribution:
            holdings += contribution * target_weights
        holdings *= asset_growth[period]
        if withdrawal:
            fraction = withdrawal / np.maximum(holdings.sum(axis=1), 1e-300)
            holdings -= holdings * fraction[:, None]
            holdings = np.maximum(holdings, 0.0)
        value_before_rebalance = holdings.sum(axis=1)
        if (period + 1) % rebalance_frequency == 0:
            target_holdings = value_before_rebalance[:, None] * target_weights
            turnover = np.abs(target_holdings - holdings).sum(axis=1)
            costs = turnover * cost_rate
            value_after_costs = value_before_rebalance - costs
            holdings = value_after_costs[:, None] * target_weights
            wealth[period] = value_after_costs
        else:
            wealth[period] = value_before_rebalance

    if not np.isfinite(wealth).all():
        raise ValueError("Portfolio wealth contains non-finite values.")
    frame = pd.DataFrame(wealth, columns=[f"path_{i}" for i in range(result.returns.shape[1])])
    frame.attrs.update({"margin_calls": 0})
    return frame


def summarize_terminal_wealth(wealth: pd.DataFrame) -> pd.Series:
    """Summarize terminal wealth and downside risk across Monte Carlo paths."""

    return summarize_wealth_risk(wealth)


def summarize_wealth_risk(
    wealth: pd.DataFrame,
    initial_value: float = 100.0,
    confidence: float = 0.95,
    periods_per_year: float = 12.0,
    risk_free_rate: float = 0.0,
    annual_inflation: float = 0.0,
    contribution: float = 0.0,
    withdrawal: float = 0.0,
) -> pd.Series:
    """Calculate terminal, loss-tail, drawdown, and annualized metrics.

    With ``annual_inflation > 0`` all metrics are computed on inflation-adjusted
    wealth, so results are expressed in real (purchasing power) terms. The
    Sharpe ratio uses ``risk_free_rate`` as the annualized risk-free return.
    """

    if wealth.empty or wealth.shape[1] == 0:
        raise ValueError("wealth must contain at least one simulated path.")
    if not np.isfinite(initial_value) or initial_value <= 0:
        raise ValueError("initial_value must be positive and finite.")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1.")
    if not np.isfinite(periods_per_year) or periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive and finite.")
    if not np.isfinite(risk_free_rate):
        raise ValueError("risk_free_rate must be finite.")
    if not np.isfinite(annual_inflation) or annual_inflation < 0:
        raise ValueError("annual_inflation must be a finite, non-negative number.")
    if not np.isfinite(contribution) or contribution < 0:
        raise ValueError("contribution must be a finite, non-negative number.")
    if not np.isfinite(withdrawal) or withdrawal < 0:
        raise ValueError("withdrawal must be a finite, non-negative number.")
    try:
        wealth_values = wealth.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("wealth must contain numeric values.") from exc
    if not np.isfinite(wealth_values).all():
        raise ValueError("wealth must contain only finite values.")

    if annual_inflation > 0:
        period = np.arange(1, len(wealth) + 1, dtype=float)
        deflator = (1.0 + annual_inflation) ** (-period / periods_per_year)
        wealth = wealth.mul(deflator, axis=0)

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
    annualization = periods_per_year / len(wealth)
    mean_terminal = float(terminal.mean())
    annualized_return = (mean_terminal / initial_value) ** annualization - 1.0
    annualized_volatility = (float(terminal.std(ddof=0)) / initial_value) * np.sqrt(annualization)
    sharpe_ratio = (
        float((annualized_return - risk_free_rate) / annualized_volatility)
        if annualized_volatility > 0
        else 0.0
    )
    ulcer = np.sqrt((drawdown**2).mean(axis=0))

    period_returns = (wealth / wealth.shift(1) - 1.0).to_numpy(dtype=float).ravel()
    period_returns = period_returns[np.isfinite(period_returns)]
    downside = period_returns - risk_free_rate / periods_per_year
    downside_squared = np.where(downside < 0, downside**2, 0.0)
    downside_deviation = float(np.sqrt(downside_squared.mean())) if downside_squared.size else 0.0
    annualized_downside = downside_deviation * np.sqrt(periods_per_year)
    sortino_ratio = (
        float((annualized_return - risk_free_rate) / annualized_downside) if annualized_downside > 0 else 0.0
    )
    mean_max_drawdown = float(max_drawdown.mean())
    calmar_ratio = float(annualized_return / mean_max_drawdown) if mean_max_drawdown > 0 else 0.0
    geometric_annualized_return = float(np.exp(np.log(terminal / initial_value).mean() * annualization) - 1.0)
    skewness = terminal.skew()
    kurtosis = terminal.kurt()
    summary = {
        "mean": terminal.mean(),
        "std": terminal.std(ddof=0),
        "p05": terminal.quantile(0.05),
        "p50": terminal.quantile(0.50),
        "p95": terminal.quantile(0.95),
        "annualized_return": float(annualized_return),
        "annualized_volatility": float(annualized_volatility),
        "geometric_annualized_return": geometric_annualized_return,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "calmar_ratio": calmar_ratio,
        "probability_of_loss": float((terminal < initial_value).mean()),
        "var_95": initial_value - lower_tail,
        "expected_shortfall_95": initial_value - float(tail.mean()),
        "max_drawdown_mean": mean_max_drawdown,
        "max_drawdown_p95": float(max_drawdown.quantile(0.95)),
        "max_drawdown_worst": float(max_drawdown.max()),
        "ulcer_index_mean": float(ulcer.mean()),
        "ulcer_index_p95": float(ulcer.quantile(0.95)),
        "terminal_skewness": float(skewness) if np.isfinite(skewness) else 0.0,
        "terminal_kurtosis": float(kurtosis) if np.isfinite(kurtosis) else 0.0,
    }

    if contribution or withdrawal:
        period_count = len(wealth)
        period_index = np.arange(1, period_count + 1, dtype=float)
        deflator = (1.0 + annual_inflation) ** (-period_index / periods_per_year)
        real_contributions = contribution * deflator
        real_withdrawals = withdrawal * deflator
        previous = np.vstack([np.full(wealth.shape[1], initial_value), wealth_values[:-1]])
        previous *= np.concatenate(([1.0], deflator[:-1]))[:, None]
        current = wealth_values * deflator[:, None]
        denominator = previous + real_contributions[:, None]
        numerator = current + real_withdrawals[:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            period_returns_with_flows = numerator / denominator - 1.0
        period_returns_with_flows[(denominator <= 0) | (current <= 0)] = np.nan

        path_growth = np.full(wealth.shape[1], np.nan, dtype=float)
        for path in range(wealth.shape[1]):
            path_returns = period_returns_with_flows[:, path]
            if np.isfinite(path_returns).all() and np.all(1.0 + path_returns > 0):
                path_growth[path] = np.exp(np.log1p(path_returns).sum())
        valid_growth = path_growth[np.isfinite(path_growth)]
        valid_period_returns = period_returns_with_flows[np.isfinite(period_returns_with_flows)]
        if valid_growth.size:
            flow_adjusted_return = float(
                np.mean(np.power(valid_growth, periods_per_year / period_count) - 1.0)
            )
        else:
            flow_adjusted_return = np.nan
        flow_adjusted_volatility = (
            float(np.std(valid_period_returns, ddof=0) * np.sqrt(periods_per_year))
            if valid_period_returns.size
            else np.nan
        )
        mean_period_return = (
            float(np.mean(valid_period_returns)) if valid_period_returns.size else np.nan
        )
        flow_adjusted_sharpe = (
            float((mean_period_return * periods_per_year - risk_free_rate) / flow_adjusted_volatility)
            if np.isfinite(flow_adjusted_volatility) and flow_adjusted_volatility > 0
            else 0.0
        )
        summary.update(
            {
                "cash_flow_adjusted_annualized_return": flow_adjusted_return,
                "cash_flow_adjusted_volatility": flow_adjusted_volatility,
                "cash_flow_adjusted_sharpe_ratio": flow_adjusted_sharpe,
                "total_contributed": float(contribution * period_count),
                "total_withdrawn": float(withdrawal * period_count),
                "net_external_cash_flow": float((contribution - withdrawal) * period_count),
            }
        )

    return pd.Series(summary)
