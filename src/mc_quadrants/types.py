from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RegimeMoments:
    """Expected returns and covariance estimates for one macro regime."""

    mean: pd.Series
    covariance: pd.DataFrame
    correlation: pd.DataFrame
    observations: int


@dataclass(frozen=True)
class ScenarioModel:
    """Complete calibrated model used by the simulator."""

    states: list[str]
    transition_matrix: pd.DataFrame
    moments: dict[str, RegimeMoments]
    frequency: str = "M"
    metadata: dict[str, Any] = field(default_factory=dict)
    historical_returns: dict[str, pd.DataFrame] = field(default_factory=dict)

    @property
    def assets(self) -> list[str]:
        if not self.states:
            raise ValueError("ScenarioModel must define at least one state.")
        first_state = self.states[0]
        return list(self.moments[first_state].mean.index)

    def validate(self) -> None:
        if not self.states:
            raise ValueError("ScenarioModel must define at least one state.")
        if len(set(self.states)) != len(self.states):
            raise ValueError("ScenarioModel states must be unique.")
        if self.transition_matrix.index.has_duplicates or self.transition_matrix.columns.has_duplicates:
            raise ValueError("Transition matrix labels must be unique.")

        missing_states = set(self.states).difference(self.transition_matrix.index)
        if missing_states:
            raise ValueError(f"Transition matrix missing rows: {sorted(missing_states)}")
        missing_columns = set(self.states).difference(self.transition_matrix.columns)
        if missing_columns:
            raise ValueError(f"Transition matrix missing columns: {sorted(missing_columns)}")

        transition = self.transition_matrix.loc[self.states, self.states]
        values = transition.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Transition matrix must contain only finite values.")
        if (values < 0).any():
            raise ValueError("Transition matrix probabilities cannot be negative.")

        row_sums = transition.sum(axis=1)
        if not np.allclose(row_sums.to_numpy(dtype=float), 1.0):
            raise ValueError("Transition matrix rows must sum to 1.")

        missing_moments = set(self.states).difference(self.moments)
        if missing_moments:
            raise ValueError(f"Missing regime moments: {sorted(missing_moments)}")

        assets = list(self.moments[self.states[0]].mean.index)
        if not assets:
            raise ValueError("Regime moments must define at least one asset.")
        for state in self.states:
            moments = self.moments[state]
            if list(moments.mean.index) != assets:
                raise ValueError(f"Regime mean assets do not match for state: {state}")
            if not moments.covariance.index.equals(moments.covariance.columns):
                raise ValueError(f"Covariance labels must match for state: {state}")
            if list(moments.covariance.index) != assets:
                raise ValueError(f"Covariance assets do not match for state: {state}")
            if not moments.correlation.index.equals(moments.correlation.columns):
                raise ValueError(f"Correlation labels must match for state: {state}")
            if list(moments.correlation.index) != assets:
                raise ValueError(f"Correlation assets do not match for state: {state}")
            if not np.isfinite(moments.mean.to_numpy(dtype=float)).all():
                raise ValueError(f"Regime mean contains non-finite values for state: {state}")
            if not np.isfinite(moments.covariance.to_numpy(dtype=float)).all():
                raise ValueError(f"Covariance contains non-finite values for state: {state}")
            if not np.isfinite(moments.correlation.to_numpy(dtype=float)).all():
                raise ValueError(f"Correlation contains non-finite values for state: {state}")


@dataclass(frozen=True)
class SimulationResult:
    """Raw simulated regime and asset-return paths."""

    returns: np.ndarray
    regimes: np.ndarray
    assets: list[str]
    states: list[str]
    frequency: str
    distribution: str = "normal"
    degrees_of_freedom: float | None = None
    transition_concentration: float | None = None

    def returns_frame(self) -> pd.DataFrame:
        periods, paths, _ = self.returns.shape
        index = pd.MultiIndex.from_product(
            [range(periods), range(paths)],
            names=["period", "path"],
        )
        flattened = self.returns.reshape(periods * paths, len(self.assets))
        return pd.DataFrame(flattened, index=index, columns=self.assets)

    def regimes_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.regimes, columns=[f"path_{i}" for i in range(self.regimes.shape[1])])
