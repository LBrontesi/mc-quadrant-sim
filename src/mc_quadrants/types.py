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
        first_state = self.states[0]
        return list(self.moments[first_state].mean.index)

    def validate(self) -> None:
        missing_states = set(self.states).difference(self.transition_matrix.index)
        if missing_states:
            raise ValueError(f"Transition matrix missing rows: {sorted(missing_states)}")

        row_sums = self.transition_matrix.loc[self.states, self.states].sum(axis=1)
        if not np.allclose(row_sums.to_numpy(dtype=float), 1.0):
            raise ValueError("Transition matrix rows must sum to 1.")

        missing_moments = set(self.states).difference(self.moments)
        if missing_moments:
            raise ValueError(f"Missing regime moments: {sorted(missing_moments)}")


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
