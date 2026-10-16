import numpy as np
import pandas as pd
import pytest

from scripts.investigate_allocation_realism import (
    nonoverlapping_count,
    paired_block_returns,
    portfolio_statistics,
)


def test_statistics_use_simple_returns_and_count_unrecovered_spells():
    logs = np.log(np.array([[2.0, 0.5], [0.5, 2.0], [0.5, 0.5], [2.0, 2.0]]))[:, None, :]
    mixed = portfolio_statistics(logs, 0.5)
    assert mixed["cagr"][0] == pytest.approx((1.25**2) ** 3 - 1)
    assert mixed["max_drawdown"][0] == pytest.approx(0.5)
    assert mixed["max_underwater_months"][0] == 1
    assert mixed["underwater_at_end"][0] == 0
    unrecovered = portfolio_statistics(logs[:3], 1.0)
    assert unrecovered["max_drawdown"][0] == pytest.approx(0.75)
    assert unrecovered["max_underwater_months"][0] == 2
    assert unrecovered["underwater_at_end"][0] == 1


def test_block_benchmark_preserves_joint_observations_and_seed():
    values = np.array([[0.1, -0.1], [0.2, -0.2], [0.3, -0.3]])
    first = paired_block_returns(values, 24, 16, 42)
    np.testing.assert_array_equal(first, paired_block_returns(values, 24, 16, 42))
    np.testing.assert_allclose(first[:, :, 0], -first[:, :, 1])
    assert set(first[:, :, 0].ravel()) == {0.1, 0.2, 0.3}


def test_overlapping_decades_do_not_count_as_independent_evidence():
    origins = pd.date_range("2000-01-31", periods=60, freq="ME")
    assert nonoverlapping_count(origins, 120) == 1
    assert nonoverlapping_count(origins, 12) == 5
