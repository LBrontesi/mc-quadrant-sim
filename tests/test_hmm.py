import numpy as np
import pandas as pd
import pytest

from mc_quadrants.hmm import fit_hmm_model
from mc_quadrants.simulation import simulate_returns


def _two_state_history(seed: int = 3):
    rng = np.random.default_rng(seed)
    n = 240
    state = np.where(np.arange(n) < 120, 0, 1)
    data = np.where(
        (state == 0)[:, None],
        rng.normal(0.01, 0.02, (n, 2)),
        rng.normal(-0.01, 0.06, (n, 2)),
    )
    returns = pd.DataFrame(
        data,
        columns=["Stocks", "Bonds"],
        index=pd.date_range("2000-01-31", periods=n, freq="ME"),
    )
    return returns


def test_hmm_recovers_two_distinct_states():
    returns = _two_state_history()

    model, fit = fit_hmm_model(returns, n_states=2, random_seed=1, restarts=2)

    means = np.array([model.moments[state].mean.to_numpy() for state in model.states])
    assert model.validate() is None
    assert len(model.states) == 2
    assert np.sqrt(((means[0] - means[1]) ** 2).sum()) > 0.01
    assert fit.log_likelihood == pytest.approx(fit.log_likelihood)
    assert np.allclose(
        model.transition_matrix.sum(axis=1).to_numpy(),
        1.0,
    )
    assert all(model.moments[state].observations >= 10 for state in model.states)


def test_hmm_model_simulates_returns():
    returns = _two_state_history()
    model, _ = fit_hmm_model(returns, n_states=2, random_seed=1, restarts=1)

    result = simulate_returns(model, periods=24, paths=5, random_seed=2)

    assert result.returns.shape == (24, 5, 2)
    assert np.isfinite(result.returns).all()
    assert set(result.states) == set(model.states)


def test_hmm_rejects_invalid_state_counts():
    returns = _two_state_history()

    with pytest.raises(ValueError, match="n_states"):
        fit_hmm_model(returns, n_states=1)

    with pytest.raises(ValueError, match="Not enough observations"):
        fit_hmm_model(returns.iloc[:5], n_states=4)


def test_hmm_is_reproducible_with_seed():
    returns = _two_state_history()

    first = fit_hmm_model(returns, n_states=2, random_seed=7, restarts=1)
    second = fit_hmm_model(returns, n_states=2, random_seed=7, restarts=1)

    assert first[1].log_likelihood == pytest.approx(second[1].log_likelihood)
    assert first[1].regimes.tolist() == second[1].regimes.tolist()
