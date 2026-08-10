import numpy as np
import pandas as pd
import pytest

from mc_quadrants.validation import walk_forward_validation


def _sample_history():
    rng = np.random.default_rng(9)
    dates = pd.date_range("1990-01-31", periods=240, freq="ME")
    growth = rng.normal(2.0, 1.5, len(dates))
    inflation = rng.normal(3.0, 1.0, len(dates))
    macro = pd.DataFrame({"growth": growth, "inflation": inflation}, index=dates)
    shock = rng.normal(0.0, 0.05, len(dates))
    returns = pd.DataFrame(
        {
            "Stocks": 0.008 + 0.5 * shock + rng.normal(0, 0.02, len(dates)),
            "Bonds": 0.002 + rng.normal(0, 0.015, len(dates)),
        },
        index=dates,
    )
    return returns, macro


def test_walk_forward_validation_reports_predictive_metrics():
    returns, macro = _sample_history()

    result = walk_forward_validation(
        returns,
        macro,
        growth_col="growth",
        inflation_col="inflation",
        growth_threshold="median",
        inflation_threshold="median",
        min_train_periods=60,
        step=10,
    )

    assert result.summary["splits"] >= 10
    assert {
        "regime_log_likelihood_mean",
        "regime_student_t_log_likelihood_mean",
        "unconditional_log_likelihood_mean",
        "advantage_mean",
        "regime_hit_rate",
        "transition_brier_score",
        "transition_log_score_mean",
        "actual_switches_per_decade",
        "predicted_switches_per_decade",
        "duration_log_score_mean",
        "rolling_vintage_expected_duration_std",
    }.issubset(result.summary.index)
    assert "advantage_vs_student_t" in result.splits
    assert "switch_brier_score" in result.splits
    assert "completed_duration" in result.splits
    assert "vintage_expected_duration" in result.splits
    assert (result.splits["advantage"] > 0).any()
    assert (result.splits["regime_hit"] <= 1).all()
    assert (result.splits["regime_hit"] >= 0).all()


def test_walk_forward_validation_rejects_short_history():
    returns, macro = _sample_history()

    with pytest.raises(ValueError, match="at least"):
        walk_forward_validation(
            returns.iloc[:20],
            macro.iloc[:20],
            growth_col="growth",
            inflation_col="inflation",
            growth_threshold="median",
            inflation_threshold="median",
            min_train_periods=60,
        )


def test_walk_forward_uses_causal_thresholds():
    returns, macro = _sample_history()

    with_window = walk_forward_validation(
        returns,
        macro,
        growth_col="growth",
        inflation_col="inflation",
        growth_threshold="median",
        inflation_threshold="median",
        min_train_periods=60,
        step=20,
        threshold_window=12,
    )

    assert with_window.summary["splits"] > 0
    assert np.isfinite(with_window.summary["advantage_mean"])


def test_walk_forward_default_does_not_use_future_thresholds():
    returns, macro = _sample_history()
    baseline = walk_forward_validation(
        returns,
        macro,
        growth_col="growth",
        inflation_col="inflation",
        growth_threshold="median",
        inflation_threshold="median",
        min_train_periods=60,
        step=20,
    )

    changed_future = macro.copy()
    changed_future.iloc[-1, changed_future.columns.get_loc("growth")] += 100.0
    changed = walk_forward_validation(
        returns,
        changed_future,
        growth_col="growth",
        inflation_col="inflation",
        growth_threshold="median",
        inflation_threshold="median",
        min_train_periods=60,
        step=20,
    )

    assert np.allclose(
        baseline.splits["advantage"].iloc[:2].to_numpy(),
        changed.splits["advantage"].iloc[:2].to_numpy(),
    )


def test_walk_forward_allows_return_dates_without_exact_macro_matches():
    returns, macro = _sample_history()
    shifted_returns = returns.copy()
    shifted_index = shifted_returns.index[:-1].append(pd.DatetimeIndex([pd.Timestamp("2025-10-31")]))
    shifted_returns.index = shifted_index

    result = walk_forward_validation(
        shifted_returns,
        macro,
        growth_col="growth",
        inflation_col="inflation",
        growth_threshold="median",
        inflation_threshold="median",
        min_train_periods=60,
        step=20,
    )

    assert result.summary["splits"] > 0


def test_walk_forward_tail_validation_uses_selected_portfolio_weights():
    returns, macro = _sample_history()
    result = walk_forward_validation(
        returns,
        macro,
        growth_col="growth",
        inflation_col="inflation",
        growth_threshold="median",
        inflation_threshold="median",
        min_train_periods=60,
        step=60,
        weights={"Stocks": 1.0, "Bonds": 0.0},
    )

    first_date = pd.Timestamp(result.splits.iloc[0]["date"])
    assert result.splits.iloc[0]["portfolio_return"] == pytest.approx(
        returns.loc[first_date, "Stocks"]
    )


def test_walk_forward_rejects_zero_portfolio_weights():
    returns, macro = _sample_history()

    with pytest.raises(ValueError, match="non-zero sum"):
        walk_forward_validation(
            returns,
            macro,
            growth_col="growth",
            inflation_col="inflation",
            growth_threshold="median",
            inflation_threshold="median",
            min_train_periods=60,
            weights={"Stocks": 0.0, "Bonds": 0.0},
        )
