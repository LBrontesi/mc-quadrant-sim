from mc_quadrants.demo import _demo_history


def test_demo_history_covers_extended_asset_universe():
    macro, returns = _demo_history(seed=7)

    assert macro.index[0].year == 1990
    assert macro.index[-1].year == 2024
    assert len(macro) == 420
    assert len(returns.columns) == 8
    assert returns.index.equals(macro.index)
