import numpy as np
import pandas as pd

from mc_quadrants.return_comparison import build_return_comparison
from mc_quadrants.types import RegimeMoments, ScenarioModel


def model(volatility=.04):
    assets = ['STOCK']
    moments = RegimeMoments(
        pd.Series([.005], index=assets),
        pd.DataFrame([[volatility**2]], index=assets, columns=assets),
        pd.DataFrame([[1.]], index=assets, columns=assets), 100,
    )
    return ScenarioModel(['State'], pd.DataFrame([[1.]]), {'State': moments})


def test_comparison_has_common_bins_and_full_sample_tail_probabilities():
    result = build_return_comparison(model(), samples=32768)
    row = result['rows'][0]
    assert result == build_return_comparison(model(), samples=32768)
    for name in ['mnts', 'student_t']:
        law = row[name]
        assert len(law['density']) == len(row['x']) == 100
        assert np.all(np.diff(law['cdf']) >= 0)
        assert 0 <= law['cdf'][0] <= law['cdf'][-1] <= 100
        width = row['x'][1] - row['x'][0]
        assert np.isclose(sum(law['density']) * width, 1 - law['outside_range'] / 100)
        assert law['quantiles'] == sorted(law['quantiles'])
    # Variance-matched t(5): its 5% quantile is roughly mean - 1.56 sigma.
    assert abs(row['student_t']['quantiles'][1] - (.5 - 1.56 * 4)) < .25


def test_zero_volatility_produces_finite_point_mass():
    row = build_return_comparison(model(0), samples=128)['rows'][0]
    for name in ['mnts', 'student_t']:
        assert np.isfinite(row[name]['density']).all()
        np.testing.assert_allclose(row[name]['quantiles'], [.5] * 5)
        assert row[name]['loss_probability'] == 0
