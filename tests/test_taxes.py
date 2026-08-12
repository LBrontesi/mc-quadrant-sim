import numpy as np

from mc_quadrants.taxes import ITALY_LOSS_CARRY_YEARS, _advance_tax_year


def test_italian_losses_expire_after_four_subsequent_tax_years():
    buckets = np.zeros((ITALY_LOSS_CARRY_YEARS + 1, 1), dtype=float)
    buckets[-1, 0] = 10.0

    for _ in range(ITALY_LOSS_CARRY_YEARS):
        _advance_tax_year(buckets)
        assert buckets.sum() == 10.0

    _advance_tax_year(buckets)

    assert buckets.sum() == 0.0
