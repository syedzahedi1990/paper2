"""Check finite-population audit coverage by enumerating actual random samples."""
from itertools import combinations

import pytest

from paper2.edit_generation import exact_error_rate_upper


@pytest.mark.parametrize("population,sample_size", [(7, 3), (10, 5), (12, 8), (8, 8)])
def test_exact_bound_covers_true_defect_rate_for_every_fixed_defective_set_size(population, sample_size):
    # Every equally likely sample is enumerated; no CDF-inversion implementation is reused.
    samples = list(combinations(range(population), sample_size))
    for defective_count in range(population + 1):
        covered = 0
        for selected in samples:
            observed = sum(index < defective_count for index in selected)
            upper = exact_error_rate_upper(population, sample_size, observed, confidence=.95)
            covered += upper >= defective_count / population
        assert covered / len(samples) >= .95


def test_default_grid_allows_one_original_flaw_but_cannot_erase_a_second():
    assert exact_error_rate_upper(3000, 100, 1) <= .05
    assert exact_error_rate_upper(3000, 100, 2) > .05
