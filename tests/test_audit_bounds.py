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


@pytest.mark.parametrize("population,sample_size", [(7, 3), (10, 5), (8, 8)])
def test_two_sided_flaw_interval_covers_every_fixed_population(population, sample_size):
    from paper2.edit_generation import _audit_rate
    samples = list(combinations(range(population), sample_size))
    for defective_count in range(population + 1):
        covered = 0
        for selected in samples:
            result = _audit_rate(set(range(population)), set(selected), set(range(defective_count)), .95)
            lower, upper = result["ci95"]
            assert lower <= upper
            covered += lower <= defective_count / population <= upper
        assert covered / len(samples) >= .95
