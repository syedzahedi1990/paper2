"""Explicit completion provenance for inference and comparisons.

Missing protocol fields describe historical v1 files only. They can be read
together for reproduction, but never silently pooled with explicit v2 results.
"""
from .training.config import LEGACY_COMPLETION_PROTOCOL, NATURAL_COMPLETION_PROTOCOL

PROTOCOLS = (LEGACY_COMPLETION_PROTOCOL, NATURAL_COMPLETION_PROTOCOL)


def completion_protocol(rows):
    observed = set()
    missing = 0
    count = 0
    for row in rows:
        count += 1
        values = {row[key] for key in ('completion_protocol', 'source_completion_protocol')
                  if row.get(key) is not None}
        if not values:
            missing += 1
            values = {LEGACY_COMPLETION_PROTOCOL}
        if len(values) != 1 or not values <= set(PROTOCOLS):
            raise ValueError('Unknown or conflicting completion protocol in one result')
        observed.update(values)
    if len(observed) > 1:
        raise ValueError('Mixed completion protocols cannot be pooled or compared; analyze each protocol separately')
    return {'completion_protocol': next(iter(observed), None),
            'protocol_records': count, 'historical_records_without_protocol': missing,
            'missing_protocol_policy': 'Historical v1 only; never inferred as natural_answer_cap_v2'}
