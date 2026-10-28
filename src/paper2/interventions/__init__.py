"""Auditable, within-checkpoint critical-token component interventions (Test 3)."""

from .components import (
    Capture,
    ComponentInterventionEngine,
    EncodedInput,
    InterventionResult,
    Site,
    encode_input,
    signed_permutation_delta,
)
from .scoring import compare_checkpoints, score_consequence_pattern, component_summary
from .runner import run_component_test

__all__ = [
    "Capture", "ComponentInterventionEngine", "EncodedInput", "InterventionResult",
    "Site", "encode_input", "signed_permutation_delta", "score_consequence_pattern",
    "compare_checkpoints", "component_summary",
    "run_component_test",
]
