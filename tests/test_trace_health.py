"""Descriptive raw-completion diagnostics and measured protocol compatibility."""
from copy import deepcopy
from dataclasses import replace

import pytest

from paper2.training.config import NATURAL_COMPLETION_PROTOCOL, TrainingConfig
from paper2.training.resources import full_pool_projection, profile_arrangement_mismatches, resource_plan
from paper2.training.trace_health import summarize_trace_health


def raw(world='a', *, seed=1, text='The box is here.\nAnswer: A', ids=None, **changes):
    return {'id': f'{world}-direct', 'world_id': world, 'format': 'native_reasoning',
            'arm': 'reasoning_rl', 'dataset': 'clean', 'model': 'fixture', 'revision': 'f' * 40,
            'step': 0, 'seed': seed, 'checkpoint_id': f'fixture-{seed}',
            'completion_protocol': NATURAL_COMPLETION_PROTOCOL,
            'completion': text, 'completion_ids': [1, 2, 3, 4] if ids is None else ids, **changes}


def panel(rows, **kwargs):
    return summarize_trace_health(rows, samples=200, **kwargs)['panels'][0]


def test_boundary_completion_is_not_a_cap_failure_and_has_no_editability_guess():
    rows = [raw('a', text='Answer: A.', ids=[7] * 384),
            raw('b', text='loop loop\nAnswer:', ids=[8] * 384)]
    result = panel(rows)
    assert result['cap_reached']['estimate'] == 1
    assert result['cap_failure']['estimate'] == .5
    assert result['answer_complete']['estimate'] == .5
    assert result['token_max_frequency_ratio']['estimate'] == 1
    assert result['token_distinct_ratio']['estimate'] == 1 / 384
    assert result['editability']['status'] == 'not_assessed'
    assert result['editability']['counts']['not_assessed'] == 2
    assert result['editability']['editable_among_editor_assessed']['estimate'] is None
    assert result['editability']['human_audit'] == 'not_evaluated_by_this_helper'


def test_health_averages_seed_means_within_each_story_and_keeps_arms_separate():
    # Two views in one seed do not give that seed extra weight.
    rows = [raw('a', seed=1, ids=[1, 1, 1, 1]), raw('a', seed=2, ids=[1, 2, 3, 4]),
            raw('b', seed=1, ids=[1, 1, 1, 1]), raw('b', seed=2, ids=[1, 2, 3, 4])]
    rows += [{**row, 'id': row['id'] + '-other'} for row in rows]
    result = panel(rows)
    assert result['token_distinct_ratio']['estimate'] == .625
    assert result['token_distinct_ratio']['ci95'] == [.625, .625]
    assert result['stories'] == 2 and result['training_seeds'] == [1, 2]
    multiple = summarize_trace_health(rows + [{**row, 'arm': 'reasoning_sft'} for row in rows], samples=50)
    assert {item['identity']['arm'] for item in multiple['panels']} == {'reasoning_rl', 'reasoning_sft'}


def test_edit_statuses_are_explicit_and_condition_on_assessed_records():
    rows = [raw('a'), raw('b'), raw('c')]
    edits = [{**rows[0], 'edit_status': 'editable'}, {**rows[1], 'edit_status': 'contradicted'}]
    result = panel(rows, editor_statuses=edits)['editability']
    assert result['status'] == 'partially_assessed'
    assert result['counts']['not_assessed'] == 1
    assert result['status_rates']['editable']['estimate'] == pytest.approx(1 / 3)
    assert result['editable_among_editor_assessed']['estimate'] == .5
    assert result['editor_assessed_records'] == 2
    with pytest.raises(ValueError, match='exactly one'):
        panel(rows, editor_statuses=edits + [edits[0]])
    with pytest.raises(ValueError, match='exactly one'):
        panel(rows, editor_statuses=[{**raw('other'), 'status': 'editable'}])


def test_exact_trace_commonness_ignores_only_complete_terminal_answer_and_preserves_whitespace():
    rows = [raw('a', text='same trace\nAnswer: A'), raw('b', text='same trace\nAnswer: B.'),
            raw('c', text='same trace \nAnswer: A')]
    result = panel(rows)['exact_common_trace']
    assert result['estimate'] == pytest.approx(2 / 3)
    assert result['distinct_exact_traces'] == 2
    assert result['observed_modal_trace_occurrences'] == 2
    assert result['ci95'] is not None
    assert rows[1]['completion'].endswith('B.')  # No mutation or rescoring.


def test_empty_or_immediate_only_checkpoint_is_explicitly_unassessed():
    result = summarize_trace_health([{**raw(), 'format': 'immediate'}])
    assert result['status'] == 'not_assessed' and result['panels'] == []
    assert result['bootstrap_samples'] == 20_000
    empty = panel([raw(text='', ids=[])])
    assert empty['token_distinct_ratio']['estimate'] is None
    assert empty['token_distinct_ratio']['observations'] == 0
    assert empty['cap_failure']['estimate'] == 0


@pytest.mark.parametrize('change,match', [
    ({'completion_protocol': 'legacy_forced_answer_v1'}, 'explicitly bind'),
    ({'forced_delimiter': True}, 'injected'),
    ({'completion_tokens': 5}, 'length'),
    ({'cap_failure': True}, 'metadata'),
])
def test_raw_health_integrity_and_legacy_reinterpretation_are_rejected(change, match):
    with pytest.raises(ValueError, match=match):
        panel([raw(**change)])


def test_duplicate_or_incomplete_seed_panel_is_not_silently_reweighted():
    with pytest.raises(ValueError, match='Duplicate'):
        panel([raw(), raw()])
    with pytest.raises(ValueError, match='same complete'):
        panel([raw('a', seed=1), raw('a', seed=2), raw('b', seed=2)])


def test_native_evaluation_field_aliases_preserve_raw_counting():
    row = raw(text='Answer: A.', ids=[1, 2, 3])
    row['completion_text'] = row.pop('completion')
    row['completion_generated_ids'] = row.pop('completion_ids')
    assert panel([row])['answer_complete']['estimate'] == 1


def test_actual_runner_checkpoint_adapter_digest_is_preserved():
    digest = 'a' * 64
    binding = panel([raw(adapter_sha256=digest)])['checkpoint_bindings'][0]
    assert binding['adapter_sha256'] == digest
    assert panel([raw(checkpoint_adapter_sha256=digest)])['checkpoint_bindings'][0]['adapter_sha256'] == digest


def test_revised_resource_budget_and_profile_identity_do_not_reuse_legacy_measurements():
    legacy = TrainingConfig()
    revised = replace(legacy, completion_protocol=NATURAL_COMPLETION_PROTOCOL, beta=.04)
    assert resource_plan(legacy, 2000, 120, 3)['evaluation_max_new_tokens_per_generation'] == 391
    plan = resource_plan(revised, 2000, 120, 3)
    assert plan['evaluation_max_new_tokens_per_generation'] == 384
    assert plan['completion_protocol'] == NATURAL_COMPLETION_PROTOCOL and plan['beta'] == .04
    historical = legacy.to_dict()
    historical.pop('beta')
    historical.pop('completion_protocol')
    assert profile_arrangement_mismatches(historical, legacy) == []
    assert profile_arrangement_mismatches(historical, revised) == ['beta', 'completion_protocol']
    assert profile_arrangement_mismatches(revised.to_dict(), revised) == []
    assert profile_arrangement_mismatches(revised.to_dict(), legacy) == ['beta', 'completion_protocol']


def test_resource_projection_requires_the_revised_measured_arrangement():
    config = TrainingConfig(stage='profile', max_steps=5,
                            completion_protocol=NATURAL_COMPLETION_PROTOCOL, beta=.04)
    with pytest.raises(ValueError, match='actual measured_config'):
        full_pool_projection(config, 2000, 120, 3, {})
    wrong = deepcopy(config.to_dict())
    wrong['beta'] = 0
    with pytest.raises(ValueError, match='measured arrangement'):
        full_pool_projection(config, 2000, 120, 3, {}, measured_config=wrong)
    result = full_pool_projection(config, 2000, 120, 3, {}, measured_config=config.to_dict())
    assert result['measured_arrangement']['beta'] == .04
    assert result['projected_total_seconds'] is None  # No invented measurements.
