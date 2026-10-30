"""Deterministic native-outcome selection, failures, tampering and lineage."""
import json
import shutil

import pytest

from paper2.io import file_digest, write_json, read_jsonl, write_jsonl
from paper2.training.selection_results import (compute_panel_selection, freeze_panel_selection,
                                               verify_panel_selection)
from panel_fixture import make_panel_fixture, refresh_candidate


@pytest.fixture(autouse=True)
def remove_this_tests_full_panel(tmp_path):
    # These CPU fixtures contain full scheduled panels. Retaining dozens after
    # the test consumes gigabytes of operator disk without adding evidence.
    yield
    shutil.rmtree(tmp_path / 'panel', ignore_errors=True)


def run_selection(fixture, output=None):
    args = (fixture['binding'], file_digest(fixture['binding']), fixture['status'],
            fixture['policy'], file_digest(fixture['policy']))
    attribution = {'reviewed_by': 'CPU fixture, not an approval', 'rationale': 'Deterministic integrity test'}
    return freeze_panel_selection(*args, output, **attribution) if output else compute_panel_selection(*args, **attribution)


@pytest.fixture
def panel(tmp_path):
    return make_panel_fixture(tmp_path / 'panel')


def test_actual_final_native_family_objective_not_best_checkpoint_or_immediate(tmp_path):
    scores = {'reasoning_rl_clean_seed1_lr-x1_constant': 120, 'answer_rl_clean_seed1_lr-x1_constant': 0,
              'reasoning_rl_clean_seed1_lr-x3_constant': 90, 'answer_rl_clean_seed1_lr-x3_constant': 90,
              'label_sft_clean_seed1_lr-x10_constant': 100}
    fixture = make_panel_fixture(tmp_path / 'panel', scores=scores, empty_reasoning=True)
    output = tmp_path / 'selection.json'
    result = run_selection(fixture, output)
    assert result['selected_settings']['rl']['learning_rate'] == 6e-5
    assert result['selected_settings']['sft']['learning_rate'] == 1e-5
    assert result['selected_reasoning_rl_candidate'] == 'reasoning_rl_clean_seed1_lr-x3_constant'
    assert result['selected_answer_rl_candidate'] == 'answer_rl_clean_seed1_lr-x3_constant'
    assert verify_panel_selection(output, file_digest(output)) == result
    assert result['scientific_gate_decision'] is None and result['confirmation_authorized'] is False
    assert len(result['selected_run_evidence']) == 3
    with pytest.raises(FileExistsError):
        run_selection(fixture, output)


def test_empty_trace_winner_is_not_replaced_by_more_readable_candidate(tmp_path):
    fixture = make_panel_fixture(tmp_path / 'panel', scores={
        'reasoning_rl_clean_seed1_lr-x30_constant': 119,
        'answer_rl_clean_seed1_lr-x30_constant': 119}, empty_reasoning=True)
    assert run_selection(fixture)['selected_settings']['rl']['learning_rate'] == .0006


def test_exact_ties_take_smaller_rate_and_preserve_every_candidate(panel):
    result = run_selection(panel)
    assert len(result['candidates']) == 12
    assert result['selected_settings']['rl']['learning_rate'] == 2e-5
    assert result['selected_settings']['sft']['learning_rate'] == 1e-6


def test_failed_candidate_excludes_entire_rl_pair_without_fabricated_zero(tmp_path):
    failed = 'answer_rl_clean_seed1_lr-x1_constant'
    fixture = make_panel_fixture(tmp_path / 'panel', failures=(failed,))
    result = run_selection(fixture)
    assert result['failures'] == [failed] and result['warnings']
    assert result['selected_settings']['rl']['learning_rate'] == 6e-5
    excluded, = result['excluded_settings']['rl']
    assert excluded['score'] is excluded['correct'] is excluded['n'] is None
    assert set(excluded['candidates']) == {failed, 'reasoning_rl_clean_seed1_lr-x1_constant'}
    failure = next(row for row in result['candidates'] if row['candidate'] == failed)
    assert failure['failure']['no_score_assigned'] and failure['launch_log']['sha256']


@pytest.mark.parametrize('family', ['answer_rl', 'label_sft'])
def test_no_eligible_family_produces_no_selection(tmp_path, family):
    failed = [f'{family}_clean_seed1_lr-x{rate}_constant' for rate in (1, 3, 10, 30)]
    fixture = make_panel_fixture(tmp_path / 'panel', failures=failed)
    with pytest.raises(ValueError, match='No eligible complete setting'):
        run_selection(fixture, tmp_path / 'no-receipt.json')
    assert not (tmp_path / 'no-receipt.json').exists()


@pytest.mark.parametrize('status', ['not_started', 'running', 'stopping'])
def test_fail_fast_unattempted_or_running_candidates_never_count_as_complete(panel, status):
    state = json.loads(panel['status'].read_text())
    state['status'] = 'failed'
    state['candidates']['label_sft_clean_seed1_lr-x30_constant']['status'] = status
    write_json(panel['status'], state)
    with pytest.raises(ValueError, match='terminal attempts'):
        run_selection(panel)


def test_failure_rule_cannot_be_added_after_the_failure(tmp_path):
    fixture = make_panel_fixture(tmp_path / 'panel', failures=('label_sft_clean_seed1_lr-x30_constant',))
    policy = json.loads(fixture['policy'].read_text())
    policy['frozen_utc'] = '2026-10-30T03:00:00+00:00'
    write_json(fixture['policy'], policy)
    with pytest.raises(ValueError, match='precede the observed failure'):
        run_selection(fixture)


@pytest.mark.parametrize('role', ['binding', 'registry', 'decision', 'corpus', 'training', 'development', 'environment', 'frozen'])
def test_changed_frozen_source_or_corpus_cannot_be_relabelled(panel, role):
    path = panel[role]
    if role == 'frozen':
        value = json.loads(path.read_text())
        value['wrapper_sha256'] = 'f' * 64
        write_json(path, value)
    else:
        path.write_text(path.read_text() + '\n')
    with pytest.raises(ValueError):
        run_selection(panel)


@pytest.mark.parametrize('mutation', ['native_correct', 'story', 'missing', 'duplicate', 'format', 'protocol', 'completion',
                                    'metric', 'exposure', 'source', 'adapter', 'terminal', 'retry'])
def test_rebound_invalid_measurement_or_attempt_cannot_select(panel, mutation):
    name = 'reasoning_rl_clean_seed1_lr-x1_constant'
    run = panel['root'] / 'runs' / name
    if mutation in ('native_correct', 'story', 'missing', 'duplicate', 'format', 'protocol', 'completion'):
        path = run / 'checkpoint_predictions.jsonl'
        rows = read_jsonl(path)
        index = next(i for i, row in enumerate(rows) if row['step'] == 2000 and row['format'] == 'native_reasoning')
        if mutation == 'missing':
            rows.pop(index)
        elif mutation == 'duplicate':
            rows[index] = rows[index + 1]
        else:
            field, value = {'native_correct': ('correct', False), 'story': ('world_id', 'foreign'),
                            'format': ('format', 'immediate'), 'protocol': ('completion_protocol', 'legacy_forced_answer_v1'),
                            'completion': ('completion', 'know ' * 384)}[mutation]
            rows[index][field] = value
        write_jsonl(path, rows)
    elif mutation == 'metric':
        path = run / 'checkpoint_evaluations.jsonl'
        rows = read_jsonl(path)
        rows[-1]['accuracy'] = .999
        write_jsonl(path, rows)
    elif mutation == 'exposure':
        path = run / 'compute_report.json'
        value = json.loads(path.read_text())
        value['prompt_exposures'] = 1999
        write_json(path, value)
    elif mutation == 'source':
        path = run / 'run_manifest.json'
        value = json.loads(path.read_text())
        value['source_identity'] = {'commit': 'f' * 40}
        write_json(path, value)
    elif mutation == 'adapter':
        (run / 'evaluated_checkpoints/checkpoint-2000/adapter_model.safetensors').write_bytes(b'changed')
    elif mutation == 'retry':
        events = read_jsonl(panel['events'])
        events.append({'event': 'candidate_started', 'candidate': name})
        write_jsonl(panel['events'], events)
    else:
        state = json.loads(panel['status'].read_text())
        state['candidates'][name]['exit_code'] = 1
        write_json(panel['status'], state)
    refresh_candidate(panel, name)
    with pytest.raises(ValueError):
        run_selection(panel)


def test_hand_written_winner_or_rehashed_receipt_is_not_authority(panel, tmp_path):
    output = tmp_path / 'selection.json'
    result = run_selection(panel, output)
    result['selected_reasoning_rl_candidate'] = 'reasoning_rl_clean_seed1_lr-x30_constant'
    write_json(output, result)
    with pytest.raises(ValueError, match='recomputed actual panel'):
        verify_panel_selection(output, file_digest(output))


def test_real_selection_derives_confirmation_and_gate_rechecks_every_family(panel, tmp_path):
    from paper2.training.confirmatory_selection import (export_selected_confirmatory_configs,
                                                        verify_confirmatory_selection)
    output = tmp_path / 'selection.json'
    result = run_selection(panel, output)
    exported = export_selected_confirmatory_configs(output, tmp_path / 'confirmation',
                                                    selection_sha256=file_digest(output))
    binding = exported['development_selection']
    review = {'calibration': {'manifest_sha256': result['corpus_manifest_sha256']},
              'development_selection': binding, 'gates': {'G1': {'evidence': [binding],
              'learning_curves': [{'arm': arm, **item} for arm, item in result['selected_run_evidence'].items()]}}}
    assert len(exported['configs']) == 24
    for arm in ('reasoning_rl', 'answer_rl', 'label_sft', 'reasoning_sft'):
        entry = next(row for row in exported['configs'] if row['arm'] == arm)
        config = json.loads((tmp_path / 'confirmation' / entry['path']).read_text())
        verified = verify_confirmatory_selection(config, review, tmp_path)
        assert verified['family'] == ('rl' if arm.endswith('_rl') else 'sft')
        with pytest.raises(ValueError, match='settings differ'):
            verify_confirmatory_selection({**config, 'learning_rate': config['learning_rate'] * 30}, review, tmp_path)
