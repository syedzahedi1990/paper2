"""Revised G1 provenance and profile compatibility are integrity-only checks."""
import json
from dataclasses import replace

import pytest

from paper2.gates import _g1_editor_measurements, verify_g1_learning_curves
from paper2.io import file_digest, read_jsonl, write_json, write_jsonl
from paper2.training.config import NATURAL_COMPLETION_PROTOCOL, TrainingConfig
from paper2.training.model_plan import assert_profile_identity, calibrated_model_plan
from paper2.training.trace_health import summarize_trace_health
from test_g1_curves import g1_panel, rebind  # noqa: F401


@pytest.fixture
def revised_g1(request):
    gate, root = request.getfixturevalue('g1_panel')
    gate['status'] = 'pending'
    for arm in ('reasoning_rl', 'answer_rl', 'label_sft'):
        output = root / arm
        path = output / 'run_manifest.json'
        manifest = json.loads(path.read_text())
        manifest['config']['completion_protocol'] = NATURAL_COMPLETION_PROTOCOL
        manifest['config']['beta'] = .04 if arm.endswith('_rl') else 0
        write_json(path, manifest)
        metrics = [{**row, 'completion_protocol': NATURAL_COMPLETION_PROTOCOL}
                   for row in read_jsonl(output / 'checkpoint_evaluations.jsonl')]
        predictions = []
        for row in read_jsonl(output / 'checkpoint_predictions.jsonl'):
            value = {**row, 'completion_protocol': NATURAL_COMPLETION_PROTOCOL}
            if row['format'] == 'native_reasoning':
                value.update({'completion': 'Answer: A.' if row['correct'] else 'loop loop',
                              'completion_ids': [1, 2, 3, 4] if row['correct'] else [9] * 384})
            predictions.append(value)
        write_jsonl(output / 'checkpoint_evaluations.jsonl', metrics)
        write_jsonl(output / 'checkpoint_predictions.jsonl', predictions)
        health = []
        for step in sorted({row['step'] for row in metrics}):
            identity = next(row for row in metrics if row['step'] == step)
            observed = [row for row in predictions if row['step'] == step and row['format'] == 'native_reasoning']
            health.append({**identity, 'native_reasoning_assessed': bool(observed),
                           'trace_health': summarize_trace_health(observed),
                           'editability': {'status': 'not_assessed'}})
        write_jsonl(output / 'checkpoint_trace_health.jsonl', health)
        receipt_path = output / 'g1_learning_evidence.json'
        receipt = json.loads(receipt_path.read_text())
        for key in ('baseline', 'latest', 'immediate_latest'):
            receipt[key]['completion_protocol'] = NATURAL_COMPLETION_PROTOCOL
        write_json(receipt_path, receipt)
        rebind(gate, root, arm)
    return gate, root


def test_revised_g1_reports_unfavorable_health_and_unassessed_editability_without_pass_marks(revised_g1):
    gate, root = revised_g1
    result = verify_g1_learning_curves(gate, root, samples=100)
    assert set(result) == {'reasoning_rl', 'answer_rl', 'label_sft'}
    for record in result['reasoning_rl']['trace_health']:
        panel = record['trace_health']['panels'][0]
        assert panel['cap_failure']['estimate'] == .5
        assert panel['editability']['status'] == 'not_assessed'
    assert all(row['trace_health']['status'] == 'not_assessed' for row in result['label_sft']['trace_health'])
    assert result['reasoning_rl']['trace_health_artifact']['sha256']
    assert result['reasoning_rl']['editor_editability']['status'] == 'not_assessed'


def test_revised_g1_pass_cannot_treat_missing_editor_measurement_as_assessed(revised_g1):
    gate, root = revised_g1
    gate['status'] = 'pass'
    with pytest.raises(ValueError, match='actual fixed-editor editability evidence'):
        verify_g1_learning_curves(gate, root, samples=20)


def test_trace_health_artifact_hash_is_required(revised_g1):
    gate, root = revised_g1
    path = root / 'reasoning_rl' / 'checkpoint_trace_health.jsonl'
    path.write_text(path.read_text() + '\n')
    with pytest.raises(ValueError, match='trace_health artifact missing or modified'):
        verify_g1_learning_curves(gate, root, samples=20)


@pytest.mark.parametrize('change,match', [
    ('missing', 'every scheduled checkpoint'), ('flag', 'assessment flag'),
    ('binding', 'checkpoint binding'), ('summary', 'summary disagrees'),
    ('invent_editability', 'bind later editor outcomes separately'),
])
def test_rehashed_health_still_requires_complete_bound_truthful_measurements(revised_g1, change, match):
    gate, root = revised_g1
    path = root / 'reasoning_rl' / 'checkpoint_trace_health.jsonl'
    rows = read_jsonl(path)
    if change == 'missing':
        rows.pop()
    elif change == 'flag':
        rows[0]['native_reasoning_assessed'] = False
    elif change == 'binding':
        rows[0]['adapter_sha256'] = 'f' * 64
    elif change == 'summary':
        rows[0]['trace_health']['panels'][0]['cap_failure']['estimate'] = 0
    else:
        rows[0]['editability']['status'] = 'assessed'
    write_jsonl(path, rows)
    rebind(gate, root, 'reasoning_rl')
    with pytest.raises(ValueError, match=match):
        verify_g1_learning_curves(gate, root, samples=20)


def test_g1_cannot_mix_old_and_revised_runs(revised_g1):
    gate, root = revised_g1
    path = root / 'answer_rl' / 'run_manifest.json'
    manifest = json.loads(path.read_text())
    manifest['config']['completion_protocol'] = 'legacy_forced_answer_v1'
    manifest['config']['beta'] = 0
    write_json(path, manifest)
    rebind(gate, root, 'answer_rl')
    with pytest.raises(ValueError, match='common completion protocol'):
        verify_g1_learning_curves(gate, root, samples=20)


def test_model_plan_requires_revised_beta_and_protocol_in_nonbootstrap_profile():
    plan = calibrated_model_plan({}, '.')
    legacy = TrainingConfig()
    revised = replace(legacy, completion_protocol=NATURAL_COMPLETION_PROTOCOL, beta=.04)
    assert_profile_identity(revised.to_dict(), {'config': revised.to_dict()}, plan)
    for field, value in [('beta', 0), ('completion_protocol', 'legacy_forced_answer_v1')]:
        measured = {**revised.to_dict(), field: value}
        with pytest.raises(ValueError, match='measured execution arrangement'):
            assert_profile_identity(revised.to_dict(), {'config': measured}, plan)
    old = legacy.to_dict()
    old.pop('beta')
    old.pop('completion_protocol')
    assert_profile_identity(legacy.to_dict(), {'config': old}, plan)
    with pytest.raises(ValueError, match='measured execution arrangement'):
        assert_profile_identity(revised.to_dict(), {'config': old}, plan)


@pytest.fixture
def classified_checkpoint(tmp_path):
    from paper2.edit_generation import generate_edit_controls
    from test_generated_edits import Editor, fixture
    traces, manifest, plan = fixture(worlds=2, sources=1)
    editor = Editor(mutate=lambda _: {'status': 'not_editable', 'reason_code': 'belief_not_stated',
                                     'explanation': 'Retained model classification; semantic audit is separate.'})
    generated = generate_edit_controls(editor, traces, manifest, plan)

    def save(name, value):
        path = tmp_path / name
        write_json(path, value)
        return {'path': name, 'sha256': file_digest(path)}

    raw, mapping = [], []
    for trace in traces:
        row = {'arm': 'reasoning_rl', 'dataset': 'clean', 'model': 'fixture', 'step': 0, 'seed': 1,
               'revision': trace['source_checkpoint']['revision'], 'checkpoint_id': trace['source_checkpoint']['id'],
               'adapter_sha256': 'a' * 64, 'world_id': trace['world_id'], 'id': trace['record_id'],
               'completion_protocol': NATURAL_COMPLETION_PROTOCOL, 'format': 'native_reasoning',
               'completion': trace['trace'] + '\nAnswer: A.', 'completion_ids': list(range(20))}
        raw.append(row)
        mapping.append({key: row[key] for key in ('checkpoint_id', 'adapter_sha256', 'id')})
        mapping[-1]['artifact_id'] = trace['artifact_id']
    evidence = {'schema_version': 1, 'completion_protocol': NATURAL_COMPLETION_PROTOCOL,
                'editor_plan': save('plan.json', plan), 'generated_edits': save('generated.json', generated),
                'classifications': mapping}
    binding = save('editability.json', evidence)
    gate = {'status': 'pass', 'editability_evidence': binding, 'evidence': [binding]}
    return gate, tmp_path, raw, evidence


def test_separate_validated_editor_outcomes_can_be_unfavorable_without_rewriting_training_health(classified_checkpoint):
    gate, root, raw, _ = classified_checkpoint
    before = summarize_trace_health(raw)
    result = _g1_editor_measurements(gate, root, raw)
    panel = result['trace_health_with_editor_outcomes']['panels'][0]
    assert result['status'] == 'measured'
    assert panel['editability']['status'] == 'assessed'
    assert panel['editability']['counts']['belief_not_stated'] == 2
    assert panel['editability']['editable_among_editor_assessed']['estimate'] == 0
    assert result['human_audit'] == 'not_evaluated_by_this_helper'
    assert summarize_trace_health(raw) == before


@pytest.mark.parametrize('change,match', [('missing', 'every native checkpoint'),
                                        ('adapter', 'adapter binding'), ('trace', 'actual native checkpoint trace')])
def test_editor_measurements_bind_actual_checkpoint_trace_and_complete_panel(classified_checkpoint, change, match):
    gate, root, raw, evidence = classified_checkpoint
    if change == 'missing':
        evidence['classifications'].pop()
    elif change == 'adapter':
        evidence['classifications'][0]['adapter_sha256'] = 'b' * 64
    else:
        raw[0]['completion'] = 'A different generated trace\nAnswer: A'
    path = root / 'editability.json'
    write_json(path, evidence)
    gate['editability_evidence']['sha256'] = file_digest(path)
    with pytest.raises(ValueError, match=match):
        _g1_editor_measurements(gate, root, raw)
