"""Revised G1 provenance and profile compatibility are integrity-only checks."""
import json
from dataclasses import replace

import pytest

from paper2.gates import _g1_editor_measurements, verify_g1_learning_curves
from paper2.io import file_digest, read_jsonl, write_json, write_jsonl
from paper2.training.config import LEGACY_COMPLETION_PROTOCOL, NATURAL_COMPLETION_PROTOCOL, TrainingConfig
from paper2.training.model_plan import assert_profile_identity, calibrated_model_plan
from paper2.training.trace_health import summarize_trace_health
from paper2.training.selection import amended_experiment_matrix
from test_g1_curves import g1_panel, rebind  # noqa: F401


@pytest.fixture
def revised_g1(request, monkeypatch):
    gate, root = request.getfixturevalue('g1_panel')
    gate['status'] = 'pending'
    configs = {config.arm: config.to_dict() for config in amended_experiment_matrix('pilot')}
    model_identity = {key: configs['reasoning_rl'][key] for key in ('model', 'revision')}
    plan = {'pilot': model_identity, 'confirmatory': model_identity,
            'amendment_sha256': configs['reasoning_rl']['model_amendment_sha256']}
    # File-backed model-amendment verification has separate integrity tests.
    monkeypatch.setattr('paper2.training.model_plan.calibrated_model_plan', lambda *_: plan)
    for arm in ('reasoning_rl', 'answer_rl', 'label_sft'):
        output = root / arm
        path = output / 'run_manifest.json'
        manifest = json.loads(path.read_text())
        manifest['config'].update({key: configs[arm][key] for key in
                                  ('model', 'revision', 'model_amendment_sha256',
                                   'completion_protocol', 'beta', 'lr_scheduler_type', 'warmup_ratio')})
        write_json(path, manifest)
        metrics = [{**row, **model_identity, 'completion_protocol': NATURAL_COMPLETION_PROTOCOL}
                   for row in read_jsonl(output / 'checkpoint_evaluations.jsonl')]
        predictions = []
        for row in read_jsonl(output / 'checkpoint_predictions.jsonl'):
            value = {**row, **model_identity, 'completion_protocol': NATURAL_COMPLETION_PROTOCOL}
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
            receipt[key].update({**model_identity, 'completion_protocol': NATURAL_COMPLETION_PROTOCOL})
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
    assert result['reasoning_rl']['eligible_for_current_g1'] is True
    assert result['reasoning_rl']['gate_decision'] == 'not_applied_by_evidence_inspection'
    assert result['reasoning_rl']['trace_health_lengths']['schema_version'] == 2


@pytest.mark.parametrize('field,value', [('beta', 0.), ('lr_scheduler_type', 'constant_with_warmup'),
                                       ('warmup_ratio', .05)])
def test_current_g1_rejects_superseded_training_settings_even_with_revised_parser(revised_g1, field, value):
    gate, root = revised_g1
    path = root / 'reasoning_rl' / 'run_manifest.json'
    manifest = json.loads(path.read_text())
    manifest['config'][field] = value
    write_json(path, manifest)
    rebind(gate, root, 'reasoning_rl')
    with pytest.raises(ValueError, match='Superseded configuration'):
        verify_g1_learning_curves(gate, root, samples=20)


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


def test_all_legacy_mistral_arms_are_historical_evidence_not_a_current_g1(revised_g1):
    gate, root = revised_g1
    gate['status'] = 'pass'  # An archived pass cannot lift current eligibility.
    for arm in ('reasoning_rl', 'answer_rl', 'label_sft'):
        output = root / arm
        path = output / 'run_manifest.json'
        manifest = json.loads(path.read_text())
        manifest['config'].update(completion_protocol=LEGACY_COMPLETION_PROTOCOL, beta=0.)
        write_json(path, manifest)
        for filename in ('checkpoint_evaluations.jsonl', 'checkpoint_predictions.jsonl'):
            write_jsonl(output / filename, [{**row, 'completion_protocol': LEGACY_COMPLETION_PROTOCOL}
                                            for row in read_jsonl(output / filename)])
        receipt_path = output / 'g1_learning_evidence.json'
        receipt = json.loads(receipt_path.read_text())
        receipt['completion_protocol'] = LEGACY_COMPLETION_PROTOCOL
        for key in ('baseline', 'latest', 'immediate_latest'):
            receipt[key]['completion_protocol'] = LEGACY_COMPLETION_PROTOCOL
        write_json(receipt_path, receipt)
        rebind(gate, root, arm)
    with pytest.raises(ValueError, match='Superseded configuration'):
        verify_g1_learning_curves(gate, root, samples=20)
    historical = verify_g1_learning_curves(gate, root, samples=20, historical=True)
    assert all(row['eligible_for_current_g1'] is False for row in historical.values())
    assert all(row['gate_decision'] == 'not_applied_by_evidence_inspection' for row in historical.values())


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
