"""Confirmation uses computed panel winners, never template learning rates."""
from dataclasses import replace
import json

import pytest

from paper2.io import file_digest, write_json
from paper2.training.confirmatory_selection import (
    export_selected_confirmatory_configs, verify_confirmatory_selection,
)
from paper2.training.selection import amended_experiment_matrix


@pytest.fixture
def selected(tmp_path, monkeypatch):
    """Isolate gate binding from the panel verifier's full artifact fixtures."""
    receipt = tmp_path / 'selection.json'
    settings = {'rl': {'learning_rate': 6e-5, 'lr_scheduler_type': 'constant', 'warmup_ratio': 0.},
                'sft': {'learning_rate': 1e-5, 'lr_scheduler_type': 'constant', 'warmup_ratio': 0.}}
    curves = {}
    for arm in ('reasoning_rl', 'answer_rl', 'label_sft'):
        path = tmp_path / f'{arm}-learning.json'
        write_json(path, {'arm': arm})
        curves[arm] = {'path': path.name, 'sha256': file_digest(path)}
    result = {'method': 'amended_pilot_family_selection_v1', 'corpus_manifest_sha256': 'a' * 64,
              'selected_settings': settings, 'selected_run_evidence': curves}
    write_json(receipt, result)
    calls = []
    def checked(path, expected_sha256=None):
        # Tests below exercise how the gate consumes the independently verified
        # winner, not a substitute implementation of panel selection.
        calls.append((path, expected_sha256))
        assert path == receipt
        if expected_sha256 is not None:
            assert expected_sha256 == file_digest(receipt)
        return result
    monkeypatch.setattr('paper2.training.confirmatory_selection._verified_selection', checked)
    binding = {'path': receipt.name, 'sha256': file_digest(receipt)}
    review = {'calibration': {'manifest_sha256': 'a' * 64},
              'development_selection': binding,
              'gates': {'G1': {'evidence': [binding],
                               'learning_curves': [{'arm': arm, **entry} for arm, entry in curves.items()]}}}
    return receipt, result, review, calls


@pytest.mark.parametrize('arm', ['reasoning_rl', 'answer_rl', 'label_sft', 'reasoning_sft'])
def test_each_arm_uses_the_computed_family_setting(selected, arm):
    receipt, result, review, calls = selected
    config = next(config for config in amended_experiment_matrix('confirmatory') if config.arm == arm)
    family = 'rl' if config.rl else 'sft'
    chosen = replace(config, **result['selected_settings'][family])
    authorized = verify_confirmatory_selection(chosen.to_dict(), review, receipt.parent)
    assert authorized['family'] == family
    assert authorized['selected_setting'] == result['selected_settings'][family]
    assert authorized['sha256'] == file_digest(receipt)
    assert calls == [(receipt, file_digest(receipt))]


@pytest.mark.parametrize('field,value', [('learning_rate', 2e-5), ('lr_scheduler_type', 'linear'),
                                         ('warmup_ratio', .05)])
def test_provisional_or_modified_settings_cannot_launch(selected, field, value):
    receipt, result, review, _ = selected
    config = amended_experiment_matrix('confirmatory')[0].to_dict()
    config.update(result['selected_settings']['rl'])
    config[field] = value
    with pytest.raises(ValueError, match=f'selected family setting.*{field}'):
        verify_confirmatory_selection(config, review, receipt.parent)


def test_sft_cannot_use_the_rl_family_rate(selected):
    receipt, result, review, _ = selected
    config = next(config for config in amended_experiment_matrix('confirmatory') if config.arm == 'reasoning_sft')
    with pytest.raises(ValueError, match='selected family setting.*learning_rate'):
        verify_confirmatory_selection(replace(config, **result['selected_settings']['rl']).to_dict(), review,
                                      receipt.parent)


def test_missing_selection_or_modified_receipt_blocks_confirmation(selected):
    receipt, _, review, calls = selected
    config = amended_experiment_matrix('confirmatory')[0].to_dict()
    missing = {key: value for key, value in review.items() if key != 'development_selection'}
    with pytest.raises(ValueError, match='bound computed development_selection'):
        verify_confirmatory_selection(config, missing, receipt.parent)
    receipt.write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='missing or modified'):
        verify_confirmatory_selection(config, review, receipt.parent)
    assert not calls


def test_selection_must_be_in_reviewed_g1_evidence(selected):
    receipt, _, review, calls = selected
    review['gates']['G1']['evidence'] = []
    with pytest.raises(ValueError, match='included in reviewed G1'):
        verify_confirmatory_selection(amended_experiment_matrix('confirmatory')[0].to_dict(), review,
                                      receipt.parent)
    assert not calls


def test_selection_corpus_and_all_selected_g1_arms_are_bound(selected):
    receipt, result, review, _ = selected
    config = replace(amended_experiment_matrix('confirmatory')[0], **result['selected_settings']['rl']).to_dict()
    review['calibration']['manifest_sha256'] = 'b' * 64
    with pytest.raises(ValueError, match='different frozen corpus'):
        verify_confirmatory_selection(config, review, receipt.parent)
    review['calibration']['manifest_sha256'] = 'a' * 64
    # Keep the correct chosen reasoning candidate, but try to substitute the
    # answer arm from another rate. The shared-family selection still forbids it.
    answer = next(row for row in review['gates']['G1']['learning_curves'] if row['arm'] == 'answer_rl')
    answer['path'] = 'some-preferred-other-run.json'
    with pytest.raises(ValueError, match='G1 answer_rl curve must be the candidate chosen'):
        verify_confirmatory_selection(config, review, receipt.parent)


def test_verifier_is_reinvoked_even_for_a_rehashed_receipt(selected, monkeypatch):
    receipt, _, review, _ = selected
    forged = json.loads(receipt.read_text())
    forged['selected_settings']['rl']['learning_rate'] = 2e-5
    write_json(receipt, forged)
    review['development_selection']['sha256'] = file_digest(receipt)
    def reject(*_, **__):
        raise ValueError('Computed winner differs from actual final panel scores')
    monkeypatch.setattr('paper2.training.confirmatory_selection._verified_selection', reject)
    with pytest.raises(ValueError, match='actual final panel scores'):
        verify_confirmatory_selection(amended_experiment_matrix('confirmatory')[0].to_dict(), review,
                                      receipt.parent)


def test_export_derives_all_24_configs_without_overwriting_templates(selected, tmp_path):
    receipt, result, _, calls = selected
    output = tmp_path / 'selected-confirmation'
    exported = export_selected_confirmatory_configs(receipt, output, selection_sha256=file_digest(receipt))
    assert len(exported['configs']) == 24
    assert exported['development_selection'] == {'path': str(receipt), 'sha256': file_digest(receipt)}
    for item in exported['configs']:
        config = json.loads((output / item['path']).read_text())
        family = 'rl' if config['arm'].endswith('_rl') else 'sft'
        assert {key: config[key] for key in result['selected_settings'][family]} == result['selected_settings'][family]
        assert config['completion_protocol'] == 'natural_answer_cap_v2'
        assert config['beta'] == (.04 if family == 'rl' else 0.)
        assert config['stage'] == 'confirmatory'
        assert file_digest(output / item['path']) == item['sha256']
    assert len(calls) == 1
    before = {path.name: file_digest(path) for path in output.iterdir()}
    with pytest.raises(ValueError, match='refusing to overwrite'):
        export_selected_confirmatory_configs(receipt, output)
    assert before == {path.name: file_digest(path) for path in output.iterdir()}


def test_a_named_and_hashed_winner_without_actual_scores_cannot_authorize_or_export(tmp_path):
    # This calls the real panel verifier: merely naming the correct-looking
    # method, candidate and rate cannot replace the actual frozen panel data.
    path = tmp_path / 'handwritten-selection.json'
    write_json(path, {'schema_version': 1, 'method': 'amended_pilot_family_selection_v1',
                     'status': 'frozen', 'reviewed_by': 'Reviewer',
                     'selected_reasoning_rl_candidate': 'reasoning_rl_clean_seed1_lr-x1_constant',
                     'selected_settings': {'rl': {'learning_rate': 2e-5,
                         'lr_scheduler_type': 'constant', 'warmup_ratio': 0.}}})
    binding = {'path': path.name, 'sha256': file_digest(path)}
    review = {'development_selection': binding, 'gates': {'G1': {'evidence': [binding]}}}
    with pytest.raises(ValueError, match='bound candidate_execution_binding'):
        verify_confirmatory_selection(amended_experiment_matrix('confirmatory')[0].to_dict(), review, tmp_path)
    output = tmp_path / 'must-not-exist'
    with pytest.raises(ValueError, match='bound candidate_execution_binding'):
        export_selected_confirmatory_configs(path, output, selection_sha256=file_digest(path))
    assert not output.exists()
