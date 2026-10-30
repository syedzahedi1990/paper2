"""Capability amendments change explicit identities without bypassing gate review."""
import json
from dataclasses import replace

import pytest

from paper2.data import VIEWS
from paper2.gates import (_validate_frozen_g4_band, inspect_historical_training_evidence,
                          freeze_calibration, validate_calibration_manifest)
from paper2.io import digest, file_digest, write_json, write_jsonl
from paper2.training.config import MODEL_LAYERS, MODEL_REVISIONS, TrainingConfig, experiment_matrix
from paper2.training.model_plan import (assert_config_model_plan, assert_profile_identity,
                                        calibrated_model_plan)


def bound(path):
    return {'path': str(path), 'sha256': file_digest(path)}


@pytest.fixture
def amended(tmp_path, request):
    # These are synthetic receipts for integrity tests, never experiment evidence.
    manifest = tmp_path / 'manifest.json'
    dev_name = 'evaluation/development__clean.jsonl'
    development = [{'id': f'{i}-{view}', 'world_id': str(i), 'view': view, 'answer': 'A',
                    'split': 'development', 'dataset': 'clean'} for i in range(120) for view in VIEWS]
    write_jsonl(tmp_path / dev_name, development)
    write_json(manifest, {'synthetic_unit_fixture': True, 'corpus_hash': 'f' * 64,
                         'files': {dev_name: {'sha256': file_digest(tmp_path / dev_name)}}})
    target = getattr(request, 'param', 'Qwen/Qwen2.5-7B-Instruct')
    identities = {stage: {'model': target, 'revision': MODEL_REVISIONS[target]}
                  for stage in ('pilot', 'confirmatory')}
    screens = []
    for name in ('Qwen/Qwen2.5-3B-Instruct', target):
        revision = MODEL_REVISIONS[name]
        output = tmp_path / name.split('/')[1]
        output.mkdir()
        protocol = {'purpose': 'development_capability_only', 'model': name, 'revision': revision,
                    'manifest_sha256': file_digest(manifest), 'trace_cap': 384,
                    'corpus_hash': 'f' * 64, 'development_sha256': file_digest(tmp_path / dev_name),
                    'script_sha256': 'd' * 64, 'implementation_sha256': {'example.py': 'e' * 64},
                    'native_procedure': 'uninterrupted_g1_generation_and_reward_v1',
                    'native_views': ['direct'], 'immediate_views': list(VIEWS)}
        write_json(output / 'protocol.json', protocol)
        expected = [('immediate', view) for view in VIEWS] + [('native_reasoning', 'direct')]
        records = [{'world_id': str(i), 'protocol_sha256': digest(protocol),
                    'predictions': [{'id': f'{i}-{view}', 'world_id': str(i), 'format': fmt, 'view': view, 'answer': 'A',
                                     'split': 'development', 'dataset': 'clean', 'reader': name}
                                    for fmt, view in expected]} for i in range(120)]
        for record in records:
            record['record_sha256'] = digest(record)
        write_jsonl(output / 'worlds.jsonl', records)
        write_json(output / 'summary.json', {'completed_worlds': 120,
                   'protocol_sha256': digest(protocol),
                   'worlds_sha256': file_digest(output / 'worlds.jsonl')})
        screens.append({kind: bound(output / f'{kind}.json' if kind != 'worlds' else output / 'worlds.jsonl')
                        for kind in ('protocol', 'summary', 'worlds')})
    amendment = {'schema_version': 1, 'kind': 'model_capability_amendment', 'model_plan': identities,
                 'manifest_sha256': file_digest(manifest), 'reviewed_by': 'Unit-test reviewer',
                 'rationale': 'Synthetic fixture exercising identity checks, not a scientific decision.',
                 'capability_screens': screens}
    amendment_path = tmp_path / 'model-amendment.json'
    write_json(amendment_path, amendment)
    calibration = {'frozen': True, 'manifest_path': str(manifest), 'manifest_sha256': file_digest(manifest),
                   'settings': {'model_plan': {**identities, 'amendment': bound(amendment_path)},
                                'g4_layer_policy': 'all_after_embedding_layer',
                                'g4_test_layers': list(range(1, MODEL_LAYERS[target])),
                                'difficulty_review': {'estimate': .4, 'ci95': [.3, .5],
                                  'relation_to_chance_and_ceiling': 'Synthetic unit test only.'}}}
    return calibration, tmp_path, amendment


def rebind_amendment(calibration, root, amendment):
    path = root / 'model-amendment.json'
    write_json(path, amendment)
    calibration['settings']['model_plan']['amendment'] = bound(path)


def test_legacy_factory_retains_three_b_pilot_seven_b_confirmation():
    plan = calibrated_model_plan({}, '.')
    assert plan['amendment_sha256'] is None
    for stage in ('pilot', 'confirmatory'):
        for config in experiment_matrix(stage):
            config.validate()
            assert_config_model_plan(config.to_dict(), plan)
    assert MODEL_LAYERS['mistralai/Mistral-Small-24B-Instruct-2501'] == 40


@pytest.mark.parametrize('stage', ['pilot', 'profile', 'confirmatory'])
def test_nondefault_training_identity_requires_explicit_hash_and_supported_revision(stage):
    model = 'mistralai/Mistral-Small-24B-Instruct-2501'
    config = TrainingConfig(stage=stage, model=model, max_steps=5 if stage == 'profile' else None)
    with pytest.raises(ValueError, match='protocol amendment'):
        config.validate()
    replace(config, model_amendment_sha256='a' * 64).validate()
    with pytest.raises(ValueError, match='immutable SHA256'):
        replace(config, model_amendment_sha256='not-a-hash').validate()
    with pytest.raises(ValueError, match='supported immutable model profile'):
        replace(config, revision='b' * 40, model_amendment_sha256='a' * 64).validate()


@pytest.mark.parametrize('amended', ['Qwen/Qwen2.5-7B-Instruct',
                                     'mistralai/Mistral-Small-24B-Instruct-2501'], indirect=True)
def test_amended_pilot_freezes_and_profiles_without_confirmatory_gate_circularity(amended, monkeypatch):
    calibration, root, _ = amended
    monkeypatch.setattr('paper2.data.assert_calibration_corpus', lambda _: {'integrity_passed': True})
    evidence = root / 'g0.json'
    write_json(evidence, {'synthetic_unit_fixture': True})
    gate = root / 'review.json'
    review = freeze_calibration(gate, calibration['manifest_path'], calibration['settings'], evidence,
                                'Unit-test reviewer', 'Synthetic integrity test only.')
    plan = calibrated_model_plan(review['calibration'], root)
    config = TrainingConfig(stage='profile', model=plan['pilot']['model'], max_steps=5,
                            model_amendment_sha256=plan['amendment_sha256'])
    config.validate()
    inspection = inspect_historical_training_evidence(config.to_dict(), gate, None)
    assert inspection['eligible_for_current_launch'] is False
    result = inspection['historical_review']
    assert result['gates']['G1']['status'] == 'pending'
    assert result['paid_compute']['approved'] is False
    # Calling the same model confirmatory cannot reuse this pilot authorization.
    with pytest.raises(ValueError, match='profiling report'):
        inspect_historical_training_evidence(replace(config, stage='confirmatory').to_dict(), gate, None)
    validate_calibration_manifest(review['calibration'], root=root)
    review['gates']['G0']['evidence'].pop()  # An unreviewed substitution is not a frozen amendment.
    write_json(gate, review)
    with pytest.raises(ValueError, match='included in reviewed G0 evidence'):
        inspect_historical_training_evidence(config.to_dict(), gate, None)


def test_model_amendment_and_capability_artifact_tampering_are_rejected(amended):
    calibration, root, amendment = amended
    plan = calibrated_model_plan(calibration, root)
    assert plan['pilot'] == plan['confirmatory']
    config = TrainingConfig(model=plan['pilot']['model'], model_amendment_sha256=plan['amendment_sha256'])
    assert assert_config_model_plan(config.to_dict(), plan) == 28
    for changes in ({'revision': 'f' * 40}, {'model_amendment_sha256': None},
                    {'model': 'Qwen/Qwen2.5-3B-Instruct'}):
        with pytest.raises(ValueError, match='frozen model plan'):
            assert_config_model_plan({**config.to_dict(), **changes}, plan)
    (root / 'model-amendment.json').write_text('modified')
    with pytest.raises(ValueError, match='missing or modified'):
        calibrated_model_plan(calibration, root)
    rebind_amendment(calibration, root, amendment)
    from pathlib import Path
    Path(amendment['capability_screens'][0]['worlds']['path']).write_text('modified')
    with pytest.raises(ValueError, match='missing or modified'):
        calibrated_model_plan(calibration, root)


@pytest.mark.parametrize('change,match', [
    ('target', 'identities or targets'), ('record_digest', 'record digest'),
    ('development_digest', 'corpus-bound'), ('implementation', 'different measurement implementations')])
def test_rehashed_capability_evidence_still_must_match_frozen_data_and_common_measurement(amended, change, match):
    from pathlib import Path
    calibration, root, amendment = amended
    screen = amendment['capability_screens'][1]
    protocol_path = Path(screen['protocol']['path'])
    protocol = json.loads(protocol_path.read_text())
    # Actual writer uses canonical protocol hashes inside receipts, and byte
    # hashes in the external evidence list. These are intentionally different.
    assert digest(protocol) != file_digest(protocol_path)
    worlds_path = Path(screen['worlds']['path'])
    rows = [json.loads(line) for line in worlds_path.read_text().splitlines()]
    if change == 'target':
        rows[0]['predictions'][0]['answer'] = 'B'
    elif change == 'development_digest':
        protocol['development_sha256'] = 'a' * 64
    elif change == 'implementation':
        protocol['implementation_sha256']['example.py'] = 'a' * 64
    write_json(protocol_path, protocol)
    for row in rows:
        row['protocol_sha256'] = digest(protocol)
        row['record_sha256'] = digest({key: value for key, value in row.items() if key != 'record_sha256'})
    if change == 'record_digest':
        rows[0]['record_sha256'] = 'a' * 64
    write_jsonl(worlds_path, rows)
    summary_path = Path(screen['summary']['path'])
    summary = json.loads(summary_path.read_text())
    summary.update({'protocol_sha256': digest(protocol), 'worlds_sha256': file_digest(worlds_path)})
    write_json(summary_path, summary)
    screen.update({'protocol': bound(protocol_path), 'summary': bound(summary_path), 'worlds': bound(worlds_path)})
    rebind_amendment(calibration, root, amendment)
    with pytest.raises(ValueError, match=match):
        calibrated_model_plan(calibration, root)


@pytest.mark.parametrize('field,value', [('reviewed_by', ''), ('manifest_sha256', 'f' * 64),
                                       ('capability_screens', [])])
def test_amendment_requires_named_corpus_bound_capability_review(amended, field, value):
    calibration, root, amendment = amended
    amendment[field] = value
    rebind_amendment(calibration, root, amendment)
    with pytest.raises(ValueError, match='amendment'):
        calibrated_model_plan(calibration, root)


def test_selected_confirmation_model_also_needs_completed_capability_evidence(amended):
    calibration, root, amendment = amended
    model = 'mistralai/Mistral-Small-24B-Instruct-2501'
    identity = {'model': model, 'revision': MODEL_REVISIONS[model]}
    amendment['model_plan']['confirmatory'] = identity
    calibration['settings']['model_plan']['confirmatory'] = identity
    rebind_amendment(calibration, root, amendment)
    with pytest.raises(ValueError, match='both selected model identities'):
        calibrated_model_plan(calibration, root)


def test_partial_or_wrong_split_capability_panels_do_not_authorize_amendment(amended):
    from pathlib import Path
    calibration, root, amendment = amended
    screen = amendment['capability_screens'][1]
    path = Path(screen['worlds']['path'])
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]['predictions'][0]['split'] = 'confirmation'
    rows[0]['record_sha256'] = digest({key: value for key, value in rows[0].items() if key != 'record_sha256'})
    write_jsonl(path, rows)
    screen['worlds'] = bound(path)
    summary_path = Path(screen['summary']['path'])
    summary = json.loads(summary_path.read_text())
    summary['worlds_sha256'] = file_digest(path)
    write_json(summary_path, summary)
    screen['summary'] = bound(summary_path)
    rebind_amendment(calibration, root, amendment)
    with pytest.raises(ValueError, match='clean development panels'):
        calibrated_model_plan(calibration, root)


def test_g4_band_tracks_declared_pilot_geometry_not_original_3b(amended):
    calibration, root, _ = amended
    plan = calibrated_model_plan(calibration, root)
    _validate_frozen_g4_band(calibration['settings'], plan=plan)
    for layers in (list(range(1, 36)), list(range(1, 10))):
        with pytest.raises(ValueError, match='full layer geometry'):
            _validate_frozen_g4_band({**calibration['settings'], 'g4_test_layers': layers}, plan=plan)


def test_full_runs_need_own_model_revision_arm_profile_and_explicit_bootstrap_is_limited():
    plan = calibrated_model_plan({}, '.')
    pilot = TrainingConfig(stage='profile').to_dict()
    target = TrainingConfig(stage='confirmatory', model=plan['confirmatory']['model'], arm='reasoning_sft')
    with pytest.raises(ValueError, match='same model, revision, arm'):
        assert_profile_identity(target.to_dict(), {'config': pilot}, plan)
    with pytest.raises(ValueError, match='same model, revision, arm'):
        assert_profile_identity(replace(target, profile=True).to_dict(), {'config': pilot}, plan)
    bootstrap = replace(target, profile=True, target_profile_bootstrap=True, max_steps=5)
    bootstrap.validate()
    assert_profile_identity(bootstrap.to_dict(), {'config': pilot}, plan)
    with pytest.raises(ValueError, match='calibrated pilot model'):
        assert_profile_identity(bootstrap.to_dict(), {'config': {**pilot, 'revision': 'c' * 40}}, plan)
    for invalid in (replace(bootstrap, profile=False), replace(bootstrap, stage='pilot')):
        with pytest.raises(ValueError):
            invalid.validate()
    own_profile = {'config': target.to_dict()}
    assert_profile_identity(target.to_dict(), own_profile, plan)
    for field, value in (('model', plan['pilot']['model']), ('revision', 'f' * 40), ('arm', 'label_sft')):
        with pytest.raises(ValueError, match='same model, revision, arm'):
            assert_profile_identity(target.to_dict(), {'config': {**target.to_dict(), field: value}}, plan)


def test_bootstrap_profile_projection_removes_its_temporary_profile_exception():
    from paper2.training.resources import full_pool_projection
    config = TrainingConfig(stage='confirmatory', model='Qwen/Qwen2.5-7B-Instruct',
                            profile=True, target_profile_bootstrap=True, max_steps=5)
    projected = full_pool_projection(config, 2000, 120, 3, {})['target_config']
    assert projected['profile'] is False and projected['target_profile_bootstrap'] is False
    TrainingConfig(**projected).validate()


@pytest.mark.parametrize('config', [TrainingConfig(stage='profile'), TrainingConfig(profile=True),
    TrainingConfig(stage='confirmatory', model='Qwen/Qwen2.5-7B-Instruct',
                   profile=True, target_profile_bootstrap=True)])
def test_production_profiles_cannot_silently_expand_to_a_full_training_pool(config):
    with pytest.raises(ValueError, match='explicit positive max_steps'):
        config.validate()
    replace(config, max_steps=5).validate()
