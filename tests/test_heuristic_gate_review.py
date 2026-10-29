import pytest

from paper2.gates import freeze_calibration, validate_calibration_manifest
from paper2.io import write_json


def test_g0_records_named_difficulty_review_without_a_heuristic_waiver(tmp_path, monkeypatch):
    manifest, evidence = tmp_path / 'manifest.json', tmp_path / 'evidence.json'
    write_json(manifest, {'corpus_hash': 'actual-corpus'})
    write_json(evidence, {'estimate': .5, 'ci95': [.4, .6], 'heuristics': {'lookup': .9}})
    calls = []

    def audit(path):
        calls.append(path)
        return {'integrity_passed': True, 'empirical_calibration_status': 'requires_named_review'}

    monkeypatch.setattr('paper2.data.assert_calibration_corpus', audit)
    path = tmp_path / 'review.json'
    with pytest.raises(ValueError, match='difficulty_review'):
        freeze_calibration(path, manifest, {}, evidence, 'Reviewer', 'Development calibration')
    assert not path.exists()
    settings = {'g4_layer_policy': 'all_after_embedding_layer', 'difficulty_review': {'estimate': .5, 'ci95': [.4, .6],
                'relation_to_chance_and_ceiling': 'Difficulty interval and every heuristic were examined.'}}
    review = freeze_calibration(path, manifest, settings, evidence, 'Reviewer', 'Development calibration')
    assert calls == [manifest]
    assert validate_calibration_manifest(review['calibration'])['empirical_calibration_status'] == 'requires_named_review'
    assert 'residual_heuristic_review' not in review['calibration']['settings']
    write_json(manifest, {'corpus_hash': 'different-corpus'})
    with pytest.raises(ValueError, match='modified after calibration'):
        validate_calibration_manifest(review['calibration'])
