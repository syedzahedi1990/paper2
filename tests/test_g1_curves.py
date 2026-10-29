"""Integrity-only G1 evidence checks; flat learning curves remain reviewable."""
import json

import pytest

from paper2.gates import verify_g1_learning_curves
from paper2.io import file_digest, read_jsonl, write_json, write_jsonl
from paper2.training.config import TrainingConfig
from paper2.training.runner import finalize_g1_learning_evidence


@pytest.fixture
def g1_panel(tmp_path):
    gate = {'status': 'pass', 'reviewed_by': 'Pilot reviewer', 'rationale': 'Reviewed every arm',
            'evidence': [], 'learning_curves': []}
    for arm in ('reasoning_rl', 'answer_rl', 'label_sft'):
        output = tmp_path / arm
        output.mkdir()
        final = 250 if arm == 'label_sft' else 2000
        config = TrainingConfig(arm=arm, max_steps=final, eval_steps=final, immediate_eval_steps=final)
        manifest = {'status': 'complete', 'config': config.to_dict(), 'synthetic_integration_test': False,
                    'training_examples': 2000, 'training_sha256': '1' * 64, 'corpus_manifest_sha256': '2' * 64,
                    'training_budget': {'resolved_max_steps': final}}
        write_json(output / 'run_manifest.json', manifest)
        write_json(output / 'compute_report.json', {'completed_optimizer_steps': final, 'unique_worlds_seen': 2000})
        metrics, predictions = [], []
        native = 'native_reasoning' if arm == 'reasoning_rl' else 'native_answer'
        for step in (0, final):
            identity = {key: config.to_dict()[key] for key in ('arm', 'dataset', 'seed', 'model', 'revision')}
            identity.update({'step': step, 'checkpoint_id': f'{arm}-{step}', 'adapter_sha256': '3' * 64,
                             'split': 'development', 'view': 'direct', 'evaluation_dataset': 'clean'})
            for fmt in ('immediate', native):
                metrics.append({**identity, 'format': fmt, 'n': 120, 'accuracy': .5})
                predictions.extend({**identity, 'format': fmt, 'id': f'dev-{i}', 'world_id': f'world-{i}',
                                    'correct': i % 2 == 0} for i in range(120))
        write_jsonl(output / 'checkpoint_evaluations.jsonl', metrics)
        write_jsonl(output / 'checkpoint_predictions.jsonl', predictions)
        receipt_path = output / 'g1_learning_evidence.json'
        write_json(receipt_path, {'primary_format': native, 'baseline': metrics[1], 'latest': metrics[3],
                                  'immediate_latest': metrics[2]})
        finalize_g1_learning_evidence(output)
        item = {'path': str(receipt_path.relative_to(tmp_path)), 'sha256': file_digest(receipt_path)}
        gate['evidence'].append(dict(item))
        gate['learning_curves'].append({**item, 'arm': arm, 'interpretation': 'Flat curve; reviewer assesses the optimization implication.'})
    return gate, tmp_path


def rebind(gate, root, arm='label_sft'):
    """Deliberate invalid-structure tests update hashes so structure is checked."""
    path = root / arm / 'g1_learning_evidence.json'
    finalize_g1_learning_evidence(path.parent)
    for collection in ('evidence', 'learning_curves'):
        for item in gate[collection]:
            if root / item['path'] == path:
                item['sha256'] = file_digest(path)


def test_flat_learning_curves_are_complete_evidence_without_improvement_threshold(g1_panel):
    gate, root = g1_panel
    result = verify_g1_learning_curves(gate, root, samples=200, expected_manifest_sha256='2' * 64)
    assert set(result) == {'reasoning_rl', 'answer_rl', 'label_sft'}
    assert all(item['native_accuracy_gain'] == item['immediate_accuracy_gain'] == 0 for item in result.values())


def test_missing_sft_arm_is_not_a_complete_g1_review(g1_panel):
    gate, root = g1_panel
    gate['learning_curves'].pop()
    with pytest.raises(ValueError, match='all three pilot arms'):
        verify_g1_learning_curves(gate, root, samples=200)


def test_each_arm_needs_reviewer_interpretation(g1_panel):
    gate, root = g1_panel
    gate['learning_curves'][-1]['interpretation'] = ''
    with pytest.raises(ValueError, match='reviewer interpretation for label_sft'):
        verify_g1_learning_curves(gate, root, samples=200)


@pytest.mark.parametrize('filename', ['checkpoint_evaluations.jsonl', 'checkpoint_predictions.jsonl',
                                      'run_manifest.json', 'compute_report.json'])
def test_tampered_learning_curve_artifacts_are_rejected(g1_panel, filename):
    gate, root = g1_panel
    path = root / 'label_sft' / filename
    path.write_text(path.read_text() + '\n')
    with pytest.raises(ValueError, match='artifact missing or modified'):
        verify_g1_learning_curves(gate, root, samples=200)


@pytest.mark.parametrize('step', [0, 250])
def test_baseline_and_final_are_required_in_both_formats(g1_panel, step):
    gate, root = g1_panel
    path = root / 'label_sft' / 'checkpoint_evaluations.jsonl'
    write_jsonl(path, [row for row in read_jsonl(path) if not (row['step'] == step and row['format'] == 'immediate')])
    rebind(gate, root)
    with pytest.raises(ValueError, match='complete scheduled native and immediate'):
        verify_g1_learning_curves(gate, root, samples=200)


@pytest.mark.parametrize("field,value", [("synthetic_integration_test", True), ("status", "failed")])
def test_failed_or_synthetic_run_cannot_substitute_for_completed_pilot(g1_panel, field, value):
    gate, root = g1_panel
    path = root / 'label_sft' / 'run_manifest.json'
    manifest = json.loads(path.read_text())
    manifest[field] = value
    write_json(path, manifest)
    rebind(gate, root)
    with pytest.raises(ValueError, match='completed non-synthetic'):
        verify_g1_learning_curves(gate, root, samples=200)


def test_aggregate_metrics_must_match_the_complete_prediction_panel(g1_panel):
    gate, root = g1_panel
    path = root / 'label_sft' / 'checkpoint_predictions.jsonl'
    rows = read_jsonl(path)
    rows[0]['correct'] = not rows[0]['correct']
    write_jsonl(path, rows)
    rebind(gate, root)
    with pytest.raises(ValueError, match='aggregate accuracy disagrees'):
        verify_g1_learning_curves(gate, root, samples=200)


def test_all_arms_must_share_frozen_corpus_and_development_stories(g1_panel):
    gate, root = g1_panel
    with pytest.raises(ValueError, match='different frozen corpus'):
        verify_g1_learning_curves(gate, root, samples=200, expected_manifest_sha256='4' * 64)
    path = root / 'label_sft' / 'checkpoint_predictions.jsonl'
    rows = read_jsonl(path)
    rows[0]['world_id'] = 'other-world'
    write_jsonl(path, rows)
    rebind(gate, root)
    with pytest.raises(ValueError, match='same development stories'):
        verify_g1_learning_curves(gate, root, samples=200)


def test_g1_intervals_pair_each_story_across_baseline_and_final(g1_panel):
    gate, root = g1_panel
    # Same successes/failures at both endpoints give uncertain accuracies but an
    # exactly zero paired difference. An unpaired bootstrap would miss this.
    result = verify_g1_learning_curves(gate, root, samples=500)
    for arm in result.values():
        assert arm["bootstrap_samples"] == 500 and arm["bootstrap_seed"] == 2026
        assert arm["native_gain_interval"] == {"estimate": 0., "ci95": [0., 0.], "stories": 120}
        assert arm["immediate_gain_interval"]["ci95"] == [0., 0.]
        for curve in arm["curves"]:
            interval = curve["accuracy_interval"]
            assert interval["estimate"] == .5 and interval["stories"] == 120
            assert interval["ci95"][0] < .5 < interval["ci95"][1]


def test_verify_g1_cli_writes_report_without_changing_pending_gate(g1_panel, monkeypatch):
    from paper2.cli import main
    import paper2.gates
    gate, root = g1_panel
    gate["status"] = "pending"
    review = root / "review.json"
    write_json(review, {"gates": {"G1": gate}, "calibration": {"manifest_sha256": "2" * 64}})
    original = review.read_bytes()
    # Exercise real validation and reporting while keeping this CLI unit fast.
    verifier = paper2.gates.verify_g1_learning_curves
    monkeypatch.setattr(paper2.gates, "verify_g1_learning_curves",
                        lambda *args, **kwargs: verifier(*args, **kwargs, samples=200))
    output = root / "g1-report.json"
    main(["verify-g1", "--review", str(review), "--output", str(output)])
    report = json.loads(output.read_text())
    assert report["review_sha256"] == file_digest(review)
    assert set(report["learning_curves"]) == {"reasoning_rl", "answer_rl", "label_sft"}
    assert all(item["native_gain_interval"]["ci95"] == [0., 0.] for item in report["learning_curves"].values())
    assert review.read_bytes() == original
    with pytest.raises(ValueError, match="overwrite|review"):
        main(["verify-g1", "--review", str(review), "--output", str(review)])
    assert review.read_bytes() == original


def test_g1_gain_interval_tracks_paired_changes_not_just_marginal_accuracies(g1_panel):
    gate, root = g1_panel
    path = root / 'label_sft' / 'checkpoint_predictions.jsonl'
    rows = read_jsonl(path)
    for row in rows:
        if row['step'] == 250 and row['format'] == 'native_answer':
            row['correct'] = not row['correct']
    write_jsonl(path, rows)
    rebind(gate, root)
    result = verify_g1_learning_curves(gate, root, samples=500)['label_sft']
    # Both endpoint accuracies remain .5, but successes and failures exchange.
    assert result['native_gain_interval']['estimate'] == 0
    assert result['native_gain_interval']['ci95'][0] < 0 < result['native_gain_interval']['ci95'][1]
    assert result['immediate_gain_interval']['ci95'] == [0., 0.]
