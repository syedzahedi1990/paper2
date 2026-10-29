"""Profile-only resource snapshots around real pinned tiny-model training."""
import json

import pytest


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def fixture(arm, *, profiling=True):
    from paper2.training.config import TrainingConfig
    from test_training_integration import rows, tiny_fixture
    model, tokenizer = tiny_fixture()
    config = TrainingConfig(
        arm=arm, stage='profile' if profiling else 'pilot', model=model.config._name_or_path,
        max_steps=3, eval_steps=2, immediate_eval_steps=2, save_steps=2,
        bf16=False, gradient_checkpointing=True, lora_rank=2, lora_alpha=4, beta=0.0,
    )
    dev = [{**rows()[0], 'id': 'dev', 'world_id': 'dev', 'split': 'development'}]
    if config.reasoning:
        # Short real generations keep this resource-ordering regression cheap.
        # Both development and GRPO still execute the actual generate machinery.
        import torch
        from transformers import LogitsProcessorList
        generate = model.generate

        class ShortGrammar:
            def __init__(self):
                self.prefix = None

            def __call__(self, input_ids, scores):
                if self.prefix is None:
                    self.prefix = input_ids.shape[1]
                step = input_ids.shape[1] - self.prefix
                token = [10, tokenizer.encode('\nAnswer:', add_special_tokens=False)[0], 3, 1][min(step, 3)]
                masked = torch.full_like(scores, float('-inf'))
                masked[:, token] = scores[:, token]
                return masked

        def constrained_generate(*args, **kwargs):
            kwargs['logits_processor'] = LogitsProcessorList(list(kwargs.get('logits_processor', [])) + [ShortGrammar()])
            return generate(*args, **kwargs)

        model.generate = constrained_generate
    return config, model, tokenizer, rows(), dev


@pytest.mark.integration
@pytest.mark.parametrize('arm', ['label_sft', 'answer_rl', 'reasoning_rl'])
def test_real_profile_records_baseline_then_each_step_before_evaluation(tmp_path, monkeypatch, arm):
    from paper2.training import runner
    config, model, tokenizer, train, dev = fixture(arm)
    path = tmp_path / 'resource_steps.jsonl'
    evaluation_observations = []
    original_evaluate = runner._development_predictions

    def observe_evaluation(*args, **kwargs):
        evaluation_observations.append([row['step'] for row in read_rows(path)] if path.exists() else [])
        return original_evaluate(*args, **kwargs)

    monkeypatch.setattr(runner, '_development_predictions', observe_evaluation)
    result = runner.run_training(config, train, tmp_path, dev_rows=dev,
                                 synthetic_model=model, synthetic_tokenizer=tokenizer)
    snapshots = read_rows(path)
    assert [row['step'] for row in snapshots] == [0, 1, 2, 3]
    assert evaluation_observations == [[], [], [0, 1, 2], [0, 1, 2], [0, 1, 2, 3], [0, 1, 2, 3]]
    assert [row['event'] for row in snapshots] == ['baseline_after_initial_evaluation'] + ['optimizer_step_before_evaluation'] * 3
    assert all(row['segment_id'] == row['segment_start_step'] == 0 for row in snapshots)
    assert all(row['resume_from_checkpoint'] is None and row['synthetic'] is True for row in snapshots)
    for key in ['elapsed_wall_seconds', *[key for key in snapshots[0] if key.startswith('cumulative_')]]:
        values = [row[key] for row in snapshots]
        assert all(value >= 0 for value in values)
        assert values == sorted(values), key
    # Baseline is after the initial panel; steps 1 and 2 include no new panel.
    assert [row['cumulative_evaluation_panels'] for row in snapshots] == [1, 1, 1, 2]
    assert [row['cumulative_evaluation_native_panels'] for row in snapshots] == [1, 1, 1, 2]
    assert [row['cumulative_evaluation_immediate_panels'] for row in snapshots] == [1, 1, 1, 2]
    for field in ['evaluation_seconds', 'evaluation_native_seconds', 'evaluation_immediate_seconds', 'evaluation_generated_tokens']:
        key = f'cumulative_{field}'
        assert snapshots[0][key] == snapshots[1][key] == snapshots[2][key]
        assert snapshots[-1][key] < result['compute'][field]
    for field in ['generation_seconds', 'generation_and_scoring_seconds', 'update_seconds', 'optimizer_seconds',
                  'generated_tokens', 'sampled_completions']:
        assert snapshots[0][f'cumulative_{field}'] == 0
        assert snapshots[-1][f'cumulative_{field}'] == result['compute'][field]
    assert result['compute']['evaluation_panels'] == 3
    assert result['compute']['evaluation_native_generations'] == len(dev) * 3
    assert result['compute']['evaluation_immediate_forward_calls'] == len(dev) * 3
    if config.rl:
        assert [row['cumulative_sampled_completions'] for row in snapshots] == [0, 8, 16, 24]
    else:
        assert all(row['cumulative_generated_tokens'] == row['cumulative_sampled_completions'] == 0 for row in snapshots)
    assert snapshots[-1]['elapsed_wall_seconds'] < result['compute']['wall_seconds']
    assert snapshots[-1]['cumulative_update_seconds'] >= snapshots[-1]['cumulative_optimizer_seconds'] > 0
    profile = json.loads((tmp_path / 'profile_report.json').read_text())
    assert profile['evaluation_panels'] == 3
    assert profile['evaluation_seconds'] > snapshots[-1]['cumulative_evaluation_seconds']


@pytest.mark.integration
def test_profile_resume_appends_explicit_segment_and_preserves_all_previous_observations(tmp_path):
    from paper2.training.runner import run_training
    config, model, tokenizer, train, dev = fixture('answer_rl')
    run_training(config, train, tmp_path, dev_rows=dev, synthetic_model=model, synthetic_tokenizer=tokenizer)
    path = tmp_path / 'resource_steps.jsonl'
    original_bytes = path.read_bytes()
    _, fresh_model, fresh_tokenizer, _, _ = fixture('answer_rl')
    checkpoint = tmp_path / 'checkpoint-2'
    result = run_training(config, train, tmp_path, dev_rows=dev,
                          synthetic_model=fresh_model, synthetic_tokenizer=fresh_tokenizer,
                          resume_from_checkpoint=str(checkpoint))
    assert path.read_bytes().startswith(original_bytes)
    snapshots = read_rows(path)
    assert [row['step'] for row in snapshots] == [0, 1, 2, 3, 2, 3]
    assert [row['segment_id'] for row in snapshots] == [0, 0, 0, 0, 1, 1]
    resumed = snapshots[4:]
    assert all(row['segment_start_step'] == 2 and row['resume_from_checkpoint'] == str(checkpoint.resolve()) for row in resumed)
    assert [row['event'] for row in resumed] == ['baseline_after_initial_evaluation', 'optimizer_step_before_evaluation']
    assert [row['cumulative_sampled_completions'] for row in resumed] == [16, 24]
    assert resumed[-1]['cumulative_sampled_completions'] == result['compute']['sampled_completions']
    assert resumed[1]['elapsed_wall_seconds'] > resumed[0]['elapsed_wall_seconds']
    before_refusal = path.read_bytes()
    with pytest.raises(ValueError, match='already contains a run'):
        run_training(config, train, tmp_path, dev_rows=dev, synthetic_model=fresh_model, synthetic_tokenizer=fresh_tokenizer)
    assert path.read_bytes() == before_refusal


@pytest.mark.integration
def test_nonprofile_training_does_not_add_resource_snapshots(tmp_path):
    from paper2.training.runner import run_training
    config, model, tokenizer, train, _ = fixture('label_sft', profiling=False)
    run_training(config, train, tmp_path, synthetic_model=model, synthetic_tokenizer=tokenizer)
    assert not (tmp_path / 'resource_steps.jsonl').exists()


def test_profile_snapshots_reject_orphaned_existing_evidence_without_explicit_resume(tmp_path):
    from paper2.training.runner import _ProfileResourceSnapshots
    path = tmp_path / 'resource_steps.jsonl'
    original = '{"segment_id": 0, "step": 0}\n'
    path.write_text(original)
    with pytest.raises(ValueError, match='already exist'):
        _ProfileResourceSnapshots(path, None, synthetic=True)
    assert path.read_text() == original
