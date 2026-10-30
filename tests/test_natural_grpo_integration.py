"""Exercise actual pinned GRPO reward, KL reference and failed-token loss path."""
import json
import pytest

pytestmark = pytest.mark.integration


def test_natural_grpo_has_zero_cap_rewards_before_advantages_and_base_reference(tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    pytest.importorskip('trl')
    from transformers import LogitsProcessorList
    from test_training_integration import tiny_fixture, rows
    from paper2.training.config import TrainingConfig, NATURAL_COMPLETION_PROTOCOL
    from paper2.training import runner
    model, tok = tiny_fixture()
    original_generate = model.generate

    class MixedGrammar:
        def __init__(self):
            self.prefix = None

        def __call__(self, input_ids, scores):
            if self.prefix is None:
                self.prefix = input_ids.shape[1]
            step = input_ids.shape[1] - self.prefix
            out = torch.full_like(scores, float('-inf'))
            for index in range(len(input_ids)):
                # Half the group never writes an answer; half naturally finishes.
                token = 10 if index % 2 == 0 else (10 if step == 0 else
                    tok.encode('\nAnswer:', add_special_tokens=False)[0] if step == 1 else
                    3 if step == 2 else 1)
                out[index, token] = scores[index, token]
            return out

    def generated(*args, **kwargs):
        kwargs['logits_processor'] = LogitsProcessorList(list(kwargs.get('logits_processor', [])) + [MixedGrammar()])
        return original_generate(*args, **kwargs)

    model.generate = generated
    observed = []
    original_factory = runner._make_trainers

    def factory(config, ledger):
        sft, grpo = original_factory(config, ledger)

        class InspectGRPO(grpo):
            def _generate_and_score_completions(self, inputs):
                result = super()._generate_and_score_completions(inputs)
                observed.append({key: result[key].detach().clone() for key in
                                 ('advantages', 'ref_per_token_logps', 'completion_mask', 'completion_ids')})
                assert self.beta == .04
                return result
        return sft, InspectGRPO

    monkeypatch.setattr(runner, '_make_trainers', factory)
    config = TrainingConfig(arm='reasoning_rl', model=model.config._name_or_path,
        completion_protocol=NATURAL_COMPLETION_PROTOCOL, beta=.04, max_steps=1,
        batch_size=1, gradient_accumulation_steps=8, bf16=False, gradient_checkpointing=False,
        lora_rank=2, lora_alpha=4, save_steps=1, eval_steps=1, immediate_eval_steps=1)
    training = [{**row, 'answer': 'A'} for row in rows()]
    report = runner.run_training(config, training, tmp_path, synthetic_model=model, synthetic_tokenizer=tok)
    assert observed
    for result in observed:
        assert torch.isfinite(result['ref_per_token_logps']).all()
        assert (result['advantages'][::2] < 0).all()
        assert (result['advantages'][1::2] > 0).all()
        assert (result['completion_mask'][::2].sum(dim=1) == 384).all()
        assert (result['completion_ids'][::2] == 10).all()
    assert report['compute']['forced_format_tokens'] == report['compute']['trace_cap_hits'] == 0
    assert report['compute']['natural_cap_reached'] == report['compute']['natural_cap_failures'] == 4
    assert report['compute']['optimized_tokens'] >= 4 * 384
    manifest = json.loads((tmp_path / 'run_manifest.json').read_text())
    assert manifest['config']['completion_protocol'] == NATURAL_COMPLETION_PROTOCOL


@pytest.mark.parametrize('arm', ['reasoning_rl', 'answer_rl', 'label_sft'])
def test_revised_all_arm_learning_evidence_and_unmeasured_cheap_checkpoints(tmp_path, arm):
    torch = pytest.importorskip('torch')
    pytest.importorskip('trl')
    from transformers import LogitsProcessorList
    from test_training_integration import tiny_fixture, rows
    from paper2.training.config import TrainingConfig, NATURAL_COMPLETION_PROTOCOL
    from paper2.training.runner import run_training
    model, tok = tiny_fixture()
    original_generate = model.generate
    reasoning = arm == 'reasoning_rl'
    # This toy WordLevel decoder otherwise inserts a space before punctuation.
    tok.clean_up_tokenization_spaces = True

    class NaturalGrammar:
        def __init__(self):
            self.prefix = None

        def __call__(self, ids, scores):
            if self.prefix is None:
                self.prefix = ids.shape[1]
            step = ids.shape[1] - self.prefix
            sequence = ([10, tok.encode('\nAnswer:', add_special_tokens=False)[0], 3, 11, 1]
                        if reasoning else [3, 11, 1])
            token = sequence[min(step, len(sequence) - 1)]
            out = torch.full_like(scores, float('-inf'))
            out[:, token] = scores[:, token]
            return out

    def generated(*args, **kwargs):
        kwargs['logits_processor'] = LogitsProcessorList(list(kwargs.get('logits_processor', [])) + [NaturalGrammar()])
        return original_generate(*args, **kwargs)

    model.generate = generated
    config = TrainingConfig(arm=arm, model=model.config._name_or_path,
        completion_protocol=NATURAL_COMPLETION_PROTOCOL, beta=.04 if arm != 'label_sft' else 0,
        max_steps=2, batch_size=1, gradient_accumulation_steps=8, bf16=False,
        gradient_checkpointing=False, lora_rank=2, lora_alpha=4, save_steps=2,
        eval_steps=2, immediate_eval_steps=1)
    training = [{**row, 'answer': 'A'} for row in rows()]
    dev = [{**row, 'id': 'dev-' + row['id'], 'world_id': 'dev-' + row['world_id'],
            'split': 'development'} for row in training[:2]]
    run_training(config, training, tmp_path, dev_rows=dev, synthetic_model=model, synthetic_tokenizer=tok)
    health = [json.loads(line) for line in (tmp_path / 'checkpoint_trace_health.jsonl').read_text().splitlines()]
    assert [row['step'] for row in health] == [0, 1, 2]
    assert health[1]['native_reasoning_assessed'] is False
    assert health[1]['trace_health']['status'] == 'not_assessed'
    assert health[0]['native_reasoning_assessed'] is reasoning
    assert all(row['editability']['status'] == 'not_assessed' for row in health)
    evidence = json.loads((tmp_path / 'g1_learning_evidence.json').read_text())
    from paper2.io import file_digest
    assert evidence['artifacts']['trace_health']['sha256'] == file_digest(tmp_path / 'checkpoint_trace_health.jsonl')
    metrics = [json.loads(line) for line in (tmp_path / 'checkpoint_evaluations.jsonl').read_text().splitlines()]
    assert all(row['accuracy'] == 1 for row in metrics if row['format'].startswith('native'))
