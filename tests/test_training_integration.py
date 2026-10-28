"""Real offline CPU optimizer tests: random tiny Qwen, no downloaded weights."""
import json
from pathlib import Path
import pytest

pytestmark = pytest.mark.integration


def tiny_fixture():
    torch = pytest.importorskip('torch')
    pytest.importorskip('trl')
    from tokenizers import AddedToken, Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
    torch.set_num_threads(1)
    vocab = {'[PAD]': 0, '[EOS]': 1, '[UNK]': 2, 'A': 3, 'B': 4, 'Answer': 5, ':': 6,
             'Alice': 7, 'Bob': 8, 'ball': 9, 'box': 10, '.': 11, 'basket': 12, 'believes': 13}
    backend = Tokenizer(WordLevel(vocab, unk_token='[UNK]'))
    backend.pre_tokenizer = Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token='[EOS]', pad_token='[PAD]', unk_token='[UNK]')
    tok.add_tokens([AddedToken('\nAnswer:', normalized=False, special=False)])
    tok.model_input_names = ['input_ids', 'attention_mask']
    tok.chat_template = "{% for message in messages %}{{ message['content'] + '\n' }}{% endfor %}"
    config = Qwen2Config(vocab_size=len(tok), hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                         num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=4096,
                         eos_token_id=1, pad_token_id=0, bos_token_id=1)
    model = Qwen2ForCausalLM(config)
    model.config._name_or_path = 'paper2-random-tiny-qwen'
    return model, tok


def rows():
    return [{'id': str(i), 'world_id': str(i), 'split': 'train', 'dataset': 'clean', 'view': 'direct',
             'story': 'Alice ball box. Bob ball basket.', 'question': 'Alice believes ball?',
             'choices': {'A': 'box', 'B': 'basket'}, 'answer': 'A' if i % 2 else 'B'} for i in range(8)]


@pytest.mark.parametrize('arm', ['label_sft', 'reasoning_sft', 'answer_rl', 'reasoning_rl'])
def test_real_trl_optimizer_and_saved_adapter(tmp_path, arm):
    from paper2.training.config import TrainingConfig
    from paper2.training.runner import run_training
    model, tok = tiny_fixture()
    config = TrainingConfig(arm=arm, stage='confirmatory' if arm == 'reasoning_sft' else 'pilot', model='paper2-random-tiny-qwen', max_steps=2, batch_size=1,
                            gradient_accumulation_steps=8, bf16=False, gradient_checkpointing=False,
                            save_steps=2, eval_steps=1, lora_rank=2, lora_alpha=4, beta=0.0)
    if config.rl:
        # Real generation with a tiny fixed-format grammar creates mixed binary
        # rewards, making this an actual optimizer-change test, not just a call test.
        import torch
        from transformers import LogitsProcessorList
        original_generate = model.generate
        class TestGrammar:
            def __init__(self):
                self.prefix = None
            def __call__(self, input_ids, scores):
                if self.prefix is None:
                    self.prefix = input_ids.shape[1]
                step = input_ids.shape[1] - self.prefix
                if config.reasoning:
                    allowed = [10] if step == 0 else tok.encode('\nAnswer:', add_special_tokens=False) if step == 1 else [3, 4] if step == 2 else [1]
                else:
                    allowed = [3, 4] if step == 0 else [1]
                masked = torch.full_like(scores, float('-inf'))
                masked[:, allowed] = scores[:, allowed]
                return masked
        def constrained_generate(*args, **kwargs):
            processors = list(kwargs.get('logits_processor', []))
            kwargs['logits_processor'] = LogitsProcessorList(processors + [TestGrammar()])
            return original_generate(*args, **kwargs)
        model.generate = constrained_generate
    dev = [{**row, 'id': 'dev-' + row['id'], 'world_id': 'dev-' + row['world_id'], 'split': 'development'}
           for row in rows()[:2]] if config.rl else None
    report = run_training(config, rows(), tmp_path, dev_rows=dev, synthetic_model=model, synthetic_tokenizer=tok,
                          trace_builder=lambda row: 'Alice believes ball box.')
    assert (tmp_path / 'adapter' / 'adapter_model.safetensors').exists()
    assert report['manifest']['status'] == 'complete'
    assert report['manifest']['trainable_parameters'] > 0
    compute = report['compute']
    assert compute['optimized_tokens'] > 0
    assert compute['checkpoint_seconds'] > 0
    if config.rl:
        assert compute['sampled_completions'] == 16
        assert compute['generated_tokens'] > 0
        assert compute['generation_seconds'] > 0
        assert compute['prompt_exposures'] == 2
        assert compute['unique_worlds_seen'] == 2
        metrics = [json.loads(line) for line in (tmp_path / 'checkpoint_evaluations.jsonl').read_text().splitlines()]
        assert len(metrics) == 6
        assert {row['step'] for row in metrics} == {0, 1, 2}
        assert {row['format'] for row in metrics} == {'immediate', 'native_reasoning' if config.reasoning else 'native_answer'}
        assert all((Path(row['path']) / 'adapter_model.safetensors').is_file() for row in metrics)
        assert all(row['dataset'] == 'clean' and row['seed'] == 1 and row['view'] == 'direct' for row in metrics)
        assert not (tmp_path / 'checkpoint-1').exists()
        assert (tmp_path / 'evaluated_checkpoints' / 'checkpoint-1').is_dir()
        evidence = json.loads((tmp_path / 'g1_learning_evidence.json').read_text())
        assert evidence['baseline']['step'] == 0
        assert evidence['latest']['step'] == 2
        assert evidence['primary_format'].startswith('native_')
        assert any(p.detach().abs().sum() > 0 for name, p in model.named_parameters() if 'lora_B' in name)
    else:
        assert compute['sampled_completions'] == 0
        assert compute['prompt_exposures'] == 16
        assert compute['unique_worlds_seen'] == 8
        assert compute['supervised_tokens'] > 0


def test_trace_cap_only_forces_delimiter_and_masks_its_tokens():
    torch = pytest.importorskip('torch')
    from paper2.training.generation import TraceCapProcessor
    _, tok = tiny_fixture()
    cap = TraceCapProcessor(tok, cap=3)
    ids = torch.tensor([[7, 8]])
    for step in range(3):
        scores = torch.zeros(1, len(tok))
        assert torch.equal(cap(ids, scores), scores)
        ids = torch.cat([ids, torch.tensor([[10]])], dim=1)
    forced = []
    for token in cap.delimiter_ids:
        scores = cap(ids, torch.zeros(1, len(tok)))
        assert int(scores.argmax()) == token
        forced.append(token)
        ids = torch.cat([ids, torch.tensor([[token]])], dim=1)
    mask = cap.mask_forced(torch.ones((1, 3 + len(forced) + 1)))
    assert mask[0, :3].tolist() == [1, 1, 1]
    assert mask[0, 3:3 + len(forced)].sum() == 0
    assert mask[0, -1] == 1


def test_reasoning_cap_keeps_delimiter_in_attention_during_real_grpo(tmp_path):
    import functools
    torch = pytest.importorskip('torch')
    pytest.importorskip('trl')
    from transformers import LogitsProcessorList
    from paper2.training.config import TrainingConfig
    from paper2.training.runner import run_training
    model, tok = tiny_fixture()
    original_generate = model.generate
    delimiter_length = len(tok.encode('\nAnswer: ', add_special_tokens=False))
    class LongTraceGrammar:
        def __init__(self):
            self.prefix = None
        def __call__(self, input_ids, scores):
            if self.prefix is None:
                self.prefix = input_ids.shape[1]
            step = input_ids.shape[1] - self.prefix
            if 384 <= step < 384 + delimiter_length:
                return scores  # The production cap processor supplies the delimiter.
            allowed = [10] if step < 384 else [3, 4] if step == 384 + delimiter_length else [1]
            masked = torch.full_like(scores, float('-inf'))
            masked[:, allowed] = scores[:, allowed]
            return masked
    def constrained_generate(*args, **kwargs):
        kwargs['logits_processor'] = LogitsProcessorList(list(kwargs.get('logits_processor', [])) + [LongTraceGrammar()])
        return original_generate(*args, **kwargs)
    model.generate = constrained_generate
    original_forward = model.forward
    masks = []
    @functools.wraps(original_forward)
    def capture_forward(*args, **kwargs):
        if torch.is_grad_enabled() and kwargs.get('attention_mask') is not None:
            masks.append(kwargs['attention_mask'].detach().clone())
        return original_forward(*args, **kwargs)
    model.forward = capture_forward
    config = TrainingConfig(arm='reasoning_rl', model='paper2-random-tiny-qwen', max_steps=2,
                            bf16=False, gradient_checkpointing=False, save_steps=2, eval_steps=2,
                            lora_rank=2, lora_alpha=4, beta=0.0)
    report = run_training(config, rows(), tmp_path, synthetic_model=model, synthetic_tokenizer=tok)
    compute = report['compute']
    assert compute['trace_cap_hits'] == 16
    assert compute['forced_format_tokens'] == 16 * delimiter_length
    assert any(p.detach().abs().sum() > 0 for name, p in model.named_parameters() if 'lora_B' in name)
    assert compute['optimized_tokens'] == compute['generated_tokens'] - compute['forced_format_tokens']
    assert masks
    assert all(mask.eq(1).all() for mask in masks), 'The forced delimiter must stay visible to attention'


def test_g1_native_reasoning_can_succeed_when_immediate_answer_fails():
    torch = pytest.importorskip('torch')
    from types import SimpleNamespace
    from paper2.training.config import TrainingConfig
    from paper2.training.runner import _development_predictions
    _, tok = tiny_fixture()

    class TraceOnlyModel:
        device = torch.device('cpu')
        training = True
        def train(self, mode=True):
            self.training = mode
        def eval(self):
            self.train(False)
        def __call__(self, **encoded):
            logits = torch.zeros(1, encoded['input_ids'].shape[1], len(tok))
            logits[..., tok.unk_token_id] = 10
            return SimpleNamespace(logits=logits)
        def generate(self, input_ids, **kwargs):
            completion = tok.encode('box\nAnswer: A', add_special_tokens=False) + [tok.eos_token_id]
            return torch.cat([input_ids, torch.tensor([completion])], dim=1)

    model = TraceOnlyModel()
    example = dict(rows()[1], split='development')
    results = _development_predictions(model, tok, [example], TrainingConfig(), torch)
    assert results[0]['format'] == 'immediate' and not results[0]['correct']
    assert results[1]['format'] == 'native_reasoning' and results[1]['correct']
    assert model.training, 'Evaluation must restore the training state'


def test_one_epoch_grpo_budget_actually_visits_every_training_world(tmp_path):
    from paper2.training.config import TrainingConfig
    from paper2.training.runner import run_training
    model, tok = tiny_fixture()
    config = TrainingConfig(arm='answer_rl', model='paper2-random-tiny-qwen', max_steps=None,
                            bf16=False, gradient_checkpointing=False, save_steps=8,
                            lora_rank=2, lora_alpha=4, beta=0.0)
    report = run_training(config, rows(), tmp_path, synthetic_model=model, synthetic_tokenizer=tok)
    assert report['manifest']['training_budget']['resolved_max_steps'] == 8
    assert report['compute']['prompt_exposures'] == 8
    assert report['compute']['sampled_completions'] == 64
    assert report['compute']['unique_worlds_seen'] == 8
    assert report['compute']['training_pool_coverage'] == 1
