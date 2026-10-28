import json
import pytest
from paper2.training.config import TrainingConfig, experiment_matrix, LORA_MODULES
from paper2.training.data import answer_reward, validate_rows, make_sft_rows, make_rl_rows
from paper2.training.ledger import ComputeLedger


def row(i=0, **changes):
    return {'id': str(i), 'world_id': str(i), 'split': 'train', 'dataset': 'clean', 'view': 'direct',
            'story': 'Alice put a ball in the box. Bob left.', 'question': 'Where does Alice believe the ball is?',
            'choices': {'A': 'box', 'B': 'basket'}, 'answer': 'A', **changes}


class Tokenizer:
    eos_token_id = 0
    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return '\n'.join(message['content'] for message in messages)


def test_matrix_exact_proposal():
    pilot = experiment_matrix('pilot')
    main = experiment_matrix('confirmatory')
    assert len(pilot) == 3
    assert {c.arm for c in pilot} == {'reasoning_rl', 'answer_rl', 'label_sft'}
    assert len(main) == 24
    assert {c.seed for c in main} == {1, 2, 3}
    assert {c.dataset for c in main} == {'clean', 'shortcut'}
    assert all(c.lora_modules == LORA_MODULES and c.group_size == 8 and c.trace_cap == 384 for c in main)
    for config in pilot + main:
        config.validate()


@pytest.mark.parametrize('changes', [{'view': 'search'}, {'split': 'confirmation'}, {'dataset': 'shortcut'}, {'answer': 'box'}, {'targets': []}])
def test_training_boundary_rejects(changes):
    with pytest.raises(ValueError):
        validate_rows([row(**changes)], TrainingConfig())


def test_duplicate_world_rejected():
    with pytest.raises(ValueError):
        validate_rows([row(), row()], TrainingConfig())


def test_only_direct_outcome_rewarded():
    assert answer_reward(completions=[' A', 'A because I know', 'B'], answer=['A'] * 3) == [1., 0., 0.]
    assert answer_reward(completions=['Bob saw the box.\nAnswer: A', 'Answer: A', 'Reasoning\nAnswer: B'], answer=['A'] * 3, reasoning=True) == [1., 0., 0.]
    assert answer_reward(completions=['x' * 385 + '\nAnswer: A'], answer=['A'], reasoning=True, tokenizer=Tokenizer()) == [0.]


def test_explicit_supervision_mask_and_strip_privileged_metadata():
    config = TrainingConfig(arm='label_sft')
    tok = Tokenizer()
    item = row(world={'belief': 'secret'}, trace='privileged trace')
    example = make_sft_rows([item], config, tok)[0]
    supervised = [v for v, keep in zip(example['input_ids'], example['completion_mask']) if keep]
    assert supervised == [ord(' '), ord('A'), 0]
    rl = make_rl_rows([item], TrainingConfig(), tok)[0]
    assert set(rl) == {'prompt', 'answer', 'world_id'}
    assert 'privileged' not in rl['prompt'] and 'secret' not in rl['prompt']


def test_reasoning_sft_source_and_cap():
    config = TrainingConfig(arm='reasoning_sft', stage='confirmatory', model='Qwen/Qwen2.5-7B-Instruct')
    tok = Tokenizer()
    item = row(trace='Alice observed the ball in the box.', trace_source=config.trace_source)
    example = make_sft_rows([item], config, tok)[0]
    target = ''.join(chr(t) for t, m in zip(example['input_ids'], example['completion_mask']) if m and t)
    assert target == item['trace'] + '\nAnswer: A'
    with pytest.raises(ValueError):
        make_sft_rows([dict(item, trace_source='unspecified')], config, tok)
    with pytest.raises(ValueError):
        make_sft_rows([dict(item, trace='x' * 385)], config, tok)


def test_ledger_no_fabricated_gpu_time(tmp_path):
    ledger = ComputeLedger(sampled_completions=8, generated_tokens=24, generation_seconds=2)
    ledger.save(tmp_path / 'compute.json')
    saved = json.loads((tmp_path / 'compute.json').read_text())
    assert saved['gpu_seconds'] == 0
    assert saved['generation_tokens_per_second'] == 12
