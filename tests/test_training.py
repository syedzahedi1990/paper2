import json
import pytest
from paper2.training.config import TrainingConfig, experiment_matrix, LORA_MODULES
from paper2.training.data import answer_reward, validate_rows, make_sft_rows, make_rl_rows
from paper2.training.ledger import ComputeLedger
from paper2.training.resources import full_pool_projection, resource_plan


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


def test_default_budget_covers_entire_training_pool_in_both_objectives():
    rl = TrainingConfig(arm='reasoning_rl')
    sft = TrainingConfig(arm='label_sft')
    assert rl.training_budget(2000)['resolved_max_steps'] == 2000
    assert sft.training_budget(2000)['resolved_max_steps'] == 250
    assert rl.training_budget(2000)['planned_prompt_exposures'] == 2000
    assert sft.training_budget(2000)['planned_prompt_exposures'] == 2000
    assert rl.learning_rate == 2e-5 and sft.learning_rate == 1e-6
    assert rl.beta == sft.beta == 0
    assert (rl.batch_size, rl.gradient_accumulation_steps) == (1, 8)
    assert (sft.batch_size, sft.gradient_accumulation_steps) == (2, 4)
    assert TrainingConfig(learning_rate=2e-5).learning_rate == 2e-5
    with pytest.raises(ValueError, match='full training-pool pass'):
        TrainingConfig(max_steps=250).training_budget(2000)
    assert TrainingConfig(stage='profile', max_steps=5).training_budget(2000)['resolved_max_steps'] == 5
    assert TrainingConfig(max_steps=2).training_budget(8, synthetic=True)['resolved_max_steps'] == 2


def test_unique_world_coverage_is_not_completion_count():
    ledger = ComputeLedger(training_examples=4)
    ledger.observe_worlds(['a'] * 8 + ['b'] * 8)
    ledger.observe_worlds(['b', 'c'])
    assert ledger.to_dict()['unique_worlds_seen'] == 3
    assert ledger.to_dict()['training_pool_coverage'] == .75


def test_evaluation_budget_is_sparse_for_rl_and_keeps_base_final_and_profile():
    rl = TrainingConfig()
    sft = TrainingConfig(arm='label_sft')
    plan = resource_plan(rl, 2000, 120, 3)
    assert plan['evaluation_native_steps'] == [0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000]
    assert plan['evaluation_immediate_steps'] == list(range(0, 2001, 50))
    assert plan['evaluation_steps'] == plan['evaluation_immediate_steps']
    assert plan['evaluation_native_panels'] == 9
    assert plan['evaluation_immediate_panels'] == 41
    assert plan['evaluation_immediate_forward_calls'] == 4920
    assert plan['evaluation_native_generations'] == 1080
    assert plan['evaluation_generated_token_upper_bound'] == 1080 * 391
    assert plan['retained_evaluated_adapters'] == 41
    assert resource_plan(sft, 2000, 120, 3)['evaluation_panels'] == 11
    assert rl.evaluation_schedule(503) == [0, 250, 500, 503]
    assert TrainingConfig(stage='profile', max_steps=5, eval_steps=5).evaluation_schedule(5) == [0, 5]
    assert TrainingConfig(eval_steps=200).evaluation_schedule(503) == [0, 200, 400, 503]


def test_shipped_learning_configs_use_the_documented_evaluation_budget():
    from pathlib import Path
    from paper2.training.config import load_config
    root = Path(__file__).resolve().parents[1]
    for path in (root / 'configs').glob('*/*.json'):
        config = load_config(path)
        assert config.eval_steps == (250 if config.rl else 25)
        assert config.immediate_eval_steps == (50 if config.rl else 25)
        assert resource_plan(config, 2000, 120, 3)['evaluation_native_panels'] == (9 if config.rl else 11)
        assert resource_plan(config, 2000, 120, 3)['evaluation_immediate_panels'] == (41 if config.rl else 11)
    profile = load_config(root / 'configs/profile_3b.json')
    assert resource_plan(profile, 2000, 120, 3)['evaluation_steps'] == [0, 5]


def test_projection_scales_training_and_evaluation_separately_and_reports_storage():
    config = TrainingConfig(stage='profile', max_steps=5, eval_steps=5, save_steps=5)
    measured = dict(completed_optimizer_steps=5, wall_seconds=30, evaluation_seconds=20,
                    checkpoint_seconds=4, evaluation_panels=2, evaluated_adapter_save_count=2,
                    evaluation_native_panels=2, evaluation_immediate_panels=2,
                    evaluation_native_seconds=18, evaluation_immediate_seconds=2,
                    evaluated_adapter_save_seconds=2, resumable_checkpoint_save_count=1,
                    resumable_checkpoint_save_seconds=1, final_adapter_save_seconds=1,
                    max_evaluated_adapter_bytes=100)
    projection = full_pool_projection(config, 2000, 120, 3, measured)
    assert projection['projected_seconds']['development_native_evaluation'] == 81
    assert projection['projected_seconds']['development_immediate_evaluation'] == 41
    assert projection['projected_seconds']['training_and_orchestration'] == 2400
    assert projection['projected_total_seconds'] == 2604
    assert projection['projected_evaluated_adapter_bytes'] == 4100
    assert projection['plan']['resumable_checkpoint_writes'] == 40
    assert projection['status'] == 'planning_estimate_requires_longer_gpu_profile_and_review'
    incomplete = full_pool_projection(config, 2000, 120, 3, {})
    assert incomplete['projected_total_seconds'] is None
    assert incomplete['projected_evaluated_adapter_bytes'] is None
    old_profile = {k: v for k, v in measured.items() if k != 'evaluation_immediate_seconds'}
    assert full_pool_projection(config, 2000, 120, 3, old_profile)['projected_total_seconds'] is None


def test_independent_evaluation_grids_keep_all_measured_adapters_without_extra_generations():
    config = TrainingConfig(stage='profile', max_steps=7, eval_steps=3, immediate_eval_steps=2)
    plan = resource_plan(config, 8, 2, 1)
    assert plan['evaluation_native_steps'] == [0, 3, 6, 7]
    assert plan['evaluation_immediate_steps'] == [0, 2, 4, 6, 7]
    assert plan['evaluation_steps'] == [0, 2, 3, 4, 6, 7]
    assert plan['retained_evaluated_adapters'] == 6
    assert plan['evaluation_native_generations'] == 8
    assert plan['evaluation_immediate_forward_calls'] == 10
    assert resource_plan(config, 8, 0, 1)['retained_evaluated_adapters'] == 0
    with pytest.raises(ValueError, match='intervals must be positive'):
        TrainingConfig(immediate_eval_steps=0).validate()


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
