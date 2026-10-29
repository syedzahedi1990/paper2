"""Preregistered training choices; pilot and confirmatory matrices are explicit."""
from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path
from typing import Any

ARMS = ('label_sft', 'reasoning_sft', 'answer_rl', 'reasoning_rl')
MODEL_REVISIONS = {
    'Qwen/Qwen2.5-3B-Instruct': 'aa8e72537993ba99e69dfaafa59ed015b17504d1',
    'Qwen/Qwen2.5-7B-Instruct': 'a09a35458c702b33eeacc393d103063234e8bc28',
}
LORA_MODULES = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj')


@dataclass(frozen=True)
class TrainingConfig:
    arm: str = 'reasoning_rl'
    stage: str = 'pilot'
    model: str = 'Qwen/Qwen2.5-3B-Instruct'
    dataset: str = 'clean'
    seed: int = 1
    revision: str | None = None
    group_size: int = 8
    trace_cap: int = 384
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_modules: tuple[str, ...] = LORA_MODULES
    max_prompt_length: int = 2048
    # A GRPO step contains group_size completions, not group_size new stories.
    # Resolve epochs against prompt groups so the default actually covers the pool.
    max_steps: int | None = None
    num_train_epochs: float = 1.0
    batch_size: int | None = None
    gradient_accumulation_steps: int | None = None
    learning_rate: float | None = None
    lr_scheduler_type: str = 'constant_with_warmup'
    warmup_ratio: float = 0.05
    beta: float = 0.0
    temperature: float = 0.8
    top_p: float = 1.0
    logging_steps: int = 1
    # eval_steps is the native completion cadence; immediate scoring is cheaper.
    eval_steps: int | None = None
    immediate_eval_steps: int | None = None
    save_steps: int = 50
    save_total_limit: int = 6
    gradient_checkpointing: bool = True
    bf16: bool = True
    load_in_4bit: bool = False
    attn_implementation: str = 'sdpa'
    trace_source: str = 'paper1_symbolic_oracle_v1'
    profile: bool = False

    def __post_init__(self) -> None:
        if self.revision is None:
            object.__setattr__(self, 'revision', MODEL_REVISIONS.get(self.model, 'synthetic'))
        if self.learning_rate is None:
            object.__setattr__(self, 'learning_rate', 2e-5 if self.rl else 1e-6)
        if self.batch_size is None:
            object.__setattr__(self, 'batch_size', 1 if self.rl else 2)
        if self.gradient_accumulation_steps is None:
            object.__setattr__(self, 'gradient_accumulation_steps', 8 if self.rl else 4)
        if self.eval_steps is None:
            # Native reasoning evaluation is sequential and can dominate a run.
            # Immediate scoring and retained RL snapshots use their own cadence.
            object.__setattr__(self, 'eval_steps', 250 if self.rl else 25)
        if self.immediate_eval_steps is None:
            # Preserve short legacy/profile schedules without extra generations.
            object.__setattr__(self, 'immediate_eval_steps', min(self.eval_steps, 50 if self.rl else 25))

    def evaluation_schedule(self, max_steps: int) -> list[int]:
        """Native completion schedule, including base and off-interval final."""
        if max_steps < 1:
            raise ValueError('Evaluation schedule needs positive optimizer steps')
        return sorted({0, max_steps, *range(self.eval_steps, max_steps + 1, self.eval_steps)})

    def immediate_evaluation_schedule(self, max_steps: int) -> list[int]:
        """Cheap matching measurements, each with a retained adapter."""
        if max_steps < 1:
            raise ValueError('Evaluation schedule needs positive optimizer steps')
        return sorted({0, max_steps, *range(self.immediate_eval_steps, max_steps + 1, self.immediate_eval_steps)})

    @property
    def reasoning(self) -> bool:
        return self.arm.startswith('reasoning')

    @property
    def rl(self) -> bool:
        return self.arm.endswith('_rl')

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result['lora_modules'] = list(self.lora_modules)
        return result

    def training_budget(self, examples: int, *, synthetic: bool = False) -> dict[str, Any]:
        """Resolve a one-device optimizer budget in units of distinct prompt groups."""
        if examples < 1:
            raise ValueError('Training budget needs a nonempty corpus')
        prompts_per_step = self.batch_size * self.gradient_accumulation_steps
        if self.rl:
            prompts_per_step //= self.group_size
        if prompts_per_step < 1:
            raise ValueError('Training batch must contain at least one prompt group')
        full_pass_steps = math.ceil(examples / prompts_per_step)
        steps = self.max_steps or math.ceil(self.num_train_epochs * full_pass_steps)
        profiling = self.profile or self.stage == 'profile'
        if not synthetic and not profiling and steps < full_pass_steps:
            raise ValueError('Production learning runs require at least one full training-pool pass; use profile for short runs')
        return {'resolved_max_steps': steps, 'prompt_groups_per_optimizer_step': prompts_per_step,
                'minimum_full_pass_steps': full_pass_steps,
                'planned_prompt_exposures': steps * prompts_per_step,
                'requested_epochs': self.num_train_epochs, 'explicit_max_steps': self.max_steps,
                'require_full_pool_coverage': not synthetic and not profiling}

    def validate(self, *, synthetic: bool = False) -> None:
        if self.arm not in ARMS:
            raise ValueError(f'Unknown arm {self.arm!r}')
        if self.stage not in ('pilot', 'confirmatory', 'profile'):
            raise ValueError('stage must be pilot, confirmatory or profile')
        if self.dataset not in ('clean', 'shortcut'):
            raise ValueError('dataset must be clean or shortcut')
        if self.group_size != 8 or self.trace_cap != 384:
            raise ValueError('Proposal fixes GRPO group size 8 and trace cap 384')
        if tuple(self.lora_modules) != LORA_MODULES:
            raise ValueError('Fixed LoRA targets must include q/k/v/o and gate/up/down projections')
        if (self.max_steps is not None and self.max_steps < 1) or self.batch_size < 1 or self.gradient_accumulation_steps < 1:
            raise ValueError('Training sizes must be positive')
        if not math.isfinite(self.num_train_epochs) or self.num_train_epochs <= 0:
            raise ValueError('num_train_epochs must be finite and positive')
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError('learning_rate must be finite and positive')
        if not 0 <= self.warmup_ratio < 1:
            raise ValueError('warmup_ratio must be between zero and one')
        if self.lr_scheduler_type not in ('constant', 'constant_with_warmup', 'linear', 'cosine'):
            raise ValueError('Unsupported preregistered learning-rate schedule')
        if min(self.eval_steps, self.immediate_eval_steps, self.save_steps, self.logging_steps) < 1:
            raise ValueError('Logging/evaluation/save intervals must be positive')
        if self.rl and self.batch_size * self.gradient_accumulation_steps % self.group_size:
            raise ValueError('Effective single-GPU batch must be divisible by group size 8')
        if self.stage in ('pilot', 'profile') and (self.dataset != 'clean' or self.seed != 1 or self.arm == 'reasoning_sft'):
            raise ValueError('Pilot uses clean data, seed 1, reasoning RL/answer RL/label SFT only')
        if not synthetic:
            expected = 'Qwen/Qwen2.5-7B-Instruct' if self.stage == 'confirmatory' else 'Qwen/Qwen2.5-3B-Instruct'
            if self.model != expected:
                raise ValueError(f'{self.stage} requires {expected}; capability exceptions need a protocol amendment')
            if len(self.revision) != 40 or any(c not in '0123456789abcdef' for c in self.revision):
                raise ValueError('Production model revisions must be immutable 40-character commit hashes')
        if self.stage == 'confirmatory' and self.seed not in (1, 2, 3):
            raise ValueError('Confirmatory seeds are 1, 2, 3')
        if self.arm == 'reasoning_sft' and self.trace_source != 'paper1_symbolic_oracle_v1':
            raise ValueError('Reasoning SFT uses the documented paper1_symbolic_oracle_v1 source; changes require a protocol amendment')


def load_config(path: str | Path) -> TrainingConfig:
    data = json.loads(Path(path).read_text())
    unknown = set(data) - {field.name for field in fields(TrainingConfig)}
    if unknown:
        raise ValueError(f'Unknown training config keys: {sorted(unknown)}')
    if 'lora_modules' in data:
        data['lora_modules'] = tuple(data['lora_modules'])
    result = TrainingConfig(**data)
    result.validate()
    return result


def experiment_matrix(stage: str) -> list[TrainingConfig]:
    if stage == 'pilot':
        return [TrainingConfig(arm=arm) for arm in ('reasoning_rl', 'answer_rl', 'label_sft')]
    if stage == 'confirmatory':
        return [TrainingConfig(arm=arm, stage=stage, model='Qwen/Qwen2.5-7B-Instruct', dataset=dataset, seed=seed)
                for arm in ARMS for dataset in ('clean', 'shortcut') for seed in (1, 2, 3)]
    raise ValueError('Matrix stage must be pilot or confirmatory')
