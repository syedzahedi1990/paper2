"""Count evaluation explicitly and project only measured same-arm resource rates."""
from __future__ import annotations
from dataclasses import replace
from typing import Any

from .config import LEGACY_COMPLETION_PROTOCOL, TrainingConfig
from .generation import completion_token_budget


PROFILE_ARRANGEMENT_KEYS = (
    'model', 'revision', 'arm', 'model_amendment_sha256', 'group_size', 'trace_cap',
    'lora_rank', 'lora_alpha', 'lora_modules', 'batch_size', 'gradient_accumulation_steps',
    'max_prompt_length', 'load_in_4bit', 'attn_implementation', 'bf16', 'gradient_checkpointing',
    'beta', 'completion_protocol',
)


def profile_arrangement_mismatches(profile_config: dict[str, Any],
                                   config: TrainingConfig) -> list[str]:
    """Compare measured execution arrangements, retaining only legacy defaults.

    Historical profiles may predate explicit beta/protocol fields. Their absent
    fields mean beta zero and the old forced-answer protocol, never the revised
    protocol. Learning rates and optimization duration are deliberately not
    execution-arrangement keys.
    """
    measured = {**profile_config}
    measured.setdefault('beta', 0.0)
    measured.setdefault('completion_protocol', LEGACY_COMPLETION_PROTOCOL)
    target = config.to_dict()
    if 'lora_modules' in measured:
        measured['lora_modules'] = list(measured['lora_modules'])
    return [key for key in PROFILE_ARRANGEMENT_KEYS if measured.get(key) != target[key]]


def resource_plan(config: TrainingConfig, training_examples: int, development_examples: int,
                  delimiter_tokens: int, *, synthetic: bool = False) -> dict[str, Any]:
    budget = config.training_budget(training_examples, synthetic=synthetic)
    steps = budget['resolved_max_steps']
    native_evaluations = config.evaluation_schedule(steps) if development_examples else []
    immediate_evaluations = config.immediate_evaluation_schedule(steps) if development_examples else []
    evaluations = sorted(set(native_evaluations) | set(immediate_evaluations))
    native_budget = completion_token_budget(
        reasoning=config.reasoning, completion_protocol=config.completion_protocol,
        trace_cap=config.trace_cap, delimiter_length=delimiter_tokens)
    native_generations = len(native_evaluations) * development_examples
    checkpoints = sorted({steps, *range(config.save_steps, steps + 1, config.save_steps)})
    return {
        'schema_version': 3,
        'completion_protocol': config.completion_protocol,
        'beta': config.beta,
        'optimizer_steps': steps,
        'training_prompt_exposures': budget['planned_prompt_exposures'],
        'training_rollout_completions': budget['planned_prompt_exposures'] * config.group_size if config.rl else 0,
        'evaluation_steps': evaluations, 'evaluation_panels': len(evaluations),
        'evaluation_native_steps': native_evaluations, 'evaluation_native_panels': len(native_evaluations),
        'evaluation_immediate_steps': immediate_evaluations, 'evaluation_immediate_panels': len(immediate_evaluations),
        'development_worlds_per_panel': development_examples,
        'evaluation_native_generations': native_generations,
        'evaluation_immediate_forward_calls': len(immediate_evaluations) * development_examples,
        'evaluation_native_generation_batch_size': 1,
        'evaluation_max_new_tokens_per_generation': native_budget,
        'evaluation_generated_token_upper_bound': native_generations * native_budget,
        'retained_evaluated_adapters': len(evaluations),
        'resumable_checkpoint_writes': len(checkpoints),
        'retained_resumable_checkpoints_upper_bound': min(len(checkpoints), config.save_total_limit),
        'final_adapter_copies': 1,
        'storage_note': 'Evaluated snapshots, rotating resumable checkpoints and the final adapter are separate copies.',
    }


def full_pool_projection(config: TrainingConfig, training_examples: int, development_examples: int,
                         delimiter_tokens: int, measured: dict[str, Any], *, synthetic: bool = False,
                         measured_config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Project an ordinary one-pool run from this arm/model's measured profile.

    A short profile is not a compute approval. In particular, early greedy trace
    lengths need not predict the learned model's lengths. Missing measured phases
    deliberately leave the total estimate unset.
    """
    if measured_config is None:
        if config.completion_protocol != LEGACY_COMPLETION_PROTOCOL or config.beta != 0.0:
            raise ValueError('Revised protocol/beta resource projections require the actual measured_config')
        measured_binding = {'completion_protocol': LEGACY_COMPLETION_PROTOCOL, 'beta': 0.0}
        binding_scope = 'Legacy unbound ledger; only historical forced-answer protocol and beta zero.'
    else:
        mismatch = profile_arrangement_mismatches(measured_config, config)
        if mismatch:
            raise ValueError(f'Resource projection differs from measured arrangement: {mismatch}')
        measured_binding = {key: measured_config.get(key) for key in PROFILE_ARRANGEMENT_KEYS}
        measured_binding['completion_protocol'] = measured_config.get('completion_protocol', LEGACY_COMPLETION_PROTOCOL)
        measured_binding['beta'] = measured_config.get('beta', 0.0)
        binding_scope = 'Explicit configuration of the measured resource ledger.'
    target = replace(config, stage='pilot' if config.stage == 'profile' else config.stage,
                     profile=False, target_profile_bootstrap=False, max_steps=None, num_train_epochs=1.0,
                     eval_steps=250 if config.rl else 25, immediate_eval_steps=50 if config.rl else 25, save_steps=50)
    plan = resource_plan(target, training_examples, development_examples, delimiter_tokens, synthetic=synthetic)
    completed_steps = measured.get('completed_optimizer_steps', 0)
    native_panels = measured.get('evaluation_native_panels', 0)
    immediate_panels = measured.get('evaluation_immediate_panels', 0)
    measured_saves = measured.get('evaluated_adapter_save_count', 0)
    measured_checkpoints = measured.get('resumable_checkpoint_save_count', 0)
    training_seconds = max(0.0, measured.get('wall_seconds', 0.0) - measured.get('evaluation_seconds', 0.0)
                           - measured.get('checkpoint_seconds', 0.0))
    rates = {
        'training_and_orchestration_seconds_per_step': training_seconds / completed_steps if completed_steps else None,
        'native_evaluation_seconds_per_panel': measured['evaluation_native_seconds'] / native_panels if native_panels and 'evaluation_native_seconds' in measured else None,
        'immediate_evaluation_seconds_per_panel': measured['evaluation_immediate_seconds'] / immediate_panels if immediate_panels and 'evaluation_immediate_seconds' in measured else None,
        'evaluated_adapter_save_seconds_per_copy': measured.get('evaluated_adapter_save_seconds', 0.0) / measured_saves if measured_saves else None,
        'resumable_checkpoint_seconds_per_write': measured.get('resumable_checkpoint_save_seconds', 0.0) / measured_checkpoints if measured_checkpoints else None,
    }
    def scaled(rate, count):
        return rates[rate] * count if rates[rate] is not None else (0.0 if count == 0 else None)
    seconds = {
        'training_and_orchestration': scaled('training_and_orchestration_seconds_per_step', plan['optimizer_steps']),
        'development_native_evaluation': scaled('native_evaluation_seconds_per_panel', plan['evaluation_native_panels']),
        'development_immediate_evaluation': scaled('immediate_evaluation_seconds_per_panel', plan['evaluation_immediate_panels']),
        'evaluated_adapter_saves': scaled('evaluated_adapter_save_seconds_per_copy', plan['retained_evaluated_adapters']),
        'resumable_checkpoint_saves': scaled('resumable_checkpoint_seconds_per_write', plan['resumable_checkpoint_writes']),
        'final_adapter_save': measured.get('final_adapter_save_seconds'),
    }
    snapshot_bytes = measured.get('max_evaluated_adapter_bytes')
    return {
        'status': 'planning_estimate_requires_longer_gpu_profile_and_review',
        'same_arm_only': config.arm, 'same_model_only': config.model,
        'measured_arrangement': measured_binding, 'measured_arrangement_scope': binding_scope,
        'target_config': target.to_dict(), 'plan': plan, 'measured_rates': rates,
        'projected_seconds': seconds,
        'projected_total_seconds': sum(seconds.values()) if all(v is not None for v in seconds.values()) else None,
        'projected_evaluated_adapter_bytes': snapshot_bytes * plan['retained_evaluated_adapters'] if snapshot_bytes else None,
        'storage_scope': 'Evaluated adapters with tokenizer/config only; excludes rotating optimizer checkpoints, final adapter, base model and datasets.',
        'limitations': [
            'An initial five-step profile does not establish steady-state throughput or a 15–25 hour pilot budget.',
            'Native development generations remain sequential; learned trace lengths may exceed the profiled lengths.',
            'Training and evaluation rates apply only to this arm, model, device and generation arrangement.',
            'Loading the base model precedes measured training wall time; repeat-run startup and later Tests 1–3 are additional costs.',
            'Use occupied GPU time, all arms, all seeds and storage costs in the reviewed cost estimate.',
        ],
    }
