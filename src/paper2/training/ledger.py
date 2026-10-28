"""Auditable counters; timing is synchronized on CUDA, not estimated from tokens."""
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass
class ComputeLedger:
    training_examples: int = 0
    answer_supervision: str = ''
    prompt_exposures: int = 0
    observed_world_ids: list[str] = field(default_factory=list)
    completion_prompt_instances: int = 0
    sampled_completions: int = 0
    generated_tokens: int = 0
    optimized_tokens: int = 0
    supervised_tokens: int = 0
    forced_format_tokens: int = 0
    trace_cap_hits: int = 0
    truncated_completions: int = 0
    generation_seconds: float = 0.0
    generation_and_scoring_seconds: float = 0.0
    update_seconds: float = 0.0
    optimizer_seconds: float = 0.0
    checkpoint_seconds: float = 0.0
    completed_optimizer_steps: int = 0
    evaluated_adapter_save_seconds: float = 0.0
    evaluated_adapter_save_count: int = 0
    resumable_checkpoint_save_seconds: float = 0.0
    resumable_checkpoint_save_count: int = 0
    final_adapter_save_seconds: float = 0.0
    max_evaluated_adapter_bytes: int = 0
    evaluation_seconds: float = 0.0
    evaluation_panels: int = 0
    evaluation_native_seconds: float = 0.0
    evaluation_immediate_seconds: float = 0.0
    evaluation_native_panels: int = 0
    evaluation_immediate_panels: int = 0
    evaluation_native_generations: int = 0
    evaluation_immediate_forward_calls: int = 0
    evaluation_generated_tokens: int = 0
    wall_seconds: float = 0.0
    gpu_seconds: float = 0.0
    peak_cuda_memory_bytes: int = 0
    learning_curve: list[dict[str, Any]] = field(default_factory=list)

    def observe_worlds(self, world_ids) -> None:
        self.observed_world_ids = sorted(set(self.observed_world_ids).union(world_ids))

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result['unique_worlds_seen'] = len(self.observed_world_ids)
        result['training_pool_coverage'] = len(self.observed_world_ids) / self.training_examples if self.training_examples else None
        result['generation_tokens_per_second'] = self.generated_tokens / self.generation_seconds if self.generation_seconds else None
        result['truncation_rate'] = self.truncated_completions / self.sampled_completions if self.sampled_completions else 0.0
        result['trace_cap_hit_rate'] = self.trace_cap_hits / self.sampled_completions if self.sampled_completions else 0.0
        result['definitions'] = {
            'prompt_exposures': 'One story per sampled group for RL; one story per SFT forward/backward pass.',
            'unique_worlds_seen': 'Distinct world IDs actually passed to SFT loss or RL rollout generation, retained across resume.',
            'completion_prompt_instances': 'One prompt per sampled completion, including the GRPO group repetitions.',
            'optimized_tokens': 'Non-masked loss-token presentations, counting repeated optimization separately.',
            'update_seconds': 'Forward/backward plus optimizer steps, excluding generation/scoring; includes trainer bookkeeping inside training_step.',
            'gpu_seconds': 'Elapsed occupied CUDA device time for one process/device; includes generation, updates, checkpointing and evaluation.',
            'trace_cap_hits': 'Rollouts in which the 384-token cap required an injected answer delimiter.',
            'evaluation_panels': 'Actual evaluation events, which may include immediate, native or both formats; includes repeats after resume.',
            'evaluation_native_panels': 'Complete native-format development passes, timed separately from immediate scoring.',
            'evaluation_immediate_panels': 'Complete immediate-answer development passes; each measured step retains an adapter.',
            'evaluation_generated_tokens': 'Native development completion tokens; excluded from training rollout token counts.',
            'max_evaluated_adapter_bytes': 'Largest measured adapter snapshot directory, including tokenizer and configuration files.',
        }
        return result

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + '\n')
