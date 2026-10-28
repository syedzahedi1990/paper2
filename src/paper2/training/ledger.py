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
    evaluation_seconds: float = 0.0
    wall_seconds: float = 0.0
    gpu_seconds: float = 0.0
    peak_cuda_memory_bytes: int = 0
    learning_curve: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result['generation_tokens_per_second'] = self.generated_tokens / self.generation_seconds if self.generation_seconds else None
        result['truncation_rate'] = self.truncated_completions / self.sampled_completions if self.sampled_completions else 0.0
        result['trace_cap_hit_rate'] = self.trace_cap_hits / self.sampled_completions if self.sampled_completions else 0.0
        result['definitions'] = {
            'prompt_exposures': 'One story per sampled group for RL; one story per SFT forward/backward pass.',
            'completion_prompt_instances': 'One prompt per sampled completion, including the GRPO group repetitions.',
            'optimized_tokens': 'Non-masked loss-token presentations, counting repeated optimization separately.',
            'update_seconds': 'Forward/backward plus optimizer steps, excluding generation/scoring; includes trainer bookkeeping inside training_step.',
            'gpu_seconds': 'Elapsed occupied CUDA device time for one process/device; includes generation, updates, checkpointing and evaluation.',
            'trace_cap_hits': 'Rollouts in which the 384-token cap required an injected answer delimiter.',
        }
        return result

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + '\n')
