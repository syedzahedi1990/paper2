"""Strict direct-only training boundary and explicit completion loss masks."""
from __future__ import annotations
import hashlib
import json
import re
from typing import Any, Callable
from .config import TrainingConfig

TRACE_INFORMATION = {
    'paper1_symbolic_oracle_v1': {
        'source': 'Deterministic symbolic rollout of the task world and observation/communication rules.',
        'extra_information': 'Privileged intermediate belief-state updates from the generator; outcome RL and label SFT do not receive these trace targets.',
        'supervision': 'Trace text plus the direct-question answer only; no consequence-question targets.',
    }
}


def validate_rows(rows: list[dict[str, Any]], config: TrainingConfig) -> None:
    if not rows:
        raise ValueError('Training data is empty')
    seen = set()
    for row in rows:
        if row.get('split') != 'train' or row.get('view') != 'direct':
            raise ValueError('Only split=train, view=direct may enter a training dataset')
        if row.get('dataset') != config.dataset:
            raise ValueError('Training dataset does not match configured clean/shortcut arm')
        answer = row.get('answer')
        if not isinstance(answer, str) or answer not in tuple('ABCDEF') or answer not in row.get('choices', {}):
            raise ValueError('Training answer must be a single valid choice letter A-F')
        key = row.get('world_id', row.get('story_id', row.get('id')))
        if not key or key in seen:
            raise ValueError('Each training world must occur exactly once')
        seen.add(key)
        if not isinstance(row.get('story'), str) or not isinstance(row.get('question'), str):
            raise ValueError('Training rows need story and question strings')
        forbidden = {'targets', 'consequence_targets', 'evaluation_targets', 'rewards_by_view'}
        if forbidden.intersection(row):
            raise ValueError('Evaluation/consequence target fields cannot enter the training boundary')


def completion_text(row: dict[str, Any], config: TrainingConfig, tokenizer: Any,
                    trace_builder: Callable | None = None) -> str:
    answer = row['answer']
    if not config.reasoning:
        return ' ' + answer
    if row.get('trace') is not None:
        if row.get('trace_source') != config.trace_source:
            raise ValueError('Reasoning SFT trace source does not match the preregistered config')
        trace = row['trace'].strip()
    else:
        if trace_builder is None:
            from paper2.data import build_oracle_trace
            trace_builder = build_oracle_trace
        trace = trace_builder(row).strip()
    if not trace or 'Answer:' in trace:
        raise ValueError('Oracle trace must be nonempty and must not contain an answer delimiter')
    if len(tokenizer.encode(trace, add_special_tokens=False)) > config.trace_cap:
        raise ValueError('Oracle trace exceeds 384-token cap; shorten its template before training')
    return trace + '\nAnswer: ' + answer


def make_rl_rows(rows: list[dict[str, Any]], config: TrainingConfig, tokenizer: Any) -> list[dict[str, Any]]:
    from paper2.prompts import render_prompt
    validate_rows(rows, config)
    # The trainer never receives world states, consequence labels, or an oracle trace.
    return [{'prompt': render_prompt(tokenizer, row, reasoning=config.reasoning),
             'answer': row['answer'], 'world_id': row.get('world_id', row.get('id'))}
            for row in rows]


def make_sft_rows(rows: list[dict[str, Any]], config: TrainingConfig, tokenizer: Any,
                  trace_builder: Callable | None = None) -> list[dict[str, Any]]:
    from paper2.prompts import render_prompt
    validate_rows(rows, config)
    result = []
    for row in rows:
        prompt = render_prompt(tokenizer, row, reasoning=config.reasoning)
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        completion = completion_text(row, config, tokenizer, trace_builder)
        # Explicit concatenation keeps the prompt boundary exact even with BPE merges.
        target_ids = tokenizer.encode(completion, add_special_tokens=False) + [tokenizer.eos_token_id]
        if len(prompt_ids) > config.max_prompt_length:
            raise ValueError('Prompt exceeds configured bound; story truncation is prohibited')
        result.append({'input_ids': prompt_ids + target_ids,
                       'attention_mask': [1] * (len(prompt_ids) + len(target_ids)),
                       'completion_mask': [0] * len(prompt_ids) + [1] * len(target_ids)})
    return result


def parse_answer(text: str, reasoning: bool) -> str | None:
    if reasoning:
        match = re.fullmatch(r'.*?\nAnswer:\s*([A-F])\s*', text, flags=re.DOTALL)
    else:
        match = re.fullmatch(r'\s*([A-F])\s*', text)
    return match.group(1) if match else None


def answer_reward(*, completions: list[Any], answer: list[str], reasoning: bool = False,
                  tokenizer: Any = None, trace_cap: int = 384, **_: Any) -> list[float]:
    """Only direct-answer correctness is rewarded; no trace/length/format bonus."""
    scores = []
    for completion, target in zip(completions, answer, strict=True):
        if isinstance(completion, list):
            completion = completion[-1]['content']
        valid = parse_answer(completion, reasoning) == target
        if reasoning and tokenizer is not None:
            trace = completion.rsplit('\nAnswer:', 1)[0]
            valid &= len(tokenizer.encode(trace, add_special_tokens=False)) <= trace_cap
        scores.append(float(valid))
    return scores


def rows_digest(rows: list[dict[str, Any]]) -> str:
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
