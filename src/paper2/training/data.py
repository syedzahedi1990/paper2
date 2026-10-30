"""Strict direct-only training boundary and explicit completion loss masks."""
from __future__ import annotations
import hashlib
import json
import re
from numbers import Integral
from typing import Any, Callable
from .config import (COMPLETION_PROTOCOLS, LEGACY_COMPLETION_PROTOCOL,
                     NATURAL_COMPLETION_PROTOCOL, TrainingConfig)
from .generation import completion_token_budget

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
    completion = trace + '\nAnswer: ' + answer
    capped_text = completion if config.completion_protocol == NATURAL_COMPLETION_PROTOCOL else trace
    if len(tokenizer.encode(capped_text, add_special_tokens=False)) > config.trace_cap:
        raise ValueError('Oracle trace exceeds 384-token cap; shorten its template before training')
    return completion


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
        if (config.reasoning and config.completion_protocol == NATURAL_COMPLETION_PROTOCOL
                and len(target_ids) > config.trace_cap):
            raise ValueError('Oracle completion including EOS exceeds the total sampled-token budget')
        if len(prompt_ids) > config.max_prompt_length:
            raise ValueError('Prompt exceeds configured bound; story truncation is prohibited')
        result.append({'input_ids': prompt_ids + target_ids,
                       'world_id': row.get('world_id', row.get('id')),
                       'attention_mask': [1] * (len(prompt_ids) + len(target_ids)),
                       'completion_mask': [0] * len(prompt_ids) + [1] * len(target_ids)})
    return result


def parse_answer(text: str, reasoning: bool, *,
                 completion_protocol: str = LEGACY_COMPLETION_PROTOCOL) -> str | None:
    if completion_protocol not in COMPLETION_PROTOCOLS:
        raise ValueError('Unknown completion protocol')
    if completion_protocol == NATURAL_COMPLETION_PROTOCOL:
        # No minimum trace length: a naturally sampled answer may start the
        # completion. Only one adjacent terminal period is optional.
        pattern = r'(?:.*\n)?Answer:\s*([A-F])\.?\s*' if reasoning else r'\s*([A-F])\.?\s*'
        match = re.fullmatch(pattern, text, flags=re.DOTALL)
        return match.group(1) if match else None
    if reasoning:
        match = re.fullmatch(r'.*?\nAnswer:\s*([A-F])\s*', text, flags=re.DOTALL)
    else:
        match = re.fullmatch(r'\s*([A-F])\s*', text)
    return match.group(1) if match else None


def _sampled_ids(completion_ids: Any) -> list[int]:
    """Require actual unpadded generation IDs; never reconstruct them from text."""
    if hasattr(completion_ids, 'tolist'):
        completion_ids = completion_ids.tolist()
    if (not isinstance(completion_ids, (list, tuple))
            or any(not isinstance(token, Integral) or isinstance(token, bool) or token < 0
                   for token in completion_ids)):
        raise ValueError('Actual unpadded completion_ids must be a sequence of token IDs')
    return [int(token) for token in completion_ids]


def natural_completion_outcome(text: str, *, reasoning: bool, completion_ids: Any,
                               trace_cap: int = 384, eos_token_id: int | None = None) -> dict[str, Any]:
    """Describe v2 completion validity from the actual sampled-token sequence.

    TRL supplies IDs through the first EOS inclusive, excluding batch padding.
    Other callers must supply the same unpadded sequence. EOS is optional: an
    answer completed at token 384 is valid. These fields impose no trace-content
    or length preference and are separate from direct-answer correctness.
    """
    ids = _sampled_ids(completion_ids)
    budget = completion_token_budget(reasoning=reasoning, completion_protocol=NATURAL_COMPLETION_PROTOCOL,
                                     trace_cap=trace_cap)
    parsed = parse_answer(text, reasoning, completion_protocol=NATURAL_COMPLETION_PROTOCOL)
    within_budget = len(ids) <= budget
    complete = parsed is not None
    cap_reached = len(ids) >= budget
    cap_failure = reasoning and cap_reached and not complete
    invalid = not within_budget or not complete
    eos_seen = eos_token_id is not None and eos_token_id in ids
    if cap_failure:
        termination = 'cap_failure'
    elif invalid:
        termination = 'invalid_completion'
    elif eos_seen:
        termination = 'ordinary_eos'
    elif cap_reached:
        termination = 'valid_at_budget'
    else:
        termination = 'complete_without_eos'
    return {'parsed_answer': parsed, 'prediction': parsed if not invalid else None,
            'sampled_tokens': len(ids), 'token_budget': budget, 'answer_complete': complete,
            'within_token_budget': within_budget, 'cap_reached': cap_reached,
            'cap_failure': cap_failure, 'invalid_completion': invalid, 'eos_seen': eos_seen,
            'termination_reason': termination}


def answer_reward(*, completions: list[Any], answer: list[str], reasoning: bool = False,
                  tokenizer: Any = None, trace_cap: int = 384,
                  completion_protocol: str = LEGACY_COMPLETION_PROTOCOL,
                  completion_ids: list[Any] | None = None, **_: Any) -> list[float]:
    """Only direct-answer correctness is rewarded; no trace/length/format bonus."""
    if completion_protocol not in COMPLETION_PROTOCOLS:
        raise ValueError('Unknown completion protocol')
    if completion_protocol == NATURAL_COMPLETION_PROTOCOL:
        if reasoning and completion_ids is None:
            raise ValueError('Natural reasoning reward requires actual unpadded completion_ids')
        if completion_ids is not None and len(completion_ids) != len(completions):
            raise ValueError('completion_ids must align with every completion')
    scores = []
    for index, (completion, target) in enumerate(zip(completions, answer, strict=True)):
        if isinstance(completion, list):
            completion = completion[-1]['content']
        valid = parse_answer(completion, reasoning, completion_protocol=completion_protocol) == target
        if completion_protocol == NATURAL_COMPLETION_PROTOCOL and completion_ids is not None:
            outcome = natural_completion_outcome(completion, reasoning=reasoning,
                                                  completion_ids=completion_ids[index], trace_cap=trace_cap,
                                                  eos_token_id=getattr(tokenizer, 'eos_token_id', None))
            valid = outcome['prediction'] == target
        elif completion_protocol == LEGACY_COMPLETION_PROTOCOL and reasoning and tokenizer is not None:
            trace = completion.rsplit('\nAnswer:', 1)[0]
            valid &= len(tokenizer.encode(trace, add_special_tokens=False)) <= trace_cap
        scores.append(float(valid))
    return scores


def rows_digest(rows: list[dict[str, Any]]) -> str:
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
