"""Explicit generation budgets, including the retained legacy delimiter rescue.

Natural-answer v2 spends its whole reasoning-completion budget on sampled
tokens. It uses no TraceCapProcessor and gets no extra delimiter/answer budget.
The legacy processor below remains unchanged for interpreting prior runs.
"""
from typing import Any
from .config import COMPLETION_PROTOCOLS, LEGACY_COMPLETION_PROTOCOL, NATURAL_COMPLETION_PROTOCOL


def completion_token_budget(*, reasoning: bool, completion_protocol: str = LEGACY_COMPLETION_PROTOCOL,
                            trace_cap: int = 384, delimiter_length: int = 0) -> int:
    """Return the generation limit, not a reward or a trace-quality criterion."""
    if completion_protocol not in COMPLETION_PROTOCOLS:
        raise ValueError('Unknown completion protocol')
    if type(trace_cap) is not int or trace_cap < 1:
        raise ValueError('Completion cap must be a positive integer')
    if type(delimiter_length) is not int or delimiter_length < 0:
        raise ValueError('Delimiter length must be a nonnegative integer')
    if not reasoning:
        return 4
    if completion_protocol == NATURAL_COMPLETION_PROTOCOL:
        return trace_cap
    return trace_cap + delimiter_length + 4


class TraceCapProcessor:
    def __init__(self, tokenizer: Any, cap: int = 384):
        self.tokenizer = tokenizer
        self.cap = cap
        from paper2.prompts import ANSWER_DELIMITER
        self.delimiter_ids = tokenizer.encode(ANSWER_DELIMITER, add_special_tokens=False)
        if not self.delimiter_ids:
            raise ValueError('Answer delimiter must tokenize to a nonempty sequence')
        self.prompt_length: int | None = None
        self.capped_rows: set[int] = set()
        self.forced_positions: dict[int, list[int]] = {}

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        if self.prompt_length is None:
            self.prompt_length = input_ids.shape[1]
        generated_length = input_ids.shape[1] - self.prompt_length
        if generated_length < self.cap:
            return scores
        for i in range(input_ids.shape[0]):
            if i not in self.capped_rows and generated_length == self.cap:
                if self.tokenizer.eos_token_id in input_ids[i, self.prompt_length:].tolist():
                    continue
                text = self.tokenizer.decode(input_ids[i, self.prompt_length:], skip_special_tokens=True)
                if '\nAnswer:' not in text:
                    self.capped_rows.add(i)
            offset = generated_length - self.cap
            if i in self.capped_rows and offset < len(self.delimiter_ids):
                scores[i].fill_(float('-inf'))
                scores[i, self.delimiter_ids[offset]] = 0
                self.forced_positions.setdefault(i, []).append(generated_length)
        return scores

    def mask_forced(self, mask: Any) -> Any:
        result = mask.clone()
        for row, positions in self.forced_positions.items():
            for position in positions:
                if position < result.shape[1]:
                    result[row, position] = 0
        return result
