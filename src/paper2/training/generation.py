"""Stop a trace at the preregistered cap and insert only the answer delimiter.

The injected delimiter has no optimized loss tokens. The subsequent answer is
sampled normally and receives only the ordinary correctness reward.
"""
from typing import Any


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
