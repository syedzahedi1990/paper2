"""Shared formatting for training, direct answering, trace readers and swaps."""
SYSTEM = "Follow the story exactly. Characters know only what they observe or are told."
ANSWER_DELIMITER = "\nAnswer:"


def training_prompt(row, reasoning=False):
    choices = "\n".join(f"{label}. {location}" for label, location in row["choices"].items())
    instruction = ("Explain the relevant observations and beliefs, then finish with a new line "
                   "Answer: followed by exactly one option letter.") if reasoning else (
                       "Answer the question immediately with exactly one option letter.")
    return f"Story:\n{row['story']}\n\nQuestion: {row['question']}\n{choices}\n\n{instruction}"


def render_prompt(tokenizer, row, reasoning=False, trace=None):
    # The reader's context never names the trace source or changes by checkpoint.
    user = training_prompt(row, reasoning=reasoning or trace is not None)
    text = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True,
    )
    if trace is not None:
        return text + "Reasoning:\n" + trace.rstrip() + ANSWER_DELIMITER
    return text + ("Reasoning:\n" if reasoning else "Answer:")


def prequestion_prompt(tokenizer, row):
    user = (f"Story:\n{row['story']}\n\nDescribe the observations and beliefs in this story. "
            "No question has been revealed. End the summary with <END_SUMMARY>.")
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True,
    ) + "Reasoning:\n"


def answer_token_ids(tokenizer, prompt, choices):
    """Require one context-correct token per answer; never silently average pieces."""
    prefix = tokenizer.encode(prompt, add_special_tokens=False)
    result = {}
    for label in choices:
        full = tokenizer.encode(prompt + " " + label, add_special_tokens=False)
        if full[:len(prefix)] != prefix or len(full) != len(prefix) + 1:
            raise ValueError(f"Answer {label!r} is not one prefix-stable token; fix the prefill")
        result[label] = full[-1]
    if len(set(result.values())) != len(result):
        raise ValueError("Answer labels have collided")
    return result
