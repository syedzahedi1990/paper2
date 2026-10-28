"""Test 1 and native-format evaluation; all scoring uses ordinary task targets."""
from dataclasses import asdict
from .prompts import render_prompt


def result_row(row, answer, reader, format_name="immediate", **metadata):
    keep = ("id", "rendering_id", "world_id", "view", "split", "dataset", "cue_condition", "cue_location",
            "cue_answer", "cue_agrees_direct", "cue_agrees_evaluated", "answer", "target_location", "target_scope", "belief_depth")
    return {**{k: row[k] for k in keep if k in row}, **asdict(answer),
            "reader": reader, "format": format_name, "correct": answer.prediction == row["answer"],
            **metadata}


def evaluate_immediate(backend, rows, reader=None):
    for row in rows:
        prompt = render_prompt(backend.tokenizer, row)
        answer = backend.answer(prompt, row["choices"])
        yield result_row(row, answer, reader or backend.checkpoint_id)


def evaluate_native(backend, rows, reasoning=False, reader=None, native_answer=False):
    if not reasoning and not native_answer:
        yield from evaluate_immediate(backend, rows, reader)
        return
    from .backend import Answer
    from .training.data import parse_answer
    from .traces import generate_trace
    for row in rows:
        trace = generate_trace(backend, row, source=reader or backend.checkpoint_id) if reasoning else None
        prompt = trace["answer_prompt"] if trace else render_prompt(backend.tokenizer, row)
        prefix_ids = trace["answer_prefix_ids"] if trace else None
        generated = backend.generate(prompt, max_new_tokens=4, prefix_ids=prefix_ids)
        prediction = parse_answer(generated["text"], reasoning=False)
        if prediction not in row["choices"]:
            prediction = None
        ids = generated["generated_ids"]
        answer = Answer(prediction, ids[0] if ids else -1, generated["text"], None, {})
        yield result_row(row, answer, reader or backend.checkpoint_id,
                         "native_reasoning" if reasoning else "native_answer",
                         trace=trace["trace"] if trace else None, trace_tokens=trace["trace_tokens"] if trace else 0,
                         answer_generated_ids=ids, answer_tokens=generated["tokens"],
                         truncated=generated["truncated"] or (trace["truncated"] if trace else False),
                         forced_delimiter=trace["forced_delimiter"] if trace else False,
                         natural_continuation=trace["replay_eligible"] if trace else True)
