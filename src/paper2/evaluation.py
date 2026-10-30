"""Test 1 and native-format evaluation; all scoring uses ordinary task targets."""
from dataclasses import asdict
from .prompts import render_prompt


def result_row(row, answer, reader, format_name="immediate", **metadata):
    keep = ("id", "rendering_id", "world_id", "view", "split", "dataset", "cue_condition", "cue_location",
            "cue_answer", "cue_agrees_direct", "cue_agrees_evaluated", "answer", "target_location", "target_scope", "belief_depth",
            "belief_edit_change", "event_rewrite_change", "critical_event_kind", "training_seed", "arm")
    return {**{k: row[k] for k in keep if k in row}, **asdict(answer),
            "reader": reader, "format": format_name, "correct": answer.prediction == row["answer"],
            **metadata}


def evaluate_immediate(backend, rows, reader=None):
    for row in rows:
        prompt = render_prompt(backend.tokenizer, row)
        answer = backend.answer(prompt, row["choices"])
        yield result_row(row, answer, reader or backend.checkpoint_id)


def evaluate_native(backend, rows, reasoning=False, reader=None, native_answer=False, *,
                    completion_protocol="legacy_forced_answer_v1"):
    if completion_protocol not in ("legacy_forced_answer_v1", "natural_answer_cap_v2"):
        raise ValueError("Unknown completion protocol")
    if not reasoning and not native_answer:
        for result in evaluate_immediate(backend, rows, reader):
            yield {**result, 'completion_protocol': completion_protocol}
        return
    from .backend import Answer
    from .training.data import parse_answer
    from .traces import generate_trace
    for row in rows:
        trace = generate_trace(backend, row, source=reader or backend.checkpoint_id,
                               completion_protocol=completion_protocol) if reasoning else None
        if reasoning and completion_protocol == "natural_answer_cap_v2":
            # The answer belongs to the same bounded sample as the trace. A
            # second generation here would rescue missing/incomplete answers.
            ids = trace["answer_generated_ids"]
            answer = Answer(trace["native_prediction"], ids[0] if ids else -1,
                            trace["completion_text"], None, {})
            yield result_row(row, answer, reader or backend.checkpoint_id, "native_reasoning",
                             trace=trace["trace"], trace_tokens=trace["trace_tokens"],
                             trace_token_attribution=trace["trace_token_attribution"],
                             trace_characters=trace["trace_characters"],
                             trace_whitespace_words=trace["trace_whitespace_words"],
                             answer_generated_ids=ids, answer_tokens=len(ids),
                             completion_protocol=completion_protocol,
                             completion_text=trace["completion_text"],
                             completion_generated_ids=trace["completion_generated_ids"],
                             completion_tokens=trace["completion_tokens"],
                             answer_complete=trace["answer_complete"], answer_status=trace["answer_status"],
                             cap_reached=trace["cap_reached"], eos_seen=trace["eos_seen"],
                             cap_failure=trace["native_outcome"]["cap_failure"],
                             invalid_completion=trace["native_outcome"]["invalid_completion"],
                             termination_reason=trace["native_outcome"]["termination_reason"],
                             truncated=trace["truncated"], forced_delimiter=False,
                             natural_continuation=trace["replay_eligible"])
            continue
        prompt = trace["answer_prompt"] if trace else render_prompt(backend.tokenizer, row)
        prefix_ids = trace["answer_prefix_ids"] if trace else None
        generated = backend.generate(prompt, max_new_tokens=4, prefix_ids=prefix_ids)
        prediction = parse_answer(generated["text"], reasoning=False,
                                  completion_protocol=completion_protocol)
        if prediction not in row["choices"]:
            prediction = None
        ids = generated["generated_ids"]
        answer = Answer(prediction, ids[0] if ids else -1, generated["text"], None, {})
        protocol_metadata = ({"completion_protocol": completion_protocol}
                             if completion_protocol != "legacy_forced_answer_v1" else {})
        yield result_row(row, answer, reader or backend.checkpoint_id,
                         "native_reasoning" if reasoning else "native_answer",
                         trace=trace["trace"] if trace else None, trace_tokens=trace["trace_tokens"] if trace else 0,
                         answer_generated_ids=ids, answer_tokens=generated["tokens"],
                         truncated=generated["truncated"] or (trace["truncated"] if trace else False),
                         forced_delimiter=trace["forced_delimiter"] if trace else False,
                         natural_continuation=trace["replay_eligible"] if trace else True,
                         **protocol_metadata)
