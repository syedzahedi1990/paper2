"""Crossed trace-source/reader test with explicit, auditable edit contracts."""
from copy import deepcopy
import re
from .io import digest
from .prompts import ANSWER_DELIMITER, answer_token_ids, render_prompt, prequestion_prompt
from .evaluation import result_row

CONTROL_NAMES = ("original", "own_replay", "paraphrase", "semantic_minimal", "semantic_regenerated",
                 "answer_cue", "belief_summary", "irrelevant", "none")


def generate_trace(backend, row, source, timing="question_dependent"):
    if timing not in ("question_dependent", "prequestion"):
        raise ValueError("Unknown question timing")
    prompt = (render_prompt(backend.tokenizer, row, reasoning=True) if timing == "question_dependent"
              else prequestion_prompt(backend.tokenizer, row))
    stop = ANSWER_DELIMITER if timing == "question_dependent" else "<END_SUMMARY>"
    generated = backend.generate(prompt, max_new_tokens=384, stop=stop)
    text = generated["text"]
    forced = stop not in text
    trace = text.split(stop, 1)[0].rstrip()
    answer_prompt = render_prompt(backend.tokenizer, row, trace=trace)
    # For ordinary question-dependent traces, verify exact token replay. Decode/re-encode can
    # change BPE boundaries, so the original token prefix remains the replay authority.
    raw_prefix = generated["prompt_ids"] + generated["generated_ids"]
    exact_text = backend.tokenizer.decode(raw_prefix, skip_special_tokens=False)
    replay_eligible = timing == "question_dependent" and not forced and exact_text.endswith(ANSWER_DELIMITER)
    if replay_eligible:
        answer_prompt = exact_text
        prefix_ids = raw_prefix
    else:
        prefix_ids = backend.token_ids(answer_prompt)
    # A cap-inserted/reconstructed delimiter is a separate inference condition,
    # not evidence that an ordinary sampled continuation was reproduced.
    ordinary_token, ordinary_prediction = None, None
    if replay_eligible:
        continuation = backend.generate(answer_prompt, max_new_tokens=1, prefix_ids=prefix_ids)
        if len(continuation["generated_ids"]) != 1:
            raise ValueError("Ordinary answer continuation must produce exactly one token")
        ordinary_token = continuation["generated_ids"][0]
        alphabet = answer_token_ids(backend.tokenizer, answer_prompt, row["choices"])
        ordinary_prediction = next((label for label, token in alphabet.items() if token == ordinary_token), None)
    return {"schema_version": 1, "trace_id": digest([source, row["id"], timing]), "source": source,
            "record_id": row["id"], "world_id": row["world_id"], "view": row["view"],
            "cue_condition": row.get("cue_condition"), "cue_location": row.get("cue_location"),
            "question_timing": timing, "trace": trace, "trace_sha256": digest(trace),
            "trace_tokens": len(backend.token_ids(trace)), "truncated": generated["truncated"],
            "forced_delimiter": forced, "answer_prompt": answer_prompt, "answer_prefix_ids": prefix_ids,
            "replay_eligible": replay_eligible,
            "replay_ineligibility_reason": None if replay_eligible else (
                "question_not_yet_revealed" if timing == "prequestion" else
                "forced_delimiter" if forced else "delimiter_not_at_exact_generation_boundary"),
            "ordinary_answer_token": ordinary_token, "ordinary_prediction": ordinary_prediction}


def collect_traces(backend, rows, source, timing="question_dependent"):
    rows = list(rows)
    shared = {}
    direct = {(r["world_id"], r.get("cue_condition"), r.get("cue_location")): r["story"]
              for r in rows if r["view"] == "direct"}
    for row in rows:
        # One fixed pre-question trace for all consequences of the SAME rendered base story.
        # Consequence rows may add query events: use base_story when provided.
        key = (row["world_id"], row.get("cue_condition"), row.get("cue_location"))
        if timing == "prequestion":
            base_story = row.get("base_story", direct.get(key))
            if base_story is None:
                raise ValueError("Prequestion traces require base_story or a matching direct-view record")
            if key not in shared:
                basis = {**row, "story": base_story}
                shared[key] = generate_trace(backend, basis, source, timing)
                shared[key]["base_story_sha256"] = digest(base_story)
            elif shared[key]["base_story_sha256"] != digest(base_story):
                raise ValueError("Prequestion views changed the underlying rendered base story")
            trace = deepcopy(shared[key])
            trace.update(record_id=row["id"], view=row["view"],
                         trace_id=digest([source, row["id"], timing]))
            trace["answer_prompt"] = render_prompt(backend.tokenizer, row, trace=trace["trace"])
            trace["answer_prefix_ids"] = backend.token_ids(trace["answer_prompt"])
            # No claim of own ordinary continuation exists before a question is revealed.
            trace["ordinary_answer_token"] = None
            trace["ordinary_prediction"] = None
            yield trace
        else:
            yield generate_trace(backend, row, source, timing)


def prepare_edit_manifest(traces):
    """Prepare reviewable contracts; semantic validity requires a named human review."""
    return {"schema_version": 1, "edits": [
        {"trace_id": trace["trace_id"], "trace_sha256": trace["trace_sha256"],
         "reviewed_by": None, "approved": False,
         "paraphrase": {"text": None, "meaning_preserved": False},
         "semantic": {"start": None, "end": None, "old": None, "new": None,
                      "fact_before": None, "fact_after": None, "expected_answers": {},
                      "only_intended_fact_changed": False},
         "irrelevant": {"start": None, "end": None, "old": None, "new": None,
                        "irrelevant_to_answers": False},
         "belief_summary": {"text": None, "task_grounded": False}}
        for trace in traces]}


def _replace(trace, edit):
    start, end = edit.get("start"), edit.get("end")
    if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start < end <= len(trace):
        raise ValueError("Edit needs valid character offsets")
    if trace[start:end] != edit.get("old") or not isinstance(edit.get("new"), str):
        raise ValueError("Edit span does not match source text")
    if edit["old"] == edit["new"]:
        raise ValueError("An edit must change the selected text")
    return trace[:start] + edit["new"] + trace[end:]


def validate_edit(trace, contract):
    if contract.get("trace_id") != trace["trace_id"] or contract.get("trace_sha256") != digest(trace["trace"]):
        raise ValueError("Edit manifest is not bound to this exact trace")
    if not contract.get("approved") or not contract.get("reviewed_by"):
        raise ValueError("Trace semantic controls require a reviewed edit manifest (G3)")
    semantic, irrelevant = contract["semantic"], contract["irrelevant"]
    _replace(trace["trace"], semantic)
    _replace(trace["trace"], irrelevant)
    if not semantic.get("only_intended_fact_changed") or not semantic.get("expected_answers"):
        raise ValueError("Semantic edit needs an audited intermediate fact and predicted consequences")
    if not semantic.get("fact_before") or not semantic.get("fact_after"):
        raise ValueError("Name the intermediate fact being edited")
    if not irrelevant.get("irrelevant_to_answers"):
        raise ValueError("Irrelevant edit must be reviewed")
    sem_size = max(len(semantic["old"]), len(semantic["new"]))
    irr_size = max(len(irrelevant["old"]), len(irrelevant["new"]))
    if not .5 <= irr_size / max(sem_size, 1) <= 2:
        raise ValueError("Irrelevant edit must be comparable in text length (0.5–2x)")
    if not contract["paraphrase"].get("meaning_preserved") or not contract["paraphrase"].get("text"):
        raise ValueError("Paraphrase must have reviewed preserved meaning")
    if not contract["belief_summary"].get("task_grounded") or not contract["belief_summary"].get("text"):
        raise ValueError("Belief summary must be checked against the task state")
    for text in [contract["paraphrase"]["text"], contract["belief_summary"]["text"], semantic["new"]]:
        if re.search(r"(?:^|\n)Answer\s*:", text, flags=re.I):
            raise ValueError("Answer delimiter leakage in an intermediate trace control")


def run_crossed_reader(backend, rows, traces, reader, edits=None,
                       conditions=("original", "own_replay", "answer_cue", "none"), *,
                       regeneration_seed=2026, regeneration_temperature=.7, regeneration_top_p=.95):
    """Run once per reader, always on both sources; minimal/regenerated edits run side by side."""
    records = {r["id"]: r for r in rows}
    contracts = {e["trace_id"]: e for e in (edits or {}).get("edits", [])}
    if set(conditions) - set(CONTROL_NAMES):
        raise ValueError("Unknown trace control")
    needs_edits = bool(set(conditions) & {"semantic_minimal", "semantic_regenerated", "paraphrase", "irrelevant", "belief_summary"})
    regenerated_cache, fixed_prequestion = {}, {}
    for trace in traces:
        row = records[trace["record_id"]]
        contract = contracts.get(trace["trace_id"])
        if needs_edits:
            if contract is None:
                raise ValueError(f"Missing edit contract for {trace['trace_id']}")
            validate_edit(trace, contract)
        for condition in conditions:
            if condition == "own_replay" and (reader != trace["source"] or trace["question_timing"] == "prequestion"):
                continue
            text, prefix_ids = trace["trace"], None
            expected = None
            regeneration = None
            if condition == "none":
                text = None
            elif condition == "answer_cue":
                text = f"The evaluated answer is {row['answer']}."
            elif condition == "paraphrase":
                text = contract["paraphrase"]["text"]
            elif condition == "belief_summary":
                text = contract["belief_summary"]["text"]
            elif condition == "irrelevant":
                text = _replace(text, contract["irrelevant"])
            elif condition in ("semantic_minimal", "semantic_regenerated"):
                edit = contract["semantic"]
                expected = edit["expected_answers"].get(row["view"])
                if expected not in row["choices"]:
                    raise ValueError("Semantic edit must predict the evaluated view")
                text = _replace(text, edit)
                if condition == "semantic_regenerated":
                    edited_prefix = trace["trace"][:edit["start"]] + edit["new"]
                    if trace["question_timing"] == "prequestion":
                        if "base_story" not in row:
                            raise ValueError("Prequestion regeneration requires an explicit base_story")
                        basis = {**row, "story": row["base_story"]}
                        prompt = prequestion_prompt(backend.tokenizer, basis) + edited_prefix
                        stop = "<END_SUMMARY>"
                    else:
                        prompt = render_prompt(backend.tokenizer, row, reasoning=True) + edited_prefix
                        stop = ANSWER_DELIMITER
                    remaining = 384 - len(backend.token_ids(edited_prefix))
                    if remaining < 1:
                        raise ValueError("Edited prefix exhausts the 384-token trace budget")
                    cache_key = digest([trace["source"], trace["world_id"], trace.get("cue_condition"),
                                        trace.get("cue_location"), prompt, stop])
                    if cache_key not in regenerated_cache:
                        sample_seed = int(digest([regeneration_seed, cache_key])[:15], 16)
                        generated = backend.generate(prompt, max_new_tokens=remaining, stop=stop,
                                                     do_sample=True, seed=sample_seed,
                                                     temperature=regeneration_temperature, top_p=regeneration_top_p)
                        regenerated_cache[cache_key] = (edited_prefix + generated["text"].split(stop, 1)[0],
                                                       {"truncated": generated["truncated"],
                                                        "forced_delimiter": stop not in generated["text"],
                                                        "generated_tokens": generated["tokens"],
                                                        "regenerated_by": reader,
                                                        "do_sample": True, "seed": sample_seed,
                                                        "temperature": regeneration_temperature,
                                                        "top_p": regeneration_top_p})
                    text, regeneration = regenerated_cache[cache_key]
            if trace["question_timing"] == "prequestion" and condition not in ("answer_cue", "none"):
                fixed_key = (trace["source"], row["world_id"], row.get("cue_condition"),
                             row.get("cue_location"), condition)
                if fixed_key in fixed_prequestion and fixed_prequestion[fixed_key] != text:
                    raise ValueError("Prequestion supplied trace must remain fixed across consequence views")
                fixed_prequestion[fixed_key] = text
            prompt = render_prompt(backend.tokenizer, row, trace=text)
            if condition == "own_replay":
                prompt = trace["answer_prompt"]
                prefix_ids = trace["answer_prefix_ids"]
            answer = backend.answer(prompt, row["choices"], prefix_ids=prefix_ids)
            yield result_row(row, answer, reader, "trace_reader", source=trace["source"], condition=condition,
                             question_timing=trace["question_timing"], trace_id=trace["trace_id"],
                             expected_edited_answer=expected,
                             follows_edit=(answer.prediction == expected) if expected else None,
                             replay_eligible=trace.get("replay_eligible", False),
                             source_forced_delimiter=trace["forced_delimiter"],
                             replay_ineligibility_reason=trace.get("replay_ineligibility_reason"),
                             replay_matches=(answer.token_id == trace["ordinary_answer_token"])
                             if condition == "own_replay" and trace.get("replay_eligible", False) else None,
                             regeneration=regeneration,
                             supplied_trace=text, trace_sha256=digest(text),
                             trace_tokens=len(backend.token_ids(text)) if text is not None else 0)
