"""Frozen source traces, preregistered edits, and a crossed reader experiment.

Readers never generate a supplied trace. Native direct-question traces may be
frozen before later consequence questions; ``prequestion`` is a legacy CLI alias
for that condition, not an instruction to generate a novel story summary.
"""
from copy import deepcopy
import re

from .io import digest
from .prompts import ANSWER_DELIMITER, answer_token_ids, render_prompt
from .evaluation import result_row

CONTROL_NAMES = ("original", "own_replay", "own_rerendered", "paraphrase", "semantic_minimal",
                 "semantic_regenerated", "answer_cue", "belief_summary", "irrelevant", "none")
EDIT_CONTROLS = frozenset(("semantic_minimal", "semantic_regenerated", "paraphrase", "irrelevant"))
TIMINGS = ("question_dependent", "direct_fixed")


def _timing(value):
    value = "direct_fixed" if value == "prequestion" else value
    if value not in TIMINGS:
        raise ValueError("Unknown question timing")
    return value


def _story_key(row):
    return row["world_id"], row.get("cue_condition"), row.get("cue_location")


def _checkpoint(backend):
    return {"id": str(backend.checkpoint_id), "revision": getattr(backend, "revision", None)}


def _artifact_id(trace):
    return trace.get("artifact_id", trace["trace_id"])


def _validate_source_trace(trace):
    if trace.get("schema_version") != 2:
        raise ValueError("Regenerate source traces with schema v2 native-question timing")
    if digest(trace["trace"]) != trace.get("trace_sha256"):
        raise ValueError("Frozen source trace text hash changed")
    expected = digest([trace["source"], trace["source_checkpoint"], trace["basis_record_id"],
                       _timing(trace["question_timing"]), trace["trace_sha256"]])
    if trace.get("artifact_id") != expected or trace["trace_id"] != digest([expected, trace["record_id"]]):
        raise ValueError("Frozen source artifact identity changed")


def symbolic_belief_summary(row):
    """An automatic base-state control, without answer labels or final answers.

    The same state description is supplied across all consequence views. Oracle
    predictions are stored as metadata only; they are never part of the prompt.
    """
    from .data.world import StoryWorld, VIEWS, CONFIG, answer_label, ordinary_target
    if "world" not in row:
        raise ValueError("Symbolic belief summaries require an evaluation record with its world")
    world = StoryWorld.from_dict(row["world"])
    state = world.state()
    sentences = []
    for obj in (world.focal_object, world.distractor_object):
        for word in CONFIG.words():
            location = state.values[(word, obj)]
            if not word:
                sentences.append(f"The {obj} is physically in the {location}.")
            elif len(word) <= len(world.focal_word):
                subject = " believes that ".join(word)
                sentences.append(f"{subject} believes the {obj} is in the {location}.")
    return {"text": " ".join(sentences), "method": "symbolic_base_state_v1",
            "world_sha256": digest(row["world"]), "task_grounded": True,
            "expected_answers": {view: answer_label(ordinary_target(world, view)) for view in VIEWS}}


def generate_trace(backend, row, source, timing="question_dependent"):
    timing = _timing(timing)
    if timing == "direct_fixed" and row["view"] != "direct":
        raise ValueError("A fixed trace must be generated from the native direct question")
    prompt = render_prompt(backend.tokenizer, row, reasoning=True)
    generated = backend.generate(prompt, max_new_tokens=384, stop=ANSWER_DELIMITER)
    text = generated["text"]
    forced = ANSWER_DELIMITER not in text
    trace = text.split(ANSWER_DELIMITER, 1)[0].rstrip()
    rerendered_prompt = render_prompt(backend.tokenizer, row, trace=trace)
    rerendered_ids = backend.token_ids(rerendered_prompt)
    # Exact sampled tokens define the ordinary continuation. Rendering is a
    # separate control: stripping whitespace or BPE boundaries can alter it.
    raw_prefix = generated["prompt_ids"] + generated["generated_ids"]
    exact_text = backend.tokenizer.decode(raw_prefix, skip_special_tokens=False)
    replay_eligible = not forced and exact_text.endswith(ANSWER_DELIMITER)
    answer_prompt = exact_text if replay_eligible else rerendered_prompt
    prefix_ids = raw_prefix if replay_eligible else rerendered_ids
    ordinary_token, ordinary_prediction = None, None
    if replay_eligible:
        continuation = backend.generate(answer_prompt, max_new_tokens=1, prefix_ids=prefix_ids)
        if len(continuation["generated_ids"]) != 1:
            raise ValueError("Ordinary answer continuation must produce exactly one token")
        ordinary_token = continuation["generated_ids"][0]
        alphabet = answer_token_ids(backend.tokenizer, answer_prompt, row["choices"])
        ordinary_prediction = next((label for label, token in alphabet.items() if token == ordinary_token), None)
    checkpoint = _checkpoint(backend)
    artifact = digest([source, checkpoint, row["id"], timing, digest(trace)])
    result = {"schema_version": 2, "artifact_id": artifact,
              "trace_id": digest([artifact, row["id"]]), "source": source,
              "source_checkpoint": checkpoint, "basis_record_id": row["id"],
              "record_id": row["id"], "world_id": row["world_id"], "view": row["view"],
              "cue_condition": row.get("cue_condition"), "cue_location": row.get("cue_location"),
              "question_timing": timing, "trace": trace, "trace_sha256": digest(trace),
              "trace_tokens": len(backend.token_ids(trace)), "truncated": generated["truncated"],
              "generation_prompt": prompt, "generation_prompt_ids": generated["prompt_ids"],
              "generated_trace_ids": generated["generated_ids"],
              "forced_delimiter": forced, "answer_prompt": answer_prompt, "answer_prefix_ids": prefix_ids,
              "rerendered_answer_prompt": rerendered_prompt, "rerendered_answer_prefix_ids": rerendered_ids,
              "replay_token_ids_match": raw_prefix == rerendered_ids if replay_eligible else None,
              "replay_eligible": replay_eligible,
              "replay_ineligibility_reason": None if replay_eligible else (
                  "forced_delimiter" if forced else "delimiter_not_at_exact_generation_boundary"),
              "ordinary_answer_token": ordinary_token, "ordinary_prediction": ordinary_prediction}
    if "world" in row:
        result["belief_summary"] = symbolic_belief_summary(row)
    return result


def collect_traces(backend, rows, source, timing="question_dependent"):
    rows, timing = list(rows), _timing(timing)
    shared = {}
    direct = {_story_key(row): row for row in rows if row["view"] == "direct"}
    for row in rows:
        if timing == "question_dependent":
            yield generate_trace(backend, row, source, timing)
            continue
        key = _story_key(row)
        if key not in direct:
            raise ValueError("Fixed direct traces require the matching direct-view evaluation record")
        basis = direct[key]
        if row.get("base_story", basis["story"]) != basis["story"]:
            raise ValueError("Fixed direct views changed the underlying rendered base story")
        if key not in shared:
            shared[key] = generate_trace(backend, basis, source, timing)
            shared[key]["base_story_sha256"] = digest(basis["story"])
        trace = deepcopy(shared[key])
        trace.update(record_id=row["id"], view=row["view"], trace_id=digest([trace["artifact_id"], row["id"]]))
        if row["id"] != basis["id"]:
            trace["answer_prompt"] = render_prompt(backend.tokenizer, row, trace=trace["trace"])
            trace["answer_prefix_ids"] = backend.token_ids(trace["answer_prompt"])
            trace["rerendered_answer_prompt"] = trace["answer_prompt"]
            trace["rerendered_answer_prefix_ids"] = trace["answer_prefix_ids"]
            trace["ordinary_answer_token"] = None
            trace["ordinary_prediction"] = None
            trace["replay_eligible"] = False
            trace["replay_token_ids_match"] = None
            trace["replay_ineligibility_reason"] = "different_question_from_native_generation"
        yield trace


def prepare_edit_manifest(traces, rows=None, *, max_worlds=12, seed=2026):
    """Freeze a deterministic world subset before reviewing model outputs.

    Selection uses only world IDs and the preregistered seed, not correctness,
    trace content, or observed edit effects. Shared direct traces get one edit
    contract for all their consequence projections.
    """
    traces = list(traces)
    if type(max_worlds) is not int or max_worlds < 1 or type(seed) is not int or seed < 0:
        raise ValueError("Edit subset requires positive max_worlds and a nonnegative integer seed")
    records = {row["id"]: row for row in rows or ()}
    universe = sorted({trace["world_id"] for trace in traces})
    selected = sorted(sorted(universe, key=lambda world: digest([seed, world]))[:max_worlds])
    selection = {"method": "sha256_seeded_world_rank_v1", "seed": seed, "max_worlds": max_worlds,
                 "universe_world_ids": universe, "selected_world_ids": selected}
    entries = {}
    for trace in traces:
        _validate_source_trace(trace)
        if trace["world_id"] not in selected:
            continue
        key = _artifact_id(trace)
        if key in entries:
            if entries[key]["trace_sha256"] != trace["trace_sha256"]:
                raise ValueError("A shared artifact has inconsistent trace text")
            entries[key]["trace_ids"].append(trace["trace_id"])
            entries[key]["record_ids"].append(trace["record_id"])
            continue
        summary = trace.get("belief_summary")
        if not summary and trace["record_id"] in records:
            summary = symbolic_belief_summary(records[trace["record_id"]])
        entries[key] = {"artifact_id": key, "trace_ids": [trace["trace_id"]],
                       "record_ids": [trace["record_id"]], "world_id": trace["world_id"],
                       "source": trace["source"], "source_checkpoint": trace.get("source_checkpoint"),
                       "question_timing": trace["question_timing"], "trace_sha256": trace["trace_sha256"],
                       "reviewed_by": None, "approved": False,
                       "paraphrase": {"text": None, "meaning_preserved": False},
                       "semantic": {"start": None, "end": None, "old": None, "new": None,
                                    "fact_before": None, "fact_after": None, "expected_answers": {},
                                    "only_intended_fact_changed": False},
                       "irrelevant": {"start": None, "end": None, "old": None, "new": None,
                                      "irrelevant_to_answers": False},
                       "belief_summary": summary}
    for entry in entries.values():
        entry["trace_ids"].sort()
        entry["record_ids"].sort()
    return {"schema_version": 2, "selection": selection, "selection_sha256": digest(selection),
            "edits": [entries[key] for key in sorted(entries)]}


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
    if (contract.get("artifact_id", contract.get("trace_id")) != _artifact_id(trace)
            or contract.get("trace_sha256") != digest(trace["trace"])):
        raise ValueError("Edit manifest is not bound to this exact trace")
    if contract.get("source_checkpoint") != trace.get("source_checkpoint"):
        raise ValueError("Edit manifest source checkpoint changed")
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
    for text in (contract["paraphrase"]["text"], semantic["new"], irrelevant["new"]):
        if re.search(r"(?:^|\n)Answer\s*:", text, flags=re.I):
            raise ValueError("Answer delimiter leakage in an intermediate trace control")


def _contracts(edits):
    if not edits:
        return {}, None
    selection = edits.get("selection")
    if edits.get("schema_version") != 2 or not selection or digest(selection) != edits.get("selection_sha256"):
        raise ValueError("Edits require a frozen schema-v2 subset manifest")
    expected = sorted(sorted(selection["universe_world_ids"],
                             key=lambda world: digest([selection["seed"], world]))[:selection["max_worlds"]])
    if expected != selection["selected_world_ids"]:
        raise ValueError("Edit subset differs from its preregistered deterministic selection")
    result = {}
    for contract in edits["edits"]:
        if contract["artifact_id"] in result:
            raise ValueError("Duplicate edit contract for a source artifact")
        if contract["world_id"] not in selection["selected_world_ids"]:
            raise ValueError("Edit contract is outside the preregistered world subset")
        result[contract["artifact_id"]] = contract
    return result, selection


def regenerate_trace_artifacts(backend, rows, traces, source, edits, *, seed=2026, temperature=.7, top_p=.95):
    """Generate a continuation ONCE with its source checkpoint, then save it.

    The resulting JSONL is the input for both crossed readers. A reader-side
    continuation, even with the same random seed, is a different intervention.
    """
    records = {row["id"]: row for row in rows}
    contracts, selection = _contracts(edits)
    if selection is None:
        raise ValueError("Source regeneration requires a preregistered edit manifest")
    seen = set()
    for trace in traces:
        _validate_source_trace(trace)
        artifact = _artifact_id(trace)
        if trace["source"] != source or trace["world_id"] not in selection["selected_world_ids"]:
            continue
        if artifact in seen:
            continue
        if trace.get("source_checkpoint") != _checkpoint(backend):
            raise ValueError("Regeneration backend is not the checkpoint that produced the source trace")
        contract = contracts.get(artifact)
        if contract is None:
            raise ValueError(f"Missing edit contract for selected artifact {artifact}")
        validate_edit(trace, contract)
        basis = records.get(trace.get("basis_record_id", trace["record_id"]))
        if basis is None:
            raise ValueError("Regeneration requires the original source question record")
        native_prompt = render_prompt(backend.tokenizer, basis, reasoning=True)
        if native_prompt != trace.get("generation_prompt"):
            raise ValueError("Regeneration source prompt changed from the frozen trace artifact")
        edit = contract["semantic"]
        edited_prefix = trace["trace"][:edit["start"]] + edit["new"]
        remaining = 384 - len(backend.token_ids(edited_prefix))
        if remaining < 1:
            raise ValueError("Edited prefix exhausts the 384-token trace budget")
        sample_seed = int(digest([seed, artifact, digest(contract)])[:15], 16)
        generated = backend.generate(native_prompt + edited_prefix, max_new_tokens=remaining,
                                     stop=ANSWER_DELIMITER, do_sample=True, seed=sample_seed,
                                     temperature=temperature, top_p=top_p)
        text = edited_prefix + generated["text"].split(ANSWER_DELIMITER, 1)[0].rstrip()
        result = {"schema_version": 2, "artifact_id": artifact, "source": source,
                  "source_checkpoint": trace["source_checkpoint"], "basis_record_id": basis["id"],
                  "source_trace_sha256": trace["trace_sha256"], "edit_contract_sha256": digest(contract),
                  "selection_sha256": edits["selection_sha256"], "trace": text, "trace_sha256": digest(text),
                  "regeneration": {"truncated": generated["truncated"],
                                   "forced_delimiter": ANSWER_DELIMITER not in generated["text"],
                                   "generated_tokens": generated["tokens"], "regenerated_by": source,
                                   "do_sample": True, "seed": sample_seed, "temperature": temperature,
                                   "top_p": top_p}}
        result["artifact_sha256"] = digest(result)
        seen.add(artifact)
        yield result


def _frozen_regeneration(trace, contract, edits, frozen):
    if frozen is None:
        raise ValueError("Missing frozen source-regenerated artifact; run regenerate-traces before either reader")
    payload = {key: value for key, value in frozen.items() if key != "artifact_sha256"}
    if digest(payload) != frozen.get("artifact_sha256") or digest(frozen["trace"]) != frozen.get("trace_sha256"):
        raise ValueError("Frozen regeneration artifact hash changed")
    expected = {"artifact_id": _artifact_id(trace), "source": trace["source"],
                "source_checkpoint": trace.get("source_checkpoint"), "source_trace_sha256": trace["trace_sha256"],
                "edit_contract_sha256": digest(contract), "selection_sha256": edits["selection_sha256"]}
    if any(frozen.get(key) != value for key, value in expected.items()):
        raise ValueError("Frozen regeneration is not bound to this source, trace, edit, and subset")
    if frozen["regeneration"].get("regenerated_by") != trace["source"]:
        raise ValueError("Only the trace source may regenerate an edited continuation")
    return frozen["trace"], {**frozen["regeneration"], "artifact_sha256": frozen["artifact_sha256"]}


def run_crossed_reader(backend, rows, traces, reader, edits=None,
                       conditions=("original", "own_replay", "own_rerendered", "answer_cue", "none"), *,
                       regenerated=None):
    """Read identical saved source text in every cell; never generate here."""
    records = {row["id"]: row for row in rows}
    contracts, selection = _contracts(edits)
    if set(conditions) - set(CONTROL_NAMES):
        raise ValueError("Unknown trace control")
    if set(conditions) & EDIT_CONTROLS and selection is None:
        raise ValueError("Edited controls require a preregistered edit subset")
    frozen = {}
    for artifact in regenerated or ():
        key = artifact["artifact_id"]
        if key in frozen:
            raise ValueError("Duplicate frozen regeneration artifact")
        frozen[key] = artifact
    fixed_direct = {}
    for trace in traces:
        _validate_source_trace(trace)
        row = records[trace["record_id"]]
        if trace["world_id"] != row["world_id"] or trace["view"] != row["view"]:
            raise ValueError("Source trace is attached to a different evaluation world or view")
        artifact_id = _artifact_id(trace)
        contract = contracts.get(artifact_id)
        selected = selection is not None and trace["world_id"] in selection["selected_world_ids"]
        if selection and trace["world_id"] not in selection["universe_world_ids"]:
            raise ValueError("Reader traces fall outside the preregistered world universe")
        if selected and set(conditions) & EDIT_CONTROLS:
            if contract is None:
                raise ValueError(f"Missing edit contract for selected artifact {artifact_id}")
            validate_edit(trace, contract)
        for condition in conditions:
            if condition in EDIT_CONTROLS and not selected:
                continue
            replay = condition in ("own_replay", "own_rerendered")
            if replay and (reader != trace["source"] or row["id"] != trace.get("basis_record_id", row["id"])):
                continue
            if replay and trace.get("source_checkpoint") != _checkpoint(backend):
                raise ValueError("Own-trace replay requires the original source checkpoint")
            text, prefix_ids, expected, regeneration = trace["trace"], None, None, None
            summary = None
            if condition == "none":
                text = None
            elif condition == "answer_cue":
                text = f"The evaluated answer is {row['answer']}."
            elif condition == "paraphrase":
                text = contract["paraphrase"]["text"]
            elif condition == "belief_summary":
                summary = symbolic_belief_summary(row)
                text = summary["text"]
                expected = summary["expected_answers"][row["view"]]
                if expected != row["answer"]:
                    raise ValueError("Symbolic summary predictions disagree with the evaluation target")
            elif condition == "irrelevant":
                text = _replace(text, contract["irrelevant"])
            elif condition in ("semantic_minimal", "semantic_regenerated"):
                expected = contract["semantic"]["expected_answers"].get(row["view"])
                if expected not in row["choices"]:
                    raise ValueError("Semantic edit must predict the evaluated view")
                text = _replace(text, contract["semantic"])
                if condition == "semantic_regenerated":
                    text, regeneration = _frozen_regeneration(trace, contract, edits, frozen.get(artifact_id))
            if _timing(trace["question_timing"]) == "direct_fixed" and condition not in ("answer_cue", "none"):
                fixed_key = (trace["source"], *_story_key(row), condition)
                if fixed_key in fixed_direct and fixed_direct[fixed_key] != text:
                    raise ValueError("Fixed direct trace must remain fixed across consequence views")
                fixed_direct[fixed_key] = text
            prompt = render_prompt(backend.tokenizer, row, trace=text)
            if condition == "own_replay":
                prompt, prefix_ids = trace["answer_prompt"], trace["answer_prefix_ids"]
            elif condition == "own_rerendered":
                # Intentionally reconstruct through the ordinary reader formatter.
                prefix_ids = backend.token_ids(prompt)
            answer = backend.answer(prompt, row["choices"], prefix_ids=prefix_ids)
            matched = (answer.token_id == trace["ordinary_answer_token"]) if replay and trace.get("replay_eligible") else None
            yield result_row(row, answer, reader, "trace_reader", source=trace["source"], condition=condition,
                             source_checkpoint=trace.get("source_checkpoint"), source_training_seed=trace.get("training_seed"),
                             question_timing=_timing(trace["question_timing"]), trace_id=trace["trace_id"],
                             artifact_id=artifact_id, source_trace_sha256=trace["trace_sha256"],
                             edit_subset_sha256=edits["selection_sha256"] if selected else None,
                             edit_subset_selected=selected,
                             expected_edited_answer=expected if condition in EDIT_CONTROLS else None,
                             expected_summary_answer=expected if summary else None,
                             follows_edit=(answer.prediction == expected) if condition in EDIT_CONTROLS and expected else None,
                             replay_eligible=trace.get("replay_eligible", False),
                             source_forced_delimiter=trace["forced_delimiter"],
                             replay_ineligibility_reason=trace.get("replay_ineligibility_reason"),
                             replay_mode=("raw_token_ids" if condition == "own_replay" else "rerendered_prompt") if replay else None,
                             replay_token_ids_match=trace.get("replay_token_ids_match") if replay else None,
                             ordinary_answer_token=trace.get("ordinary_answer_token") if replay else None,
                             replay_matches=matched,
                             raw_replay_matches=matched if condition == "own_replay" else None,
                             rerendered_replay_matches=matched if condition == "own_rerendered" else None,
                             regeneration=regeneration, symbolic_summary=summary,
                             supplied_trace=text, trace_sha256=digest(text),
                             trace_tokens=len(backend.token_ids(text)) if text is not None else 0)
