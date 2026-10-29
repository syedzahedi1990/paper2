"""Frozen source traces, preregistered edits, and a crossed reader experiment.

Readers never generate a supplied trace. Native direct-question traces may be
frozen before later consequence questions; ``prequestion`` is a legacy CLI alias
for that condition, not an instruction to generate a novel story summary.
"""
from copy import deepcopy

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
    """Supply only the queried belief as an oracle-premise positive control.

    This is not the model's own summary. Other physical/character states and the
    per-view answer labels remain metadata, never supplied summary text.
    """
    from .data.world import StoryWorld, VIEWS, answer_label, ordinary_target
    if "world" not in row:
        raise ValueError("Symbolic belief summaries require an evaluation record with its world")
    world = StoryWorld.from_dict(row["world"])
    location = world.state().values[(world.focal_word, world.focal_object)]
    subject = " believes that ".join(world.focal_word)
    return {"text": f"{subject} believes the {world.focal_object} is in the {location}.",
            "method": "symbolic_queried_belief_v2", "world_sha256": digest(row["world"]),
            "task_grounded": True, "focal_word": list(world.focal_word),
            "focal_object": world.focal_object, "location": location,
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


def symbolic_edit_prediction(row, value=None, *, stated_location=None):
    """Propose a focal-belief edit and compute all eight consequences.

    The reviewer still locates and checks the actual trace span. This helper
    never asserts that a free-form trace expresses only the proposed fact.
    """
    from .data.world import StoryWorld, VIEWS, LOCATIONS, answer_label, ordinary_target, rewrite_forecast
    world = StoryWorld.from_dict(row["world"])
    truth_before = ordinary_target(world, "direct")
    before = truth_before if stated_location is None else stated_location
    if before not in LOCATIONS:
        raise ValueError("The trace-stated belief must name a task location")
    value = row["scope_edit_location"] if value is None else value
    if stated_location is not None and value == before:
        # The initial proposed destination can itself be an incorrect stated belief.
        # Freeze a deterministic alternative before observing reader responses.
        value = min((location for location in LOCATIONS if location not in (before, truth_before)),
                    key=lambda location: digest(["trace_edit_target_v2", world.world_id, before, location]))
    if value not in LOCATIONS or value == before or (stated_location is not None and value == truth_before):
        raise ValueError("Symbolic edit must change the focal belief")
    subject = " believes that ".join(world.focal_word)
    premise = f"{subject} believes the {world.focal_object} is in the "
    oracle = {"method": "focal_belief_edit_v1" if stated_location is None else "focal_trace_belief_edit_v2",
              "world_sha256": digest(row["world"]), "word": list(world.focal_word),
              "object": world.focal_object, "before": before, "after": value}
    if stated_location is not None:
        oracle["truth_before"] = truth_before
    prediction = {"fact_before": premise + before, "fact_after": premise + value,
            "expected_answers": {view: answer_label(rewrite_forecast(world, view, value, "belief_edit"))
                                 for view in VIEWS},
            "oracle": oracle}
    if stated_location is not None:
        prediction["expected_before_answers"] = {
            view: answer_label(rewrite_forecast(world, view, before, "belief_edit")) for view in VIEWS}
    return prediction


def _validate_oracle_edit(row, contract):
    semantic = contract["semantic"]
    oracle = semantic.get("oracle")
    if oracle is not None:
        stated = oracle.get("before") if oracle.get("method") == "focal_trace_belief_edit_v2" else None
        expected = symbolic_edit_prediction(row, oracle.get("after"), stated_location=stated)
        if any(semantic.get(key) != value for key, value in expected.items()):
            raise ValueError("Semantic edit differs from its symbolic fact or oracle consequences")
        if stated is not None and semantic.get("stated_location") != stated:
            raise ValueError("Semantic edit differs from the trace-stated location")


def prepare_edit_manifest(traces, rows=None, *, max_worlds=None, seed=2026, stage="pilot"):
    """Freeze a deterministic world subset before reviewing model outputs.

    Selection uses only world IDs and the preregistered seed, not correctness,
    trace content, or observed edit effects. Shared direct traces get one edit
    contract for all their consequence projections.
    """
    traces = list(traces)
    if type(seed) is not int or seed < 0:
        raise ValueError("Edit subset requires a nonnegative integer seed")
    records = {row["id"]: row for row in rows or ()}
    universe = sorted({trace["world_id"] for trace in traces})
    from .precision import edit_precision_plan
    precision = edit_precision_plan(len(universe), stage=stage, max_worlds=max_worlds)
    if stage == "confirmatory" and set(universe) != {row["world_id"] for row in records.values()}:
        raise ValueError("Confirmatory edit planning needs traces covering the complete evaluation world pool")
    if stage == "confirmatory":
        from .data.world import VIEWS
        if any(_timing(trace["question_timing"]) != "direct_fixed" for trace in traces):
            raise ValueError("Confirmatory edits require frozen direct-question traces (direct_fixed)")
        panels, sources = {}, {}
        for row in records.values():
            panels.setdefault(_story_key(row), set()).add(row["view"])
        for trace in traces:
            source_key = (trace["source"], digest(trace["source_checkpoint"]), trace.get("training_seed"))
            sources.setdefault(source_key, set()).add(trace["record_id"])
        if any(views != set(VIEWS) for views in panels.values()) or any(
                record_ids != set(records) for record_ids in sources.values()):
            raise ValueError("Confirmatory edits require complete eight-view panels for every source and world")
    max_worlds = precision["selected_worlds"]
    selected = sorted(sorted(universe, key=lambda world: digest([seed, world]))[:max_worlds])
    selection = {"method": "sha256_seeded_world_rank_v1", "seed": seed, "max_worlds": max_worlds,
                 "universe_world_ids": universe, "selected_world_ids": selected, "precision": precision,
                 "timing_policy": "direct_fixed_confirmatory" if stage == "confirmatory" else "pilot_diagnostic"}
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
        prediction = symbolic_edit_prediction(records[trace["record_id"]]) if trace["record_id"] in records else {}
        if prediction:
            from .data.world import LOCATIONS
            prediction["trace_edit_options"] = {
                location: symbolic_edit_prediction(records[trace["record_id"]], stated_location=location)
                for location in LOCATIONS}
        entries[key] = {"artifact_id": key, "trace_ids": [trace["trace_id"]],
                       "record_ids": [trace["record_id"]], "world_id": trace["world_id"],
                       "source": trace["source"], "source_checkpoint": trace.get("source_checkpoint"),
                       "question_timing": trace["question_timing"], "trace_sha256": trace["trace_sha256"],
                       "reviewed_by": None, "approved": False,
                       "paraphrase": {"text": None, "meaning_preserved": False},
                       "semantic": {"start": None, "end": None, "old": None, "new": None,
                                    "fact_before": None, "fact_after": None, "expected_answers": {},
                                    "only_intended_fact_changed": False, **prediction},
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


def validate_edit(trace, contract, *, batch_review=None):
    if (contract.get("artifact_id", contract.get("trace_id")) != _artifact_id(trace)
            or contract.get("trace_sha256") != digest(trace["trace"])):
        raise ValueError("Edit manifest is not bound to this exact trace")
    if contract.get("source_checkpoint") != trace.get("source_checkpoint"):
        raise ValueError("Edit manifest source checkpoint changed")
    sampled = (batch_review is not None and batch_review.get("bound_contract_hashes", {}).get(
               contract["artifact_id"]) == digest(contract))
    if batch_review is not None and not sampled:
        raise ValueError("Generated edit differs from its sampled-audit binding")
    if not sampled and (not contract.get("approved") or not contract.get("reviewed_by")):
        raise ValueError("Trace semantic controls require a reviewed edit manifest (G3)")
    semantic, irrelevant = contract["semantic"], contract["irrelevant"]
    _replace(trace["trace"], semantic)
    _replace(trace["trace"], irrelevant)
    if (not sampled and not semantic.get("only_intended_fact_changed")) or not semantic.get("expected_answers"):
        raise ValueError("Semantic edit needs an audited intermediate fact and predicted consequences")
    if not semantic.get("fact_before") or not semantic.get("fact_after"):
        raise ValueError("Name the intermediate fact being edited")
    if not sampled and not irrelevant.get("irrelevant_to_answers"):
        raise ValueError("Irrelevant edit must be reviewed")
    if (not sampled and not contract["paraphrase"].get("meaning_preserved")) or not contract["paraphrase"].get("text"):
        raise ValueError("Paraphrase must have reviewed preserved meaning")
    # Preserve a source answer statement, but do not introduce a new one.
    from .edit_generation import introduced_answer_statements
    for text in (contract["paraphrase"]["text"], _replace(trace["trace"], semantic),
                 _replace(trace["trace"], irrelevant)):
        if introduced_answer_statements(trace["trace"], text):
            raise ValueError("New answer statement leakage in an intermediate trace control")


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
    if "precision" in selection:
        from .precision import edit_precision_plan
        plan = selection["precision"]
        expected_plan = edit_precision_plan(len(selection["universe_world_ids"]), stage=plan["stage"],
                        max_worlds=selection["max_worlds"])
        if plan != expected_plan:
            raise ValueError("Edit precision plan was altered")
        if plan["stage"] == "confirmatory" and (selection.get("timing_policy") != "direct_fixed_confirmatory"
                or any(_timing(contract["question_timing"]) != "direct_fixed" for contract in edits["edits"])):
            raise ValueError("Confirmatory edit manifests require the frozen direct-question timing policy")
    result = {}
    for contract in edits["edits"]:
        if contract["artifact_id"] in result:
            raise ValueError("Duplicate edit contract for a source artifact")
        if contract["world_id"] not in selection["selected_world_ids"]:
            raise ValueError("Edit contract is outside the preregistered world subset")
        result[contract["artifact_id"]] = contract
    return result, selection


def _batch_review(edits, audit, *, required=True):
    if edits and "generation_plan" in edits:
        if audit is None:
            if required:
                raise ValueError("Model-generated controls require their completed preregistered sample audit")
            return None
        from .edit_generation import validate_batch_audit
        return validate_batch_audit(edits, audit)
    if audit is not None:
        raise ValueError("Sample audit supplied without a model-generated edit batch")
    return None


def _effective_contract(contract, batch_review):
    if contract is not None and batch_review is not None:
        return batch_review["effective_contracts"][contract["artifact_id"]]
    return contract


def _edit_status(contract, batch_review):
    if contract is None:
        return None, None
    if batch_review is None:
        return ("editable", None) if contract.get("approved") else ("not_assessed", "not_yet_audited")
    status = contract.get("generation_status")
    if status == "ready":
        return "editable", None
    if status == "not_editable":
        return contract["reason_code"], contract.get("reason")
    return "generation_failed", contract.get("reason") or "; ".join(
        contract.get("generation", {}).get("errors", [])) or "Editor output could not be used"


def regenerate_trace_artifacts(backend, rows, traces, source, edits, *, seed=2026, temperature=.7, top_p=.95,
                               edit_audit=None):
    """Generate a continuation ONCE with its source checkpoint, then save it.

    The resulting JSONL is the input for both crossed readers. A reader-side
    continuation, even with the same random seed, is a different intervention.
    """
    records = {row["id"]: row for row in rows}
    contracts, selection = _contracts(edits)
    batch_review = _batch_review(edits, edit_audit)
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
        contract = _effective_contract(contracts.get(artifact), batch_review)
        if contract is None:
            raise ValueError(f"Missing edit contract for selected artifact {artifact}")
        if batch_review and _edit_status(contract, batch_review)[0] != "editable":
            seen.add(artifact)
            continue
        validate_edit(trace, contract, batch_review=batch_review)
        basis = records.get(trace.get("basis_record_id", trace["record_id"]))
        if basis is None:
            raise ValueError("Regeneration requires the original source question record")
        _validate_oracle_edit(basis, contract)
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
                  "edit_audit_sha256": batch_review["audit_sha256"] if batch_review else None,
                  "regeneration": {"truncated": generated["truncated"],
                                   "forced_delimiter": ANSWER_DELIMITER not in generated["text"],
                                   "generated_tokens": generated["tokens"], "regenerated_by": source,
                                   "do_sample": True, "seed": sample_seed, "temperature": temperature,
                                   "top_p": top_p}}
        result["artifact_sha256"] = digest(result)
        seen.add(artifact)
        yield result


def _frozen_regeneration(trace, contract, edits, frozen, batch_review=None):
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
    if frozen.get("edit_audit_sha256") != (batch_review["audit_sha256"] if batch_review else None):
        raise ValueError("Frozen regeneration is bound to a different edit audit")
    if frozen["regeneration"].get("regenerated_by") != trace["source"]:
        raise ValueError("Only the trace source may regenerate an edited continuation")
    return frozen["trace"], {**frozen["regeneration"], "artifact_sha256": frozen["artifact_sha256"]}


def run_crossed_reader(backend, rows, traces, reader, edits=None,
                       conditions=("original", "own_replay", "own_rerendered", "answer_cue", "none"), *,
                       regenerated=None, edit_audit=None):
    """Read identical saved source text in every cell; never generate here."""
    records = {row["id"]: row for row in rows}
    contracts, selection = _contracts(edits)
    batch_review = _batch_review(edits, edit_audit, required=bool(set(conditions) & EDIT_CONTROLS))
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
        initial_contract = contracts.get(artifact_id)
        contract = _effective_contract(initial_contract, batch_review)
        selected = selection is not None and trace["world_id"] in selection["selected_world_ids"]
        edit_status, edit_status_reason = _edit_status(contract, batch_review) if selected else (None, None)
        initial_status, initial_reason = _edit_status(initial_contract, batch_review) if selected else (None, None)
        if selection and trace["world_id"] not in selection["universe_world_ids"]:
            raise ValueError("Reader traces fall outside the preregistered world universe")
        if selected and set(conditions) & EDIT_CONTROLS:
            if contract is None:
                raise ValueError(f"Missing edit contract for selected artifact {artifact_id}")
            if batch_review is None or edit_status == "editable":
                validate_edit(trace, contract, batch_review=batch_review)
                _validate_oracle_edit(row, contract)
        for condition in conditions:
            if condition in EDIT_CONTROLS and not selected:
                continue
            if condition in EDIT_CONTROLS and batch_review and edit_status != "editable":
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
                    text, regeneration = _frozen_regeneration(trace, contract, edits, frozen.get(artifact_id), batch_review)
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
                             edit_status=edit_status, edit_status_reason=edit_status_reason,
                             edit_initial_status=initial_status, edit_initial_status_reason=initial_reason,
                             edit_human_corrected=bool(initial_contract and contract and digest(initial_contract) != digest(contract)),
                             edit_stated_location=contract["semantic"].get("stated_location") if contract else None,
                             edit_source_belief_correct=(contract["semantic"]["oracle"]["before"] ==
                                 contract["semantic"]["oracle"].get("truth_before", contract["semantic"]["oracle"]["before"]))
                                 if contract and edit_status == "editable" and contract["semantic"].get("oracle") else None,
                             edit_review_mode=("sampled_model_generated" if batch_review else "individual_human")
                                 if selected and condition in EDIT_CONTROLS else None,
                             edit_audit_sha256=batch_review["audit_sha256"] if batch_review and selected else None,
                             edit_audit_quality=batch_review["audit_error_rate"]["by_source_checkpoint"][
                                 digest([contract["source"], contract["source_checkpoint"]])]["original_ready"]
                                 if batch_review and selected and contract else None,
                             edit_audit_original_flaw=artifact_id in batch_review["audit_error_rate"]["original_failed_artifact_ids"]
                                 if batch_review and selected else None,
                             edit_length_ratios={
                                 "paraphrase_to_original": len(contract["paraphrase"]["text"]) / max(1, len(trace["trace"])),
                                 "irrelevant_to_semantic": max(len(contract["irrelevant"]["old"]), len(contract["irrelevant"]["new"])) /
                                     max(1, len(contract["semantic"]["old"]), len(contract["semantic"]["new"]))}
                                 if contract and edit_status == "editable" else None,
                             edit_generation_plan_sha256=batch_review["plan_sha256"] if batch_review and selected else None,
                             edit_precision_plan=selection.get("precision") if selected else None,
                             expected_edited_answer=expected if condition in EDIT_CONTROLS else None,
                             expected_before_edit_answer=contract["semantic"].get("expected_before_answers", {}).get(
                                 row["view"], row["answer"]) if condition in EDIT_CONTROLS else None,
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
