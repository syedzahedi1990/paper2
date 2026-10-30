"""Prospective editor diagnostics on deterministic training-only oracle traces.

These are symbolic inputs, not model observations, native completions, or Test 2
sources. Preparing them makes no model call and supplies no human approval.
"""
from copy import deepcopy

from .data.world import StoryWorld, build_oracle_trace, render_record
from .edit_generation import (generation_outcome_report, prepare_generation_plan,
                              validate_generated_batch)
from .io import digest
from .traces import prepare_edit_manifest

PURPOSE = "editor_oracle_calibration"
EDITOR_MODEL = "Qwen/Qwen2.5-7B-Instruct"
EDITOR_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"


def prepare_oracle_calibration(rows, *, model_id=EDITOR_MODEL, revision=EDITOR_REVISION,
                               worlds=12, seed=2026, max_new_tokens=2048):
    """Rank clean training-world IDs before observing any editor output.

    Rows must include the corresponding training-world state. File/corpus hashes
    belong in the caller's exclusive preparation receipt; this payload also binds
    the full validated input rows and every selected source row by content hash.
    No tokenizer, model weights, GPU, or held-out predictions are consulted.
    """
    rows = list(rows)
    if model_id != EDITOR_MODEL or revision != EDITOR_REVISION:
        raise ValueError("Oracle calibration must use the fixed study editor and revision")
    if type(worlds) is not int or worlds < 1 or type(seed) is not int or seed < 0:
        raise ValueError("Calibration needs a positive story count and nonnegative seed")
    if max_new_tokens != 2048:
        raise ValueError("Oracle calibration retains the 2048-token editor budget")
    records = {}
    for row in rows:
        if (row.get("split") != "train" or row.get("view") != "direct"
                or row.get("dataset") != "clean" or row.get("world", {}).get("split") != "train"):
            raise ValueError("Oracle calibration requires clean direct training rows and training worlds")
        world = StoryWorld.from_dict(row["world"])
        canonical = render_record(world, "direct", row.get("cue_condition", "none"),
                                  cue_location=row.get("cue_location"), dataset="clean")
        if any(row.get(field) != canonical[field] for field in (
                "id", "world_id", "story", "question", "answer", "choices")):
            raise ValueError("Calibration training row disagrees with its symbolic world")
        if world.world_id in records:
            raise ValueError("Calibration training worlds must occur exactly once")
        records[world.world_id] = row
    if len(records) < worlds:
        raise ValueError("Training pool is smaller than the preregistered calibration sample")
    selected = sorted(sorted(records, key=lambda key: digest([PURPOSE, seed, key]))[:worlds])
    source, checkpoint = "training_symbolic_oracle", {"id": "paper1_symbolic_oracle_v1", "revision": None}
    traces = []
    for world_id in selected:
        row = records[world_id]
        text = build_oracle_trace(row)
        artifact = digest([source, checkpoint, row["id"], "direct_fixed", digest(text)])
        traces.append({"schema_version": 2, "artifact_id": artifact,
                       "trace_id": digest([artifact, row["id"]]), "source": source,
                       "source_checkpoint": checkpoint, "basis_record_id": row["id"],
                       "record_id": row["id"], "world_id": world_id, "view": "direct",
                       "question_timing": "direct_fixed", "trace": text, "trace_sha256": digest(text),
                       "purpose": PURPOSE, "source_kind": "deterministic_training_oracle",
                       "model_generated": False, "replay_eligible": False})
    manifest = prepare_edit_manifest(traces, [records[key] for key in selected],
                                     max_worlds=worlds, seed=seed, stage="pilot")
    plan = prepare_generation_plan(manifest, model_id=model_id, revision=revision,
                                   seed=seed, max_new_tokens=max_new_tokens,
                                   audit_sample_size=100, audit_seed=seed)
    calibration = {"schema_version": 1, "purpose": PURPOSE,
                   "selection_method": "sha256_training_world_rank_v1", "seed": seed,
                   "training_population_size": len(records), "selected_world_ids": selected,
                   "input_rows_sha256": digest([records[key] for key in sorted(records)]),
                   "selected_row_sha256": {key: digest(records[key]) for key in selected},
                   "traces_sha256": digest(traces), "manifest_sha256": digest(manifest),
                   "generation_plan_sha256": plan["plan_sha256"],
                   "execution": {"attention": "sdpa", "do_sample": False, "seed": seed,
                                 "max_new_tokens": max_new_tokens},
                   "interpretation": "Prospective editor infrastructure diagnostic only. No numerical quality "
                                     "cutoff; machine-valid edits still need semantic review. No study outcome used."}
    calibration["calibration_sha256"] = digest(calibration)
    return {"traces": traces, "manifest": manifest, "generation_plan": plan, "calibration": calibration}


def report_oracle_calibration(preparation, batch):
    """Report every frozen calibration input and operational failure cause."""
    record = preparation["calibration"]
    if record.get("calibration_sha256") != digest({key: value for key, value in record.items()
                                                   if key != "calibration_sha256"}):
        raise ValueError("Oracle calibration preparation changed")
    traces, manifest, plan = (preparation[name] for name in ("traces", "manifest", "generation_plan"))
    if (record.get("purpose") != PURPOSE or digest(traces) != record.get("traces_sha256")
            or digest(manifest) != record.get("manifest_sha256")
            or plan.get("plan_sha256") != record.get("generation_plan_sha256")
            or batch.get("generation_plan") != plan
            or any(trace.get("purpose") != PURPOSE for trace in traces)):
        raise ValueError("Oracle calibration batch is not bound to its frozen inputs")
    contracts = validate_generated_batch(batch)
    report = {"schema_version": 1, "purpose": PURPOSE,
              "calibration_sha256": record["calibration_sha256"],
              "batch_sha256": batch["generated_batch"]["batch_sha256"],
              "outcomes": generation_outcome_report(contracts.values()),
              "items": [{"artifact_id": trace["artifact_id"], "world_id": trace["world_id"],
                         "source_characters": len(trace["trace"]),
                         "source_words": len(trace["trace"].split()),
                         "generation_status": contracts[trace["artifact_id"]]["generation_status"],
                         "failure_code": contracts[trace["artifact_id"]]["generation"].get("failure_code"),
                         "not_editable": deepcopy(contracts[trace["artifact_id"]].get("not_editable"))}
                        for trace in traces],
              "semantic_approval": False, "interpretation": record["interpretation"]}
    report["report_sha256"] = digest(report)
    return report


def _read(path):
    import json
    from pathlib import Path
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _reference(path):
    from pathlib import Path
    from .io import file_digest
    path = Path(path).resolve()
    return {"path": str(path), "sha256": file_digest(path)}


def _write_new(path, value):
    import json
    import os
    from pathlib import Path
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _append(stream, value):
    import json
    import os
    stream.write(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def _training_inputs(training, worlds_path, corpus_manifest):
    from .io import file_digest, read_jsonl
    from .data.corpus import CORPUS_SCHEMA
    inputs = {"training": _reference(training), "worlds": _reference(worlds_path),
              "corpus_manifest": _reference(corpus_manifest)}
    corpus = _read(corpus_manifest)
    expected = digest({key: corpus.get(key) for key in ("schema_version", "config", "files", "shortcut_audit")})
    if (corpus.get("schema_version") != CORPUS_SCHEMA or corpus.get("corpus_hash") != expected
            or corpus.get("config", {}).get("train") != 2000):
        raise ValueError("Calibration requires the frozen 2000-story corpus manifest")
    for name, path in (("training/clean.jsonl", training), ("worlds/train.jsonl", worlds_path)):
        if corpus.get("files", {}).get(name) != {"rows": 2000, "sha256": file_digest(path)}:
            raise ValueError("Calibration input file differs from its frozen corpus binding")
    training_rows, world_rows = read_jsonl(training), read_jsonl(worlds_path)
    if len(training_rows) != 2000 or len(world_rows) != 2000:
        raise ValueError("Calibration needs all 2000 training rows and worlds")
    worlds = {}
    for row in world_rows:
        world = StoryWorld.from_dict(row)
        if world.split != "train" or world.world_id in worlds:
            raise ValueError("Calibration worlds must be distinct training worlds")
        worlds[world.world_id] = row
    if {row.get("world_id") for row in training_rows} != set(worlds):
        raise ValueError("Clean training export and world inventory differ")
    joined = []
    for row in training_rows:
        if "world" in row:
            raise ValueError("Clean training export must retain its original world-free boundary")
        joined.append({**row, "world": worlds[row["world_id"]]})
    # Recheck every input after parsing; a changing file cannot be frozen.
    if any(file_digest(ref["path"]) != ref["sha256"] for ref in inputs.values()):
        raise ValueError("Calibration inputs changed while preparing their bindings")
    return joined, inputs, corpus["corpus_hash"]


def prepare_files(training, worlds_path, corpus_manifest, output):
    """Create a new directory containing one immutable, complete preparation."""
    from pathlib import Path
    from .training.source_identity import current_source_identity
    rows, inputs, corpus_hash = _training_inputs(training, worlds_path, corpus_manifest)
    value = prepare_oracle_calibration(rows, worlds=12, seed=2026)
    value.update(schema_version=1, purpose=PURPOSE, input_files=inputs,
                 corpus_hash=corpus_hash, source_identity=current_source_identity())
    value["preparation_sha256"] = digest(value)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    path = output / "preparation.json"
    _write_new(path, value)
    return {"status": "prepared", "preparation": _reference(path), "planned_calls": 12,
            "max_output_tokens": 12 * 2048, "model_calls": 0,
            "scientific_gate_decision": None, "human_audit": "not_performed"}


def _prepared_file(preparation, expected_sha, *, execution=False):
    from pathlib import Path
    from .io import file_digest
    from .edit_generation import PROMPT_VERSION, _validate_plan
    path = Path(preparation).resolve()
    if file_digest(path) != expected_sha:
        raise ValueError("Calibration preparation file hash differs from the frozen expected hash")
    value = _read(path)
    if (value.get("schema_version") != 1 or value.get("purpose") != PURPOSE
            or value.get("preparation_sha256") != digest({key: item for key, item in value.items()
                                                          if key != "preparation_sha256"})):
        raise ValueError("Calibration preparation content changed")
    refs = value.get("input_files", {})
    if set(refs) != {"training", "worlds", "corpus_manifest"}:
        raise ValueError("Calibration preparation input references are incomplete")
    if any(file_digest(ref["path"]) != ref.get("sha256") for ref in refs.values()):
        raise ValueError("Calibration preparation source input changed")
    rows, inputs, corpus_hash = _training_inputs(*(refs[name]["path"] for name in (
        "training", "worlds", "corpus_manifest")))
    rebuilt = prepare_oracle_calibration(rows, worlds=12, seed=2026)
    if (any(value.get(key) != item for key, item in rebuilt.items())
            or value.get("input_files") != inputs or value.get("corpus_hash") != corpus_hash):
        raise ValueError("Calibration preparation differs from deterministic training-only inputs")
    _validate_plan(value["generation_plan"])
    if value["generation_plan"]["prompt_version"] != PROMPT_VERSION:
        raise ValueError("New calibration execution requires the prospective quotation prompt")
    if execution:
        from .training.source_identity import current_source_identity
        if value.get("source_identity") != current_source_identity():
            raise ValueError("Calibration source changed after preparation; preserve this preparation and freeze a new one")
    return path, value


def _resource_summary(contracts):
    import math
    observed = []
    for contract in contracts:
        generation = contract["generation"]
        measurement = (generation.get("generation_metadata") or {}).get("g1_resource_measurement")
        if measurement is None:
            continue
        if (measurement.get("kind") != "observed_editor_call"
                or measurement.get("attention_implementation") != "sdpa"
                or measurement.get("prompt_sha256") != generation["prompt_sha256"]
                or type(measurement.get("input_tokens")) is not int or measurement["input_tokens"] < 0
                or type(measurement.get("wall_seconds")) not in (int, float)
                or not math.isfinite(measurement["wall_seconds"]) or measurement["wall_seconds"] < 0
                or (measurement.get("output_tokens") is not None
                    and (type(measurement["output_tokens"]) is not int
                         or not 0 <= measurement["output_tokens"] <= 2048))):
            raise ValueError("Invalid measured calibration editor resources")
        observed.append(measurement)
    outputs = [row for row in observed if row["output_tokens"] is not None]
    return {"planned_calls": 12, "completed_contracts": len(contracts),
            "observed_calls_with_timing": len(observed), "calls_without_timing": len(contracts) - len(observed),
            "observed_inference_wall_seconds": sum(row["wall_seconds"] for row in observed) if observed else None,
            "observed_input_tokens": sum(row["input_tokens"] for row in observed) if observed else None,
            "observed_output_tokens": sum(row["output_tokens"] for row in outputs) if outputs else None,
            "calls_without_output_token_count": len(contracts) - len(outputs),
            "measurement_status": "observed_editor_calls" if observed else "not_measured",
            "interpretation": "Observed calls only; excludes model loading, idle/transfer time and human review. "
                              "Unavailable token counts remain unavailable. No rental debit or approval inferred."}


def generate_file(preparation, expected_sha, output, *, device="auto", resume=False, backend=None):
    """Run the fixed twelve calls once, resuming only durable completed records.

    An in-flight interrupted call is explicitly unresolved and cannot be sampled
    again. The preparation owns one permanent output claim, even across different
    attempted output paths; the OS lock prevents concurrent callers.
    """
    import os
    from pathlib import Path
    from .edit_generation import generate_edit_controls
    from .io import read_jsonl
    # Lazy import: preparing and reporting never load torch or model weights.
    from .training.g1_editability import (SDPAEditor, _MeasuredEditor, _JournalOnlyEditor,
                                         _job_lock, _journal_rows, _CallStartWriteFailure)
    prepared_path, prepared = _prepared_file(preparation, expected_sha, execution=True)
    plan, manifest, traces = (prepared[key] for key in ("generation_plan", "manifest", "traces"))
    output = Path(output).resolve()
    journal, started = Path(str(output) + ".journal.jsonl"), Path(str(output) + ".started.jsonl")
    claim = prepared_path.parent / "generation.job.json"
    identity = {"schema_version": 1, "purpose": PURPOSE, "preparation": _reference(prepared_path),
                "plan_sha256": plan["plan_sha256"], "output": str(output), "attention_implementation": "sdpa",
                "planned_calls": 12, "source_identity": prepared["source_identity"]}
    if output.exists():
        raise FileExistsError("Calibration batch already exists; no overwrite or resampling")
    output.parent.mkdir(parents=True, exist_ok=True)
    with _job_lock(prepared_path.parent / "generation.lock"):
        if output.exists():
            raise FileExistsError("Calibration batch already exists; no overwrite or resampling")
        if resume:
            if not all(path.is_file() for path in (journal, started, claim)) or _read(claim) != identity:
                raise ValueError("Resume requires this exact calibration claim and both journals")
        elif any(path.exists() for path in (journal, started, claim)):
            raise ValueError("Calibration already has a generation claim; use explicit resume for its original output")
        completed = _journal_rows(journal, plan)
        starts = read_jsonl(started) if started.exists() else []
        ordered = [contract["artifact_id"] for contract in manifest["edits"]]
        if ([row["artifact_id"] for row in completed] != ordered[:len(completed)]
                or len(starts) != len(completed)):
            raise ValueError("Calibration has an interrupted unresolved call or noncanonical journal; no resampling")
        for begin, contract in zip(starts, completed):
            expected = {"artifact_id": contract["artifact_id"], "prompt_sha256": contract["generation"]["prompt_sha256"],
                        "plan_sha256": plan["plan_sha256"]}
            if begin != expected:
                raise ValueError("Calibration call-start record differs from the completed result")
        if not resume:
            _write_new(claim, identity)
        calls = 0
        with journal.open("a" if resume else "x", encoding="utf-8", newline="\n") as stream, \
                started.open("a" if resume else "x", encoding="utf-8", newline="\n") as begin_stream:
            for handle in (stream, begin_stream):
                handle.flush()
                os.fsync(handle.fileno())
            if len(completed) == 12:
                editor = _JournalOnlyEditor(completed, plan)
            else:
                measured = _MeasuredEditor(backend if backend is not None else SDPAEditor(device=device))

                class ClaimedEditor:
                    checkpoint_id, revision, tokenizer = measured.checkpoint_id, measured.revision, measured.tokenizer

                    def token_ids(self, text):
                        return measured.token_ids(text)

                    @property
                    def last_measurement(self):
                        return measured.last_measurement

                    def generate(self, prompt, **kwargs):
                        try:
                            _append(begin_stream, {"artifact_id": ordered[len(completed) + calls],
                                "prompt_sha256": digest(prompt), "plan_sha256": plan["plan_sha256"]})
                        except Exception as error:
                            # This is a job-persistence failure, not a model-output
                            # outcome. Do not let the generic editor catch it and
                            # continue with later contracts or unrecorded calls.
                            raise _CallStartWriteFailure(str(error)) from error
                        return measured.generate(prompt, **kwargs)

                editor = ClaimedEditor()

            def persist(contract):
                nonlocal calls
                measurement = editor.last_measurement
                if measurement is not None and measurement["prompt_sha256"] == contract["generation"]["prompt_sha256"]:
                    metadata = contract["generation"].get("generation_metadata") or {}
                    metadata["g1_resource_measurement"] = measurement
                    contract["generation"]["generation_metadata"] = metadata
                    contract["generation"]["generation_sha256"] = digest({key: value for key, value
                        in contract["generation"].items() if key != "generation_sha256"})
                _append(stream, contract)
                calls += 1

            generate_edit_controls(editor, traces, manifest, plan, on_result=persist, completed_results=completed)
        durable = _journal_rows(journal, plan)
        expected_starts = [{"artifact_id": contract["artifact_id"],
                            "prompt_sha256": contract["generation"]["prompt_sha256"],
                            "plan_sha256": plan["plan_sha256"]} for contract in durable]
        if len(durable) != 12 or read_jsonl(started) != expected_starts:
            raise ValueError("Calibration call-start evidence is incomplete; preserve both journals without resampling")
        result = generate_edit_controls(_JournalOnlyEditor(durable, plan), traces, manifest, plan,
                                        completed_results=durable)
        _write_new(output, result)
        return {"status": "complete", "new_calls": calls, "generated": _reference(output),
                "journal": _reference(journal), "started": _reference(started),
                "resources": _resource_summary(durable), "scientific_gate_decision": None,
                "human_audit": "not_performed"}


def report_file(preparation, expected_sha, batch, output):
    """Persist a descriptive calibration report; never approve the editor."""
    from pathlib import Path
    _, prepared = _prepared_file(preparation, expected_sha)
    generated = _read(batch)
    value = report_oracle_calibration(prepared, generated)
    value.update(preparation=_reference(preparation), generated_batch=_reference(batch),
                 resources=_resource_summary(generated["edits"]))
    value.pop("report_sha256")
    value["report_sha256"] = digest(value)
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_new(output, value)
    return {"status": "complete", "report": _reference(output), "resources": value["resources"],
            "scientific_gate_decision": None, "human_audit": "not_performed"}
