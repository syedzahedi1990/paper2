"""Composable experiment commands. No command provisions or purchases compute."""
import argparse
from dataclasses import replace
from pathlib import Path
import json
import sys
from .io import read_jsonl, write_jsonl, write_json, file_digest


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _backend(args):
    from .backend import HFBackend
    return HFBackend(args.model, args.adapter, device=args.device, revision=args.revision)


def _model_args(parser):
    parser.add_argument("--model", default="mistralai/Mistral-Small-24B-Instruct-2501")
    parser.add_argument("--revision", help="Defaults to the proposal model's pinned immutable revision")
    parser.add_argument("--adapter")
    parser.add_argument("--device", default="auto")


def parser():
    p = argparse.ArgumentParser(prog="paper2", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate", help="Generate and hash the controlled corpus")
    gen.add_argument("--output", required=True)
    gen.add_argument("--seed", type=int, default=20261102)
    gen.add_argument("--train", type=int, default=2000)
    gen.add_argument("--development", type=int, default=120)
    gen.add_argument("--confirmation", type=int, default=120)
    gen.add_argument("--distractor-events", type=int, default=2)
    gen.add_argument("--hidden-move", action="store_true", help="Development calibration difficulty variant")
    gen.add_argument("--depths", type=int, nargs="+", default=[1, 2])
    verify = sub.add_parser("verify-data")
    verify.add_argument("--data", required=True)
    shortcuts = sub.add_parser("audit-shortcuts")
    shortcuts.add_argument("--data", required=True)
    shortcuts.add_argument("--output", required=True)
    audit = sub.add_parser("audit-tokenizer")
    audit.add_argument("--input", nargs="+", required=True)
    audit.add_argument("--model", default="mistralai/Mistral-Small-24B-Instruct-2501")
    audit.add_argument("--revision", help="Defaults to the supported model's immutable pinned revision")
    audit.add_argument("--output", required=True)
    matrix = sub.add_parser("matrix")
    matrix.add_argument("--stage", choices=["pilot", "confirmatory"], required=True)
    matrix.add_argument("--output", required=True)
    matrix.add_argument("--development-panel", action="store_true", help="Export the declared pilot candidates without running them")
    matrix.add_argument("--design", choices=["amended_20261030", "legacy_proposal"], default="amended_20261030")
    failure_policy = sub.add_parser("freeze-selection-failure-policy", help="Freeze failure treatment before observing a failed candidate")
    for flag in ("output", "reviewer", "rationale"):
        failure_policy.add_argument("--" + flag, required=True)
    select = sub.add_parser("select-panel", help="Compute final-native family winners from all twelve terminal attempts")
    for flag in ("execution-binding", "sha256", "panel-status", "failure-policy", "failure-policy-sha256", "output", "reviewer", "rationale"):
        select.add_argument("--" + flag, required=True)
    verify_selection = sub.add_parser("verify-panel-selection", help="Recompute a frozen selection against actual saved outcomes")
    verify_selection.add_argument("--selection", required=True)
    verify_selection.add_argument("--sha256", required=True)
    selected_configs = sub.add_parser("selected-confirmatory-configs", help="Export configurations bound to computed family winners; never launch")
    selected_configs.add_argument("--selection", required=True)
    selected_configs.add_argument("--selection-sha256", required=True)
    selected_configs.add_argument("--output", required=True)
    gates = sub.add_parser("init-gates")
    gates.add_argument("--output", required=True)
    g1 = sub.add_parser("verify-g1", help="Summarize complete all-arm learning evidence without approving the gate")
    g1.add_argument("--review", required=True)
    g1.add_argument("--output", required=True)
    g1.add_argument("--historical", action="store_true", help="Inspect historical evidence; cannot approve the current G1 gate")
    health = sub.add_parser("trace-health", help="Create additive length diagnostics from saved predictions without generation")
    health.add_argument("--input", required=True)
    health.add_argument("--original-health", help="Verify frozen v1 summaries before deriving lengths")
    health.add_argument("--output", required=True)
    health.add_argument("--tokenizer", help="Optional tokenizer for separately labelled trace-token attribution")
    health.add_argument("--revision", help="Immutable tokenizer revision; required with --tokenizer")
    editability = sub.add_parser("g1-editability", help="Classify saved native checkpoint traces; never regenerate them")
    edit_sub = editability.add_subparsers(dest="editability_command", required=True)
    inspect_candidate = edit_sub.add_parser("inspect-completed", help="Inspect one finished candidate before selection without launching the editor")
    for flag in ("execution-binding", "sha256", "panel-status", "candidate", "output"):
        inspect_candidate.add_argument("--" + flag, required=True)
    for name in ("prepare", "generate", "finalize", "resources"):
        task = edit_sub.add_parser(name)
        task.add_argument("--input", required=True, help="Preparation binding for prepare; prepared.json otherwise")
        task.add_argument("--sha256", required=True, help="Exact SHA256 of --input")
        task.add_argument("--output", required=True)
        if name == "generate":
            task.add_argument("--device", default="auto")
            task.add_argument("--resume", action="store_true")
            task.add_argument("--max-calls", type=int, help="Bound a measured editor profile, retaining finished contracts")
        if name == "finalize":
            task.add_argument("--generated", required=True)
        if name == "resources":
            task.add_argument("--journal")
            task.add_argument("--hourly-gbp", type=float)
    calibration = sub.add_parser("editor-calibration", help="Prospective training-oracle editor diagnostic; no study outcomes")
    cal_sub = calibration.add_subparsers(dest="calibration_command", required=True)
    cal_prepare = cal_sub.add_parser("prepare")
    for flag in ("training", "worlds", "corpus-manifest", "output"):
        cal_prepare.add_argument("--" + flag, required=True)
    for name in ("generate", "report"):
        task = cal_sub.add_parser(name)
        for flag in ("input", "sha256", "output"):
            task.add_argument("--" + flag, required=True)
        if name == "generate":
            task.add_argument("--device", default="auto")
            task.add_argument("--resume", action="store_true")
        else:
            task.add_argument("--batch", required=True)
    freeze = sub.add_parser("freeze-calibration")
    for flag in ("review", "manifest", "settings", "evidence", "reviewer", "rationale"):
        freeze.add_argument("--" + flag, required=True)
    for command in ("train", "profile"):
        train = sub.add_parser(command)
        for flag in ("config", "input", "output", "gates", "corpus-manifest", "development"):
            train.add_argument("--" + flag, required=True)
        train.add_argument("--profile-report", required=command == "train")
        train.add_argument("--resume-from-checkpoint")
    for command in ("evaluate", "traces", "regenerate-traces", "crossed", "components", "opentom-evaluate"):
        run = sub.add_parser(command)
        run.add_argument("--input", required=True)
        run.add_argument("--output", required=True)
        _model_args(run)
        run.add_argument("--training-seed", type=int, help="Required for trained-adapter inference; used for within-story seed averaging")
        run.add_argument("--arm")
        if command in ("evaluate", "opentom-evaluate"):
            run.add_argument("--reasoning", action="store_true", help="Native reasoning format; otherwise immediate")
        if command == "opentom-evaluate":
            run.add_argument("--opentom-protocol", choices=["official", "adapted"], default="official")
        if command in ("evaluate", "traces", "components"):
            run.add_argument("--completion-protocol", choices=["legacy_forced_answer_v1", "natural_answer_cap_v2"],
                             required=True, help="Explicit versioned completion policy; legacy requires deliberate selection")
        if command == "evaluate":
            run.add_argument("--format", choices=["immediate", "native_answer", "native_reasoning"], default="immediate")
        if command in ("evaluate", "crossed"):
            run.add_argument("--reader", required=True, help="Stable checkpoint alias, e.g. base or reasoning_rl_seed1")
        if command == "traces":
            run.add_argument("--source", required=True)
            run.add_argument("--timing", choices=["question_dependent", "direct_fixed", "prequestion"], default="question_dependent")
        if command == "regenerate-traces":
            run.add_argument("--source", required=True)
            run.add_argument("--traces", nargs="+", required=True)
            run.add_argument("--edits", required=True)
            run.add_argument("--seed", type=int, default=2026)
            run.add_argument("--edit-audit", help="Completed preregistered sample audit for generated controls")
        if command == "crossed":
            run.add_argument("--traces", nargs="+", required=True)
            run.add_argument("--edits")
            run.add_argument("--edit-audit", help="Same completed sample audit for both readers")
            run.add_argument("--controls", choices=["basic", "full"], default="full")
            run.add_argument("--regenerated", nargs="+", help="Frozen source-generated continuations, shared across readers")
            run.add_argument("--comparison-seed", type=int, help="Paired base/trained comparison replicate; inferred from source traces when possible")
        if command == "components":
            selection = run.add_mutually_exclusive_group()
            selection.add_argument("--layers", type=int, nargs="+", help="Secondary diagnostic layer mask; primary defaults to layers 1 through last")
            selection.add_argument("--all-layers", action="store_true", help="Additional diagnostic including layer 0; primary G4 remains layers 1 through last")
            run.add_argument("--seed", type=int, default=1)
            run.add_argument("--layer-mode", choices=["joint", "single", "both"], default="both")
            run.add_argument("--residual-control-layers", type=int, nargs="+",
                             help="Additional residual diagnostics; does not declare a G4 test band")
            run.add_argument("--g4-test-layers", type=int, nargs="+",
                             help="Frozen contiguous band tested jointly for K, V and its aligned residual control")
            run.add_argument("--donor-model", help="Explicit secondary cross-model analysis on identical input tokens")
            run.add_argument("--donor-adapter")
            run.add_argument("--donor-revision")
    edits = sub.add_parser("prepare-edits")
    edits.add_argument("--traces", nargs="+", required=True)
    edits.add_argument("--output", required=True)
    edits.add_argument("--input", required=True, help="Evaluation rows with symbolic worlds")
    edits.add_argument("--max-worlds", type=int, help="Pilot defaults to 12; confirmatory uses all candidate worlds")
    edits.add_argument("--stage", choices=["pilot", "confirmatory"], default="pilot")
    edits.add_argument("--seed", type=int, default=2026)
    plan = sub.add_parser("plan-edit-generation", help="Freeze editor model, prompt, decoding and audit sample")
    for flag in ("edits", "model", "revision", "output"):
        plan.add_argument("--" + flag, required=True)
    plan.add_argument("--seed", type=int, default=2026)
    plan.add_argument("--max-new-tokens", type=int, default=2048)
    plan.add_argument("--audit-sample-size", type=int, default=100)
    plan.add_argument("--audit-seed", type=int, default=2026)
    plan.add_argument("--audit-confidence", type=float, default=.95)
    generate_edits = sub.add_parser("generate-edits", help="Generate frozen controls with the preregistered editor")
    for flag in ("edits", "plan", "output"):
        generate_edits.add_argument("--" + flag, required=True)
    generate_edits.add_argument("--traces", nargs="+", required=True)
    generate_edits.add_argument("--device", default="auto")
    generate_edits.add_argument("--resume", action="store_true", help="Reuse the existing per-contract journal without resampling finished outputs")
    for command in ("prepare-edit-audit", "verify-edit-audit"):
        audit = sub.add_parser(command)
        audit.add_argument("--edits", required=True)
        audit.add_argument("--output", required=True)
        if command == "verify-edit-audit":
            audit.add_argument("--audit", required=True)
    analyze = sub.add_parser("analyze")
    analyze.add_argument("kind", choices=["accuracy", "transfer", "crossed", "cues", "components", "opentom", "match"])
    analyze.add_argument("--input", required=True)
    analyze.add_argument("--baseline")
    analyze.add_argument("--output", required=True)
    analyze.add_argument("--expected-runs", help="JSON list of prespecified arm/dataset/seed run identities for matching")
    for flag in ("base-reader", "trained-reader", "base-source", "trained-source"):
        analyze.add_argument("--" + flag)
    plot = sub.add_parser("plot", help="Render measured estimates and intervals without decision cutoffs")
    plot.add_argument("kind", choices=["quadrant", "heuristics", "h3"])
    plot.add_argument("--input", required=True, help="Quadrant/H3 manifest JSON or model result JSONL")
    plot.add_argument("--audit", help="Required for heuristics: shortcut audit JSON")
    plot.add_argument("--split", choices=["train", "development", "confirmation"])
    plot.add_argument("--output", required=True, help="PNG, PDF or SVG; a JSON data sidecar is also saved")
    imp = sub.add_parser("opentom-import")
    imp.add_argument("--input", required=True)
    imp.add_argument("--output", required=True)
    imp.add_argument("--source-url")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    command = args.command
    if command == "generate":
        from .data import generate_corpus
        result = generate_corpus(args.output, seed=args.seed, train=args.train, development=args.development,
                                 confirmation=args.confirmation, depths=tuple(args.depths),
                                 distractor_events=args.distractor_events, hidden_move=args.hidden_move)
        print(json.dumps({"manifest": str(Path(args.output) / "manifest.json"), "corpus_hash": result["corpus_hash"]}))
    elif command == "verify-data":
        from .data import verify_corpus
        print(json.dumps(verify_corpus(args.data)))
    elif command == "audit-shortcuts":
        from .data import verify_corpus
        verify_corpus(args.data)
        write_json(args.output, _json(Path(args.data) / "manifest.json")["shortcut_audit"])
    elif command == "audit-tokenizer":
        from transformers import AutoTokenizer
        from .data import audit_tokenizer_lengths
        from .training.config import MODEL_REVISIONS
        rows = [row for path in args.input for row in read_jsonl(path)]
        revision = args.revision or MODEL_REVISIONS.get(args.model)
        import re
        if not revision or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("Tokenizer audit requires an immutable model revision")
        result = audit_tokenizer_lengths(rows, AutoTokenizer.from_pretrained(args.model, revision=revision))
        result['tokenizer'] = {'model': args.model, 'revision': revision}
        write_json(args.output, result)
        if not result["passed"]:
            raise ValueError("G2 tokenizer length matching failed; see audit output")
    elif command == "matrix":
        from .training.config import experiment_matrix
        if args.development_panel:
            if args.stage != "pilot":
                raise ValueError("Development selection uses pilot data and arms only")
            from .training.selection import export_development_panel
            export_development_panel(args.output, design=args.design)
        else:
            from .training.selection import amended_experiment_matrix
            configs = (amended_experiment_matrix(args.stage) if args.design == "amended_20261030"
                       else experiment_matrix(args.stage))
            for config in configs:
                name = f"{config.arm}_{config.dataset}_seed{config.seed}.json"
                write_json(Path(args.output) / name, config.to_dict())
    elif command == "freeze-selection-failure-policy":
        from datetime import datetime, timezone
        from .training.selection import selection_failure_policy
        if not args.reviewer.strip() or not args.rationale.strip():
            raise ValueError("Failure policy needs attribution and rationale")
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump({"schema_version": 1, "status": "frozen", "frozen_utc": datetime.now(timezone.utc).isoformat(),
                       "reviewed_by": args.reviewer, "rationale": args.rationale,
                       "policy": selection_failure_policy()}, stream, indent=2, sort_keys=True)
            stream.write("\n")
        print(json.dumps({"path": str(output.resolve()), "sha256": file_digest(output)}))
    elif command == "select-panel":
        from .training.selection_results import freeze_panel_selection
        result = freeze_panel_selection(args.execution_binding, args.sha256, args.panel_status,
                    args.failure_policy, args.failure_policy_sha256, args.output,
                    reviewed_by=args.reviewer, rationale=args.rationale)
        print(json.dumps({"path": str(Path(args.output).resolve()), "sha256": file_digest(args.output),
                          "selected_settings": result['selected_settings'], "failures": result['failures']}))
    elif command == "verify-panel-selection":
        from .training.selection_results import verify_panel_selection
        result = verify_panel_selection(args.selection, args.sha256)
        print(json.dumps({"verified": True, "selected_settings": result['selected_settings'],
                          "scientific_gate_decision": None}))
    elif command == "selected-confirmatory-configs":
        from .training.confirmatory_selection import export_selected_confirmatory_configs
        result = export_selected_confirmatory_configs(args.selection, args.output,
                                                       selection_sha256=args.selection_sha256)
        print(json.dumps(result))
    elif command == "init-gates":
        from .gates import initial_review
        if Path(args.output).exists():
            raise ValueError("Review already exists; refusing to erase gate decisions")
        write_json(args.output, initial_review())
    elif command == "freeze-calibration":
        from .gates import freeze_calibration
        freeze_calibration(args.review, args.manifest, _json(args.settings), args.evidence, args.reviewer, args.rationale)
    elif command == "verify-g1":
        from .gates import verify_g1_learning_curves
        review = _json(args.review)
        if Path(args.output).resolve() == Path(args.review).resolve():
            raise ValueError("G1 evidence output must not overwrite the review")
        result = verify_g1_learning_curves(review["gates"]["G1"], Path(args.review).parent,
                                          expected_manifest_sha256=review["calibration"]["manifest_sha256"],
                                          calibration=review["calibration"], historical=args.historical)
        write_json(args.output, {"schema_version": 1, "review_sha256": file_digest(args.review),
                                "learning_curves": result,
                                "note": "Evidence completeness only. No reviewer decision or numerical pass mark is inferred."})
    elif command == "trace-health":
        from .training.trace_health_reanalysis import reanalyze_trace_health
        tokenizer = provenance = None
        if bool(args.tokenizer) != bool(args.revision):
            raise ValueError("Specify both --tokenizer and its immutable --revision, or neither")
        if args.tokenizer:
            import re
            if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
                raise ValueError("Tokenizer revision must be an immutable commit")
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, revision=args.revision)
            provenance = {"model": args.tokenizer, "revision": args.revision}
        reanalyze_trace_health(args.input, args.output, original_health_path=args.original_health,
                               tokenizer=tokenizer, tokenizer_provenance=provenance)
    elif command == "g1-editability":
        from .training import g1_editability
        if args.editability_command == "inspect-completed":
            print(json.dumps(g1_editability.inspect_completed_candidate(args.execution_binding, args.sha256,
                                     args.panel_status, args.candidate, args.output)))
        elif args.editability_command == "prepare":
            print(json.dumps(g1_editability.prepare(args.input, args.sha256, args.output)))
        elif args.editability_command == "generate":
            print(json.dumps(g1_editability.generate(args.input, args.sha256, args.output, device=args.device,
                                   resume=args.resume, max_calls=args.max_calls)))
        elif args.editability_command == "finalize":
            print(json.dumps(g1_editability.finalize(args.input, args.sha256, args.generated, args.output)))
        else:
            if Path(args.output).exists():
                raise FileExistsError("Resource report output must be a new file")
            write_json(args.output, g1_editability.resource_report(args.input, args.sha256,
                       journal_path=args.journal, hourly_gbp=args.hourly_gbp))
    elif command == "editor-calibration":
        from . import editor_calibration
        if args.calibration_command == "prepare":
            result = editor_calibration.prepare_files(args.training, args.worlds, args.corpus_manifest, args.output)
        elif args.calibration_command == "generate":
            result = editor_calibration.generate_file(args.input, args.sha256, args.output,
                                                      device=args.device, resume=args.resume)
        else:
            result = editor_calibration.report_file(args.input, args.sha256, args.batch, args.output)
        print(json.dumps(result))
    elif command in ("train", "profile"):
        from .training.config import load_config
        from .training.runner import run_training
        config = load_config(args.config)
        if command == "profile":
            config = replace(config, stage="confirmatory" if config.stage == "confirmatory" else "profile", profile=True)
        dev = [r for r in read_jsonl(args.development) if r["view"] == "direct"]
        run_training(config, read_jsonl(args.input), args.output, gate_file=args.gates,
                     profile_report=args.profile_report, corpus_manifest=args.corpus_manifest,
                     dev_rows=dev, resume_from_checkpoint=args.resume_from_checkpoint)
    elif command == "prepare-edits":
        from .traces import prepare_edit_manifest
        write_json(args.output, prepare_edit_manifest([r for p in args.traces for r in read_jsonl(p)],
                    read_jsonl(args.input), max_worlds=args.max_worlds, seed=args.seed,
                    stage=args.stage))
    elif command == "plan-edit-generation":
        from .edit_generation import prepare_generation_plan
        if Path(args.output).exists():
            raise ValueError("Generation plan already exists; refusing to overwrite preregistration")
        write_json(args.output, prepare_generation_plan(_json(args.edits), model_id=args.model,
                     revision=args.revision, seed=args.seed, max_new_tokens=args.max_new_tokens,
                     audit_sample_size=args.audit_sample_size, audit_seed=args.audit_seed,
                     audit_confidence=args.audit_confidence))
    elif command == "generate-edits":
        from .edit_generation import generate_edit_controls, _validate_plan
        from .backend import HFBackend
        output = Path(args.output)
        journal = Path(str(output) + ".journal.jsonl")
        if output.exists():
            raise ValueError("Generated batch already exists; refusing to repeat or overwrite generation")
        if journal.exists() != args.resume:
            raise ValueError("Existing journal requires --resume; a fresh run must omit --resume")
        plan = _json(args.plan)
        _validate_plan(plan)
        completed = read_jsonl(journal) if args.resume else []
        backend = HFBackend(plan["model"]["id"], device=args.device, revision=plan["model"]["revision"])
        output.parent.mkdir(parents=True, exist_ok=True)
        with journal.open("a", encoding="utf-8") as stream:
            def save_contract(contract):
                import os
                stream.write(json.dumps(contract, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            result = generate_edit_controls(backend, [r for p in args.traces for r in read_jsonl(p)],
                         _json(args.edits), plan, on_result=save_contract, completed_results=completed)
        write_json(output, result)
    elif command == "prepare-edit-audit":
        from .edit_generation import prepare_batch_audit
        if Path(args.output).exists():
            raise ValueError("Audit file already exists; refusing to overwrite human decisions")
        write_json(args.output, prepare_batch_audit(_json(args.edits)))
    elif command == "verify-edit-audit":
        from .edit_generation import validate_batch_audit
        write_json(args.output, validate_batch_audit(_json(args.edits), _json(args.audit)))
    elif command == "plot":
        from .plotting import (load_quadrant_data, plot_quadrant,
                               accuracy_reference_data, plot_accuracy_references)
        if args.kind == "quadrant":
            plot_quadrant(load_quadrant_data(args.input), args.output)
        elif args.kind == "h3":
            from .h3 import load_h3_data
            from .plotting import plot_h3
            plot_h3(load_h3_data(args.input), args.output)
        else:
            if not args.audit:
                raise ValueError("Heuristic reference plots require --audit")
            plot_accuracy_references(accuracy_reference_data(read_jsonl(args.input), _json(args.audit),
                                      split=args.split), args.output)
    elif command == "opentom-import":
        from .opentom import import_opentom
        import_opentom(args.input, args.output, args.source_url)
    elif command == "analyze":
        from . import analysis
        from .protocols import completion_protocol
        rows = read_jsonl(args.input)
        baseline = read_jsonl(args.baseline) if args.baseline else []
        protocol = completion_protocol(rows + baseline) if args.kind != "opentom" else {}
        if args.kind in ("transfer", "crossed", "components") and not args.baseline:
            raise ValueError("This analysis requires --baseline")
        if args.kind == "accuracy":
            result = analysis.accuracy_summary(rows)
        elif args.kind == "transfer":
            result = analysis.transfer(read_jsonl(args.baseline), rows)
        elif args.kind == "crossed":
            result = analysis.crossed_summary(read_jsonl(args.baseline), rows,
                        base_reader=args.base_reader, trained_reader=args.trained_reader,
                        base_source=args.base_source, trained_source=args.trained_source)
        elif args.kind == "cues":
            result = analysis.cue_summary(rows)
        elif args.kind == "components":
            from .interventions import component_summary
            result = component_summary(read_jsonl(args.baseline), rows)
        elif args.kind == "opentom":
            from .opentom import summarize_opentom
            result = summarize_opentom(rows)
        else:
            result = analysis.match_checkpoints(rows, expected_runs=_json(args.expected_runs) if args.expected_runs else None)
        write_json(args.output, {**result, **protocol})
    else:
        if args.adapter and args.training_seed is None:
            raise ValueError("Trained adapter inference requires --training-seed for valid confirmatory aggregation")
        if getattr(args, "donor_adapter", None) and not args.donor_model:
            raise ValueError("--donor-adapter needs --donor-model")
        adapter_hashes = {}
        for field in ("adapter", "donor_adapter"):
            value = getattr(args, field, None)
            if value:
                directory = Path(value).resolve()
                weights = directory / "adapter_model.safetensors"
                if not weights.is_file():
                    raise ValueError(f"--{field.replace('_', '-')} requires a retained local adapter with safetensors weights")
                setattr(args, field, str(directory))
                adapter_hashes[field] = file_digest(weights)
        backend = _backend(args)
        rows = read_jsonl(args.input)
        if command == "evaluate":
            from .evaluation import evaluate_native
            results = evaluate_native(backend, rows,
                        reasoning=args.reasoning or args.format == "native_reasoning", reader=args.reader,
                        native_answer=args.format == "native_answer", completion_protocol=args.completion_protocol)
        elif command == "traces":
            from .traces import collect_traces
            results = collect_traces(backend, rows, args.source, args.timing, completion_protocol=args.completion_protocol)
        elif command == "regenerate-traces":
            from .traces import regenerate_trace_artifacts
            results = regenerate_trace_artifacts(backend, rows,
                         [r for p in args.traces for r in read_jsonl(p)], args.source,
                         _json(args.edits), seed=args.seed, edit_audit=_json(args.edit_audit) if args.edit_audit else None)
        elif command == "crossed":
            from .traces import run_crossed_reader, CONTROL_NAMES
            conditions = CONTROL_NAMES if args.controls == "full" else ("original", "own_replay", "own_rerendered", "answer_cue", "none")
            source_traces = [r for p in args.traces for r in read_jsonl(p)]
            source_seeds = {r.get("training_seed") for r in source_traces if r.get("training_seed") is not None}
            if len(source_seeds) > 1:
                raise ValueError("Run each base/trained seed pair separately, then combine matched result files for analysis")
            if args.comparison_seed is None and source_seeds:
                args.comparison_seed = next(iter(source_seeds))
            if source_seeds and args.comparison_seed not in source_seeds:
                raise ValueError("Comparison seed differs from the trained trace source")
            if args.training_seed is not None and args.comparison_seed != args.training_seed:
                raise ValueError("Trained reader and source must belong to the same prespecified comparison seed")
            results = run_crossed_reader(backend, rows, source_traces, args.reader,
                                        _json(args.edits) if args.edits else None, conditions,
                                        regenerated=[r for p in (args.regenerated or []) for r in read_jsonl(p)],
                                        edit_audit=_json(args.edit_audit) if args.edit_audit else None)
        elif command == "components":
            from .interventions import Site, run_component_test
            primary_layers = list(range(1, backend.model.config.num_hidden_layers))
            layers = list(range(backend.model.config.num_hidden_layers)) if args.all_layers else (args.layers or primary_layers)
            sites = [Site(layer, component) for layer in layers for component in ("k", "v", "residual")]
            donor_backend = None
            if args.donor_model:
                from .backend import HFBackend
                donor_backend = HFBackend(args.donor_model, args.donor_adapter, device=args.device, revision=args.donor_revision)
            results = run_component_test(backend, rows, sites, seed=args.seed,
                         layer_mode=args.layer_mode, donor_backend=donor_backend,
                         residual_control_layers=args.residual_control_layers, g4_test_layers=args.g4_test_layers or (None if donor_backend else primary_layers))
        elif command == "opentom-evaluate":
            from .opentom import evaluate_opentom
            results = evaluate_opentom(backend, rows, args.reasoning, protocol=args.opentom_protocol)
        else:
            raise ValueError(command)
        if command != "regenerate-traces":
            results = [{**row, "training_seed": args.training_seed,
                        **({"completion_protocol": args.completion_protocol} if command == "components" else {}),
                        **({"comparison_seed": args.comparison_seed} if command == "crossed" and args.comparison_seed is not None else {}),
                        **({"arm": args.arm} if args.arm else {})} for row in results]
        write_jsonl(args.output, results)
        for field, expected_hash in adapter_hashes.items():
            if file_digest(Path(getattr(args, field)) / "adapter_model.safetensors") != expected_hash:
                raise ValueError("Adapter weights changed during inference; no completed receipt written")
        write_json(args.output + ".run.json", {"command": vars(args), "input_sha256": file_digest(args.input),
                   "output_sha256": file_digest(args.output), "checkpoint": backend.checkpoint_id,
                   "adapter_sha256": adapter_hashes.get("adapter"),
                   "donor_adapter_sha256": adapter_hashes.get("donor_adapter"),
                   "resolved_revision": getattr(backend.model.config, "_commit_hash", None),
                   "status": "completed"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
