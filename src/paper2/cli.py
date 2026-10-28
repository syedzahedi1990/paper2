"""Composable experiment commands. No command provisions or purchases compute."""
import argparse
from dataclasses import replace
from pathlib import Path
import json
import sys
from .io import read_jsonl, write_jsonl, write_json, file_digest


def _json(path):
    return json.loads(Path(path).read_text())


def _backend(args):
    from .backend import HFBackend
    return HFBackend(args.model, args.adapter, device=args.device, revision=args.revision)


def _model_args(parser):
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
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
    audit = sub.add_parser("audit-tokenizer")
    audit.add_argument("--input", nargs="+", required=True)
    audit.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    audit.add_argument("--output", required=True)
    matrix = sub.add_parser("matrix")
    matrix.add_argument("--stage", choices=["pilot", "confirmatory"], required=True)
    matrix.add_argument("--output", required=True)
    gates = sub.add_parser("init-gates")
    gates.add_argument("--output", required=True)
    freeze = sub.add_parser("freeze-calibration")
    for flag in ("review", "manifest", "settings", "evidence", "reviewer", "rationale"):
        freeze.add_argument("--" + flag, required=True)
    for command in ("train", "profile"):
        train = sub.add_parser(command)
        for flag in ("config", "input", "output", "gates", "corpus-manifest", "development"):
            train.add_argument("--" + flag, required=True)
        train.add_argument("--profile-report", required=command == "train")
        train.add_argument("--resume-from-checkpoint")
    for command in ("evaluate", "traces", "crossed", "components", "opentom-evaluate"):
        run = sub.add_parser(command)
        run.add_argument("--input", required=True)
        run.add_argument("--output", required=True)
        _model_args(run)
        if command in ("evaluate", "opentom-evaluate"):
            run.add_argument("--reasoning", action="store_true", help="Native reasoning format; otherwise immediate")
        if command == "evaluate":
            run.add_argument("--format", choices=["immediate", "native_answer", "native_reasoning"], default="immediate")
        if command in ("evaluate", "crossed"):
            run.add_argument("--reader", required=True, help="Stable checkpoint alias, e.g. base or reasoning_rl_seed1")
        if command == "traces":
            run.add_argument("--source", required=True)
            run.add_argument("--timing", choices=["question_dependent", "prequestion"], default="question_dependent")
        if command == "crossed":
            run.add_argument("--traces", nargs="+", required=True)
            run.add_argument("--edits")
            run.add_argument("--controls", choices=["basic", "full"], default="full")
        if command == "components":
            run.add_argument("--layers", type=int, nargs="+", required=True)
            run.add_argument("--seed", type=int, default=1)
    edits = sub.add_parser("prepare-edits")
    edits.add_argument("--traces", nargs="+", required=True)
    edits.add_argument("--output", required=True)
    analyze = sub.add_parser("analyze")
    analyze.add_argument("kind", choices=["accuracy", "transfer", "crossed", "cues", "components", "opentom", "match"])
    analyze.add_argument("--input", required=True)
    analyze.add_argument("--baseline")
    analyze.add_argument("--output", required=True)
    analyze.add_argument("--target-accuracy", type=float)
    analyze.add_argument("--tolerance", type=float, default=.02)
    for flag in ("base-reader", "trained-reader", "base-source", "trained-source"):
        analyze.add_argument("--" + flag)
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
    elif command == "audit-tokenizer":
        from transformers import AutoTokenizer
        from .data import audit_tokenizer_lengths
        rows = [row for path in args.input for row in read_jsonl(path)]
        result = audit_tokenizer_lengths(rows, AutoTokenizer.from_pretrained(args.model))
        write_json(args.output, result)
        if not result["passed"]:
            raise ValueError("G2 tokenizer length matching failed; see audit output")
    elif command == "matrix":
        from .training.config import experiment_matrix
        for config in experiment_matrix(args.stage):
            name = f"{config.arm}_{config.dataset}_seed{config.seed}.json"
            write_json(Path(args.output) / name, config.to_dict())
    elif command == "init-gates":
        from .gates import initial_review
        if Path(args.output).exists():
            raise ValueError("Review already exists; refusing to erase gate decisions")
        write_json(args.output, initial_review())
    elif command == "freeze-calibration":
        from .gates import freeze_calibration
        freeze_calibration(args.review, args.manifest, _json(args.settings), args.evidence, args.reviewer, args.rationale)
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
        write_json(args.output, prepare_edit_manifest([r for p in args.traces for r in read_jsonl(p)]))
    elif command == "opentom-import":
        from .opentom import import_opentom
        import_opentom(args.input, args.output, args.source_url)
    elif command == "analyze":
        from . import analysis
        rows = read_jsonl(args.input)
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
            from .interventions import compare_checkpoints
            from collections import defaultdict
            effects = compare_checkpoints(read_jsonl(args.baseline), rows)
            clusters = defaultdict(lambda: defaultdict(list))
            for effect in effects:
                key = f"{effect['layer']}|{effect['component']}|{effect['control']}"
                clusters[key][effect["pair_id"]].append(effect["training_change_in_effect"])
            result = {"paired_effects": effects, "summary": {k: analysis.cluster_interval(v) for k, v in clusters.items()}}
        elif args.kind == "opentom":
            from .opentom import summarize_opentom
            result = summarize_opentom(rows)
        else:
            if args.target_accuracy is None:
                raise ValueError("--target-accuracy required for matching")
            result = analysis.match_checkpoints(rows, args.target_accuracy, args.tolerance)
        write_json(args.output, result)
    else:
        backend = _backend(args)
        rows = read_jsonl(args.input)
        if command == "evaluate":
            from .evaluation import evaluate_native
            results = evaluate_native(backend, rows,
                        reasoning=args.reasoning or args.format == "native_reasoning", reader=args.reader,
                        native_answer=args.format == "native_answer")
        elif command == "traces":
            from .traces import collect_traces
            results = collect_traces(backend, rows, args.source, args.timing)
        elif command == "crossed":
            from .traces import run_crossed_reader, CONTROL_NAMES
            conditions = CONTROL_NAMES if args.controls == "full" else ("original", "own_replay", "answer_cue", "none")
            results = run_crossed_reader(backend, rows, [r for p in args.traces for r in read_jsonl(p)], args.reader,
                                        _json(args.edits) if args.edits else None, conditions)
        elif command == "components":
            from .interventions import Site, run_component_test
            sites = [Site(layer, component) for layer in args.layers for component in ("k", "v", "residual")]
            results = run_component_test(backend, rows, sites, seed=args.seed)
        elif command == "opentom-evaluate":
            from .opentom import evaluate_opentom
            results = evaluate_opentom(backend, rows, args.reasoning)
        else:
            raise ValueError(command)
        write_jsonl(args.output, results)
        write_json(args.output + ".run.json", {"command": vars(args), "input_sha256": file_digest(args.input),
                   "output_sha256": file_digest(args.output), "checkpoint": backend.checkpoint_id,
                   "resolved_revision": getattr(backend.model.config, "_commit_hash", None),
                   "status": "completed"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
