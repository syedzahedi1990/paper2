"""Evidence-backed pilot review. Gates never pass by absence of evidence."""
import json
import math
from datetime import date
from pathlib import Path
from .io import file_digest, write_json

GATES = ("G0", "G1", "G2", "G3", "G4")
G4_LAYER_POLICY = "all_after_embedding_layer"


def initial_review():
    return {"schema_version": 1, "gates": {g: {"status": "pending", "evidence": [], "rationale": ""} for g in GATES},
            "calibration": {"frozen": False, "manifest_sha256": None, "manifest_path": None, "settings": {}},
            "paid_compute": {"approved": False, "approved_by": None, "gbp_limit": 500,
                             "estimated_gbp": None, "exchange_rate_date": None},
            "note": "Review measurements, not whether a preferred hypothesis won. No model results yet."}


def verify_evidence(gate, root):
    if gate.get("status") != "pass" or not gate.get("rationale") or not gate.get("reviewed_by"):
        raise ValueError("Gate requires a pass decision, reviewer and rationale")
    if not gate.get("evidence"):
        raise ValueError("Gate has no evidence files")
    for item in gate["evidence"]:
        path = Path(root) / item["path"]
        if not path.is_file() or file_digest(path) != item["sha256"]:
            raise ValueError(f"Gate evidence missing or modified: {item['path']}")


def verify_g1_learning_curves(gate, root, *, expected_manifest_sha256=None, seed=2026, samples=20_000):
    """Validate every pilot arm and report pointwise/paired story intervals.

    Intervals describe the fixed seed-1 pilot over sampled stories, not training
    seed variability. They never impose an improvement or significance cutoff.
    """
    from .analysis import cluster_interval
    from .io import read_jsonl
    if type(samples) is not int or samples < 1:
        raise ValueError("G1 bootstrap samples must be a positive integer")
    entries = gate.get("learning_curves")
    arms = {"reasoning_rl", "answer_rl", "label_sft"}
    if (not isinstance(entries, list) or len(entries) != len(arms)
            or any(not isinstance(item, dict) for item in entries)
            or {item.get("arm") for item in entries} != arms):
        raise ValueError("G1 requires learning_curves for all three pilot arms: reasoning_rl, answer_rl and label_sft")
    results = {}
    common_worlds = None
    common_training_hash = None
    common_revision = None
    for entry in entries:
        arm = entry["arm"]
        if not isinstance(entry.get("interpretation"), str) or not entry["interpretation"].strip():
            raise ValueError(f"G1 needs a reviewer interpretation for {arm}, including any lack of movement")
        path = Path(root) / entry.get("path", "")
        if not path.is_file() or file_digest(path) != entry.get("sha256"):
            raise ValueError(f"G1 {arm} learning evidence missing or modified")
        if not any((Path(root) / item["path"]).resolve() == path.resolve()
                   and item["sha256"] == entry["sha256"] for item in gate.get("evidence", [])):
            raise ValueError("G1 learning-curve files must be included in reviewed evidence")
        receipt = json.loads(path.read_text())
        if receipt.get("schema_version") != 3:
            raise ValueError("G1 requires finalized schema-3 learning evidence with hashed run artifacts")
        artifacts = {}
        for name in ("manifest", "compute", "evaluations", "predictions"):
            item = receipt.get("artifacts", {}).get(name, {})
            artifact = path.parent / item.get("path", "")
            if not artifact.is_file() or file_digest(artifact) != item.get("sha256"):
                raise ValueError(f"G1 {arm} {name} artifact missing or modified")
            artifacts[name] = (read_jsonl(artifact) if name in ("evaluations", "predictions")
                               else json.loads(artifact.read_text()))
        manifest, compute = artifacts["manifest"], artifacts["compute"]
        config = manifest.get("config", {})
        identity_keys = ("arm", "dataset", "seed", "model", "revision", "stage")
        identity = {key: config.get(key) for key in identity_keys}
        if (manifest.get("status") != "complete" or receipt.get("run_status") != "complete"
                or manifest.get("synthetic_integration_test") is not False or receipt.get("synthetic") is not False
                or identity != receipt.get("run_identity")
                or identity["arm"] != arm or identity["dataset"] != "clean" or identity["seed"] != 1
                or identity["stage"] != "pilot" or config.get("profile") is not False
                or identity["model"] != "Qwen/Qwen2.5-3B-Instruct"):
            raise ValueError(f"G1 {arm} must be a completed non-synthetic 3B clean seed-1 pilot run")
        from .training.config import TrainingConfig
        run_config = TrainingConfig(**config)
        run_config.validate()
        if common_revision is not None and identity["revision"] != common_revision:
            raise ValueError("G1 pilot arms used different base-model revisions")
        common_revision = identity["revision"]
        if (expected_manifest_sha256 is not None
                and manifest.get("corpus_manifest_sha256") != expected_manifest_sha256):
            raise ValueError("G1 learning evidence belongs to a different frozen corpus")
        if manifest.get("training_examples") != 2000 or compute.get("unique_worlds_seen") != 2000:
            raise ValueError("G1 pilot curves must follow complete exposure to the 2,000-story pool")
        training_hash = manifest.get("training_sha256")
        if not isinstance(training_hash, str) or len(training_hash) != 64:
            raise ValueError("G1 pilot run requires its training-data hash")
        if common_training_hash is not None and training_hash != common_training_hash:
            raise ValueError("G1 pilot arms used different training rows")
        common_training_hash = training_hash
        final = manifest.get("training_budget", {}).get("resolved_max_steps")
        if (type(final) is not int or final != run_config.training_budget(2000)["resolved_max_steps"]
                or compute.get("completed_optimizer_steps") != final):
            raise ValueError("G1 curves require a completed optimizer budget and final checkpoint")
        native = "native_reasoning" if arm == "reasoning_rl" else "native_answer"
        schedules = {"immediate": run_config.immediate_evaluation_schedule(final),
                     native: run_config.evaluation_schedule(final)}
        expected = {(step, fmt) for fmt, steps in schedules.items() for step in steps}
        metrics = artifacts["evaluations"]
        if (len(metrics) != len(expected)
                or {(row.get("step"), row.get("format")) for row in metrics} != expected):
            raise ValueError("G1 requires complete scheduled native and immediate curves, including step 0 and final")
        predictions = artifacts["predictions"]
        grouped = {}
        for row in predictions:
            grouped.setdefault((row.get("step"), row.get("format")), []).append(row)
        if set(grouped) != expected:
            raise ValueError("G1 prediction panels differ from the scheduled learning curves")
        for metric in metrics:
            panel = grouped[(metric["step"], metric["format"])]
            worlds = {row.get("world_id") for row in panel}
            if (len(panel) != 120 or len(worlds) != 120 or None in worlds
                    or len({row.get("id") for row in panel}) != 120 or metric.get("n") != 120):
                raise ValueError("G1 requires complete 120-story development panels")
            if common_worlds is not None and worlds != common_worlds:
                raise ValueError("G1 learning curves must use the same development stories in every arm and checkpoint")
            common_worlds = worlds
            for row in [metric, *panel]:
                if (any(row.get(key) != identity[key] for key in identity_keys if key != "stage")
                        or row.get("split") != "development" or row.get("view") != "direct"
                        or row.get("evaluation_dataset") != "clean"
                        or row.get("checkpoint_id") != metric.get("checkpoint_id")
                        or row.get("adapter_sha256") != metric.get("adapter_sha256")):
                    raise ValueError("G1 learning-curve identity or checkpoint binding is inconsistent")
            if any(type(row.get("correct")) is not bool for row in panel):
                raise ValueError("G1 predictions require boolean correctness measurements")
            accuracy = sum(row["correct"] for row in panel) / len(panel)
            if (type(metric.get("accuracy")) not in (int, float)
                    or not math.isfinite(metric["accuracy"]) or abs(metric["accuracy"] - accuracy) > 1e-12):
                raise ValueError("G1 aggregate accuracy disagrees with the raw prediction panel")
        by_key = {(row["step"], row["format"]): row for row in metrics}
        if (receipt.get("primary_format") != native or receipt.get("baseline") != by_key[(0, native)]
                or receipt.get("latest") != by_key[(final, native)]
                or receipt.get("immediate_latest") != by_key[(final, "immediate")]):
            raise ValueError("G1 receipt boundaries disagree with the complete learning curves")
        curves = [{**metric, "accuracy_interval": cluster_interval(
                    {row["world_id"]: [int(row["correct"])]
                     for row in grouped[(metric["step"], metric["format"])]}, seed=seed, samples=samples)}
                  for metric in metrics]
        gains = {}
        for fmt in (native, "immediate"):
            before = {row["world_id"]: int(row["correct"]) for row in grouped[(0, fmt)]}
            gains[fmt] = cluster_interval({row["world_id"]: [int(row["correct"]) - before[row["world_id"]]]
                                          for row in grouped[(final, fmt)]}, seed=seed, samples=samples)
        results[arm] = {"interpretation": entry["interpretation"], "curves": curves,
                        "native_accuracy_gain": gains[native]["estimate"],
                        "immediate_accuracy_gain": gains["immediate"]["estimate"],
                        "native_gain_interval": gains[native], "immediate_gain_interval": gains["immediate"],
                        "bootstrap_samples": samples, "bootstrap_seed": seed,
                        "uncertainty": "95% pointwise whole-story bootstrap accuracy intervals and paired baseline-to-final gain intervals; fixed pilot seed 1, not training-seed uncertainty."}
    return results


def verify_g4_positive_control(gate, root, *, expected_test_layers=None):
    """Bind G4 to measured, nontrivial residual evidence, never a layer-0 replay.

    A named reviewer still judges flat nulls, preservation and useful precision.
    This check rejects missing or structurally ineligible evidence. Effect sizes
    and intervals are returned for review without an automatic numerical cutoff.
    """
    from .interventions.controls import residual_control_summary
    from .io import read_jsonl
    selected = gate.get("positive_control", {})
    required = ("path", "sha256", "mask_id", "checkpoint_id")
    if not all(isinstance(selected.get(key), str) and selected[key] for key in required):
        raise ValueError("G4 requires a selected nontrivial residual positive-control evidence file, mask and checkpoint")
    test_layers = selected.get("test_layers")
    if (not isinstance(test_layers, list) or not test_layers
            or any(type(layer) is not int or layer <= 0 for layer in test_layers)
            or test_layers != list(range(min(test_layers), max(test_layers) + 1))):
        raise ValueError("G4 requires an explicit contiguous positive_control.test_layers band")
    if expected_test_layers is not None and test_layers != expected_test_layers:
        raise ValueError("G4 positive-control test band differs from the frozen calibration band")
    if selected["mask_id"] != ",".join(map(str, test_layers)):
        raise ValueError("G4 residual mask must exactly match the declared test band")
    path = Path(root) / selected["path"]
    if not path.is_file() or file_digest(path) != selected["sha256"]:
        raise ValueError("G4 positive-control evidence missing or modified")
    if not any((Path(root) / item["path"]).resolve() == path.resolve() and item["sha256"] == selected["sha256"]
               for item in gate.get("evidence", [])):
        raise ValueError("G4 positive-control file must be included in the reviewed evidence")
    rows = [row for row in read_jsonl(path) if row.get("mask_id") == selected["mask_id"]
            and row.get("checkpoint_id") == selected["checkpoint_id"]]
    candidates = residual_control_summary(rows)["candidates"]
    if len(candidates) != 1 or not candidates[0]["eligible"]:
        raise ValueError("G4 requires eligible residual evidence aligned with the declared K/V test band, excluding layer zero and retaining downstream attention")
    if candidates[0]["test_layers"] != test_layers:
        raise ValueError("G4 evidence test band differs from the selected test band")
    geometry = {row.get("model_num_hidden_layers") for row in rows}
    if len(geometry) != 1 or test_layers != list(range(1, next(iter(geometry)))):
        raise ValueError("G4 primary evidence requires every layer from 1 through the final model layer")
    return candidates[0]


def assert_training_allowed(config_dict, gate_file, profile_report):
    stage = config_dict.get("stage", "pilot")
    if stage not in ("profile", "pilot", "confirmatory"):
        raise ValueError("Unknown experiment stage")
    model = config_dict.get("model_name_or_path", config_dict.get("model_id", config_dict.get("model", "")))
    paid_required = stage == "confirmatory" or "7b" in model.lower()
    if not gate_file or (not profile_report and (stage != "profile" or paid_required)):
        raise ValueError("Training needs a calibration gate file and a completed profiling report")
    path = Path(gate_file)
    review = json.loads(path.read_text())
    calibration = review.get("calibration", {})
    if not calibration.get("frozen") or not calibration.get("manifest_sha256"):
        raise ValueError("G0: calibrate on development stories and freeze the corpus before training")
    verify_evidence(review["gates"]["G0"], path.parent)
    validate_calibration_manifest(calibration)
    if stage == "profile" and not paid_required:
        return review
    profile = json.loads(Path(profile_report).read_text())
    if profile.get("status") != "completed":
        raise ValueError("Profiling did not complete")
    if profile.get("synthetic") is not False:
        raise ValueError("A synthetic or unspecified profile cannot authorize real-model training")
    measured = ("generation_seconds", "update_seconds", "checkpoint_seconds", "peak_cuda_memory_bytes",
                "generated_tokens", "sampled_completions", "truncation_rate",
                "evaluation_seconds", "evaluation_panels", "evaluation_native_generations",
                "evaluation_generated_tokens", "max_evaluated_adapter_bytes",
                "evaluation_native_seconds", "evaluation_immediate_seconds",
                "evaluation_native_panels", "evaluation_immediate_panels")
    for field in measured:
        value = profile.get(field)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Profiling must report a finite nonnegative measurement: {field}")
    from .training.config import ARMS
    profiled_arm = profile.get("config", {}).get("arm")
    if profiled_arm not in ARMS:
        raise ValueError("Profiling measurement requires its declared training arm")
    # A target-size bootstrap profile may cite the earlier pilot profile. A full
    # training run must use measurements for its own arm, not another objective.
    requested_arm = config_dict.get("arm")
    if (requested_arm is not None and requested_arm != profiled_arm
            and stage != "profile" and not config_dict.get("profile", False)):
        raise ValueError("Training requires a profile of the same arm")
    if profile["update_seconds"] <= 0:
        raise ValueError("Profiling must measure an optimization update")
    if profiled_arm.endswith("_rl"):
        throughput = profile.get("generation_tokens_per_second")
        if (type(throughput) not in (int, float) or not math.isfinite(throughput) or throughput <= 0
                or any(profile[field] <= 0 for field in
                       ("generation_seconds", "generated_tokens", "sampled_completions"))):
            raise ValueError("RL profiling must measure actual rollout generation and sampled completions")
    else:
        # SFT has no training rollouts. Native development generation is measured
        # separately below; fabricating rollout throughput would corrupt costs.
        if (any(profile[field] != 0 for field in
                ("generation_seconds", "generated_tokens", "sampled_completions"))
                or profile.get("generation_tokens_per_second") is not None):
            raise ValueError("SFT profiling must report zero training rollouts and undefined rollout throughput")
        for field in ("supervised_tokens", "optimized_tokens", "prompt_exposures"):
            value = profile.get(field)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"SFT profiling must measure supervised optimization: {field}")
    if any(profile[field] <= 0 for field in ("evaluation_seconds", "evaluation_panels",
                   "evaluation_native_generations", "evaluation_generated_tokens", "max_evaluated_adapter_bytes",
                   "evaluation_native_seconds", "evaluation_immediate_seconds",
                   "evaluation_native_panels", "evaluation_immediate_panels")):
        raise ValueError("Profiling must measure native development evaluation and retained-adapter storage")
    if not 0 <= profile["truncation_rate"] <= 1:
        raise ValueError("Profiling truncation rate must be between zero and one")
    if paid_required:
        for gate in GATES:
            verify_evidence(review["gates"][gate], path.parent)
        verify_g1_learning_curves(review["gates"]["G1"], path.parent,
                                  expected_manifest_sha256=calibration["manifest_sha256"])
        settings = calibration.get("settings", {})
        _validate_frozen_g4_band(settings)
        test_layers = settings.get("g4_test_layers")
        if test_layers is None:
            raise ValueError("G4 pilot evidence band must be declared in frozen calibration settings.g4_test_layers")
        # This list binds the reviewed pilot evidence, not the confirmatory
        # architecture. Each model uses the frozen policy with its own geometry.
        verify_g4_positive_control(review["gates"]["G4"], path.parent, expected_test_layers=test_layers)
        paid = review.get("paid_compute", {})
        if not paid.get("approved") or not paid.get("approved_by"):
            raise ValueError("Paid 7B runs require the proposal's explicit cost approval after gate review")
        if paid.get("estimated_gbp") is None or not paid.get("exchange_rate_date"):
            raise ValueError("Record the profiling-based cost and exchange-rate date")
        estimate, limit = paid["estimated_gbp"], paid.get("gbp_limit", 500)
        if any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
               for value in (estimate, limit)):
            raise ValueError("Cost estimate and ceiling must be finite positive GBP amounts")
        try:
            date.fromisoformat(paid["exchange_rate_date"])
        except (TypeError, ValueError) as error:
            raise ValueError("Exchange-rate date must be an ISO calendar date") from error
        if estimate > min(limit, 500):
            raise ValueError("Estimate exceeds the proposal's £500 ceiling")
    return review


def freeze_calibration(review_path, manifest, settings, evidence, reviewer, rationale):
    if not isinstance(reviewer, str) or not reviewer.strip() or not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("Calibration freeze requires a named reviewer and rationale")
    path = Path(review_path)
    review = json.loads(path.read_text()) if path.exists() else initial_review()
    if review["calibration"]["frozen"]:
        raise ValueError("Calibration is already frozen; use a new preregistered experiment lineage")
    from .data import assert_calibration_corpus
    _validate_difficulty_review(settings)
    _validate_frozen_g4_band(settings)
    assert_calibration_corpus(manifest)
    review["calibration"] = {"frozen": True, "manifest_sha256": file_digest(manifest),
                             "manifest_path": str(Path(manifest).resolve()), "settings": settings}
    review["gates"]["G0"] = {"status": "pass", "reviewed_by": reviewer, "rationale": rationale,
                               "evidence": [{"path": str(Path(evidence).resolve()), "sha256": file_digest(evidence)}]}
    write_json(path, review)
    return review


def validate_calibration_manifest(calibration):
    """Old or modified corpora cannot reuse a G0 approval after the generator fix."""
    from .data import assert_calibration_corpus
    manifest = calibration.get("manifest_path")
    if not manifest or not Path(manifest).is_file():
        raise ValueError("G0 requires a current audited corpus manifest path; recalibrate old lineages")
    if file_digest(manifest) != calibration.get("manifest_sha256"):
        raise ValueError("G0 corpus manifest was modified after calibration")
    _validate_difficulty_review(calibration.get("settings", {}))
    _validate_frozen_g4_band(calibration.get("settings", {}))
    return assert_calibration_corpus(manifest)


def _validate_difficulty_review(settings):
    """Require measured difficulty and interpretation, never impose a score cutoff."""
    review = settings.get("difficulty_review")
    if not isinstance(review, dict):
        raise ValueError("G0 calibration requires settings.difficulty_review with an estimate, interval and interpretation")
    estimate, interval = review.get("estimate"), review.get("ci95")
    if (type(estimate) not in (int, float) or not math.isfinite(estimate) or not 0 <= estimate <= 1
            or not isinstance(interval, list) or len(interval) != 2
            or any(type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1
                   for value in interval) or interval[0] > interval[1]):
        raise ValueError("G0 difficulty review needs a finite probability estimate and ordered interval within [0,1]")
    interpretation = review.get("relation_to_chance_and_ceiling")
    if not isinstance(interpretation, str) or not interpretation.strip():
        raise ValueError("G0 difficulty review needs a reviewer interpretation relative to chance and ceiling")


def _validate_frozen_g4_band(settings):
    if settings.get("g4_layer_policy") != G4_LAYER_POLICY:
        raise ValueError("Frozen calibration requires g4_layer_policy='all_after_embedding_layer' for each model's layers 1 through last")
    layers = settings.get("g4_test_layers")
    if layers is not None and (not isinstance(layers, list) or not layers
            or any(type(layer) is not int or layer < 1 for layer in layers)
            or layers != list(range(1, max(layers) + 1))):
        raise ValueError("Frozen G4 pilot test layers must include every layer from 1 through the pilot model's final layer")
