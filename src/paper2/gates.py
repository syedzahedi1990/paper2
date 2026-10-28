"""Evidence-backed pilot review. Gates never pass by absence of evidence."""
import json
import math
from datetime import date
from pathlib import Path
from .io import file_digest, write_json

GATES = ("G0", "G1", "G2", "G3", "G4")


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


def verify_g4_positive_control(gate, root, *, expected_test_layers=None):
    """Bind G4 to measured, nontrivial residual evidence, never a layer-0 replay.

    A named reviewer still judges flat nulls, preservation and useful precision.
    This check rejects missing, structurally ineligible or nonpositive evidence;
    it does not declare G4 passed automatically from the architecture or code.
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
    effect = candidates[0]["effects"].get("natural|changed_accuracy", {})
    if effect.get("estimate") is None or effect["estimate"] <= 0:
        raise ValueError("G4 residual positive control did not improve changed-view consequence accuracy over own replay")
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
                "generated_tokens", "sampled_completions", "generation_tokens_per_second", "truncation_rate",
                "evaluation_seconds", "evaluation_panels", "evaluation_native_generations",
                "evaluation_generated_tokens", "max_evaluated_adapter_bytes",
                "evaluation_native_seconds", "evaluation_immediate_seconds",
                "evaluation_native_panels", "evaluation_immediate_panels")
    for field in measured:
        value = profile.get(field)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Profiling must report a finite nonnegative measurement: {field}")
    if any(profile[field] <= 0 for field in ("generation_seconds", "update_seconds", "generated_tokens",
                                            "sampled_completions", "generation_tokens_per_second")):
        raise ValueError("Profiling must measure actual generation and an optimization update")
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
        test_layers = calibration.get("settings", {}).get("g4_test_layers")
        if test_layers is None:
            raise ValueError("G4 test band must be declared in frozen calibration settings.g4_test_layers")
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
    _validate_frozen_g4_band(settings)
    assert_calibration_corpus(manifest, acknowledged_heuristic_risks=_heuristic_risk_review(manifest, settings))
    review["calibration"] = {"frozen": True, "manifest_sha256": file_digest(manifest),
                             "manifest_path": str(Path(manifest).resolve()), "settings": settings}
    review["gates"]["G0"] = {"status": "pass", "reviewed_by": reviewer, "rationale": rationale,
                               "evidence": [{"path": str(Path(evidence).resolve()), "sha256": file_digest(evidence)}]}
    write_json(path, review)
    return review


def validate_calibration_manifest(calibration):
    """Old or modified corpora cannot reuse a G0 approval after the generator fix."""
    from .data import assert_calibration_corpus
    _validate_frozen_g4_band(calibration.get("settings", {}))
    manifest = calibration.get("manifest_path")
    if not manifest or not Path(manifest).is_file():
        raise ValueError("G0 requires a current audited corpus manifest path; recalibrate old lineages")
    if file_digest(manifest) != calibration.get("manifest_sha256"):
        raise ValueError("G0 corpus manifest was modified after calibration")
    return assert_calibration_corpus(manifest, acknowledged_heuristic_risks=
                                     _heuristic_risk_review(manifest, calibration.get("settings", {})))


def _heuristic_risk_review(manifest, settings):
    """Known weak heuristics need a corpus-bound decision, never an implicit pass."""
    acknowledgement = settings.get("residual_heuristic_review")
    if acknowledgement is None:
        return False
    corpus_hash = json.loads(Path(manifest).read_text()).get("corpus_hash")
    if (not isinstance(acknowledgement, dict) or acknowledgement.get("accepted") is not True
            or not corpus_hash or acknowledgement.get("corpus_hash") != corpus_hash
            or not isinstance(acknowledgement.get("rationale"), str)
            or not acknowledgement["rationale"].strip()):
        raise ValueError("G0 residual heuristic review requires accepted=true, the current corpus_hash and a rationale")
    return True


def _validate_frozen_g4_band(settings):
    layers = settings.get("g4_test_layers")
    if layers is not None and (not isinstance(layers, list) or not layers
            or any(type(layer) is not int or layer < 1 for layer in layers)
            or layers != list(range(min(layers), max(layers) + 1))):
        raise ValueError("Frozen G4 test layers must be a sorted contiguous band starting after layer zero")
