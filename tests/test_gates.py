import json

import pytest

from paper2.gates import assert_training_allowed, freeze_calibration, initial_review
from paper2.io import file_digest, write_json


@pytest.fixture
def reviewed(tmp_path):
    evidence = tmp_path / "evidence.json"
    write_json(evidence, {"measured": True})
    review = initial_review()
    review["calibration"] = {"frozen": True, "manifest_sha256": "0" * 64, "settings": {}}
    review["gates"] = {name: {"status": "pass", "reviewed_by": "Reviewer", "rationale": "Audited measurements",
                              "evidence": [{"path": evidence.name, "sha256": file_digest(evidence)}]}
                       for name in review["gates"]}
    review["paid_compute"] = {"approved": True, "approved_by": "User", "estimated_gbp": 100,
                               "gbp_limit": 500, "exchange_rate_date": "2026-10-28"}
    gate = tmp_path / "gates.json"
    write_json(gate, review)
    profile = {"status": "completed", "synthetic": False, "generation_seconds": 2, "update_seconds": 1,
               "checkpoint_seconds": .1, "peak_cuda_memory_bytes": 1000, "generated_tokens": 16,
               "sampled_completions": 8, "generation_tokens_per_second": 8, "truncation_rate": 0}
    report = tmp_path / "profile.json"
    write_json(report, profile)
    return gate, report, review, profile


def test_pilot_uses_measured_profile_and_does_not_need_paid_approval(reviewed):
    gate, report, review, profile = reviewed
    review["paid_compute"]["approved"] = False
    write_json(gate, review)
    assert_training_allowed({"stage": "pilot", "model": "Qwen2.5-3B-Instruct"}, gate, report)
    profile["synthetic"] = True
    write_json(report, profile)
    with pytest.raises(ValueError, match="synthetic"):
        assert_training_allowed({"stage": "pilot", "model": "Qwen2.5-3B-Instruct"}, gate, report)


def test_7b_profiling_cannot_bypass_paid_gate(reviewed):
    gate, report, review, _ = reviewed
    assert_training_allowed({"stage": "profile", "model": "Qwen2.5-3B-Instruct"}, gate, None)
    with pytest.raises(ValueError, match="profiling report"):
        assert_training_allowed({"stage": "profile", "model": "Qwen2.5-7B-Instruct"}, gate, None)
    review["paid_compute"]["approved"] = False
    write_json(gate, review)
    with pytest.raises(ValueError, match="explicit cost approval"):
        assert_training_allowed({"stage": "profile", "model": "Qwen2.5-7B-Instruct"}, gate, report)


@pytest.mark.parametrize("estimate", [float("nan"), float("inf"), -1, 0, 501])
def test_nonfinite_negative_and_overbudget_estimates_fail_closed(reviewed, estimate):
    gate, report, review, _ = reviewed
    review["paid_compute"]["estimated_gbp"] = estimate
    gate.write_text(json.dumps(review))  # Deliberately malformed external JSON.
    with pytest.raises(ValueError, match="finite positive|ceiling"):
        assert_training_allowed({"stage": "confirmatory", "model": "Qwen2.5-7B-Instruct"}, gate, report)


def test_evidence_tampering_and_empty_profiles_are_rejected(reviewed):
    gate, report, review, _ = reviewed
    write_json(report, {"status": "completed", "synthetic": False})
    with pytest.raises(ValueError, match="measurement"):
        assert_training_allowed({"stage": "pilot", "model": "Qwen2.5-3B-Instruct"}, gate, report)
    evidence = gate.parent / review["gates"]["G0"]["evidence"][0]["path"]
    evidence.write_text("changed")
    with pytest.raises(ValueError, match="modified"):
        assert_training_allowed({"stage": "profile", "model": "Qwen2.5-3B-Instruct"}, gate, None)


def test_freeze_requires_named_review_before_writing(tmp_path):
    manifest = tmp_path / "manifest.json"
    write_json(manifest, {})
    with pytest.raises(ValueError, match="named reviewer"):
        freeze_calibration(tmp_path / "gates.json", manifest, {}, manifest, "", "reason")
    assert not (tmp_path / "gates.json").exists()
