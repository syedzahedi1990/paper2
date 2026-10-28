import pytest

from paper2.gates import freeze_calibration, validate_calibration_manifest
from paper2.io import file_digest, write_json


def test_g0_requires_explicit_corpus_bound_weak_heuristic_review(tmp_path, monkeypatch):
    manifest, evidence = tmp_path / "manifest.json", tmp_path / "evidence.json"
    write_json(manifest, {"corpus_hash": "actual-corpus"})
    write_json(evidence, {"base_accuracy": .5})
    calls = []

    def audit(path, *, acknowledged_heuristic_risks=False):
        calls.append(acknowledged_heuristic_risks)
        if not acknowledged_heuristic_risks:
            raise ValueError("Residual heuristics require review")
        return {"calibration_ready": False, "required_controls_passed": True}

    monkeypatch.setattr("paper2.data.assert_calibration_corpus", audit)
    path = tmp_path / "review.json"
    with pytest.raises(ValueError, match="Residual heuristics"):
        freeze_calibration(path, manifest, {}, evidence, "Reviewer", "Development calibration")
    assert not path.exists()
    settings = {"residual_heuristic_review": {"accepted": True, "corpus_hash": "wrong", "rationale": "Known risks"}}
    with pytest.raises(ValueError, match="current corpus_hash"):
        freeze_calibration(path, manifest, settings, evidence, "Reviewer", "Development calibration")
    settings["residual_heuristic_review"]["corpus_hash"] = "actual-corpus"
    review = freeze_calibration(path, manifest, settings, evidence, "Reviewer", "Development calibration")
    assert calls == [False, True]
    assert validate_calibration_manifest(review["calibration"])["calibration_ready"] is False
    write_json(manifest, {"corpus_hash": "different-corpus"})
    review["calibration"]["manifest_sha256"] = file_digest(manifest)
    with pytest.raises(ValueError, match="current corpus_hash"):
        validate_calibration_manifest(review["calibration"])
