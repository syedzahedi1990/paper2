"""G4 must not be certified by reconstruction-like or disconnected residuals."""
import copy

import pytest

from paper2.data import VIEWS
from paper2.gates import verify_g4_positive_control
from paper2.interventions.controls import residual_control_summary
from paper2.interventions import component_summary
from paper2.io import file_digest, write_jsonl


def controls(layers=(1,), *, positive=True, test_layers="same"):
    test_layers = list(layers) if test_layers == "same" else test_layers
    rows = []
    for pair in ("pair-a", "pair-b"):
        for control in ("natural", "own", "signed_permutation"):
            for view in VIEWS:
                target = "B" if view in ("direct", "search") else "A"
                rows.append({"checkpoint_id": "test-checkpoint", "pair_id": pair, "world_id": pair,
                             "analysis": "primary_within_checkpoint", "view": view,
                             "layers": list(layers), "mask_id": ",".join(map(str, layers)),
                             "model_num_hidden_layers": 4, "component": "residual", "control": control,
                             "g4_test_layers": test_layers,
                             # Claims do not override the native receipts or layer coordinates.
                             "g4_control_eligible": True, "positive_control": True,
                             "ordinary_target": "A", "forecast_target": target,
                             "forecasts": {"event_rewrite": target}, "forecast_probability": .5,
                             "prediction": target if positive and control == "natural" else "A",
                             "receipt": {"analysis": "primary_within_checkpoint", "control": control,
                                         "checkpoint_id": "test-checkpoint", "donor_checkpoint_id": "test-checkpoint",
                                         "residual_site": "decoder_block_output", "layer_indexing": "zero_based",
                                         "model_num_hidden_layers": 4, "g4_test_layers": test_layers,
                                         "sites": [{"layer": layer, "component": "residual", "other_token_rows_exact": True}
                                                   for layer in layers]}})
    for residual in list(rows):
        for component in ("k", "v"):
            row = copy.deepcopy(residual)
            row["component"] = component
            for site in row["receipt"]["sites"]:
                site["component"] = component
            rows.append(row)
    return rows


def gate_for(tmp_path, rows):
    path = tmp_path / "components.jsonl"
    write_jsonl(path, rows)
    evidence = {"path": path.name, "sha256": file_digest(path)}
    return {"positive_control": {**evidence, "mask_id": rows[0]["mask_id"], "checkpoint_id": "test-checkpoint",
                                 "test_layers": rows[0]["g4_test_layers"]},
            "evidence": [evidence]}


@pytest.mark.parametrize("layers,reason", [((0, 1, 2, 3), "layer_zero"), ((0,), "layer_zero"), ((3,), "downstream")])
def test_layer_zero_and_final_only_cannot_certify_g4_even_with_perfect_predictions(tmp_path, layers, reason):
    rows = controls(layers)
    summary = residual_control_summary(rows)
    assert summary["eligible_masks"] == []
    assert reason in summary["candidates"][0]["ineligible_reasons"][0]
    with pytest.raises(ValueError, match="G4"):
        verify_g4_positive_control(gate_for(tmp_path, rows), tmp_path)
    comparison = component_summary(rows, rows)
    assert comparison["g4_positive_controls"]["after"]["eligible_masks"] == []


def test_nontrivial_measured_control_reports_pattern_effect_and_requires_hashed_evidence(tmp_path):
    rows = controls((1, 2))
    summary = residual_control_summary(rows)
    assert summary["eligible_masks"] == ["1,2"]
    assert summary["empirical_gate_passed"] is False
    candidate = summary["candidates"][0]
    assert candidate["effects"]["natural|changed_accuracy"] == {"estimate": 1., "ci95": [1., 1.], "stories": 2}
    assert candidate["effects"]["natural|intact_accuracy"]["estimate"] == 0
    assert candidate["effects"]["signed_permutation|changed_accuracy"]["estimate"] == 0
    gate = gate_for(tmp_path, rows)
    assert verify_g4_positive_control(gate, tmp_path)["eligible"]
    gate["evidence"] = []
    with pytest.raises(ValueError, match="reviewed evidence"):
        verify_g4_positive_control(gate, tmp_path)


def test_geometry_does_not_make_a_zero_effect_a_positive_control(tmp_path):
    rows = controls(positive=False)
    assert residual_control_summary(rows)["eligible_masks"] == ["1"]
    with pytest.raises(ValueError, match="did not improve"):
        verify_g4_positive_control(gate_for(tmp_path, rows), tmp_path)


def test_missing_receipt_incomplete_panel_and_changed_targets_fail_closed(tmp_path):
    rows = controls()
    missing = copy.deepcopy(rows)
    missing[0]["receipt"] = None
    with pytest.raises(ValueError, match="eligible residual evidence"):
        verify_g4_positive_control(gate_for(tmp_path, missing), tmp_path)
    with pytest.raises(ValueError, match="complete eight-view"):
        residual_control_summary(rows[1:])
    changed = copy.deepcopy(rows)
    changed[0]["forecasts"]["event_rewrite"] = "C"
    with pytest.raises(ValueError, match="forecast targets"):
        residual_control_summary(changed)
    with pytest.raises(ValueError, match="selected nontrivial"):
        verify_g4_positive_control({"evidence": []}, tmp_path)


def test_interior_layers_never_certify_an_undeclared_or_different_test_band(tmp_path):
    for declared in (None, [2]):
        rows = controls((1,), test_layers=declared)
        summary = residual_control_summary(rows)
        assert summary["eligible_masks"] == []
        assert "test_band" in summary["candidates"][0]["ineligible_reasons"][0]
        with pytest.raises(ValueError, match="G4"):
            verify_g4_positive_control(gate_for(tmp_path, rows), tmp_path)


def test_control_requires_complete_same_band_key_value_receipts(tmp_path):
    rows = controls((1, 2))
    missing = [row for row in rows if row["component"] != "k"]
    assert residual_control_summary(missing)["eligible_masks"] == []
    with pytest.raises(ValueError, match="aligned with the declared K/V"):
        verify_g4_positive_control(gate_for(tmp_path, missing), tmp_path)
    bad_receipt = copy.deepcopy(rows)
    next(row for row in bad_receipt if row["component"] == "v")["receipt"]["g4_test_layers"] = [1]
    assert residual_control_summary(bad_receipt)["eligible_masks"] == []


def test_gate_binds_exact_control_band_to_frozen_calibration(tmp_path):
    rows = controls((1, 2))
    gate = gate_for(tmp_path, rows)
    assert verify_g4_positive_control(gate, tmp_path, expected_test_layers=[1, 2])["eligible"]
    with pytest.raises(ValueError, match="frozen calibration band"):
        verify_g4_positive_control(gate, tmp_path, expected_test_layers=[2])
    gate["positive_control"]["test_layers"] = [1]
    with pytest.raises(ValueError, match="exactly match"):
        verify_g4_positive_control(gate, tmp_path)
