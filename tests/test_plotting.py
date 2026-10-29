"""Synthetic fixtures test plot plumbing; they are not experimental findings."""
import json

import pytest

from paper2.plotting import (accuracy_reference_data, load_quadrant_data, plot_accuracy_references,
                             plot_quadrant, quadrant_data)


def point():
    return {"label": "Test fixture", "arm": "reasoning_rl", "dataset": "clean", "reader": "trained",
            "source": "trained", "edit_condition": "semantic_minimal",
            "transfer": {"estimand": "trained_minus_base_direct_answer_accuracy", "training_seeds": [1, 2, 3],
                         "effects": {"unrewarded": {"estimate": .1, "ci95": [-.05, .2], "stories": 120}}},
            "crossed": {"controls": {"trained|trained|semantic_minimal|direct_fixed": {
                "story_precision": {"editability": {"candidate_worlds": 120, "editable_worlds": 85}, "metrics": {
                    "changed_response_increase_vs_original": {"estimate": .2, "ci95": [.1, .35], "stories": 85,
                        "audit_sensitivity": {"status": "reported_sensitivity_bound", "maximum_absolute_bias": 1.5,
                                              "sensitivity_range": [-1., 1.], "simultaneous_confidence": .95}}}}}}}}


def test_quadrant_extracts_both_measured_intervals_without_labels_or_cutoffs(tmp_path):
    entry = point()
    for kind in ("transfer", "crossed"):
        (tmp_path / f"{kind}.json").write_text(json.dumps(entry.pop(kind)))
        entry[kind + "_path"] = kind + ".json"
    manifest = tmp_path / "points.json"
    manifest.write_text(json.dumps({"points": [entry]}))
    result = load_quadrant_data(manifest)
    assert result["points"][0]["x"]["estimate"] == .1
    assert result["points"][0]["y"]["ci95"] == [.1, .35]
    assert result["points"][0]["y"]["audit_sensitivity"]["sensitivity_range"] == [-1., 1.]
    assert result["points"][0]["y"]["audit_sensitivity"]["maximum_absolute_bias"] == 1.5
    assert result["points"][0]["editability"]["editable_worlds"] == 85
    assert not any("threshold" in key or "quadrant" in key or "class" in key for key in result["points"][0])
    assert result["manifest"]["sha256"]


def test_question_dependent_trace_is_not_substituted_for_fixed_direct_measurement():
    entry = point()
    control = entry["crossed"]["controls"].pop("trained|trained|semantic_minimal|direct_fixed")
    entry["crossed"]["controls"]["trained|trained|semantic_minimal|question_dependent"] = control
    with pytest.raises(ValueError, match="frozen direct-question"):
        quadrant_data([entry])


def test_unestimable_axis_is_reported_without_making_up_an_interval():
    entry = point()
    entry["transfer"]["effects"]["unrewarded"]["ci95"] = None
    result = quadrant_data([entry])
    assert result["points"] == [] and len(result["omitted"]) == 1
    with pytest.raises(ValueError, match="at least one measured point"):
        plot_quadrant(result, "unused.png")


def reference_fixture():
    rows = [{"world_id": str(world), "view": "direct", "answer": "A", "prediction": "A" if world else "B",
             "split": "confirmation", "cue_condition": "clean", "belief_depth": 1, "critical_event_kind": "move",
             "reader": "base", "format": "immediate", "dataset": "clean"} for world in range(4)]
    metric = {"score": .25, "ci95": [0., .75], "n": 4, "interval": "fixture_reference_interval"}
    audit = {"confirmation": {"audit_version": "fixture", "strata": {"depth1__move": {
        "n": 4, "world_ids": [str(i) for i in range(4)],
        "metrics": {"first_event": metric, "equal_outer": metric},
        "residual_heuristics": {"metrics": {"most_mentioned_location": metric}, "semantic_baselines": {}}}}}}
    return rows, audit


def test_accuracy_figure_uses_named_rules_and_matching_story_ids_not_relation_probabilities():
    rows, audit = reference_fixture()
    result = accuracy_reference_data(rows, audit, split="confirmation", samples=100)
    assert result["models"][0]["estimate"] == .75
    assert {row["label"] for row in result["references"]} == {"first_event", "most_mentioned_location"}
    assert result["world_identity_verified"] is True
    rows[0]["world_id"] = "different"
    with pytest.raises(ValueError, match="different worlds"):
        accuracy_reference_data(rows, audit, samples=100)


def test_plot_renderers_save_figures_with_their_actual_numeric_inputs(tmp_path):
    pytest.importorskip("matplotlib")
    quadrant = quadrant_data([point()])
    result = plot_quadrant(quadrant, tmp_path / "fixture-quadrant.png")
    assert (tmp_path / "fixture-quadrant.png").stat().st_size > 0
    assert json.loads((tmp_path / "fixture-quadrant.png.json").read_text()) == quadrant
    assert result["figure_sha256"]
    rows, audit = reference_fixture()
    reference = accuracy_reference_data(rows, audit, samples=100)
    plot_accuracy_references(reference, tmp_path / "fixture-reference.svg")
    assert (tmp_path / "fixture-reference.svg").stat().st_size > 0
