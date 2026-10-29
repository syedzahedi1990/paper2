"""Saved adapters are selected without an invented matching window."""
import pytest

from paper2.analysis import match_checkpoints


def measured(tmp_path, run, step, accuracy, identifier=None):
    arm, dataset, seed = run
    identifier = identifier or f"{arm}-{dataset}-{seed}-{step}"
    path = tmp_path / identifier
    path.mkdir(exist_ok=True)
    (path / "adapter_config.json").write_text("{}")
    (path / "adapter_model.safetensors").write_bytes(b"fixture")
    return {"arm": arm, "dataset": dataset, "seed": seed, "step": step,
            "split": "development", "view": "direct", "format": "immediate",
            "checkpoint_path": str(path), "checkpoint_id": identifier, "accuracy": accuracy}


def test_highest_common_reached_accuracy_preserves_runs_and_reports_actual_gaps(tmp_path):
    first, second = ("reasoning_rl", "clean", 1), ("reasoning_rl", "shortcut", 1)
    rows = [measured(tmp_path, first, 0, .2), measured(tmp_path, first, 50, .6),
            measured(tmp_path, first, 100, .8), measured(tmp_path, second, 0, .2),
            measured(tmp_path, second, 50, .7)]
    result = match_checkpoints(rows)
    assert result["target_accuracy"] == .7
    assert result["matches"]["reasoning_rl|clean|seed=1"]["checkpoint"]["step"] == 50
    assert result["matches"]["reasoning_rl|clean|seed=1"]["absolute_accuracy_gap"] == pytest.approx(.1)
    assert result["matches"]["reasoning_rl|shortcut|seed=1"]["actual_accuracy"] == .7
    assert not any("tolerance" in key or "pass" in key for key in result)


def test_ties_use_earlier_step_then_stable_id_and_step_zero_can_be_selected(tmp_path):
    first, second = ("label_sft", "clean", 1), ("reasoning_rl", "clean", 1)
    rows = [measured(tmp_path, first, 0, .6, "a-base-adapter"),
            measured(tmp_path, first, 50, .8), measured(tmp_path, second, 0, .7, "b-base-adapter"),
            measured(tmp_path, second, 0, .7, "a-second-base-adapter")]
    rows.append({"arm": "base", "split": "development", "format": "immediate", "view": "direct", "accuracy": .1})
    result = match_checkpoints(rows)
    assert result["target_accuracy"] == .7 and result["excluded_base_records"] == 1
    assert result["matches"]["label_sft|clean|seed=1"]["checkpoint"]["step"] == 0
    assert result["matches"]["reasoning_rl|clean|seed=1"]["checkpoint"]["checkpoint_id"] == "a-second-base-adapter"


def test_expected_comparison_set_detects_missing_runs_without_assuming_whole_grid(tmp_path):
    run = ("reasoning_rl", "clean", 2)
    rows = [measured(tmp_path, run, 50, .8)]
    expected = [{"arm": run[0], "dataset": run[1], "seed": run[2]}]
    assert match_checkpoints(rows, expected_runs=expected)["comparison_set_source"] == "validated_expected_runs"
    with pytest.raises(ValueError, match="missing="):
        match_checkpoints(rows, expected_runs=expected + [{**expected[0], "seed": 3}])
    with pytest.raises(ValueError, match="duplicates"):
        match_checkpoints(rows, expected_runs=expected * 2)


def test_conflicting_measurements_and_nonfinite_accuracies_are_rejected(tmp_path):
    row = measured(tmp_path, ("reasoning_rl", "clean", 1), 50, .8)
    with pytest.raises(ValueError, match="Conflicting"):
        match_checkpoints([row, {**row, "accuracy": .7}])
    with pytest.raises(ValueError, match="measured accuracy"):
        match_checkpoints([{**row, "accuracy": float("nan")}])
    assert match_checkpoints([row, row])["matches"]["reasoning_rl|clean|seed=1"]["evaluated_checkpoints"] == 1
