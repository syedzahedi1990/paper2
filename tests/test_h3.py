"""H3 fixtures exercise estimands and provenance, not measured research outcomes."""
import copy
import json

import pytest

from paper2.data import VIEWS
from paper2.h3 import h3_cross_arm_data, load_h3_data
from paper2.io import file_digest, write_json, write_jsonl
from paper2.plotting import plot_h3
from paper2.training.config import ARMS


def corpus_manifest(tmp_path):
    path = tmp_path / "corpus.json"
    write_json(path, {"files": {"evaluation/confirmation__clean.jsonl": {"sha256": "c" * 64},
                                 "pairs/confirmation.jsonl": {"sha256": "d" * 64}}})
    return path


@pytest.fixture(scope="module")
def raw_panel():
    def evaluation(seed, arm):
        return [{"world_id": f"w{world:03}", "view": view, "answer": "A",
                 "prediction": "A" if seed == 1 and world < 60 else "B",
                 "training_seed": seed, "arm": arm, "split": "confirmation", "cue_condition": "clean",
                 "cue_location": "box", "format": "immediate"}
                for world in range(120) for view in VIEWS]

    def components(rows, checkpoint):
        output = []
        for row in rows:
            forecast = "B" if row["view"] == "direct" else "A"
            for component in ("k", "v", "residual"):
                for control in ("natural", "own", "signed_permutation"):
                    successful = row["training_seed"] == 1 and int(row["world_id"][1:]) < 60 and control == "natural"
                    output.append({**row, "pair_id": "pair-" + row["world_id"],
                                   "checkpoint_id": checkpoint, "donor_checkpoint_id": checkpoint,
                                   "analysis": "primary_within_checkpoint", "forecast_account": "event_rewrite",
                                   "model_num_hidden_layers": 3, "layers": [1, 2], "mask_id": "1,2",
                                   "component": component, "control": control, "ordinary_target": "A",
                                   "forecast_target": forecast, "forecasts": {"event_rewrite": forecast},
                                   "prediction": forecast if successful else row["prediction"]})
        return output

    before = evaluation(None, None)
    base = {"evaluation": before, "components": components(before, "model"), "checkpoint": "model",
            "model": "model", "revision": "a" * 40}
    entries = []
    for arm in ARMS:
        for dataset in ("clean", "shortcut"):
            runs = []
            for seed in (1, 2, 3):
                checkpoint = f"{arm}/{dataset}/{seed}"
                rows = evaluation(seed, arm)
                runs.append({"seed": seed, "checkpoint": checkpoint, "evaluation": rows,
                             "components": components(rows, checkpoint), "training_manifest": {
                                 "config": {"arm": arm, "dataset": dataset, "seed": seed, "stage": "confirmatory",
                                            "model": "model", "revision": "a" * 40},
                                 "corpus_manifest_sha256": "b" * 64, "status": "complete", "synthetic_integration_test": False}})
            entries.append({"arm": arm, "dataset": dataset, "runs": runs})
    return base, entries


def test_h3_averages_seeds_within_story_and_keeps_paired_own_and_null_controls(raw_panel):
    base, entries = raw_panel
    result = h3_cross_arm_data(base, entries, samples=100)
    assert len(result["points"]) == 24
    for point in result["points"]:
        assert point["x"]["estimate"] == pytest.approx(1 / 6)
        assert point["y"]["estimate"] == pytest.approx(1 / 6)
        assert point["x"]["stories"] == point["y"]["stories"] == 120
        assert point["x"]["ci95"] == point["y"]["ci95"]
        assert point["controls"]["signed_permutation|training_change_in_effect"]["estimate"] == 0
        assert point["controls"]["own|training_change_in_effect"]["estimate"] == 0
    assert result["layers"] == [1, 2]
    assert result["training_seeds"] == [1, 2, 3]


def test_h3_y_is_change_in_paired_effect_not_change_in_raw_natural_success(raw_panel):
    original_base, entries = raw_panel
    base = dict(original_base)
    base["evaluation"] = [{**row, "prediction": "B" if row["view"] == "direct" else "A"}
                          if int(row["world_id"][1:]) < 30 else row for row in base["evaluation"]]
    base["components"] = [{**row, "prediction": row["forecast_target"]}
                          if int(row["world_id"][1:]) < 30 and row["control"] == "own" else row
                          for row in base["components"]]
    result = h3_cross_arm_data(base, entries, samples=100)
    for point in result["points"]:
        assert point["controls"]["natural|before_effect"]["estimate"] == -.25
        assert point["controls"]["natural|after_effect"]["estimate"] == pytest.approx(1 / 6)
        assert point["y"]["estimate"] == pytest.approx(5 / 12)
        assert point["x"]["estimate"] == pytest.approx(-1 / 12)


@pytest.mark.parametrize("defect,message", [
    ("point", "eight confirmatory"), ("seed", "exactly training seeds"),
    ("dataset", "training provenance"), ("checkpoint", "component identity"),
    ("story", "stories/ordinary targets"), ("null", "complete primary"),
    ("view", "complete view"), ("forecast", "forecast support"),
])
def test_h3_rejects_mixed_or_incomplete_support(raw_panel, defect, message):
    base, original = raw_panel
    entries = [dict(entry, runs=list(entry["runs"])) for entry in original]
    run = copy.copy(entries[0]["runs"][0])
    entries[0]["runs"][0] = run
    if defect == "point":
        entries.pop()
    elif defect == "seed":
        entries[0]["runs"] = entries[0]["runs"][:2]
    elif defect == "dataset":
        run["training_manifest"] = copy.deepcopy(run["training_manifest"])
        run["training_manifest"]["config"]["dataset"] = "shortcut"
    else:
        run["components"] = list(run["components"])
        if defect in ("null", "view"):
            run["components"] = [row for row in run["components"] if not (
                row["world_id"] == "w000" and row["component"] == "k" and row["control"] == "signed_permutation"
                and (defect == "null" or row["view"] == "direct"))]
        elif defect == "forecast":
            run["components"] = [{**row, "forecasts": {"event_rewrite": "C"}, "forecast_target": "C"}
                                 if row["view"] == "direct" else row for row in run["components"]]
        else:
            field = "checkpoint_id" if defect == "checkpoint" else "world_id"
            run["components"][0] = {**run["components"][0], field: "wrong"}
    with pytest.raises(ValueError, match=message):
        h3_cross_arm_data(base, entries, samples=10)


def test_h3_loader_rejects_modified_output_receipt(tmp_path, raw_panel):
    base, _ = raw_panel
    corpus = corpus_manifest(tmp_path)
    for name, command in (("evaluation", "evaluate"), ("components", "components")):
        path = tmp_path / f"{name}.jsonl"
        write_jsonl(path, base[name])
        write_json(str(path) + ".run.json", {"status": "completed", "output_sha256": file_digest(path),
                   "input_sha256": "c" * 64, "checkpoint": "model", "resolved_revision": "a" * 40,
                   "adapter_sha256": None,
                   "command": {"command": command, "training_seed": None, "model": "model"}})
    manifest = tmp_path / "h3.json"
    write_json(manifest, {"checkpoint_selection": "final", "corpus_manifest_path": corpus.name, "base": {
        "evaluation_path": "evaluation.jsonl", "components_path": "components.jsonl"}, "points": []})
    with (tmp_path / "evaluation.jsonl").open("a") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="receipt/hash"):
        load_h3_data(manifest)


@pytest.mark.parametrize("selection,defect,message", [
    ("final", None, None), ("matched", None, None),
    ("final", "adapter_missing", "must record adapter_sha256"),
    ("final", "adapter_disagree", "adapter hashes differ"),
    ("final", "adapter_registry", "adapter hash differs from the selected"),
    ("final", "adapter_invalid", "valid adapter hash"),
    ("final", "base_adapter", "base must have a null adapter"),
    ("final", "corpus_missing", "requires corpus_manifest_path"),
    ("final", "corpus_tampered", "training corpus hash differs"),
    ("final", "cross_corpus", "training corpus hash differs"),
    ("final", "corpus_input_missing", "must bind clean confirmation"),
    ("final", "input_missing", "require source-input hashes"),
    ("final", "input_mismatch", "base inference inputs differ"),
])
def test_h3_loader_binds_registered_checkpoints_and_verifies_matching(tmp_path, monkeypatch, selection, defect, message):
    # Loader contracts are tested with small opaque payloads; the full 120-story
    # numerical/support validation is exercised independently above.
    corpus = corpus_manifest(tmp_path)
    def outputs(prefix, checkpoint, seed, arm):
        paths = {}
        for kind, command in (("evaluation", "evaluate"), ("components", "components")):
            filename = f"{prefix}-{kind}.jsonl"
            path = tmp_path / filename
            write_jsonl(path, [{"fixture": prefix}])
            write_json(str(path) + ".run.json", {"status": "completed", "output_sha256": file_digest(path),
                       "input_sha256": ("c" if kind == "evaluation" else "d") * 64,
                       "checkpoint": checkpoint, "resolved_revision": "a" * 40,
                       "adapter_sha256": None if seed is None else "e" * 64,
                       "command": {"command": command, "training_seed": seed, "arm": arm, "model": "model"}})
            paths[kind + "_path"] = filename
        return paths

    manifest = {"checkpoint_selection": selection, "corpus_manifest_path": corpus.name,
                "base": outputs("base", "model", None, None), "points": []}
    selected_paths = []
    for arm in ARMS:
        for dataset in ("clean", "shortcut"):
            point = {"arm": arm, "dataset": dataset, "runs": []}
            for seed in (1, 2, 3):
                prefix = f"{arm}-{dataset}-{seed}"
                limiting = not selected_paths
                step = 2 if selection == "final" or limiting else 1
                checkpoint = f"{prefix}/checkpoint-{step}"
                selected_paths.append(checkpoint)
                spec = {**outputs(prefix, checkpoint, seed, arm), "seed": seed,
                        "training_manifest_path": prefix + ".json", "checkpoint_evaluations_path": prefix + "-registry.jsonl"}
                identity = {"arm": arm, "dataset": dataset, "seed": seed, "model": "model", "revision": "a" * 40}
                write_json(tmp_path / spec["training_manifest_path"], {"config": {**identity, "stage": "confirmatory", "immediate_eval_steps": 1},
                           "corpus_manifest_sha256": file_digest(corpus),
                           "training_budget": {"resolved_max_steps": 2}})
                write_jsonl(tmp_path / spec["checkpoint_evaluations_path"], [
                    {**identity, "step": i, "format": "immediate", "split": "development", "view": "direct",
                     "checkpoint_id": f"{prefix}/checkpoint-{i}", "adapter_sha256": "e" * 64,
                     "accuracy": accuracy} for i, accuracy in enumerate([.1, .2, .4] if limiting else [.1, .4, .6])])
                point["runs"].append(spec)
            manifest["points"].append(point)
    path = tmp_path / "manifest.json"
    write_json(path, manifest)
    observed = []

    def record(base, entries):
        observed.extend(run["checkpoint"] for entry in entries for run in entry["runs"])
        assert base["checkpoint"] == "model"
        return {"figure": "fixture"}

    monkeypatch.setattr("paper2.h3.h3_cross_arm_data", record)
    if defect:
        run = manifest["points"][0]["runs"][0]
        if defect.startswith("adapter") or defect in ("base_adapter", "input_missing", "input_mismatch"):
            specification = manifest["base"] if defect in ("base_adapter", "input_missing", "input_mismatch") else run
            for kind in ("evaluation", "components"):
                receipt_path = tmp_path / (specification[kind + "_path"] + ".run.json")
                receipt = json.loads(receipt_path.read_text())
                if defect == "adapter_missing":
                    receipt.pop("adapter_sha256")
                elif defect == "adapter_disagree":
                    receipt["adapter_sha256"] = ("f" if kind == "evaluation" else "e") * 64
                elif defect in ("adapter_registry", "base_adapter"):
                    receipt["adapter_sha256"] = "f" * 64
                elif defect == "adapter_invalid":
                    receipt["adapter_sha256"] = "z" * 64
                elif defect == "input_missing":
                    receipt.pop("input_sha256")
                else:
                    receipt["input_sha256"] = "f" * 64
                write_json(receipt_path, receipt)
        elif defect == "corpus_missing":
            manifest.pop("corpus_manifest_path")
            write_json(path, manifest)
        elif defect in ("corpus_tampered", "corpus_input_missing"):
            data = json.loads(corpus.read_text())
            if defect == "corpus_tampered":
                data["changed_after_training"] = True
            else:
                data["files"].pop("pairs/confirmation.jsonl")
            write_json(corpus, data)
        else:
            training = tmp_path / run["training_manifest_path"]
            data = json.loads(training.read_text())
            data["corpus_manifest_sha256"] = "f" * 64
            write_json(training, data)
        with pytest.raises(ValueError, match=message):
            load_h3_data(path)
        assert observed == []
        return
    result = load_h3_data(path)
    assert observed == selected_paths and len(observed) == 24
    assert result["checkpoint_selection"] == selection
    assert result["corpus_manifest"]["sha256"] == file_digest(corpus)
    assert result["confirmation_input_sha256"] == {"evaluation": "c" * 64, "components": "d" * 64}
    if selection == "matched":
        registry_path = tmp_path / manifest["points"][0]["runs"][1]["checkpoint_evaluations_path"]
        registry = [json.loads(line) for line in registry_path.read_text().splitlines()]
        registry[1]["accuracy"] = .1
        write_jsonl(registry_path, registry)
        with pytest.raises(ValueError, match="matching rule"):
            load_h3_data(path)


def test_h3_plot_has_three_complete_panels_and_numeric_sidecar(tmp_path, raw_panel):
    pytest.importorskip("matplotlib")
    data = h3_cross_arm_data(*raw_panel, samples=100)
    output = tmp_path / "h3.png"
    result = plot_h3(data, output)
    assert output.stat().st_size > 0 and result["figure_sha256"]
    assert json.loads(output.with_suffix(".png.json").read_text()) == data
    data["points"].pop()
    with pytest.raises(ValueError, match="all eight"):
        plot_h3(data, output)
