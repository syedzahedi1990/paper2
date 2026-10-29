"""Descriptive cross-arm H3 figure from complete, provenance-bound raw panels."""
from collections import defaultdict
from decimal import Decimal
import json
import math
from pathlib import Path

import numpy as np

from .analysis import cluster_interval, transfer
from .data import VIEWS
from .interventions.scoring import score_consequence_pattern
from .io import digest, file_digest, read_jsonl
from .training.config import ARMS

COMPONENTS = ("k", "v", "residual")
CONTROLS = ("natural", "own", "signed_permutation")


def _sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _evaluation(rows, seed, arm):
    indexed = {}
    for row in rows:
        if (row.get("format") != "immediate" or row.get("split") != "confirmation"
                or row.get("cue_condition") != "clean" or row.get("training_seed") != seed
                or (arm is not None and row.get("arm") != arm)):
            raise ValueError("H3 requires immediate clean confirmation evaluations with matching arm and seed")
        key = row["world_id"], row["view"]
        if key in indexed:
            raise ValueError("H3 evaluation repeats a story/view")
        indexed[key] = row
    worlds = {world for world, _ in indexed}
    if len(worlds) != 120 or set(indexed) != {(world, view) for world in worlds for view in VIEWS}:
        raise ValueError("H3 requires complete eight-view panels on all 120 confirmation stories")
    return indexed


def _patterns(rows, evaluation, checkpoint, seed, arm):
    groups, pairs, geometry, forecasts = defaultdict(dict), {}, set(), {}
    for row in rows:
        if row.get("control") not in CONTROLS:
            continue
        if row.get("analysis") != "primary_within_checkpoint":
            raise ValueError("H3 cannot use secondary cross-model swaps")
        layers = row.get("model_num_hidden_layers")
        if type(layers) is not int or layers < 2:
            raise ValueError("H3 requires recorded model layer geometry")
        # Additional single-layer diagnostics are retained in their source file,
        # but never substituted for the preregistered full primary path.
        primary = list(range(1, layers))
        if row.get("layers") != primary:
            continue
        if (row.get("mask_id") != ",".join(map(str, primary))
                or row.get("checkpoint_id") != checkpoint
                or row.get("donor_checkpoint_id") != checkpoint
                or row.get("training_seed") != seed or row.get("split") != "confirmation"
                or row.get("forecast_account") != "event_rewrite"
                or (arm is not None and row.get("arm") != arm)
                or row.get("component") not in COMPONENTS):
            raise ValueError("H3 component identity, forecast account or primary mask differs")
        world, view = row["world_id"], row["view"]
        target = evaluation.get((world, view))
        if target is None or target["answer"] != row.get("ordinary_target"):
            raise ValueError("H3 component stories/ordinary targets differ from Test 1")
        if row["control"] == "own" and row["prediction"] != target["prediction"]:
            raise ValueError("H3 own-control prediction differs from the same checkpoint's Test 1 answer")
        if row.get("forecast_target") != row.get("forecasts", {}).get("event_rewrite"):
            raise ValueError("H3 component forecast target is inconsistent")
        pair = row["pair_id"]
        if world in pairs and pairs[world] != pair:
            raise ValueError("H3 requires one matched event pair per story")
        pairs[world] = pair
        forecast = (row["ordinary_target"], row["forecast_target"], row["forecasts"])
        if (world, view) in forecasts and forecasts[world, view] != forecast:
            raise ValueError("H3 forecasts differ across component controls")
        forecasts[world, view] = forecast
        key = world, row["component"], row["control"]
        if view in groups[key]:
            raise ValueError("H3 component panel repeats a view")
        groups[key][view] = row
        geometry.add(layers)
    worlds = {world for world, _ in evaluation}
    if (len(geometry) != 1 or len(set(pairs.values())) != len(worlds)
            or set(groups) != {(world, component, control) for world in worlds
                              for component in COMPONENTS for control in CONTROLS}):
        raise ValueError("H3 requires complete primary K/V/residual natural/own/null panels")
    patterns = {}
    for key, observations in groups.items():
        patterns[key] = score_consequence_pattern(
            {view: row["prediction"] for view, row in observations.items()},
            {view: row["ordinary_target"] for view, row in observations.items()},
            {view: row["forecasts"] for view, row in observations.items()})["joint_correct"]
    return patterns, {"pairs": pairs, "forecasts_sha256": digest(sorted(forecasts.items())),
                      "num_hidden_layers": geometry.pop()}


def h3_cross_arm_data(base, entries, *, seed=2026, samples=20_000):
    """Compute both axes; entries contain the loader's raw, verified artifacts.

    Three fixed training seeds are averaged inside each story. Every bootstrap
    uses the same ordered stories and resampling seed, preserving paired support.
    No cross-arm inferential association is estimated from the eight points.
    """
    expected = {(arm, dataset) for arm in ARMS for dataset in ("clean", "shortcut")}
    identities = [(entry["arm"], entry["dataset"]) for entry in entries]
    if len(identities) != 8 or set(identities) != expected:
        raise ValueError("H3 requires exactly the eight confirmatory arm/dataset combinations")
    baseline = _evaluation(base["evaluation"], None, None)
    old, support = _patterns(base["components"], baseline, base["checkpoint"], None, None)
    points = []
    corpus_hashes = set()
    for entry in sorted(entries, key=lambda item: (item["arm"], item["dataset"])):
        runs = entry["runs"]
        if len(runs) != 3 or {run["seed"] for run in runs} != {1, 2, 3}:
            raise ValueError("H3 requires exactly training seeds 1, 2 and 3 for every point")
        trained_rows, outcomes, checkpoints, sources = [], [], {}, []
        for run in sorted(runs, key=lambda item: item["seed"]):
            config = run["training_manifest"]["config"]
            if (config.get("arm") != entry["arm"] or config.get("dataset") != entry["dataset"]
                    or config.get("seed") != run["seed"] or config.get("stage") != "confirmatory"
                    or config.get("model") != base["model"] or config.get("revision") != base["revision"]
                    or run["training_manifest"].get("status") != "complete"
                    or run["training_manifest"].get("synthetic_integration_test") is not False):
                raise ValueError("H3 training provenance differs from the declared arm/dataset/seed/base")
            corpus = run["training_manifest"].get("corpus_manifest_sha256")
            if not isinstance(corpus, str) or len(corpus) != 64:
                raise ValueError("H3 requires the frozen training corpus hash")
            corpus_hashes.add(corpus)
            current = _evaluation(run["evaluation"], run["seed"], entry["arm"])
            if set(current) != set(baseline):
                raise ValueError("H3 seed/checkpoint story supports differ")
            for key, row in current.items():
                if (row["answer"], row.get("cue_location")) != (baseline[key]["answer"], baseline[key].get("cue_location")):
                    raise ValueError("H3 ordinary targets or cue renderings differ between checkpoints")
            new, run_support = _patterns(run["components"], current, run["checkpoint"], run["seed"], entry["arm"])
            if support != run_support:
                raise ValueError("H3 matched pairs, forecast support or architecture differ")
            trained_rows.extend(run["evaluation"])
            outcomes.append(new)
            checkpoints[str(run["seed"])] = run["checkpoint"]
            sources.append(run.get("input_sources"))
        x = transfer(base["evaluation"], trained_rows, seed, samples)["effects"]["unrewarded"]
        for component in COMPONENTS:
            measurements = {}
            for control in CONTROLS:
                for quantity in ("before", "after", "before_effect", "after_effect", "training_change_in_effect"):
                    values = {}
                    for world in sorted(support["pairs"]):
                        b = float(old[world, component, control])
                        bo = float(old[world, component, "own"])
                        a = float(np.mean([panel[world, component, control] for panel in outcomes]))
                        ao = float(np.mean([panel[world, component, "own"] for panel in outcomes]))
                        values[world] = [{"before": b, "after": a, "before_effect": b - bo,
                                          "after_effect": a - ao, "training_change_in_effect": (a - ao) - (b - bo)}[quantity]]
                    measurements[f"{control}|{quantity}"] = cluster_interval(values, seed, samples)
            points.append({"label": entry.get("label", f"{entry['arm']} / {entry['dataset']}"),
                           "arm": entry["arm"], "dataset": entry["dataset"], "component": component,
                           "x": x, "y": measurements["natural|training_change_in_effect"],
                           "controls": measurements, "training_seeds": [1, 2, 3],
                           "checkpoints": checkpoints, "input_sources": sources})
    if len(corpus_hashes) != 1:
        raise ValueError("H3 compared runs must share one frozen corpus")
    return {"schema_version": 1, "figure": "h3_descriptive_cross_arm", "points": points,
            "x_label": "Unrewarded no-trace accuracy transfer (trained − base)",
            "y_label": "Change in natural − own joint event-rewrite correctness",
            "evaluation_condition": "clean", "split": "confirmation", "training_seeds": [1, 2, 3],
            "layers": list(range(1, support["num_hidden_layers"])),
            "world_ids_sha256": digest(sorted(support["pairs"])), "support": support,
            "corpus_manifest_sha256": next(iter(corpus_hashes)), "base_sources": base.get("input_sources"),
            "bootstrap_samples": samples, "bootstrap_seed": seed,
            "uncertainty": "Paired whole-story percentile 95% intervals; three fixed seeds averaged within story.",
            "interpretation": "Eight descriptive arm/dataset points per component; no correlation test, fitted trend, p-value or category."}


def load_h3_data(manifest_path):
    """Load raw JSONL outputs and verify their completed inference receipts.

    Paths are relative to the manifest. Every output requires its original
    ``.run.json`` receipt. Trained checkpoints must be registered in the supplied
    checkpoint_evaluations JSONL, which binds the retained adapter hash.
    """
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("checkpoint_selection") not in ("final", "matched"):
        raise ValueError("H3 manifest must identify final or development-matched checkpoint selection")

    def artifact(name, item, sources, *, jsonl=False):
        if not isinstance(item.get(name + "_path"), str) or not item[name + "_path"]:
            raise ValueError(f"H3 requires {name}_path")
        source = (path.parent / item[name + "_path"]).resolve()
        sources[name] = {"path": str(source), "sha256": file_digest(source)}
        return read_jsonl(source) if jsonl else json.loads(source.read_text(encoding="utf-8"))

    corpus_sources = {}
    corpus = artifact("corpus_manifest", manifest, corpus_sources)
    corpus_sha256 = corpus_sources["corpus_manifest"]["sha256"]
    expected_inputs = {kind: corpus.get("files", {}).get(filename, {}).get("sha256") for kind, filename in (
        ("evaluation", "evaluation/confirmation__clean.jsonl"), ("components", "pairs/confirmation.jsonl"))}
    if any(not _sha256(value) for value in expected_inputs.values()):
        raise ValueError("H3 corpus manifest must bind clean confirmation evaluation and pair input hashes")

    def outputs(item, expected_seed, expected_arm):
        result, sources, receipts = {}, {}, {}
        for kind, command in (("evaluation", "evaluate"), ("components", "components")):
            result[kind] = artifact(kind, item, sources, jsonl=True)
            receipt_path = Path(sources[kind]["path"] + ".run.json")
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            args = receipt.get("command", {})
            if (receipt.get("status") != "completed" or receipt.get("output_sha256") != sources[kind]["sha256"]
                    or args.get("command") != command or args.get("training_seed") != expected_seed
                    or (expected_arm is not None and args.get("arm") != expected_arm)):
                raise ValueError("H3 output receipt/hash/arm/seed is inconsistent")
            sources[kind + "_receipt"] = {"path": str(receipt_path), "sha256": file_digest(receipt_path)}
            receipts[kind] = receipt
        left, right = receipts["evaluation"], receipts["components"]
        for field in ("checkpoint", "resolved_revision"):
            if not left.get(field) or left[field] != right.get(field):
                raise ValueError("H3 Test 1 and Test 3 use different checkpoints or revisions")
        if left["command"].get("model") != right["command"].get("model"):
            raise ValueError("H3 inference base models differ")
        if any("adapter_sha256" not in receipt for receipt in receipts.values()):
            raise ValueError("H3 inference receipts must record adapter_sha256, including null for the base")
        if left["adapter_sha256"] != right["adapter_sha256"]:
            raise ValueError("H3 Test 1 and Test 3 adapter hashes differ")
        if expected_seed is None:
            if left["adapter_sha256"] is not None or any(receipt["command"].get("adapter") is not None for receipt in receipts.values()):
                raise ValueError("H3 base must have a null adapter hash and no adapter")
        elif not _sha256(left["adapter_sha256"]):
            raise ValueError("H3 trained inference receipts require a valid adapter hash")
        result.update(checkpoint=left["checkpoint"], model=left["command"]["model"],
                      revision=left["resolved_revision"], input_sources=sources,
                      adapter_sha256=left["adapter_sha256"],
                      input_hashes={kind: receipt.get("input_sha256") for kind, receipt in receipts.items()})
        if any(not _sha256(value) for value in result["input_hashes"].values()):
            raise ValueError("H3 inference receipts require source-input hashes")
        return result

    base = outputs(manifest["base"], None, None)
    if base["checkpoint"] != base["model"]:
        raise ValueError("H3 base reference must be the untrained model, without an adapter")
    if base["input_hashes"] != expected_inputs:
        raise ValueError("H3 base inference inputs differ from the declared confirmation corpus")
    entries = []
    registries = []
    for item in manifest["points"]:
        entry = {"arm": item["arm"], "dataset": item["dataset"], "label": item.get("label", f"{item['arm']} / {item['dataset']}"), "runs": []}
        for spec in item["runs"]:
            run = outputs(spec, spec["seed"], item["arm"])
            if run["input_hashes"] != base["input_hashes"]:
                raise ValueError("H3 checkpoints were evaluated on different source inputs")
            run["seed"] = spec["seed"]
            run["training_manifest"] = artifact("training_manifest", spec, run["input_sources"])
            if run["training_manifest"].get("corpus_manifest_sha256") != corpus_sha256:
                raise ValueError("H3 training corpus hash differs from the declared corpus manifest")
            registry = artifact("checkpoint_evaluations", spec, run["input_sources"], jsonl=True)
            immediate = [row for row in registry if row.get("format") == "immediate"]
            config = run["training_manifest"]["config"]
            final_step = run["training_manifest"]["training_budget"]["resolved_max_steps"]
            cadence = config.get("immediate_eval_steps")
            identity = {"arm": item["arm"], "dataset": item["dataset"], "seed": spec["seed"],
                        "model": base["model"], "revision": base["revision"], "split": "development", "view": "direct"}
            if (type(final_step) is not int or final_step < 1 or type(cadence) is not int or cadence < 1
                    or len(immediate) != len({row.get("step") for row in immediate})
                    or {row.get("step") for row in immediate} != {0, final_step, *range(cadence, final_step + 1, cadence)}
                    or any(any(row.get(key) != value for key, value in identity.items())
                           or type(row.get("accuracy")) not in (int, float)
                           or not math.isfinite(row["accuracy"]) or not 0 <= row["accuracy"] <= 1
                           or not isinstance(row.get("checkpoint_id"), str)
                           for row in immediate)):
                raise ValueError("H3 requires complete scheduled development registries with consistent run identities")
            matches = [row for row in registry if row.get("checkpoint_id") == run["checkpoint"] and row.get("format") == "immediate"]
            if (len(matches) != 1
                    or not _sha256(matches[0].get("adapter_sha256"))):
                raise ValueError("H3 selected checkpoint is missing or inconsistent in its training registry")
            if run["adapter_sha256"] != matches[0]["adapter_sha256"]:
                raise ValueError("H3 inference adapter hash differs from the selected checkpoint registry")
            if manifest["checkpoint_selection"] == "final":
                if matches[0]["step"] != run["training_manifest"]["training_budget"]["resolved_max_steps"]:
                    raise ValueError("H3 final comparison must use each run's final retained checkpoint")
            if run["model"] != base["model"] or run["revision"] != base["revision"]:
                raise ValueError("H3 trained inference differs from the base reference model")
            run["input_sources"]["selected_checkpoint"] = matches[0]
            registries.append((immediate, matches[0]))
            entry["runs"].append(run)
        entries.append(entry)
    if manifest["checkpoint_selection"] == "matched":
        target = min(max(row["accuracy"] for row in rows) for rows, _ in registries)
        for rows, selected in registries:
            closest = min(rows, key=lambda row: (abs(Decimal(str(row["accuracy"])) - Decimal(str(target))),
                                                  row["step"], row["checkpoint_id"]))
            if selected != closest:
                raise ValueError("H3 selected checkpoint does not follow the common development-accuracy matching rule")
    result = h3_cross_arm_data(base, entries)
    result["checkpoint_selection"] = manifest["checkpoint_selection"]
    result["manifest"] = {"path": str(path.resolve()), "sha256": file_digest(path)}
    result["corpus_manifest"] = corpus_sources["corpus_manifest"]
    result["confirmation_input_sha256"] = expected_inputs
    return result
