"""Figures from measured analysis artifacts; no quadrant labels or cutoffs."""
from collections import defaultdict
import json
import math
from pathlib import Path

from .analysis import accuracy_summary
from .io import digest, file_digest, write_json


def _measurement(value, name):
    if not isinstance(value, dict):
        raise ValueError(f"Missing measured {name}")
    estimate, interval = value.get("estimate"), value.get("ci95")
    if estimate is None or interval is None:
        return None
    if (type(estimate) not in (int, float) or not math.isfinite(estimate)
            or not isinstance(interval, (list, tuple)) or len(interval) != 2
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in interval)
            or interval[0] > interval[1]):
        raise ValueError(f"Invalid measured estimate or interval for {name}")
    return {"estimate": estimate, "ci95": list(interval), "stories": value.get("stories")}


def quadrant_data(entries):
    """Extract the two measured axes from transfer and crossed-analysis JSON.

    Each entry names label/arm/dataset, reader/source and edit_condition, and
    embeds its actual transfer and crossed summaries. Fixed-direct semantic
    minimal and regenerated interventions remain explicitly distinct conditions.
    """
    if isinstance(entries, dict):
        entries = entries.get("points")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Quadrant input needs a nonempty list of measured point specifications")
    points, omitted = [], []
    seen = set()
    for item in entries:
        if any(not isinstance(item.get(key), str) or not item[key] for key in
               ("label", "arm", "dataset", "reader", "source", "edit_condition")):
            raise ValueError("Each point needs label, arm, dataset, reader, source and edit_condition")
        if item["label"] in seen:
            raise ValueError("Plot labels must be unique")
        seen.add(item["label"])
        condition = item["edit_condition"]
        if condition not in ("semantic_minimal", "semantic_regenerated"):
            raise ValueError("The edit axis needs semantic_minimal or semantic_regenerated")
        transfer, crossed = item.get("transfer", {}), item.get("crossed", {})
        if transfer.get("estimand") != "trained_minus_base_direct_answer_accuracy":
            raise ValueError("The transfer axis requires an actual Test 1 transfer summary")
        key = f"{item['reader']}|{item['source']}|{condition}|direct_fixed"
        control = crossed.get("controls", {}).get(key)
        if control is None:
            raise ValueError("The edit axis requires the specified frozen direct-question control")
        story = control.get("story_precision", {})
        x = _measurement(transfer.get("effects", {}).get("unrewarded"), "unrewarded no-trace transfer")
        edit_metric = story.get("metrics", {}).get("changed_response_increase_vs_original")
        y = _measurement(edit_metric,
                         "targeted-edit effect relative to the original trace")
        identity = {k: item[k] for k in ("label", "arm", "dataset", "reader", "source", "edit_condition")}
        if x is None or y is None:
            omitted.append({**identity, "reason": "Estimate or story interval is unavailable on at least one axis"})
            continue
        # Sensitivity is distinct from the bootstrap interval drawn on the plot.
        # Retain the complete [-1,1] outcome bound in its numeric sidecar.
        y["audit_sensitivity"] = edit_metric.get("audit_sensitivity", {
            "status": "unavailable", "reason": "Source summary has no audit sensitivity evidence",
            "maximum_absolute_bias": None, "sensitivity_range": None})
        points.append({**identity, "x": x, "y": y, "question_timing": "direct_fixed",
                       "training_seeds": transfer.get("training_seeds"), "editability": story.get("editability"),
                       "transfer_sha256": digest(transfer), "crossed_sha256": digest(crossed),
                       "input_sources": item.get("input_sources")})
    return {"schema_version": 1, "figure": "two_measured_axes", "points": points, "omitted": omitted,
            "x_label": "No-trace transfer on unrewarded views (trained − base)",
            "y_label": "Targeted-edit response change (edited − original trace)",
            "edit_axis_scope": "Change-required consequences of direct_fixed traces, conditional on editability",
            "interpretation": "Points and 95% story intervals only. No regions, cutoffs, quadrant assignments or stored-belief claims."}


def load_quadrant_data(manifest_path):
    """Resolve transfer_path/crossed_path relative to a saved plot manifest."""
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    entries = manifest.get("points") if isinstance(manifest, dict) else manifest
    if not isinstance(entries, list):
        raise ValueError("Plot manifest must contain a points list")
    loaded = []
    for item in entries:
        entry, sources = dict(item), {}
        for field in ("transfer", "crossed"):
            source_path = item.get(field + "_path")
            if not isinstance(source_path, str) or not source_path:
                raise ValueError(f"Plot point requires {field}_path")
            source = (path.parent / source_path).resolve()
            entry[field] = json.loads(source.read_text(encoding="utf-8"))
            sources[field] = {"path": str(source), "sha256": file_digest(source)}
        entry["input_sources"] = sources
        loaded.append(entry)
    result = quadrant_data(loaded)
    result["manifest"] = {"path": str(path.resolve()), "sha256": file_digest(path)}
    return result


def _pyplot():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("Plot rendering requires the optional plotting dependencies: pip install 'paper2-reasoning-rl[plot]'") from error
    return plt


def _save(figure, output, data, plt):
    output = Path(output)
    if output.suffix.lower() not in (".png", ".pdf", ".svg"):
        raise ValueError("Figure output must be PNG, PDF or SVG")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)
    sidecar = output.with_suffix(output.suffix + ".json")
    write_json(sidecar, data)
    return {"figure_path": str(output.resolve()), "data_path": str(sidecar.resolve()),
            "figure_sha256": file_digest(output)}


def plot_quadrant(data, output):
    if data.get("figure") != "two_measured_axes" or not data.get("points"):
        raise ValueError("A two-axis figure requires at least one measured point with both intervals")
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(8, 6), layout="constrained")
    colors = plt.get_cmap("tab10")
    for index, point in enumerate(data["points"]):
        x, y = _measurement(point["x"], "x axis"), _measurement(point["y"], "y axis")
        if x is None or y is None or point.get("question_timing") != "direct_fixed":
            raise ValueError("Each plotted point needs direct_fixed estimates and both intervals")
        color = colors(index % 10)
        axis.hlines(y["estimate"], *x["ci95"], color=color, linewidth=1.5)
        axis.vlines(x["estimate"], *y["ci95"], color=color, linewidth=1.5)
        axis.scatter([x["estimate"]], [y["estimate"]], color=[color], s=45, label=point["label"], zorder=3)
    axis.set(xlabel=data["x_label"], ylabel=data["y_label"], title="Two measurements with 95% story intervals")
    axis.grid(alpha=.15)
    axis.legend(loc="best", frameon=False)
    return _save(figure, output, data, plt)


def plot_h3(data, output):
    """Eight descriptive arm/dataset points for each of K, V and residual."""
    from .h3 import COMPONENTS
    from .training.config import ARMS
    expected = {(arm, dataset) for arm in ARMS for dataset in ("clean", "shortcut")}
    if data.get("figure") != "h3_descriptive_cross_arm":
        raise ValueError("H3 plotting requires the verified descriptive cross-arm analysis")
    panels = {component: [point for point in data.get("points", []) if point.get("component") == component]
              for component in COMPONENTS}
    if len(data.get("points", [])) != 24 or any(len(points) != 8 or {
            (point["arm"], point["dataset"]) for point in points} != expected for points in panels.values()):
        raise ValueError("H3 plotting requires all eight arm/dataset points for each component")
    plt = _pyplot()
    figure, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True, layout="constrained")
    colors = {arm: plt.get_cmap("tab10")(index) for index, arm in enumerate(ARMS)}
    for axis, component in zip(axes, COMPONENTS):
        for point in panels[component]:
            x, y = _measurement(point["x"], "H3 transfer"), _measurement(point["y"], "H3 component change")
            if x is None or y is None:
                raise ValueError("H3 plotting needs measured intervals on both axes for every point")
            color = colors[point["arm"]]
            axis.hlines(y["estimate"], *x["ci95"], color=color, linewidth=1)
            axis.vlines(x["estimate"], *y["ci95"], color=color, linewidth=1)
            axis.scatter(x["estimate"], y["estimate"], color=[color], s=40,
                         marker="o" if point["dataset"] == "clean" else "^", label=point["label"], zorder=3)
        axis.set(title={"k": "Keys", "v": "Values", "residual": "Residual stream"}[component],
                 xlabel=data["x_label"])
        axis.grid(alpha=.15)
    axes[0].set_ylabel(data["y_label"])
    axes[-1].legend(loc="best", fontsize="small", frameon=False)
    figure.suptitle("Descriptive cross-arm H3 comparison · 95% story intervals")
    return _save(figure, output, data, plt)


def accuracy_reference_data(model_rows, audit, *, split=None, seed=2026, samples=20_000):
    """Pair clean direct-question model scores with measured symbolic references.

    Relation probabilities are not classifiers and are deliberately omitted.
    The depth-two last-both-observed semantic solver is labelled as an intended
    rule, rather than being described as a weak shortcut.
    """
    rows = [row for row in model_rows if row.get("view") == "direct"]
    if not rows:
        raise ValueError("Accuracy references require direct-question model results")
    splits = {row.get("split") for row in rows}
    if len(splits) != 1 or None in splits or (split is not None and splits != {split}):
        raise ValueError("Model results must use one explicitly matching audit split")
    split = split or next(iter(splits))
    corpus_hash = audit.get("corpus_hash")
    selected = audit.get("shortcut_audit", audit)
    selected = selected.get(split, selected)
    if not isinstance(selected.get("strata"), dict):
        raise ValueError("The selected shortcut audit has no depth/event strata")
    if any(row.get("cue_condition", "clean") != "clean" for row in rows):
        raise ValueError("This comparison requires clean cue model rows matching the clean audit")
    model_groups = defaultdict(list)
    for row in rows:
        depth, event = row.get("belief_depth"), row.get("critical_event_kind")
        if depth not in (1, 2) or event not in ("move", "tell"):
            raise ValueError("Model rows need belief_depth and critical_event_kind for matched audit strata")
        label = row.get("arm") or row.get("reader")
        if not label:
            raise ValueError("Model rows require a reader or arm label")
        model_groups[(f"depth{depth}__{event}", label, row.get("training_dataset", row.get("dataset")),
                      row.get("format", "immediate"))].append(row)
    models, references = [], []
    matched_world_ids = True
    for (stratum, label, dataset, format_name), observations in sorted(model_groups.items(), key=lambda item: repr(item[0])):
        cell = selected["strata"].get(stratum)
        if cell is None:
            raise ValueError("Model stratum is missing from the shortcut audit")
        worlds = {row["world_id"] for row in observations}
        if len(worlds) != cell["n"]:
            raise ValueError("Model and heuristic audit have different story counts in a stratum")
        if "world_ids" in cell:
            if worlds != set(cell["world_ids"]):
                raise ValueError("Model and heuristic audit refer to different worlds")
        else:
            matched_world_ids = False
        summary = accuracy_summary(observations, seed, samples)
        models.append({"stratum": stratum, "label": f"{label} / {dataset} / {format_name}",
                       "arm_or_reader": label, "dataset": dataset, "format": format_name,
                       "estimate": summary["accuracy"], "ci95": summary["ci95"], "stories": summary["stories"],
                       "kind": "model", "interval": summary["uncertainty"]})
    predictor_names = {"first_event", "last_event", "initial_location", "clean_cue", "exclude_outer", "exclude_inner", "exclude_both"}
    for stratum in sorted({entry["stratum"] for entry in models}):
        cell = selected["strata"][stratum]
        families = [(name, result, "heuristic") for name, result in cell.get("metrics", {}).items() if name in predictor_names]
        residual = cell.get("residual_heuristics", {})
        families += [(name, result, "heuristic") for name, result in residual.get("metrics", {}).items()]
        families += [(name, result, "intended_rule") for name, result in residual.get("semantic_baselines", {}).items()]
        for name, result, kind in sorted(families):
            measured = {"estimate": result.get("estimate", result.get("score")), "ci95": result.get("ci95"),
                        "stories": result.get("n")}
            if _measurement(measured, name) is None:
                raise ValueError("Heuristic plots require measured reference intervals; regenerate the current descriptive audit")
            references.append({**measured, "stratum": stratum, "label": name, "kind": kind,
                               "interval": result.get("interval"), "reference_expectation": result.get("reference_expectation")})
    return {"schema_version": 1, "figure": "accuracy_and_heuristic_references", "split": split,
            "models": models, "references": references, "audit_version": selected.get("audit_version"),
            "audit_sha256": digest(selected), "corpus_hash": corpus_hash,
            "world_identity_verified": matched_world_ids, "model_results_sha256": digest(rows),
            "bootstrap_samples": samples, "bootstrap_seed": seed,
            "interpretation": "Measured clean direct-question accuracy beside named rules, stratified by depth/event. "
                              "Model intervals bootstrap stories; reference interval methods are retained from the audit. "
                              "No pass threshold or inferred chance-level guarantee."}


def plot_accuracy_references(data, output):
    if data.get("figure") != "accuracy_and_heuristic_references" or not data.get("models"):
        raise ValueError("Reference figure needs measured model results")
    plt = _pyplot()
    strata = sorted({point["stratum"] for point in data["models"]})
    columns = min(2, len(strata))
    rows = math.ceil(len(strata) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(8 * columns, 7 * rows), squeeze=False, layout="constrained")
    for axis, stratum in zip(axes.flat, strata):
        points = [point for point in data["models"] + data["references"] if point["stratum"] == stratum]
        labels = []
        for index, point in enumerate(points):
            measurement = _measurement(point, point["label"])
            if measurement is None:
                raise ValueError("Reference points need estimates and intervals")
            color = "#235789" if point["kind"] == "model" else "#2c7a4b" if point["kind"] == "intended_rule" else "#777777"
            axis.hlines(index, *measurement["ci95"], color=color, linewidth=1.2)
            axis.scatter([measurement["estimate"]], [index], color=color, marker="D" if point["kind"] == "model" else "o", s=30)
            labels.append(point["label"] + (" (intended rule)" if point["kind"] == "intended_rule" else ""))
        axis.set(yticks=range(len(points)), yticklabels=labels, xlim=(0, 1), xlabel="Direct-question accuracy", title=stratum.replace("__", " / "))
        axis.invert_yaxis()
        axis.grid(axis="x", alpha=.15)
    for axis in list(axes.flat)[len(strata):]:
        axis.set_visible(False)
    return _save(figure, output, data, plt)
