"""Task-specific Test 3 forecasts, kept separate from ordinary task accuracy."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
from paper2.data import VIEWS


def score_consequence_pattern(predictions: Mapping[str, str | None],
                              baseline_targets: Mapping[str, str],
                              forecasts: Mapping[str, Mapping[str, str]],
                              account: str = "event_rewrite") -> dict[str, Any]:
    """Score a whole matched story, including required changes and preservation.

    A partial pattern is an error, not joint success. The caller supplies the
    generator's per-view forecasts, rather than reusing Test 1 ordinary targets.
    ``event_rewrite`` is the prespecified natural-event-swap primary forecast;
    alternative paper-one accounts remain individually reported diagnostics.
    """
    if set(baseline_targets) != set(VIEWS) or set(predictions) != set(baseline_targets) or set(forecasts) != set(baseline_targets):
        raise ValueError("Predictions, ordinary targets and forecasts must cover identical complete view sets")
    targets = {view: forecasts[view][account] for view in baseline_targets}
    changed = {v for v in targets if targets[v] != baseline_targets[v]}
    intact = set(targets) - changed
    correct = {v: predictions[v] == targets[v] for v in targets}
    average = lambda views: sum(correct[v] for v in views) / len(views) if views else None
    return {
        "account": account, "forecast_accuracy": average(targets),
        "changed_accuracy": average(changed), "intact_accuracy": average(intact),
        "joint_correct": all(correct.values()),
        "changed_views": sorted(changed), "intact_views": sorted(intact),
        "ordinary_accuracy": sum(predictions[v] == baseline_targets[v] for v in targets) / len(targets),
        "per_view_correct": correct,
    }


def _mask(row):
    return row.get("mask_id", str(row.get("layer")))


def compare_checkpoints(before: Sequence[Mapping[str, Any]], after: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Pair identical within-checkpoint swaps, retaining view and required-change status."""
    from paper2.analysis import _seed
    seeds = {_seed(r) for r in after}
    if {_seed(r) for r in before} == {None} and seeds and None not in seeds:
        before = [{**r, "training_seed": seed} for seed in seeds for r in before]

    def index(rows):
        output = {}
        for row in rows:
            if row.get("analysis", "primary_within_checkpoint") != "primary_within_checkpoint":
                continue
            if row["control"] not in ("natural", "own", "signed_permutation"):
                continue
            key = (_seed(row), row["pair_id"], row["view"], _mask(row), row["component"], row["control"])
            if key in output:
                raise ValueError("Duplicate intervention key; compare one arm/dataset at a time")
            output[key] = row
        return output
    left, right = index(before), index(after)
    if not left or set(left) != set(right):
        raise ValueError("Checkpoint comparison requires the same complete set of swaps")
    result = []
    for key, row in left.items():
        if key[-1] == "own":
            continue
        own_key = (*key[:-1], "own")
        if own_key not in left:
            raise ValueError("Every intervention needs its own-component control")
        for candidate in (right[key], left[own_key], right[own_key]):
            if candidate["forecast_target"] != row["forecast_target"]:
                raise ValueError("Checkpoint comparisons changed forecast targets")
        old = row["forecast_probability"] - left[own_key]["forecast_probability"]
        new = right[key]["forecast_probability"] - right[own_key]["forecast_probability"]
        result.append({"training_seed": key[0], "pair_id": key[1], "view": key[2], "mask_id": key[3],
                       "layers": row.get("layers", [row.get("layer")]), "component": key[4],
                       "change_required": row.get("change_required", row.get("ordinary_target") != row["forecast_target"]),
                       "control": key[5], "before_effect": old, "after_effect": new,
                       "training_change_in_effect": new - old})
    return result


def component_summary(before, after):
    """Report whole-story consequences alongside component probability contrasts.

    Fixed seeds are averaged within stories, then stories are bootstrapped.
    We report changed and intact outcomes separately; a pooled probability effect
    alone is never the H3 consequence-pattern result. Each checkpoint's paired
    intervention-minus-own effect is reported alongside the training contrast.
    """
    from collections import defaultdict
    from paper2.analysis import _seed, cluster_interval
    from .controls import residual_control_summary
    effects = compare_checkpoints(before, after)
    clusters = defaultdict(lambda: defaultdict(list))
    for row in effects:
        prefix = f"{row['mask_id']}|{row['component']}|{row['control']}"
        scope = "changed" if row["change_required"] else "intact"
        for suffix in (scope, "view:" + row["view"]):
            group = prefix + "|" + suffix
            # Preserve the original unsuffixed training-contrast keys.
            clusters[group][row["pair_id"]].append(row["training_change_in_effect"])
            for effect in ("before_effect", "after_effect"):
                clusters[group + "|" + effect][row["pair_id"]].append(row[effect])

    def patterns(rows):
        groups = defaultdict(dict)
        for row in rows:
            if row.get("analysis", "primary_within_checkpoint") != "primary_within_checkpoint":
                continue
            if row["control"] not in ("natural", "own", "signed_permutation"):
                continue
            key = (_seed(row), row["pair_id"], _mask(row), row["component"], row["control"])
            if row["view"] in groups[key]:
                raise ValueError("Duplicate consequence view")
            groups[key][row["view"]] = row
        output = {}
        for key, views in groups.items():
            output[key] = score_consequence_pattern(
                {v: r["prediction"] for v, r in views.items()},
                {v: r["ordinary_target"] for v, r in views.items()},
                {v: r["forecasts"] for v, r in views.items()})
        return output

    seeds = {_seed(r) for r in after}
    if {_seed(r) for r in before} == {None} and seeds and None not in seeds:
        before = [{**r, "training_seed": seed} for seed in seeds for r in before]
    old, new = patterns(before), patterns(after)
    if set(old) != set(new):
        raise ValueError("Consequence patterns must cover matching complete panels")
    panels = defaultdict(set)
    for key in new:
        panels[key[0]].add(key[1:])
    if len(panels) > 1 and any(x != next(iter(panels.values())) for x in panels.values()):
        raise ValueError("Every seed must cover the same complete component panel")
    pattern_clusters = defaultdict(lambda: defaultdict(list))
    for key, score in new.items():
        training_seed, pair, mask, component, control = key
        own_key = (*key[:-1], "own")
        if own_key not in new:
            raise ValueError("Consequence pattern requires an own control")
        prefix = f"{mask}|{component}|{control}"
        for metric in ("changed_accuracy", "intact_accuracy", "joint_correct", "ordinary_accuracy"):
            for stage, all_scores in (("before", old), ("after", new)):
                value = all_scores[key][metric]
                if value is not None:
                    pattern_clusters[prefix + "|" + metric + "|" + stage][pair].append(float(value))
                    own_value = all_scores[own_key][metric]
                    if control != "own" and own_value is not None:
                        effect = float(value) - float(own_value)
                        pattern_clusters[prefix + "|" + metric + "|" + stage + "_effect"][pair].append(effect)
            if score[metric] is not None and old[key][metric] is not None:
                delta = float(score[metric]) - float(old[key][metric])
                pattern_clusters[prefix + "|" + metric + "|training_change"][pair].append(delta)
                if control != "own" and new[own_key][metric] is not None and old[own_key][metric] is not None:
                    effect = delta - (float(new[own_key][metric]) - float(old[own_key][metric]))
                    pattern_clusters[prefix + "|" + metric + "|training_change_in_effect"][pair].append(effect)
    return {"paired_effects": effects,
            "probability_effects": {k: cluster_interval(v) for k, v in sorted(clusters.items())},
            "consequence_patterns": {k: cluster_interval(v) for k, v in sorted(pattern_clusters.items())},
            "g4_positive_controls": {"before": residual_control_summary(before), "after": residual_control_summary(after)},
            "uncertainty": "Fixed seeds averaged within stories; paired whole-story bootstrap.",
            "effect_definition": "Before/after effects pair each intervention with its own-component control at that checkpoint; unsuffixed probability_effects remain the after-minus-before change in effect.",
            "forecast_account": "event_rewrite"}
