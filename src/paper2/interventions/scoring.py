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


def compare_checkpoints(before: Sequence[Mapping[str, Any]], after: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Pair the same within-checkpoint swaps before/after training.

    Returns per-story/view probability effects for downstream story-clustered
    uncertainty estimation. It does not label an effect as a representation or
    estimate a proportion of training gain 'stored' in a component.
    """
    def index(rows):
        output = {}
        for row in rows:
            if row.get("analysis", "primary_within_checkpoint") != "primary_within_checkpoint":
                continue
            if row["control"] not in ("natural", "own", "signed_permutation"):
                continue
            key = (row["pair_id"], row["view"], row["layer"], row["component"], row["control"])
            if key in output:
                raise ValueError("Duplicate intervention key; compare one checkpoint at a time")
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
        result.append({"pair_id": key[0], "view": key[1], "layer": key[2], "component": key[3],
                       "control": key[4], "before_effect": old, "after_effect": new,
                       "training_change_in_effect": new - old})
    return result
