"""Preregistered symbolic shortcut checks, separate for depth and event kind.

A rule that returns several candidates is scored as uniform choice among those
candidates (its exact expected accuracy), not an optimistically selected answer.
The simultaneous bounds are screening thresholds, not hypothesis-test results.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import math
from typing import Any, Iterable, Mapping

from .world import LOCATIONS, StoryWorld, ordinary_target

AUDIT_VERSION = "paper2.shortcuts.v3"
RELATION_WEIGHTS = {"both": 1, "outer_only": 5, "inner_only": 5, "neither": 25}


def relation_quota(count: int) -> dict[str, int]:
    """Largest-remainder integer allocation of the independent six-label null."""
    if count < 0:
        raise ValueError("Relation quota count must be nonnegative")
    result = {relation: count * weight // 36 for relation, weight in RELATION_WEIGHTS.items()}
    priority = sorted(RELATION_WEIGHTS, key=lambda relation: -(count * RELATION_WEIGHTS[relation] % 36))
    for relation in priority[:count - sum(result.values())]:
        result[relation] += 1
    return result


def audit_shortcuts(worlds: Iterable[StoryWorld], *,
                    cue_rows: Iterable[Mapping[str, Any]] = (),
                    minimum_cell_size: int = 24,
                    familywise_alpha: float = 0.01,
                    expected_depths: Iterable[int] | None = None,
                    require_cues: bool = False,
                    require_relation_quotas: bool = False,
                    require_clean_cue_quota: bool = False) -> dict[str, Any]:
    """Audit positional, initial, cue and both-sign nested-belief shortcuts.

    Expected equality is 1/6 for either independent first-order belief, 1/36
    for both, and 25/36 for differing from both. The last rate must NOT be
    compared with 1/2: with six labels, independence naturally yields a majority
    of disagreements. Candidate-exclusion accuracies test usable answer rules.
    Only clean/independent cues should be near chance; planted cue conditions
    are reported against their intended rates separately by the corpus manifest.
    """
    if minimum_cell_size < 1 or not 0 < familywise_alpha < 1:
        raise ValueError("Invalid shortcut-audit threshold")
    worlds = list(worlds)
    cues: dict[str, list[str]] = defaultdict(list)
    for row in cue_rows:
        if row.get("view") == "direct" and row.get("cue_condition") in ("clean", "independent"):
            if row.get("cue_location") not in LOCATIONS:
                raise ValueError("Shortcut audit requires valid clean cue locations")
            cues[row["world_id"]].append(row["cue_location"])
    grouped: dict[str, list[StoryWorld]] = defaultdict(list)
    for world in worlds:
        grouped[f"depth{len(world.focal_word)}__{world.critical_event.kind}"].append(world)
    depths = set(expected_depths) if expected_depths is not None else {len(world.focal_word) for world in worlds}
    if not depths or depths - {1, 2}:
        raise ValueError("Shortcut audit requires expected depths one and/or two")
    expected_strata = {f"depth{depth}__{kind}" for depth in depths for kind in ("move", "tell")}
    missing_strata = sorted(expected_strata - set(grouped))
    unexpected_strata = sorted(set(grouped) - expected_strata)
    missing_cues = sum(world.world_id not in cues for world in worlds)
    coverage_passed = not missing_strata and not unexpected_strata and (not require_cues or not missing_cues)
    # At most 4 strata x 13 metrics. A fixed denominator prevents an omitted
    # stratum or unavailable cue from silently weakening the threshold.
    comparisons = 52
    strata = {}
    for key, cell in sorted(grouped.items()):
        values: dict[str, list[float]] = defaultdict(list)
        expectations: dict[str, float] = {}
        relation_counts: Counter[str] = Counter()
        for world in cell:
            target = ordinary_target(world, "direct")
            relevant = [event for event in world.events if event.object == world.focal_object]
            predictions = {"first_event": relevant[0].location, "last_event": relevant[-1].location,
                           "initial_location": world.initial_locations[world.focal_object]}
            for rule, predicted in predictions.items():
                values[rule].append(float(predicted == target))
                expectations[rule] = 1 / 6
            if world.world_id in cues:
                values["clean_cue"].append(sum(cue == target for cue in cues[world.world_id]) / len(cues[world.world_id]))
                expectations["clean_cue"] = 1 / 6
            if len(world.focal_word) == 2:
                state = world.state()
                outer, inner = (state.get((agent,), world.focal_object) for agent in world.focal_word)
                relation = ("both" if target == outer == inner else "outer_only" if target == outer
                            else "inner_only" if target == inner else "neither")
                relation_counts[relation] += 1
                relations = {"equal_outer": target == outer, "equal_inner": target == inner,
                             "equal_both": target == outer == inner,
                             "different_from_both": target != outer and target != inner}
                nulls = {"equal_outer": 1 / 6, "equal_inner": 1 / 6,
                         "equal_both": 1 / 36, "different_from_both": 25 / 36}
                for rule, value in relations.items():
                    values[rule].append(float(value))
                    expectations[rule] = nulls[rule]
                for rule, excluded in (("exclude_outer", {outer}), ("exclude_inner", {inner}),
                                       ("exclude_both", {outer, inner})):
                    candidates = set(LOCATIONS) - excluded
                    values[rule].append(float(target in candidates) / len(candidates))
                    expectations[rule] = 1 / 6
        metrics = {}
        for rule, observations in sorted(values.items()):
            count = len(observations)
            score = sum(observations) / count
            tolerance = math.sqrt(math.log(2 * comparisons / familywise_alpha) / (2 * count))
            if require_relation_quotas and rule in ("equal_outer", "equal_inner", "equal_both", "different_from_both"):
                # Each joint quota differs from its ideal real count by <1;
                # either marginal sums two such cells. This exact finite-design
                # bound replaces a loose sampling bound for planned relations.
                tolerance = min(tolerance, (2 if rule in ("equal_outer", "equal_inner") else 1) / count)
            if require_clean_cue_quota and rule == "clean_cue":
                tolerance = min(tolerance, 1 / count)
            sufficient = count >= minimum_cell_size
            # Wide held-out confidence bounds must never admit a deterministic
            # relation. Do not require equal_both>0: its six-label null is only
            # 1/36, so zero cases in a 30-world cell is ordinary sampling noise.
            needs_both_signs = rule in ("equal_outer", "equal_inner", "different_from_both")
            nondegenerate = 0 < score < 1 if needs_both_signs else True
            metrics[rule] = {"n": count, "score": score, "null_expectation": expectations[rule],
                             "absolute_deviation": abs(score - expectations[rule]),
                             "maximum_deviation": tolerance,
                             "sufficient": sufficient, "requires_both_relation_outcomes": needs_both_signs,
                             "nondegenerate": nondegenerate,
                             "passed": sufficient and nondegenerate and abs(score - expectations[rule]) <= tolerance}
        planned = relation_quota(len(cell)) if key.startswith("depth2") else None
        observed = {relation: relation_counts[relation] for relation in RELATION_WEIGHTS} if planned is not None else None
        quota_passed = planned is None or not require_relation_quotas or observed == planned
        clean_planned = (len(cell) + 5) // 6
        clean_observed = sum(values.get("clean_cue", []))
        clean_passed = not require_clean_cue_quota or (len(values.get("clean_cue", [])) == len(cell)
                                                       and clean_observed == clean_planned)
        strata[key] = {"n": len(cell), "passed": quota_passed and clean_passed and all(metric["passed"] for metric in metrics.values()),
                       "clean_cue_quota": {"required": require_clean_cue_quota, "planned_correct": clean_planned,
                                           "observed_correct": clean_observed, "passed": clean_passed},
                       "metrics": metrics, "relation_quota": {"required": require_relation_quotas and planned is not None,
                                                               "planned": planned, "observed": observed,
                                                               "passed": quota_passed}}
    return {"audit_version": AUDIT_VERSION, "passed": bool(strata) and coverage_passed and all(cell["passed"] for cell in strata.values()),
            "coverage": {"expected_strata": sorted(expected_strata), "missing_strata": missing_strata,
                         "unexpected_strata": unexpected_strata, "cues_required": require_cues,
                         "worlds_missing_clean_cue": missing_cues, "passed": coverage_passed},
            "worlds": len(worlds), "minimum_cell_size": minimum_cell_size,
            "familywise_alpha": familywise_alpha, "simultaneous_comparisons": comparisons,
            "threshold": "Two-sided Hoeffding bounds; required joint relation and heldout clean-cue quotas must match exactly",
            "candidate_rule_scoring": "Uniform random selection from retained candidates; exact expected accuracy",
            "cue_reference": "Only clean/independent cues; planted agree/conflict/shortcut rates are not chance controls",
            "strata": strata}
