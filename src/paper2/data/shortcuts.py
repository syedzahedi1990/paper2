"""Preregistered symbolic shortcut checks, separate for depth and event kind.

A rule that returns several candidates is scored as uniform choice among those
candidates (its exact expected accuracy), not an optimistically selected answer.
Reported intervals describe uncertainty; no heuristic score decides a gate.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import math
from typing import Any, Iterable, Mapping

from .world import AGENTS, LOCATIONS, StoryWorld, ordinary_target

AUDIT_VERSION = "paper2.shortcuts.v5"
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


def _mode_candidates(locations: Iterable[str]) -> set[str]:
    counts = Counter(locations)
    maximum = max(counts.values())
    return {location for location, count in counts.items() if count == maximum}


def _score_summary(observations: list[float], reference: float, confidence: float) -> dict[str, Any]:
    """Pointwise bounded-score interval, never an acceptance threshold.

    The Hoeffding interval assumes independent story observations. Exact quota
    totals are design checks, not estimates inferred from these intervals.
    """
    n = len(observations)
    score = sum(observations) / n
    radius = math.sqrt(math.log(2 / (1 - confidence)) / (2 * n))
    return {"n": n, "score": score, "estimate": score,
            "ci95": [max(0., score - radius), min(1., score + radius)] if confidence == .95 else None,
            "interval": [max(0., score - radius), min(1., score + radius)],
            "confidence": confidence, "interval_method": "pointwise_hoeffding_bounded_story_score",
            "reference_expectation": reference, "difference_from_reference": score - reference}


def _residual_heuristics(cell: list[StoryWorld], position_count: int,
                         confidence: float) -> dict[str, Any]:
    """Describe positional, frequency and observation heuristics without screening."""
    scores: dict[str, list[float]] = defaultdict(list)
    ties: Counter[str] = Counter()
    missing: Counter[str] = Counter()
    semantic_scores = []
    for world in cell:
        target = ordinary_target(world, "direct")
        relevant = [event for event in world.events if event.object == world.focal_object]
        initial = world.initial_locations[world.focal_object]
        for position in range(position_count):
            rule = f"event_position_{position + 1}"
            # The rule has a declared fallback, so variable-length stories do
            # not condition scoring on whether a favourable position exists.
            location = relevant[position].location if position < len(relevant) else initial
            missing[rule] += position >= len(relevant)
            scores[rule].append(float(location == target))
        frequency_sources = {
            "most_mentioned_location": [event.location for event in relevant],
            "most_mentioned_location_with_initial": [initial] + [event.location for event in relevant],
            "most_mentioned_story_location": list(world.initial_locations.values()) +
                                              [event.location for event in world.events],
        }
        for rule, locations in frequency_sources.items():
            candidates = _mode_candidates(locations)
            ties[rule] += len(candidates) > 1
            scores[rule].append(float(target in candidates) / len(candidates))
        jointly_observed = [event for event in relevant if set(AGENTS).issubset(event.agents)]
        joint_location = jointly_observed[-1].location if jointly_observed else initial
        joint_score = float(joint_location == target)
        if len(world.focal_word) == 2:
            # Under the inherited state engine, both observers in the queried
            # word is precisely the correct update condition at depth two.
            semantic_scores.append(joint_score)
        else:
            scores["last_both_observed"].append(joint_score)
            missing["last_both_observed"] += not jointly_observed
    metrics = {}
    for rule, observations in sorted(scores.items()):
        metrics[rule] = {**_score_summary(observations, 1 / len(LOCATIONS), confidence),
                         "tie_worlds": ties[rule], "initial_fallback_worlds": missing[rule]}
    semantic_baselines = {}
    if semantic_scores:
        semantic_baselines["last_both_observed"] = {
            **_score_summary(semantic_scores, 1., confidence),
            "interpretation": "Exact intended depth-two update rule; not a chance-level shortcut control",
        }
    return {"metrics": metrics, "semantic_baselines": semantic_baselines}


def audit_shortcuts(worlds: Iterable[StoryWorld], *,
                    cue_rows: Iterable[Mapping[str, Any]] = (),
                    confidence: float = .95,
                    expected_depths: Iterable[int] | None = None,
                    require_cues: bool = False,
                    require_relation_quotas: bool = False,
                    require_clean_cue_quota: bool = False) -> dict[str, Any]:
    """Audit required controls and remaining positional/frequency heuristics.

    Expected equality is 1/6 for either independent first-order belief, 1/36
    for both, and 25/36 for differing from both. The last rate must NOT be
    compared with 1/2: with six labels, independence naturally yields a majority
    of disagreements. Candidate-exclusion accuracies test usable answer rules.
    Only clean/independent cues should be near chance; planted cue conditions
    are reported against their intended rates separately by the corpus manifest.
    Every score and interval is descriptive. Only exact declared design quotas
    and required coverage receive an integrity pass/fail result. Scientific G0
    suitability remains a named reviewer decision informed by all measurements.
    """
    if type(confidence) not in (int, float) or not 0 < confidence < 1:
        raise ValueError("Shortcut interval confidence must be between zero and one")
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
    position_count = max((sum(event.object == world.focal_object for event in world.events)
                          for world in worlds), default=0)
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
        metrics = {rule: _score_summary(observations, expectations[rule], confidence)
                   for rule, observations in sorted(values.items())}
        planned = relation_quota(len(cell)) if key.startswith("depth2") else None
        observed = {relation: relation_counts[relation] for relation in RELATION_WEIGHTS} if planned is not None else None
        quota_passed = planned is None or not require_relation_quotas or observed == planned
        clean_planned = (len(cell) + 5) // 6
        clean_observed = sum(values.get("clean_cue", []))
        clean_passed = not require_clean_cue_quota or (len(values.get("clean_cue", [])) == len(cell)
                                                       and clean_observed == clean_planned)
        integrity_passed = quota_passed and clean_passed
        residual = _residual_heuristics(cell, position_count, confidence)
        strata[key] = {"n": len(cell), "world_ids": sorted(world.world_id for world in cell),
                       "integrity_passed": integrity_passed,
                       "residual_heuristics": residual,
                       "clean_cue_quota": {"required": require_clean_cue_quota, "planned_correct": clean_planned,
                                           "observed_correct": clean_observed, "passed": clean_passed},
                       "metrics": metrics, "relation_quota": {"required": require_relation_quotas and planned is not None,
                                                               "planned": planned, "observed": observed,
                                                               "passed": quota_passed}}
    integrity_passed = bool(strata) and coverage_passed and all(cell["integrity_passed"] for cell in strata.values())
    return {"audit_version": AUDIT_VERSION, "integrity_passed": integrity_passed,
            "scientific_decision": "Requires named review of estimates, intervals, difficulty and heuristic use; no automatic accuracy cutoffs",
            "coverage": {"expected_strata": sorted(expected_strata), "missing_strata": missing_strata,
                         "unexpected_strata": unexpected_strata, "cues_required": require_cues,
                         "worlds_missing_clean_cue": missing_cues, "passed": coverage_passed},
            "worlds": len(worlds), "confidence": confidence,
            "interval_interpretation": "Pointwise bounded-score intervals assuming independent stories; not simultaneous screening bounds. Exact planned quotas are separate integrity checks.",
            "candidate_rule_scoring": "Uniform random selection from retained candidates; exact expected accuracy",
            "residual_rule_definitions": {
                "event_position_N": "Nth focal-object event (one-based); use focal initial location if absent",
                "most_mentioned_location": "Mode of focal-object event locations; uniform choice among tied modes",
                "most_mentioned_location_with_initial": "Same mode including focal initial location once",
                "most_mentioned_story_location": "Mode over all initial and event locations; inventory adds one of each and cannot alter the modes",
                "last_both_observed": "Last focal-object event received by both characters, else focal initial location; semantic solver at depth two",
            },
            "cue_reference": "Only clean/independent cues; planted agree/conflict/shortcut rates are not chance controls",
            "strata": strata}
