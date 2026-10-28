"""Preregistered story-count planning, distinct from measured uncertainty."""
import math
from statistics import NormalDist


def edit_precision_plan(universe_worlds, *, stage="pilot", max_worlds=None, half_width=.10):
    """Plan a 95% interval for a per-arm story mean bounded in [0, 1].

    The worst-case-variance normal approximation is a planning calculation,
    not a guaranteed interval and not a power calculation for arm differences.
    The target applies to editable stories, whose count is unknown at planning.
    Actual whole-story intervals must also meet the frozen precision target.
    """
    if stage not in ("pilot", "confirmatory"):
        raise ValueError("Edit stage must be pilot or confirmatory")
    if type(universe_worlds) is not int or universe_worlds < 1:
        raise ValueError("Edit precision planning requires at least one candidate world")
    if type(half_width) not in (int, float) or not math.isfinite(half_width) or not 0 < half_width < .5:
        raise ValueError("Edit precision half-width must be finite and between zero and .5")
    if max_worlds is not None and (type(max_worlds) is not int or max_worlds < 1):
        raise ValueError("Edit subset requires positive max_worlds")
    z = NormalDist().inv_cdf(.975)
    required = math.ceil(z * z / (4 * half_width * half_width))
    requested = max_worlds if max_worlds is not None else (12 if stage == "pilot" else universe_worlds)
    selected = min(requested, universe_worlds)
    if stage == "confirmatory" and selected < required:
        raise ValueError(f"Confirmatory edit precision requires at least {required} independent candidate worlds, and ultimately editable worlds; "
                         f"only {selected} selected. Preregister more worlds or a wider precision target.")
    return {"stage": stage, "confidence": .95, "target_half_width": half_width,
            "method": "conditional_editable_story_rate_planning_v2", "outcome_range": [0, 1],
            "estimand": "Per-source-checkpoint targeted-edit compliance or preservation conditional on editable traces, averaged within story",
            "required_worlds": required, "required_eligible_worlds": required, "selected_worlds": selected,
            "candidate_selection": "Preregister candidate stories before assessing editability; confirmatory default uses the full pool",
            "minimum_editable_fraction_for_target": required / selected,
            "planned_half_width_if_all_editable": z / (2 * math.sqrt(selected)),
            "candidate_pool_sufficient": selected >= required,
            "limitations": "Approximate planning only; candidate count does not establish editable count. "
                           "Every source checkpoint needs enough editable worlds and actual story intervals; both quadrant axes need review. "
                           "This does not size paired arm differences, subgroups, or ratios of remaining errors."}
