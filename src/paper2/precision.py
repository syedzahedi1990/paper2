"""Preregistered story-count planning, distinct from measured uncertainty."""
import math
from statistics import NormalDist


def edit_precision_plan(universe_worlds, *, stage="pilot", max_worlds=None, half_width=.10):
    """Plan a 95% interval for a per-arm story mean bounded in [0, 1].

    The worst-case-variance normal approximation is a planning calculation,
    not a guaranteed interval and not a power calculation for arm differences.
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
    requested = max_worlds if max_worlds is not None else (12 if stage == "pilot" else required)
    selected = min(requested, universe_worlds)
    if stage == "confirmatory" and selected < required:
        raise ValueError(f"Confirmatory edit precision requires at least {required} independent worlds; "
                         f"only {selected} selected. Preregister more worlds or a wider precision target.")
    return {"stage": stage, "confidence": .95, "target_half_width": half_width,
            "method": "worst_case_variance_normal_planning_v1", "outcome_range": [0, 1],
            "estimand": "Per-arm targeted-edit compliance or preservation averaged within story",
            "required_worlds": required, "selected_worlds": selected,
            "planned_half_width": z / (2 * math.sqrt(selected)),
            "planning_sufficient": selected >= required,
            "limitations": "Approximate planning only; actual story intervals and both quadrant axes need review. "
                           "This does not size paired arm differences, subgroups, or ratios of remaining errors."}
