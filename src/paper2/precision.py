"""Candidate selection metadata; uncertainty is reported from observed outcomes."""


def edit_precision_plan(universe_worlds, *, stage="pilot", max_worlds=None):
    """Freeze candidate coverage without an invented precision pass criterion.

    Confirmation evaluates the entire frozen story pool. Editability and actual
    confidence intervals are outcomes, not eligibility thresholds for reporting.
    The historical function name is retained for callers and saved field names.
    """
    if stage not in ("pilot", "confirmatory"):
        raise ValueError("Edit stage must be pilot or confirmatory")
    if type(universe_worlds) is not int or universe_worlds < 1:
        raise ValueError("Edit candidate selection requires at least one world")
    if max_worlds is not None and (type(max_worlds) is not int or max_worlds < 1):
        raise ValueError("Edit subset requires positive max_worlds")
    if stage == "confirmatory" and max_worlds is not None and max_worlds != universe_worlds:
        raise ValueError("Confirmatory edits use the entire frozen candidate pool")
    requested = max_worlds if max_worlds is not None else (12 if stage == "pilot" else universe_worlds)
    selected = min(requested, universe_worlds)
    return {"stage": stage, "method": "candidate_coverage_without_precision_threshold_v3",
            "selected_worlds": selected,
            "estimand": "Per-arm targeted-edit effects conditional on editable traces, averaging fixed seeds within story",
            "candidate_selection": "All frozen confirmation worlds; pilot subset selected before assessing editability",
            "limitations": "Report eligible counts, estimates and whole-story 95% intervals. "
                           "Neither candidate counts nor interval widths assign a pass, quadrant or substantive reading."}
