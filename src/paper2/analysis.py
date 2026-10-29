"""Proposal estimands with paired story-level uncertainty and explicit denominators."""
from collections import defaultdict
import numpy as np
from .data import VIEWS

JOINT_ALIGNMENT = (
    "Joint correctness requires all eight views of the same underlying world, cue condition and literal cue location. "
    "The cue condition is anchored to the direct-belief target and remains fixed across consequences. "
    "Incomplete eight-view patterns are counted separately and never treated as joint successes."
)


def _seed(row):
    return row.get("comparison_seed", row.get("training_seed", row.get("seed")))


def _index(rows, *, require_balanced_seeds=True):
    result = {}
    for row in rows:
        key = (_seed(row), row["world_id"], row["view"], row.get("cue_condition", "clean"), row.get("cue_location", ""))
        if key in result:
            raise ValueError(f"Duplicate evaluation key: {key}; select one checkpoint/format first")
        result[key] = row
    panels = defaultdict(set)
    for key in result:
        panels[key[0]].add(key[1:])
    if require_balanced_seeds and len(panels) > 1 and any(panel != next(iter(panels.values())) for panel in panels.values()):
        raise ValueError("Every training seed must cover the same complete story/view panel")
    return result


def paired_rows(base, other, *, require_balanced_seeds=True):
    base, other = list(base), list(other)
    seeds = {_seed(r) for r in other}
    if {_seed(r) for r in base} == {None} and seeds and None not in seeds:
        base = [{**r, "training_seed": seed} for seed in seeds for r in base]
    a, b = (_index(base, require_balanced_seeds=require_balanced_seeds),
            _index(other, require_balanced_seeds=require_balanced_seeds))
    if set(a) != set(b):
        raise ValueError("Paired comparisons require exactly the same worlds, views and cue conditions")
    for key in sorted(a, key=repr):
        if a[key]["answer"] != b[key]["answer"]:
            raise ValueError("Ordinary task targets differ between paired readers")
        yield a[key], b[key]


def _baseline_subset(baseline, observations, *, require_balanced_seeds=True):
    """Select a prespecified control panel, preserving the reader's own baseline."""
    seeds = {_seed(r) for r in observations}
    if {_seed(r) for r in baseline} == {None} and seeds and None not in seeds:
        baseline = [{**r, "comparison_seed": seed} for seed in seeds for r in baseline]
    left, right = (_index(baseline, require_balanced_seeds=require_balanced_seeds),
                   _index(observations, require_balanced_seeds=require_balanced_seeds))
    if not set(right) <= set(left):
        raise ValueError("Control subset contains questions missing from its reader baseline")
    return [left[key] for key in right]


def cluster_interval(values, seed=2026, samples=20_000):
    """values is world_id -> list of paired observation effects."""
    if not values:
        return {"estimate": None, "ci95": None, "stories": 0}
    # Equal story weighting prevents a world with missing views receiving extra weight.
    means = np.asarray([np.mean(v) for _, v in sorted(values.items())], dtype=float)
    estimate = float(means.mean())
    if len(means) < 2:
        return {"estimate": estimate, "ci95": None, "stories": len(means)}
    rng = np.random.default_rng(seed)
    boot = means[rng.integers(0, len(means), size=(samples, len(means)))].mean(axis=1)
    return {"estimate": estimate, "ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
            "stories": len(means)}


def _ratio_interval(values, seed=2026, samples=20_000):
    """Bootstrap story-level (numerator, denominator) pairs together."""
    array = np.asarray([value for _, value in sorted(values.items())], dtype=float)
    if not len(array):
        return {"estimate": None, "ci95": None, "stories": 0, "valid_bootstrap_draws": 0}
    numerator, denominator = array.sum(axis=0)
    estimate = float(numerator / denominator) if denominator else None
    interval, valid_draws = None, 0
    if denominator and len(array) > 1:
        rng = np.random.default_rng(seed)
        draws = array[rng.integers(0, len(array), size=(samples, len(array)))].sum(axis=1)
        valid = draws[:, 1] > 0
        valid_draws = int(valid.sum())
        if valid_draws:
            interval = np.quantile(draws[valid, 0] / draws[valid, 1], [.025, .975]).tolist()
    return {"estimate": estimate, "ci95": interval, "stories": len(array),
            "valid_bootstrap_draws": valid_draws}


def _joint_patterns(rows):
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        groups[(_seed(row), row["world_id"], row.get("cue_condition", "clean"), row.get("cue_location"))][row["view"]].append(row)
    patterns, incomplete = [], 0
    for (_training_seed, world, condition, cue), views in groups.items():
        if set(views) != set(VIEWS):
            incomplete += 1
            continue
        if any(len(values) != 1 for values in views.values()):
            raise ValueError("Joint correctness requires exactly one record per view and cue rendering")
        correct = all(views[view][0]["prediction"] == views[view][0]["answer"] for view in VIEWS)
        patterns.append((world, condition, cue, correct))
    return patterns, incomplete


def accuracy_summary(rows, seed=2026, samples=20_000):
    rows = list(rows)
    _index(rows)
    by_view, overall = defaultdict(lambda: defaultdict(lambda: defaultdict(list))), defaultdict(lambda: defaultdict(list))
    for row in rows:
        correct = int(row["prediction"] == row["answer"])
        by_view[row["view"]][row["world_id"]][_seed(row)].append(correct)
        overall[row["world_id"]][_seed(row)].append(correct)

    def interval(groups):
        return cluster_interval({world: [float(np.mean(scores)) for scores in seeds.values()]
                                 for world, seeds in groups.items()}, seed, samples)

    total = interval(overall)
    views = {}
    for view, groups in sorted(by_view.items()):
        result = interval(groups)
        observations = [value for seeds in groups.values() for values in seeds.values() for value in values]
        views[view] = {**result, "n": len(observations), "accuracy": result["estimate"],
                       "micro_accuracy": float(np.mean(observations)) if observations else None}
    patterns, incomplete = _joint_patterns(rows)
    joint = [correct for _, _, _, correct in patterns]
    joint_by_world = defaultdict(list)
    for world, _, _, correct in patterns:
        joint_by_world[world].append(int(correct))
    joint_interval = cluster_interval(joint_by_world, seed, samples)
    return {"n": len(rows), "accuracy": total["estimate"], "ci95": total["ci95"], "stories": total["stories"],
            "micro_accuracy": float(np.mean([row["prediction"] == row["answer"] for row in rows])) if rows else None,
            "per_view": views, "joint_all_eight": joint_interval["estimate"],
            "joint_all_eight_ci95": joint_interval["ci95"], "joint_all_eight_stories": joint_interval["stories"],
            "joint_complete_stories": len(joint), "joint_incomplete_stories": incomplete,
            "joint_alignment": JOINT_ALIGNMENT, "invalid_answers": sum(row["prediction"] is None for row in rows),
            "bootstrap_samples": samples, "bootstrap_seed": seed,
            "uncertainty": "95% whole-story bootstrap; fixed seeds averaged within each story before resampling stories."}


def transfer(base, trained, seed=2026, samples=20_000):
    base, trained = list(base), list(trained)
    groups = defaultdict(lambda: defaultdict(list))
    ratio_groups = defaultdict(lambda: defaultdict(lambda: np.zeros(4)))
    preserved, baseline_correct, repairs, errors = 0, 0, 0, 0
    for a, b in paired_rows(base, trained):
        ac, bc = a["prediction"] == a["answer"], b["prediction"] == b["answer"]
        delta = int(bc) - int(ac)
        groups[a["view"]][a["world_id"]].append(delta)
        if a["view"] != "direct":
            groups["unrewarded"][a["world_id"]].append(delta)
        scope = a.get("target_scope", "unspecified")
        groups[f"scope:{scope}"][a["world_id"]].append(delta)
        if a["view"] != "direct":
            groups[f"unrewarded_scope:{scope}"][a["world_id"]].append(delta)
        if ac:
            baseline_correct += 1
            preserved += int(bc)
        else:
            errors += 1
            repairs += int(bc)
        ratio_groups[a["world_id"]][_seed(b)] += [int(ac and bc), int(ac), int(not ac and bc), int(not ac)]
    ratios = {world: np.mean(list(seeds.values()), axis=0) for world, seeds in ratio_groups.items()}
    preservation = _ratio_interval({world: values[:2] for world, values in ratios.items()}, seed, samples)
    repair = _ratio_interval({world: values[2:] for world, values in ratios.items()}, seed, samples)
    return {"estimand": "trained_minus_base_direct_answer_accuracy",
            "effects": {k: cluster_interval(v, seed, samples) for k, v in sorted(groups.items())},
            "preservation": {"correct_before": baseline_correct, "still_correct": preserved,
                             **preservation, "rate": preservation["estimate"]},
            "error_repair": {"errors_before": errors, "fixed": repairs,
                             **repair, "fraction": repair["estimate"]},
            "training_seeds": sorted({_seed(r) for r in trained}, key=repr),
            "bootstrap_samples": samples, "bootstrap_seed": seed,
            "uncertainty": "Fixed training seeds averaged within each story before whole-story bootstrap; not uncertainty over a population of seeds.",
            "scope_definition": "Dependence under the specified symbolic belief edit; event-rewrite scope belongs to Test 3.",
            "interpretation": "Transfer to direct-answer inference; does not identify stored beliefs."}


def trace_benefit(baseline, supplied, seed=2026, samples=20_000, *, require_balanced_seeds=True):
    by_world = defaultdict(lambda: [0, 0, 0, 0])
    for a, b in paired_rows(baseline, supplied, require_balanced_seeds=require_balanced_seeds):
        ac, bc = a["prediction"] == a["answer"], b["prediction"] == b["answer"]
        v = by_world[a["world_id"]]
        v[0] += int(not ac)
        v[1] += int(not ac and bc)
        v[2] += int(ac)
        v[3] += int(ac and not bc)
    values = np.asarray(list(by_world.values()), dtype=int)
    if not len(values):
        return {"fraction_errors_fixed": None, "baseline_errors": 0, "ci95": None}
    errors, fixed, correct, broken = values.sum(axis=0).tolist()
    ci = None
    if errors and len(values) > 1:
        rng = np.random.default_rng(seed)
        sums = values[rng.integers(0, len(values), size=(samples, len(values)))].sum(axis=1)
        valid = sums[:, 0] > 0
        ci = np.quantile(sums[valid, 1] / sums[valid, 0], [.025, .975]).tolist()
    return {"fraction_errors_fixed": fixed / errors if errors else None, "ci95": ci,
            "baseline_errors": errors, "errors_fixed": fixed, "baseline_correct": correct,
            "correct_broken": broken, "regression_rate": broken / correct if correct else None,
            "net_accuracy_gain": (fixed - broken) / (errors + correct),
            "ceiling": errors == 0}


def _crossed_original_contrasts(baselines, cells, identities, seed=2026, samples=20_000):
    """Paired cluster bootstrap of differences in reader-specific error repair."""
    br, tr, bs, ts = identities
    timings = sorted({key[3] for key in cells if key[2] == "original"})
    output = {}
    weights = {
        "trace_source_effect_for_base_reader": [-1, 1, 0, 0],
        "trace_source_effect_for_trained_reader": [0, 0, -1, 1],
        "reader_effect_for_base_trace": [-1, 0, 1, 0],
        "reader_effect_for_trained_trace": [0, -1, 0, 1],
        "interaction": [1, -1, -1, 1],
    }
    for timing in timings:
        keys = [(reader, source, "original", timing) for reader in (br, tr) for source in (bs, ts)]
        if not all(key in cells for key in keys):
            output[timing] = {"status": "incomplete_2x2"}
            continue
        values = defaultdict(lambda: np.zeros(6, dtype=int))
        # Columns: base errors, trained errors, repaired errors in BB/BT/TB/TT.
        for index, reader in enumerate((br, tr)):
            for row in baselines[reader]:
                values[row["world_id"]][index] += int(row["prediction"] != row["answer"])
        # Pairing the baselines across readers also checks the shared question set.
        list(paired_rows(baselines[br], baselines[tr]))
        for index, key in enumerate(keys):
            for baseline, supplied in paired_rows(baselines[key[0]], cells[key]):
                values[baseline["world_id"]][index + 2] += int(
                    baseline["prediction"] != baseline["answer"] and supplied["prediction"] == supplied["answer"])
        array = np.asarray([value for _, value in sorted(values.items())])
        if not len(array):
            output[timing] = {"status": "no_stories"}
            continue

        def benefits(totals):
            denominator = totals[..., [0, 0, 1, 1]]
            return np.divide(totals[..., 2:], denominator, out=np.full(denominator.shape, np.nan), where=denominator > 0)

        estimates = benefits(array.sum(axis=0))
        boot = None
        if len(array) > 1:
            rng = np.random.default_rng(seed)
            boot = benefits(array[rng.integers(0, len(array), size=(samples, len(array)))].sum(axis=1))
        contrasts = {}
        for name, coefficients in weights.items():
            coefficients = np.asarray(coefficients)
            selected = coefficients != 0
            estimate = estimates[selected] @ coefficients[selected]
            interval, valid_draws = None, 0
            if boot is not None:
                draws = boot[:, selected] @ coefficients[selected]
                draws = draws[np.isfinite(draws)]
                valid_draws = len(draws)
                if valid_draws:
                    interval = np.quantile(draws, [.025, .975]).tolist()
            contrasts[name] = {"estimate": float(estimate) if np.isfinite(estimate) else None,
                               "ci95": interval, "valid_bootstrap_draws": valid_draws,
                               "stories": len(array)}
        output[timing] = {"status": "estimated", "contrasts": contrasts,
                          "estimand": "Difference in fractions of each reader's own no-trace errors repaired",
                          "bootstrap_seed": seed, "bootstrap_samples": samples}
    return output


EDIT_STATUSES = ("editable", "belief_not_stated", "contradicted", "ambiguous", "no_safe_control", "generation_failed", "not_assessed")
EDIT_CONDITIONS = frozenset(("semantic_minimal", "semantic_regenerated", "paraphrase", "irrelevant"))


def _source_checkpoint_key(row):
    from .io import digest
    return (row.get("source", "unspecified"), digest(row.get("source_checkpoint")),
            row.get("question_timing", "question_dependent"))


def _editability_counts(rows):
    """Count source artifacts once, despite eight views and two crossed readers."""
    artifacts = {}
    for row in rows:
        status = row.get("edit_status")
        initial = row.get("edit_initial_status", status)
        if status not in EDIT_STATUSES or initial not in EDIT_STATUSES:
            raise ValueError("Selected edit candidates need an explicit supported edit_status")
        fallback = (_source_checkpoint_key(row), _seed(row), row["world_id"],
                    row.get("cue_condition"), row.get("cue_location"))
        key = row.get("artifact_id", fallback)
        value = (row["world_id"], status, initial, bool(row.get("edit_human_corrected", False)),
                 row.get("edit_source_belief_correct"))
        if key in artifacts and artifacts[key] != value:
            raise ValueError("One source artifact has conflicting editability statuses or correction history")
        artifacts[key] = value

    def categories(index):
        return {status: {"artifacts": sum(value[index] == status for value in artifacts.values()),
                         "worlds": len({value[0] for value in artifacts.values() if value[index] == status})}
                for status in EDIT_STATUSES}

    final, initial = categories(1), categories(2)
    return {"candidate_artifacts": len(artifacts),
            "candidate_worlds": len({value[0] for value in artifacts.values()}),
            "editable_artifacts": final["editable"]["artifacts"], "editable_worlds": final["editable"]["worlds"],
            "editable_fraction": final["editable"]["artifacts"] / len(artifacts) if artifacts else None,
            "categories": final, "initial_categories": initial,
            "editable_source_belief_correctness": {
                label: sum(value[1] == "editable" and value[4] is truth for value in artifacts.values())
                for label, truth in (("correct", True), ("incorrect", False), ("unknown", None))},
            "human_corrected_artifacts": sum(value[3] for value in artifacts.values()),
            "correction_history_note": "Initial machine failures remain counted after human correction; final eligibility defines the conditional edit estimand.",
            "category_world_counts_note": "World counts can overlap categories when multiple cue renderings are selected."}


def _editability_intervals(rows, seed=2026, samples=20_000):
    """One source artifact per cue rendering; average source seeds within story."""
    artifacts = {}
    for row in rows:
        source_seed = row.get("source_training_seed", _seed(row))
        fallback = (_source_checkpoint_key(row), source_seed, row["world_id"],
                    row.get("cue_condition"), row.get("cue_location"))
        key = row.get("artifact_id", fallback)
        value = (row["world_id"], source_seed, row["edit_status"], row.get("edit_initial_status", row["edit_status"]))
        if key in artifacts and artifacts[key] != value:
            raise ValueError("One source artifact has conflicting editability seed metadata")
        artifacts[key] = value
    final, initial = {}, {}
    for index, target in ((2, final), (3, initial)):
        for status in EDIT_STATUSES:
            groups = defaultdict(lambda: defaultdict(list))
            for value in artifacts.values():
                groups[value[0]][value[1]].append(int(value[index] == status))
            target[status] = cluster_interval({world: [float(np.mean(values)) for values in seeds.values()]
                                               for world, seeds in groups.items()}, seed, samples)
    return {"category_rates": final, "initial_category_rates": initial,
            "editable_fraction": final["editable"]["estimate"], "editable_fraction_ci95": final["editable"]["ci95"],
            "source_training_seeds": sorted({value[1] for value in artifacts.values()}, key=repr),
            "bootstrap_samples": samples, "bootstrap_seed": seed,
            "uncertainty": "95% whole-story bootstrap; source artifacts counted once, cue renderings averaged within source seed, then fixed source seeds within story."}


def _editability_summary(cells, seed=2026, samples=20_000):
    """Expose noneditable candidates and support differences, never drop them."""
    groups = defaultdict(list)
    for (reader, source, condition, timing), rows in cells.items():
        if condition != "original":
            continue
        for row in rows:
            if row.get("edit_subset_selected"):
                if row.get("edit_status") is None and row.get("edit_precision_plan"):
                    raise ValueError("Selected candidate panel is missing editability status")
                if row.get("edit_status") is not None:
                    groups[_source_checkpoint_key(row)].append(row)
    if not groups:
        return {"status": "not_recorded", "by_source_checkpoint": {}, "by_source_arm": {}, "paired_source_support": []}
    reports, eligible_sets, candidate_sets = {}, {}, {}
    for key, rows in sorted(groups.items()):
        label = "|".join(key)
        plans = [row.get("edit_precision_plan") for row in rows]
        if any(plan != plans[0] for plan in plans):
            raise ValueError("Source checkpoint mixes candidate precision plans")
        plan = plans[0]
        _validate_edit_precision_plan(plan)
        report = _editability_counts(rows)
        report["micro_editable_fraction"] = report["editable_fraction"]
        report.update(_editability_intervals(rows, seed, samples))
        support_key = lambda row: (row["world_id"], row.get("cue_condition", "clean"), row.get("cue_location"))
        eligible_sets[key] = {support_key(row) for row in rows if row["edit_status"] == "editable"}
        candidate_sets[key] = {support_key(row) for row in rows}
        if plan and plan["stage"] == "confirmatory":
            if report["candidate_worlds"] != plan["selected_worlds"]:
                raise ValueError("Confirmatory editability report is missing planned candidate worlds")
            # Each reader/comparison replicate must retain every candidate panel.
            panels = defaultdict(list)
            for row in rows:
                panels[(row.get("reader"), _seed(row))].append(row)
            for panel in panels.values():
                if len({row["world_id"] for row in panel}) != plan["selected_worlds"]:
                    raise ValueError("Confirmatory editability replicate is missing candidate worlds")
                _index(panel)
                if _joint_patterns(panel)[1]:
                    raise ValueError("Confirmatory editability needs complete eight-view candidate panels")
        report.update(source=key[0], source_checkpoint=rows[0].get("source_checkpoint"), question_timing=key[2],
                      precision_plan=plan,
                      conditional_effect_status="requires_edited_outcomes" if report["editable_worlds"] else (
                          "awaiting_editability_assessment" if report["categories"]["not_assessed"]["artifacts"]
                          else "not_estimable_no_editable_worlds"),
                      precision_interpretation="Report actual whole-story confidence intervals; no precision cutoff")
        reports[label] = report
    arm_groups = defaultdict(list)
    for (source, checkpoint, timing), rows in groups.items():
        arm_groups[(source, timing)].extend(rows)
    arm_reports = {}
    for (source, timing), rows in sorted(arm_groups.items()):
        report = _editability_counts(rows)
        report["micro_editable_fraction"] = report["editable_fraction"]
        report.update(_editability_intervals(rows, seed, samples))
        report.update(source=source, question_timing=timing,
                      source_checkpoints=sorted({key[1] for key in groups if key[0] == source and key[2] == timing}),
                      scope="Selected source-arm/dataset comparison; do not merge different trained arms under one source label")
        arm_reports[f"{source}|{timing}"] = report
    from itertools import combinations
    support = []
    for left, right in combinations(sorted(groups), 2):
        if left[2] != right[2]:
            continue
        same_candidates = candidate_sets[left] == candidate_sets[right]
        common = eligible_sets[left] & eligible_sets[right]
        identical = same_candidates and eligible_sets[left] == eligible_sets[right]
        support.append({"left": "|".join(left), "right": "|".join(right),
                        "candidate_pools_equal": same_candidates,
                        "left_editable_worlds": len({key[0] for key in eligible_sets[left]}),
                        "right_editable_worlds": len({key[0] for key in eligible_sets[right]}),
                        "common_editable_worlds": len({key[0] for key in common}),
                        "common_world_cue_panels": len(common), "identical_editable_support": identical,
                        "own_subset_estimands_comparable": identical,
                        "interpretation": "Primary conditional estimates retain each source's own support. Separate paired_edit_source_comparisons explicitly estimates any shared-support contrast."})
    return {"status": "reported", "by_source_checkpoint": reports, "by_source_arm": arm_reports,
            "paired_source_support": support,
            "interpretation": "Editability is a separate outcome. Conditional edit effects do not describe absent, contradictory or failed traces."}


def _edit_audit_sensitivity(rows, estimate, *, difference=False):
    """Worst-case effect of flawed controls under the reported sampling bounds.

    The scientific estimator weights worlds equally, then seeds equally within
    story, then views equally within seed. An audit bounds a count of flawed
    source contracts, so allocate that count to the largest contract weights;
    a global flaw fraction must not be assigned independently to every arm.
    """
    import math
    from .io import digest
    if not rows or estimate is None:
        return {"status": "not_estimable", "maximum_absolute_bias": None, "sensitivity_range": None}
    panels = defaultdict(lambda: defaultdict(list))
    for row in rows:
        panels[row["world_id"]][_seed(row)].append(row)
    weights, representatives = defaultdict(float), {}
    for seeds in panels.values():
        for observations in seeds.values():
            weight = 1 / (len(panels) * len(seeds) * len(observations))
            for row in observations:
                artifact = row.get("artifact_id")
                if not artifact:
                    return {"status": "unavailable", "reason": "Contract identities are missing", "maximum_absolute_bias": None,
                            "sensitivity_range": None}
                key = _source_checkpoint_key(row), artifact
                weights[key] += weight
                representatives[key] = row
    by_source = defaultdict(list)
    for key, weight in weights.items():
        by_source[key[0]].append((weight, representatives[key]))
    source_bounds, weight_cap, confidences, families = [], 0., set(), set()
    for source, contracts in sorted(by_source.items()):
        quality_values = [row.get("edit_audit_quality") for _, row in contracts]
        quality = quality_values[0]
        if not isinstance(quality, dict) or any(digest(value) != digest(quality) for value in quality_values):
            return {"status": "unavailable", "reason": "Missing or inconsistent source-checkpoint audit quality", "maximum_absolute_bias": None,
                    "sensitivity_range": None}
        population = quality.get("population_size")
        upper = quality.get("simultaneous_one_sided_upper_rate")
        confidence = quality.get("simultaneous_confidence")
        family_size = quality.get("source_family_size")
        # Changed controls are no longer the originally sampled machine outputs.
        # Count their entire weight conservatively, including formerly unready
        # contracts made editable by human corrections.
        changed = [(weight, row) for weight, row in contracts if row.get("edit_human_corrected")
                   or row.get("edit_initial_status", "editable") != "editable"]
        unchanged = [(weight, row) for weight, row in contracts if not row.get("edit_human_corrected")
                     and row.get("edit_initial_status", "editable") == "editable"]
        if (type(population) is not int or population < len(unchanged)
                or type(confidence) not in (int, float) or not math.isfinite(confidence) or not 0 < confidence < 1
                or type(family_size) is not int or family_size < len(by_source)
                or (unchanged and (type(upper) not in (int, float) or not math.isfinite(upper) or not 0 <= upper <= 1))):
            return {"status": "unavailable", "reason": "No valid simultaneous original-ready source bound", "maximum_absolute_bias": None,
                    "sensitivity_range": None}
        correction_weight = sum(weight for weight, _ in changed)
        flawed_count_cap = min(population, math.ceil(population * upper)) if unchanged else 0
        sampled_weight_cap = sum(sorted((weight for weight, _ in unchanged), reverse=True)[:flawed_count_cap])
        cap = min(sum(weight for weight, _ in contracts), correction_weight + sampled_weight_cap)
        weight_cap += cap
        confidences.add(confidence)
        families.add(family_size)
        source_bounds.append({"source_checkpoint": "|".join(source), "original_ready_population": population,
                              "audit_sample_size": quality.get("sample_size"), "observed_flaws": quality.get("flaws"),
                              "simultaneous_upper_flaw_rate": upper, "source_family_size": quality.get("source_family_size"),
                              "original_ready_flaw_count_cap": flawed_count_cap,
                              "metric_contracts": len(contracts), "metric_weight": sum(w for w, _ in contracts),
                              "changed_contracts_worst_case_weight": correction_weight,
                              "maximum_affected_metric_weight": cap})
    if len(confidences) != 1 or len(families) != 1:
        return {"status": "unavailable", "reason": "Inconsistent simultaneous audit families", "maximum_absolute_bias": None,
                "sensitivity_range": None}
    width, lower, upper = (2., -1., 1.) if difference else (1., 0., 1.)
    cap = min(width, width * weight_cap)
    return {"status": "reported_sensitivity_bound", "maximum_absolute_bias": cap,
            "sensitivity_range": [max(lower, estimate - cap), min(upper, estimate + cap)],
            "simultaneous_confidence": next(iter(confidences)),
            "by_source_checkpoint": source_bounds,
            "interpretation": "Worst-case sensitivity, not a second confidence interval or an acceptance rule. "
                              "Source-specific simultaneous original-ready flaw-count bounds are allocated to the largest actual contract weights. "
                              "Human-changed or originally unready controls receive their full weight conservatively. "
                              "The bound assumes the recorded random audit and human flaw judgements are valid; it does not establish semantic truth."}


def _conditional_edit_metrics(rows, original, plan, seed, samples):
    groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in rows:
        world = row["world_id"]
        follows = int(row["prediction"] == row["expected_edited_answer"])
        groups["all_predicted_consequences"][world][_seed(row)].append(follows)
        metric = "changed_target_compliance" if row["expected_edited_answer"] != row.get("expected_before_edit_answer", row["answer"]) else "intact_target_preservation"
        groups[metric][world][_seed(row)].append(follows)
    if original is not None:
        baseline = _baseline_subset(original, rows, require_balanced_seeds=False)
        for before, after in paired_rows(baseline, rows, require_balanced_seeds=False):
            if after["expected_edited_answer"] != after.get("expected_before_edit_answer", after["answer"]):
                effect = int(after["prediction"] == after["expected_edited_answer"]) - int(
                    before["prediction"] == after["expected_edited_answer"])
                groups["changed_response_increase_vs_original"][after["world_id"]][_seed(after)].append(effect)
    metrics = {}
    for name in ("all_predicted_consequences", "changed_target_compliance", "intact_target_preservation",
                 "changed_response_increase_vs_original"):
        metrics[name] = cluster_interval({world: [float(np.mean(values)) for values in seeds.values()]
                                          for world, seeds in groups[name].items()}, seed=seed, samples=samples)
        selected = rows if name == "all_predicted_consequences" else [
            row for row in rows if (row["expected_edited_answer"] != row.get("expected_before_edit_answer", row["answer"]))
            == (name != "intact_target_preservation")]
        metrics[name]["audit_sensitivity"] = _edit_audit_sensitivity(
            selected, metrics[name]["estimate"], difference=name == "changed_response_increase_vs_original")

    return metrics


def _validate_edit_precision_plan(plan):
    if plan is not None:
        from .precision import edit_precision_plan
        expected = edit_precision_plan(plan["selected_worlds"], stage=plan["stage"],
                     max_worlds=plan["selected_worlds"])
        if plan != expected:
            raise ValueError("Invalid targeted-edit precision plan")


def _edit_precision_summary(rows, original, seed, samples):
    """Conditional edit precision uses actual eligible stories in every checkpoint."""
    _index(rows, require_balanced_seeds=False)
    candidates = [row for row in original or () if row.get("edit_subset_selected")]
    plans = [row.get("edit_precision_plan") for row in rows + candidates]
    plan = plans[0] if plans else None
    if any(candidate != plan for candidate in plans):
        raise ValueError("Targeted edits mix different precision plans")
    _validate_edit_precision_plan(plan)
    explicit_status = bool(candidates) or any(row.get("edit_status") is not None for row in rows)
    if plan and plan["stage"] == "confirmatory":
        if not candidates:
            raise ValueError("Confirmatory targeted edits require the complete candidate editability panel")
        if any(row.get("question_timing") != "direct_fixed" for row in rows + candidates):
            raise ValueError("Primary confirmatory edit precision requires frozen direct-question traces")
        # Keep the original candidate panel balanced across training seeds.
        _index(candidates)
        candidate_groups = defaultdict(list)
        for row in candidates:
            candidate_groups[_source_checkpoint_key(row)].append(row)
        for group in candidate_groups.values():
            if len({row["world_id"] for row in group}) != plan["selected_worlds"]:
                raise ValueError("Confirmatory targeted-edit results are missing planned candidate worlds")
            if _joint_patterns(group)[1]:
                raise ValueError("Confirmatory targeted edits require complete eight-view candidate panels")
    if explicit_status:
        if any(row.get("edit_status") != "editable" for row in rows):
            raise ValueError("Actual edit outcomes must be explicitly editable")
        _editability_counts(candidates)
        eligible = [row for row in candidates if row["edit_status"] == "editable"]
        observed = _index(rows, require_balanced_seeds=False)
        expected = _index(eligible, require_balanced_seeds=False)
        if set(observed) != set(expected):
            raise ValueError("Edited outcomes must cover every editable candidate exactly once; no silent world loss")
        if any(_source_checkpoint_key(observed[key]) != _source_checkpoint_key(expected[key])
               or observed[key].get("artifact_id") != expected[key].get("artifact_id") for key in observed):
            raise ValueError("Edited outcomes changed their candidate source or artifact identity")
    if plan and plan["stage"] == "confirmatory" and _joint_patterns(rows)[1]:
        raise ValueError("Confirmatory targeted edits require complete eight-view panels")
    metrics = _conditional_edit_metrics(rows, original, plan, seed, samples)
    groups, candidate_groups = defaultdict(list), defaultdict(list)
    for row in rows:
        groups[_source_checkpoint_key(row)].append(row)
    for row in candidates:
        candidate_groups[_source_checkpoint_key(row)].append(row)
    checkpoints = {}
    for key in sorted(set(groups) | set(candidate_groups)):
        selected = groups[key]
        subgroup = _conditional_edit_metrics(selected, original, plan, seed, samples)
        counts = _editability_counts(candidate_groups[key]) if explicit_status else {
            "candidate_worlds": None, "editable_worlds": len({row["world_id"] for row in selected})}
        checkpoints["|".join(key)] = {"source_checkpoint": (selected or candidate_groups[key])[0].get("source_checkpoint"),
                                     "editability": counts, "metrics": subgroup}
    return {"plan": plan, "metrics": metrics, "by_source_checkpoint": checkpoints,
            "editability": _editability_counts(candidates) if explicit_status else None,
            "estimand": "Compliance or preservation conditional on the source trace being editable",
            "interpretation": "Fixed views and available replicates averaged within story; checkpoint-specific eligible sets and intervals are reported. "
                              "Candidate stories never substitute for editable stories. Report estimates and intervals without precision pass criteria or quadrant labels."}


def _trace_control_summary(cells, seed=2026, samples=20_000):
    output = {}
    for key, rows in sorted(cells.items()):
        reader, source, condition, timing = key
        if condition == "original":
            continue
        conditional = condition in EDIT_CONDITIONS
        summary = {"n": len(rows)}
        if condition in ("own_replay", "own_rerendered"):
            eligible = [r for r in rows if r.get("replay_eligible", False)]
            checked = [r for r in eligible if r.get("replay_matches") is not None]
            summary.update(eligible=len(eligible), ineligible=len(rows) - len(eligible),
                           forced_delimiter=sum(bool(r.get("source_forced_delimiter")) for r in rows),
                           checked=len(checked), matches=sum(bool(r["replay_matches"]) for r in checked),
                           match_rate=float(np.mean([r["replay_matches"] for r in checked])) if checked else None,
                           replay_mode="raw_token_ids" if condition == "own_replay" else "rerendered_prompt",
                           token_prefix_equal=sum(r.get("replay_token_ids_match") is True for r in eligible),
                           eligibility_rate=len(eligible) / len(rows) if rows else None)
        else:
            summary["ordinary_accuracy"] = float(np.mean([r["prediction"] == r["answer"] for r in rows])) if rows else None
            original = cells.get((reader, source, "original", timing))
            summary["original_comparison_available"] = original is not None
            if original is not None:
                pairs = list(paired_rows(_baseline_subset(original, rows, require_balanced_seeds=not conditional),
                                         rows, require_balanced_seeds=not conditional))
                same_output = lambda a, b: (a["token_id"] == b["token_id"]
                    if "token_id" in a and "token_id" in b else
                    a["prediction"] is not None and a["prediction"] == b["prediction"])
                summary["agreement_with_original"] = float(np.mean([same_output(a, b) for a, b in pairs])) if pairs else None
                summary["agreement_definition"] = "Same next-token ID; fallback is the same valid task label when token IDs are absent"
                correct = [(a, b) for a, b in pairs if a["prediction"] == a["answer"]]
                summary["original_correct"] = len(correct)
                summary["ordinary_correct_broken"] = sum(b["prediction"] != b["answer"] for _, b in correct)
                summary["ordinary_regression_rate"] = summary["ordinary_correct_broken"] / len(correct) if correct else None
            if condition in ("semantic_minimal", "semantic_regenerated"):
                if any(r.get("expected_edited_answer") is None for r in rows):
                    raise ValueError("Targeted edit summaries require an explicit predicted consequence")
                summary["predicted_consequence_rate"] = float(np.mean([
                    r["prediction"] == r["expected_edited_answer"] for r in rows])) if rows else None
                changed = [r for r in rows if r["expected_edited_answer"] != r.get("expected_before_edit_answer", r["answer"])]
                intact = [r for r in rows if r["expected_edited_answer"] == r.get("expected_before_edit_answer", r["answer"])]
                summary["changed_target_n"] = len(changed)
                summary["changed_target_compliance"] = float(np.mean([
                    r["prediction"] == r["expected_edited_answer"] for r in changed])) if changed else None
                summary["intact_target_n"] = len(intact)
                summary["intact_target_preservation"] = float(np.mean([
                    r["prediction"] == r["expected_edited_answer"] for r in intact])) if intact else None
                summary["story_precision"] = _edit_precision_summary(rows, original, seed, samples)
                metrics = summary["story_precision"]["metrics"]
                summary["predicted_consequence_rate"] = metrics["all_predicted_consequences"]["estimate"]
                summary["changed_target_compliance"] = metrics["changed_target_compliance"]["estimate"]
                summary["intact_target_preservation"] = metrics["intact_target_preservation"]["estimate"]
                summary["accuracy_weighting"] = "Equal worlds after averaging views within seed and fixed seeds within story"
        output["|".join(key)] = summary
    return output


def _paired_edit_source_comparisons(cells, seed=2026, samples=20_000):
    """Explicit supplemental paired contrasts on shared editable observations.

    This deliberately does not replace each source's own conditional estimand.
    Shared support is model-dependent and is neither random nor a causal design.
    """
    from itertools import combinations
    grouped = defaultdict(lambda: defaultdict(list))
    for (reader, source, condition, timing), rows in cells.items():
        if condition not in ("semantic_minimal", "semantic_regenerated"):
            continue
        for row in rows:
            if row.get("edit_status") == "editable":
                grouped[(reader, condition, timing)][_source_checkpoint_key(row)].append(row)
    output = []
    for (reader, condition, timing), sources in sorted(grouped.items()):
        for left_key, right_key in combinations(sorted(sources), 2):
            left = _index(sources[left_key], require_balanced_seeds=False)
            right = _index(sources[right_key], require_balanced_seeds=False)
            shared = set(left) & set(right)
            groups = defaultdict(lambda: defaultdict(list))
            for key in sorted(shared, key=repr):
                a, b = left[key], right[key]
                if a["answer"] != b["answer"]:
                    raise ValueError("Paired edited sources have different ordinary targets")
                difference = int(b["prediction"] == b["expected_edited_answer"]) - int(
                    a["prediction"] == a["expected_edited_answer"])
                world = a["world_id"]
                groups["all_predicted_consequences"][world].append(difference)
                changed_a = a["expected_edited_answer"] != a.get("expected_before_edit_answer", a["answer"])
                changed_b = b["expected_edited_answer"] != b.get("expected_before_edit_answer", b["answer"])
                if changed_a and changed_b:
                    groups["changed_for_both_sources"][world].append(difference)
                elif not changed_a and not changed_b:
                    groups["intact_for_both_sources"][world].append(difference)
            output.append({"reader": reader, "condition": condition, "question_timing": timing,
                           "left": "|".join(left_key), "right": "|".join(right_key),
                           "status": "estimated_on_common_support" if shared else "no_common_editable_support",
                           "left_editable_worlds": len({row["world_id"] for row in left.values()}),
                           "right_editable_worlds": len({row["world_id"] for row in right.values()}),
                           "common_records": len(shared), "common_worlds": len({left[key]["world_id"] for key in shared}),
                           "common_comparison_seeds": sorted({key[0] for key in shared}, key=repr),
                           "metrics": {name: cluster_interval(values, seed, samples) for name, values in groups.items()},
                           "estimand": "Right-minus-left edit compliance on the explicitly reported shared editable world/view/cue/comparison-seed panel",
                           "interpretation": "Supplemental paired conditional comparison with observed story-level uncertainty. "
                                             "Eligibility is model-dependent; do not generalize this contrast to all candidates or call it a causal arm effect."})
    return output


def crossed_summary(no_trace, crossed, *, base_reader=None, trained_reader=None,
                    base_source=None, trained_source=None, seed=2026, samples=20_000):
    baselines, cells = defaultdict(list), defaultdict(list)
    for r in no_trace:
        baselines[r["reader"]].append(r)
    for r in crossed:
        cells[(r["reader"], r["source"], r.get("condition", "original"),
               r.get("question_timing", "question_dependent"))].append(r)
    # Reuse a single untrained reference across fixed training seeds without
    # counting it as a different story. Require the full panel for each seed.
    for reader, baseline in list(baselines.items()):
        seeds = {_seed(row) for key, rows in cells.items() if key[0] == reader and key[2] == "original" for row in rows}
        if {_seed(row) for row in baseline} == {None} and seeds and None not in seeds:
            baselines[reader] = [{**row, "comparison_seed": seed} for seed in seeds for row in baseline]
    supplied = {}
    for row in crossed:
        from .io import digest
        if "supplied_trace" in row and digest(row["supplied_trace"]) != row.get("trace_sha256"):
            raise ValueError("Supplied trace text does not match its recorded hash")
        if row.get("condition", "original") == "original" and "source_trace_sha256" in row:
            if row["source_trace_sha256"] != row.get("trace_sha256"):
                raise ValueError("Original reader trace differs from its frozen source artifact")
        if row.get("trace_id") is not None and "trace_sha256" not in row:
            raise ValueError("Trace reader records require supplied-text hashes")
        if "trace_sha256" not in row or row.get("condition") in ("own_replay", "own_rerendered"):
            continue
        key = (_seed(row), row["source"], row["world_id"], row["view"], row.get("cue_condition"),
               row.get("cue_location"), row.get("condition", "original"), row.get("question_timing", "question_dependent"))
        if key in supplied and supplied[key] != row["trace_sha256"]:
            raise ValueError("Crossed readers received different trace text for a fixed source/control")
        supplied[key] = row["trace_sha256"]
    output = {}
    for key, rows in sorted(cells.items()):
        if key[0] not in baselines:
            raise ValueError(f"Missing own no-trace baseline for {key[0]}")
        # Primary original cells need the full panel. Reviewed edit controls and
        # fixed-direct replay use their explicitly recorded subsets.
        conditional = key[2] in EDIT_CONDITIONS
        baseline = baselines[key[0]] if key[2] == "original" else _baseline_subset(
            baselines[key[0]], rows, require_balanced_seeds=not conditional)
        output["|".join(key)] = trace_benefit(baseline, rows, seed=seed, samples=samples,
                                            require_balanced_seeds=not conditional)
        if conditional:
            output["|".join(key)]["estimand_scope"] = "Source-specific editable subset; eligibility may differ across checkpoints"
        output["|".join(key)]["evaluated_records"] = len(rows)
    identities = (base_reader, trained_reader, base_source, trained_source)
    if all(value is None for value in identities):
        readers = {key[0] for key in cells if key[2] == "original"}
        sources = {key[1] for key in cells if key[2] == "original"}
        if readers == sources == {"base", "trained"}:
            identities = ("base", "trained", "base", "trained")
    elif any(value is None for value in identities):
        raise ValueError("Specify all four base/trained reader/source identities or none")
    if all(value is not None for value in identities):
        if identities[0] == identities[1] or identities[2] == identities[3]:
            raise ValueError("Base and trained reader/source identities must be distinct")
        if not {identities[0], identities[1]} <= set(baselines):
            raise ValueError("Missing designated reader baselines")
        contrasts = _crossed_original_contrasts(baselines, cells, identities, seed=seed, samples=samples)
    else:
        contrasts = {"status": "identities_required", "reason": "Supply unambiguous base/trained reader and source labels"}
    return {"cells": output, "original_2x2": contrasts, "controls": _trace_control_summary(cells, seed, samples),
            "editability": _editability_summary(cells, seed, samples),
            "paired_edit_source_comparisons": _paired_edit_source_comparisons(cells, seed, samples),
            "identities": dict(zip(("base_reader", "trained_reader", "base_source", "trained_source"), identities)),
            "supplied_text_identity_verified": bool(supplied),
            "uncertainty": "Whole-story bootstrap with complete fixed-seed panels for original traces; conditional edit panels report available eligible replicates and checkpoint-specific intervals. No inference over a population of seeds.",
            "interpretation": "Reader/source interaction is not a fraction of gain stored in a trace."}


def match_checkpoints(dev_summaries, *, expected_runs=None):
    """Match saved checkpoints at the highest accuracy every compared run reaches.

    The comparison set is explicit in the output. Optional expected_runs checks
    that a caller supplied every intended arm/dataset/seed, without assuming that
    every comparison must include the complete experimental grid.
    """
    from pathlib import Path
    from decimal import Decimal
    import math
    from .io import file_digest
    from .training.config import ARMS
    dev_summaries = list(dev_summaries)
    if any(row.get("split") != "development" for row in dev_summaries):
        raise ValueError("Accuracy matching must use development data only")

    def run_key(row):
        if any(row.get(field) is None for field in ("arm", "dataset", "seed")):
            raise ValueError("Matching needs explicit arm/dataset/seed identities")
        if row["arm"] not in ARMS:
            raise ValueError("Unknown trained arm in matching comparison set")
        return f"{row['arm']}|{row['dataset']}|seed={row['seed']}"

    by_run, identities = defaultdict(list), {}
    base_records = 0
    for row in dev_summaries:
        if row.get("arm") in ("base", "untrained_base") or (
                row.get("arm") is None and row.get("checkpoint_kind") == "base"):
            base_records += 1
            continue
        if row.get("format") != "immediate" or row.get("view") != "direct":
            continue
        key = run_key(row)
        if (type(row.get("step")) is not int or row["step"] < 0
                or not isinstance(row.get("checkpoint_path"), str)
                or type(row.get("accuracy")) not in (int, float)
                or not math.isfinite(row["accuracy"]) or not 0 <= row["accuracy"] <= 1):
            raise ValueError("Matching needs a measured accuracy, nonnegative step and saved checkpoint_path")
        checkpoint = Path(row["checkpoint_path"])
        if not checkpoint.is_dir() or not (checkpoint / "adapter_config.json").is_file():
            raise ValueError(f"Evaluated adapter does not exist: {checkpoint}")
        weights = checkpoint / "adapter_model.safetensors"
        if not weights.is_file():
            weights = checkpoint / "adapter_model.bin"
        if not weights.is_file():
            raise ValueError(f"Evaluated adapter weights do not exist: {checkpoint}")
        if row.get("adapter_sha256") and file_digest(weights) != row["adapter_sha256"]:
            raise ValueError("Evaluated adapter has changed since measurement")
        identity = row.get("checkpoint_id", row["checkpoint_path"])
        if not isinstance(identity, str) or not identity:
            raise ValueError("Matching requires a stable checkpoint ID")
        record_key = key, row["step"], identity
        if record_key in identities:
            if identities[record_key] != row:
                raise ValueError("Conflicting measurements for one evaluated checkpoint")
            continue
        identities[record_key] = row
        by_run[key].append(row)
    if not by_run:
        raise ValueError("Matching requires saved immediate development checkpoints for trained-arm runs")
    declared = None
    if expected_runs is not None:
        if not isinstance(expected_runs, list) or not expected_runs:
            raise ValueError("Expected runs must be a nonempty list of arm/dataset/seed identities")
        declared = [run_key(row) for row in expected_runs]
        if len(declared) != len(set(declared)):
            raise ValueError("Expected matching runs contain duplicates")
        if set(declared) != set(by_run):
            missing, unexpected = sorted(set(declared) - set(by_run)), sorted(set(by_run) - set(declared))
            raise ValueError(f"Matching comparison set differs from expected runs: missing={missing}, unexpected={unexpected}")
    maxima = {key: max(row["accuracy"] for row in rows) for key, rows in by_run.items()}
    target = min(maxima.values())
    matches = {}
    for key, rows in sorted(by_run.items()):
        checkpoint = min(rows, key=lambda row: (abs(Decimal(str(row["accuracy"])) - Decimal(str(target))), row["step"],
                                               row.get("checkpoint_id", row["checkpoint_path"])))
        gap = checkpoint["accuracy"] - target
        matches[key] = {"checkpoint": checkpoint, "maximum_reached_accuracy": maxima[key],
                        "actual_accuracy": checkpoint["accuracy"], "accuracy_gap": gap,
                        "absolute_accuracy_gap": abs(gap), "evaluated_checkpoints": len(rows)}
    return {"method": "highest_common_reached_accuracy_nearest_saved_checkpoint_v1",
            "target_accuracy": target, "comparison_set": sorted(by_run),
            "comparison_set_source": "validated_expected_runs" if declared is not None else "supplied_development_measurements",
            "excluded_base_records": base_records, "matches": matches,
            "tie_break": "Nearest accuracy, then earlier step, then lexicographic checkpoint ID",
            "interpretation": "The target is the minimum of the compared runs' maximum measured direct-answer accuracies. "
                              "Report achieved gaps; no tolerance cutoff or interpolation. Step-zero adapters may be selected. "
                              "An input-defined set does not prove that omitted experimental runs were supplied."}


def cue_summary(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["cue_condition"]].append(row)
    result = {k: accuracy_summary(v) for k, v in groups.items()}
    for key, group in groups.items():
        world_accuracy, view_accuracy = defaultdict(list), defaultdict(lambda: defaultdict(list))
        for row in group:
            correct = row["prediction"] == row["answer"]
            world_accuracy[row["world_id"]].append(correct)
            view_accuracy[row["view"]][row["world_id"]].append(correct)
        result[key]["micro_accuracy"] = result[key].get("micro_accuracy", result[key]["accuracy"])
        result[key]["accuracy"] = float(np.mean([np.mean(values) for values in world_accuracy.values()])) if world_accuracy else None
        result[key]["accuracy_weighting"] = "equal_worlds_after_averaging_renderings"
        for view, worlds in view_accuracy.items():
            result[key]["per_view"][view]["micro_accuracy"] = result[key]["per_view"][view].get("micro_accuracy", result[key]["per_view"][view]["accuracy"])
            result[key]["per_view"][view]["accuracy"] = float(np.mean([np.mean(values) for values in worlds.values()]))
        joint_worlds = defaultdict(list)
        for world, _, _, correct in _joint_patterns(group)[0]:
            joint_worlds[world].append(correct)
        result[key]["micro_joint_all_eight"] = result[key]["joint_all_eight"]
        result[key]["joint_all_eight"] = float(np.mean([np.mean(values) for values in joint_worlds.values()])) if joint_worlds else None
        result[key]["joint_unique_worlds"] = len(joint_worlds)
        direct = [r for r in group if r["view"] == "direct" and r.get("cue_answer") is not None]
        result[key]["cue_present_direct_n"] = len(direct)
        following = defaultdict(list)
        for row in direct:
            following[row["world_id"]].append(row["prediction"] == row["cue_answer"])
        result[key]["direct_cue_following_rate"] = float(np.mean([
            np.mean(values) for values in following.values()])) if following else None
    if "agree" in groups and "conflict" in groups:
        # Independent has six and conflict five counterbalanced renderings per world.
        # Average those replicates within world/view before taking a paired contrast.
        condition_means = {}
        for condition in ("agree", "conflict"):
            values = defaultdict(list)
            for row in groups[condition]:
                values[(row["world_id"], row["view"])].append(row["prediction"] == row["answer"])
            condition_means[condition] = {k: float(np.mean(v)) for k, v in values.items()}
        if set(condition_means["agree"]) != set(condition_means["conflict"]):
            raise ValueError("Cue contrasts require the same underlying worlds and views")
        effects, by_view = defaultdict(list), defaultdict(lambda: defaultdict(list))
        for (world, view), value in condition_means["agree"].items():
            delta = value - condition_means["conflict"][(world, view)]
            effects[world].append(delta)
            by_view[view][world].append(delta)
        result["direct_agree_minus_conflict"] = cluster_interval(by_view.get("direct", {}))
        result["per_view_agree_minus_conflict"] = {view: cluster_interval(values) for view, values in by_view.items()}
        result["pooled_direct_anchored_cue_effect"] = cluster_interval(effects)
        result["cue_contrast_definition"] = ("Cue conditions are assigned against the direct answer and held fixed across views. "
            "The pooled effect is an overall response to that manipulation, not agreement with every evaluated answer.")
    return result
