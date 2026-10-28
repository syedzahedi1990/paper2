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


def _index(rows):
    result = {}
    for row in rows:
        key = (_seed(row), row["world_id"], row["view"], row.get("cue_condition", "clean"), row.get("cue_location", ""))
        if key in result:
            raise ValueError(f"Duplicate evaluation key: {key}; select one checkpoint/format first")
        result[key] = row
    panels = defaultdict(set)
    for key in result:
        panels[key[0]].add(key[1:])
    if len(panels) > 1 and any(panel != next(iter(panels.values())) for panel in panels.values()):
        raise ValueError("Every training seed must cover the same complete story/view panel")
    return result


def paired_rows(base, other):
    base, other = list(base), list(other)
    seeds = {_seed(r) for r in other}
    if {_seed(r) for r in base} == {None} and seeds and None not in seeds:
        base = [{**r, "training_seed": seed} for seed in seeds for r in base]
    a, b = _index(base), _index(other)
    if set(a) != set(b):
        raise ValueError("Paired comparisons require exactly the same worlds, views and cue conditions")
    for key in sorted(a, key=repr):
        if a[key]["answer"] != b[key]["answer"]:
            raise ValueError("Ordinary task targets differ between paired readers")
        yield a[key], b[key]


def _baseline_subset(baseline, observations):
    """Select a prespecified control panel, preserving the reader's own baseline."""
    seeds = {_seed(r) for r in observations}
    if {_seed(r) for r in baseline} == {None} and seeds and None not in seeds:
        baseline = [{**r, "comparison_seed": seed} for seed in seeds for r in baseline]
    left, right = _index(baseline), _index(observations)
    if not set(right) <= set(left):
        raise ValueError("Control subset contains questions missing from its reader baseline")
    return [left[key] for key in right]


def cluster_interval(values, seed=2026, samples=2000):
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


def accuracy_summary(rows):
    _index(rows)
    by_view = defaultdict(list)
    for row in rows:
        correct = row["prediction"] == row["answer"]
        by_view[row["view"]].append(correct)
    patterns, incomplete = _joint_patterns(rows)
    joint = [correct for _, _, _, correct in patterns]
    return {"n": len(rows), "accuracy": float(np.mean([r["prediction"] == r["answer"] for r in rows])) if rows else None,
            "per_view": {v: {"n": len(x), "accuracy": float(np.mean(x))} for v, x in sorted(by_view.items())},
            "joint_all_eight": float(np.mean(joint)) if joint else None,
            "joint_complete_stories": len(joint), "joint_incomplete_stories": incomplete,
            "joint_alignment": JOINT_ALIGNMENT,
            "invalid_answers": sum(r["prediction"] is None for r in rows)}


def transfer(base, trained, seed=2026):
    groups = defaultdict(lambda: defaultdict(list))
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
    return {"estimand": "trained_minus_base_direct_answer_accuracy",
            "effects": {k: cluster_interval(v, seed) for k, v in sorted(groups.items())},
            "preservation": {"correct_before": baseline_correct, "still_correct": preserved,
                             "rate": preserved / baseline_correct if baseline_correct else None},
            "error_repair": {"errors_before": errors, "fixed": repairs,
                             "fraction": repairs / errors if errors else None},
            "training_seeds": sorted({_seed(r) for r in trained}, key=repr),
            "uncertainty": "Fixed training seeds averaged within each story before whole-story bootstrap; not uncertainty over a population of seeds.",
            "scope_definition": "Dependence under the specified symbolic belief edit; event-rewrite scope belongs to Test 3.",
            "interpretation": "Transfer to direct-answer inference; does not identify stored beliefs."}


def trace_benefit(baseline, supplied, seed=2026, samples=2000):
    by_world = defaultdict(lambda: [0, 0, 0, 0])
    for a, b in paired_rows(baseline, supplied):
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


def _crossed_original_contrasts(baselines, cells, identities, seed=2026, samples=2000):
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


def _trace_control_summary(cells):
    output = {}
    for key, rows in sorted(cells.items()):
        reader, source, condition, timing = key
        if condition == "original":
            continue
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
                pairs = list(paired_rows(_baseline_subset(original, rows), rows))
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
                changed = [r for r in rows if r["expected_edited_answer"] != r["answer"]]
                intact = [r for r in rows if r["expected_edited_answer"] == r["answer"]]
                summary["changed_target_n"] = len(changed)
                summary["changed_target_compliance"] = float(np.mean([
                    r["prediction"] == r["expected_edited_answer"] for r in changed])) if changed else None
                summary["intact_target_n"] = len(intact)
                summary["intact_target_preservation"] = float(np.mean([
                    r["prediction"] == r["answer"] for r in intact])) if intact else None
        output["|".join(key)] = summary
    return output


def crossed_summary(no_trace, crossed, *, base_reader=None, trained_reader=None,
                    base_source=None, trained_source=None, seed=2026, samples=2000):
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
        baseline = baselines[key[0]] if key[2] == "original" else _baseline_subset(baselines[key[0]], rows)
        output["|".join(key)] = trace_benefit(baseline, rows, seed=seed, samples=samples)
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
    return {"cells": output, "original_2x2": contrasts, "controls": _trace_control_summary(cells),
            "identities": dict(zip(("base_reader", "trained_reader", "base_source", "trained_source"), identities)),
            "supplied_text_identity_verified": bool(supplied),
            "uncertainty": "Whole-story bootstrap after averaging fixed-seed error/repair counts within story; no inference over a population of seeds.",
            "interpretation": "Reader/source interaction is not a fraction of gain stored in a trace."}


def match_checkpoints(dev_summaries, target_accuracy, tolerance=.02):
    """Select an existing evaluated adapter separately for every arm/dataset/seed."""
    from pathlib import Path
    from .io import file_digest
    if not 0 <= target_accuracy <= 1 or tolerance < 0:
        raise ValueError("Invalid matching target or tolerance")
    if any(r.get("split") != "development" for r in dev_summaries):
        raise ValueError("Accuracy matching must use development data only")
    by_run = defaultdict(list)
    for row in dev_summaries:
        if row.get("format") != "immediate" or row.get("view") != "direct":
            continue
        if any(row.get(field) is None for field in ("arm", "dataset", "seed", "step", "checkpoint_path")):
            raise ValueError("Matching needs arm/dataset/seed/step and a saved checkpoint_path")
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
        key = f"{row['arm']}|{row['dataset']}|seed={row['seed']}"
        by_run[key]  # Retain runs with no acceptable match, rather than dropping seeds.
        if abs(row["accuracy"] - target_accuracy) <= tolerance:
            by_run[key].append(row)
    return {key: {"status": "matched", "checkpoint": min(rows, key=lambda r: (abs(r["accuracy"] - target_accuracy), r["step"]))}
            if rows else {"status": "no_match", "checkpoint": None}
            for key, rows in sorted(by_run.items())}


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
        result[key]["micro_accuracy"] = result[key]["accuracy"]
        result[key]["accuracy"] = float(np.mean([np.mean(values) for values in world_accuracy.values()])) if world_accuracy else None
        result[key]["accuracy_weighting"] = "equal_worlds_after_averaging_renderings"
        for view, worlds in view_accuracy.items():
            result[key]["per_view"][view]["micro_accuracy"] = result[key]["per_view"][view]["accuracy"]
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
