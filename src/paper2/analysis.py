"""Proposal estimands with paired story-level uncertainty and explicit denominators."""
from collections import defaultdict
import numpy as np
from .data import VIEWS

JOINT_ALIGNMENT = (
    "Joint correctness requires all eight views of the same underlying world, cue condition and literal cue location. "
    "The cue condition is anchored to the direct-belief target and remains fixed across consequences. "
    "Incomplete eight-view patterns are counted separately and never treated as joint successes."
)


def _index(rows):
    result = {}
    for row in rows:
        key = (row["world_id"], row["view"], row.get("cue_condition", "clean"), row.get("cue_location", ""))
        if key in result:
            raise ValueError(f"Duplicate evaluation key: {key}; select one checkpoint/format first")
        result[key] = row
    return result


def paired_rows(base, other):
    a, b = _index(base), _index(other)
    if set(a) != set(b):
        raise ValueError("Paired comparisons require exactly the same worlds, views and cue conditions")
    for key in sorted(a):
        if a[key]["answer"] != b[key]["answer"]:
            raise ValueError("Ordinary task targets differ between paired readers")
        yield a[key], b[key]


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
        groups[(row["world_id"], row.get("cue_condition", "clean"), row.get("cue_location"))][row["view"]].append(row)
    patterns, incomplete = [], 0
    for (world, condition, cue), views in groups.items():
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
        if condition == "own_replay":
            eligible = [r for r in rows if r.get("replay_eligible", False)]
            checked = [r for r in eligible if r.get("replay_matches") is not None]
            summary.update(eligible=len(eligible), ineligible=len(rows) - len(eligible),
                           forced_delimiter=sum(bool(r.get("source_forced_delimiter")) for r in rows),
                           checked=len(checked), matches=sum(bool(r["replay_matches"]) for r in checked),
                           match_rate=float(np.mean([r["replay_matches"] for r in checked])) if checked else None)
        else:
            summary["ordinary_accuracy"] = float(np.mean([r["prediction"] == r["answer"] for r in rows])) if rows else None
            original = cells.get((reader, source, "original", timing))
            summary["original_comparison_available"] = original is not None
            if original is not None:
                pairs = list(paired_rows(original, rows))
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
    output = {}
    for key, rows in sorted(cells.items()):
        if key[0] not in baselines:
            raise ValueError(f"Missing own no-trace baseline for {key[0]}")
        # Each timing/control cell must contain exactly the baseline questions.
        output["|".join(key)] = trace_benefit(baselines[key[0]], rows, seed=seed, samples=samples)
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
            "interpretation": "Reader/source interaction is not a fraction of gain stored in a trace."}


def match_checkpoints(dev_summaries, target_accuracy, tolerance=.02):
    if any(r.get("split") != "development" for r in dev_summaries):
        raise ValueError("Accuracy matching must use development data only")
    by_arm = defaultdict(list)
    for row in dev_summaries:
        if row.get("format") != "immediate" or row.get("view") != "direct":
            raise ValueError("Match on immediate direct-belief accuracy only")
        if abs(row["accuracy"] - target_accuracy) <= tolerance:
            by_arm[row["arm"]].append(row)
    return {arm: min(rows, key=lambda r: (abs(r["accuracy"] - target_accuracy), r["step"]))
            for arm, rows in sorted(by_arm.items())}


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
        effects = defaultdict(list)
        for (world, view), value in condition_means["agree"].items():
            effects[world].append(value - condition_means["conflict"][(world, view)])
        result["agree_minus_conflict"] = cluster_interval(effects)
    return result
