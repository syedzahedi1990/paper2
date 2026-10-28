import copy
import random

import pytest

from paper2.analysis import (_edit_precision_summary, _editability_summary, _paired_edit_source_comparisons,
                             _trace_control_summary, crossed_summary)
from paper2.data.world import VIEWS, answer_label, generate_world, ordinary_target, render_record, rewrite_forecast
from paper2.precision import edit_precision_plan
from paper2.traces import symbolic_edit_prediction, _validate_oracle_edit


def observations(n, plan, seeds=(1,)):
    return [{"world_id": str(i), "view": view, "answer": "A", "expected_edited_answer": "B",
             "prediction": "B" if i % 2 else "A", "training_seed": seed, "edit_precision_plan": plan,
             "question_timing": "direct_fixed"}
            for i in range(n) for seed in seeds for view in VIEWS]


def test_confirmatory_sample_size_is_planned_before_effects_and_small_pool_fails():
    plan = edit_precision_plan(120, stage="confirmatory")
    assert plan["required_worlds"] == plan["required_eligible_worlds"] == 97
    assert plan["selected_worlds"] == 120
    assert plan["minimum_editable_fraction_for_target"] == 97 / 120
    assert plan["planned_half_width_if_all_editable"] <= .10
    with pytest.raises(ValueError, match="at least 97"):
        edit_precision_plan(120, stage="confirmatory", max_worlds=12)
    with pytest.raises(ValueError, match="at least 385"):
        edit_precision_plan(120, stage="confirmatory", half_width=.05)
    assert edit_precision_plan(120)["selected_worlds"] == 12


def test_repeated_views_and_seeds_do_not_create_precision_and_pilot_never_certifies_it():
    plan = edit_precision_plan(120)
    rows = observations(12, plan)
    result = _edit_precision_summary(rows, None, 2026, 1000)["metrics"]["changed_target_compliance"]
    replicated = _edit_precision_summary(observations(12, plan, (1, 2, 3)), None, 2026, 1000)
    assert result == replicated["metrics"]["changed_target_compliance"]
    assert result["stories"] == 12 and result["observed_half_width"] > .20
    assert result["confirmatory_precision_ready"] is False
    for row in rows:
        row["prediction"] = "B"
    at_ceiling = _edit_precision_summary(rows, None, 2026, 1000)["metrics"]["changed_target_compliance"]
    assert at_ceiling["observed_half_width"] == 0
    assert at_ceiling["confirmatory_precision_ready"] is False


def candidate_rows(plan, *, seeds=(1,), statuses=None, n=120):
    rows = observations(n, plan, seeds)
    for row in rows:
        seed, world = row["training_seed"], int(row["world_id"])
        row.update(source="trained", reader="reader", source_checkpoint={"id": f"checkpoint-{seed}"},
                   source_training_seed=seed, edit_subset_selected=True,
                   artifact_id=f"checkpoint-{seed}-world-{world}", condition="original",
                   expected_before_edit_answer="A",
                   edit_status=statuses(seed, world) if statuses else "editable")
    return rows


def edited_rows(candidates, condition="semantic_minimal"):
    return [dict(row, condition=condition) for row in candidates if row["edit_status"] == "editable"]


def test_confirmatory_report_checks_actual_precision_and_missing_worlds():
    plan = edit_precision_plan(120, stage="confirmatory")
    original = candidate_rows(plan)
    rows = edited_rows(original)
    result = _edit_precision_summary(rows, original, 2026, 3000)["metrics"]["changed_target_compliance"]
    assert result["stories"] == 120 and result["observed_half_width"] < .11
    assert result["confirmatory_precision_ready"] == (result["observed_half_width"] <= .1)
    with pytest.raises(ValueError, match="complete candidate editability panel"):
        _edit_precision_summary(rows, None, 2026, 100)
    with pytest.raises(ValueError, match="missing planned candidate worlds"):
        _edit_precision_summary(rows[:-len(VIEWS)], original[:-len(VIEWS)], 2026, 100)
    with pytest.raises(ValueError, match="no silent world loss"):
        _edit_precision_summary(rows[:-len(VIEWS)], original, 2026, 100)
    with pytest.raises(ValueError, match="complete eight-view"):
        _edit_precision_summary([row for row in rows if row["view"] == "direct"],
                                [row for row in original if row["view"] == "direct"], 2026, 100)


def test_precision_counts_eligible_worlds_instead_of_120_candidates_and_keeps_denominators():
    plan = edit_precision_plan(120, stage="confirmatory")
    original = candidate_rows(plan, statuses=lambda seed, world: "editable" if world < 96 else "belief_not_stated")
    rows = edited_rows(original)
    result = _edit_precision_summary(rows, original, 2026, 1000)
    metric = result["metrics"]["changed_target_compliance"]
    assert metric["stories"] == 96 and metric["confirmatory_precision_ready"] is False
    assert result["editability"]["candidate_worlds"] == 120
    assert result["editability"]["editable_worlds"] == 96
    assert result["editability"]["categories"]["belief_not_stated"]["artifacts"] == 24
    with pytest.raises(ValueError, match="explicitly editable"):
        _edit_precision_summary([dict(row, edit_status="contradicted") for row in rows], original, 2026, 100)


def test_different_seed_eligibility_gets_actual_checkpoint_intervals_without_union_inflation():
    plan = edit_precision_plan(120, stage="confirmatory")
    original = candidate_rows(plan, seeds=(1, 2), statuses=lambda seed, world:
        "editable" if (world < 96 if seed == 1 else world >= 24) else "contradicted")
    result = _edit_precision_summary(edited_rows(original), original, 2026, 1000)
    assert result["metrics"]["changed_target_compliance"]["stories"] == 120
    assert result["metrics"]["changed_target_compliance"]["confirmatory_precision_ready"] is False
    assert len(result["by_source_checkpoint"]) == 2
    for subgroup in result["by_source_checkpoint"].values():
        assert subgroup["editability"]["candidate_worlds"] == 120
        assert subgroup["metrics"]["changed_target_compliance"]["stories"] == 96
        assert not subgroup["metrics"]["changed_target_compliance"]["confirmatory_precision_ready"]


def test_editability_report_counts_artifacts_once_and_keeps_all_failure_categories():
    plan = edit_precision_plan(120, stage="confirmatory")
    kinds = ("editable", "belief_not_stated", "contradicted", "ambiguous", "no_safe_control", "generation_failed")
    original = candidate_rows(plan, statuses=lambda seed, world: kinds[world % len(kinds)])
    cells = {(reader, "trained", "original", "direct_fixed"):
             [dict(row, reader=reader) for row in original] for reader in ("base", "trained")}
    report = _editability_summary(cells)
    subgroup = next(iter(report["by_source_checkpoint"].values()))
    assert subgroup["candidate_artifacts"] == subgroup["candidate_worlds"] == 120
    assert subgroup["editable_worlds"] == 20 and not subgroup["eligible_count_sufficient"]
    assert all(subgroup["categories"][kind]["artifacts"] == 20 for kind in kinds)
    assert subgroup["editable_fraction"] == 1 / 6
    cells[("base", "trained", "original", "direct_fixed")][0]["edit_status"] = "editable" if original[0]["edit_status"] != "editable" else "contradicted"
    with pytest.raises(ValueError, match="conflicting editability"):
        _editability_summary(cells)


def test_wrong_stated_belief_defines_changed_and_intact_using_its_oracle_before_forecast():
    plan = edit_precision_plan(120)
    original = candidate_rows(plan, n=12)
    # Changing a wrong source belief B to C leaves this question's source
    # consequence B intact, although the ordinary correct target is A.
    for row in original:
        row.update(expected_before_edit_answer="B", expected_edited_answer="B", prediction="B")
    rows = edited_rows(original)
    cells = {("reader", "trained", "original", "direct_fixed"): original,
             ("reader", "trained", "semantic_minimal", "direct_fixed"): rows}
    control = _trace_control_summary(cells, 2026, 100)["reader|trained|semantic_minimal|direct_fixed"]
    assert control["changed_target_n"] == 0
    assert control["intact_target_n"] == 12 * len(VIEWS)
    assert control["intact_target_preservation"] == 1
    assert control["ordinary_accuracy"] == 0
    assert control["story_precision"]["metrics"]["changed_target_compliance"]["stories"] == 0


def test_paired_source_comparison_reports_explicit_common_support_and_paired_interval():
    plan = edit_precision_plan(120)
    left = candidate_rows(plan, n=12, statuses=lambda seed, world: "editable" if world < 10 else "ambiguous")
    right = [dict(row, source="other", source_checkpoint={"id": "other-checkpoint"},
                  artifact_id="other-" + row["world_id"], edit_status="editable" if int(row["world_id"]) >= 2 else "contradicted")
             for row in left]
    cells = {("reader", "trained", "original", "direct_fixed"): left,
             ("reader", "other", "original", "direct_fixed"): right,
             ("reader", "trained", "semantic_minimal", "direct_fixed"): edited_rows(left),
             ("reader", "other", "semantic_minimal", "direct_fixed"):
                 [dict(row, prediction="B") for row in edited_rows(right)]}
    support = _editability_summary(cells)["paired_source_support"][0]
    assert support["left_editable_worlds"] == support["right_editable_worlds"] == 10
    assert support["common_editable_worlds"] == 8 and not support["identical_editable_support"]
    contrast = _paired_edit_source_comparisons(cells, 2026, 1000)[0]
    assert contrast["status"] == "estimated_on_common_support"
    assert contrast["common_worlds"] == 8 and contrast["common_records"] == 8 * len(VIEWS)
    assert abs(contrast["metrics"]["all_predicted_consequences"]["estimate"]) == .5
    assert contrast["metrics"]["all_predicted_consequences"]["ci95"] is not None
    assert contrast["confirmatory_precision_ready"] is False


def test_crossed_analysis_accepts_seed_specific_editable_sets_and_exposes_them():
    plan = edit_precision_plan(120)
    original = candidate_rows(plan, n=12, seeds=(1, 2), statuses=lambda seed, world:
        "editable" if world < (8 if seed == 1 else 10) else "belief_not_stated")
    baseline = [dict(row, condition="none") for row in original]
    result = crossed_summary(baseline, original + edited_rows(original), samples=100)
    assert result["editability"]["status"] == "reported"
    assert len(result["editability"]["by_source_checkpoint"]) == 2
    assert result["controls"]["reader|trained|semantic_minimal|direct_fixed"]["story_precision"]["editability"]["candidate_worlds"] == 12



def test_human_corrections_do_not_erase_machine_errors_or_noneditable_outcomes():
    plan = edit_precision_plan(120)
    original = candidate_rows(plan, n=12)
    for row in original:
        changed = int(row["world_id"]) < 3
        row.update(edit_initial_status="generation_failed" if changed else "editable",
                   edit_human_corrected=changed)
    report = _editability_summary({("reader", "trained", "original", "direct_fixed"): original})
    counts = next(iter(report["by_source_checkpoint"].values()))
    assert counts["editable_artifacts"] == 12
    assert counts["initial_categories"]["generation_failed"]["artifacts"] == 3
    assert counts["categories"]["generation_failed"]["artifacts"] == 0
    assert counts["human_corrected_artifacts"] == 3


def test_all_noneditable_checkpoint_is_reported_without_fabricating_edit_effect():
    plan = edit_precision_plan(120, stage="confirmatory")
    original = candidate_rows(plan, statuses=lambda seed, world: "belief_not_stated")
    result = crossed_summary([dict(row, condition="none") for row in original], original, samples=100)
    counts = next(iter(result["editability"]["by_source_checkpoint"].values()))
    assert counts["candidate_worlds"] == 120 and counts["editable_worlds"] == 0
    assert counts["conditional_effect_status"] == "not_estimable_no_editable_worlds"
    assert result["controls"] == {} and result["paired_edit_source_comparisons"] == []
    precision = _edit_precision_summary([], original, 2026, 100)
    assert precision["metrics"]["changed_target_compliance"]["estimate"] is None
    assert precision["metrics"]["changed_target_compliance"]["confirmatory_precision_ready"] is False


def test_missing_editability_status_cannot_hide_an_entire_candidate_panel():
    plan = edit_precision_plan(120, stage="confirmatory")
    original = candidate_rows(plan)
    for row in original:
        row.pop("edit_status")
    with pytest.raises(ValueError, match="missing editability status"):
        _editability_summary({("reader", "trained", "original", "direct_fixed"): original})


def test_basic_preaudit_candidates_remain_visible_as_not_assessed():
    plan = edit_precision_plan(120)
    original = candidate_rows(plan, n=12, statuses=lambda seed, world: "not_assessed")
    report = crossed_summary([dict(row, condition="none") for row in original], original, samples=100)
    counts = next(iter(report["editability"]["by_source_checkpoint"].values()))
    assert counts["categories"]["not_assessed"]["artifacts"] == 12
    assert counts["editable_worlds"] == 0
    assert counts["conditional_effect_status"] == "awaiting_editability_assessment"
    assert report["controls"] == {}


def test_correctness_counts_are_separate_from_whether_the_stated_belief_is_editable():
    plan = edit_precision_plan(120)
    original = candidate_rows(plan, n=12)
    for row in original:
        row["edit_source_belief_correct"] = int(row["world_id"]) < 4
    report = _editability_summary({("reader", "trained", "original", "direct_fixed"): original})
    counts = next(iter(report["by_source_checkpoint"].values()))
    assert counts["editable_worlds"] == 12
    assert counts["editable_source_belief_correctness"] == {"correct": 4, "incorrect": 8, "unknown": 0}


@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize("kind", ["move", "tell"])
def test_oracle_populates_semantic_edit_consequences_and_rejects_tampering(depth, kind):
    world = generate_world(random.Random(20), depth=depth, event_kind=kind)
    row = render_record(world, "direct")
    semantic = symbolic_edit_prediction(row)
    assert semantic["oracle"]["before"] == ordinary_target(world, "direct")
    assert semantic["oracle"]["before"] != semantic["oracle"]["after"]
    assert semantic["expected_answers"] == {view: answer_label(rewrite_forecast(
        world, view, row["scope_edit_location"], "belief_edit")) for view in VIEWS}
    _validate_oracle_edit(row, {"semantic": semantic})
    wrong = copy.deepcopy(semantic)
    wrong["expected_answers"]["world"] = "invalid"
    with pytest.raises(ValueError, match="oracle consequences"):
        _validate_oracle_edit(row, {"semantic": wrong})
