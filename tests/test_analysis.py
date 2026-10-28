import pytest

from paper2.analysis import accuracy_summary, crossed_summary, cue_summary, trace_benefit, transfer
from paper2.data import LOCATIONS, VIEWS


def row(world, prediction, reader="base", view="direct", **extra):
    return {"world_id": world, "view": view, "cue_condition": "none", "cue_location": None,
            "answer": "A", "prediction": prediction, "reader": reader, **extra}


def test_crossed_benefit_uses_each_readers_remaining_errors_and_reports_regressions():
    baseline = [row("1", "B"), row("2", "B"), row("3", "A"), row("4", "A"),
                row("1", "B", "trained"), row("2", "A", "trained"),
                row("3", "A", "trained"), row("4", "A", "trained")]
    crossed = [row(str(i), p, reader, source="trained", condition="original")
               for reader in ("base", "trained") for i, p in enumerate(("A", "B", "B", "A"), 1)]
    summary = crossed_summary(baseline, crossed)["cells"]
    base, trained = summary["base|trained|original|question_dependent"], summary["trained|trained|original|question_dependent"]
    assert base["fraction_errors_fixed"] == .5 and trained["fraction_errors_fixed"] == 1
    assert base["correct_broken"] == 1 and trained["correct_broken"] == 2
    assert base["net_accuracy_gain"] == 0 and trained["net_accuracy_gain"] == -.25


def test_ceiling_is_undefined_and_mismatched_questions_are_rejected():
    result = trace_benefit([row("1", "A")], [row("1", "B")])
    assert result["ceiling"] and result["fraction_errors_fixed"] is None
    assert result["regression_rate"] == 1
    with pytest.raises(ValueError, match="exactly the same"):
        trace_benefit([row("1", "A")], [row("2", "A")])


def test_unrewarded_transfer_excludes_direct_and_preservation_counts_losses():
    baseline = [row("1", "B"), row("1", "B", view="search"), row("1", "A", view="world")]
    trained = [row("1", "A"), row("1", "A", view="search"), row("1", "B", view="world")]
    result = transfer(baseline, trained)
    assert result["effects"]["direct"]["estimate"] == 1
    assert result["effects"]["unrewarded"]["estimate"] == 0
    assert result["preservation"]["rate"] == 0


def test_invalid_answer_is_never_counted_as_following_an_absent_cue():
    result = cue_summary([row("1", None, cue_answer=None)])
    assert result["none"]["direct_cue_following_rate"] is None
    assert result["none"]["cue_present_direct_n"] == 0


def test_original_two_by_two_reports_source_reader_effects_and_interaction():
    baseline = [row("1", "B"), row("2", "B"), row("3", "A"),
                row("1", "B", "trained"), row("2", "A", "trained"), row("3", "A", "trained")]
    crossed = []
    predictions = {("base", "base"): ("B", "B", "A"),
                   ("base", "trained"): ("A", "B", "A"),
                   ("trained", "base"): ("B", "A", "A"),
                   ("trained", "trained"): ("A", "A", "A")}
    for (reader, source), values in predictions.items():
        crossed.extend(row(str(i), value, reader, source=source, condition="original")
                       for i, value in enumerate(values, 1))
    result = crossed_summary(baseline, crossed, samples=200)["original_2x2"]["question_dependent"]
    effects = result["contrasts"]
    assert effects["trace_source_effect_for_base_reader"]["estimate"] == .5
    assert effects["trace_source_effect_for_trained_reader"]["estimate"] == 1
    assert effects["reader_effect_for_base_trace"]["estimate"] == 0
    assert effects["reader_effect_for_trained_trace"]["estimate"] == .5
    assert effects["interaction"]["estimate"] == .5
    assert effects["interaction"]["valid_bootstrap_draws"] > 0


def test_controls_separate_replay_eligibility_from_targeted_edit_compliance():
    baseline = [row("1", "A")]
    crossed = [row("1", "A", source="base", condition="original"),
               row("1", "A", source="base", condition="own_replay", replay_eligible=False,
                   replay_matches=None, source_forced_delimiter=True),
               row("1", "B", source="base", condition="semantic_minimal", expected_edited_answer="B")]
    controls = crossed_summary(baseline, crossed)["controls"]
    replay = controls["base|base|own_replay|question_dependent"]
    assert replay["forced_delimiter"] == replay["ineligible"] == 1
    assert replay["checked"] == 0 and replay["match_rate"] is None
    edited = controls["base|base|semantic_minimal|question_dependent"]
    assert edited["predicted_consequence_rate"] == 1
    assert edited["ordinary_regression_rate"] == 1
    assert edited["changed_target_compliance"] == 1


def test_two_different_invalid_outputs_are_not_counted_as_trace_control_agreement():
    baseline = [row("1", "A")]
    crossed = [row("1", None, source="base", condition="original", token_id=20),
               row("1", None, source="base", condition="irrelevant", token_id=21)]
    summary = crossed_summary(baseline, crossed)["controls"]["base|base|irrelevant|question_dependent"]
    assert summary["agreement_with_original"] == 0


def test_cue_summary_weights_worlds_equally_when_rendering_counts_differ():
    rows = [row("1", "A" if i < 5 else "B", cue_condition="independent", cue_location=str(i), cue_answer="A")
            for i in range(6)]
    rows.append(row("2", "B", cue_condition="independent", cue_location="0", cue_answer="A"))
    result = cue_summary(rows)["independent"]
    assert result["micro_accuracy"] == pytest.approx(5 / 7)
    assert result["accuracy"] == pytest.approx(5 / 12)
    assert result["direct_cue_following_rate"] == pytest.approx(5 / 12)


def test_agree_joint_holds_direct_belief_cue_fixed_across_all_consequence_targets():
    rows = []
    for index, view in enumerate(VIEWS):
        answer = "ABCDEF"[index % 6]
        rows.append(row("1", answer, view=view, answer=answer, cue_condition="agree",
                        cue_location=LOCATIONS[0], cue_answer="A"))
    result = accuracy_summary(rows)
    assert result["joint_all_eight"] == 1 and result["joint_complete_stories"] == 1
    assert "literal cue location" in result["joint_alignment"]
    rows[-1]["prediction"] = None
    assert accuracy_summary(rows)["joint_all_eight"] == 0


def test_conflict_joint_uses_five_fixed_cues_excluding_direct_target_for_every_view():
    rows = []
    for index, view in enumerate(VIEWS):
        answer = "ABCDEF"[index % 6]
        alternatives = list(LOCATIONS[1:])
        for rank, location in enumerate(alternatives):
            rows.append(row("1", None if view == "report" and rank == 2 else answer, view=view,
                            answer=answer, cue_condition="conflict", cue_location=location,
                            cue_answer="ABCDEF"[LOCATIONS.index(location)]))
    summary = cue_summary(rows)["conflict"]
    assert summary["joint_complete_stories"] == 5
    assert summary["joint_all_eight"] == .8
    incomplete = accuracy_summary(rows[:-1])
    assert incomplete["joint_complete_stories"] == 4
    assert incomplete["joint_incomplete_stories"] == 1
