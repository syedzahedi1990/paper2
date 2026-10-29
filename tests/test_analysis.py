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


def test_unrewarded_scope_does_not_include_the_rewarded_direct_answer():
    before = [row('w', 'B', target_scope='belief_dependent'),
              row('w', 'A', view='search', target_scope='belief_dependent')]
    after = [{**before[0], 'prediction': 'A'}, {**before[1], 'prediction': 'B'}]
    result = transfer(before, after)['effects']
    assert result['scope:belief_dependent']['estimate'] == 0
    assert result['unrewarded_scope:belief_dependent']['estimate'] == -1


def test_seed_effects_are_averaged_inside_story_not_bootstrapped_as_extra_stories():
    before = [row('one', 'B', view='search'), row('two', 'B', view='search')]
    after = [{**r, 'training_seed': seed, 'prediction': 'A' if seed == 1 else 'B'}
             for seed in (1, 2, 3) for r in before]
    estimate = transfer(before, after)['effects']['unrewarded']
    assert estimate['stories'] == 2
    assert estimate['estimate'] == pytest.approx(1 / 3)
    assert estimate['ci95'] == pytest.approx([1 / 3, 1 / 3])
    with pytest.raises(ValueError, match='Every training seed'):
        transfer(before, after[:-1])


def test_joint_accuracy_reports_story_interval_with_seeds_averaged_inside_story():
    rows = [row(str(world), "A" if world == 0 or seed == 1 else "B", view=view, training_seed=seed)
            for world in range(2) for seed in (1, 2) for view in VIEWS]
    result = accuracy_summary(rows)
    assert result["bootstrap_samples"] == 20_000
    assert result["joint_all_eight"] == .75
    assert result["joint_all_eight_stories"] == 2
    assert result["joint_all_eight_ci95"] == [.5, 1.]
    assert result["joint_complete_stories"] == 4


def test_preservation_and_repair_report_paired_story_ratio_intervals():
    base = [row(str(world), prediction, view=view) for world in range(2)
            for prediction, view in (("A", "direct"), ("B", "search"))]
    trained = [dict(item, training_seed=seed, prediction="A" if item["world_id"] == "0" or seed == 1 else "B")
               for seed in (1, 2) for item in base]
    result = transfer(base, trained)
    assert result["bootstrap_samples"] == 20_000
    for name in ("preservation", "error_repair"):
        assert result[name]["estimate"] == .75
        assert result[name]["ci95"] == [.5, 1.]
        assert result[name]["stories"] == 2
        assert result[name]["valid_bootstrap_draws"] == 20_000
    ceiling = transfer([row("one", "A"), row("two", "A")], [row("one", "A"), row("two", "A")])
    assert ceiling["error_repair"]["estimate"] is None and ceiling["error_repair"]["ci95"] is None


def test_cue_contrast_separates_direct_from_oppositely_affected_consequences():
    rows = [row('w', 'A', cue_condition='agree'), row('w', 'B', cue_condition='conflict'),
            row('w', 'B', view='world', cue_condition='agree'),
            row('w', 'A', view='world', cue_condition='conflict')]
    result = cue_summary(rows)
    assert result['direct_agree_minus_conflict']['estimate'] == 1
    assert result['per_view_agree_minus_conflict']['world']['estimate'] == -1
    assert result['pooled_direct_anchored_cue_effect']['estimate'] == 0


def test_crossed_controls_can_use_prespecified_subset_but_original_cannot():
    baseline = [row('one', 'B'), row('two', 'B')]
    original = [{**r, 'source': 'base', 'condition': 'original'} for r in baseline]
    edited = row('one', 'A', source='base', condition='semantic_minimal', expected_edited_answer='A')
    summary = crossed_summary(baseline, original + [edited])
    assert summary['cells']['base|base|semantic_minimal|question_dependent']['baseline_errors'] == 1
    assert summary['controls']['base|base|semantic_minimal|question_dependent']['predicted_consequence_rate'] == 1
    with pytest.raises(ValueError, match='exactly the same'):
        crossed_summary(baseline, original[:1])


def test_crossed_analysis_rejects_different_reader_trace_text():
    baseline = [row('w', 'B'), row('w', 'B', reader='trained')]
    crossed = [row('w', 'A', reader=reader, source='base', condition='original',
                   trace_sha256=reader) for reader in ('base', 'trained')]
    with pytest.raises(ValueError, match='different trace text'):
        crossed_summary(baseline, crossed)


def test_rerendered_replay_reports_failures_separately_from_raw_identity():
    baseline = [row('w', 'B')]
    crossed = [row('w', 'B', source='base', condition='original'),
               row('w', 'B', source='base', condition='own_replay', replay_eligible=True,
                   replay_matches=True, replay_token_ids_match=False),
               row('w', 'A', source='base', condition='own_rerendered', replay_eligible=True,
                   replay_matches=False, replay_token_ids_match=False)]
    controls = crossed_summary(baseline, crossed)['controls']
    assert controls['base|base|own_replay|question_dependent']['match_rate'] == 1
    assert controls['base|base|own_rerendered|question_dependent']['match_rate'] == 0
    assert controls['base|base|own_rerendered|question_dependent']['token_prefix_equal'] == 0


def test_matching_preserves_seeds_and_requires_saved_measured_weights(tmp_path):
    from paper2.analysis import match_checkpoints
    from paper2.io import file_digest
    adapter = tmp_path / 'checkpoint'
    adapter.mkdir()
    (adapter / 'adapter_config.json').write_text('{}')
    weights = adapter / 'adapter_model.safetensors'
    weights.write_bytes(b'fixture')
    common = dict(arm='reasoning_rl', dataset='clean', step=25, split='development',
                  view='direct', checkpoint_path=str(adapter), adapter_sha256=file_digest(weights))
    records = [{**common, 'seed': seed, 'format': 'immediate', 'accuracy': score}
               for seed, score in ((1, .5), (2, .51), (3, .9))]
    records.append({**common, 'seed': 1, 'format': 'native_reasoning', 'accuracy': .7})
    result = match_checkpoints(records)
    assert len(result['matches']) == 3 and result['target_accuracy'] == .5
    assert result['matches']['reasoning_rl|clean|seed=3']['absolute_accuracy_gap'] == .4
    assert result['matches']['reasoning_rl|clean|seed=1']['checkpoint']['step'] == 25
    weights.write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed'):
        match_checkpoints(records)


def test_crossed_seed_pool_reuses_base_baseline_without_triple_counting_repair_rate():
    baseline = [row('w', 'B')]
    baseline.extend(row('w', 'B', reader='trained', training_seed=seed) for seed in (1, 2, 3))
    crossed = [row('w', 'A' if seed == 1 else 'B', reader=reader, source=source,
                   condition='original', comparison_seed=seed)
               for seed in (1, 2, 3) for reader in ('base', 'trained') for source in ('base', 'trained')]
    result = crossed_summary(baseline, crossed)
    for cell in result['cells'].values():
        assert cell['fraction_errors_fixed'] == pytest.approx(1 / 3)
    assert result['original_2x2']['question_dependent']['contrasts']['interaction']['estimate'] == 0


def test_crossed_analysis_rejects_tampered_text_and_stale_source_hashes():
    from paper2.io import digest
    baseline = [row('w', 'B')]
    observation = row('w', 'A', source='base', condition='original', supplied_trace='changed',
                      trace_sha256=digest('original'), source_trace_sha256=digest('original'))
    with pytest.raises(ValueError, match='recorded hash'):
        crossed_summary(baseline, [observation])
    observation['trace_sha256'] = digest('changed')
    with pytest.raises(ValueError, match='frozen source'):
        crossed_summary(baseline, [observation])
