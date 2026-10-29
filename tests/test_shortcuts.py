from dataclasses import replace
import json
import random

import pytest

from paper2.data.shortcuts import relation_quota

from paper2.data import (CorpusConfig, LOCATIONS, assert_calibration_corpus, audit_shortcuts,
                         generate_corpus, generate_world, generate_world_splits,
                         ordinary_target, render_record, verify_corpus, heldout_clean_cues)


def test_default_stratified_shortcuts_and_both_nested_relation_directions():
    worlds = generate_world_splits(CorpusConfig())["train"]
    result = audit_shortcuts(worlds, cue_rows=[render_record(world, "direct", "clean") for world in worlds])
    assert result["integrity_passed"]
    assert "passed" not in result and "requires_heuristic_review" not in result
    assert set(result["strata"]) == {f"depth{depth}__{kind}" for depth in (1, 2) for kind in ("move", "tell")}
    for name, cell in result["strata"].items():
        assert cell["world_ids"] == sorted(world.world_id for world in worlds
                                          if name == f"depth{len(world.focal_word)}__{world.critical_event.kind}")
        metrics = cell["metrics"]
        assert {"first_event", "last_event", "initial_location", "clean_cue"} <= set(metrics)
        if name.startswith("depth2"):
            assert metrics["equal_both"]["reference_expectation"] == 1 / 36
            assert metrics["different_from_both"]["reference_expectation"] == 25 / 36
            for rule in ("equal_outer", "equal_inner", "exclude_outer", "exclude_inner", "exclude_both"):
                assert metrics[rule]["reference_expectation"] == 1 / 6
        residual = cell["residual_heuristics"]
        assert "flagged_rules" not in residual and "all_near_chance" not in residual
        for position in range(1, 10):
            assert residual["metrics"][f"event_position_{position}"]["n"] == cell["n"]
        for metric in [*metrics.values(), *residual["metrics"].values()]:
            assert metric["ci95"][0] <= metric["estimate"] <= metric["ci95"][1]
            assert not {"passed", "sufficient", "maximum_deviation", "requires_review"}.intersection(metric)
        if name.startswith("depth1"):
            assert "last_both_observed" in residual["metrics"]
        else:
            assert "last_both_observed" not in residual["metrics"]
            assert residual["semantic_baselines"]["last_both_observed"]["score"] == 1


@pytest.mark.parametrize("kind", ["move", "tell"])
@pytest.mark.parametrize("force_equal", [False, True])
@pytest.mark.parametrize("count", [30, 400])
def test_audit_reports_always_equal_and_always_different_nested_beliefs_without_score_gate(kind, force_equal, count):
    rng = random.Random(705)
    worlds = []
    for _ in range(count):
        world = generate_world(rng, depth=2, event_kind=kind, distractor_events=0)
        target = ordinary_target(world, "direct")
        location = target if force_equal else rng.choice([loc for loc in LOCATIONS if loc != target])
        events = list(world.events)
        # The final two events privately update one character each, while leaving
        # the queried nested belief intact. Force either failure mode explicitly.
        events[-2:] = [replace(event, location=location) for event in events[-2:]]
        worlds.append(replace(world, events=tuple(events)))
    result = audit_shortcuts(worlds)
    metric = "equal_both" if force_equal else "different_from_both"
    assert result["strata"][f"depth2__{kind}"]["metrics"][metric]["score"] == 1
    report = result["strata"][f"depth2__{kind}"]["metrics"][metric]
    assert report["ci95"][1] == 1 and "passed" not in report


@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize("kind", ["move", "tell"])
def test_audit_reports_perfect_first_event_lookup_without_numeric_screen(depth, kind):
    rng = random.Random(901)
    worlds = []
    for _ in range(80):
        world = generate_world(rng, depth=depth, event_kind=kind, distractor_events=0)
        events = list(world.events)
        events[0] = replace(events[0], location=ordinary_target(world, "direct"))
        worlds.append(replace(world, events=tuple(events)))
    result = audit_shortcuts(worlds)
    report = result["strata"][f"depth{depth}__{kind}"]["metrics"]["first_event"]
    assert report["score"] == 1 and report["ci95"][1] == 1
    assert "passed" not in report


def test_hidden_moves_do_not_restore_position_or_reverse_belief_shortcuts():
    worlds = generate_world_splits(CorpusConfig(train=2000, hidden_move=True))["train"]
    assert audit_shortcuts(worlds)["integrity_passed"]


def test_smoke_corpus_and_old_schema_cannot_pass_calibration(tmp_path):
    manifest = generate_corpus(tmp_path, CorpusConfig(train=10, development=1, confirmation=1))
    assert manifest["empirical_calibration_status"] == "requires_named_review"
    assert verify_corpus(tmp_path)["verified"]
    with pytest.raises(ValueError, match="quotas|coverage|story splits"):
        assert_calibration_corpus(tmp_path / "manifest.json")
    manifest["schema_version"] = "paper2.corpus.v2"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Obsolete corpus schema"):
        assert_calibration_corpus(tmp_path)


def test_missing_event_stratum_and_missing_cues_fail_required_coverage():
    rng = random.Random(171)
    worlds = [generate_world(rng, depth=1, event_kind="move") for _ in range(30)]
    result = audit_shortcuts(worlds, expected_depths=(1, 2), require_cues=True)
    assert not result["integrity_passed"]
    assert result["coverage"]["missing_strata"] == ["depth1__tell", "depth2__move", "depth2__tell"]
    assert result["coverage"]["worlds_missing_clean_cue"] == 30


@pytest.mark.parametrize("hidden_move", [False, True])
def test_joint_relations_use_declared_quotas_in_every_split_and_event_kind(hidden_move):
    splits = generate_world_splits(CorpusConfig(hidden_move=hidden_move))
    for split, worlds in splits.items():
        audit = audit_shortcuts(worlds, require_relation_quotas=True)
        assert audit["integrity_passed"]
        for key, cell in audit["strata"].items():
            if not key.startswith("depth2"):
                continue
            quota = cell["relation_quota"]
            assert quota["required"] and quota["passed"]
            assert quota["observed"] == quota["planned"] == relation_quota(cell["n"])
            if split != "train":
                assert quota["observed"] == {"both": 1, "outer_only": 4, "inner_only": 4, "neither": 21}
                assert "maximum_deviation" not in cell["metrics"]["different_from_both"]


def test_exact_quota_integrity_rejects_small_relation_imbalance():
    worlds = generate_world_splits(CorpusConfig())["development"]
    for index, world in enumerate(worlds):
        if len(world.focal_word) != 2 or world.critical_event.kind != "move":
            continue
        target = ordinary_target(world, "direct")
        if not all(world.state().get((agent,), world.focal_object) != target for agent in world.focal_word):
            continue
        events = list(world.events)
        last = max(i for i, event in enumerate(events) if event.object == world.focal_object and world.agent in event.agents)
        events[last] = replace(events[last], location=target)
        worlds[index] = replace(world, events=tuple(events))
        break
    result = audit_shortcuts(worlds, require_relation_quotas=True)
    assert not result["integrity_passed"]
    assert not result["strata"]["depth2__move"]["relation_quota"]["passed"]


def test_clean_cue_quota_rejects_incidental_heldout_correlation():
    worlds = generate_world_splits(CorpusConfig())["confirmation"]
    cues = heldout_clean_cues(worlds)
    rows = [render_record(world, "direct", "clean", cue_location=cues[world.world_id]) for world in worlds]
    audit = audit_shortcuts(worlds, cue_rows=rows, require_clean_cue_quota=True)
    assert audit["integrity_passed"]
    assert all(cell["clean_cue_quota"]["planned_correct"] == 5 for cell in audit["strata"].values())
    for row in rows:
        if row["belief_depth"] == 2 and row["critical_event_kind"] == "tell" and not row["cue_agrees_direct"]:
            row["cue_location"] = row["target_location"]
            break
    audit = audit_shortcuts(worlds, cue_rows=rows, require_clean_cue_quota=True)
    assert not audit["integrity_passed"]
    assert not audit["strata"]["depth2__tell"]["clean_cue_quota"]["passed"]


def test_frequency_ties_use_uniform_accuracy_and_missing_positions_use_declared_fallback():
    world = generate_world(random.Random(731), depth=1, event_kind="move", deciding_position=1,
                           distractor_events=0, target_location="box")
    locations = ("basket", "box", "box", "box", "box", "basket", "basket", "basket", "shelf")
    world = replace(world, events=tuple(replace(event, location=location)
                                      for event, location in zip(world.events, locations)))
    audit = audit_shortcuts([world])
    metric = audit["strata"]["depth1__move"]["residual_heuristics"]["metrics"]["most_mentioned_location"]
    assert metric["score"] == 0.5 and metric["tie_worlds"] == 1
    shorter = replace(world, events=world.events[:-1],
                      initial_locations={**world.initial_locations, world.focal_object: "box"})
    audit = audit_shortcuts([world, shorter])
    metric = audit["strata"]["depth1__move"]["residual_heuristics"]["metrics"]["event_position_9"]
    assert metric["n"] == 2 and metric["score"] == 0.5 and metric["initial_fallback_worlds"] == 1


def test_calibration_corpus_checks_integrity_not_heuristic_accuracy(monkeypatch):
    from paper2.data import corpus
    verified = {"integrity_passed": True, "empirical_calibration_status": "requires_named_review",
                "story_counts": {"train": 2000, "development": 120, "confirmation": 120},
                "shortcut_audit": {"train": {"audit_version": "paper2.shortcuts.v5"}}}
    monkeypatch.setattr(corpus, "verify_corpus", lambda _: verified)
    assert assert_calibration_corpus("any-path") == verified
    verified["integrity_passed"] = False
    with pytest.raises(ValueError, match="quotas|coverage"):
        assert_calibration_corpus("any-path")


def test_tiny_strata_still_report_intervals_without_an_invented_minimum_size():
    world = generate_world(random.Random(731), depth=1, event_kind="move")
    metric = audit_shortcuts([world])["strata"]["depth1__move"]["metrics"]["first_event"]
    assert metric["n"] == 1 and metric["ci95"] == [0., 1.]
    assert "sufficient" not in metric


def test_old_shortcut_audit_requires_new_manifest_even_when_story_schema_is_current(tmp_path):
    manifest = generate_corpus(tmp_path, CorpusConfig(train=10, development=1, confirmation=1))
    manifest["shortcut_audit"]["train"]["audit_version"] = "paper2.shortcuts.v3"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Obsolete shortcut audit"):
        verify_corpus(tmp_path)
