import json
import random

import pytest

from paper2.data import (AGENTS, OBJECTS, LOCATIONS, VIEWS, CONFIG, StoryWorld, generate_world,
                         render_record, ordinary_target, matched_pair, rewrite_forecast,
                         validate_cue_renderings, build_oracle_trace)
from paper2.data._vendor.tom_epistemic_generator import Event, EpistemicState


def test_original_state_engine_tracks_second_order_observers_and_tells():
    assert len(OBJECTS) == 20 and len(LOCATIONS) == 6 and len(AGENTS) == 2
    state = EpistemicState.initial(CONFIG, {obj: "box" for obj in OBJECTS})
    state = state.apply(Event("move", "key", "basket", frozenset({"Alice"})))
    assert state.get((), "key") == state.get(("Alice",), "key") == "basket"
    assert state.get(("Alice", "Bob"), "key") == "box"
    state = state.apply(Event("tell", "key", "shelf", frozenset(AGENTS)))
    assert state.get((), "key") == "basket"
    assert state.get(("Alice", "Bob"), "key") == state.get(("Bob", "Alice"), "key") == "shelf"


@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize("kind", ["move", "tell"])
def test_eight_ordinary_targets_and_single_event_rewrite(depth, kind):
    world = generate_world(random.Random(37), depth=depth, event_kind=kind, distractor_events=0)
    assert StoryWorld.from_dict(json.loads(json.dumps(world.to_dict()))) == world
    pair = matched_pair(world)
    recipient, donor = pair["recipient_world"], pair["donor_world"]
    differences = [i for i, events in enumerate(zip(recipient["events"], donor["events"])) if events[0] != events[1]]
    assert differences == [world.critical_event_index]
    assert {key for key in recipient["events"][0] if recipient["events"][0][key] != donor["events"][0][key]} == {"location"}
    assert tuple(row["view"] for row in pair["recipient"]) == VIEWS
    for view, row, donor_row in zip(VIEWS, pair["recipient"], pair["donor"]):
        assert row["choices"][row["answer"]] == ordinary_target(world, view)
        assert pair["forecasts"][view]["event_rewrite"] == donor_row["answer"]
        start, stop = row["critical_value_span"]
        assert row["story"][start:stop] == world.critical_event.location
        assert row["story"][slice(*row["critical_event_span"])].startswith(world.agent if depth == 1 else "Alice and Bob")
    assert ordinary_target(world, "observed_overwrite") == world.overwrite_location
    assert ordinary_target(world, "unobserved_overwrite") == world.critical_event.location
    assert ordinary_target(world, "report") == world.critical_event.location
    assert ordinary_target(world, "irrelevant_object") == world.initial_locations[world.distractor_object]
    if kind == "tell":
        assert ordinary_target(world, "world") == world.initial_locations[world.focal_object]
        assert not pair["event_rewrite_change_mask"]["world"]


def test_paper_one_depth_one_account_patterns():
    world = generate_world(random.Random(1), depth=1, event_kind="move", distractor_events=0)
    donor = world.donor_location
    changed = {view for view in VIEWS if rewrite_forecast(world, view, donor, "belief_edit") != ordinary_target(world, view)}
    assert changed == {"direct", "search", "unobserved_overwrite", "report"}
    changed = {view for view in VIEWS if rewrite_forecast(world, view, donor, "event_rewrite") != ordinary_target(world, view)}
    assert changed == {"direct", "world", "search", "unobserved_overwrite", "report"}


def test_cue_twins_match_exactly_and_independent_is_fully_crossed():
    world = generate_world(random.Random(2))
    for view in VIEWS:
        target = ordinary_target(world, view)
        direct_target = ordinary_target(world, "direct")
        rows = [render_record(world, view, "agree")]
        rows += [render_record(world, view, "independent", cue_location=loc) for loc in LOCATIONS]
        rows += [render_record(world, view, "conflict", cue_location=loc) for loc in LOCATIONS if loc != direct_target]
        assert validate_cue_renderings(rows)["matched"]
        assert len({row["id"] for row in rows}) == 12
        assert sum(row["cue_agrees"] for row in rows if row["cue_condition"] == "independent") == 1
        assert all(row["target_location"] == target for row in rows)


def test_cue_condition_is_fixed_to_direct_belief_across_all_eight_views():
    world = generate_world(random.Random(22), depth=2)
    rows = [render_record(world, view, "agree") for view in VIEWS]
    assert len({row["cue_location"] for row in rows}) == 1
    assert len({row["base_story"] for row in rows}) == 1
    assert all(row["cue_agrees"] and row["cue_agrees_direct"] for row in rows)
    overwritten = next(row for row in rows if row["view"] == "observed_overwrite")
    assert not overwritten["cue_agrees_evaluated"]
    # A cue conflicting with the rewarded answer can agree with an unrewarded
    # consequence. Its condition must never change when that question is asked.
    conflict = render_record(world, "observed_overwrite", "conflict", cue_location=world.overwrite_location)
    assert conflict["cue_agrees_evaluated"] and not conflict["cue_agrees_direct"]


def test_prequestion_base_story_omits_later_consequence_events():
    world = generate_world(random.Random(3), depth=2)
    direct = render_record(world, "direct", "independent", cue_location="shelf")
    report = render_record(world, "report", "independent", cue_location="shelf")
    assert direct["story"] == report["base_story"]
    assert "accepts the report" not in report["base_story"]
    assert "accepts the report" in report["story"]


def test_hidden_move_and_symbolic_reasoning_source():
    for depth in (1, 2):
        world = generate_world(random.Random(10), depth=depth, hidden_move=True)
        assert ordinary_target(world, "world") != ordinary_target(world, "direct")
        row = render_record(world, "direct")
        trace = build_oracle_trace(row)
        assert f"choice {row['answer']}" in trace
        assert "belief remains" in trace
    with pytest.raises(ValueError):
        build_oracle_trace(render_record(world, "world"))
