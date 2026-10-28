"""The edit subset and symbolic controls do not depend on an ML installation."""
import copy
import random

import pytest

from paper2.backend import Answer
from paper2.data.world import generate_world, render_record, ordinary_target, answer_label, VIEWS
from paper2.io import digest
from paper2.traces import (prepare_edit_manifest, run_crossed_reader, symbolic_belief_summary,
                           _contracts, validate_edit)


def trace_record(world, view="direct", source="base"):
    row = render_record(world, view)
    text = "Alice believes box and waits."
    basis_id = render_record(world, "direct")["id"]
    checkpoint = {"id": source, "revision": None}
    artifact = digest([source, checkpoint, basis_id, "direct_fixed", digest(text)])
    return row, {"schema_version": 2, "trace_id": digest([artifact, row["id"]]), "artifact_id": artifact,
                 "record_id": row["id"], "basis_record_id": render_record(world, "direct")["id"],
                 "world_id": world.world_id, "view": view, "source": source,
                 "question_timing": "direct_fixed", "trace": text, "trace_sha256": digest(text),
                 "source_checkpoint": {"id": source, "revision": None},
                 "replay_eligible": False, "forced_delimiter": True}


def test_edit_subset_is_deterministic_source_paired_and_deduplicated():
    rng = random.Random(44)
    worlds = [generate_world(rng) for _ in range(20)]
    pairs = [trace_record(world, view, source) for world in worlds for source in ("base", "trained")
             for view in VIEWS]
    rows, traces = zip(*pairs)
    manifest = prepare_edit_manifest(traces, rows, max_worlds=3, seed=17)
    shuffled = list(reversed(traces))
    again = prepare_edit_manifest(shuffled, reversed(rows), max_worlds=3, seed=17)
    assert manifest == again
    assert len(manifest["edits"]) == 6  # Three worlds x two sources; eight views share one review.
    assert len(manifest["selection"]["selected_world_ids"]) == 3
    assert len(manifest["selection"]["universe_world_ids"]) == 20
    assert all(len(edit["trace_ids"]) == 8 and len(edit["record_ids"]) == 8 for edit in manifest["edits"])
    assert all(edit["belief_summary"]["task_grounded"] for edit in manifest["edits"])
    assert all(not edit["approved"] and edit["reviewed_by"] is None for edit in manifest["edits"])
    _contracts(manifest)
    changed = copy.deepcopy(manifest)
    changed["selection"]["selected_world_ids"].pop()
    changed["selection_sha256"] = digest(changed["selection"])
    with pytest.raises(ValueError, match="preregistered deterministic"):
        _contracts(changed)


@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize("kind", ["move", "tell"])
def test_symbolic_control_is_fixed_and_predicts_each_ordinary_consequence(depth, kind):
    world = generate_world(random.Random(123), depth=depth, event_kind=kind)
    rows = [render_record(world, view) for view in VIEWS]
    summaries = [symbolic_belief_summary(row) for row in rows]
    assert all(summary == summaries[0] for summary in summaries)
    assert "Answer:" not in summaries[0]["text"]
    assert "choice" not in summaries[0]["text"]
    assert summaries[0]["expected_answers"] == {
        view: answer_label(ordinary_target(world, view)) for view in VIEWS}


def test_manual_controls_require_human_review_and_fail_closed_for_selected_world():
    row, trace = trace_record(generate_world(random.Random(3)))
    manifest = prepare_edit_manifest([trace], [row], max_worlds=1)
    with pytest.raises(ValueError, match="reviewed edit"):
        validate_edit(trace, manifest["edits"][0])
    manifest["edits"].clear()
    with pytest.raises(ValueError, match="Missing edit contract"):
        list(run_crossed_reader(None, [row], [trace], "trained", manifest, conditions=("semantic_minimal",)))


class SummaryReader:
    def token_ids(self, text):
        return list(text.encode())

    def answer(self, prompt, choices, prefix_ids=None):
        return Answer("A", 1, " A", "A", {})

    class tokenizer:
        @staticmethod
        def apply_chat_template(messages, **kwargs):
            return "\n".join(message["content"] for message in messages)


def test_full_world_baselines_remain_available_with_manual_subset():
    worlds = [generate_world(random.Random(seed)) for seed in range(5)]
    rows, traces = zip(*(trace_record(world) for world in worlds))
    manifest = prepare_edit_manifest(traces, rows, max_worlds=1)
    # Automatic symbolic controls do not require human reviews or selected-world filtering.
    output = list(run_crossed_reader(SummaryReader(), rows, traces, "trained", manifest,
                                     conditions=("original", "none", "belief_summary")))
    assert len(output) == 15
    assert len({row["world_id"] for row in output}) == 5
    assert sum(row["edit_subset_selected"] for row in output) == 3
    assert all(row["expected_summary_answer"] == row["answer"] for row in output
               if row["condition"] == "belief_summary")


def test_original_reader_rejects_stale_trace_hash_even_without_any_edits():
    row, trace = trace_record(generate_world(random.Random(18)))
    trace["trace"] += " An unreviewed change."
    with pytest.raises(ValueError, match="trace text hash changed"):
        list(run_crossed_reader(SummaryReader(), [row], [trace], "trained", conditions=("original",)))


def test_recomputed_text_hash_does_not_silently_change_frozen_artifact_identity():
    row, trace = trace_record(generate_world(random.Random(18)))
    trace["trace"] += " An unreviewed change."
    trace["trace_sha256"] = digest(trace["trace"])
    with pytest.raises(ValueError, match="artifact identity changed"):
        list(run_crossed_reader(SummaryReader(), [row], [trace], "trained", conditions=("original",)))
