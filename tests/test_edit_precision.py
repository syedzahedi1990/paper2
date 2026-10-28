import copy
import random

import pytest

from paper2.analysis import _edit_precision_summary
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
    assert plan["required_worlds"] == plan["selected_worlds"] == 97
    assert plan["planned_half_width"] <= .10
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


def test_confirmatory_report_checks_actual_precision_and_missing_worlds():
    plan = edit_precision_plan(120, stage="confirmatory")
    rows = observations(97, plan)
    result = _edit_precision_summary(rows, None, 2026, 3000)["metrics"]["changed_target_compliance"]
    assert result["stories"] == 97 and result["observed_half_width"] < .11
    assert result["confirmatory_precision_ready"] == (result["observed_half_width"] <= .1)
    with pytest.raises(ValueError, match="missing planned worlds"):
        _edit_precision_summary(rows[:-len(VIEWS)], None, 2026, 100)
    with pytest.raises(ValueError, match="complete eight-view panels"):
        _edit_precision_summary([row for row in rows if row["view"] == "direct"], None, 2026, 100)


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
