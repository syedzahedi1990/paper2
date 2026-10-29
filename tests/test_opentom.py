"""External evaluation options, ambiguity handling, and auditable benchmark rules."""
import copy
import json

import pytest

from paper2.opentom import (SCORING_PROTOCOL, evaluate_opentom, import_opentom, question_protocol,
                            score_opentom, summarize_opentom)


def record(kind="fine_location", answer="the basket", options=None):
    types = {"fine_location": "location-fo", "coarse_location": "location-fo", "fullness": "multihop-so",
             "accessibility": "multihop-so", "attitude": "attitude"}
    questions = {"fine_location": "Where does Alice locate the cup?", "coarse_location": "Is it in its initial location?",
                 "fullness": "How does the box's fullness change?", "accessibility": "How does accessibility change?",
                 "attitude": "What is your attitude towards Bob's action?"}
    row = {"id": kind, "story": "Alice observes Bob moving the cup.", "dataset": "opentom", "split": "external_test",
           "question_type": types[kind], "question": questions[kind], "answer_kind": kind, "answer_text": answer,
           "plot_info": {"original_place": "box", "move_to_place": "the basket"}, "scoring_protocol": SCORING_PROTOCOL}
    row["answer_options"] = options or question_protocol({"type": types[kind], "question": questions[kind]}, row["plot_info"])[1]
    return row


@pytest.mark.parametrize("kind,answer,prediction", [
    ("fine_location", "the basket", "basket"),
    ("fine_location", "the basket", "It is in the basket."),
    ("coarse_location", "Yes", "Yes."),
    ("coarse_location", "No", "No."),
    ("fullness", "less full", "emptier"),
    ("fullness", "less full", "more empty"),
    ("fullness", "more full", "fuller"),
    ("fullness", "equally full", "equally full"),
    ("accessibility", "less accessible", "less accessible"),
    ("attitude", "positive", "A"),
    ("attitude", "neutral", "B"),
    ("attitude", "negative", "Some reasoning.\n\nAnswer: C."),
])
def test_type_specific_scoring_preserves_upstream_valid_forms(kind, answer, prediction):
    result = score_opentom(prediction, record(kind, answer))
    assert result["correct"] and result["upstream_correct"]


@pytest.mark.parametrize("kind,answer,prediction", [
    ("fine_location", "the basket", "box or basket"),
    ("coarse_location", "No", "yes or no"),
    ("coarse_location", "No", "unknown"),
    ("fullness", "less full", "more full or less full"),
    ("accessibility", "more accessible", "more accessible or less accessible"),
    ("attitude", "positive", "positive or negative"),
])
def test_ambiguous_or_unparsed_outputs_are_not_credited(kind, answer, prediction):
    result = score_opentom(prediction, record(kind, answer))
    assert result["canonical_prediction"] is None and result["prediction_label"] is None
    assert not result["correct"]


def test_exact_options_resolve_nested_place_names_while_upstream_tie_is_retained():
    row = record(answer="the kitchen pantry", options=["pantry", "the kitchen pantry"])
    row["plot_info"] = {"original_place": "pantry", "move_to_place": "the kitchen pantry"}
    scored = score_opentom("the kitchen pantry", row)
    assert scored["correct"] and scored["prediction_label"] == 2
    assert not scored["upstream_correct"] and scored["upstream_prediction_label"] == 3
    assert scored["upstream_report_included"]  # Pinned evaluator counts a location tie as wrong.


def test_documented_metadata_typo_is_corrected_independently_of_gold():
    row = record(answer="center of dining table")
    row["plot_info"] = {"original_place": "bottle", "move_to_place": "center of dinint table"}
    question = {"question": row["question"], "type": row["question_type"], "answer": "not used"}
    kind, options = question_protocol(question, row["plot_info"])
    assert options == ["bottle", "center of dining table"]
    row.update(answer_kind=kind, answer_options=options)
    scored = score_opentom("center of dining table", row)
    assert scored["correct"] and not scored["upstream_gold_valid"]
    # Other records in that same narrative retain the typo in their gold text.
    row["answer_text"] = "center of dinint table"
    assert score_opentom("center of dining table", row)["correct"]


class StubBackend:
    checkpoint_id = "stub-reader"

    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.prompts = []

    class tokenizer:
        @staticmethod
        def apply_chat_template(messages, **kwargs):
            return "\n".join(message["content"] for message in messages)

    def generate(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return {"text": next(self.outputs), "truncated": False}


@pytest.mark.parametrize("kind,answer", [("fine_location", "the basket"), ("coarse_location", "Yes"),
                                        ("fullness", "less full"), ("accessibility", "less accessible"),
                                        ("attitude", "neutral")])
def test_real_evaluation_prompts_show_options_without_gold_conditioning(kind, answer):
    row = record(kind, answer)
    backend = StubBackend([answer, answer])
    result = next(evaluate_opentom(backend, [row], protocol="adapted"))
    changed = copy.deepcopy(row)
    changed["answer_text"] = next(option for option in row["answer_options"] if option.lower() != answer.lower())
    next(evaluate_opentom(backend, [changed], protocol="adapted"))
    assert backend.prompts[0] == backend.prompts[1]
    assert all(repr(option) in backend.prompts[0] for option in row["answer_options"])
    assert result["correct"] and result["scoring_protocol"] == SCORING_PROTOCOL
    if kind == "attitude":
        assert "assuming that you observed the action?" in backend.prompts[0]


def test_reader_rejects_answer_options_not_derived_from_question_and_plot():
    row = record()
    row["answer_options"] = ["the basket"]
    with pytest.raises(ValueError, match="question/plot-derived"):
        next(evaluate_opentom(StubBackend([]), [row]))


def test_reasoning_is_scored_from_final_answer_and_missing_delimiter_is_visible():
    backend = StubBackend(["The character saw the move.", " basket"])
    result = next(evaluate_opentom(backend, [record()], reasoning=True, protocol="adapted"))
    assert result["correct"] and result["forced_delimiter"]
    assert result["trace"] == "The character saw the move."
    assert backend.prompts[-1].endswith("\nAnswer:")


def test_import_keeps_raw_data_and_records_gold_compatibility_audit(tmp_path):
    source = tmp_path / "source.json"
    source.write_text(json.dumps([{"narrative": "Story.", "plot_info": {"original_place": "drawer", "move_to_place": "another drawer"},
                                  "question": {"question": "Where?", "type": "location-fo", "answer": "another drawer"}}]))
    output = tmp_path / "rows.jsonl"
    rows = import_opentom(source, output)
    assert rows[0]["answer_text"] == "another drawer"
    assert rows[0]["plot_info"]["move_to_place"] == "another drawer"
    provenance = json.loads((tmp_path / "rows.jsonl.provenance.json").read_text())
    assert provenance["gold_self_score_correct"] == 1
    assert provenance["upstream_gold_self_score_failures"] == 1
    assert "not exact published" in provenance["prompt_adaptation"]


def test_summary_separates_subtypes_macro_f1_and_invalid_output_denominators():
    rows = [record("coarse_location", answer) for answer in ("Yes", "No", "No")]
    results = [{**row, **score_opentom(prediction, row)} for row, prediction in zip(rows, ("Yes", "Yes", "?"))]
    summary = summarize_opentom(results)
    assert summary["accuracy"] == pytest.approx(1 / 3)
    subtype = summary["by_subtype"]["location-fo|coarse_location"]
    assert subtype["macro_f1"] == pytest.approx(1 / 3)
    assert subtype["invalid_predictions"] == 1
    assert subtype["upstream_diagnostic"]["included"] == 2
    assert subtype["upstream_diagnostic"]["accuracy_excluding_invalid"] == .5


def test_pinned_official_prompt_and_lexical_scorer_parity():
    from pathlib import Path
    from paper2.opentom import OPENTOM_REVISION, official_prompt, upstream_score
    fixture = json.loads((Path(__file__).parent / 'fixtures/opentom_official_parity.json').read_text())
    assert fixture['revision'] == OPENTOM_REVISION
    for case in fixture['cases']:
        row = record(case['kind'], case['answer'])
        row.update(story=case['story'], question=case['question'])
        assert official_prompt(row, case['reasoning']) == case['official_prompt']
        for example in case['scores']:
            actual = upstream_score(example['prediction'], row)
            assert actual['upstream_prediction_label'] == example['prediction_label']
            assert actual['upstream_target_label'] == example['target_label']


def test_default_official_run_has_no_paper2_answer_injection_and_parses_upstream_cot():
    from paper2.opentom import OFFICIAL_PROTOCOL, OFFICIAL_SYSTEM, official_prompt
    backend = StubBackend(['The character saw the move.\nTherefore, the basket.'])
    result = next(evaluate_opentom(backend, [record()], reasoning=True))
    assert backend.prompts == [OFFICIAL_SYSTEM + '\n' + official_prompt(record(), True)]
    assert len(backend.prompts) == 1 and not backend.prompts[0].endswith('Answer:')
    assert result['prediction_text'] == 'Therefore, the basket.'
    assert result['correct'] and not result['forced_delimiter']
    assert result['evaluation_protocol'] == OFFICIAL_PROTOCOL and result['primary_metric'] == 'macro_f1'
    assert result['raw_generation'].startswith('The character')


def test_official_primary_f1_retains_upstream_exclusions_and_story_intervals():
    backend = StubBackend(['Yes', 'Yes', '?'])
    rows = [record('coarse_location', answer) for answer in ('Yes', 'No', 'No')]
    results = list(evaluate_opentom(backend, rows))
    report = summarize_opentom(results, samples=100)
    subtype = report['by_subtype']['location-fo|coarse_location']
    assert subtype['macro_f1'] == pytest.approx(1 / 3)
    assert subtype['accuracy'] == .5 and subtype['accuracy_full_denominator'] == pytest.approx(1 / 3)
    assert subtype['included'] == 2 and subtype['excluded'] == 1
    assert subtype['bootstrap']['unit'] == 'whole_story' and subtype['bootstrap']['stories'] == 1
    assert subtype['macro_f1_ci95'] == pytest.approx([1 / 3, 1 / 3])


def test_official_and_adapted_results_and_readers_never_pool():
    row = record('coarse_location', 'No')
    primary = list(evaluate_opentom(StubBackend(['No']), [row]))
    secondary = list(evaluate_opentom(StubBackend(['No']), [row], protocol='adapted'))
    other = {**primary[0], 'reader': 'other-checkpoint'}
    result = summarize_opentom(primary + secondary + [other], samples=20)
    assert len(result['by_run']) == 3
    assert any(value.get('role') == 'secondary_adapted_check' for value in result['by_run'].values())
