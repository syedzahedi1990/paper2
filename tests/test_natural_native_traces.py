"""Native v2 samples stop at their budget; supplied-trace controls stay separate."""
from copy import deepcopy

import pytest

from paper2.backend import Answer
from paper2.evaluation import evaluate_native
from paper2.traces import (_validate_source_trace, collect_traces, generate_trace,
                           prepare_edit_manifest, regenerate_trace_artifacts, run_crossed_reader)


PROTOCOL = "natural_answer_cap_v2"


class Tokenizer:
    eos_token_id = 0

    @staticmethod
    def apply_chat_template(messages, **kwargs):
        return "\n".join(message["content"] for message in messages) + "\nAssistant:"

    @staticmethod
    def encode(text, add_special_tokens=False):
        # Canonical re-encoding merges xx, while sampled x,x remains legal.
        # This deliberately distinguishes real sampled counts from re-encoding.
        result, i = [], 0
        while i < len(text):
            if text[i:i + 2] == "xx":
                result.append(2000)
                i += 2
            elif text[i:i + 1] == " " and text[i + 1:i + 2] in tuple("ABCDEF"):
                result.append(1000 + ord(text[i + 1]))
                i += 2
            else:
                result.append(ord(text[i]) + 10)
                i += 1
        return result

    @staticmethod
    def decode(ids, skip_special_tokens=True):
        return "".join("" if i == 0 and skip_special_tokens else "<eos>" if i == 0 else
                       "xx" if i == 2000 else "yyy" if i == 3000 else
                       " " + chr(i - 1000) if 1000 <= i < 1200 else
                       chr(i - 10) for i in ids)


class Backend:
    tokenizer = Tokenizer()
    checkpoint_id = "source-model"
    revision = "fixed-revision"

    def __init__(self, completion, *, sampled_ids=None, eos=True):
        self.ids = self.token_ids(completion) if sampled_ids is None else sampled_ids
        if eos:
            self.ids = [*self.ids, 0]
        self.calls, self.answer_calls = [], []

    def token_ids(self, text):
        return self.tokenizer.encode(text)

    def generate(self, prompt, **kwargs):
        self.calls.append({"prompt": prompt, **kwargs})
        if len(self.calls) > 1:
            pytest.fail("Native completion was rescued with an extra generation")
        return {"text": self.tokenizer.decode(self.ids), "prompt_ids": self.token_ids(prompt),
                "generated_ids": self.ids, "tokens": len(self.ids),
                "truncated": len(self.ids) == kwargs["max_new_tokens"]}

    def answer(self, prompt, choices, prefix_ids=None):
        self.answer_calls.append({"prompt": prompt, "prefix_ids": prefix_ids})
        return Answer("A", self.token_ids(" A")[0], " A", "A", {})


def row(view="direct"):
    return {"id": view, "world_id": "world", "view": view, "split": "development",
            "story": "Alice sees the cup in the box.", "question": "Where does Alice believe it is?",
            "choices": {"A": "box", "B": "basket"}, "answer": "A"}


def at_cap(suffix):
    # Exactly384 actual sampled tokens; canonical text re-encoding is shorter.
    suffix_ids = Tokenizer.encode(suffix)
    return [ord("x") + 10] * (384 - len(suffix_ids)) + suffix_ids


@pytest.mark.parametrize("suffix,prediction", [("\nAnswer: A", "A"), ("\nAnswer: A.", "A"),
                                                ("\nAnswer:", None), ("", None)])
def test_literal_cap_counts_sampled_tokens_and_never_rescues(suffix, prediction):
    ids = at_cap(suffix)
    backend = Backend("", sampled_ids=ids, eos=False)
    result, = evaluate_native(backend, [row()], reasoning=True, completion_protocol=PROTOCOL)
    assert len(backend.calls) == 1
    assert backend.calls[0]["max_new_tokens"] == 384 and backend.calls[0].get("stop") is None
    assert result["completion_tokens"] == 384
    assert len(backend.token_ids(result["completion_text"])) < 384
    assert result["prediction"] == prediction and result["correct"] == (prediction == "A")
    assert result["cap_reached"] and result["truncated"] and not result["eos_seen"]
    assert result["cap_failure"] == (prediction is None)
    assert result["answer_complete"] == (prediction is not None)
    assert result["termination_reason"] == ("valid_at_budget" if prediction else "cap_failure")
    assert result["forced_delimiter"] is False and not backend.answer_calls


def test_early_eos_missing_delimiter_is_retained_without_an_answer_attempt():
    backend = Backend("Alice believes box.")
    result, = evaluate_native(backend, [row()], reasoning=True, completion_protocol=PROTOCOL)
    assert not result["correct"] and result["prediction"] is None
    assert result["answer_status"] == "missing_delimiter"
    assert not result["cap_reached"] and not result["truncated"] and result["eos_seen"]
    assert not result["forced_delimiter"] and not result["cap_failure"]
    assert result["invalid_completion"] and len(backend.calls) == 1


@pytest.mark.parametrize("text", ["Alice believes box.\nAnswer:", "Alice believes box.\nAnswer: G",
                                  "Alice believes box.\nAnswer: A because", "Answer:"])
def test_incomplete_native_trace_matches_health_and_editor_input_without_stripping(text):
    from hashlib import sha256
    from paper2.training.trace_health import summarize_trace_health

    backend = Backend(text)
    trace = generate_trace(backend, row(), "base", completion_protocol=PROTOCOL)
    assert trace["trace"] == text and trace["answer_complete"] is False
    _validate_source_trace(trace)
    manifest = prepare_edit_manifest([trace])
    assert manifest["edits"][0]["trace_sha256"] == trace["trace_sha256"]
    raw = {**row(), "format": "native_reasoning", "completion_protocol": PROTOCOL,
           "completion": text, "completion_ids": trace["completion_generated_ids"]}
    health = summarize_trace_health([raw], samples=10)["panels"][0]
    assert health["exact_common_trace"]["observed_modal_trace_sha256"] == sha256(text.encode()).hexdigest()


def test_complete_boundary_answer_is_valid_when_reencoding_would_exceed_cap():
    suffix = Tokenizer.encode("\nAnswer: A")
    ids = [3000] * (384 - len(suffix)) + suffix
    backend = Backend("", sampled_ids=ids, eos=False)
    result, = evaluate_native(backend, [row()], reasoning=True, completion_protocol=PROTOCOL)
    assert len(backend.token_ids(result["completion_text"])) > 384
    assert result["completion_tokens"] == 384 and result["correct"]
    assert result["termination_reason"] == "valid_at_budget"


@pytest.mark.parametrize("protocol,correct", [("legacy_forced_answer_v1", False), (PROTOCOL, True)])
def test_native_answer_period_parsing_is_bound_to_selected_protocol(protocol, correct):
    backend = Backend(" A.")
    result, = evaluate_native(backend, [row()], native_answer=True, completion_protocol=protocol)
    assert result["correct"] is correct
    assert len(backend.calls) == 1 and backend.calls[0]["max_new_tokens"] == 4


def test_naturally_immediate_reasoning_answer_has_empty_trace_and_no_minimum_length():
    backend = Backend("Answer: A.")
    trace = generate_trace(backend, row(), "base", completion_protocol=PROTOCOL)
    assert trace["trace"] == "" and trace["trace_tokens"] == 0
    assert trace["answer_complete"] and trace["native_prediction"] == "A"
    assert trace["replay_eligible"] and trace["ordinary_prediction"] == "A"
    _validate_source_trace(trace)


def test_incomplete_own_replay_does_not_call_reader_but_supplied_trace_controls_do():
    backend = Backend("Alice believes box.")
    trace = generate_trace(backend, row(), "base", completion_protocol=PROTOCOL)
    own, original, rerendered = run_crossed_reader(
        backend, [row()], [trace], "base", conditions=("own_replay", "original", "own_rerendered"))
    assert own["prediction"] is None and own["replay_evaluated"] is False
    assert own["source_answer_complete"] is False and own["source_forced_delimiter"] is False
    assert own["replay_ineligibility_reason"] == "missing_natural_delimiter"
    assert original["correct"] and rerendered["correct"]
    assert rerendered["replay_matches"] is None
    assert len(backend.calls) == 1 and len(backend.answer_calls) == 2
    assert all(call["prompt"].endswith("\nAnswer:") for call in backend.answer_calls)


def test_complete_own_replay_uses_the_sampled_prefix_without_regeneration():
    backend = Backend("Alice believes box. \nAnswer: A.")
    trace = generate_trace(backend, row(), "base", completion_protocol=PROTOCOL)
    replay, = run_crossed_reader(backend, [row()], [trace], "base", conditions=("own_replay",))
    assert replay["raw_replay_matches"] and replay["replay_evaluated"]
    assert backend.answer_calls[0]["prefix_ids"] == trace["answer_prefix_ids"]
    assert trace["replay_token_ids_match"] is False  # Rerendering strips the space.
    assert len(backend.calls) == 1


def test_direct_fixed_shares_one_complete_native_sample_and_edit_review_remains_unapproved():
    backend = Backend("Alice believes box and waits.\nAnswer: A")
    records = [row("observed_overwrite"), row()]
    traces = list(collect_traces(backend, records, "base", timing="direct_fixed", completion_protocol=PROTOCOL))
    assert len(backend.calls) == 1
    assert traces[0]["artifact_id"] == traces[1]["artifact_id"]
    assert traces[0]["completion_generated_ids"] == traces[1]["completion_generated_ids"]
    assert not traces[0]["replay_eligible"] and traces[1]["replay_eligible"]
    for trace in traces:
        _validate_source_trace(trace)
    manifest = prepare_edit_manifest(traces)
    assert len(manifest["edits"]) == 1
    assert manifest["edits"][0]["reviewed_by"] is None and not manifest["edits"][0]["approved"]


def test_reviewed_supplied_edit_controls_still_use_one_source_regeneration():
    trace = generate_trace(Backend("Alice believes box and waits.\nAnswer: A"), row(), "base",
                           completion_protocol=PROTOCOL)
    manifest = prepare_edit_manifest([trace])
    contract = manifest["edits"][0]
    text = trace["trace"]
    start, irrelevant = text.index("box"), text.index("waits")
    contract.update(approved=True, reviewed_by="synthetic test reviewer",
                    semantic={"start": start, "end": start + 3, "old": "box", "new": "basket",
                              "fact_before": "Alice believes box", "fact_after": "Alice believes basket",
                              "expected_answers": {"direct": "B"}, "only_intended_fact_changed": True},
                    irrelevant={"start": irrelevant, "end": irrelevant + 5, "old": "waits", "new": "rests",
                                "irrelevant_to_answers": True},
                    paraphrase={"text": "Alice thinks box and waits.", "meaning_preserved": True})
    source = Backend(" and waits.\nAnswer:")
    regenerated = list(regenerate_trace_artifacts(source, [row()], [trace], "base", manifest))
    assert len(source.calls) == 1 and source.calls[0]["stop"] == "\nAnswer:"
    conditions = ("paraphrase", "semantic_minimal", "semantic_regenerated", "irrelevant")
    cells = [list(run_crossed_reader(Backend("unused"), [row()], [trace], reader, manifest,
                                     conditions=conditions, regenerated=regenerated))
             for reader in ("base", "trained")]
    assert [r["supplied_trace"] for r in cells[0]] == [r["supplied_trace"] for r in cells[1]]
    assert len(cells[0]) == 4


@pytest.mark.parametrize("field,value", [("completion_protocol", "legacy_forced_answer_v1"),
                                         ("completion_text", "changed")])
def test_new_trace_identity_binds_protocol_and_full_sample(field, value):
    trace = generate_trace(Backend("A premise.\nAnswer: A"), row(), "base", completion_protocol=PROTOCOL)
    changed = deepcopy(trace)
    changed[field] = value
    with pytest.raises(ValueError, match="protocol|completion hash"):
        _validate_source_trace(changed)


def test_backend_cannot_silently_exceed_the_bound():
    backend = Backend("", sampled_ids=at_cap("\nAnswer: A") + [0], eos=False)
    with pytest.raises(ValueError, match="exceeded"):
        list(evaluate_native(backend, [row()], reasoning=True, completion_protocol=PROTOCOL))


def test_backend_cannot_count_padding_after_eos_as_sampled_tokens():
    backend = Backend("", sampled_ids=Tokenizer.encode("Answer: A") + [0, 0], eos=False)
    with pytest.raises(ValueError, match="unpadded"):
        list(evaluate_native(backend, [row()], reasoning=True, completion_protocol=PROTOCOL))


def test_unknown_completion_protocol_is_rejected():
    with pytest.raises(ValueError, match="Unknown"):
        list(evaluate_native(Backend(""), [row()], reasoning=True, completion_protocol="typo"))
    with pytest.raises(ValueError, match="Unknown"):
        generate_trace(Backend(""), row(), "base", completion_protocol="typo")
