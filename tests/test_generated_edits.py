"""Model output is a candidate; a frozen random human audit grants batch use."""
from copy import deepcopy
import json
import random
import re

import pytest

from paper2.data.world import generate_world, render_record
from paper2.edit_generation import (CHECKS, CONTROL_PROMPT, generate_edit_controls,
                                    prepare_batch_audit, prepare_generation_plan,
                                    validate_batch_audit, validate_generated_batch,
                                    exact_error_rate_upper, introduced_answer_statements)
from paper2.io import digest
from paper2.traces import prepare_edit_manifest, symbolic_edit_prediction, validate_edit


REVISION = "a" * 40


def fixture(worlds=2, sources=2, *, audit_sample_size=100, stated_location=None, answer_statement=False):
    rows, traces = [], []
    for seed in range(worlds):
        row = render_record(generate_world(random.Random(seed)), "direct")
        rows.append(row)
        premise = symbolic_edit_prediction(row)["fact_before"]
        if stated_location is not None:
            premise = premise.rsplit(" ", 1)[0] + " " + stated_location
        text = premise + ". Then we can consider the events."
        if answer_statement:
            text += " The answer is A."
        for number in range(sources):
            source = f"source-{number}"
            checkpoint = {"id": source, "revision": "b" * 40}
            artifact = digest([source, checkpoint, row["id"], "direct_fixed", digest(text)])
            traces.append({"schema_version": 2, "artifact_id": artifact,
                           "trace_id": digest([artifact, row["id"]]), "source": source,
                           "source_checkpoint": checkpoint, "basis_record_id": row["id"],
                           "record_id": row["id"], "world_id": row["world_id"], "view": "direct",
                           "question_timing": "direct_fixed", "trace": text, "trace_sha256": digest(text)})
    manifest = prepare_edit_manifest(traces, rows, max_worlds=worlds)
    plan = prepare_generation_plan(manifest, model_id="fixed-editor", revision=REVISION,
                                   audit_sample_size=audit_sample_size)
    return traces, manifest, plan


class Editor:
    checkpoint_id = "fixed-editor"
    revision = REVISION

    class tokenizer:
        chat_template = "<user>{{ user_content }}</user><assistant>"

        @staticmethod
        def apply_chat_template(messages, *, tokenize, add_generation_prompt):
            assert tokenize is False and add_generation_prompt is True
            assert len(messages) == 1 and messages[0]["role"] == "user"
            return "<user>" + messages[0]["content"] + "</user><assistant>"

    def __init__(self, mutate=None):
        self.calls = []
        self.mutate = mutate

    def generate(self, prompt, **decoding):
        self.calls.append((prompt, decoding))
        assert prompt.startswith("<user>" + CONTROL_PROMPT) and prompt.endswith("</user><assistant>")
        payload = json.loads(prompt[len("<user>" + CONTROL_PROMPT):-len("</user><assistant>")])
        trace = payload["trace"]
        before = re.search(r" is in the (\w+)", trace).group(1)
        after = payload["target_by_stated_location"][before]
        output = {"status": "ready", "stated_location": before,
                  "paraphrase": trace.replace("Then we can", "Next we may"),
                  "semantic": {"old": before, "new": after},
                  "irrelevant": {"old": "Then", "new": "Next"}}
        if self.mutate:
            output = self.mutate(output)
        return {"text": output if isinstance(output, str) else json.dumps(output),
                "truncated": False, "tokens": 100, "seed": decoding["seed"]}


def approved_audit(batch):
    audit = prepare_batch_audit(batch)
    for item in audit["items"]:
        item["reviewed_by"] = "Human reviewer"
        item["checks"] = {check: True for check in item["checks"]}
    return audit


def rehash_batch(batch):
    evidence = batch["generated_batch"]
    evidence["contract_content_sha256"] = digest(batch["edits"])
    evidence["batch_sha256"] = digest({key: value for key, value in evidence.items() if key != "batch_sha256"})


def test_generation_freezes_model_prompt_oracle_and_never_claims_semantic_approval():
    traces, manifest, plan = fixture()
    editor, journal = Editor(), []
    batch = generate_edit_controls(editor, traces, manifest, plan, on_result=journal.append)
    assert len(editor.calls) == len(journal) == 4
    assert validate_generated_batch(batch)
    for contract, template in zip(batch["edits"], manifest["edits"]):
        assert contract["generation_status"] == "ready"
        assert contract["approved"] is False and contract["reviewed_by"] is None
        assert contract["paraphrase"]["meaning_preserved"] is False
        assert contract["semantic"]["only_intended_fact_changed"] is False
        selected = template["semantic"]["trace_edit_options"][contract["semantic"]["stated_location"]]
        assert contract["semantic"]["expected_answers"] == selected["expected_answers"]
        assert contract["semantic"]["oracle"] == selected["oracle"]
        assert "expected_answers" not in contract["generation"]["input"]
        assert contract["generation"]["model"] == plan["model"]
        assert contract["generation"]["prompt_format"] == "single_user_chat_v1"
        assert contract["generation"]["prompt"].startswith("<user>")
    assert all(decode["do_sample"] is False for _, decode in editor.calls)
    assert digest(manifest) == plan["input_manifest_sha256"]
    assert plan["resource_counts"]["contract_generations"] == 4
    assert plan["resource_counts"]["max_output_tokens"] == 4 * plan["decoding"]["max_new_tokens"]


def test_audit_sample_is_fixed_before_generation_and_covers_every_checkpoint():
    traces, manifest, plan = fixture(worlds=2, sources=12, audit_sample_size=2)
    batch = generate_edit_controls(Editor(), traces, manifest, plan)
    audit = prepare_batch_audit(batch)
    assert [item["artifact_id"] for item in audit["items"]] == plan["audit"]["sample_artifact_ids"]
    assert len(plan["audit"]["uniform_artifact_ids"]) == 2
    assert len({item["source"] for item in audit["items"]}) == 12
    assert audit["sample_size"] <= 14  # Global sample plus missing sources, not 60 per source.
    assert "Uniform random seed" in plan["audit"]["sampling_assumptions"]
    assert prepare_batch_audit(batch) == audit


def test_batch_acceptance_binds_every_contract_without_individual_approval():
    traces, manifest, plan = fixture()
    batch = generate_edit_controls(Editor(), traces, manifest, plan)
    approval = validate_batch_audit(batch, approved_audit(batch))
    assert approval["bound_contract_hashes"] == {
        contract["artifact_id"]: digest(contract) for contract in batch["edits"]}
    assert approval["sample_size"] <= approval["population_size"]
    assert all(contract["approved"] is False for contract in batch["edits"])
    trace_index = {trace["artifact_id"]: trace for trace in traces}
    for contract in batch["edits"]:
        validate_edit(trace_index[contract["artifact_id"]], contract, batch_review=approval)
    with pytest.raises(ValueError, match="human reviewer"):
        validate_batch_audit(batch, prepare_batch_audit(batch))


@pytest.mark.parametrize("mutation", ["missing_check", "missing_item", "changed_sample", "no_reviewer"])
def test_human_audit_rejects_missing_or_substituted_checks(mutation):
    traces, manifest, plan = fixture(worlds=5)
    batch = generate_edit_controls(Editor(), traces, manifest, plan)
    audit = approved_audit(batch)
    if mutation == "missing_check":
        del audit["items"][0]["checks"][CHECKS[0]]
    elif mutation == "missing_item":
        audit["items"].pop()
    elif mutation == "changed_sample":
        audit["items"][0]["artifact_id"] = "different"
    else:
        audit["items"][0]["reviewed_by"] = " "
    with pytest.raises(ValueError):
        validate_batch_audit(batch, audit)


@pytest.mark.parametrize("mutation", ["invalid_json", "missing_fact", "answer_leak", "wrong_span",
                                       "extra_fact", "overlap", "duplicate_key"])
def test_invalid_model_outputs_are_retained_as_operational_failures_without_usable_controls(mutation):
    def mutate(output):
        if mutation == "invalid_json":
            return "```json\n{}\n```"
        if mutation == "missing_fact":
            return {"unresolved": "The focal belief is not stated."}
        if mutation == "answer_leak":
            output["paraphrase"] += "\nAnswer: A"
        elif mutation == "wrong_span":
            output["semantic"]["old"] += " absent"
        elif mutation == "extra_fact":
            output["semantic"]["new"] += " and Bob knows"
        elif mutation == "overlap":
            output["irrelevant"] = deepcopy(output["semantic"])
        elif mutation == "duplicate_key":
            return '{"paraphrase":"first","paraphrase":"second"}'
        return output

    traces, manifest, plan = fixture(worlds=1, sources=1)
    batch = generate_edit_controls(Editor(mutate), traces, manifest, plan)
    contract = batch["edits"][0]
    assert contract["generation_status"] == "error"
    assert contract["generation"]["raw_output"]
    assert contract["generation"]["errors"]
    assert len(batch["edits"]) == len(manifest["edits"])
    certificate = validate_batch_audit(batch, approved_audit(batch))
    assert certificate["status_counts"] == {"error": 1}
    assert certificate["bound_contract_hashes"] == {}
    assert certificate["failed_contract_hashes"] == {contract["artifact_id"]: digest(contract)}


@pytest.mark.parametrize("mutation", ["wrong_model", "wrong_revision", "changed_template", "changed_plan"])
def test_bindings_fail_before_any_inference(mutation):
    traces, manifest, plan = fixture()
    editor = Editor()
    if mutation == "wrong_model":
        editor.checkpoint_id = "other-editor"
    elif mutation == "wrong_revision":
        editor.revision = "main"
    elif mutation == "changed_template":
        manifest["edits"][0]["semantic"]["fact_after"] += " changed"
    else:
        plan["decoding"]["max_new_tokens"] += 1
    with pytest.raises(ValueError):
        generate_edit_controls(editor, traces, manifest, plan)
    assert editor.calls == []


def test_plan_rejects_mutable_revision_and_post_generation_registration():
    traces, manifest, plan = fixture()
    with pytest.raises(ValueError, match="immutable"):
        prepare_generation_plan(manifest, model_id="editor", revision="main")
    batch = generate_edit_controls(Editor(), traces, manifest, plan)
    with pytest.raises(ValueError, match="before populating"):
        prepare_generation_plan(batch, model_id="editor", revision=REVISION)


def test_changing_any_contract_invalidates_prior_sample_audit():
    traces, manifest, plan = fixture()
    batch = generate_edit_controls(Editor(), traces, manifest, plan)
    audit = approved_audit(batch)
    batch["edits"][-1]["paraphrase"]["text"] += " altered"
    with pytest.raises(ValueError, match="content"):
        validate_batch_audit(batch, audit)
    rehash_batch(batch)
    with pytest.raises(ValueError, match="frozen model output"):
        validate_batch_audit(batch, audit)


def test_relabeling_machine_outputs_as_individually_reviewed_is_rejected():
    traces, manifest, plan = fixture()
    batch = generate_edit_controls(Editor(), traces, manifest, plan)
    batch["edits"][0]["approved"] = True
    rehash_batch(batch)
    with pytest.raises(ValueError, match="individual human approval"):
        validate_generated_batch(batch)


def test_resume_uses_exact_frozen_outputs_without_regenerating_them():
    traces, manifest, plan = fixture()
    journal = []
    original = generate_edit_controls(Editor(), traces, manifest, plan, on_result=journal.append)
    resumed, emitted = Editor(), []
    batch = generate_edit_controls(resumed, traces, manifest, plan,
                                    completed_results=journal[:2], on_result=emitted.append)
    assert len(resumed.calls) == len(emitted) == 2
    assert batch == original
    with pytest.raises(ValueError, match="duplicate"):
        generate_edit_controls(Editor(), traces, manifest, plan, completed_results=journal + journal[:1])
    changed = deepcopy(journal[:1])
    changed[0]["generation"]["raw_output"] += " "
    editor = Editor()
    with pytest.raises(ValueError, match="Generation record hash"):
        generate_edit_controls(editor, traces, manifest, plan, completed_results=changed)
    assert editor.calls == []


def test_resume_does_not_retry_unresolved_outputs():
    traces, manifest, plan = fixture()
    unresolved = generate_edit_controls(Editor(lambda _: {"unresolved": "No focal premise"}), traces, manifest, plan)
    editor = Editor()
    batch = generate_edit_controls(editor, traces, manifest, plan, completed_results=unresolved["edits"])
    assert editor.calls == []
    assert batch == unresolved
    assert validate_batch_audit(batch, approved_audit(batch))["status_counts"] == {"error": 4}


def test_inference_failure_is_recorded_without_dropping_contracts():
    class FailedEditor(Editor):
        def generate(self, *args, **kwargs):
            raise RuntimeError("inference unavailable")

    traces, manifest, plan = fixture()
    batch = generate_edit_controls(FailedEditor(), traces, manifest, plan)
    assert batch["generated_batch"]["ready_count"] == 0
    assert len(batch["edits"]) == 4
    assert all(c["generation"]["errors"] == ["RuntimeError: inference unavailable"] for c in batch["edits"])
    editor = Editor()
    assert generate_edit_controls(editor, traces, manifest, plan, completed_results=batch["edits"]) == batch
    assert editor.calls == []


def test_returned_prompt_tokens_must_match_frozen_native_prompt():
    class WrongPromptEditor(Editor):
        @staticmethod
        def token_ids(text):
            return list(text.encode())

        def generate(self, prompt, **decoding):
            result = super().generate(prompt, **decoding)
            result["prompt_ids"] = self.token_ids("different prompt")
            return result

    traces, manifest, plan = fixture(worlds=1, sources=1)
    batch = generate_edit_controls(WrongPromptEditor(), traces, manifest, plan)
    assert batch["edits"][0]["generation_status"] == "error"
    assert "prompt tokens" in batch["edits"][0]["generation"]["errors"][0]


@pytest.mark.parametrize("use_initial_destination", [False, True])
def test_incorrect_stated_belief_is_edited_without_replacing_it_with_truth(use_initial_destination):
    from paper2.data.world import LOCATIONS
    _, initial, _ = fixture(worlds=1, sources=1)
    truth = initial["edits"][0]["semantic"]["oracle"]["before"]
    wrong = (initial["edits"][0]["semantic"]["oracle"]["after"] if use_initial_destination
             else next(location for location in LOCATIONS if location != truth))
    traces, manifest, plan = fixture(worlds=1, sources=1, stated_location=wrong)
    batch = generate_edit_controls(Editor(), traces, manifest, plan)
    semantic = batch["edits"][0]["semantic"]
    assert semantic["old"] == semantic["stated_location"] == wrong
    assert semantic["oracle"]["truth_before"] == truth
    assert semantic["new"] not in (wrong, truth)
    assert semantic["expected_before_answers"] != semantic["expected_answers"]
    assert validate_batch_audit(batch, approved_audit(batch))["status_counts"] == {"ready": 1}


@pytest.mark.parametrize("reason", ["belief_not_stated", "contradicted", "ambiguous", "no_safe_control"])
def test_scientific_not_editable_outcomes_remain_in_batch_and_audit(reason):
    traces, manifest, plan = fixture()
    editor = Editor(lambda _: {"status": "not_editable", "reason_code": reason,
                               "explanation": "The trace does not support a single safe focal-belief edit."})
    batch = generate_edit_controls(editor, traces, manifest, plan)
    assert batch["generated_batch"]["ready_count"] == 0
    assert len(validate_generated_batch(batch)) == 4
    assert all(contract["generation_status"] == "not_editable" for contract in batch["edits"])
    certificate = validate_batch_audit(batch, approved_audit(batch))
    assert certificate["bound_contract_hashes"] == {}
    assert len(certificate["not_editable_contract_hashes"]) == 4
    assert len(certificate["effective_contracts"]) == 4
    assert certificate["status_counts"] == {"not_editable": 4}
    resumed = Editor()
    assert generate_edit_controls(resumed, traces, manifest, plan, completed_results=batch["edits"]) == batch
    assert resumed.calls == []


def test_existing_answer_statements_are_preserved_but_new_labels_are_rejected():
    traces, manifest, plan = fixture(answer_statement=True)
    batch = generate_edit_controls(Editor(), traces, manifest, plan)
    assert validate_generated_batch(batch)
    assert all("The answer is A." in c["paraphrase"]["text"] for c in batch["edits"])
    assert introduced_answer_statements("The answer is A.", "The answer is B.")
    assert introduced_answer_statements("The answer is A.", "The answer is A. The answer is A.")
    assert not introduced_answer_statements("The answer is A.", "Our answer is A.")
    assert not introduced_answer_statements("Answer: A", "Answer: A")


@pytest.mark.parametrize("original,changed", [
    ("Answer: (A)", "Answer: (B)"),
    ("The answer is (A).", "The answer is (B)."),
    ("The answer is [A].", "The answer is [B]."),
    ("The answer is **A**.", "The answer is **B**."),
    ("**Answer:** **(A)**", "**Answer:** **(B)**"),
    ("The answer is (a).", "The answer is (b)."),
    ("The answer is **a**.", "The answer is **b**."),
    ("No answer letter here.", "No answer letter here. The correct option is (C)."),
])
def test_common_formatted_answer_labels_detect_addition_and_allow_exact_preservation(original, changed):
    assert introduced_answer_statements(original, changed)
    assert not introduced_answer_statements(original, original)


@pytest.mark.parametrize("prose", ["The answer is a box.", "The answer is the drawer.",
                                    "The answer is probably the box.", "The choice is clear.",
                                    "The answer is not clear."])
def test_ordinary_prose_is_not_misclassified_as_an_explicit_answer_label(prose):
    assert not introduced_answer_statements("", prose)


def test_exact_error_bound_allows_one_failure_with_default_sample_but_not_coarse_sample():
    assert exact_error_rate_upper(2425, 100, 1) < .05
    assert exact_error_rate_upper(2425, 60, 1) > .05
    assert exact_error_rate_upper(40, 40, 1) == .025
    assert exact_error_rate_upper(40, 40, 0) == 0
    assert exact_error_rate_upper(2425, 100, 2) > .05
    assert exact_error_rate_upper(20, 1, 0) == .95  # Exact boundary: 1/20 == alpha, no float drift.


def corrected_sample(audit, batch, *, item_index=0, to_not_editable=False):
    item = audit["items"][item_index]
    original = next(c for c in batch["edits"] if c["artifact_id"] == item["artifact_id"])
    item["checks"]["paraphrase_meaning_preserved"] = False
    controls = json.loads(original["generation"]["raw_output"])
    controls["paraphrase"] = controls["paraphrase"].replace("Next we may", "Afterward we can")
    if to_not_editable:
        item["checks"]["editability_classification_correct"] = False
        controls = {"status": "not_editable", "reason_code": "contradicted",
                    "explanation": "Human review identified incompatible final focal beliefs."}
    item["correction"] = {"controls": controls, "reason": "Correct the sampled control after human inspection.",
                          "reviewed_by": "Human second reviewer",
                          "checks": {check: True for check in (CHECKS[:1] if to_not_editable else CHECKS)}}
    return original, item


def test_small_audit_failure_can_be_corrected_without_erasing_original_or_resampling():
    traces, manifest, plan = fixture(worlds=20, sources=2)
    batch = generate_edit_controls(Editor(), traces, manifest, plan)
    before = digest(batch)
    audit = approved_audit(batch)
    original, item = corrected_sample(audit, batch)
    certificate = validate_batch_audit(batch, audit)
    corrected = certificate["effective_contracts"][item["artifact_id"]]
    assert digest(batch) == before
    assert corrected["generation"] == original["generation"]
    assert corrected["human_correction"]["original_contract_sha256"] == digest(original)
    assert corrected["paraphrase"]["text"] != original["paraphrase"]["text"]
    assert certificate["bound_contract_hashes"][item["artifact_id"]] == digest(corrected)
    assert certificate["audit_error_rate"]["failure_count"] == 1
    assert certificate["audit_error_rate"]["one_sided_upper_rate"] == .025
    assert certificate["audit_error_rate"]["original_ready"]["one_sided_upper_rate"] == .025
    assert corrected["approved"] is False
    trace = next(t for t in traces if t["artifact_id"] == item["artifact_id"])
    validate_edit(trace, corrected, batch_review=certificate)
    item["correction"]["checks"]["paraphrase_meaning_preserved"] = False
    with pytest.raises(ValueError, match="re-review"):
        validate_batch_audit(batch, audit)


def test_corrected_editability_classification_still_counts_as_original_failure():
    traces, manifest, plan = fixture(worlds=20, sources=2)
    batch = generate_edit_controls(Editor(), traces, manifest, plan)
    audit = approved_audit(batch)
    _, item = corrected_sample(audit, batch, to_not_editable=True)
    certificate = validate_batch_audit(batch, audit)
    assert certificate["status_counts"] == {"ready": 39, "not_editable": 1}
    assert certificate["audit_error_rate"]["original_ready"]["population_size"] == 40
    assert certificate["audit_error_rate"]["failure_count"] == 1
    assert item["artifact_id"] in certificate["not_editable_contract_hashes"]
    assert item["artifact_id"] not in certificate["bound_contract_hashes"]


def test_multiple_sample_failures_are_reported_without_a_numeric_cap():
    traces, manifest, plan = fixture(worlds=20, sources=2)
    batch = generate_edit_controls(Editor(), traces, manifest, plan)
    audit = approved_audit(batch)
    corrected_sample(audit, batch, item_index=0)
    corrected_sample(audit, batch, item_index=1)
    certificate = validate_batch_audit(batch, audit)
    assert certificate["audit_complete"]
    assert certificate["audit_error_rate"]["flaw_rate"] == .05
    assert certificate["audit_error_rate"]["ci95"] == [.05, .05]


def manual_controls(contract):
    trace = contract["generation"]["input"]["trace"]
    before = re.search(r" is in the (\w+)", trace).group(1)
    after = contract["semantic"]["trace_edit_options"][before]["oracle"]["after"]
    return {"status": "ready", "stated_location": before,
            "paraphrase": trace.replace("Then we can", "Afterward we can"),
            "semantic": {"old": before, "new": after},
            "irrelevant": {"old": "Then", "new": "Next"}}


def test_human_can_correct_sampled_false_ineligibility_without_losing_original_failure():
    traces, manifest, plan = fixture(worlds=20, sources=2)
    batch = generate_edit_controls(Editor(lambda _: {
        "status": "not_editable", "reason_code": "belief_not_stated", "explanation": "No focal fact found."}),
        traces, manifest, plan)
    audit = approved_audit(batch)
    item = audit["items"][0]
    original = next(c for c in batch["edits"] if c["artifact_id"] == item["artifact_id"])
    item["checks"]["editability_classification_correct"] = False
    item["correction"] = {"controls": manual_controls(original), "reason": "Human located the stated focal belief.",
                          "reviewed_by": "Human reviewer", "checks": {check: True for check in CHECKS}}
    certificate = validate_batch_audit(batch, audit)
    assert certificate["status_counts"] == {"ready": 1, "not_editable": 39}
    assert certificate["original_status_counts"] == {"not_editable": 40}
    assert certificate["audit_error_rate"]["failure_count"] == 1
    assert certificate["effective_contracts"][item["artifact_id"]]["generation"] == original["generation"]


def test_human_can_repair_sampled_generation_error_and_original_operational_rate_remains():
    traces, manifest, plan = fixture()
    batch = generate_edit_controls(Editor(lambda _: "malformed output"), traces, manifest, plan)
    audit = approved_audit(batch)
    item = audit["items"][0]
    original = next(c for c in batch["edits"] if c["artifact_id"] == item["artifact_id"])
    assert item["checks"] == {"generation_failure_recorded_correctly": True}
    item["correction"] = {"controls": manual_controls(original), "reason": "Human supplied a valid local edit.",
                          "reviewed_by": "Human reviewer", "checks": {check: True for check in CHECKS}}
    certificate = validate_batch_audit(batch, audit)
    assert certificate["original_status_counts"] == {"error": 4}
    assert certificate["status_counts"] == {"ready": 1, "error": 3}
    assert certificate["audit_error_rate"]["failure_count"] == 0
    assert certificate["audit_error_rate"]["original_ready"]["one_sided_upper_rate"] is None
    assert certificate["effective_contracts"][item["artifact_id"]]["generation"]["raw_output"] == "malformed output"


def test_audited_batch_runs_both_readers_on_same_controls_and_source_regeneration():
    from paper2.backend import Answer
    from paper2.prompts import render_prompt
    from paper2.traces import regenerate_trace_artifacts, run_crossed_reader

    class Reader:
        class tokenizer:
            @staticmethod
            def apply_chat_template(messages, **kwargs):
                return "\n".join(message["content"] for message in messages) + "\nAssistant:"
        revision = "b" * 40

        def __init__(self, checkpoint):
            self.checkpoint_id = checkpoint
            self.generated = 0

        def token_ids(self, text):
            return list(text.encode())

        def answer(self, prompt, choices, prefix_ids=None):
            return Answer("A", 65, "A", "A", {})

        def generate(self, prompt, **kwargs):
            self.generated += 1
            return {"text": ". The narration continues.\nAnswer:", "truncated": False, "tokens": 7}

    traces, manifest, plan = fixture()
    rows = [render_record(generate_world(random.Random(seed)), "direct") for seed in range(2)]
    records = {row["id"]: row for row in rows}
    # Reader prompts use the ordinary system/user formatter.
    for trace in traces:
        trace.update(forced_delimiter=False, replay_eligible=False,
                     generation_prompt=render_prompt(Reader.tokenizer, records[trace["record_id"]], reasoning=True))
    batch = generate_edit_controls(Editor(), traces, manifest, plan)
    audit = approved_audit(batch)
    readers = [Reader("source-0"), Reader("source-1")]
    with pytest.raises(ValueError, match="completed preregistered sample audit"):
        list(run_crossed_reader(readers[0], rows, traces, "source-0", batch,
                                conditions=("semantic_minimal",)))
    frozen = [item for backend in readers for item in regenerate_trace_artifacts(
        backend, rows, traces, backend.checkpoint_id, batch, edit_audit=audit)]
    assert [reader.generated for reader in readers] == [2, 2]
    conditions = ("paraphrase", "irrelevant", "semantic_minimal", "semantic_regenerated")
    outputs = [list(run_crossed_reader(backend, rows, traces, backend.checkpoint_id, batch,
                                      conditions=conditions, regenerated=frozen, edit_audit=audit))
               for backend in readers]
    assert [reader.generated for reader in readers] == [2, 2]  # Reading never regenerates anything.
    assert [row["supplied_trace"] for row in outputs[0]] == [row["supplied_trace"] for row in outputs[1]]
    assert all(row["edit_review_mode"] == "sampled_model_generated" for row in outputs[0])
    assert all(row["edit_audit_sha256"] == digest(audit) for row in outputs[0])


@pytest.mark.parametrize("with_human_repair", [False, True])
def test_mixed_outcomes_and_human_repairs_flow_through_reader_and_source_artifacts(with_human_repair):
    from paper2.backend import Answer
    from paper2.prompts import render_prompt
    from paper2.traces import regenerate_trace_artifacts, run_crossed_reader

    class Reader:
        class tokenizer:
            @staticmethod
            def apply_chat_template(messages, **kwargs):
                return "\n".join(message["content"] for message in messages) + "\nAssistant:"
        revision = "b" * 40

        def __init__(self, checkpoint):
            self.checkpoint_id, self.generated = checkpoint, 0

        def token_ids(self, text):
            return list(text.encode())

        def answer(self, prompt, choices, prefix_ids=None):
            return Answer("A", 65, "A", "A", {})

        def generate(self, prompt, **kwargs):
            self.generated += 1
            return {"text": ". Further reasoning.\nAnswer:", "truncated": False, "tokens": 5}

    class MixedEditor(Editor):
        def generate(self, prompt, **kwargs):
            output = super().generate(prompt, **kwargs)
            if not with_human_repair and len(self.calls) % 3 == 1:
                output["text"] = json.dumps({"status": "not_editable", "reason_code": "contradicted",
                                              "explanation": "Conflicting final focal claims."})
            elif not with_human_repair and len(self.calls) % 3 == 2:
                output["text"] = "Malformed candidate"
            return output

    worlds = 10 if with_human_repair else 3
    traces, manifest, plan = fixture(worlds=worlds, sources=2, answer_statement=True)
    rows = [render_record(generate_world(random.Random(seed)), "direct") for seed in range(worlds)]
    records = {row["id"]: row for row in rows}
    for trace in traces:
        trace.update(forced_delimiter=False, replay_eligible=False,
                     generation_prompt=render_prompt(Reader.tokenizer, records[trace["record_id"]], reasoning=True))
    batch = generate_edit_controls(MixedEditor(), traces, manifest, plan)
    audit = approved_audit(batch)
    if with_human_repair:
        corrected_sample(audit, batch)
    certificate = validate_batch_audit(batch, audit)
    readers = [Reader("source-0"), Reader("source-1")]
    frozen = [item for backend in readers for item in regenerate_trace_artifacts(
        backend, rows, traces, backend.checkpoint_id, batch, edit_audit=audit)]
    ready = [contract for contract in certificate["effective_contracts"].values()
             if contract["generation_status"] == "ready"]
    assert len(frozen) == sum(reader.generated for reader in readers) == len(ready)
    effective = certificate["effective_contracts"]
    assert all(item["edit_contract_sha256"] == digest(effective[item["artifact_id"]]) for item in frozen)
    conditions = ("original", "paraphrase", "semantic_minimal", "semantic_regenerated")
    output = list(run_crossed_reader(readers[0], rows, traces, "source-0", batch,
                                     conditions=conditions, regenerated=frozen, edit_audit=audit))
    originals = [row for row in output if row["condition"] == "original"]
    assert len(originals) == len(traces)
    assert len(output) == len(traces) + 3 * len(ready)
    if with_human_repair:
        key = audit["items"][0]["artifact_id"]
        supplied = next(row["supplied_trace"] for row in output
                        if row["artifact_id"] == key and row["condition"] == "paraphrase")
        assert supplied == effective[key]["paraphrase"]["text"]
        assert "Afterward we can" in supplied
    else:
        assert {row["edit_status"] for row in originals} == {"editable", "contradicted", "generation_failed"}
    changed_audit = deepcopy(audit)
    changed_audit["items"][0]["notes"] = "Changed review record after regeneration"
    with pytest.raises(ValueError, match="different edit audit"):
        list(run_crossed_reader(readers[0], rows, traces, "source-0", batch,
             conditions=("semantic_regenerated",), regenerated=frozen, edit_audit=changed_audit))


def test_cli_generation_audit_and_journal_recovery(tmp_path, monkeypatch):
    from paper2.cli import main
    from paper2.io import read_jsonl, write_json, write_jsonl
    traces, manifest, _ = fixture()
    templates, source = tmp_path / "templates.json", tmp_path / "traces.jsonl"
    plan_path, batch_path = tmp_path / "plan.json", tmp_path / "batch.json"
    audit_path, evidence_path = tmp_path / "audit.json", tmp_path / "evidence.json"
    write_json(templates, manifest)
    write_jsonl(source, traces)
    assert main(["plan-edit-generation", "--edits", str(templates), "--model", "fixed-editor",
                 "--revision", REVISION, "--audit-sample-size", "100", "--output", str(plan_path)]) == 0
    with pytest.raises(ValueError, match="already exists"):
        main(["plan-edit-generation", "--edits", str(templates), "--model", "fixed-editor",
              "--revision", REVISION, "--output", str(plan_path)])
    editor = Editor()
    monkeypatch.setattr("paper2.backend.HFBackend", lambda *_, **__: editor)
    command = ["generate-edits", "--edits", str(templates), "--plan", str(plan_path),
               "--traces", str(source), "--output", str(batch_path)]
    # Seed a valid partially completed journal, as after a process interruption.
    completed = generate_edit_controls(Editor(), traces, manifest, json.loads(plan_path.read_text()))
    write_jsonl(str(batch_path) + ".journal.jsonl", completed["edits"][:1])
    with pytest.raises(ValueError, match="requires --resume"):
        main(command)
    assert main(command + ["--resume"]) == 0
    assert len(editor.calls) == 3
    assert len(read_jsonl(str(batch_path) + ".journal.jsonl")) == 4
    batch = json.loads(batch_path.read_text())
    assert batch == completed
    assert main(["prepare-edit-audit", "--edits", str(batch_path), "--output", str(audit_path)]) == 0
    with pytest.raises(ValueError, match="human reviewer"):
        main(["verify-edit-audit", "--edits", str(batch_path), "--audit", str(audit_path),
              "--output", str(evidence_path)])
    write_json(audit_path, approved_audit(batch))
    assert main(["verify-edit-audit", "--edits", str(batch_path), "--audit", str(audit_path),
                 "--output", str(evidence_path)]) == 0
    assert json.loads(evidence_path.read_text())["population_size"] == 4


def test_sparse_audit_reports_uncertainty_without_a_pass_mark():
    traces, manifest, plan = fixture(worlds=10, sources=12, audit_sample_size=2)
    batch = generate_edit_controls(Editor(), traces, manifest, plan)
    audit = approved_audit(batch)
    for index, item in enumerate(audit["items"]):
        item["checks"]["lengths_comparable"] = False
        corrected_sample(audit, batch, item_index=index, to_not_editable=index == 0)
    certificate = validate_batch_audit(batch, audit)
    assert certificate["status_counts"]["not_editable"] == 1
    assert certificate["original_status_counts"] == {"ready": 120}
    assert certificate["audit_error_rate"]["failure_count"] == len(audit["items"])
    assert certificate["audit_error_rate"]["flaw_rate"] == 1
    assert certificate["audit_error_rate"]["one_sided_upper_rate"] == 1
    reports = certificate["audit_error_rate"]["by_source_checkpoint"].values()
    assert any(item["original_ready"]["sample_size"] == 0 for item in reports)
    for report in reports:
        ready = report["original_ready"]
        assert ready["source_family_size"] == 12
        assert ready["simultaneous_one_sided_upper_rate"] >= ready["one_sided_upper_rate"]
        if ready["sample_size"] == 0:
            assert ready["flaw_rate"] is None and ready["ci95"] == [0, 1]


def test_arbitrary_length_ratio_is_reportable_human_quality_not_machine_failure():
    traces, manifest, plan = fixture(worlds=1, sources=1)
    def longer(output):
        output["paraphrase"] *= 3
        output["irrelevant"]["new"] = "After carefully considering all the events"
        return output
    batch = generate_edit_controls(Editor(longer), traces, manifest, plan)
    assert batch["edits"][0]["generation_status"] == "ready"
    assert "lengths_comparable" in prepare_batch_audit(batch)["items"][0]["checks"]


@pytest.mark.parametrize("classification_only", [False, True])
def test_known_sampled_flaw_requires_repair_without_numeric_batch_threshold(classification_only):
    traces, manifest, plan = fixture(worlds=1, sources=1)
    def not_editable(output):
        return {"status": "not_editable", "reason_code": "belief_not_stated",
                "explanation": "Editor reports no focal belief."}
    batch = generate_edit_controls(Editor(not_editable if classification_only else None), traces, manifest, plan)
    audit = approved_audit(batch)
    check = "editability_classification_correct" if classification_only else "semantic_only_intended_fact_changed"
    audit["items"][0]["checks"][check] = False
    with pytest.raises(ValueError, match="documented correction and re-review"):
        validate_batch_audit(batch, audit)
    original = batch["edits"][0]
    audit["items"][0]["correction"] = {
        "controls": manual_controls(original), "reason": "Human repaired the known failed control/classification.",
        "reviewed_by": "Human second reviewer", "checks": {name: True for name in CHECKS}}
    result = validate_batch_audit(batch, audit)
    assert result["audit_complete"] and result["audit_error_rate"]["flaw_rate"] == 1
    assert result["audit_error_rate"]["failure_count"] == 1
    assert original["artifact_id"] in result["bound_contract_hashes"]
