"""Offline HF integration for replay, frozen pre-question traces and resampling."""

import copy
import string

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers
from transformers import LogitsProcessor, LogitsProcessorList, PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

from paper2.backend import HFBackend
from paper2.prompts import render_prompt
from paper2.traces import (collect_traces, generate_trace, run_crossed_reader,
                           prepare_edit_manifest, regenerate_trace_artifacts)
from paper2.evaluation import evaluate_native


@pytest.fixture
def backend():
    vocabulary = {value: index for index, value in enumerate(["[PAD]", "[EOS]", "[UNK]", *string.printable, *[" " + v for v in "ABCDEF"]])}
    core = Tokenizer(models.WordLevel(vocabulary, unk_token="[UNK]"))
    core.pre_tokenizer = pre_tokenizers.Split(Regex(r" [A-F]|[\s\S]"), behavior="isolated")
    core.decoder = decoders.Fuse()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=core, pad_token="[PAD]", eos_token="[EOS]", unk_token="[UNK]")
    tokenizer.chat_template = "{% for message in messages %}{{ message['role'] + ': ' + message['content'] + '\n' }}{% endfor %}assistant: "
    torch.manual_seed(80)
    config = Qwen2Config(vocab_size=len(vocabulary), hidden_size=16, intermediate_size=24,
                         num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                         max_position_embeddings=4096, eos_token_id=1, pad_token_id=0)
    config._attn_implementation = "eager"
    result = HFBackend.__new__(HFBackend)
    result.model, result.tokenizer, result.torch = Qwen2ForCausalLM(config).eval(), tokenizer, torch
    result.checkpoint_id = "base"
    return result


def record(view="direct"):
    return {"id": view, "world_id": "one-world", "view": view, "split": "development",
            "dataset": "clean", "cue_condition": "none", "cue_location": None,
            "story": "Alice sees the cup in the box." + (" A later move occurs." if view != "direct" else ""),
            "base_story": "Alice sees the cup in the box.", "question": "Where does Alice believe the cup is?",
            "choices": dict(zip("ABCDEF", ("box", "basket", "shelf", "drawer", "cabinet", "closet"))),
            "answer": "A"}


def test_native_answer_completion_scores_later_label_without_calling_it_single_token(backend, monkeypatch):
    force_fixture_trace(backend, monkeypatch, lambda prompt, cap: " \t A")
    output = list(evaluate_native(backend, [record()], reader="base", native_answer=True))
    assert output[0]["format"] == "native_answer"
    assert output[0]["prediction"] == "A" and output[0]["correct"]
    assert output[0]["answer_tokens"] > 1
    assert output[0]["probabilities"] == {}  # No false first-token probability for a multi-token answer.


def force_fixture_trace(backend, monkeypatch, choose_text):
    """Force fixture trace tokens through real HF generate/model forward calls."""
    original = backend.model.generate
    calls = []

    def generate(*args, **kwargs):
        prompt_ids = kwargs["input_ids"][0].tolist()
        prompt = backend.tokenizer.decode(prompt_ids, skip_special_tokens=False)
        calls.append({"prompt": prompt, "sample": kwargs.get("do_sample"), "budget": kwargs["max_new_tokens"]})
        text = choose_text(prompt, kwargs["max_new_tokens"])
        if text is not None:
            forced = backend.token_ids(text)
            offset = len(prompt_ids)

            class ForceFixture(LogitsProcessor):
                def __call__(self, input_ids, scores):
                    index = input_ids.shape[-1] - offset
                    scores.fill_(-float("inf"))
                    scores[:, forced[index] if index < len(forced) else backend.tokenizer.eos_token_id] = 0
                    return scores

            kwargs["logits_processor"] = LogitsProcessorList([ForceFixture()])
        return original(*args, **kwargs)

    monkeypatch.setattr(backend.model, "generate", generate)
    return calls


def test_exact_generation_continuation_replays_and_forced_delimiter_is_ineligible(backend, monkeypatch):
    force_fixture_trace(backend, monkeypatch, lambda prompt, cap: "Alice believes box.\nAnswer:" if cap > 1 else None)
    row = record()
    trace = generate_trace(backend, row, "base")
    assert trace["replay_eligible"] and not trace["forced_delimiter"]
    output = list(run_crossed_reader(backend, [row], [trace], "base", conditions=("own_replay",)))
    assert output[0]["replay_matches"] is True
    assert trace["answer_prefix_ids"] == backend.token_ids(trace["answer_prompt"])
    # EOS before any answer delimiter must not fabricate an ordinary replay.
    monkeypatch.undo()
    force_fixture_trace(backend, monkeypatch, lambda prompt, cap: "Alice believes box." if cap > 1 else None)
    trace = generate_trace(backend, row, "base")
    assert trace["forced_delimiter"] and not trace["replay_eligible"]
    assert trace["ordinary_answer_token"] is None
    output = list(run_crossed_reader(backend, [row], [trace], "base", conditions=("own_replay",)))
    assert output[0]["replay_matches"] is None


def reviewed_manifest(traces):
    manifest = prepare_edit_manifest(traces)
    by_artifact = {trace["artifact_id"]: trace for trace in traces}
    for contract in manifest["edits"]:
        trace = by_artifact[contract["artifact_id"]]
        text = trace["trace"]
        start, irrelevant = text.index("box"), text.index("waits")
        contract.update(reviewed_by="test reviewer", approved=True,
                        semantic={"start": start, "end": start + 3, "old": "box", "new": "basket",
                                  "fact_before": "Alice believes box", "fact_after": "Alice believes basket",
                                  "expected_answers": {"direct": "B", "observed_overwrite": "A"},
                                  "only_intended_fact_changed": True},
                        irrelevant={"start": irrelevant, "end": irrelevant + 5, "old": "waits", "new": "rests",
                                    "irrelevant_to_answers": True},
                        paraphrase={"text": "Alice thinks box and waits.", "meaning_preserved": True})
    return manifest


def test_direct_fixed_uses_native_question_and_holds_trace_fixed(backend, monkeypatch):
    calls = force_fixture_trace(backend, monkeypatch, lambda prompt, cap: "Alice believes box.\nAnswer:" if cap > 1 else None)
    rows = [record("observed_overwrite"), record("direct")]
    # Legacy alias now names the native direct question condition, never a new summary prompt.
    traces = list(collect_traces(backend, rows, "base", timing="prequestion"))
    assert len(calls) == 2  # Native trace, then its ordinary one-token continuation.
    assert "A later move occurs" not in calls[0]["prompt"]
    assert "Question:" in calls[0]["prompt"] and "<END_SUMMARY>" not in calls[0]["prompt"]
    assert traces[0]["trace"] == traces[1]["trace"]
    assert traces[0]["artifact_id"] == traces[1]["artifact_id"]
    assert traces[0]["trace_id"] != traces[1]["trace_id"]
    assert not traces[0]["replay_eligible"] and traces[1]["replay_eligible"]
    assert all(trace["question_timing"] == "direct_fixed" for trace in traces)
    output = list(run_crossed_reader(backend, rows, traces, "base", conditions=("original", "own_replay", "own_rerendered")))
    assert len(output) == 4  # Native direct question alone has ordinary replay claims.
    wrong = copy.deepcopy(traces)
    wrong[1]["trace"] += " Changed after seeing the question."
    with pytest.raises(ValueError, match="hash changed|remain fixed"):
        list(run_crossed_reader(backend, rows, wrong, "base", conditions=("original",)))
    with pytest.raises(ValueError, match="matching direct"):
        list(collect_traces(backend, [rows[0]], "base", timing="direct_fixed"))


def test_rerendered_replay_records_token_boundary_difference(backend, monkeypatch):
    force_fixture_trace(backend, monkeypatch, lambda prompt, cap: "Alice believes box. \nAnswer:" if cap > 1 else None)
    row = record()
    trace = generate_trace(backend, row, "base")
    assert trace["replay_eligible"] and trace["replay_token_ids_match"] is False
    original, rerendered = list(run_crossed_reader(backend, [row], [trace], "base",
                                                  conditions=("own_replay", "own_rerendered")))
    expected = backend.answer(render_prompt(backend.tokenizer, row, trace=trace["trace"]), row["choices"])
    assert original["raw_replay_matches"] is True
    assert rerendered["replay_mode"] == "rerendered_prompt"
    assert rerendered["token_id"] == expected.token_id
    assert rerendered["rerendered_replay_matches"] == (expected.token_id == trace["ordinary_answer_token"])
    assert rerendered["replay_token_ids_match"] is False


def test_source_regenerates_once_and_both_readers_receive_identical_frozen_text(backend, monkeypatch):
    def choose(prompt, budget):
        if budget == 1:
            return None
        if prompt.endswith("Reasoning:\n"):
            return "Alice believes box and waits.\nAnswer:"
        return " and waits.\nAnswer:"
    calls = force_fixture_trace(backend, monkeypatch, choose)
    rows = [record("observed_overwrite"), record("direct")]
    traces = list(collect_traces(backend, rows, "base", timing="direct_fixed"))
    manifest = reviewed_manifest(traces)
    assert len(manifest["edits"]) == 1  # Not eight duplicated human reviews.
    regenerated = list(regenerate_trace_artifacts(backend, rows, traces, "base", manifest))
    assert len(regenerated) == 1 and len(calls) == 3
    assert calls[-1]["sample"] is True
    assert "Question:" in calls[-1]["prompt"] and "A later move occurs" not in calls[-1]["prompt"]

    def reader_must_not_generate(*args, **kwargs):
        pytest.fail("A reader generated its own continuation instead of reading a frozen source trace")

    monkeypatch.setattr(backend, "generate", reader_must_not_generate)
    outputs = [list(run_crossed_reader(backend, rows, traces, reader, manifest,
                                      conditions=("semantic_minimal", "semantic_regenerated"),
                                      regenerated=regenerated)) for reader in ("base", "trained")]
    supplied = [row for output in outputs for row in output if row["condition"] == "semantic_regenerated"]
    assert len({row["supplied_trace"] for row in supplied}) == 1
    assert len({row["regeneration"]["artifact_sha256"] for row in supplied}) == 1
    assert all(row["regeneration"]["regenerated_by"] == "base" for row in supplied)
    assert all(row["regeneration"]["do_sample"] for row in supplied)
    with pytest.raises(ValueError, match="Missing frozen"):
        list(run_crossed_reader(backend, rows, traces, "trained", manifest, conditions=("semantic_regenerated",)))
    tampered = copy.deepcopy(regenerated)
    tampered[0]["trace"] += " altered"
    with pytest.raises(ValueError, match="hash changed"):
        list(run_crossed_reader(backend, rows, traces, "base", manifest,
                                conditions=("semantic_regenerated",), regenerated=tampered))
    backend.checkpoint_id = "other-checkpoint"
    with pytest.raises(ValueError, match="not the checkpoint"):
        list(regenerate_trace_artifacts(backend, rows, traces, "base", manifest))


def test_actual_seeded_sampling_is_reproducible_without_rng_side_effect(backend):
    prompt = render_prompt(backend.tokenizer, record(), reasoning=True)
    state = torch.random.get_rng_state().clone()
    a = backend.generate(prompt, max_new_tokens=8, do_sample=True, seed=17)
    assert torch.equal(state, torch.random.get_rng_state())
    b = backend.generate(prompt, max_new_tokens=8, do_sample=True, seed=17)
    assert a["generated_ids"] == b["generated_ids"]
    assert a["do_sample"] and a["temperature"] == .7


def test_cli_loads_local_tiny_checkpoint_and_runs_immediate_and_native_answers(backend, tmp_path):
    from paper2.cli import main
    from paper2.io import read_jsonl, write_jsonl

    model_dir = tmp_path / "tiny-model"
    backend.model.save_pretrained(model_dir)
    backend.tokenizer.save_pretrained(model_dir)
    input_file = tmp_path / "items.jsonl"
    write_jsonl(input_file, [record()])
    for format_name in ("immediate", "native_answer"):
        output = tmp_path / f"{format_name}.jsonl"
        assert main(["evaluate", "--model", str(model_dir), "--device", "cpu", "--reader", "base",
                     "--completion-protocol", "natural_answer_cap_v2",
                     "--input", str(input_file), "--output", str(output), "--format", format_name]) == 0
        result, = read_jsonl(output)
        assert result["format"] == format_name
        assert result["correct"] == (result["prediction"] == "A")
        if format_name == "native_answer":
            assert 1 <= result["answer_tokens"] <= 4
        assert output.with_suffix(".jsonl.run.json").is_file()
