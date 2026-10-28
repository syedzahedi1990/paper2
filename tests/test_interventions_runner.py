"""End-to-end Test 3 on real generated worlds with a tiny offline tokenizer/model."""

import json
import random
import copy
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

from paper2.backend import HFBackend
from paper2.data import generate_world, matched_pair
from paper2.interventions import Site, run_component_test, score_consequence_pattern


def tiny_backend(pair, num_layers=2):
    vocabulary = {"[UNK]": 0, "[PAD]": 1}
    split = pre_tokenizers.Whitespace()
    # Build an offline vocabulary from the complete local fixture's characters.
    for row in pair["recipient"] + pair["donor"]:
        for token, _ in split.pre_tokenize_str(row["story"] + " " + row["question"] + " A B C D E F Answer:"):
            vocabulary.setdefault(token, len(vocabulary))
    core = Tokenizer(models.WordLevel(vocabulary, unk_token="[UNK]"))
    core.pre_tokenizer = split
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=core, unk_token="[UNK]", pad_token="[PAD]")
    tokenizer.chat_template = "{% for message in messages %}{{ message['role'] + ': ' + message['content'] + '\n' }}{% endfor %}assistant: "
    torch.manual_seed(0)
    config = Qwen2Config(vocab_size=len(vocabulary), hidden_size=16, intermediate_size=24,
                         num_hidden_layers=num_layers, num_attention_heads=2, num_key_value_heads=1,
                         max_position_embeddings=2048)
    config._attn_implementation = "eager"
    backend = HFBackend.__new__(HFBackend)
    backend.model = Qwen2ForCausalLM(config).eval()
    backend.torch, backend.tokenizer, backend.checkpoint_id = torch, tokenizer, "random-test-only"
    return backend


@pytest.mark.parametrize("event_kind", ["move", "tell"])
def test_real_pairs_run_with_forecasts_and_serializable_receipts(event_kind, monkeypatch):
    pair = matched_pair(generate_world(random.Random(13), event_kind=event_kind, distractor_events=0))
    backend = tiny_backend(pair)
    sites = [Site(layer, component) for layer in (0, 1) for component in ("k", "v", "residual")]
    calls = []
    original = backend.model.forward
    def count_forward(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(backend.model, "forward", count_forward)
    rows = run_component_test(backend, [pair], sites, seed=8)
    assert len(rows) == 8 * (2 + 3 * 3)
    # Two captures and nine swaps per view: each mask is ONE native forward,
    # rather than averaging separate single-layer runs.
    assert len(calls) == 8 * (2 + 3 * 3)
    json.dumps(rows, allow_nan=False)
    assert not any(row["positive_control"] for row in rows)
    assert {row["g4_control_reason"] for row in rows if row["component"] == "residual"} == {
        "includes_layer_zero_input_reconstruction_risk"}
    for row in rows:
        assert row["forecast_target"] == pair["forecasts"][row["view"]]["event_rewrite"]
        if row["control"] == "own":
            assert row["prediction"] == row["baseline_prediction"]
        if row["receipt"]:
            assert row["receipt"]["analysis"] == "primary_within_checkpoint"
            assert row['layers'] == [0, 1] and row['mask_id'] == '0,1'
            assert [site['layer'] for site in row['receipt']['sites']] == [0, 1]
            assert {site['component'] for site in row['receipt']['sites']} == {row['component']}
            assert all(site['other_token_rows_exact'] for site in row['receipt']['sites'])
    baseline = {row["view"]: row["ordinary_target"] for row in rows if row["control"] == "baseline"}
    predictions = {row["view"]: row["prediction"] for row in rows if row["component"] == "k" and row["control"] == "natural"}
    score = score_consequence_pattern(predictions, baseline, pair["forecasts"])
    assert isinstance(score["joint_correct"], bool)
    if event_kind == "tell":
        assert "world" in score["intact_views"]
    assert set(score['changed_views']) == {view for view, changed in pair['event_rewrite_change_mask'].items() if changed}
    with pytest.raises(ValueError, match="positive-control"):
        run_component_test(backend, [pair], [Site(0, "k")])


def test_joint_and_single_masks_preserve_head_selection():
    from paper2.interventions.runner import intervention_masks
    sites = [Site(layer, component, (0,) if component != 'residual' else None)
             for layer in (0, 1) for component in ('k', 'v', 'residual')]
    masks = intervention_masks(sites, 'both')
    assert len(masks) == 9
    assert {mask for mask, _, _ in masks} == {(0, 1), (0,), (1,)}
    assert all(site.heads == (0,) for _, component, selected in masks if component != 'residual' for site in selected)
    with pytest.raises(ValueError, match='repeat'):
        intervention_masks(sites + [sites[0]])


def test_explicit_nonzero_residual_control_keeps_full_depth_key_value_masks():
    from paper2.interventions.controls import residual_control_summary
    pair = matched_pair(generate_world(random.Random(13), distractor_events=0))
    backend = tiny_backend(pair, num_layers=3)
    sites = [Site(layer, component) for layer in range(3) for component in ("k", "v", "residual")]
    rows = run_component_test(backend, [pair], sites, residual_control_layers=[1])
    assert len(rows) == 8 * (2 + 4 * 3)
    assert {tuple(row["layers"]) for row in rows if row["component"] in ("k", "v")} == {(0, 1, 2)}
    controls = [row for row in rows if row["positive_control"]]
    assert len(controls) == 8 and all(row["layers"] == [1] for row in controls)
    assert residual_control_summary(rows)["eligible_masks"] == ["1"]
    assert not residual_control_summary(rows)["empirical_gate_passed"]
    with pytest.raises(ValueError, match="greater than zero"):
        run_component_test(backend, [pair], sites, residual_control_layers=[0, 1])
    with pytest.raises(ValueError, match="no_downstream_attention"):
        run_component_test(backend, [pair], sites, residual_control_layers=[2])


def test_nonzero_residual_is_real_intervention_without_donor_reconstruction():
    from paper2.interventions import ComponentInterventionEngine, encode_input
    from paper2.prompts import render_prompt
    pair = matched_pair(generate_world(random.Random(13), distractor_events=0))
    backend = tiny_backend(pair, num_layers=3)
    inputs = []
    for row in (pair["recipient"][0], pair["donor"][0]):
        prompt = render_prompt(backend.tokenizer, row, reasoning=False)
        origin = prompt.index(row["story"])
        a, b = row["critical_value_span"]
        inputs.append(encode_input(backend.tokenizer, prompt, (origin + a, origin + b)))
    engine = ComponentInterventionEngine(backend.model, backend.checkpoint_id)
    sites = [Site(layer, "residual") for layer in range(3)]
    recipient, donor = [engine.capture(value, sites) for value in inputs]
    natural = engine.swap(inputs[0], recipient, donor, [Site(1, "residual")])
    assert not torch.equal(natural.next_token_logits, recipient.next_token_logits)
    assert not torch.equal(natural.next_token_logits, donor.next_token_logits)
    # These are block-output hooks. Even an all-layer patch need not exactly
    # reconstruct the donor: noncritical rows retain earlier recipient effects.
    # It is nevertheless conservatively excluded from empirical G4 evidence.
    reconstruction_risk = engine.swap(inputs[0], recipient, donor, sites)
    assert not torch.equal(reconstruction_risk.next_token_logits, donor.next_token_logits)
    own = engine.swap(inputs[0], recipient, donor, [Site(1, "residual")], control="own")
    assert torch.equal(own.next_token_logits, recipient.next_token_logits)
    disconnected = engine.swap(inputs[0], recipient, donor, [Site(2, "residual")])
    assert torch.equal(disconnected.next_token_logits, recipient.next_token_logits)


def test_cross_checkpoint_runner_requires_identical_complete_tokens():
    pair = matched_pair(generate_world(random.Random(17), distractor_events=0))
    backend = tiny_backend(pair)
    donor = copy.copy(backend)
    donor.model = copy.deepcopy(backend.model)
    donor.checkpoint_id = 'random-trained-test-only'
    with torch.no_grad():
        next(donor.model.parameters()).add_(.01)
    sites = [Site(layer, component) for layer in (0, 1) for component in ('k', 'v', 'residual')]
    rows = run_component_test(backend, [pair], sites, donor_backend=donor)
    assert all(row['analysis'] == 'secondary_cross_model' for row in rows)
    assert all(not row['positive_control'] and not row['change_required'] for row in rows)
    for row in rows:
        assert row['forecast_target'] == row['ordinary_target']
        if row['receipt'] and row['control'] != 'own':
            receipt = row['receipt']
            assert receipt['analysis'] == 'secondary_cross_model'
            assert receipt['recipient_prefix_sha256'] == receipt['donor_prefix_sha256']
    donor.tokenizer = copy.deepcopy(backend.tokenizer)
    donor.tokenizer.chat_template += ' A'
    with pytest.raises(ValueError, match='identical complete token inputs'):
        run_component_test(backend, [pair], sites, donor_backend=donor)


def test_cross_checkpoint_runner_checks_each_tokenizers_answer_alphabet(monkeypatch):
    import paper2.prompts as prompts
    pair = matched_pair(generate_world(random.Random(17), distractor_events=0))
    backend = tiny_backend(pair)
    donor = copy.copy(backend)
    donor.model = copy.deepcopy(backend.model)
    donor.tokenizer = copy.deepcopy(backend.tokenizer)
    donor.checkpoint_id = 'other-tokenizer'
    original = prompts.answer_token_ids
    def different_alphabet(tokenizer, prompt, choices):
        result = original(tokenizer, prompt, choices)
        if tokenizer is donor.tokenizer:
            result['A'], result['B'] = result['B'], result['A']
        return result
    monkeypatch.setattr(prompts, 'answer_token_ids', different_alphabet)
    sites = [Site(0, component) for component in ('k', 'v', 'residual')]
    with pytest.raises(ValueError, match='Answer-token IDs changed'):
        run_component_test(backend, [pair], sites, donor_backend=donor)


def test_component_cli_exposes_secondary_donor_and_preserves_joint_mask(tmp_path, monkeypatch):
    import paper2.cli as cli
    import paper2.backend as backend_module
    pair = matched_pair(generate_world(random.Random(17), distractor_events=0))
    backend = tiny_backend(pair)
    donor = copy.copy(backend)
    donor.model = copy.deepcopy(backend.model)
    donor.checkpoint_id = 'secondary-donor'
    monkeypatch.setattr(cli, '_backend', lambda args: backend)
    requests = []
    def load_donor(model, adapter, **kwargs):
        requests.append((model, adapter))
        return donor
    monkeypatch.setattr(backend_module, 'HFBackend', load_donor)
    source = tmp_path / 'pairs.jsonl'
    source.write_text(json.dumps(pair) + '\n')
    output = tmp_path / 'components.jsonl'
    assert cli.main(['components', '--model', 'tiny-recipient', '--donor-model', 'tiny-donor',
                     '--input', str(source), '--output', str(output), '--layers', '0', '1']) == 0
    assert requests == [('tiny-donor', None)]
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert all(row['analysis'] == 'secondary_cross_model' for row in rows)
    assert {tuple(row['layers']) for row in rows if row['receipt']} == {(0, 1)}
    assert json.loads(Path(str(output) + '.run.json').read_text())['status'] == 'completed'
