"""Native Mistral Small geometry, with random offline weights and no GPU."""
import json
import random

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from transformers import MistralConfig, MistralForCausalLM
import transformers.models.mistral.modeling_mistral as native

from paper2.interventions import ComponentInterventionEngine, EncodedInput, Site
from paper2.interventions.controls import residual_control_eligibility, residual_control_summary
from paper2.interventions.runner import intervention_masks, run_component_test


def mistral_model(vocab_size=53, num_layers=3, attention="eager"):
    torch.manual_seed(19)
    config = MistralConfig(vocab_size=vocab_size, hidden_size=40, intermediate_size=64,
                           num_hidden_layers=num_layers, num_attention_heads=4,
                           num_key_value_heads=2, head_dim=8, attention_dropout=0.0,
                           max_position_embeddings=4096, sliding_window=None,
                           rope_theta=100000000.0, eos_token_id=1, pad_token_id=0)
    config._attn_implementation = attention
    return MistralForCausalLM(config).eval()


@pytest.fixture
def setup():
    torch.set_num_threads(1)
    model = mistral_model()
    engine = ComponentInterventionEngine(model, "random-mistral-only")
    recipient = EncodedInput((1, 2, 3, 4, 5, 6), 2)
    donor = EncodedInput((1, 2, 9, 4, 5, 6), 2)
    return model, engine, recipient, donor


@pytest.mark.parametrize("attention", ["eager", "sdpa"])
def test_nonsquare_mistral_own_and_signed_null_native_forwards(attention):
    model = mistral_model(attention=attention)
    engine = ComponentInterventionEngine(model, "random-mistral-only")
    recipient, donor = EncodedInput((1, 2, 3, 4, 5, 6), 2), EncodedInput((1, 2, 9, 4, 5, 6), 2)
    assert model.model.layers[0].self_attn.q_proj.weight.shape == (32, 40)
    assert model.model.layers[0].self_attn.o_proj.weight.shape == (40, 32)
    assert engine.geometry == (3, 40, 2, 8)
    sites = [Site(layer, component) for layer in (1, 2) for component in ("k", "v", "residual")]
    a, b = engine.capture(recipient, sites), engine.capture(donor, sites)
    assert a.payloads["1:k"].shape == (2, 8) and a.payloads["1:residual"].shape == (40,)
    for component in ("k", "v", "residual"):
        selected = [site for site in sites if site.component == component]
        controls = engine.controls(recipient, a, b, selected, seed=17)
        assert torch.equal(controls["own"].next_token_logits, a.next_token_logits)
        assert not torch.equal(controls["natural"].next_token_logits, a.next_token_logits)
        repeat = engine.swap(recipient, a, b, selected, control="signed_permutation", seed=17)
        assert torch.equal(repeat.next_token_logits, controls["signed_permutation"].next_token_logits)
        for evidence in repeat.receipt["nulls"].values():
            assert evidence["original_delta_norm"] == pytest.approx(evidence["rotated_delta_norm"])
        assert all(site["other_token_rows_exact"] for site in repeat.receipt["sites"])
    assert all(not engine._module(site)._forward_hooks for site in sites)


@pytest.mark.parametrize("component", ["k", "v"])
def test_mistral_pre_rope_and_pre_gqa_with_selected_head(setup, monkeypatch, component):
    model, engine, recipient, donor = setup
    site = Site(1, component, (1,))
    a, b = engine.capture(recipient, [site]), engine.capture(donor, [site])
    seen, original = [], native.eager_attention_forward

    def observe(module, query, key, value, *args, **kwargs):
        if module.layer_idx == 1:
            seen.append(tuple(t.detach().clone() for t in (query, key, value)))
        return original(module, query, key, value, *args, **kwargs)

    monkeypatch.setattr(native, "eager_attention_forward", observe)
    engine.capture(recipient, [site])
    engine.swap(recipient, a, b, [site])
    (q0, k0, v0), (q1, k1, v1) = seen
    assert q0.shape[1] == 4 and k0.shape[1] == 2
    assert torch.equal(q0, q1)
    changed0, changed1 = (k0, k1) if component == "k" else (v0, v1)
    assert torch.equal(v0, v1) if component == "k" else torch.equal(k0, k1)
    assert torch.equal(changed0[:, 0], changed1[:, 0])
    assert torch.equal(changed0[:, :, :2], changed1[:, :, :2])
    assert torch.equal(changed0[:, :, 3:], changed1[:, :, 3:])
    expected = b.payloads[site.key][1]
    if component == "k":
        with torch.inference_mode():
            cos, sin = model.model.rotary_emb(torch.zeros(1, 6, 40), torch.arange(6).unsqueeze(0))
            expected = expected * cos[0, 2] + native.rotate_half(expected) * sin[0, 2]
    assert torch.equal(changed1[0, 1, 2], expected)


def test_mistral_residual_is_block_output_and_not_input_reconstruction(setup):
    _, engine, recipient, donor = setup
    sites = [Site(layer, "residual") for layer in range(3)] + [Site(0, "k"), Site(0, "v")]
    a, b = engine.capture(recipient, sites), engine.capture(donor, sites)
    band = [Site(layer, "residual") for layer in (1, 2)]
    result = engine.swap(recipient, a, b, band)
    assert not torch.equal(result.next_token_logits, a.next_token_logits)
    assert not torch.equal(result.next_token_logits, b.next_token_logits)
    last_only = engine.swap(recipient, a, b, [Site(2, "residual")])
    assert torch.equal(last_only.next_token_logits, a.next_token_logits)
    all_residuals = engine.swap(recipient, a, b, sites[:3])
    assert not torch.equal(all_residuals.next_token_logits, b.next_token_logits)
    # Layer-zero K/V were computed before the residual output was replaced.
    reconstructed = engine.swap(recipient, a, b, sites)
    assert torch.equal(reconstructed.next_token_logits, b.next_token_logits)


@pytest.mark.parametrize("projection", ["q_proj", "k_proj", "v_proj", "o_proj"])
def test_mistral_geometry_checks_actual_projection_dimensions(projection):
    model = mistral_model()
    old = getattr(model.model.layers[1].self_attn, projection)
    setattr(model.model.layers[1].self_attn, projection,
            torch.nn.Linear(old.in_features, old.out_features + 1, bias=False))
    with pytest.raises(ValueError, match="Native projection geometry"):
        ComponentInterventionEngine(model, "malformed")


def test_mistral_rejects_unmatched_donor_parameter_changes_and_cleans_failed_hooks(setup, monkeypatch):
    model, engine, recipient, donor = setup
    sites = [Site(1, component) for component in ("k", "v", "residual")]
    a, b = engine.capture(recipient, sites), engine.capture(donor, sites)
    invalid = engine.capture(EncodedInput((1, 7, 9, 4, 5, 6), 2), sites)
    with pytest.raises(ValueError, match="outside the critical token"):
        engine.swap(recipient, a, invalid, sites)
    original = model.forward

    def fail(*args, **kwargs):
        raise RuntimeError("native failure")

    monkeypatch.setattr(model, "forward", fail)
    with pytest.raises(RuntimeError, match="native failure"):
        engine.swap(recipient, a, b, sites)
    assert not engine._active and all(not engine._module(site)._forward_hooks for site in sites)
    monkeypatch.setattr(model, "forward", original)
    with torch.no_grad():
        model.model.layers[1].self_attn.k_proj.weight.add_(.01)
    with pytest.raises(ValueError, match="parameters changed"):
        engine.swap(recipient, a, b, sites)


def test_mistral_40_layer_policy_is_one_joint_band_plus_descriptive_layers():
    sites = [Site(layer, component) for layer in range(1, 40) for component in ("k", "v", "residual")]
    masks = intervention_masks(sites, mode="both", g4_test_layers=list(range(1, 40)))
    joint = [(mask, component) for mask, component, _ in masks if len(mask) > 1]
    assert joint == [(tuple(range(1, 40)), component) for component in ("k", "v", "residual")]
    assert len(masks) == 3 * (1 + 39)
    assert residual_control_eligibility(list(range(1, 40)), 40, test_layers=list(range(1, 40)))[0]
    assert not residual_control_eligibility(list(range(40)), 40, test_layers=list(range(1, 40)))[0]


@pytest.mark.parametrize("event_kind", ["move", "tell"])
def test_mistral_complete_real_world_component_panels(event_kind):
    from paper2.data import generate_world, matched_pair
    from test_interventions_runner import tiny_backend
    pair = matched_pair(generate_world(random.Random(13), event_kind=event_kind, distractor_events=0))
    backend = tiny_backend(pair)
    backend.model = mistral_model(vocab_size=len(backend.tokenizer))
    sites = [Site(layer, component) for layer in (1, 2) for component in ("k", "v", "residual")]
    rows = run_component_test(backend, [pair], sites, seed=8, g4_test_layers=[1, 2])
    assert len(rows) == 8 * (2 + 3 * 3)
    json.dumps(rows, allow_nan=False)
    for row in rows:
        assert row["forecast_target"] == pair["forecasts"][row["view"]]["event_rewrite"]
        if row["control"] == "own":
            assert row["prediction"] == row["baseline_prediction"]
    summary = residual_control_summary(rows)
    assert summary["eligible_masks"] == ["1,2"]
    assert not summary["empirical_gate_passed"]


def test_mistral_peft_capture_includes_active_adapter_and_rejects_changed_adapter(setup):
    pytest.importorskip("peft")
    from peft import LoraConfig, get_peft_model
    model, _, recipient, _ = setup
    config = LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                                        "gate_proj", "up_proj", "down_proj"],
                        task_type="CAUSAL_LM", lora_dropout=0)
    model = get_peft_model(model, config).eval()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                parameter.fill_(.02)
    engine = ComponentInterventionEngine(model, "active-lora")
    site = Site(1, "k")
    active = engine.capture(recipient, [site])
    with model.disable_adapter():
        disabled = engine.capture(recipient, [site])
        with pytest.raises(ValueError, match="parameters changed"):
            engine.swap(recipient, active, active, [site], control="own")
    assert not torch.equal(active.payloads[site.key], disabled.payloads[site.key])
    assert torch.equal(engine.swap(recipient, active, active, [site], control="own").next_token_logits,
                       active.next_token_logits)
    model.add_adapter("other", config)
    model.set_adapter("other")
    model.eval()
    with pytest.raises(ValueError, match="parameters changed"):
        engine.swap(recipient, active, active, [site], control="own")
