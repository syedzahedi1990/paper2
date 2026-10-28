"""Real tiny-model tests; no pretrained downloads or GPU are required."""

import copy

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
from transformers import Qwen2Config, Qwen2ForCausalLM

from paper2.interventions import (
    ComponentInterventionEngine, EncodedInput, Site, compare_checkpoints,
    encode_input, score_consequence_pattern, signed_permutation_delta,
)


@pytest.fixture
def setup():
    torch.manual_seed(19)
    config = Qwen2Config(vocab_size=53, hidden_size=32, intermediate_size=48,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         attention_dropout=0.0, max_position_embeddings=64)
    config._attn_implementation = "eager"
    model = Qwen2ForCausalLM(config).eval()
    engine = ComponentInterventionEngine(model, "tiny-base")
    recipient = EncodedInput((1, 2, 3, 4, 5, 6), 2)
    donor = EncodedInput((1, 2, 9, 4, 5, 6), 2)
    sites = [Site(0, c) for c in ("k", "v", "residual")]
    return model, engine, recipient, donor, sites


def test_own_controls_exact_and_natural_components_have_real_effects(setup):
    model, engine, recipient, donor, sites = setup
    a, b = engine.capture(recipient, sites), engine.capture(donor, sites)
    assert a.payloads["0:k"].shape == (2, 8)  # KV heads, not four query heads.
    for site in sites:
        controls = engine.controls(recipient, a, b, [site], seed=7)
        assert torch.equal(controls["own"].next_token_logits, a.next_token_logits)
        assert not torch.equal(controls["natural"].next_token_logits, a.next_token_logits)
        receipt = controls["natural"].receipt
        assert receipt["analysis"] == "primary_within_checkpoint"
        assert receipt["sites"][0]["other_token_rows_exact"]
        assert receipt["sites"][0]["unselected_heads_exact"]
        assert not receipt["direct_query_modification"]
    assert not model.model.layers[0]._forward_hooks
    assert not model.model.layers[0].self_attn.k_proj._forward_hooks


def test_key_swap_preserves_native_rope_gqa_queries_and_values(setup, monkeypatch):
    import transformers.models.qwen2.modeling_qwen2 as native
    model, engine, recipient, donor, _ = setup
    site = Site(0, "k", (1,))
    a, b = engine.capture(recipient, [site]), engine.capture(donor, [site])
    seen = []
    original = native.eager_attention_forward

    def observe(module, query, key, value, *args, **kwargs):
        if module.layer_idx == 0:
            seen.append(tuple(t.detach().clone() for t in (query, key, value)))
        return original(module, query, key, value, *args, **kwargs)

    monkeypatch.setattr(native, "eager_attention_forward", observe)
    engine.capture(recipient, [site])
    result = engine.swap(recipient, a, b, [site])
    (q0, k0, v0), (q1, k1, v1) = seen
    assert q0.shape[1] == 4 and k0.shape[1] == 2
    assert torch.equal(q0, q1) and torch.equal(v0, v1)
    assert torch.equal(k0[:, 0], k1[:, 0])
    assert torch.equal(k0[:, :, :2], k1[:, :, :2])
    assert torch.equal(k0[:, :, 3:], k1[:, :, 3:])
    assert not torch.equal(k0[:, 1, 2], k1[:, 1, 2])
    # The native path rotates the donor pre-RoPE key at the recipient position.
    with torch.inference_mode():
        cos, sin = model.model.rotary_emb(torch.zeros(1, 6, 32), torch.arange(6).unsqueeze(0))
        unrotated = b.payloads[site.key][1]
        expected = unrotated * cos[0, 2] + native.rotate_half(unrotated) * sin[0, 2]
    assert torch.equal(k1[0, 1, 2], expected)
    assert result.receipt["key_site"] == "post_bias_pre_native_RoPE"


def test_values_leave_keys_queries_unchanged_at_selected_layer(setup, monkeypatch):
    import transformers.models.qwen2.modeling_qwen2 as native
    _, engine, recipient, donor, _ = setup
    site = Site(0, "v", (0,))
    a, b = engine.capture(recipient, [site]), engine.capture(donor, [site])
    seen, original = [], native.eager_attention_forward

    def observe(module, query, key, value, *args, **kwargs):
        if module.layer_idx == 0:
            seen.append(tuple(t.detach().clone() for t in (query, key, value)))
        return original(module, query, key, value, *args, **kwargs)

    monkeypatch.setattr(native, "eager_attention_forward", observe)
    engine.capture(recipient, [site])
    engine.swap(recipient, a, b, [site])
    (q0, k0, v0), (q1, k1, v1) = seen
    assert torch.equal(q0, q1) and torch.equal(k0, k1)
    assert torch.equal(v0[:, 1], v1[:, 1])
    assert torch.equal(v1[0, 0, 2], b.payloads["0:v"][0])


def test_signed_null_preserves_displacement_norm_per_head_and_is_reproducible():
    r = torch.arange(24, dtype=torch.float32).reshape(3, 8)
    d = r + torch.arange(1, 25).reshape(3, 8)
    a, evidence = signed_permutation_delta(r, d, seed=57)
    b, evidence2 = signed_permutation_delta(r, d, seed=57)
    assert torch.equal(a, b) and evidence == evidence2
    assert torch.allclose((a - r).norm(dim=-1), (d - r).norm(dim=-1))
    assert not torch.equal(a, d)
    assert evidence["rounding_norm"] == [0.0, 0.0, 0.0]
    # It is the displacement that is randomized, not donor absolute vectors.
    shifted, _ = signed_permutation_delta(r + 100, d + 100, seed=57)
    assert torch.equal(shifted, a + 100)


def test_cross_model_rejected_unless_secondary_and_prefix_identical(setup):
    model, engine, recipient, donor, sites = setup
    a = engine.capture(recipient, sites)
    other = ComponentInterventionEngine(copy.deepcopy(model), "tiny-trained")
    same = other.capture(recipient, sites)
    different = other.capture(donor, sites)
    with pytest.raises(ValueError, match="explicit secondary"):
        engine.swap(recipient, a, same, sites)
    with pytest.raises(ValueError, match="identical token prefixes"):
        engine.swap(recipient, a, different, sites, secondary_cross_model=True)
    result = engine.swap(recipient, a, same, sites, secondary_cross_model=True)
    assert result.receipt["analysis"] == "secondary_cross_model"
    assert torch.equal(result.next_token_logits, a.next_token_logits)


def test_capture_mutation_checkpoint_change_and_unmatched_donor_rejected(setup):
    model, engine, recipient, donor, sites = setup
    a, b = engine.capture(recipient, sites), engine.capture(donor, sites)
    malformed = engine.capture(EncodedInput((1, 7, 9, 4, 5, 6), 2), sites)
    with pytest.raises(ValueError, match="outside the critical token"):
        engine.swap(recipient, a, malformed, sites)
    # Captures are checksummed even though tensor objects cannot be frozen.
    with torch.inference_mode():
        b.payloads["0:k"][0, 0] += 1
    with pytest.raises(ValueError, match="mutated"):
        engine.swap(recipient, a, b, sites)
    with torch.no_grad():
        next(model.parameters()).add_(0.01)
    with pytest.raises(ValueError, match="parameters changed"):
        engine.swap(recipient, a, a, sites, control="own")


def test_hooks_removed_after_failure_and_invalid_sites_fail_closed(setup, monkeypatch):
    model, engine, recipient, _, sites = setup
    original = model.forward

    def fail(**kwargs):
        raise RuntimeError("injected native failure")

    monkeypatch.setattr(model, "forward", fail)
    with pytest.raises(RuntimeError, match="injected"):
        engine.capture(recipient, sites)
    assert not engine._active
    for site in sites:
        assert not engine._module(site)._forward_hooks
    monkeypatch.setattr(model, "forward", original)
    engine.capture(recipient, sites)
    with pytest.raises(ValueError, match="KV heads"):
        engine.capture(recipient, [Site(0, "k", (2,))])
    model.train()
    with pytest.raises(ValueError, match="eval"):
        engine.capture(recipient, sites)


def test_prefix_coordinates_and_single_critical_token_validation():
    class Tokenizer:
        def __call__(self, text, **kwargs):
            assert kwargs == {"add_special_tokens": False, "return_offsets_mapping": True}
            return {"input_ids": [1, 2, 3], "offset_mapping": [(0, 3), (3, 7), (7, 10)]}

    result = encode_input(Tokenizer(), "aaa boxbbb", (4, 7), prefix_stop_char=7)
    assert result.critical_position == 1 and result.prefix_stop == 2
    with pytest.raises(ValueError, match="exactly one"):
        encode_input(Tokenizer(), "aaa boxbbb", (2, 7))
    with pytest.raises(ValueError, match="additional"):
        encode_input(Tokenizer(), "aaa boxbbb", (4, 6))
    with pytest.raises(ValueError, match="boundary"):
        encode_input(Tokenizer(), "aaa boxbbb", (4, 7), prefix_stop_char=9)


def test_swap_forecasts_preserve_intact_views_and_do_not_use_ordinary_targets():
    ordinary = {"direct": "A", "world": "A", "other_agent": "B", "search": "A",
                "observed_overwrite": "C", "unobserved_overwrite": "A", "irrelevant_object": "D", "report": "A"}
    expected = {v: ("F" if v in ("direct", "world", "search", "unobserved_overwrite", "report") else a)
                for v, a in ordinary.items()}
    forecasts = {v: {"event_rewrite": a} for v, a in expected.items()}
    scored = score_consequence_pattern(expected, ordinary, forecasts)
    assert scored["joint_correct"] and scored["changed_accuracy"] == scored["intact_accuracy"] == 1
    assert scored["ordinary_accuracy"] == 3 / 8
    bad = dict(expected, irrelevant_object="F")
    assert not score_consequence_pattern(bad, ordinary, forecasts)["joint_correct"]
    with pytest.raises(ValueError, match="complete view"):
        score_consequence_pattern({"direct": "F"}, ordinary, forecasts)


def test_before_after_comparison_subtracts_each_models_own_control():
    common = {"pair_id": "pair", "view": "direct", "layer": 0, "component": "k", "forecast_target": "B"}
    before = [dict(common, control="own", forecast_probability=.1),
              dict(common, control="natural", forecast_probability=.3)]
    after = [dict(common, control="own", forecast_probability=.5),
             dict(common, control="natural", forecast_probability=.9)]
    row, = compare_checkpoints(before, after)
    assert row["before_effect"] == pytest.approx(.2)
    assert row["after_effect"] == pytest.approx(.4)
    assert row["training_change_in_effect"] == pytest.approx(.2)
