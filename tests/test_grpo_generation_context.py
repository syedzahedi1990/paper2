"""Pinned GRPO must retain prompt context with gradient checkpointing enabled."""
import functools

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from transformers import MistralConfig, MistralForCausalLM, Qwen2Config, Qwen2ForCausalLM

from paper2.training.runner import _rollout_generation_mode


def model_fixture(family, vocab_size=29):
    torch.set_num_threads(1)
    torch.manual_seed(123)
    config_class, model_class = ((MistralConfig, MistralForCausalLM) if family == "mistral"
                                else (Qwen2Config, Qwen2ForCausalLM))
    extra = {"head_dim": 8, "sliding_window": None} if family == "mistral" else {}
    config = config_class(vocab_size=vocab_size, hidden_size=40 if family == "mistral" else 32,
                          intermediate_size=48, num_hidden_layers=2, num_attention_heads=4,
                          num_key_value_heads=2, attention_dropout=0, pad_token_id=0,
                          eos_token_id=None, **extra)
    config._attn_implementation = "eager"
    model = model_class(config)
    model.config._name_or_path = "paper2-random-tiny-" + family
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model.train()


def checkpoint_flags(model):
    return tuple((name, module.gradient_checkpointing) for name, module in model.named_modules()
                 if hasattr(module, "gradient_checkpointing"))


def generate_with_observations(model):
    observations = []

    def before(module, args, kwargs):
        cache = kwargs.get("past_key_values")
        observations.append((kwargs["input_ids"].shape[1], None if cache is None else cache.get_seq_length(),
                             module.training))

    handle = model.register_forward_pre_hook(before, with_kwargs=True)
    prompt = torch.tensor([[2, 3, 4, 5]])
    try:
        result = model.generate(prompt, attention_mask=torch.ones_like(prompt), use_cache=True,
                                max_new_tokens=4, do_sample=False, return_dict_in_generate=True,
                                output_logits=True)
    finally:
        handle.remove()
    return result, observations


@pytest.mark.parametrize("family", ["mistral", "qwen2"])
def test_pinned_training_mode_cache_bug_reproduces_context_loss(family):
    # This deliberately reproduces the old pinned-library path. no_grad() is
    # already supplied by generate; it does not disable checkpointed-layer logic.
    model = model_fixture(family)
    result, observations = generate_with_observations(model)
    assert observations == [(4, 0, True), (1, 0, True), (1, 0, True), (1, 0, True)]
    model.eval()
    with torch.no_grad():
        isolated = model(input_ids=result.sequences[:, 4:5], position_ids=torch.tensor([[4]]),
                         use_cache=False).logits[:, -1]
        full_prefix = model(input_ids=result.sequences[:, :5], use_cache=False).logits[:, -1]
    assert torch.equal(result.logits[1], isolated)
    assert not torch.allclose(result.logits[1], full_prefix, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("family", ["mistral", "qwen2"])
def test_rollout_context_uses_incremental_cache_and_full_prefix(family):
    model = model_fixture(family)
    model.model.layers[0].mlp.eval()  # Exact restoration includes mixed module modes.
    modes = tuple(module.training for module in model.modules())
    flags = checkpoint_flags(model)
    with _rollout_generation_mode(model):
        assert not any(module.training for module in model.modules())
        assert checkpoint_flags(model) == flags
        result, observations = generate_with_observations(model)
        with torch.no_grad():
            full_prefix = model(input_ids=result.sequences[:, :5], use_cache=False).logits[:, -1]
    assert observations == [(4, 0, False), (1, 4, False), (1, 5, False), (1, 6, False)]
    torch.testing.assert_close(result.logits[1], full_prefix, atol=1e-6, rtol=1e-6)
    assert tuple(module.training for module in model.modules()) == modes
    assert checkpoint_flags(model) == flags and model.is_gradient_checkpointing
    assert model.config.use_cache is False


@pytest.mark.parametrize("family", ["mistral", "qwen2"])
def test_rollout_context_restores_modes_and_checkpointing_after_exception(family):
    model = model_fixture(family)
    model.model.layers[0].mlp.eval()
    modes = tuple(module.training for module in model.modules())
    flags = checkpoint_flags(model)
    with pytest.raises(RuntimeError, match="injected generation failure"):
        with _rollout_generation_mode(model):
            assert not model.training and model.is_gradient_checkpointing
            raise RuntimeError("injected generation failure")
    assert tuple(module.training for module in model.modules()) == modes
    assert checkpoint_flags(model) == flags


@pytest.mark.integration
@pytest.mark.parametrize("family", ["mistral", "qwen2"])
@pytest.mark.parametrize("fail", [False, True])
def test_real_grpo_checkpointed_rollout_modes_cache_and_forced_delimiter(tmp_path, family, fail):
    pytest.importorskip("trl")
    from transformers import LogitsProcessorList
    from paper2.training.config import TrainingConfig
    from paper2.training.runner import run_training
    from test_training_integration import tiny_fixture, rows

    _, tokenizer = tiny_fixture()
    model = model_fixture(family, vocab_size=len(tokenizer))
    model.config.eos_token_id = model.generation_config.eos_token_id = tokenizer.eos_token_id
    delimiter_length = len(tokenizer.encode("\nAnswer: ", add_special_tokens=False))
    original_generate, original_forward = model.generate, model.forward
    observations, loss_masks = [], []
    generating = False

    class LongTraceGrammar:
        def __init__(self):
            self.prefix = None

        def __call__(self, input_ids, scores):
            if self.prefix is None:
                self.prefix = input_ids.shape[1]
            step = input_ids.shape[1] - self.prefix
            if 384 <= step < 384 + delimiter_length:
                return scores  # The real production cap supplies this delimiter.
            allowed = [10] if step < 384 else [3, 4] if step == 384 + delimiter_length else [1]
            masked = torch.full_like(scores, float("-inf"))
            masked[:, allowed] = scores[:, allowed]
            return masked

    def constrained_generate(*args, **kwargs):
        nonlocal generating
        assert not model.training and model.is_gradient_checkpointing
        assert not any(module.training for module in model.modules())
        if fail:
            raise RuntimeError("injected real rollout failure")
        kwargs["logits_processor"] = LogitsProcessorList(
            list(kwargs.get("logits_processor", [])) + [LongTraceGrammar()])
        generating = True
        try:
            return original_generate(*args, **kwargs)
        finally:
            generating = False

    @functools.wraps(original_forward)
    def forward(*args, **kwargs):
        if generating:
            cache = kwargs.get("past_key_values")
            observations.append((kwargs["input_ids"].shape[1], cache.get_seq_length(), model.training))
        if torch.is_grad_enabled() and kwargs.get("attention_mask") is not None:
            assert model.training and model.is_gradient_checkpointing
            loss_masks.append(kwargs["attention_mask"].detach().clone())
        return original_forward(*args, **kwargs)

    model.generate, model.forward = constrained_generate, forward
    config = TrainingConfig(arm="reasoning_rl", model=model.config._name_or_path, max_steps=1,
                            bf16=False, gradient_checkpointing=True, save_steps=1, eval_steps=1,
                            lora_rank=2, lora_alpha=4, beta=0.0, warmup_ratio=0.0)
    if fail:
        with pytest.raises(RuntimeError, match="injected real rollout failure"):
            run_training(config, rows(), tmp_path, synthetic_model=model, synthetic_tokenizer=tokenizer)
    else:
        report = run_training(config, rows(), tmp_path, synthetic_model=model, synthetic_tokenizer=tokenizer)
        compute = report["compute"]
        prompt_length = observations[0][0]
        assert observations[0] == (prompt_length, 0, False)
        assert all(item == (1, prompt_length + i - 1, False)
                   for i, item in enumerate(observations[1:], 1))
        assert compute["sampled_completions"] == compute["trace_cap_hits"] == 8
        assert compute["forced_format_tokens"] == 8 * delimiter_length
        assert compute["optimized_tokens"] == compute["generated_tokens"] - compute["forced_format_tokens"]
        assert compute["generation_seconds"] > 0 and compute["prompt_exposures"] == 1
        assert loss_masks and all(mask.eq(1).all() for mask in loss_masks)
        assert any(parameter.detach().abs().sum() > 0 for name, parameter in model.named_parameters()
                   if "lora_B" in name)
    assert model.training and model.is_gradient_checkpointing
    assert model.config.use_cache is False and model.generate is constrained_generate
