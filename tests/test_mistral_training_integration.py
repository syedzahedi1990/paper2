"""Run the existing optimizer acceptance checks on nonsquare tiny Mistral."""
import pytest

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("arm", ["label_sft", "answer_rl", "reasoning_rl"])
def test_real_mistral_lora_training_and_reload(tmp_path, monkeypatch, arm):
    torch = pytest.importorskip("torch")
    pytest.importorskip("trl")
    pytest.importorskip("peft")
    from peft import PeftModel
    import test_training_integration as acceptance
    from test_mistral_interventions import mistral_model

    original_fixture, models = acceptance.tiny_fixture, []

    def tiny_mistral_fixture():
        _, tokenizer = original_fixture()
        model = mistral_model(vocab_size=len(tokenizer))
        model.config._name_or_path = "paper2-random-tiny-mistral"
        models.append((model, tokenizer))
        return model, tokenizer

    # Reuse the same real TRL training, mixed-reward generation, checkpoint,
    # compute-ledger and native/immediate evaluation assertions as Qwen.
    monkeypatch.setattr(acceptance, "tiny_fixture", tiny_mistral_fixture)
    acceptance.test_real_trl_optimizer_and_saved_adapter(tmp_path, arm)
    model, tokenizer = models[0]
    assert any(parameter.detach().abs().sum() > 0 for name, parameter in model.named_parameters()
               if "lora_B" in name)
    model.eval()
    reloaded = PeftModel.from_pretrained(mistral_model(vocab_size=len(tokenizer)),
                                         tmp_path / "adapter").eval()
    inputs = tokenizer("Alice ball box. Bob ball basket.", return_tensors="pt")
    with torch.inference_mode():
        expected = model(**inputs, use_cache=False).logits
        actual = reloaded(**inputs, use_cache=False).logits
    assert torch.equal(expected, actual)
