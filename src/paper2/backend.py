"""Hugging Face inference with unconstrained, single-token primary answers."""
from dataclasses import dataclass

from .prompts import answer_token_ids


@dataclass
class Answer:
    prediction: str | None
    token_id: int
    text: str
    restricted_prediction: str | None
    probabilities: dict


class HFBackend:
    def __init__(self, model_id, adapter=None, device="auto", dtype="auto", revision=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from .training.config import MODEL_REVISIONS
        if revision is None:
            revision = MODEL_REVISIONS.get(model_id)
        self.torch = torch
        self.checkpoint_id = adapter or model_id
        self.revision = revision
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        kwargs = {"revision": revision, "torch_dtype": dtype, "attn_implementation": "eager"}
        if device == "auto":
            kwargs["device_map"] = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
        if device != "auto":
            self.model.to(device)
        self.model.eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def token_ids(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def answer(self, prompt, choices, prefix_ids=None):
        mapping = answer_token_ids(self.tokenizer, prompt, choices)
        ids = self.token_ids(prompt) if prefix_ids is None else prefix_ids
        with self.torch.inference_mode():
            tensor = self.torch.tensor([ids], device=self.model.device)
            logits = self.model(input_ids=tensor, use_cache=False).logits[0, -1].float()
        return self.answer_from_logits(logits, mapping)

    def answer_from_logits(self, logits, mapping):
        logits = logits.float().reshape(-1)
        token_id = int(logits.argmax().item())
        probs = self.torch.softmax(logits, dim=-1)
        prediction = next((key for key, value in mapping.items() if value == token_id), None)
        return Answer(prediction, token_id, self.tokenizer.decode([token_id]),
                      max(mapping, key=lambda key: float(logits[mapping[key]].item())),
                      {key: float(probs[value].item()) for key, value in mapping.items()})

    def generate(self, prompt, max_new_tokens=384, stop=None, prefix_ids=None, *,
                 do_sample=False, temperature=.7, top_p=.95, seed=None):
        from transformers import StoppingCriteria, StoppingCriteriaList
        tokenizer = self.tokenizer
        ids = self.token_ids(prompt) if prefix_ids is None else prefix_ids
        n = len(ids)

        class StopText(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return stop in tokenizer.decode(input_ids[0, n:], skip_special_tokens=False)

        if max_new_tokens < 1:
            raise ValueError("Generation needs a positive token budget")
        if do_sample and (temperature <= 0 or not 0 < top_p <= 1):
            raise ValueError("Sampling needs positive temperature and 0 < top_p <= 1")
        if seed is not None and (type(seed) is not int or seed < 0):
            raise ValueError("Generation seed must be a nonnegative integer")
        sampling = {"temperature": temperature, "top_p": top_p} if do_sample else {}
        # Seeded resampling must not change the caller's CPU/CUDA RNG state.
        devices = sorted({p.device.index for p in self.model.parameters() if p.device.type == "cuda"})
        with self.torch.random.fork_rng(devices=devices, enabled=seed is not None), self.torch.inference_mode():
            if seed is not None:
                self.torch.manual_seed(seed)
            output = self.model.generate(
                input_ids=self.torch.tensor([ids], device=self.model.device),
                max_new_tokens=max_new_tokens, do_sample=do_sample, **sampling,
                pad_token_id=self.tokenizer.pad_token_id,
                stopping_criteria=StoppingCriteriaList([StopText()]) if stop else None,
            )[0].tolist()
        generated = output[n:]
        return {"text": self.tokenizer.decode(generated, skip_special_tokens=True),
                "prompt_ids": ids, "generated_ids": generated,
                "truncated": len(generated) == max_new_tokens,
                "tokens": len(generated), "do_sample": do_sample, "seed": seed,
                "temperature": temperature if do_sample else None, "top_p": top_p if do_sample else None}
