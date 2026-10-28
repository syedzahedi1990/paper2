"""Critical-token swaps through native Hugging Face Qwen2 forward passes.

K/V payloads are taken after the projection (including bias and active LoRA)
and before native RoPE / grouped-query replication. A key is therefore rotated
at the *recipient's* position, exactly once, by the normal model code. KV heads
are distinct from query heads. Residual swaps replace the decoder block output
at the critical token. Nothing is written to model weights or the KV cache.

These are local component swaps: later computations may respond to the edit.
They do not implement paper one's optional downstream token-restoration mode.
Every call is batch size one, unpadded, in eval mode, with use_cache=False.
Layer numbers are zero based; selecting layers must precede confirmation data.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from typing import Any, Iterable, Literal


@dataclass(frozen=True)
class Site:
    layer: int
    component: Literal["k", "v", "residual"]
    heads: tuple[int, ...] | None = None

    @property
    def key(self) -> str:
        return f"{self.layer}:{self.component}"


@dataclass(frozen=True)
class EncodedInput:
    input_ids: tuple[int, ...]
    critical_position: int
    prefix_stop: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_ids", tuple(self.input_ids))
        if not self.input_ids or any(type(v) is not int or v < 0 for v in self.input_ids):
            raise ValueError("input_ids must be a nonempty sequence of nonnegative integers")
        if type(self.critical_position) is not int or not 0 <= self.critical_position < len(self.input_ids):
            raise ValueError("critical_position must identify an input token")
        if self.prefix_stop is None:
            object.__setattr__(self, "prefix_stop", self.critical_position + 1)
        if type(self.prefix_stop) is not int or not self.critical_position < self.prefix_stop <= len(self.input_ids):
            raise ValueError("prefix_stop must include the critical token")

    @property
    def prefix(self) -> tuple[int, ...]:
        return self.input_ids[:self.prefix_stop]


def encode_input(tokenizer: Any, prompt: str, critical_char_span: tuple[int, int],
                 prefix_stop_char: int | None = None) -> EncodedInput:
    """Resolve an explicit critical location span without substring guessing.

    ``prompt`` is the final serialized model input, including the answer prefill.
    This uses ``add_special_tokens=False`` because chat serialization already
    supplies its own special tokens. Multi-token critical values fail closed;
    the proposal's critical-token intervention cannot silently become a span.
    """
    start, end = critical_char_span
    if not 0 <= start < end <= len(prompt):
        raise ValueError("Invalid critical character span")
    encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
    overlap = [i for i, (a, b) in enumerate(offsets) if a < end and b > start]
    if len(overlap) != 1:
        raise ValueError("Critical value must align to exactly one tokenizer token")
    p = overlap[0]
    a, b = offsets[p]
    if a > start or b < end or prompt[a:start].strip() or prompt[end:b].strip():
        raise ValueError("Critical token contains additional non-whitespace content")
    stop = p + 1
    if prefix_stop_char is not None:
        if not end <= prefix_stop_char <= len(prompt):
            raise ValueError("prefix_stop_char must follow the critical value")
        if any(a < prefix_stop_char < b for a, b in offsets):
            raise ValueError("Prefix boundary must align to a tokenizer boundary")
        stop = max(i + 1 for i, (a, b) in enumerate(offsets) if b > a and b <= prefix_stop_char)
    return EncodedInput(tuple(int(i) for i in ids), p, stop)


def _digest_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _tensor_hash(value: Any) -> str:
    import torch
    data = value.detach().cpu().contiguous()
    metadata = str((tuple(data.shape), str(data.dtype))).encode()
    return hashlib.sha256(metadata + data.view(torch.uint8).numpy().tobytes()).hexdigest()


@dataclass(frozen=True)
class Capture:
    encoded: EncodedInput
    checkpoint_id: str
    engine_id: str
    geometry: tuple[int, int, int, int]
    parameter_signature: tuple[Any, ...]
    payloads: dict[str, Any]
    hashes: dict[str, str]
    next_token_logits: Any
    model_class: str
    architecture_sha256: str


@dataclass(frozen=True)
class InterventionResult:
    next_token_logits: Any
    receipt: dict[str, Any]


def signed_permutation_delta(recipient: Any, donor: Any, *, seed: int) -> tuple[Any, dict[str, Any]]:
    """Recipient + randomly signed/permuted donor-minus-recipient displacement.

    Coordinate permutations/signs are independent per head and reproducible on
    CPU. Calculations use float32 (float64 inputs keep float64), with one final
    cast to the recipient dtype. Requested and realized norms expose rounding.
    The null need not have zero behavioral effect: that is a measured G4 gate.
    """
    import torch
    if tuple(recipient.shape) != tuple(donor.shape) or recipient.ndim not in (1, 2):
        raise ValueError("Null payloads must have matching [width] or [heads, width] shapes")
    if not recipient.is_floating_point() or not donor.is_floating_point():
        raise ValueError("Null payloads must be floating point")
    if type(seed) is not int or seed < 0:
        raise ValueError("Null seed must be a nonnegative integer")
    dtype = torch.float64 if recipient.dtype == torch.float64 else torch.float32
    r = recipient.detach().cpu().to(dtype)
    d = donor.detach().cpu().to(dtype)
    delta = (d - r).reshape(-1, r.shape[-1])
    generator = torch.Generator(device="cpu").manual_seed(seed)
    permutations = torch.stack([torch.randperm(delta.shape[-1], generator=generator) for _ in delta])
    signs = (torch.randint(0, 2, delta.shape, generator=generator) * 2 - 1).to(dtype)
    rotated = delta.gather(-1, permutations) * signs
    requested = r + rotated.reshape_as(r)
    result = requested.to(recipient.dtype)
    norm = lambda value: value.reshape(-1, r.shape[-1]).double().norm(dim=-1).tolist()
    return result, {
        "seed": seed, "kind": "signed_coordinate_permutation_of_delta",
        "permutation_sha256": _tensor_hash(permutations), "sign_sha256": _tensor_hash(signs),
        "original_delta_norm": norm(delta), "rotated_delta_norm": norm(rotated),
        "realized_delta_norm": norm(result.to(dtype) - r),
        "rounding_norm": norm(result.to(dtype) - requested),
    }


class ComponentInterventionEngine:
    """Capture and swap Qwen2.5 components without replacing attention code.

    Pass an eval-mode ``AutoModelForCausalLM`` or a PEFT model whose underlying
    model_type is qwen2. Captures are bound to model instance, active adapter,
    parameter versions, token prefix, and payload checksums. Do not share one
    engine concurrently between threads.
    """

    def __init__(self, model: Any, checkpoint_id: str):
        if not isinstance(checkpoint_id, str) or not checkpoint_id.strip():
            raise ValueError("An explicit checkpoint_id is required")
        self.model, self.checkpoint_id = model, checkpoint_id
        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        config = base.config
        if config.model_type != "qwen2":
            raise ValueError("This audited adapter supports Qwen2/Qwen2.5 only")
        self.layers = base.model.layers
        self.n_kv = config.num_key_value_heads
        self.n_q = config.num_attention_heads
        self.width = config.hidden_size
        self.head_dim = getattr(config, "head_dim", None) or self.width // self.n_q
        if self.n_q % self.n_kv or self.head_dim * self.n_q != self.width:
            raise ValueError("Unsupported attention/GQA geometry")
        self.geometry = (len(self.layers), self.width, self.n_kv, self.head_dim)
        self.architecture_sha256 = _digest_json({name: getattr(config, name, None) for name in (
            "model_type", "vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
            "num_attention_heads", "num_key_value_heads", "head_dim", "rope_theta", "rope_scaling",
            "sliding_window", "use_sliding_window", "max_window_layers", "hidden_act", "rms_norm_eps",
        )})
        self.engine_id = uuid.uuid4().hex
        self._active = False
        self._assert_eval()

    def _assert_eval(self) -> None:
        if any(module.training for module in self.model.modules()):
            raise ValueError("Component interventions require model.eval() (dropout disabled)")

    def _signature(self) -> tuple[Any, ...]:
        adapter = repr(getattr(self.model, "active_adapters", None))
        return (adapter, tuple((name, id(p), p._version) for name, p in self.model.named_parameters()))

    def _sites(self, sites: Iterable[Site]) -> tuple[Site, ...]:
        result = tuple(sites)
        if not result or len({site.key for site in result}) != len(result):
            raise ValueError("Provide nonempty, non-overlapping component sites")
        for site in result:
            if type(site.layer) is not int or not 0 <= site.layer < len(self.layers):
                raise ValueError("Invalid zero-based layer")
            if site.component not in ("k", "v", "residual"):
                raise ValueError("Component must be k, v, or residual")
            if site.component == "residual" and site.heads is not None:
                raise ValueError("Residual swaps have no KV-head selection")
            if site.heads is not None and (not site.heads or len(set(site.heads)) != len(site.heads)
                    or any(type(h) is not int or not 0 <= h < self.n_kv for h in site.heads)):
                raise ValueError("Heads must select unique KV heads, not query heads")
        return result

    def _module(self, site: Site) -> Any:
        block = self.layers[site.layer]
        return block if site.component == "residual" else getattr(block.self_attn, site.component + "_proj")

    def _shape(self, site: Site) -> tuple[int, ...]:
        return (self.width,) if site.component == "residual" else (self.n_kv, self.head_dim)

    def _validate_capture(self, capture: Capture, sites: tuple[Site, ...], *, local: bool) -> None:
        if capture.geometry != self.geometry:
            raise ValueError("Donor and recipient component geometry differs")
        if capture.architecture_sha256 != self.architecture_sha256:
            raise ValueError("Donor and recipient model architecture or RoPE settings differ")
        if local and (capture.engine_id != self.engine_id or capture.parameter_signature != self._signature()):
            raise ValueError("Capture belongs to another model/checkpoint or model parameters changed")
        for site in sites:
            if site.key not in capture.payloads:
                raise ValueError(f"Capture is missing site {site.key}")
            value = capture.payloads[site.key]
            if tuple(value.shape) != self._shape(site) or _tensor_hash(value) != capture.hashes[site.key]:
                raise ValueError("Captured payload was mutated or has the wrong shape")

    def capture(self, encoded: EncodedInput, sites: Iterable[Site]) -> Capture:
        sites = self._sites(sites)
        signature = self._signature()
        logits, payloads, _ = self._forward(encoded, sites)
        if signature != self._signature():
            raise RuntimeError("Model parameters changed during capture")
        return Capture(encoded, self.checkpoint_id, self.engine_id, self.geometry, signature,
                       payloads, {k: _tensor_hash(v) for k, v in payloads.items()}, logits,
                       type(self.model).__qualname__, self.architecture_sha256)

    def swap(self, encoded: EncodedInput, recipient: Capture, donor: Capture,
             sites: Iterable[Site], *, control: Literal["natural", "own", "signed_permutation"] = "natural",
             seed: int = 0, secondary_cross_model: bool = False) -> InterventionResult:
        sites = self._sites(sites)
        if control not in ("natural", "own", "signed_permutation"):
            raise ValueError("Unknown component control")
        if type(seed) is not int or seed < 0:
            raise ValueError("Null seed must be a nonnegative integer")
        self._validate_capture(recipient, sites, local=True)
        if encoded.critical_position != recipient.encoded.critical_position or encoded.prefix != recipient.encoded.prefix:
            raise ValueError("Recipient capture and execution token prefix differ")
        source = recipient if control == "own" else donor
        cross = source.engine_id != self.engine_id
        if cross and not secondary_cross_model:
            raise ValueError("Cross-model swaps require explicit secondary_cross_model=True")
        self._validate_capture(source, sites, local=not cross)
        if source.encoded.critical_position != encoded.critical_position or len(source.encoded.prefix) != len(encoded.prefix):
            raise ValueError("Donor and recipient token coordinates differ")
        if cross:
            if source.encoded.prefix != encoded.prefix:
                raise ValueError("Secondary cross-model swaps require identical token prefixes")
        elif any(a != b for i, (a, b) in enumerate(zip(source.encoded.prefix, encoded.prefix))
                 if i != encoded.critical_position):
            raise ValueError("Natural matched donor changes prefix outside the critical token")
        replacements, nulls = {}, {}
        for site in sites:
            payload = source.payloads[site.key]
            if control == "signed_permutation":
                # Site-specific seeds do not depend on site traversal order.
                local_seed = int(_digest_json([seed, site.key])[:15], 16)
                payload, nulls[site.key] = signed_permutation_delta(
                    recipient.payloads[site.key], payload, seed=local_seed)
            replacements[site.key] = payload
        signature = self._signature()
        logits, _, audit = self._forward(encoded, sites, replacements=replacements)
        if signature != self._signature():
            raise RuntimeError("Model parameters changed during intervention")
        receipt = {
            "schema_version": 1, "checkpoint_id": self.checkpoint_id,
            "architecture_sha256": self.architecture_sha256,
            "donor_checkpoint_id": source.checkpoint_id, "control": control,
            "analysis": "secondary_cross_model" if cross else "primary_within_checkpoint",
            "critical_position": encoded.critical_position, "prefix_stop": encoded.prefix_stop,
            "input_ids_sha256": _digest_json(encoded.input_ids),
            "recipient_prefix_sha256": _digest_json(encoded.prefix),
            "donor_prefix_sha256": _digest_json(source.encoded.prefix),
            "key_site": "post_bias_pre_native_RoPE", "value_site": "post_bias_pre_native_GQA_repeat",
            "residual_site": "decoder_block_output", "layer_indexing": "zero_based",
            "model_num_hidden_layers": len(self.layers),
            "native_attention": True, "use_cache": False, "direct_query_modification": False,
            "downstream_computation_may_respond": True, "nulls": nulls, "sites": audit,
        }
        return InterventionResult(logits, receipt)

    def controls(self, encoded: EncodedInput, recipient: Capture, donor: Capture,
                 sites: Iterable[Site], *, seed: int = 0) -> dict[str, InterventionResult]:
        """Run natural-donor, own-component and displacement-matched null controls."""
        sites = tuple(sites)
        return {name: self.swap(encoded, recipient, donor, sites, control=name, seed=seed)
                for name in ("natural", "own", "signed_permutation")}

    def _forward(self, encoded: EncodedInput, sites: tuple[Site, ...],
                 replacements: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any], list[dict[str, Any]]]:
        import torch
        self._assert_eval()
        if self._active:
            raise RuntimeError("Concurrent or recursive intervention execution is unsupported")
        modules = [self._module(site) for site in sites]
        if any(module._forward_hooks or module._forward_pre_hooks for module in modules):
            raise ValueError("Existing hooks conflict with the selected component sites")
        self._active = True
        handles, captures, audit, observed = [], {}, [], set()
        p = encoded.critical_position

        def hook(site: Site):
            def apply(_module, _inputs, output):
                if site.key in observed:
                    raise RuntimeError("Component called more than once; cache/recomputation is unsupported")
                observed.add(site.key)
                tensor = output[0] if isinstance(output, tuple) else output
                width = self.width if site.component == "residual" else self.n_kv * self.head_dim
                if tuple(tensor.shape) != (1, len(encoded.input_ids), width):
                    raise ValueError("Native component tensor shape changed")
                payload = tensor[0, p].reshape(self._shape(site)).detach().cpu().clone()
                captures[site.key] = payload
                if replacements is None:
                    return None
                if not torch.isfinite(replacements[site.key]).all():
                    raise ValueError("Non-finite donor payload")
                mixed = tensor.clone()
                row = mixed[0, p].reshape(self._shape(site))
                target = replacements[site.key].to(device=tensor.device, dtype=tensor.dtype)
                heads = tuple(range(self.n_kv)) if site.heads is None else site.heads
                if site.component == "residual":
                    row.copy_(target)
                else:
                    for head in heads:
                        row[head].copy_(target[head])
                other_exact = torch.equal(mixed[:, :p], tensor[:, :p]) and torch.equal(mixed[:, p + 1:], tensor[:, p + 1:])
                if not other_exact:
                    raise RuntimeError("Intervention changed a noncritical token")
                if site.component != "residual":
                    for head in set(range(self.n_kv)) - set(heads):
                        if not torch.equal(row[head], payload[head].to(row.device)):
                            raise RuntimeError("Intervention changed an unselected KV head")
                audit.append({
                    "layer": site.layer, "component": site.component,
                    "kv_heads": None if site.component == "residual" else list(heads),
                    "received_sha256": _tensor_hash(payload), "donor_sha256": _tensor_hash(target),
                    "applied_sha256": _tensor_hash(row), "other_token_rows_exact": other_exact,
                    "unselected_heads_exact": True,
                })
                if isinstance(output, tuple):
                    return (mixed, *output[1:])
                return mixed
            return apply

        try:
            for site, module in zip(sites, modules):
                handles.append(module.register_forward_hook(hook(site)))
            device = self.model.get_input_embeddings().weight.device
            if device.type == "meta":
                raise ValueError("Input embeddings on meta device are unsupported")
            ids = torch.tensor([encoded.input_ids], device=device, dtype=torch.long)
            with torch.inference_mode():
                output = self.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                                    use_cache=False, return_dict=True)
            if observed != {site.key for site in sites}:
                raise RuntimeError("Not every requested native component was called")
            return output.logits[0, -1].detach().float().cpu().clone(), captures, audit
        finally:
            for handle in reversed(handles):
                handle.remove()
            self._active = False
