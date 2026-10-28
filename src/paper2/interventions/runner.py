"""Executable simultaneous component swaps and optional layer diagnostics."""
from __future__ import annotations
from dataclasses import asdict
from typing import Any, Iterable
from .components import ComponentInterventionEngine, Site, encode_input
from .controls import residual_control_eligibility


def intervention_masks(sites, mode="joint", residual_control_layers=None, g4_test_layers=None):
    """Retain diagnostics and add K/V/residual on the declared G4 test band."""
    sites = tuple(sites)
    if mode not in ("joint", "single", "both"):
        raise ValueError("Unknown layer mode")
    layers = sorted({s.layer for s in sites})
    if not layers or any({s.layer for s in sites if s.component == c} != set(layers)
                         for c in ("k", "v", "residual")):
        raise ValueError("Every layer mask needs K, V and residual positive-control sites")
    by_site = {(site.layer, site.component): site for site in sites}
    if len(by_site) != len(sites):
        raise ValueError("Layer masks cannot repeat a component site")
    masks = [tuple(layers)] if mode in ("joint", "both") else []
    if mode in ("single", "both"):
        masks.extend((layer,) for layer in layers if (layer,) not in masks)
    result = [(mask, component, tuple(by_site[layer, component] for layer in mask))
              for mask in masks for component in ("k", "v", "residual")]
    if residual_control_layers is not None:
        requested = tuple(residual_control_layers)
        if (not requested or any(type(layer) is not int or layer <= 0 for layer in requested)
                or len(set(requested)) != len(requested)):
            raise ValueError("Residual diagnostic layers must be distinct integers greater than zero")
        mask = tuple(sorted(requested))
        if not any(existing == mask and component == "residual" for existing, component, _ in result):
            result.append((mask, "residual", tuple(Site(layer, "residual") for layer in mask)))
    if g4_test_layers is not None:
        requested = tuple(g4_test_layers)
        if (not requested or any(type(layer) is not int or layer <= 0 for layer in requested)
                or len(set(requested)) != len(requested)):
            raise ValueError("G4 test layers must be distinct integers greater than zero")
        mask = tuple(sorted(requested))
        if list(mask) != list(range(mask[0], mask[-1] + 1)):
            raise ValueError("G4 test layers must declare a contiguous band")
        for component in ("k", "v", "residual"):
            if not any(existing == mask and name == component for existing, name, _ in result):
                result.append((mask, component, tuple(by_site.get((layer, component), Site(layer, component))
                                                     for layer in mask)))
    return result


def run_component_test(backend: Any, pairs: Iterable[dict[str, Any]], sites: Iterable[Site],
                       *, seed: int = 0, layer_mode="joint", donor_backend=None,
                       residual_control_layers=None, g4_test_layers=None) -> list[dict[str, Any]]:
    """Swap all selected layers at once per component, with own/null controls.

    A donor backend explicitly requests a secondary between-checkpoint analysis:
    it supplies activations on the *identical recipient input*. Its targets are
    ordinary task targets, not the primary matched-event rewrite forecasts.
    No fitted intervention, architecture band or empirical G4 success is assumed.
    """
    from paper2.prompts import answer_token_ids, render_prompt
    sites = tuple(sites)
    masks = intervention_masks(sites, layer_mode, residual_control_layers, g4_test_layers)
    engine = ComponentInterventionEngine(backend.model, backend.checkpoint_id)
    cross = donor_backend is not None
    g4_test_layers = sorted(g4_test_layers) if g4_test_layers is not None else None
    if g4_test_layers is not None:
        eligible, reason = residual_control_eligibility(g4_test_layers, len(engine.layers),
                                                       test_layers=g4_test_layers, cross=cross)
        if not eligible:
            raise ValueError(f"Residual positive-control band is ineligible: {reason}")
    # The declared G4 band always measures all three components together.
    sites = tuple({(site.layer, site.component): site for _, _, selected in masks for site in selected}.values())
    donor_engine = ComponentInterventionEngine(donor_backend.model, donor_backend.checkpoint_id) if cross else engine
    rows = []
    for pair in pairs:
        recipients = {row["view"]: row for row in pair["recipient"]}
        donors = {row["view"]: row for row in pair["donor"]}
        if set(recipients) != set(donors) or set(recipients) != set(pair["forecasts"]):
            raise ValueError("Matched-pair views and forecasts differ")
        for view, recipient_row in recipients.items():
            donor_row = recipient_row if cross else donors[view]
            if recipient_row["choices"] != donor_row["choices"]:
                raise ValueError("Matched pair must share its answer alphabet")
            encoded, prompts = {}, {}
            for role, row in (("recipient", recipient_row), ("donor", donor_row)):
                tokenizer = donor_backend.tokenizer if cross and role == "donor" else backend.tokenizer
                prompt = render_prompt(tokenizer, row, reasoning=False)
                if prompt.count(row["story"]) != 1:
                    raise ValueError("Serialized prompt must contain the story exactly once")
                origin = prompt.index(row["story"])
                a, b = row["critical_value_span"]
                encoded[role] = encode_input(tokenizer, prompt, (origin + a, origin + b))
                prompts[role] = prompt
            if cross and encoded["recipient"].input_ids != encoded["donor"].input_ids:
                raise ValueError("Cross-model CLI analysis requires identical complete token inputs")
            recipient = engine.capture(encoded["recipient"], sites)
            donor = donor_engine.capture(encoded["donor"], sites)
            mappings = {role: answer_token_ids(donor_backend.tokenizer if cross and role == "donor" else backend.tokenizer,
                                               prompt, recipient_row["choices"])
                        for role, prompt in prompts.items()}
            if mappings["recipient"] != mappings["donor"]:
                raise ValueError("Answer-token IDs changed between matched prompts")
            baseline = backend.answer_from_logits(recipient.next_token_logits, mappings["recipient"])
            forecasts = pair["forecasts"][view]
            target = recipient_row["answer"] if cross else forecasts["event_rewrite"]
            common = {
                "checkpoint_id": backend.checkpoint_id, "donor_checkpoint_id": donor_engine.checkpoint_id,
                "pair_id": pair["pair_id"], "world_id": pair["world_id"], "split": pair["split"], "view": view,
                "event_kind": pair["event_kind"],
                "analysis": "secondary_cross_model" if cross else "primary_within_checkpoint",
                "forecast_account": "ordinary_task_secondary" if cross else "event_rewrite",
                "ordinary_target": recipient_row["answer"], "donor_target": donor_row["answer"],
                "forecast_target": target, "forecasts": forecasts,
                "change_required": target != recipient_row["answer"],
                "baseline_prediction": baseline.prediction,
                "baseline_correct": baseline.prediction == recipient_row["answer"],
                "model_num_hidden_layers": len(engine.layers),
                "g4_test_layers": g4_test_layers,
            }

            def add(logits, control, *, mask=(), component=None, receipt=None, role="recipient"):
                answer = backend.answer_from_logits(logits, mappings[role])
                eligible, reason = residual_control_eligibility(mask, len(engine.layers),
                                                               test_layers=g4_test_layers, cross=cross)
                eligible = eligible and component == "residual"
                if component != "residual":
                    reason = "not_residual"
                if receipt is not None:
                    receipt = {**receipt, "g4_test_layers": g4_test_layers}
                rows.append({**common, "layer": mask[0] if len(mask) == 1 else None,
                             "layers": list(mask), "mask_id": ",".join(map(str, mask)), "component": component,
                             "control": control, "positive_control": eligible and control == "natural",
                             "positive_control_candidate": component == "residual" and control == "natural",
                             "g4_control_eligible": eligible, "g4_control_reason": reason,
                             **asdict(answer), "forecast_correct": answer.prediction == target,
                             "forecast_probability": answer.probabilities[target],
                             "ordinary_correct": answer.prediction == recipient_row["answer"],
                             "donor_correct": answer.prediction == donor_row["answer"], "receipt": receipt})
            add(recipient.next_token_logits, "baseline")
            add(donor.next_token_logits, "donor_checkpoint_input" if cross else "natural_input", role="donor")
            for mask, component, selected in masks:
                for control in ("natural", "own", "signed_permutation"):
                    result = engine.swap(encoded["recipient"], recipient, donor, selected,
                                         control=control, seed=seed, secondary_cross_model=cross)
                    add(result.next_token_logits, control, mask=mask, component=component, receipt=result.receipt)
    return rows
