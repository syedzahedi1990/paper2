"""Executable simultaneous component swaps and optional layer diagnostics."""
from __future__ import annotations
from dataclasses import asdict
from typing import Any, Iterable
from .components import ComponentInterventionEngine, Site, encode_input


def intervention_masks(sites, mode="joint"):
    """Use the same declared layer mask for K, V and the residual control."""
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
    return [(mask, component, tuple(by_site[layer, component] for layer in mask))
            for mask in masks for component in ("k", "v", "residual")]


def run_component_test(backend: Any, pairs: Iterable[dict[str, Any]], sites: Iterable[Site],
                       *, seed: int = 0, layer_mode="joint", donor_backend=None) -> list[dict[str, Any]]:
    """Swap all selected layers at once per component, with own/null controls.

    A donor backend explicitly requests a secondary between-checkpoint analysis:
    it supplies activations on the *identical recipient input*. Its targets are
    ordinary task targets, not the primary matched-event rewrite forecasts.
    No fitted intervention, architecture band or empirical G4 success is assumed.
    """
    from paper2.prompts import answer_token_ids, render_prompt
    sites = tuple(sites)
    masks = intervention_masks(sites, layer_mode)
    engine = ComponentInterventionEngine(backend.model, backend.checkpoint_id)
    cross = donor_backend is not None
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
            }

            def add(logits, control, *, mask=(), component=None, receipt=None, role="recipient"):
                answer = backend.answer_from_logits(logits, mappings[role])
                rows.append({**common, "layer": mask[0] if len(mask) == 1 else None,
                             "layers": list(mask), "mask_id": ",".join(map(str, mask)), "component": component,
                             "control": control, "positive_control": not cross and component == "residual" and control == "natural",
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
