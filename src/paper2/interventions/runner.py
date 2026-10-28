"""Executable Test 3 sweep over generator-produced matched event pairs."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Iterable

from .components import ComponentInterventionEngine, Site, encode_input


def run_component_test(backend: Any, pairs: Iterable[dict[str, Any]], sites: Iterable[Site],
                       *, seed: int = 0) -> list[dict[str, Any]]:
    """Run each site separately, alongside own, null and natural-input controls.

    Include at least one residual site for each selected layer: these are the
    positive controls needed to interpret a null K/V result. This function
    reports measurements; G4 success still requires empirical gate analysis.
    It never starts training or downloads a model. ``backend`` is an HFBackend
    with its model already loaded. All runs here are within one checkpoint.
    """
    from paper2.prompts import answer_token_ids, render_prompt

    sites = tuple(sites)
    layers = {site.layer for site in sites}
    if not sites or {site.layer for site in sites if site.component == "residual"} != layers:
        raise ValueError("Each tested layer needs a residual-stream positive-control site")
    engine = ComponentInterventionEngine(backend.model, backend.checkpoint_id)
    rows = []
    for pair in pairs:
        recipients = {row["view"]: row for row in pair["recipient"]}
        donors = {row["view"]: row for row in pair["donor"]}
        if set(recipients) != set(donors) or set(recipients) != set(pair["forecasts"]):
            raise ValueError("Matched-pair views and forecasts differ")
        for view, recipient_row in recipients.items():
            donor_row = donors[view]
            if recipient_row["choices"] != donor_row["choices"]:
                raise ValueError("Matched pair must share its answer alphabet")
            encoded, prompts = {}, {}
            for role, row in (("recipient", recipient_row), ("donor", donor_row)):
                prompt = render_prompt(backend.tokenizer, row, reasoning=False)
                if prompt.count(row["story"]) != 1:
                    raise ValueError("Serialized prompt must contain the story exactly once")
                origin = prompt.index(row["story"])
                a, b = row["critical_value_span"]
                encoded[role] = encode_input(backend.tokenizer, prompt, (origin + a, origin + b))
                prompts[role] = prompt
            recipient = engine.capture(encoded["recipient"], sites)
            donor = engine.capture(encoded["donor"], sites)
            mappings = {role: answer_token_ids(backend.tokenizer, prompt, recipient_row["choices"])
                        for role, prompt in prompts.items()}
            if mappings["recipient"] != mappings["donor"]:
                raise ValueError("Answer-token IDs changed between matched prompts")
            baseline = backend.answer_from_logits(recipient.next_token_logits, mappings["recipient"])
            forecasts = pair["forecasts"][view]
            common = {
                "checkpoint_id": backend.checkpoint_id, "pair_id": pair["pair_id"],
                "world_id": pair["world_id"], "split": pair["split"], "view": view,
                "event_kind": pair["event_kind"], "analysis": "primary_within_checkpoint",
                "ordinary_target": recipient_row["answer"], "donor_target": donor_row["answer"],
                "forecast_target": forecasts["event_rewrite"], "forecasts": forecasts,
                "change_required": forecasts["event_rewrite"] != recipient_row["answer"],
                "baseline_prediction": baseline.prediction,
                "baseline_correct": baseline.prediction == recipient_row["answer"],
            }

            def add(logits, control, *, site=None, receipt=None, role="recipient"):
                answer = backend.answer_from_logits(logits, mappings[role])
                rows.append({**common, "layer": None if site is None else site.layer,
                             "component": None if site is None else site.component,
                             "control": control, "positive_control": site is not None and site.component == "residual" and control == "natural",
                             **asdict(answer), "forecast_correct": answer.prediction == forecasts["event_rewrite"],
                             "forecast_probability": answer.probabilities[forecasts["event_rewrite"]],
                             "ordinary_correct": answer.prediction == recipient_row["answer"],
                             "donor_correct": answer.prediction == donor_row["answer"],
                             "receipt": receipt})
            add(recipient.next_token_logits, "baseline")
            add(donor.next_token_logits, "natural_input", role="donor")
            for site in sites:
                for name, result in engine.controls(encoded["recipient"], recipient, donor, [site], seed=seed).items():
                    add(result.next_token_logits, name, site=site, receipt=result.receipt)
    return rows
