"""Nontrivial residual positive controls for the empirical G4 review."""
from collections import defaultdict

from paper2.analysis import _seed, cluster_interval
from paper2.data import VIEWS


def residual_control_eligibility(layers, num_layers, *, cross=False):
    """Conservatively exclude input reconstruction and disconnected output sites.

    The engine patches block outputs, not token embeddings. A mask beginning at
    block zero is still excluded from G4 because it may approximate a donor-input
    reconstruction. At least one downstream attention block must remain for the
    critical token's residual state to influence the answer position.
    """
    if cross:
        return False, "secondary_cross_model"
    if (type(num_layers) is not int or num_layers < 2 or not layers
            or any(type(layer) is not int or not 0 <= layer < num_layers for layer in layers)
            or len(set(layers)) != len(layers)):
        return False, "missing_or_invalid_layer_geometry"
    if 0 in layers:
        return False, "includes_layer_zero_input_reconstruction_risk"
    if min(layers) == num_layers - 1:
        return False, "no_downstream_attention_block"
    return True, "nonzero_band_with_downstream_attention"


def _receipt_eligibility(row):
    layers = row.get("layers", [])
    eligible, reason = residual_control_eligibility(
        layers, row.get("model_num_hidden_layers"),
        cross=row.get("analysis") != "primary_within_checkpoint")
    if not eligible:
        return eligible, reason
    receipt = row.get("receipt") or {}
    sites = receipt.get("sites", [])
    if (receipt.get("analysis") != "primary_within_checkpoint"
            or receipt.get("residual_site") != "decoder_block_output"
            or receipt.get("layer_indexing") != "zero_based"
            or receipt.get("model_num_hidden_layers") != row["model_num_hidden_layers"]
            or receipt.get("checkpoint_id") != row.get("checkpoint_id")
            or receipt.get("donor_checkpoint_id") != row.get("checkpoint_id")
            or receipt.get("control") != row.get("control")
            or row.get("mask_id") != ",".join(map(str, sorted(layers)))
            or len(sites) != len(layers)
            or {site.get("layer") for site in sites} != set(layers)
            or any(site.get("component") != "residual" or not site.get("other_token_rows_exact") for site in sites)):
        return False, "missing_or_inconsistent_native_receipt"
    return True, reason


def residual_control_summary(rows):
    """Summarize measured candidates; structural eligibility never passes G4.

    Reports effect on changed, intact and joint predictions versus own-component
    replay, plus signed-permutation effects. G4 still needs reviewed nulls,
    preservation and precision; no architecture-only condition is a pass.
    """
    from .scoring import score_consequence_pattern
    groups = defaultdict(list)
    for row in rows:
        if row.get("component") == "residual" and row.get("control") in ("natural", "own", "signed_permutation"):
            groups[(row.get("checkpoint_id"), row.get("mask_id"))].append(row)
    result = []
    for (checkpoint, mask), group in sorted(groups.items(), key=lambda item: repr(item[0])):
        reasons = {_receipt_eligibility(row)[1] for row in group if not _receipt_eligibility(row)[0]}
        candidate = {"checkpoint_id": checkpoint, "mask_id": mask, "layers": group[0].get("layers", []),
                     "eligible": not reasons, "ineligible_reasons": sorted(reasons), "effects": {}}
        if reasons:
            result.append(candidate)
            continue
        panels = defaultdict(dict)
        for row in group:
            key = (_seed(row), row["pair_id"], row["control"])
            if row["view"] in panels[key]:
                raise ValueError("G4 residual evidence has duplicate views")
            panels[key][row["view"]] = row
        expected = {(seed, pair, control) for seed, pair, _ in panels
                    for control in ("natural", "own", "signed_permutation")}
        if set(panels) != expected or any(set(views) != set(VIEWS) for views in panels.values()):
            raise ValueError("G4 residual evidence requires complete eight-view natural, own and null panels")
        seed_panels = defaultdict(set)
        for seed, pair, _ in panels:
            seed_panels[seed].add(pair)
        if any(pairs != next(iter(seed_panels.values())) for pairs in seed_panels.values()):
            raise ValueError("G4 residual evidence must cover the same stories for every training seed")
        scores = {}
        for key, views in panels.items():
            for view, row in views.items():
                own = panels[(*key[:2], "own")][view]
                if (row["ordinary_target"] != own["ordinary_target"] or row["forecasts"] != own["forecasts"]
                        or row["forecast_target"] != row["forecasts"]["event_rewrite"]):
                    raise ValueError("G4 residual evidence changed forecast targets across controls")
            scores[key] = score_consequence_pattern(
                {view: row["prediction"] for view, row in views.items()},
                {view: row["ordinary_target"] for view, row in views.items()},
                {view: row["forecasts"] for view, row in views.items()})
        clusters = defaultdict(lambda: defaultdict(list))
        for (seed, pair, control), score in scores.items():
            if control == "own":
                continue
            own = scores[(seed, pair, "own")]
            for metric in ("changed_accuracy", "intact_accuracy", "joint_correct"):
                if score[metric] is not None:
                    clusters[f"{control}|{metric}"][pair].append(float(score[metric]) - float(own[metric]))
        candidate["effects"] = {metric: cluster_interval(values) for metric, values in sorted(clusters.items())}
        result.append(candidate)
    return {"candidates": result, "eligible_masks": sorted({r["mask_id"] for r in result if r["eligible"]}),
            "empirical_gate_passed": False,
            "review_required": "Review positive effects, intact views, flat nulls and useful story-level precision."}
