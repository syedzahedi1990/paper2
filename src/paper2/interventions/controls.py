"""Nontrivial residual positive controls for the empirical G4 review."""
from collections import defaultdict

from paper2.analysis import _seed, cluster_interval
from paper2.data import VIEWS


def residual_control_eligibility(layers, num_layers, *, test_layers=None, cross=False):
    """Conservatively exclude input reconstruction and disconnected output sites.

    The engine patches block outputs, not token embeddings. A mask beginning at
    block zero is still excluded from G4 because it may approximate a donor-input
    reconstruction. At least one downstream attention block must remain for the
    critical token's residual state to influence the answer position. An interior
    mask is not sufficient: the control must match the explicitly declared K/V
    test band, including its first layer. Unselected single-layer diagnostics
    cannot certify a different joint band.
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
    if (not isinstance(test_layers, (list, tuple)) or not test_layers
            or any(type(layer) is not int or not 0 < layer < num_layers for layer in test_layers)
            or list(test_layers) != list(range(min(test_layers), max(test_layers) + 1))):
        return False, "missing_or_invalid_preregistered_test_band"
    if sorted(layers) != list(test_layers):
        return False, "residual_band_does_not_match_preregistered_test_band"
    return True, "aligned_preregistered_band_with_downstream_attention"


def _receipt_eligibility(row, component="residual"):
    layers = row.get("layers", [])
    eligible, reason = residual_control_eligibility(
        layers, row.get("model_num_hidden_layers"),
        test_layers=row.get("g4_test_layers"),
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
            or receipt.get("g4_test_layers") != row.get("g4_test_layers")
            or row.get("component") != component
            or row.get("mask_id") != ",".join(map(str, sorted(layers)))
            or len(sites) != len(layers)
            or {site.get("layer") for site in sites} != set(layers)
            or any(site.get("component") != component or not site.get("other_token_rows_exact") for site in sites)):
        return False, "missing_or_inconsistent_native_receipt"
    return True, reason


def _aligned_component_evidence(rows, residuals, checkpoint, mask):
    """Require actual K and V conditions on the band certified by the control."""
    expected = {(_seed(row), row["pair_id"], row["view"], row["control"]): row for row in residuals}
    for component in ("k", "v"):
        component_rows = [row for row in rows if row.get("checkpoint_id") == checkpoint
                          and row.get("mask_id") == mask and row.get("component") == component
                          and row.get("control") in ("natural", "own", "signed_permutation")]
        indexed = {(_seed(row), row["pair_id"], row["view"], row["control"]): row for row in component_rows}
        if len(indexed) != len(component_rows) or set(indexed) != set(expected):
            return "missing_or_incomplete_same_band_key_value_panels"
        for key, row in indexed.items():
            residual = expected[key]
            if not _receipt_eligibility(row, component)[0]:
                return "missing_or_inconsistent_same_band_key_value_receipts"
            if (row.get("g4_test_layers") != residual.get("g4_test_layers")
                    or any(row.get(field) != residual.get(field)
                           for field in ("ordinary_target", "forecasts", "forecast_target"))):
                return "inconsistent_same_band_key_value_targets"
    return None


def residual_control_summary(rows):
    """Summarize measured candidates; structural eligibility never passes G4.

    Reports effect on changed, intact and joint predictions versus own-component
    replay, plus signed-permutation effects. G4 still needs reviewed nulls,
    preservation and precision; no architecture-only condition is a pass.
    """
    from .scoring import score_consequence_pattern
    rows = list(rows)
    groups = defaultdict(list)
    for row in rows:
        if row.get("component") == "residual" and row.get("control") in ("natural", "own", "signed_permutation"):
            groups[(row.get("checkpoint_id"), row.get("mask_id"))].append(row)
    result = []
    for (checkpoint, mask), group in sorted(groups.items(), key=lambda item: repr(item[0])):
        reasons = {_receipt_eligibility(row)[1] for row in group if not _receipt_eligibility(row)[0]}
        candidate = {"checkpoint_id": checkpoint, "mask_id": mask, "layers": group[0].get("layers", []),
                     "test_layers": group[0].get("g4_test_layers"),
                     "eligible": not reasons, "ineligible_reasons": sorted(reasons), "effects": {}}
        geometry = group[0].get("model_num_hidden_layers")
        candidate["primary_design_compatible"] = (type(geometry) is int and
            candidate["layers"] == list(range(1, geometry)))
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
        alignment_error = _aligned_component_evidence(rows, group, checkpoint, mask)
        if alignment_error:
            candidate.update(eligible=False, ineligible_reasons=[alignment_error])
            result.append(candidate)
            continue
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
            "review_required": "Structural candidates do not pass G4. The primary design uses every layer from 1 through the final layer; review measured effects and intervals, intact views and nulls without an automatic effect-size cutoff."}
