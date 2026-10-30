"""Derive confirmatory settings from the reverified development-panel winner.

The checked-in matrix is a design template, not a selected learning rate. This
module never starts training or changes a frozen gate review.
"""
from dataclasses import replace
from pathlib import Path

from ..io import file_digest, write_json
from .selection import amended_experiment_matrix


SETTING_KEYS = ("learning_rate", "lr_scheduler_type", "warmup_ratio")
SELECTED_PILOT_ARMS = ("reasoning_rl", "answer_rl", "label_sft")


def _verified_selection(path, expected_sha256=None):
    # Import lazily: exporting a design template or launching a pilot does not
    # need a completed panel or any of its private result files.
    from .selection_results import verify_panel_selection
    return verify_panel_selection(path, expected_sha256=expected_sha256)


def _family(arm):
    if arm not in ("reasoning_rl", "answer_rl", "label_sft", "reasoning_sft"):
        raise ValueError("Confirmatory selection requires a declared training arm")
    return "rl" if arm.endswith("_rl") else "sft"


def _selected_setting(selection, arm):
    family = _family(arm)
    setting = selection.get("selected_settings", {}).get(family)
    if not isinstance(setting, dict) or set(setting) != set(SETTING_KEYS):
        raise ValueError(f"Computed selection lacks the complete {family} family setting")
    return setting


def verify_confirmatory_selection(config, review, root):
    """Recompute selection, bind reviewed selected curves, and check this run.

    A receipt name, editable selected-rate field, or a rehashed forged winner
    does not suffice: the panel verifier reads the frozen registry and actual
    terminal scores again. Both SFT arms inherit the label-SFT selection, and
    both RL arms inherit the shared RL-family selection.
    """
    binding = review.get("development_selection")
    if (not isinstance(binding, dict) or not isinstance(binding.get("path"), str)
            or not binding["path"] or not isinstance(binding.get("sha256"), str)
            or len(binding["sha256"]) != 64):
        raise ValueError("Confirmatory runs require a bound computed development_selection receipt")
    path = Path(root) / binding["path"]
    if not path.is_file() or file_digest(path) != binding["sha256"]:
        raise ValueError("Confirmatory development selection receipt is missing or modified")
    g1 = review.get("gates", {}).get("G1", {})
    if not any((Path(root) / item.get("path", "")).resolve() == path.resolve()
               and item.get("sha256") == binding["sha256"] for item in g1.get("evidence", [])):
        raise ValueError("Computed development selection must be included in reviewed G1 evidence")
    selection = _verified_selection(path, expected_sha256=binding["sha256"])
    if selection.get("corpus_manifest_sha256") != review.get("calibration", {}).get("manifest_sha256"):
        raise ValueError("Computed development selection belongs to a different frozen corpus")
    selected_curves = selection.get("selected_run_evidence", {})
    for arm in SELECTED_PILOT_ARMS:
        chosen = selected_curves.get(arm, {})
        entries = [entry for entry in g1.get("learning_curves", []) if entry.get("arm") == arm]
        if (len(entries) != 1 or not chosen.get("path") or not chosen.get("sha256")
                or (Path(root) / entries[0].get("path", "")).resolve()
                != (path.parent / chosen["path"]).resolve()
                or entries[0].get("sha256") != chosen["sha256"]):
            raise ValueError(f"G1 {arm} curve must be the candidate chosen by the computed panel selection")
    setting = _selected_setting(selection, config.get("arm"))
    mismatch = [key for key in SETTING_KEYS if config.get(key) != setting[key]]
    if mismatch:
        raise ValueError("Confirmatory settings differ from the computed selected family setting: "
                         + ", ".join(mismatch))
    return {"path": str(path.resolve()), "sha256": binding["sha256"],
            "method": selection["method"], "family": _family(config["arm"]),
            "selected_setting": dict(setting),
            "corpus_manifest_sha256": selection["corpus_manifest_sha256"]}


def export_selected_confirmatory_configs(selection_receipt, output, *, selection_sha256=None):
    """Write a fresh complete 24-run grid after recomputing the panel selection.

    Export is not gate or compute approval. The launch gate independently
    rechecks the same receipt and its underlying panel, including on profiles.
    """
    source = Path(selection_receipt).resolve()
    selection = _verified_selection(source, expected_sha256=selection_sha256)
    binding = {"path": str(source), "sha256": file_digest(source)}
    destination = Path(output)
    configs = {}
    for baseline in amended_experiment_matrix("confirmatory"):
        config = replace(baseline, **_selected_setting(selection, baseline.arm))
        config.validate()
        name = f"{config.arm}_{config.dataset}_seed{config.seed}.json"
        configs[name] = config.to_dict()
    if any((destination / name).exists() for name in (*configs, "selection_manifest.json")):
        raise ValueError("Confirmatory export already exists; refusing to overwrite selected configs")
    for name, config in configs.items():
        write_json(destination / name, config)
    manifest = {"schema_version": 1, "method": "selected_confirmatory_config_export_v1",
                "development_selection": binding,
                "corpus_manifest_sha256": selection["corpus_manifest_sha256"],
                "selected_settings": selection["selected_settings"],
                "configs": [{"path": name, "sha256": file_digest(destination / name),
                             "arm": config["arm"], "dataset": config["dataset"], "seed": config["seed"]}
                            for name, config in configs.items()],
                "authorization": "Export only. All scientific gates, profile/source checks and cost approval remain required."}
    write_json(destination / "selection_manifest.json", manifest)
    return manifest
