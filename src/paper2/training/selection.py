"""Export the preregistered development panel without launching training."""
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

from .config import MODEL_REVISIONS, NATURAL_COMPLETION_PROTOCOL, experiment_matrix
from ..io import digest, file_digest, write_json

RATE_MULTIPLIERS = (1, 3, 10, 30)
CURRENT_DESIGN = "amended_20261030"
LEGACY_DESIGN = "legacy_proposal"
DESIGNS = (CURRENT_DESIGN, LEGACY_DESIGN)
CURRENT_MODEL = "mistralai/Mistral-Small-24B-Instruct-2501"
MODEL_AMENDMENT_SHA256 = "87dcbf696fb2f995e0f3632713f2fd4b829386494cb135c8502455abdca7aa88"
SCHEDULES = (("constant", 0.0),)
LEGACY_SCHEDULES = (("constant", 0.0), ("constant_with_warmup", 0.05))


def amended_experiment_matrix(stage):
    """Current reference configs, not selected rates or permission to train.

    The legacy factory remains available for reproducing the original proposal.
    The fixed amendment hash identifies the actual model decision; launch gates
    still verify its original evidence against the frozen calibration.
    """
    return [replace(config, model=CURRENT_MODEL, revision=MODEL_REVISIONS[CURRENT_MODEL],
                    model_amendment_sha256=MODEL_AMENDMENT_SHA256,
                    completion_protocol=NATURAL_COMPLETION_PROTOCOL,
                    beta=.04 if config.rl else 0., lr_scheduler_type="constant", warmup_ratio=0.,
                    save_total_limit=2)
            for config in experiment_matrix(stage)]


def assert_current_design(config):
    """Reject superseded configurations for current G1 or new real launches.

    This is a versioned design check, not an empirical performance threshold.
    Reading and reproducing historical scores uses the explicit historical APIs.
    """
    expected = {"model": CURRENT_MODEL, "revision": MODEL_REVISIONS[CURRENT_MODEL],
                "model_amendment_sha256": MODEL_AMENDMENT_SHA256,
                "completion_protocol": NATURAL_COMPLETION_PROTOCOL,
                "beta": .04 if config.get("arm", "").endswith("_rl") else 0.,
                "lr_scheduler_type": "constant", "warmup_ratio": 0.}
    mismatches = [key for key, value in expected.items() if config.get(key) != value]
    if mismatches:
        raise ValueError("Superseded configuration is ineligible for the current 2026-10-30 amendment: "
                         + ", ".join(mismatches) + ". Use explicit historical evidence inspection only.")


def _panel_baselines(design):
    if design not in DESIGNS:
        raise ValueError(f"Unknown selection design {design!r}; choose one of {DESIGNS}")
    return (amended_experiment_matrix("pilot") if design == CURRENT_DESIGN else experiment_matrix("pilot"))


def development_selection_spec(*, design=CURRENT_DESIGN):
    baselines = _panel_baselines(design)
    schedules = SCHEDULES if design == CURRENT_DESIGN else LEGACY_SCHEDULES
    spec = {
        "schema_version": 1,
        "method": "pilot_lora_rate_panel_v1",
        "rate_multipliers": list(RATE_MULTIPLIERS),
        "reference_rates": {"rl": 2e-5, "sft": 1e-6},
        "schedules": [{"lr_scheduler_type": name, "warmup_ratio": warmup} for name, warmup in schedules],
        "pilot_arms": [config.arm for config in baselines],
        "candidate_runs": 24,
        "settings_per_arm": 8,
        "selection": {
            "evaluation": "Final-step native-format accuracy on the same 120 development stories",
            "rl": "Maximize the mean over reasoning_rl and answer_rl at each shared setting",
            "sft": "Maximize label_sft accuracy; reuse the selected setting for both SFT arms",
            "ties": ["smaller learning rate", "no warmup"],
            "failures": "Retain every failed run; no automatic omission, favorable replacement or expanded search",
            "freeze": "Freeze the costed panel before pilot learning and selected family settings before confirmation",
            "confirmation": "Never use confirmation outcomes for selection",
        },
        "source_caution": "Zhong reports these rates but does not specify LoRA versus full fine-tuning. "
                          "They are reference candidates, not established LoRA optima. The approximate 10x "
                          "LoRA multiplier is empirical motivation for breadth, not a guaranteed conversion.",
        "sources": ["https://arxiv.org/html/2606.09092v1", "https://thinkingmachines.ai/blog/lora/"],
        "resource_note": "Eight settings per pilot arm: 24 runs, eight times the one-setting pilot run count. "
                         "Profile and cost all candidates, evaluation and retained adapters before selecting this panel. "
                         "No training or purchase is launched by export.",
    }
    if design == CURRENT_DESIGN:
        spec.update({"schema_version": 2, "method": "amended_pilot_lora_rate_panel_20261030",
                     "design": design, "candidate_runs": 12, "settings_per_arm": 4,
                     "model": CURRENT_MODEL, "revision": MODEL_REVISIONS[CURRENT_MODEL],
                     "model_amendment_sha256": MODEL_AMENDMENT_SHA256,
                     "completion_protocol": NATURAL_COMPLETION_PROTOCOL,
                     "beta_by_family": {"rl": .04, "sft": 0.},
                     "excluded_panels": ["superseded pre-amendment completion/KL panel"],
                     "initialization": "Each candidate independently initializes from the pinned untrained base",
                     "authorization": "Export only; requires a separately frozen costed registry and measured gates. Confirmation remains unapproved.",
                     "resource_note": "Four constant/no-warmup settings per pilot arm: 12 full-exposure runs. "
                                      "Bind current source, same-protocol profiles, corpus and cost evidence before "
                                      "candidate learning. Export does not launch training or authorize confirmation."})
    return spec


def development_selection_panel(*, design=CURRENT_DESIGN):
    """Return independently initialized configs, changing only rate and schedule."""
    configs = {}
    for baseline in _panel_baselines(design):
        for multiplier in RATE_MULTIPLIERS:
            rate = float(Decimal(str(baseline.learning_rate)) * multiplier)
            for schedule, warmup in (SCHEDULES if design == CURRENT_DESIGN else LEGACY_SCHEDULES):
                config = replace(baseline, learning_rate=rate, lr_scheduler_type=schedule, warmup_ratio=warmup)
                config.validate()
                name = f"{config.arm}_clean_seed1_lr-x{multiplier}_{schedule}.json"
                configs[name] = config.to_dict()
    return configs


def export_development_panel(output, *, design=CURRENT_DESIGN):
    """Write a fresh hashed panel; never overwrite a frozen candidate config."""
    destination = Path(output)
    configs = development_selection_panel(design=design)
    filenames = [*configs, "selection_plan.json"]
    if any((destination / filename).exists() for filename in filenames):
        raise ValueError("Development panel already exists; refusing to overwrite frozen candidates")
    for name, config in configs.items():
        write_json(destination / name, config)
    plan = {**development_selection_spec(design=design),
            "candidates": [{"config_path": name, "config_sha256": file_digest(destination / name),
                            "arm": config["arm"], "learning_rate": config["learning_rate"],
                            "lr_scheduler_type": config["lr_scheduler_type"], "warmup_ratio": config["warmup_ratio"]}
                           for name, config in configs.items()]}
    plan["plan_sha256"] = digest(plan)
    write_json(destination / "selection_plan.json", plan)
    return plan
