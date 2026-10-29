"""Export the preregistered development panel without launching training."""
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

from .config import experiment_matrix
from ..io import digest, file_digest, write_json

RATE_MULTIPLIERS = (1, 3, 10, 30)
SCHEDULES = (("constant", 0.0), ("constant_with_warmup", 0.05))


def development_selection_spec():
    return {
        "schema_version": 1,
        "method": "pilot_lora_rate_panel_v1",
        "rate_multipliers": list(RATE_MULTIPLIERS),
        "reference_rates": {"rl": 2e-5, "sft": 1e-6},
        "schedules": [{"lr_scheduler_type": name, "warmup_ratio": warmup} for name, warmup in SCHEDULES],
        "pilot_arms": [config.arm for config in experiment_matrix("pilot")],
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


def development_selection_panel():
    """Return independently initialized configs, changing only rate and schedule."""
    configs = {}
    for baseline in experiment_matrix("pilot"):
        for multiplier in RATE_MULTIPLIERS:
            rate = float(Decimal(str(baseline.learning_rate)) * multiplier)
            for schedule, warmup in SCHEDULES:
                config = replace(baseline, learning_rate=rate, lr_scheduler_type=schedule, warmup_ratio=warmup)
                config.validate()
                name = f"{config.arm}_clean_seed1_lr-x{multiplier}_{schedule}.json"
                configs[name] = config.to_dict()
    return configs


def export_development_panel(output):
    """Write a fresh hashed panel; never overwrite a frozen candidate config."""
    destination = Path(output)
    configs = development_selection_panel()
    filenames = [*configs, "selection_plan.json"]
    if any((destination / filename).exists() for filename in filenames):
        raise ValueError("Development panel already exists; refusing to overwrite frozen candidates")
    for name, config in configs.items():
        write_json(destination / name, config)
    plan = {**development_selection_spec(),
            "candidates": [{"config_path": name, "config_sha256": file_digest(destination / name),
                            "arm": config["arm"], "learning_rate": config["learning_rate"],
                            "lr_scheduler_type": config["lr_scheduler_type"], "warmup_ratio": config["warmup_ratio"]}
                           for name, config in configs.items()]}
    plan["plan_sha256"] = digest(plan)
    write_json(destination / "selection_plan.json", plan)
    return plan
