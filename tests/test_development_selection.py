"""The prospective panel spans LoRA uncertainty without changing exposure."""
import json
from pathlib import Path

import pytest

from paper2.cli import main
from paper2.io import digest, file_digest
from paper2.training.config import TrainingConfig, experiment_matrix, load_config
from paper2.training.selection import development_selection_panel, development_selection_spec


def test_panel_breadth_and_shared_family_settings_preserve_every_other_choice():
    configs = list(development_selection_panel().values())
    assert len(configs) == 24
    baselines = {config.arm: config.to_dict() for config in experiment_matrix("pilot")}
    expected = {"label_sft": {1e-6, 3e-6, 1e-5, 3e-5},
                "answer_rl": {2e-5, 6e-5, 2e-4, 6e-4},
                "reasoning_rl": {2e-5, 6e-5, 2e-4, 6e-4}}
    for arm, rates in expected.items():
        rows = [config for config in configs if config["arm"] == arm]
        assert len(rows) == 8 and {config["learning_rate"] for config in rows} == rates
        assert {(config["lr_scheduler_type"], config["warmup_ratio"]) for config in rows} == {
            ("constant", 0.0), ("constant_with_warmup", .05)}
        for config in rows:
            assert {k: v for k, v in config.items() if k not in ("learning_rate", "lr_scheduler_type", "warmup_ratio")} == {
                k: v for k, v in baselines[arm].items() if k not in ("learning_rate", "lr_scheduler_type", "warmup_ratio")}
            assert TrainingConfig(**config).training_budget(2000)["planned_prompt_exposures"] == 2000


def test_checked_in_selection_spec_matches_exported_precommit_rule():
    path = Path(__file__).resolve().parents[1] / 'configs/development_selection.json'
    assert json.loads(path.read_text()) == development_selection_spec()


def test_cli_exports_frozen_panel_with_runnable_configs_and_rejects_overwrite(tmp_path):
    output = tmp_path / 'panel'
    assert main(['matrix', '--stage', 'pilot', '--development-panel', '--output', str(output)]) == 0
    plan = json.loads((output / 'selection_plan.json').read_text())
    assert plan['plan_sha256'] == digest({k: v for k, v in plan.items() if k != 'plan_sha256'})
    for candidate in plan['candidates']:
        config = load_config(output / candidate['config_path'])
        assert config.arm == candidate['arm']
        assert file_digest(output / candidate['config_path']) == candidate['config_sha256']
    with pytest.raises(ValueError, match='overwrite'):
        main(['matrix', '--stage', 'pilot', '--development-panel', '--output', str(output)])
    with pytest.raises(ValueError, match='pilot'):
        main(['matrix', '--stage', 'confirmatory', '--development-panel', '--output', str(tmp_path / 'invalid')])
    assert not (tmp_path / 'invalid').exists()
