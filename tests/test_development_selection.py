"""The prospective panel spans LoRA uncertainty without changing exposure."""
import json
from pathlib import Path

import pytest

from paper2.cli import main
from paper2.io import digest, file_digest
from paper2.training.config import TrainingConfig, experiment_matrix, load_config
from paper2.training.selection import (CURRENT_DESIGN, LEGACY_DESIGN, amended_experiment_matrix,
                                       assert_current_design, development_selection_panel,
                                       development_selection_spec)


def test_panel_breadth_and_shared_family_settings_preserve_every_other_choice():
    configs = list(development_selection_panel().values())
    assert len(configs) == 12
    baselines = {config.arm: config.to_dict() for config in amended_experiment_matrix("pilot")}
    expected = {"label_sft": {1e-6, 3e-6, 1e-5, 3e-5},
                "answer_rl": {2e-5, 6e-5, 2e-4, 6e-4},
                "reasoning_rl": {2e-5, 6e-5, 2e-4, 6e-4}}
    for arm, rates in expected.items():
        rows = [config for config in configs if config["arm"] == arm]
        assert len(rows) == 4 and {config["learning_rate"] for config in rows} == rates
        assert {(config["lr_scheduler_type"], config["warmup_ratio"]) for config in rows} == {
            ("constant", 0.0)}
        for config in rows:
            assert_current_design(config)
            assert {k: v for k, v in config.items() if k not in ("learning_rate", "lr_scheduler_type", "warmup_ratio")} == {
                k: v for k, v in baselines[arm].items() if k not in ("learning_rate", "lr_scheduler_type", "warmup_ratio")}
            assert TrainingConfig(**config).training_budget(2000)["planned_prompt_exposures"] == 2000


def test_legacy_panel_requires_explicit_selection_and_retains_original_settings():
    configs = list(development_selection_panel(design=LEGACY_DESIGN).values())
    assert len(configs) == 24
    baselines = {config.arm: config.to_dict() for config in experiment_matrix("pilot")}
    for config in configs:
        assert {key: value for key, value in config.items()
                if key not in ("learning_rate", "lr_scheduler_type", "warmup_ratio")} == {
            key: value for key, value in baselines[config["arm"]].items()
            if key not in ("learning_rate", "lr_scheduler_type", "warmup_ratio")}
        with pytest.raises(ValueError, match="Superseded configuration"):
            assert_current_design(config)
    assert development_selection_spec(design=LEGACY_DESIGN)["candidate_runs"] == 24
    # The actual frozen amended registry uses this original family selection
    # rule verbatim. Record supersession separately, never rewrite that rule.
    assert development_selection_spec()["selection"] == development_selection_spec(design=LEGACY_DESIGN)["selection"]
    with pytest.raises(ValueError, match="Unknown selection design"):
        development_selection_panel(design="implicit_auto")


def test_checked_in_confirmation_is_current_design_but_does_not_grant_approval():
    root = Path(__file__).resolve().parents[1] / 'configs/confirmatory'
    configs = amended_experiment_matrix('confirmatory')
    assert len(configs) == 24
    for config in configs:
        config.validate()
        assert_current_design(config.to_dict())
        name = f'{config.arm}_{config.dataset}_seed{config.seed}.json'
        assert load_config(root / name) == config
    spec = development_selection_spec()
    assert spec['design'] == CURRENT_DESIGN
    assert 'unapproved' in spec['authorization']


def test_checked_in_selection_spec_matches_exported_precommit_rule():
    path = Path(__file__).resolve().parents[1] / 'configs/development_selection.json'
    assert json.loads(path.read_text()) == development_selection_spec()


def test_cli_exports_frozen_panel_with_runnable_configs_and_rejects_overwrite(tmp_path):
    output = tmp_path / 'panel'
    assert main(['matrix', '--stage', 'pilot', '--development-panel', '--output', str(output)]) == 0
    plan = json.loads((output / 'selection_plan.json').read_text())
    assert plan['plan_sha256'] == digest({k: v for k, v in plan.items() if k != 'plan_sha256'})
    assert plan['candidate_runs'] == len(plan['candidates']) == 12
    for candidate in plan['candidates']:
        config = load_config(output / candidate['config_path'])
        assert config.arm == candidate['arm']
        assert file_digest(output / candidate['config_path']) == candidate['config_sha256']
    with pytest.raises(ValueError, match='overwrite'):
        main(['matrix', '--stage', 'pilot', '--development-panel', '--output', str(output)])
    with pytest.raises(ValueError, match='pilot'):
        main(['matrix', '--stage', 'confirmatory', '--development-panel', '--output', str(tmp_path / 'invalid')])
    assert not (tmp_path / 'invalid').exists()


def test_cli_legacy_export_is_explicit_and_current_matrix_uses_amended_identity(tmp_path):
    old = tmp_path / 'historical'
    main(['matrix', '--stage', 'pilot', '--development-panel', '--design', LEGACY_DESIGN, '--output', str(old)])
    assert json.loads((old / 'selection_plan.json').read_text())['candidate_runs'] == 24
    current = tmp_path / 'confirmation'
    main(['matrix', '--stage', 'confirmatory', '--output', str(current)])
    assert len(list(current.glob('*.json'))) == 24
    for path in current.glob('*.json'):
        assert_current_design(load_config(path).to_dict())
