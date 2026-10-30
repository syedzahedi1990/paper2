"""Current launches cannot borrow an old pass or silently use old settings."""
from dataclasses import replace

import pytest

from paper2.gates import assert_training_allowed
from paper2.training.config import LEGACY_COMPLETION_PROTOCOL, TrainingConfig
from paper2.training.selection import amended_experiment_matrix, assert_current_design


@pytest.mark.parametrize('stage', ['pilot', 'profile', 'confirmatory'])
@pytest.mark.parametrize('field,value', [
    ('completion_protocol', LEGACY_COMPLETION_PROTOCOL), ('beta', 0.),
    ('model', 'Qwen/Qwen2.5-3B-Instruct'), ('revision', 'a' * 40),
    ('model_amendment_sha256', None), ('lr_scheduler_type', 'constant_with_warmup'),
    ('warmup_ratio', .05),
])
def test_superseded_launch_fails_before_any_gate_or_profile_can_authorize_it(stage, field, value):
    current = amended_experiment_matrix('pilot')[0]
    config = replace(current, stage=stage).to_dict()
    config[field] = value
    with pytest.raises(ValueError, match='Superseded configuration'):
        assert_training_allowed(config, 'not-read-old-pass.json', 'not-read-old-profile.json')


def test_old_default_and_v2_with_old_beta_are_not_implicitly_current():
    legacy = TrainingConfig().to_dict()
    with pytest.raises(ValueError, match='Superseded configuration'):
        assert_current_design(legacy)
    current = amended_experiment_matrix('pilot')[0].to_dict()
    del current['beta']
    with pytest.raises(ValueError, match='beta'):
        assert_current_design(current)


def test_sft_has_no_kl_and_current_templates_still_need_measured_authorization():
    for config in amended_experiment_matrix('confirmatory'):
        assert config.beta == (.04 if config.rl else 0.)
        with pytest.raises(ValueError, match='calibration gate file'):
            assert_training_allowed(config.to_dict(), None, None)
    sft = next(config for config in amended_experiment_matrix('pilot') if config.arm == 'label_sft')
    with pytest.raises(ValueError, match='beta'):
        assert_current_design(replace(sft, beta=.04).to_dict())
