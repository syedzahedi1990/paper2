"""Exercise the operator commands on a complete synthetic panel, without models."""
import json
import shutil

import pytest

from paper2.cli import main
from paper2.io import file_digest
from panel_fixture import make_panel_fixture


def test_cli_computes_reverifies_and_exports_selected_family_rates(tmp_path, capsys):
    panel = make_panel_fixture(tmp_path / 'panel', scores={
        'reasoning_rl_clean_seed1_lr-x3_constant': 119,
        'answer_rl_clean_seed1_lr-x3_constant': 119,
        'label_sft_clean_seed1_lr-x10_constant': 119})
    try:
        policy, selection = tmp_path / 'policy.json', tmp_path / 'selection.json'
        main(['freeze-selection-failure-policy', '--output', str(policy), '--reviewer', 'CPU test only',
              '--rationale', 'Exercise immutable prospective policy CLI'])
        assert json.loads(policy.read_text(encoding='utf-8'))['status'] == 'frozen'
        with pytest.raises(FileExistsError):
            main(['freeze-selection-failure-policy', '--output', str(policy), '--reviewer', 'CPU test only',
                  '--rationale', 'Must not overwrite'])
        main(['select-panel', '--execution-binding', str(panel['binding']), '--sha256', file_digest(panel['binding']),
              '--panel-status', str(panel['status']), '--failure-policy', str(policy),
              '--failure-policy-sha256', file_digest(policy), '--output', str(selection),
              '--reviewer', 'CPU fixture', '--rationale', 'Apply final-native family objective'])
        main(['verify-panel-selection', '--selection', str(selection), '--sha256', file_digest(selection)])
        grid = tmp_path / 'confirmation'
        main(['selected-confirmatory-configs', '--selection', str(selection),
              '--selection-sha256', file_digest(selection), '--output', str(grid)])
        manifest = json.loads((grid / 'selection_manifest.json').read_text(encoding='utf-8'))
        assert len(manifest['configs']) == 24
        assert manifest['selected_settings']['rl']['learning_rate'] == 6e-5
        assert manifest['selected_settings']['sft']['learning_rate'] == 1e-5
        assert '"verified": true' in capsys.readouterr().out
    finally:
        shutil.rmtree(tmp_path / 'panel')
