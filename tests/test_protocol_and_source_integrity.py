"""Current results cannot silently restore the legacy completion procedure."""
from copy import deepcopy
import json
import subprocess

import pytest

from paper2.analysis import accuracy_summary, transfer, crossed_summary, match_checkpoints, cue_summary
from paper2.cli import parser, main
from paper2.protocols import completion_protocol
from paper2.traces import run_crossed_reader, regenerate_trace_artifacts
from paper2.training.source_identity import current_source_identity, assert_profile_source

NATURAL = 'natural_answer_cap_v2'
LEGACY = 'legacy_forced_answer_v1'


def test_cli_requires_protocol_before_any_model_is_loaded():
    for command, flag in [('evaluate', '--reader'), ('traces', '--source')]:
        args = [command, '--input', 'in.jsonl', '--output', 'out.jsonl', flag, 'base']
        with pytest.raises(SystemExit):
            parser().parse_args(args)
        for protocol in (NATURAL, LEGACY):
            parsed = parser().parse_args(args + ['--completion-protocol', protocol])
            assert parsed.completion_protocol == protocol


def test_missing_protocol_is_explicitly_historical_and_cannot_join_new_results():
    report = completion_protocol([{}, {'completion_protocol': LEGACY}])
    assert report['completion_protocol'] == LEGACY
    assert report['historical_records_without_protocol'] == 1
    with pytest.raises(ValueError, match='Mixed completion'):
        completion_protocol([{}, {'completion_protocol': NATURAL}])
    with pytest.raises(ValueError, match='conflicting'):
        completion_protocol([{'completion_protocol': NATURAL, 'source_completion_protocol': LEGACY}])


@pytest.mark.parametrize('kind', ['accuracy', 'transfer', 'crossed', 'match', 'cues'])
def test_all_analysis_entrypoints_reject_mixed_protocols_before_computation(kind):
    old = {'completion_protocol': LEGACY}
    new = {'completion_protocol': NATURAL}
    with pytest.raises(ValueError, match='Mixed completion'):
        if kind == 'accuracy':
            accuracy_summary([old, new])
        elif kind == 'transfer':
            transfer([old], [new])
        elif kind == 'crossed':
            crossed_summary([old], [new])
        elif kind == 'match':
            match_checkpoints([old, new])
        else:
            cue_summary([old, new])


def test_cli_analysis_rejects_mix_including_baseline_without_writing_output(tmp_path):
    left, right, output = (tmp_path / name for name in ('left.jsonl', 'right.jsonl', 'out.json'))
    left.write_text(json.dumps({'completion_protocol': LEGACY}) + '\n')
    right.write_text(json.dumps({'completion_protocol': NATURAL}) + '\n')
    with pytest.raises(ValueError, match='Mixed completion'):
        main(['analyze', 'transfer', '--input', str(left), '--baseline', str(right), '--output', str(output)])
    assert not output.exists()


@pytest.mark.parametrize('function', [run_crossed_reader, regenerate_trace_artifacts])
def test_edit_only_native_sources_fail_before_any_test2_call(function):
    traces = [{'purpose': 'g1_editability_only'}]
    with pytest.raises(ValueError, match='G1 editability-only'):
        list(function(None, [], traces, 'source', None))


def test_profile_identity_binds_commit_and_actual_source_bytes(tmp_path):
    def git(*args):
        return subprocess.run(['git', '-C', str(tmp_path), *args], check=True,
                              capture_output=True, text=True).stdout.strip()
    git('init', '-q')
    source = tmp_path / 'src/paper2/example.py'
    source.parent.mkdir(parents=True)
    source.write_text('value = 1\n')
    (tmp_path / 'pyproject.toml').write_text('[project]\nname = "example"\n')
    git('add', '.')
    git('-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', 'commit', '-qm', 'initial')
    measured = current_source_identity(tmp_path)
    assert measured['commit'] == git('rev-parse', 'HEAD')
    assert measured['tracked_changes'] is False
    assert_profile_source({'source_identity': measured}, current_source_identity(tmp_path))
    for key in ('commit', 'tree', 'files_sha256'):
        changed = deepcopy(measured)
        changed[key] = {} if key == 'files_sha256' else 'a' * 40
        with pytest.raises(ValueError, match='reprofile'):
            assert_profile_source({'source_identity': measured}, changed)
    source.write_text('value = 2\n')
    with pytest.raises(ValueError, match='reprofile'):
        assert_profile_source({'source_identity': measured}, current_source_identity(tmp_path))
    with pytest.raises(ValueError, match='reprofile'):
        assert_profile_source({}, measured)


def test_source_identity_refuses_unrecorded_checkout(tmp_path):
    with pytest.raises(ValueError, match='recorded Git'):
        current_source_identity(tmp_path)
