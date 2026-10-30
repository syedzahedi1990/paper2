"""Prepared G1 inputs own one editor job, including interrupted-call evidence."""
import json
from pathlib import Path

import pytest

from paper2.training import g1_editability as helper
import test_g1_editability as fixtures

MeasuredFakeEditor = fixtures.MeasuredFakeEditor
observed_fixture = fixtures.observed_fixture
workflow_fixture = fixtures.workflow_fixture


@pytest.mark.parametrize('resume', [False, True])
def test_same_preparation_cannot_start_another_output(workflow_fixture, tmp_path, resume):
    prepared, sha = workflow_fixture
    first = tmp_path / 'first.json'
    helper.generate(prepared, sha, first, backend=MeasuredFakeEditor(), max_calls=1)
    second = MeasuredFakeEditor()
    with pytest.raises(ValueError, match='explicit resume|different prepared inputs/output'):
        helper.generate(prepared, sha, tmp_path / 'second.json', backend=second, resume=resume)
    assert second.calls == []


def test_actual_interrupted_call_is_unresolved_and_never_resampled(workflow_fixture, tmp_path):
    class Interrupted(MeasuredFakeEditor):
        def generate(self, prompt, **kwargs):
            self.calls.append(prompt)
            raise KeyboardInterrupt('Synthetic interruption during an actual call')

    prepared, sha = workflow_fixture
    output = tmp_path / 'interrupted.json'
    first = Interrupted()
    with pytest.raises(KeyboardInterrupt):
        helper.generate(prepared, sha, output, backend=first)
    assert len(first.calls) == 1
    assert len(helper.read_jsonl(str(output) + '.started.jsonl')) == 1
    assert Path(str(output) + '.journal.jsonl').read_bytes() == b''
    second = MeasuredFakeEditor()
    with pytest.raises(ValueError, match='interrupted unresolved call'):
        helper.generate(prepared, sha, output, backend=second, resume=True)
    assert second.calls == [] and not output.exists()


@pytest.mark.parametrize('kind', ['partial', 'changed', 'extra'])
def test_changed_call_start_journal_stops_before_more_inference(workflow_fixture, tmp_path, kind):
    prepared, sha = workflow_fixture
    output = tmp_path / 'generated.json'
    helper.generate(prepared, sha, output, backend=MeasuredFakeEditor(), max_calls=1)
    started = Path(str(output) + '.started.jsonl')
    if kind == 'partial':
        with started.open('a', encoding='utf-8') as stream:
            stream.write('{"artifact_id":')
    elif kind == 'extra':
        with started.open('a', encoding='utf-8') as stream:
            stream.write(started.read_text(encoding='utf-8'))
    else:
        begin, = helper.read_jsonl(started)
        begin['prompt_sha256'] = 'f' * 64
        started.write_text(json.dumps(begin) + '\n', encoding='utf-8')
    next_editor = MeasuredFakeEditor()
    with pytest.raises(ValueError):
        helper.generate(prepared, sha, output, backend=next_editor, resume=True)
    assert next_editor.calls == []


def test_different_output_cannot_bypass_preparation_os_lock(workflow_fixture, tmp_path):
    prepared, sha = workflow_fixture
    editor = MeasuredFakeEditor()
    with helper._job_lock(Path(str(prepared.resolve()) + '.generation.lock')):
        with pytest.raises(ValueError, match='Another G1 editor process'):
            helper.generate(prepared, sha, tmp_path / 'any-output.json', backend=editor)
    assert editor.calls == []


def test_final_start_result_consistency_is_checked_before_publish(workflow_fixture, tmp_path, monkeypatch):
    prepared, sha = workflow_fixture
    output = tmp_path / 'generated.json'
    original = helper.generate_edit_controls

    def corrupt_after_calls(*args, **kwargs):
        result = original(*args, **kwargs)
        started = Path(str(output) + '.started.jsonl')
        with started.open('a', encoding='utf-8') as stream:
            stream.write('{"artifact_id":"unexpected"}\n')
        return result

    monkeypatch.setattr(helper, 'generate_edit_controls', corrupt_after_calls)
    editor = MeasuredFakeEditor()
    with pytest.raises(ValueError, match='interrupted unresolved call'):
        helper.generate(prepared, sha, output, backend=editor)
    assert len(editor.calls) == 3 and not output.exists()
    assert len(helper.read_jsonl(str(output) + '.journal.jsonl')) == 3


def test_call_start_write_failure_stops_before_backend_or_later_contract(workflow_fixture, tmp_path, monkeypatch):
    prepared, sha = workflow_fixture
    output = tmp_path / 'generated.json'
    original = helper.json.dumps

    def failed_start(value, *args, **kwargs):
        if isinstance(value, dict) and set(value) == {'artifact_id', 'prompt_sha256', 'plan_sha256'}:
            raise OSError('Synthetic call-start journal failure')
        return original(value, *args, **kwargs)

    monkeypatch.setattr(helper.json, 'dumps', failed_start)
    editor = MeasuredFakeEditor()
    with pytest.raises(helper._CallStartWriteFailure, match='journal failure'):
        helper.generate(prepared, sha, output, backend=editor)
    assert editor.calls == [] and not output.exists()
    assert Path(str(output) + '.journal.jsonl').read_bytes() == b''
