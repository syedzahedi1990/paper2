"""Exact quotations remove model counting without weakening span validation."""
from copy import deepcopy
import json
import random

import pytest

from paper2.data.world import generate_world, render_record
from paper2.edit_generation import (CONTROL_PROMPT, LEGACY_CONTROL_PROMPT, LEGACY_PROMPT_VERSION,
    PROMPT_VERSION, ControlOutputError, _json_object, _parse_controls, _quote_span,
    generate_edit_controls, validate_generated_batch)
from paper2.editor_calibration import prepare_oracle_calibration, report_oracle_calibration
from paper2.io import digest, read_jsonl, write_json, write_jsonl
from test_generated_edits import Editor, fixture


def test_quotes_resolve_far_offset_without_model_counting_and_preserve_newlines():
    source = 'prefix' * 107 + '\nAlice believes the phone is in the box.\nNext event.'
    old = '\nAlice believes the phone is in the box.'
    span = _quote_span(source, {'old': old, 'new': old.replace('box', 'basket')})
    assert span['start'] == 642 and span['end'] == 642 + len(old)
    assert source[span['start']:span['end']] == old
    assert 'start' not in CONTROL_PROMPT and 'end is exclusive' not in CONTROL_PROMPT


@pytest.mark.parametrize('source,old,code', [
    ('A box and a box.', 'box', 'quote_not_unique'),
    ('aaaa', 'aaa', 'quote_not_unique'),
    ('A box.', 'basket', 'quote_not_found'),
    ('A\nbox.', r'A\nbox.', 'quote_not_found'),
])
def test_absent_and_ambiguous_quotes_have_distinct_causes(source, old, code):
    with pytest.raises(ControlOutputError) as error:
        _quote_span(source, {'old': old, 'new': 'a replacement'})
    assert error.value.code == code


def test_unique_context_resolves_repeated_location_without_relaxing_edit():
    trace = 'Bob believes the phone is in the box. Alice believes the phone is in the box.'
    old = 'Alice believes the phone is in the box.'
    result = _quote_span(trace, {'old': old, 'new': old.replace('box', 'basket')})
    assert result['start'] == trace.index('Alice')
    assert trace[:result['start']] == 'Bob believes the phone is in the box. '


@pytest.mark.parametrize('wrapper', ['```json\n%s\n```', '```\n%s\n```', '  ```json\r\n%s\r\n```\n'])
def test_one_complete_fence_is_accepted_and_canonical_offsets_recorded(wrapper):
    traces, manifest, plan = fixture(worlds=1, sources=1)
    batch = generate_edit_controls(Editor(lambda output: wrapper % json.dumps(output)), traces, manifest, plan)
    contract = next(iter(validate_generated_batch(batch).values()))
    assert contract['generation_status'] == 'ready'
    assert set(contract['semantic']) >= {'start', 'end', 'old', 'new'}
    assert contract['generation']['raw_output'].strip().startswith('```')


@pytest.mark.parametrize('raw', [
    'Here is the answer:\n```json\n{}\n```', '```json\n{}\n```\nCommentary',
    '```json\n{}\n```\n```json\n{}\n```', '```python\n{}\n```',
    '[{}]', '{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}',
])
def test_fence_support_keeps_strict_json_object_boundary(raw):
    with pytest.raises(ControlOutputError):
        _json_object(raw, allow_fence=True)


def test_failures_by_cause_keep_all_frozen_inputs_and_resume_without_new_calls():
    traces, manifest, plan = fixture(worlds=3, sources=1)

    class MixedEditor(Editor):
        def generate(self, prompt, **decoding):
            result = super().generate(prompt, **decoding)
            if len(self.calls) == 1:
                output = json.loads(result['text'])
                output['semantic']['old'] = 'Absent premise'
                result['text'] = json.dumps(output)
            elif len(self.calls) == 2:
                result['text'] = 'not JSON'
            return result

    batch = generate_edit_controls(MixedEditor(), traces, manifest, plan)
    report = batch['generated_batch']['outcome_report']
    assert report['population_size'] == 3
    assert report['status']['ready'] == {'count': 1, 'denominator': 3, 'rate': 1 / 3}
    assert set(report['operational_failure_causes']) == {'quote_not_found', 'invalid_json'}
    assert all(item['denominator'] == 3 for item in report['operational_failure_causes'].values())
    resumed = Editor()
    assert generate_edit_controls(resumed, traces, manifest, plan, completed_results=batch['edits']) == batch
    assert not resumed.calls
    changed = deepcopy(batch)
    changed['generated_batch']['outcome_report']['population_size'] = 2
    evidence = changed['generated_batch']
    evidence['batch_sha256'] = digest({key: value for key, value in evidence.items() if key != 'batch_sha256'})
    with pytest.raises(ValueError, match='cause report'):
        validate_generated_batch(changed)


def test_quote_protocol_rejects_model_offsets():
    traces, manifest, _ = fixture(worlds=1, sources=1)
    row = manifest['edits'][0]
    before = row['semantic']['oracle']['before']
    output = {'status': 'ready', 'stated_location': before, 'paraphrase': traces[0]['trace'] + ' ',
              'semantic': {'old': before, 'new': row['semantic']['oracle']['after'], 'start': 0, 'end': 1},
              'irrelevant': {'old': 'Then', 'new': 'Next'}}
    with pytest.raises(ControlOutputError) as error:
        _parse_controls(json.dumps(output), traces[0]['trace'], row, prompt_version=PROMPT_VERSION)
    assert error.value.code == 'invalid_output_schema'


def test_frozen_legacy_plan_uses_original_prompt_offsets_and_unfenced_parser():
    traces, manifest, plan = fixture(worlds=1, sources=1)
    plan.update(prompt_version=LEGACY_PROMPT_VERSION, prompt=LEGACY_CONTROL_PROMPT,
                prompt_sha256=digest(LEGACY_CONTROL_PROMPT))
    plan['plan_sha256'] = digest({key: value for key, value in plan.items() if key != 'plan_sha256'})

    class LegacyEditor(Editor):
        def generate(self, prompt, **decoding):
            assert LEGACY_CONTROL_PROMPT in prompt
            original = prompt
            prompt = prompt.replace(LEGACY_CONTROL_PROMPT, CONTROL_PROMPT)
            result = super().generate(prompt, **decoding)
            self.calls[-1] = (original, decoding)
            source = json.loads(prompt[len('<user>' + CONTROL_PROMPT):-len('</user><assistant>')])['trace']
            controls = json.loads(result['text'])
            for name in ('semantic', 'irrelevant'):
                controls[name] = _quote_span(source, controls[name])
            result['text'] = json.dumps(controls)
            return result

    batch = generate_edit_controls(LegacyEditor(), traces, manifest, plan)
    assert validate_generated_batch(batch)
    assert 'failure_code' not in batch['edits'][0]['generation']
    assert 'outcome_report' not in batch['generated_batch']
    resumed = LegacyEditor()
    assert generate_edit_controls(resumed, traces, manifest, plan, completed_results=batch['edits']) == batch
    assert not resumed.calls
    with pytest.raises(ControlOutputError):
        _json_object('```json\n{}\n```')


def test_io_unicode_and_lf_bytes_are_platform_independent(tmp_path):
    rows = [{'trace': 'A café\nAlice’s belief: 箱', 'label': 'é'}]
    jsonl = tmp_path / 'journal.jsonl'
    write_jsonl(jsonl, rows)
    expected = (json.dumps(rows[0], sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')
    assert jsonl.read_bytes() == expected
    assert read_jsonl(jsonl) == rows
    value = tmp_path / 'receipt.json'
    write_json(value, rows)
    assert value.read_bytes() == (json.dumps(rows, indent=2, sort_keys=True, allow_nan=False) + '\n').encode('utf-8')


def training_rows(count=14):
    return [render_record(generate_world(random.Random(seed), split='train'), 'direct') for seed in range(count)]


def test_oracle_calibration_is_training_only_deterministic_and_explicitly_not_model_native():
    rows = training_rows()
    prepared = prepare_oracle_calibration(rows)
    assert prepared == prepare_oracle_calibration(list(reversed(rows)))
    assert len(prepared['traces']) == 12
    assert prepared['generation_plan']['prompt_version'] == PROMPT_VERSION
    assert prepared['generation_plan']['decoding']['do_sample'] is False
    for trace in prepared['traces']:
        assert trace['purpose'] == 'editor_oracle_calibration'
        assert trace['model_generated'] is False and trace['replay_eligible'] is False
        assert not ({'generation_prompt_ids', 'completion_generated_ids', 'native_outcome'} & trace.keys())
    assert prepared['calibration']['execution']['attention'] == 'sdpa'
    assert prepared['generation_plan']['resource_counts']['max_output_tokens'] == 12 * 2048


@pytest.mark.parametrize('mutation', ['split', 'world_split', 'dataset', 'answer', 'duplicate', 'too_few'])
def test_calibration_rejects_heldout_inconsistent_or_incomplete_inputs(mutation):
    rows = training_rows()
    if mutation == 'world_split':
        rows[0]['world']['split'] = 'confirmation'
    elif mutation == 'split':
        rows[0]['split'] = 'development'
    elif mutation == 'dataset':
        rows[0]['dataset'] = 'shortcut'
    elif mutation == 'answer':
        rows[0]['answer'] = 'Z'
    elif mutation == 'duplicate':
        rows[-1] = rows[0]
    else:
        rows = rows[:11]
    with pytest.raises(ValueError):
        prepare_oracle_calibration(rows)


def test_calibration_report_preserves_failures_without_quality_approval():
    prepared = prepare_oracle_calibration(training_rows())

    class DiagnosticEditor(Editor):
        checkpoint_id = prepared['generation_plan']['model']['id']
        revision = prepared['generation_plan']['model']['revision']

        def generate(self, prompt, **decoding):
            self.calls.append((prompt, decoding))
            return {'text': 'invalid JSON', 'truncated': False}

    editor = DiagnosticEditor()
    batch = generate_edit_controls(editor, prepared['traces'], prepared['manifest'], prepared['generation_plan'])
    report = report_oracle_calibration(prepared, batch)
    assert len(editor.calls) == 12 and len(report['items']) == 12
    assert report['outcomes']['operational_failure_causes']['invalid_json']['count'] == 12
    assert report['semantic_approval'] is False
    assert all(item['source_characters'] > 0 for item in report['items'])
    changed = deepcopy(prepared)
    changed['traces'][0]['purpose'] = 'production'
    with pytest.raises(ValueError, match='frozen inputs'):
        report_oracle_calibration(changed, batch)


def test_failure_cause_is_recomputed_from_saved_output_not_only_trusted_as_label():
    from paper2.edit_generation import generation_outcome_report
    traces, manifest, plan = fixture(worlds=1, sources=1)
    batch = generate_edit_controls(Editor(lambda _: 'invalid JSON'), traces, manifest, plan)
    contract = batch['edits'][0]
    evidence = contract['generation']
    evidence['failure_code'] = 'quote_not_found'
    evidence['generation_sha256'] = digest({key: value for key, value in evidence.items() if key != 'generation_sha256'})
    certificate = batch['generated_batch']
    certificate['contract_content_sha256'] = digest(batch['edits'])
    certificate['outcome_report'] = generation_outcome_report(batch['edits'])
    certificate['batch_sha256'] = digest({key: value for key, value in certificate.items() if key != 'batch_sha256'})
    with pytest.raises(ValueError, match='failure cause'):
        validate_generated_batch(batch)


@pytest.fixture(scope='module')
def calibration_files(tmp_path_factory):
    """Synthetic training pool only; no private data, outcome logs or model calls."""
    from paper2.io import file_digest
    from paper2.data.corpus import CORPUS_SCHEMA
    root = tmp_path_factory.mktemp('oracle-calibration-inputs')
    rows = training_rows(2000)
    training, worlds, manifest = root / 'clean.jsonl', root / 'train.jsonl', root / 'manifest.json'
    write_jsonl(training, [{key: value for key, value in row.items() if key != 'world'} for row in rows])
    write_jsonl(worlds, [row['world'] for row in rows])
    corpus = {'schema_version': CORPUS_SCHEMA, 'config': {'train': 2000}, 'shortcut_audit': {},
              'files': {'training/clean.jsonl': {'rows': 2000, 'sha256': file_digest(training)},
                        'worlds/train.jsonl': {'rows': 2000, 'sha256': file_digest(worlds)}}}
    corpus['corpus_hash'] = digest(corpus)
    write_json(manifest, corpus)
    return training, worlds, manifest


@pytest.fixture
def file_preparation(calibration_files, tmp_path, monkeypatch):
    from paper2.editor_calibration import prepare_files
    from paper2.training import source_identity
    # Root may be concurrently editing another source file; this fixture tests
    # the exact identity binding separately, not the Git inventory implementation.
    monkeypatch.setattr(source_identity, 'current_source_identity', lambda: {'fixture': 'synthetic frozen source'})
    result = prepare_files(*calibration_files, tmp_path / 'prepared')
    return result['preparation']['path'], result['preparation']['sha256']


class CalibrationFakeEditor(Editor):
    from paper2.editor_calibration import EDITOR_MODEL, EDITOR_REVISION
    checkpoint_id = EDITOR_MODEL
    revision = EDITOR_REVISION
    attention_implementation = 'sdpa'

    def token_ids(self, prompt):
        return [7] * len(prompt.split())

    def synchronize(self):
        pass

    def generate(self, prompt, **decoding):
        self.calls.append((prompt, decoding))
        return {'text': 'invalid JSON', 'truncated': False, 'generated_ids': [10, 11, 2],
                'prompt_ids': self.token_ids(prompt)}


def test_file_preparation_is_exclusive_and_binds_all_inputs(calibration_files, file_preparation):
    from pathlib import Path
    from paper2.editor_calibration import prepare_files
    path, sha = file_preparation
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    assert len(value['traces']) == 12
    assert value['calibration']['training_population_size'] == 2000
    assert value['source_identity'] == {'fixture': 'synthetic frozen source'}
    assert set(value['input_files']) == {'training', 'worlds', 'corpus_manifest'}
    assert sha
    with pytest.raises(FileExistsError):
        prepare_files(*calibration_files, Path(path).parent)


def test_file_generation_and_reporting_preserve_all_failures_and_measure_actual_calls(file_preparation, tmp_path):
    from paper2.editor_calibration import generate_file, report_file
    prepared, sha = file_preparation
    output, report = tmp_path / 'batch.json', tmp_path / 'report.json'
    editor = CalibrationFakeEditor()
    generated = generate_file(prepared, sha, output, backend=editor)
    assert generated['status'] == 'complete' and generated['new_calls'] == len(editor.calls) == 12
    assert generated['resources']['observed_output_tokens'] == 36
    assert generated['resources']['observed_calls_with_timing'] == 12
    assert generated['resources']['observed_inference_wall_seconds'] >= 0
    assert report_file(prepared, sha, output, report)['status'] == 'complete'
    saved = json.loads(report.read_text(encoding='utf-8'))
    assert saved['outcomes']['operational_failure_causes']['invalid_json']['denominator'] == 12
    assert saved['outcomes']['operational_failure_causes']['invalid_json']['count'] == 12
    assert saved['semantic_approval'] is False
    with pytest.raises(FileExistsError):
        generate_file(prepared, sha, output, backend=editor, resume=True)
    with pytest.raises(FileExistsError):
        report_file(prepared, sha, output, report)
    assert len(editor.calls) == 12


def test_file_resume_never_resamples_durable_results_and_claim_prevents_other_output(file_preparation, tmp_path, monkeypatch):
    from paper2 import editor_calibration as calibration
    prepared, sha = file_preparation
    output = tmp_path / 'batch.json'
    append = calibration._append
    count = 0

    def pause_after_two_results(stream, value):
        nonlocal count
        append(stream, value)
        if 'generation' in value:
            count += 1
            if count == 2:
                raise KeyboardInterrupt('Synthetic interruption after durable result')

    first = CalibrationFakeEditor()
    monkeypatch.setattr(calibration, '_append', pause_after_two_results)
    with pytest.raises(KeyboardInterrupt):
        calibration.generate_file(prepared, sha, output, backend=first)
    monkeypatch.setattr(calibration, '_append', append)
    second = CalibrationFakeEditor()
    with pytest.raises(ValueError, match='generation claim'):
        calibration.generate_file(prepared, sha, tmp_path / 'another-output.json', backend=second)
    assert not second.calls
    done = calibration.generate_file(prepared, sha, output, backend=second, resume=True)
    assert len(first.calls) == 2 and len(second.calls) == done['new_calls'] == 10
    assert not ({call[1]['seed'] for call in first.calls} & {call[1]['seed'] for call in second.calls})
    assert done['resources']['observed_calls_with_timing'] == 12


def test_interrupted_inflight_call_blocks_retry_instead_of_resampling(file_preparation, tmp_path):
    from paper2.editor_calibration import generate_file
    prepared, sha = file_preparation
    output = tmp_path / 'batch.json'

    class Interrupted(CalibrationFakeEditor):
        def generate(self, *args, **kwargs):
            raise KeyboardInterrupt('Synthetic interruption before any result')

    with pytest.raises(KeyboardInterrupt):
        generate_file(prepared, sha, output, backend=Interrupted())
    untouched = CalibrationFakeEditor()
    with pytest.raises(ValueError, match='interrupted unresolved call'):
        generate_file(prepared, sha, output, backend=untouched, resume=True)
    assert not untouched.calls


def test_file_generation_rejects_input_hash_source_or_lock_mismatch_before_calls(file_preparation, tmp_path, monkeypatch):
    from pathlib import Path
    from paper2.editor_calibration import generate_file
    from paper2.training import source_identity
    from paper2.training.g1_editability import _job_lock
    prepared, sha = file_preparation
    editor = CalibrationFakeEditor()
    with pytest.raises(ValueError, match='expected hash'):
        generate_file(prepared, '0' * 64, tmp_path / 'batch.json', backend=editor)
    with _job_lock(Path(prepared).parent / 'generation.lock'):
        with pytest.raises(ValueError, match='Another G1 editor process'):
            generate_file(prepared, sha, tmp_path / 'batch.json', backend=editor)
    monkeypatch.setattr(source_identity, 'current_source_identity', lambda: {'fixture': 'changed source'})
    with pytest.raises(ValueError, match='source changed'):
        generate_file(prepared, sha, tmp_path / 'batch.json', backend=editor)
    assert not editor.calls


def test_finished_journal_finalizes_without_loading_editor_again(file_preparation, tmp_path, monkeypatch):
    from paper2 import editor_calibration as calibration
    from paper2.training import g1_editability
    prepared, sha = file_preparation
    output = tmp_path / 'batch.json'
    writer = calibration._write_new

    def fail_final_write(path, value):
        if path == output:
            raise OSError('Synthetic final file write failure')
        return writer(path, value)

    editor = CalibrationFakeEditor()
    monkeypatch.setattr(calibration, '_write_new', fail_final_write)
    with pytest.raises(OSError, match='final file'):
        calibration.generate_file(prepared, sha, output, backend=editor)
    assert len(editor.calls) == 12
    monkeypatch.setattr(calibration, '_write_new', writer)
    monkeypatch.setattr(g1_editability, 'SDPAEditor', lambda **kwargs: pytest.fail('No completed record may be resampled'))
    result = calibration.generate_file(prepared, sha, output, resume=True)
    assert result['new_calls'] == 0 and result['status'] == 'complete'


def test_failed_start_persistence_stops_before_any_inference_or_later_contract(file_preparation, tmp_path, monkeypatch):
    from pathlib import Path
    from paper2 import editor_calibration as calibration
    from paper2.training.g1_editability import _CallStartWriteFailure
    prepared, sha = file_preparation
    output = tmp_path / 'batch.json'
    editor = CalibrationFakeEditor()
    attempts = []

    def failed_start(stream, value):
        attempts.append(value)
        assert 'generation' not in value
        # Simulate a write that partially reached disk before its failure.
        stream.write('{"artifact_id":')
        stream.flush()
        raise OSError('Synthetic failed start journal persistence')

    monkeypatch.setattr(calibration, '_append', failed_start)
    with pytest.raises(_CallStartWriteFailure, match='failed start journal'):
        calibration.generate_file(prepared, sha, output, backend=editor)
    assert len(attempts) == 1 and not editor.calls and not output.exists()
    assert Path(str(output) + '.journal.jsonl').read_bytes() == b''
    assert Path(str(output) + '.started.jsonl').read_bytes() == b'{"artifact_id":'
    with pytest.raises(json.JSONDecodeError):
        calibration.generate_file(prepared, sha, output, backend=editor, resume=True)
    assert len(attempts) == 1 and not editor.calls
