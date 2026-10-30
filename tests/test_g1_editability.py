"""CPU-only synthetic completions/editor replies; never a scientific measurement."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import random
import shutil

import pytest

from paper2.data.world import generate_world, render_record
from paper2.edit_generation import generate_edit_controls, prepare_generation_plan, validate_generated_batch
from paper2.gates import _g1_editor_measurements
from paper2.io import digest, file_digest
from paper2.training.config import TrainingConfig
from paper2.training.data import natural_completion_outcome
from paper2.training.selection import MODEL_AMENDMENT_SHA256
from paper2.training.selection_results import freeze_panel_selection
from panel_fixture import make_panel_fixture

from paper2.training import g1_editability as helper


@pytest.fixture(autouse=True)
def remove_this_tests_full_panel(tmp_path):
    # Full CPU panels are disposable test inputs, not experiment evidence. Keep
    # only the small test outputs instead of accumulating gigabytes per suite.
    yield
    shutil.rmtree(tmp_path / 'panel', ignore_errors=True)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + '\n')
    return path


def jsonl(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(value, sort_keys=True) + '\n' for value in values))
    return path


@pytest.fixture(scope='module')
def observed_fixture():
    config = replace(TrainingConfig(), arm='reasoning_rl', model=helper.MODEL['model'], revision=helper.MODEL['revision'],
        completion_protocol=helper.NATURAL_COMPLETION_PROTOCOL, beta=.04, model_amendment_sha256=MODEL_AMENDMENT_SHA256,
        lr_scheduler_type='constant', warmup_ratio=0.).to_dict()
    development = [render_record(generate_world(random.Random(seed)), 'direct') for seed in range(120)]
    predictions = []
    for step in range(0, 2001, 250):
        for world in development:
            text, ids = 'Repetition.\nAnswer:', [9] * 384
            outcome = natural_completion_outcome(text, reasoning=True, completion_ids=ids, eos_token_id=2)
            checkpoint = f'/synthetic/run/evaluated_checkpoints/checkpoint-{step}'
            predictions.append({**outcome, 'id': world['id'], 'world_id': world['world_id'], 'answer': world['answer'],
                'arm': 'reasoning_rl', 'dataset': 'clean', 'evaluation_dataset': 'clean', 'seed': 1,
                **helper.MODEL, 'view': 'direct', 'split': 'development', 'format': 'native_reasoning',
                'completion_protocol': helper.NATURAL_COMPLETION_PROTOCOL, 'checkpoint_id': checkpoint,
                'checkpoint_path': checkpoint, 'path': checkpoint, 'step': step,
                'checkpoint_kind': 'base' if step == 0 else 'trained', 'adapter_sha256': digest(['mock', step]),
                'completion': text, 'completion_ids': ids, 'completion_tokens': 384,
                'eos_token_id': 2, 'correct': False, 'forced_delimiter': False})
    return config, development, predictions


def test_extraction_is_1080_observed_sources_with_no_fabricated_prompt_or_replay(observed_fixture):
    config, development, predictions = observed_fixture
    traces, mappings = helper.extract(predictions, development, config, {'fixture': 'synthetic'})
    assert len(traces) == len(mappings) == 1080
    assert len({row['artifact_id'] for row in traces}) == 1080
    for source, raw in zip(traces, sorted(predictions, key=lambda row: (row['step'], row['id']))):
        helper._validate_source_trace(source)
        assert source['trace'] == raw['completion']  # Incomplete answer header retained.
        assert source['generation_prompt_ids'] is None and source['generation_prompt'] is None
        assert source['generation_prompt_observed'] is False and source['replay_eligible'] is False
        assert 'answer_prefix_ids' not in source and 'answer_prompt' not in source
        assert source['observed_completion_provenance']['raw_record_sha256'] == digest(raw)
        assert source['purpose'] == helper.PURPOSE


@pytest.mark.parametrize('text,expected', [('thinking\nAnswer: A.', 'thinking'), ('Answer: A', ''),
    ('thinking\nAnswer:', 'thinking\nAnswer:'), ('thinking\nAnswer: Z', 'thinking\nAnswer: Z'),
    ('thinking\nAnswer: A. More text', 'thinking\nAnswer: A. More text')])
def test_canonical_trace_only_removes_a_valid_terminal_answer(text, expected):
    assert helper.canonical_trace(text) == expected


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'protocol', 'answer', 'adapter', 'checkpoint',
    'completion_ids', 'outcome', 'cap', 'forced', 'eos_padding', 'wrong_world'])
def test_extraction_refuses_incomplete_changed_or_inconsistent_observations(observed_fixture, mutation):
    config, development, original = observed_fixture
    rows = deepcopy(original)
    if mutation == 'missing':
        rows.pop()
    elif mutation == 'duplicate':
        rows[-1] = rows[0]
    elif mutation == 'protocol':
        rows[0]['completion_protocol'] = 'legacy_forced_answer_v1'
    elif mutation == 'answer':
        rows[0]['answer'] = 'Z'
    elif mutation == 'adapter':
        rows[0]['adapter_sha256'] = 'f' * 64
    elif mutation == 'checkpoint':
        rows[0]['checkpoint_path'] = '/other'
    elif mutation == 'completion_ids':
        rows[0]['completion_ids'] = None
    elif mutation == 'outcome':
        rows[0]['cap_failure'] = False
    elif mutation == 'cap':
        rows[0]['completion_ids'].append(8)
    elif mutation == 'forced':
        rows[0]['forced_delimiter'] = True
    elif mutation == 'eos_padding':
        rows[0]['completion_ids'][0] = 2
    else:
        rows[0]['world_id'] = 'foreign'
    with pytest.raises(ValueError):
        helper.extract(rows, development, config, {})


def test_extract_refuses_replay_scope_and_legacy_config(observed_fixture):
    config, development, rows = observed_fixture
    with pytest.raises(ValueError, match='never Test 2 or replay'):
        helper.extract(rows, development, config, {}, purpose='test2_replay')
    with pytest.raises(ValueError, match='selected amended'):
        helper.extract(rows, development, {**config, 'beta': 0}, {})


class FakeEditor:
    checkpoint_id = helper.EDITOR['model_id']
    revision = helper.EDITOR['revision']

    class tokenizer:
        chat_template = '<user>{{ user_content }}</user><assistant>'

        @staticmethod
        def apply_chat_template(messages, *, tokenize, add_generation_prompt):
            return '<user>' + messages[0]['content'] + '</user><assistant>'

    def generate(self, prompt, **kwargs):
        return {'text': json.dumps({'status': 'not_editable', 'reason_code': 'belief_not_stated',
            'explanation': 'Synthetic editor response for integration testing, not a scientific observation.'}),
            'truncated': False}


def test_existing_editor_and_g1_accept_truthful_edit_only_sources_without_replay(observed_fixture, tmp_path):
    config, development, predictions = observed_fixture
    traces, mappings = helper.extract(predictions, development, config, {})
    # Small integration subset exercises the actual schema/generator/G1 binding;
    # full-panel integrity and 1080 extraction are tested separately above.
    traces, mappings = traces[:2], mappings[:2]
    record_ids = {trace['record_id'] for trace in traces}
    rows = [row for row in development if row['id'] in record_ids]
    raw = [row for row in predictions if row['step'] == 0 and row['id'] in record_ids]
    manifest = helper.prepare_edit_manifest(traces, rows, max_worlds=120, seed=2026, stage='pilot')
    plan = prepare_generation_plan(manifest, **helper.EDITOR)
    assert plan['prompt_version'] == helper.PROMPT_VERSION
    assert plan['decoding']['max_new_tokens'] == 2048 and plan['decoding']['seed'] == 2026
    batch = generate_edit_controls(FakeEditor(), traces, manifest, plan)
    assert len(validate_generated_batch(batch)) == 2
    plan_path, batch_path = write(tmp_path / 'plan.json', plan), write(tmp_path / 'batch.json', batch)
    evidence = helper.editor_evidence(plan_path, batch_path, mappings)
    evidence_path = write(tmp_path / 'evidence.json', evidence)
    item = helper.reference(evidence_path)
    result = _g1_editor_measurements({'status': 'pending', 'evidence': [item], 'editability_evidence': item}, tmp_path, raw)
    assert result['status'] == 'measured'
    assert result['human_audit'] == 'not_evaluated_by_this_helper'
    assert result['trace_health_with_editor_outcomes']['panels'][0]['editability']['counts']['belief_not_stated'] == 2
    with pytest.raises(ValueError, match='retain all'):
        helper.editor_evidence(plan_path, batch_path, mappings[:1])


def bind_panel_for_extraction(panel, directory):
    """Freeze the actual computed winner of twelve distinct synthetic runs."""
    selection = directory / 'selection.json'
    result = freeze_panel_selection(panel['binding'], file_digest(panel['binding']), panel['status'],
        panel['policy'], file_digest(panel['policy']), selection,
        reviewed_by='Offline fixture, no scientific approval', rationale='CPU integrity testing only')
    candidate = result['selected_reasoning_rl_candidate']
    root = panel['root'] / 'runs' / candidate
    paths = dict(selection=selection, panel_status=panel['status'], registry=panel['registry'],
        config=panel['configs'][candidate], run_manifest=root / 'run_manifest.json',
        predictions=root / 'checkpoint_predictions.jsonl', learning_evidence=root / 'g1_learning_evidence.json',
        development=panel['development'], corpus_manifest=panel['corpus'],
        execution_environment=panel['environment'], candidate_execution_binding=panel['binding'],
        batch_frozen_inputs=panel['frozen'])
    binding = write(directory / 'binding.json', {'purpose': helper.PURPOSE,
        'scope': 'selected_reasoning_rl_after_complete_panel', 'candidate_name': candidate,
        'source_commit': result['source_commit'], **{key: helper.reference(path) for key, path in paths.items()}})
    return binding, paths, directory / 'output'


@pytest.fixture
def file_fixture(tmp_path):
    return bind_panel_for_extraction(make_panel_fixture(tmp_path / 'panel'), tmp_path)


def test_prepare_requires_actual_bound_complete_selection_and_never_launches(file_fixture):
    binding, _, output = file_fixture
    report = helper.prepare(binding, file_digest(binding), output)
    assert report['editor_launched'] is False and report['scientific_gate_decision'] is None
    plan = helper.read(output / 'editor_plan.json')
    assert len(plan['contract_specs']) == 1080
    assert plan['resource_counts']['max_output_tokens'] == 1080 * 2048
    assert helper.read(output / 'extraction.json')['observed_prompt_ids'] is False
    with pytest.raises(FileExistsError):
        helper.prepare(binding, file_digest(binding), output)


@pytest.mark.parametrize('role', helper.ROLES)
def test_any_bound_input_change_is_refused_before_output(file_fixture, role):
    binding, paths, output = file_fixture
    paths[role].write_text(paths[role].read_text() + '\n')
    with pytest.raises(ValueError, match='Missing or modified'):
        helper.prepare(binding, file_digest(binding), output)
    assert not output.exists()


def test_rehashed_incomplete_panel_is_not_a_selected_complete_run(file_fixture):
    binding, paths, output = file_fixture
    state = helper.read(paths['panel_status'])
    state['status'] = 'running'
    write(paths['panel_status'], state)
    refresh_extraction_bindings(binding, paths, 'panel_status', refresh_selection=True)
    with pytest.raises(ValueError, match='All twelve'):
        helper.prepare(binding, file_digest(binding), output)
    assert not output.exists()


@pytest.mark.parametrize('mutation', ['selection', 'registry', 'source', 'receipt', 'adapter', 'purpose'])
def test_rehashed_policy_changes_and_missing_adapter_never_write_outputs(file_fixture, mutation):
    binding, paths, output = file_fixture
    binding_doc = helper.read(binding)
    if mutation == 'adapter':
        raw = helper.read_jsonl(paths['predictions'])[0]
        (Path(raw['checkpoint_id']) / 'adapter_model.safetensors').write_bytes(b'changed')
    elif mutation == 'purpose':
        binding_doc['purpose'] = 'test2_replay'
    elif mutation == 'source':
        binding_doc['source_commit'] = 'f' * 40
    elif mutation == 'receipt':
        doc = helper.read(paths['learning_evidence'])
        doc['run_status'] = 'running'
        write(paths['learning_evidence'], doc)
        binding_doc['learning_evidence'] = helper.reference(paths['learning_evidence'])
    else:
        path = paths[mutation]
        doc = helper.read(path)
        if mutation == 'selection':
            doc['selected_reasoning_rl_candidate'] = 'different'
        else:
            doc['beta_by_arm']['reasoning_rl'] = 0
        write(path, doc)
        binding_doc[mutation] = helper.reference(path)
    write(binding, binding_doc)
    with pytest.raises(ValueError):
        helper.prepare(binding, file_digest(binding), output)
    assert not output.exists()


def test_existing_output_refused_before_input_access(tmp_path):
    output = tmp_path / 'existing'
    output.mkdir()
    marker = output / 'preserved.txt'
    marker.write_text('original')
    with pytest.raises(FileExistsError):
        helper.prepare(tmp_path / 'does-not-exist.json', 'f' * 64, output)
    assert marker.read_text() == 'original'


def refresh_extraction_bindings(binding, paths, *roles, refresh_selection=False):
    if refresh_selection:
        selected = helper.read(paths['selection'])
        selected['panel_status_sha256'] = file_digest(paths['panel_status'])
        selected['panel_status'] = helper.reference(paths['panel_status'])
        write(paths['selection'], selected)
        roles = (*roles, 'selection')
    doc = helper.read(binding)
    for role in roles:
        doc[role] = helper.reference(paths[role])
    write(binding, doc)


@pytest.mark.parametrize('kind', ['foreign_commit', 'same_commit_changed_inventory'])
def test_new_extraction_binding_cannot_substitute_completed_run_environment(file_fixture, kind):
    binding, paths, output = file_fixture
    fake = deepcopy(helper.read(paths['execution_environment']))
    if kind == 'foreign_commit':
        fake['source']['commit'] = 'f' * 40
    else:
        fake['runtime'] = {'package_files_sha256': {'foreign/module.py': 'f' * 64}}
    paths['execution_environment'] = write(output.parent / 'substituted-environment.json', fake)
    doc = helper.read(binding)
    doc['execution_environment'] = helper.reference(paths['execution_environment'])
    doc['source_commit'] = fake['source']['commit']
    write(binding, doc)
    with pytest.raises(ValueError, match='substituted an original frozen execution_environment|differs from the recomputed selection'):
        helper.prepare(binding, file_digest(binding), output)
    assert not output.exists()


@pytest.mark.parametrize('kind', ['original_binding', 'frozen_binding', 'frozen_environment', 'wrapper'])
def test_rehashed_extraction_must_preserve_original_completed_lineage(file_fixture, kind):
    binding, paths, output = file_fixture
    role = 'candidate_execution_binding' if kind == 'original_binding' else 'batch_frozen_inputs'
    doc = helper.read(paths[role])
    if kind == 'original_binding':
        doc['added_after_completion'] = True
    elif kind == 'frozen_binding':
        doc['binding_sha256'] = 'f' * 64
    elif kind == 'frozen_environment':
        doc['files_sha256'][str(paths['execution_environment'])] = 'f' * 64
    else:
        doc['wrapper_sha256'] = 'f' * 64
    write(paths[role], doc)
    refresh_extraction_bindings(binding, paths, role)
    with pytest.raises(ValueError, match='lineage|original frozen|wrapper|modified bound|Original source|original execution binding'):
        helper.prepare(binding, file_digest(binding), output)
    assert not output.exists()


@pytest.mark.parametrize('kind', ['exit', 'steps', 'worlds', 'config', 'protocol', 'empty_receipt'])
def test_other_candidate_completion_labels_alone_are_not_accepted(file_fixture, kind):
    binding, paths, output = file_fixture
    state = helper.read(paths['panel_status'])
    current = state['candidates']['label_sft_clean_seed1_lr-x30_constant']
    if kind == 'exit':
        current['exit_code'] = 1
    elif kind == 'steps':
        current['completion_receipt']['completed_optimizer_steps'] = 2000
    elif kind == 'worlds':
        current['completion_receipt']['observed_worlds'] = 1999
    elif kind == 'config':
        current['config_sha256'] = 'f' * 64
    elif kind == 'protocol':
        current['completion_receipt']['completion_protocol'] = 'legacy_forced_answer_v1'
    else:
        current['completion_receipt'] = {}
    write(paths['panel_status'], state)
    refresh_extraction_bindings(binding, paths, 'panel_status', refresh_selection=True)
    with pytest.raises(ValueError, match='Every candidate needs|process/config/exit|successful supervisor|Terminal completion receipt'):
        helper.prepare(binding, file_digest(binding), output)
    assert not output.exists()


def test_same_path_with_different_hash_at_later_step_is_not_a_ninth_checkpoint(observed_fixture):
    config, development, original = observed_fixture
    rows = deepcopy(original)
    checkpoint = rows[0]['checkpoint_id']
    for row in rows:
        if row['step'] == 250:
            row.update(checkpoint_id=checkpoint, checkpoint_path=checkpoint, path=checkpoint)
    with pytest.raises(ValueError, match='Nine different checkpoint identities'):
        helper.extract(rows, development, config, {})


def test_actual_adapter_hash_must_match_completed_run_receipt_on_every_checkpoint(file_fixture):
    binding, paths, output = file_fixture
    state = helper.read(paths['panel_status'])
    selected = state['candidates']['reasoning_rl_clean_seed1_lr-x1_constant']
    matching = selected['completion_receipt']['matching_adapter_sha256']
    key = next(path for path in matching if Path(path).parent.name == 'checkpoint-250')
    matching[key] = 'f' * 64
    write(paths['panel_status'], state)
    refresh_extraction_bindings(binding, paths, 'panel_status', refresh_selection=True)
    with pytest.raises(ValueError, match='raw row or original completion receipt|Terminal completion receipt'):
        helper.prepare(binding, file_digest(binding), output)
    assert not output.exists()


def test_terminal_failure_is_retained_and_selected_completed_run_can_be_prepared(tmp_path):
    failed_name = 'reasoning_rl_clean_seed1_lr-x30_constant'
    panel = make_panel_fixture(tmp_path / 'panel', failures=(failed_name,))
    failed_manifest = panel['root'] / 'runs' / failed_name / 'run_manifest.json'
    before = file_digest(failed_manifest)
    binding, paths, output = bind_panel_for_extraction(panel, tmp_path)
    selected = helper.read(paths['selection'])
    assert selected['failures'] == [failed_name]
    assert selected['selected_reasoning_rl_candidate'] != failed_name
    report = helper.prepare(binding, file_digest(binding), output)
    assert report['editor_launched'] is False
    assert len(helper.read(output / 'editor_plan.json')['contract_specs']) == 1080
    assert file_digest(failed_manifest) == before


def test_completed_candidate_inspection_needs_no_selection_or_editor(tmp_path, monkeypatch):
    panel = make_panel_fixture(tmp_path / 'panel')
    candidate = 'reasoning_rl_clean_seed1_lr-x1_constant'
    state = helper.read(panel['status'])
    # Match an ongoing real panel: only the first completed candidate is present.
    state['status'] = 'running'
    state['candidates'] = {candidate: state['candidates'][candidate]}
    write(panel['status'], state)
    monkeypatch.setattr(helper, 'prepare_generation_plan', lambda *a, **kw: pytest.fail('No editor plan in inspection'))
    monkeypatch.setattr(helper, 'SDPAEditor', lambda *a, **kw: pytest.fail('No editor call in inspection'))
    output = tmp_path / 'inspection.json'
    before = {path: file_digest(path) for path in (panel['binding'], panel['status'], panel['frozen'])}
    report = helper.inspect_completed_candidate(panel['binding'], file_digest(panel['binding']),
                                                panel['status'], candidate, output)
    assert report['source_records'] == 1080 and report['native_checkpoints'] == 9
    assert report['stories_per_checkpoint'] == 120 and len(report['adapter_sha256']) == 9
    assert not report['editor_launched'] and not report['selection_performed']
    assert not report['selected_run_claim'] and not report['g1_eligible']
    assert report['scientific_gate_decision'] is None
    assert before == {path: file_digest(path) for path in before}
    assert not (tmp_path / 'selection.json').exists()
    assert not list(tmp_path.rglob('prepared.json'))
    assert not list(tmp_path.rglob('editor_plan.json'))
    with pytest.raises(FileExistsError):
        helper.inspect_completed_candidate(panel['binding'], file_digest(panel['binding']),
                                            panel['status'], candidate, output)


@pytest.mark.parametrize('problem', ['unfinished', 'missing_adapter'])
def test_candidate_inspection_rejects_unfinished_or_missing_native_adapter(tmp_path, monkeypatch, problem):
    panel = make_panel_fixture(tmp_path / 'panel')
    candidate = 'reasoning_rl_clean_seed1_lr-x1_constant'
    if problem == 'unfinished':
        state = helper.read(panel['status'])
        state['candidates'][candidate]['status'] = 'running'
        write(panel['status'], state)
    else:
        adapter = panel['root'] / 'runs' / candidate / 'evaluated_checkpoints' / 'checkpoint-250' / 'adapter_model.safetensors'
        adapter.unlink()  # Synthetic fixture only: emulate missing retained weights.
    monkeypatch.setattr(helper, 'SDPAEditor', lambda *a, **kw: pytest.fail('No editor call in inspection'))
    output = tmp_path / 'inspection.json'
    with pytest.raises(ValueError, match='not completed|matching adapter is missing'):
        helper.inspect_completed_candidate(panel['binding'], file_digest(panel['binding']),
                                            panel['status'], candidate, output)
    assert not output.exists()

class MeasuredFakeEditor(FakeEditor):
    attention_implementation = 'sdpa'

    def __init__(self, *, fail_first=False):
        self.calls = []
        self.seeds = []
        self.fail_first = fail_first

    def token_ids(self, text):
        return [7, 8, len(text)]  # Explicit synthetic tokenizer, not a measured model tokenization.

    def generate(self, prompt, **kwargs):
        self.calls.append(prompt)
        self.seeds.append(kwargs['seed'])
        assert kwargs['max_new_tokens'] == 2048 and kwargs['do_sample'] is False
        if self.fail_first and len(self.calls) == 1:
            raise RuntimeError('Synthetic inference failure retained, not retried')
        result = super().generate(prompt, **kwargs)
        return {**result, 'prompt_ids': self.token_ids(prompt), 'generated_ids': [3, 4, 2]}


def prepared_fixture(file_fixture):
    binding, _, directory = file_fixture
    result = helper.prepare(binding, file_digest(binding), directory)
    return Path(result['prepared']['path']), result['prepared']['sha256']


def test_fixed_workload_before_generation_has_no_invented_measurements_or_spend(file_fixture):
    prepared, sha = prepared_fixture(file_fixture)
    result = helper.resource_report(prepared, sha, hourly_gbp=1.25)
    assert result['planned_calls'] == result['remaining_calls'] == 1080
    assert result['max_output_tokens_per_call'] == 2048
    assert result['max_total_output_tokens'] == 2_211_840
    assert result['measurement_status'] == 'not_measured'
    assert result['observed_inference_wall_seconds'] is None
    assert result['projected_full_output_tokens'] is None
    assert result['projected_full_inference_gbp'] is None
    assert result['spend_approval'] is False


@pytest.fixture
def workflow_fixture(observed_fixture, tmp_path, monkeypatch):
    # Full extraction/preparation/lineage is tested above with all1080 records.
    # Exercise the real generic editor/journal validators on a short observed
    # slice here; only the costly already-tested prepared-input boundary is stubbed.
    config, development, predictions = observed_fixture
    traces, mappings = helper.extract(predictions, development, config, {})
    traces, mappings = traces[:3], mappings[:3]
    ids = {row['record_id'] for row in traces}
    manifest = helper.prepare_edit_manifest(traces, [row for row in development if row['id'] in ids],
        max_worlds=120, seed=2026, stage='pilot')
    plan = prepare_generation_plan(manifest, **helper.EDITOR)
    files = {'editor_plan.json': write(tmp_path / 'editor_plan.json', plan),
             'extraction.json': write(tmp_path / 'extraction.json', {'fixture': True}),
             'classifications.mapping.json': write(tmp_path / 'classifications.mapping.json', mappings)}
    prepared = write(tmp_path / 'prepared.json', {'fixture': 'Three-contract control-flow slice'})
    monkeypatch.setattr(helper, '_prepared', lambda *args: ({}, files, plan, manifest, traces, mappings))
    return prepared, file_digest(prepared)


def test_editor_workflow_resumes_same_contracts_retains_failures_and_finalizes(workflow_fixture, tmp_path):
    prepared, sha = workflow_fixture
    output = tmp_path / 'generated.json'
    first = MeasuredFakeEditor(fail_first=True)
    pause = helper.generate(prepared, sha, output, backend=first, max_calls=2)
    assert pause['status'] == 'paused_after_requested_calls' and len(first.calls) == 2
    assert not output.exists()
    rows = helper.read_jsonl(pause['journal']['path'])
    assert rows[0]['generation_status'] == 'error' and rows[1]['generation_status'] == 'not_editable'
    assert pause['resources']['observed_calls_with_timing'] == 2
    assert pause['resources']['calls_without_output_token_count'] == 1
    first_bytes = Path(pause['journal']['path']).read_bytes()
    with pytest.raises(ValueError, match='explicit resume'):
        helper.generate(prepared, sha, output, backend=MeasuredFakeEditor())
    final_backend = MeasuredFakeEditor()
    done = helper.generate(prepared, sha, output, backend=final_backend, resume=True)
    assert done['status'] == 'complete' and len(final_backend.calls) == 1
    assert Path(pause['journal']['path']).read_bytes().startswith(first_bytes)
    assert not set(first.seeds).intersection(final_backend.seeds)
    assert len(set(first.seeds + final_backend.seeds)) == 3
    contracts = helper.validate_generated_batch(helper.read(output))
    assert len(contracts) == 3
    assert sum(row['generation_status'] == 'error' for row in contracts.values()) == 1
    resources = helper.resource_report(prepared, sha, journal_path=pause['journal']['path'], hourly_gbp=2)
    assert resources['completed_contracts'] == 3 and resources['remaining_calls'] == 0
    assert resources['observed_output_tokens'] == 2 * 3
    assert resources['observed_calls_with_timing'] == 3
    assert resources['projected_full_inference_gbp'] == pytest.approx(resources['observed_inference_wall_seconds'] / 3600 * 2)
    evidence_path = tmp_path / 'g1-editability.json'
    result = helper.finalize(prepared, sha, output, evidence_path)
    assert result['classifications'] == 3 and result['scientific_gate_decision'] is None
    assert helper.read(evidence_path)['human_audit'] == 'not_performed_by_this_helper'
    with pytest.raises(FileExistsError):
        helper.finalize(prepared, sha, output, evidence_path)
    with pytest.raises(FileExistsError):
        helper.generate(prepared, sha, output, resume=True)


def test_editor_load_failure_preserves_resumable_empty_journal(workflow_fixture, tmp_path, monkeypatch):
    prepared, sha = workflow_fixture
    output = tmp_path / 'generated.json'
    def failure(**kwargs):
        raise RuntimeError('Synthetic model load failure')
    monkeypatch.setattr(helper, 'SDPAEditor', failure)
    with pytest.raises(RuntimeError, match='load failure'):
        helper.generate(prepared, sha, output)
    journal = Path(str(output) + '.journal.jsonl')
    assert journal.exists() and journal.read_bytes() == b''
    result = helper.generate(prepared, sha, output, backend=MeasuredFakeEditor(), resume=True, max_calls=1)
    assert result['new_calls'] == 1


def test_invalid_or_changed_journal_never_calls_editor_again(workflow_fixture, tmp_path):
    prepared, sha = workflow_fixture
    output = tmp_path / 'generated.json'
    helper.generate(prepared, sha, output, backend=MeasuredFakeEditor(), max_calls=1)
    journal = Path(str(output) + '.journal.jsonl')
    with journal.open('a') as stream:
        stream.write(journal.read_text())
    backend = MeasuredFakeEditor()
    with pytest.raises(ValueError, match='duplicate'):
        helper.generate(prepared, sha, output, backend=backend, resume=True)
    assert not backend.calls


def test_sdpa_and_frozen_identity_are_mandatory_before_any_editor_call():
    backend = MeasuredFakeEditor()
    backend.attention_implementation = 'eager'
    with pytest.raises(ValueError, match='SDPA'):
        helper._MeasuredEditor(backend)
    backend.attention_implementation = 'sdpa'
    backend.revision = 'other'
    with pytest.raises(ValueError, match='revision'):
        helper._MeasuredEditor(backend)
    assert not backend.calls


def test_sdpa_constructor_loads_only_fixed_editor_without_touching_source_model(monkeypatch):
    import sys
    from types import SimpleNamespace
    calls = []
    model = SimpleNamespace(eval=lambda: None)
    tokenizer = SimpleNamespace(pad_token_id=None, eos_token_id=2)
    def load_model(name, **kwargs):
        calls.append((name, kwargs))
        return model
    fake_transformers = SimpleNamespace(
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=load_model),
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda name, **kwargs: tokenizer))
    monkeypatch.setitem(sys.modules, 'transformers', fake_transformers)
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace())
    backend = helper.SDPAEditor()
    assert calls == [(helper.EDITOR['model_id'], {'revision': helper.EDITOR['revision'],
        'torch_dtype': 'auto', 'attn_implementation': 'sdpa', 'device_map': 'auto'})]
    assert backend.checkpoint_id != helper.MODEL['model'] and tokenizer.pad_token_id == 2


def test_job_lock_prevents_duplicate_live_output_owner(tmp_path):
    with helper._job_lock(tmp_path / 'editor.lock'):
        with pytest.raises((ValueError, OSError)):
            with helper._job_lock(tmp_path / 'editor.lock'):
                pytest.fail('A second process cannot acquire the journal')
    with helper._job_lock(tmp_path / 'editor.lock'):
        pass


@pytest.mark.parametrize('changed', ['prepared', 'editor_plan.json', 'edit_only_sources.jsonl'])
def test_changed_prepared_bytes_are_rejected_before_editor_load(file_fixture, changed, monkeypatch):
    prepared, sha = prepared_fixture(file_fixture)
    target = prepared if changed == 'prepared' else prepared.parent / changed
    target.write_text(target.read_text() + '\n')
    monkeypatch.setattr(helper, 'SDPAEditor', lambda **kwargs: pytest.fail('No backend should load'))
    with pytest.raises(ValueError, match='hash mismatch|modified bound'):
        helper.generate(prepared, sha, prepared.parent / 'new.json')


def test_complete_durable_journal_recovers_without_loading_or_regenerating(workflow_fixture, tmp_path, monkeypatch):
    prepared, sha = workflow_fixture
    output = tmp_path / 'generated.json'
    backend = MeasuredFakeEditor()
    done = helper.generate(prepared, sha, output, backend=backend)
    original_bytes = output.read_bytes()
    # Simulate an interruption after the last journal fsync but before publishing
    # a batch, using only synthetic test artifacts.
    output.rename(tmp_path / 'preserved-synthetic-first-batch.json')
    monkeypatch.setattr(helper, 'SDPAEditor', lambda **kwargs: pytest.fail('Completed journal must not load a model'))
    recovered = helper.generate(prepared, sha, output, resume=True)
    assert recovered['new_calls'] == 0 and len(backend.calls) == 3
    assert output.read_bytes() == original_bytes
    assert recovered['journal']['sha256'] == done['journal']['sha256']


def test_resource_projection_refuses_missing_evidence_or_nonfinite_price(workflow_fixture, tmp_path):
    prepared, sha = workflow_fixture
    with pytest.raises(ValueError, match='missing'):
        helper.resource_report(prepared, sha, journal_path=tmp_path / 'absent.jsonl')
    with pytest.raises(ValueError, match='finite'):
        helper.resource_report(prepared, sha, hourly_gbp=float('nan'))
