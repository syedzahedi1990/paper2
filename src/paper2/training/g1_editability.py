#!/usr/bin/env python3
"""Extract observed native completions for G1 fixed-editor classification only.

Saved-token extraction and a separately invoked, resumable fixed-editor job.
Never regenerates source traces, reconstructs source prompts, or decides a gate.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import math
import os
import time
from contextlib import contextmanager

from paper2.edit_generation import (CONTROL_PROMPT, PROMPT_VERSION, prepare_generation_plan,
    validate_generated_batch, generate_edit_controls, _validate_plan, _validate_record)
from paper2.backend import HFBackend
from paper2.io import digest, file_digest, read_jsonl
from paper2.traces import _validate_source_trace, prepare_edit_manifest
from paper2.training.config import NATURAL_COMPLETION_PROTOCOL, TrainingConfig
from paper2.training.data import natural_completion_outcome, parse_answer
from paper2.training.selection import assert_current_design
from paper2.training.selection_results import verify_panel_selection

PURPOSE = 'g1_editability_only'
MODEL = {'model': 'mistralai/Mistral-Small-24B-Instruct-2501', 'revision': '9527884be6e5616bdd54de542f9ae13384489724'}
EDITOR = {'model_id': 'Qwen/Qwen2.5-7B-Instruct', 'revision': 'a09a35458c702b33eeacc393d103063234e8bc28',
          'seed': 2026, 'max_new_tokens': 2048, 'audit_sample_size': 100, 'audit_seed': 2026, 'audit_confidence': .95}
ROLES = ('selection', 'panel_status', 'registry', 'config', 'run_manifest', 'predictions',
         'learning_evidence', 'development', 'corpus_manifest', 'execution_environment',
         'candidate_execution_binding', 'batch_frozen_inputs')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def reference(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': file_digest(path)}


def bound(item, root, name):
    path = (Path(root) / item.get('path', '')).resolve()
    if not path.is_file() or file_digest(path) != item.get('sha256'):
        raise ValueError('Missing or modified bound ' + name)
    return path


def canonical_trace(text):
    """Same valid-terminal rule as native health/G1; malformed headers stay."""
    complete = parse_answer(text, True, completion_protocol=NATURAL_COMPLETION_PROTOCOL) is not None
    suffix = re.search(r'(?:^|\n)Answer:\s*[A-F]\.?\s*$', text) if complete else None
    return text[:suffix.start()] if suffix is not None else text


def extract(predictions, development, raw_config, provenance, *, purpose=PURPOSE):
    if purpose != PURPOSE:
        raise ValueError('Completion-only sources may only serve G1 editability, never Test 2 or replay')
    config = TrainingConfig(**raw_config)
    if (config.arm != 'reasoning_rl' or config.dataset != 'clean' or config.seed != 1
            or config.completion_protocol != NATURAL_COMPLETION_PROTOCOL or config.beta != .04
            or config.stage != 'pilot' or config.profile or config.max_steps is not None
            or config.model != MODEL['model'] or config.revision != MODEL['revision']):
        raise ValueError('G1 extraction requires the selected amended full-pool reasoning RL candidate')
    config.validate()
    assert_current_design(raw_config)
    steps = config.evaluation_schedule(config.training_budget(2000)['resolved_max_steps'])
    if steps != list(range(0, 2001, 250)):
        raise ValueError('Expected all nine native checkpoint measurements')
    worlds = {row['id']: row for row in development if row.get('view') == 'direct'}
    if (len(worlds) != 120 or len({row['world_id'] for row in worlds.values()}) != 120
            or any(row.get('split') != 'development' or 'world' not in row for row in worlds.values())):
        raise ValueError('Require all 120 distinct frozen development worlds with symbolic states')
    native = [row for row in predictions if row.get('format') == 'native_reasoning']
    if len(native) != 1080 or {(row.get('step'), row.get('id')) for row in native} != {
            (step, record_id) for step in steps for record_id in worlds}:
        raise ValueError('Require exactly all nine complete 120-story native panels')
    traces, mappings, checkpoints = [], [], {}
    for row in sorted(native, key=lambda item: (item['step'], item['id'])):
        world = worlds[row['id']]
        if any(row.get(key) != value for key, value in {
                'arm': config.arm, 'dataset': 'clean', 'evaluation_dataset': 'clean', 'seed': 1,
                'model': config.model, 'revision': config.revision, 'view': 'direct', 'split': 'development',
                'world_id': world['world_id'], 'answer': world['answer'],
                'completion_protocol': NATURAL_COMPLETION_PROTOCOL}.items()):
            raise ValueError('Raw native record differs from its frozen config/world/protocol')
        checkpoint = row.get('checkpoint_id')
        if (not isinstance(checkpoint, str) or not checkpoint or row.get('checkpoint_path') != checkpoint
                or row.get('path') != checkpoint or row.get('checkpoint_kind') != ('base' if row['step'] == 0 else 'trained')
                or not isinstance(row.get('adapter_sha256'), str) or not re.fullmatch('[0-9a-f]{64}', row['adapter_sha256'])):
            raise ValueError('Incomplete raw checkpoint/adapter identity')
        identity = (checkpoint, row['adapter_sha256'])
        if checkpoints.setdefault(row['step'], identity) != identity:
            raise ValueError('One native checkpoint panel refers to multiple adapters')
        text, ids = row.get('completion'), row.get('completion_ids')
        if not isinstance(text, str) or not isinstance(ids, list) or any(type(token) is not int or token < 0 for token in ids):
            raise ValueError('Observed completion text and sampled IDs are required')
        if len(ids) > 384 or row.get('forced_delimiter') is not False or type(row.get('eos_token_id')) is not int:
            raise ValueError('Invalid sampled native completion provenance')
        eos = row['eos_token_id']
        if eos in ids and ids.index(eos) != len(ids) - 1:
            raise ValueError('Observed completion must end at its first EOS without padding')
        outcome = natural_completion_outcome(text, reasoning=True, completion_ids=ids, eos_token_id=eos)
        if (any(row.get(key) != value for key, value in outcome.items())
                or row.get('completion_tokens') != len(ids)
                or type(row.get('correct')) is not bool
                or row['correct'] != (outcome['prediction'] == world['answer'])):
            raise ValueError('Recorded native outcome differs from observed completion/answer')
        trace = canonical_trace(text)
        source = 'observed_reasoning_rl_native_checkpoint'
        source_checkpoint = {'id': checkpoint, 'revision': row['revision']}
        completion_hash = digest([text, ids, None, outcome])
        artifact = digest([source, source_checkpoint, row['id'], 'direct_fixed', digest(trace),
                           NATURAL_COMPLETION_PROTOCOL, completion_hash])
        result = {'schema_version': 3, 'purpose': PURPOSE, 'artifact_id': artifact,
            'trace_id': digest([artifact, row['id']]), 'source': source, 'source_checkpoint': source_checkpoint,
            'basis_record_id': row['id'], 'record_id': row['id'], 'world_id': row['world_id'], 'view': 'direct',
            'question_timing': 'direct_fixed', 'trace': trace, 'trace_sha256': digest(trace),
            'completion_protocol': NATURAL_COMPLETION_PROTOCOL, 'completion_text': text,
            'completion_generated_ids': ids, 'completion_tokens': len(ids), 'native_outcome': outcome,
            'native_completion_sha256': completion_hash, 'generation_prompt': None, 'generation_prompt_ids': None,
            'generation_prompt_observed': False, 'prompt_provenance': 'unavailable_in_checkpoint_predictions; not reconstructed',
            'replay_eligible': False, 'replay_ineligibility_reason': 'G1 completion record lacks original prompt',
            'observed_completion_provenance': {**provenance, 'raw_record_sha256': digest(row),
                'frozen_development_record_sha256': digest(world), 'world_sha256': digest(world['world']),
                'adapter_sha256': row['adapter_sha256'], 'step': row['step'],
                'token_text_consistency': 'Observed together in hashed training record; no tokenizer reconstruction performed'},
            'training_seed': 1, 'forced_delimiter': False}
        _validate_source_trace(result)
        traces.append(result)
        mappings.append({'checkpoint_id': checkpoint, 'adapter_sha256': row['adapter_sha256'],
                         'id': row['id'], 'artifact_id': artifact, 'raw_record_sha256': digest(row)})
    if len({checkpoint for checkpoint, _ in checkpoints.values()}) != 9:
        raise ValueError('Nine different checkpoint identities must be preserved')
    return traces, mappings



def completion_lineage(paths, extraction_binding, state, registry, entry):
    """Follow the completed batch's original binding, not a new commit claim."""
    original_path = paths['candidate_execution_binding']
    original_sha = file_digest(original_path)
    original, frozen = read(original_path), read(paths['batch_frozen_inputs'])
    if (paths['batch_frozen_inputs'] != paths['panel_status'].parent / 'frozen-inputs.json'
            or state.get('binding_sha256') != original_sha or frozen.get('binding_sha256') != original_sha
            or original.get('schema_version') != 2 or original.get('purpose') != 'amended_pilot_candidate_execution'
            or original.get('completion_protocol') != NATURAL_COMPLETION_PROTOCOL
            or original.get('beta_by_arm') != {'reasoning_rl': .04, 'answer_rl': .04, 'label_sft': 0.}):
        raise ValueError('Original execution binding differs from the completed batch source lineage')
    files = frozen.get('files_sha256', {})
    if files.get(str(original_path)) != original_sha:
        raise ValueError('Original execution binding is absent or changed in frozen-inputs')
    for role in ('execution_environment', 'registry', 'corpus_manifest', 'development'):
        actual = bound(original.get(role, {}), original_path.parent, 'original ' + role)
        if actual != paths[role] or files.get(str(actual)) != file_digest(actual):
            raise ValueError('Extraction substituted an original frozen ' + role)
    config_path = (paths['registry'].parent / entry['config_path']).resolve()
    if (config_path != paths['config'] or files.get(str(config_path)) != file_digest(config_path)
            or original.get('config_sha256') != {row['config_path']: row['config_sha256'] for row in registry['candidates']}):
        raise ValueError('Selected config differs from original frozen execution binding')
    decision_path = bound(original.get('decision', {}), original_path.parent, 'original resource decision')
    decision, environment = read(decision_path), read(paths['execution_environment'])
    commit = environment.get('source', {}).get('commit')
    if (files.get(str(decision_path)) != file_digest(decision_path)
            or decision.get('execution_environment_sha256') != file_digest(paths['execution_environment'])
            or decision.get('source_commit') != commit or extraction_binding.get('source_commit') != commit
            or not isinstance(commit, str) or not re.fullmatch('[0-9a-f]{40}', commit)
            or frozen.get('wrapper_sha256') != environment.get('wrapper', {}).get('sha256')
            or not re.fullmatch('[0-9a-f]{64}', frozen.get('wrapper_sha256', ''))):
        raise ValueError('Completion source/environment differs from the original frozen decision or wrapper')
    return {'candidate_execution_binding': reference(original_path),
            'batch_frozen_inputs': reference(paths['batch_frozen_inputs']),
            'original_resource_decision': reference(decision_path),
            'execution_environment': reference(paths['execution_environment']),
            'source_commit': commit, 'wrapper_sha256': frozen['wrapper_sha256']}


def validate_completed_panel(state, entries):
    """Retained postflight receipts establish all-arm completion, never learning."""
    for entry in entries:
        name = Path(entry['config_path']).stem
        arm = entry.get('arm')
        expected_steps = 2000 if arm in ('reasoning_rl', 'answer_rl') else 250 if arm == 'label_sft' else None
        current = state['candidates'][name]
        receipt = current.get('completion_receipt', {})
        matching = receipt.get('matching_adapter_sha256', {})
        if (expected_steps is None or current.get('exit_code') != 0
                or current.get('config_sha256') != entry.get('config_sha256')
                or receipt.get('completed_optimizer_steps') != expected_steps or receipt.get('observed_worlds') != 2000
                or receipt.get('completion_protocol') != NATURAL_COMPLETION_PROTOCOL
                or not isinstance(matching, dict) or not matching
                or any(not isinstance(key, str) or not key or not re.fullmatch('[0-9a-f]{64}', value)
                       for key, value in matching.items())
                or not re.fullmatch('[0-9a-f]{64}', receipt.get('final_adapter_sha256', ''))
                or any(not re.fullmatch('[0-9a-f]{64}', current.get('artifacts', {}).get(filename, {}).get('sha256', ''))
                       for filename in ('run_manifest.json', 'checkpoint_predictions.jsonl', 'g1_learning_evidence.json'))):
            raise ValueError('Every candidate needs its actual full-pool config/exit/completion evidence')

def observed_candidate_inputs(paths, selected_state, provenance):
    """Validate original saved observations and weights without choosing a run."""
    manifest, receipt = read(paths['run_manifest']), read(paths['learning_evidence'])
    config = read(paths['config'])
    if (manifest.get('status') != 'complete' or manifest.get('config') != config
            or manifest.get('resume_from_checkpoint') is not None or manifest.get('synthetic_integration_test') is not False
            or manifest.get('corpus_manifest_sha256') != file_digest(paths['corpus_manifest'])
            or receipt.get('schema_version') != 3 or receipt.get('run_status') != 'complete'
            or receipt.get('completion_protocol') != NATURAL_COMPLETION_PROTOCOL):
        raise ValueError('Selected run must be complete, fresh, amended and corpus bound')
    for role, key in (('manifest', 'run_manifest'), ('predictions', 'predictions')):
        actual = bound(receipt.get('artifacts', {}).get(role, {}), paths['learning_evidence'].parent, role)
        if actual != paths[key]:
            raise ValueError('Learning receipt refers to another run')
    corpus = read(paths['corpus_manifest'])
    if corpus.get('files', {}).get('evaluation/development__clean.jsonl', {}).get('sha256') != file_digest(paths['development']):
        raise ValueError('Development rows do not match the frozen corpus')
    predictions, development = read_jsonl(paths['predictions']), read_jsonl(paths['development'])
    traces, mappings = extract(predictions, development, config, provenance)
    adapters = {}
    run_root = paths['run_manifest'].parent
    for row in predictions:
        if row.get('format') != 'native_reasoning':
            continue
        adapter = Path(row['checkpoint_id']) / 'adapter_model.safetensors'
        if not adapter.resolve().is_relative_to(run_root):
            raise ValueError('Native checkpoint belongs to another candidate run')
        if str(adapter) not in adapters:
            if not adapter.is_file():
                raise ValueError('Observed matching adapter is missing or changed')
            adapters[str(adapter)] = file_digest(adapter)
        if (adapters[str(adapter)] != row['adapter_sha256']
                or selected_state['completion_receipt']['matching_adapter_sha256'].get(str(adapter)) != adapters[str(adapter)]):
            raise ValueError('Observed matching adapter differs from a raw row or original completion receipt')
    return traces, mappings, adapters, development


def prepare(binding_path, expected_sha, output, *, purpose=PURPOSE):
    if purpose != PURPOSE:
        raise ValueError('This helper cannot prepare replay or Test 2 evidence')
    if Path(output).exists():
        raise FileExistsError('G1 editor input output already exists; no overwrite')
    binding_path = Path(binding_path).resolve()
    if file_digest(binding_path) != expected_sha:
        raise ValueError('Input binding hash mismatch')
    binding = read(binding_path)
    if binding.get('purpose') != PURPOSE or binding.get('scope') != 'selected_reasoning_rl_after_complete_panel':
        raise ValueError('Require an explicitly selected run after the complete amended pilot panel')
    paths = {key: bound(binding.get(key, {}), binding_path.parent, key) for key in ROLES}
    registry, state = (read(paths[key]) for key in ('registry', 'panel_status'))
    selection = verify_panel_selection(paths['selection'], binding['selection']['sha256'])
    candidate = binding.get('candidate_name')
    entries = registry.get('candidates', [])
    if selection['selected_reasoning_rl_candidate'] != candidate:
        raise ValueError('Extraction must use the recomputed selected reasoning-RL candidate')
    for role in ('panel_status', 'registry', 'candidate_execution_binding', 'batch_frozen_inputs',
                 'execution_environment', 'corpus_manifest', 'development'):
        if bound(selection[role], paths['selection'].parent, 'selection ' + role) != paths[role]:
            raise ValueError('Extraction differs from the recomputed selection: ' + role)
    if (selection['selected_configs']['reasoning_rl'] != reference(paths['config'])
            or selection['selected_run_evidence']['reasoning_rl'] != reference(paths['learning_evidence'])):
        raise ValueError('Extraction substituted the computed winning config or run')
    if registry.get('plan_sha256') != digest({key: value for key, value in registry.items() if key != 'plan_sha256'}):
        raise ValueError('Frozen registry content hash mismatch')
    entry = next((row for row in entries if Path(row['config_path']).stem == candidate), None)
    if not entry or entry.get('config_sha256') != file_digest(paths['config']):
        raise ValueError('Selected candidate config does not match the frozen registry')
    lineage = completion_lineage(paths, binding, state, registry, entry)
    selected_state = state['candidates'][candidate]
    if (selected_state.get('config_sha256') != file_digest(paths['config']) or selected_state.get('exit_code') != 0
            or selected_state.get('completion_receipt', {}).get('completed_optimizer_steps') != 2000
            or selected_state.get('completion_receipt', {}).get('observed_worlds') != 2000
            or selected_state.get('completion_receipt', {}).get('completion_protocol') != NATURAL_COMPLETION_PROTOCOL):
        raise ValueError('Completed panel records a different selected config')
    for role, filename in (('run_manifest', 'run_manifest.json'), ('predictions', 'checkpoint_predictions.jsonl'),
                           ('learning_evidence', 'g1_learning_evidence.json')):
        if selected_state.get('artifacts', {}).get(filename, {}).get('sha256') != file_digest(paths[role]):
            raise ValueError('Selected run artifact differs from completed panel receipt')
    provenance = {'binding': reference(binding_path), **{key: reference(path) for key, path in paths.items()},
                  'source_commit': binding['source_commit'], 'completion_source_lineage': lineage, 'purpose': PURPOSE}
    traces, mappings, adapters, development = observed_candidate_inputs(paths, selected_state, provenance)
    edits = prepare_edit_manifest(traces, development, max_worlds=120, seed=2026, stage='pilot')
    plan = prepare_generation_plan(edits, **EDITOR)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    artifacts = {'edit_manifest.json': edits, 'editor_plan.json': plan, 'classifications.mapping.json': mappings,
                 'extraction.json': {'schema_version': 1, 'purpose': PURPOSE, 'scope': binding['scope'],
                    'created_utc': datetime.now(timezone.utc).isoformat(), 'provenance': provenance,
                    'source_records': len(traces), 'native_checkpoints': 9, 'stories_per_checkpoint': 120,
                    'adapter_sha256': adapters, 'observed_prompt_ids': False, 'generated_editor_outputs': False,
                    'scientific_gate_decision': None, 'selection_performed': False}}
    for name, value in artifacts.items():
        with (output / name).open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write('\n')
    with (output / 'edit_only_sources.jsonl').open('x', encoding='utf-8', newline='\n') as stream:
        for trace in traces:
            stream.write(json.dumps(trace, sort_keys=True) + '\n')
    prepared = {'schema_version': 1, 'purpose': PURPOSE,
            'files': {path.name: reference(path) for path in output.iterdir()},
            'editor_launched': False, 'scientific_gate_decision': None}
    prepared['prepared_sha256'] = digest(prepared)
    _write_new(output / 'prepared.json', prepared)
    return {**prepared, 'prepared': reference(output / 'prepared.json')}


def inspect_completed_candidate(execution_binding, expected_sha, panel_status, candidate, output):
    """Inspect one finished candidate before selection, without preparing editor inputs.

    This deliberately does not produce prepared.json, a generation plan or G1
    eligibility. The full panel and computed selection remain necessary later.
    All nine native adapter files must still be present at their original paths.
    """
    output = Path(output)
    if output.exists():
        raise FileExistsError('Candidate inspection output already exists')
    original_path, status_path = Path(execution_binding).resolve(), Path(panel_status).resolve()
    if file_digest(original_path) != expected_sha:
        raise ValueError('Original execution binding hash mismatch')
    original, state = read(original_path), read(status_path)
    paths = {'candidate_execution_binding': original_path, 'panel_status': status_path,
             'batch_frozen_inputs': status_path.parent / 'frozen-inputs.json'}
    for role in ('registry', 'execution_environment', 'corpus_manifest', 'development'):
        paths[role] = bound(original.get(role, {}), original_path.parent, role)
    registry = read(paths['registry'])
    if registry.get('plan_sha256') != digest({key: value for key, value in registry.items() if key != 'plan_sha256'}):
        raise ValueError('Frozen registry content hash mismatch')
    entry = next((row for row in registry.get('candidates', [])
                  if Path(row['config_path']).stem == candidate), None)
    if entry is None or entry.get('arm') != 'reasoning_rl':
        raise ValueError('Inspection requires one registered reasoning-RL candidate')
    current = state.get('candidates', {}).get(candidate, {})
    if current.get('status') != 'completed':
        raise ValueError('Candidate is not completed; preserve existing records and inspect after completion')
    validate_completed_panel(state, [entry])
    paths['config'] = bound({'path': entry['config_path'], 'sha256': entry['config_sha256']},
                            paths['registry'].parent, 'config')
    for role, filename in (('run_manifest', 'run_manifest.json'),
                           ('predictions', 'checkpoint_predictions.jsonl'),
                           ('learning_evidence', 'g1_learning_evidence.json')):
        paths[role] = bound(current.get('artifacts', {}).get(filename, {}), status_path.parent, role)
    source = read(paths['execution_environment']).get('source', {}).get('commit')
    lineage = completion_lineage(paths, {'source_commit': source}, state, registry, entry)
    provenance = {**{key: reference(path) for key, path in paths.items()},
                  'source_commit': source, 'completion_source_lineage': lineage, 'purpose': PURPOSE}
    before = {key: file_digest(path) for key, path in paths.items()}
    traces, mappings, adapters, _ = observed_candidate_inputs(paths, current, provenance)
    if before != {key: file_digest(path) for key, path in paths.items()}:
        raise ValueError('Candidate inputs changed during inspection')
    report = {'schema_version': 1, 'kind': 'completed_candidate_extraction_inspection',
              'observed_utc': datetime.now(timezone.utc).isoformat(), 'candidate_name': candidate,
              'provenance': provenance, 'source_records': len(traces),
              'native_checkpoints': 9, 'stories_per_checkpoint': 120,
              'artifact_ids_sha256': digest([row['artifact_id'] for row in mappings]),
              'adapter_sha256': adapters, 'editor_launched': False,
              'selection_performed': False, 'selected_run_claim': False,
              'g1_eligible': False, 'scientific_gate_decision': None}
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_new(output, report)
    return {**report, 'receipt': reference(output)}


def editor_evidence(plan_path, generated_path, mappings):
    """After actual editor generation, construct the existing G1 receipt shape."""
    plan, generated = read(plan_path), read(generated_path)
    if (plan.get('model') != {'id': EDITOR['model_id'], 'revision': EDITOR['revision']}
            or plan.get('prompt_version') != PROMPT_VERSION or plan.get('prompt') != CONTROL_PROMPT
            or plan.get('decoding') != {'max_new_tokens': 2048, 'do_sample': False, 'temperature': 1.0, 'top_p': 1.0, 'seed': 2026}):
        raise ValueError('G1 editor identity, prompt and decoding must match the frozen prospective policy')
    if generated.get('generation_plan') != plan:
        raise ValueError('Generated edits differ from frozen editor plan')
    contracts = validate_generated_batch(generated, allow_errors=True)
    if len(mappings) != len(contracts) or {row['artifact_id'] for row in mappings} != set(contracts):
        raise ValueError('Editor results must retain all checkpoint classifications, including failures')
    return {'schema_version': 1, 'completion_protocol': NATURAL_COMPLETION_PROTOCOL,
            'purpose': PURPOSE, 'editor_plan': reference(plan_path), 'generated_edits': reference(generated_path),
            'classifications': mappings, 'human_audit': 'not_performed_by_this_helper'}


def _write_new(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def _prepared(path, expected_sha):
    path = Path(path).resolve()
    if file_digest(path) != expected_sha:
        raise ValueError('Prepared G1 input receipt hash mismatch')
    value = read(path)
    if (value.get('schema_version') != 1 or value.get('purpose') != PURPOSE
            or value.get('prepared_sha256') != digest({k: v for k, v in value.items() if k != 'prepared_sha256'})):
        raise ValueError('Invalid prepared G1 input receipt')
    names = {'edit_only_sources.jsonl', 'edit_manifest.json', 'editor_plan.json',
             'classifications.mapping.json', 'extraction.json'}
    if set(value.get('files', {})) != names:
        raise ValueError('Prepared G1 input set is incomplete')
    files = {name: bound(ref, path.parent, name) for name, ref in value['files'].items()}
    if any(file.parent != path.parent or file.name != name for name, file in files.items()):
        raise ValueError('Prepared G1 files must belong to the original extraction directory')
    plan, manifest = read(files['editor_plan.json']), read(files['edit_manifest.json'])
    _validate_editor_plan(plan)
    if digest(manifest) != plan['input_manifest_sha256']:
        raise ValueError('Prepared edit manifest differs from frozen editor plan')
    traces = read_jsonl(files['edit_only_sources.jsonl'])
    mappings = read(files['classifications.mapping.json'])
    extraction = read(files['extraction.json'])
    if (len(traces) != 1080 or len(mappings) != 1080 or len(plan['contract_specs']) != 1080
            or extraction.get('source_records') != 1080 or extraction.get('native_checkpoints') != 9
            or extraction.get('stories_per_checkpoint') != 120 or extraction.get('purpose') != PURPOSE
            or len({row['artifact_id'] for row in traces}) != 1080
            or {row['artifact_id'] for row in traces} != set(plan['contract_specs'])
            or {row['artifact_id'] for row in mappings} != set(plan['contract_specs'])):
        raise ValueError('G1 editor must retain all 1080 observed native records')
    for trace in traces:
        _validate_source_trace(trace)
        if (trace.get('purpose') != PURPOSE or trace.get('replay_eligible') is not False
                or trace.get('generation_prompt_observed') is not False
                or trace.get('generation_prompt') is not None or trace.get('generation_prompt_ids') is not None):
            raise ValueError('G1 sources must remain completion-only and ineligible for replay')
    return value, files, plan, manifest, traces, mappings


def _validate_editor_plan(plan):
    _validate_plan(plan)
    if (plan.get('model') != {'id': EDITOR['model_id'], 'revision': EDITOR['revision']}
            or plan.get('prompt_version') != PROMPT_VERSION or plan.get('prompt') != CONTROL_PROMPT
            or plan.get('decoding') != {'max_new_tokens': 2048, 'do_sample': False,
                'temperature': 1.0, 'top_p': 1.0, 'seed': 2026}):
        raise ValueError('G1 editor must use the fixed model, revision, prompt and decoding')


class SDPAEditor(HFBackend):
    """G1 editor only: source checkpoints are never loaded or regenerated."""
    attention_implementation = 'sdpa'

    def __init__(self, *, device='auto'):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.checkpoint_id, self.revision = EDITOR['model_id'], EDITOR['revision']
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_id, revision=self.revision)
        kwargs = {'revision': self.revision, 'torch_dtype': 'auto', 'attn_implementation': 'sdpa'}
        if device == 'auto':
            kwargs['device_map'] = 'auto'
        self.model = AutoModelForCausalLM.from_pretrained(self.checkpoint_id, **kwargs)
        if device != 'auto':
            self.model.to(device)
        self.model.eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def synchronize(self):
        devices = {parameter.device.index for parameter in self.model.parameters()
                   if parameter.device.type == 'cuda'}
        for device in devices:
            self.torch.cuda.synchronize(device)


class _MeasuredEditor:
    def __init__(self, backend):
        if getattr(backend, 'attention_implementation', None) != 'sdpa':
            raise ValueError('G1 editor requires explicit SDPA; source traces must not be regenerated')
        if (str(backend.checkpoint_id) != EDITOR['model_id']
                or backend.revision != EDITOR['revision']):
            raise ValueError('G1 editor backend differs from frozen model/revision')
        self.backend = backend
        self.checkpoint_id, self.revision = backend.checkpoint_id, backend.revision
        self.tokenizer = backend.tokenizer
        self.last_measurement = None

    def token_ids(self, text):
        return self.backend.token_ids(text)

    def generate(self, prompt, **kwargs):
        self.last_measurement = None
        input_ids = self.token_ids(prompt)
        if not isinstance(input_ids, list) or any(type(token) is not int for token in input_ids):
            raise ValueError('Editor prompt token IDs were not observed')
        sync = getattr(self.backend, 'synchronize', lambda: None)
        sync()
        start = time.perf_counter()
        output = None
        try:
            output = self.backend.generate(prompt, **kwargs)
            return output
        finally:
            sync()
            elapsed = time.perf_counter() - start
            generated = output.get('generated_ids') if isinstance(output, dict) else None
            if generated is not None and (not isinstance(generated, list)
                    or any(type(token) is not int for token in generated)):
                generated = None
            self.last_measurement = {'schema_version': 1, 'kind': 'observed_editor_call',
                'attention_implementation': 'sdpa', 'prompt_sha256': digest(prompt),
                'input_tokens': len(input_ids), 'output_tokens': None if generated is None else len(generated),
                'wall_seconds': elapsed, 'output_token_count_source': 'actual_generated_ids' if generated is not None else 'unavailable',
                'device_synchronization': 'backend_synchronize' if hasattr(self.backend, 'synchronize') else 'unavailable_test_backend'}
            if isinstance(output, dict):
                output['g1_resource_measurement'] = self.last_measurement


@contextmanager
def _job_lock(path):
    """OS lock releases on interruption; the lock file is retained as evidence."""
    stream = Path(path).open('a+b')
    acquired = False
    try:
        if os.name == 'nt':
            import msvcrt
            if stream.tell() == 0:
                stream.write(b'\0')
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        acquired = True
        yield
    except BlockingIOError as error:
        raise ValueError('Another G1 editor process owns this exact output journal') from error
    finally:
        if acquired:
            if os.name == 'nt':
                import msvcrt
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()


class _ProfilePause(BaseException):
    """Pause only after persisting a complete actual result; never an edit error."""


class _JournalOnlyEditor:
    """Finalize a fully durable journal without loading the editor again."""
    last_measurement = None

    def __init__(self, completed, plan):
        from types import SimpleNamespace
        self.checkpoint_id, self.revision = plan['model']['id'], plan['model']['revision']
        self.tokenizer = SimpleNamespace(chat_template=completed[0]['generation']['chat_template'])

    def generate(self, *args, **kwargs):
        raise RuntimeError('A completed journal must never make another model call')


def _journal_rows(path, plan):
    rows = read_jsonl(path) if Path(path).exists() else []
    seen = set()
    for row in rows:
        key = row.get('artifact_id')
        if key in seen or key not in plan['contract_specs']:
            raise ValueError('G1 generation journal contains an unknown or duplicate contract')
        _validate_record(row, plan, allow_unresolved=True)
        seen.add(key)
    return rows


class _CallStartWriteFailure(BaseException):
    """A call cannot proceed without a durable start record; never an edit error."""


def _validate_call_starts(started, completed, plan, ordered):
    starts = read_jsonl(started) if Path(started).exists() else []
    if ([row['artifact_id'] for row in completed] != ordered[:len(completed)]
            or len(starts) != len(completed)):
        raise ValueError('G1 has an interrupted unresolved call or noncanonical journal; no resampling')
    for begin, contract in zip(starts, completed):
        expected = {'artifact_id': contract['artifact_id'],
                    'prompt_sha256': contract['generation']['prompt_sha256'],
                    'plan_sha256': plan['plan_sha256']}
        if begin != expected:
            raise ValueError('G1 call-start record differs from its durable completed result')


def generate(prepared_path, expected_sha, output, *, device='auto', resume=False, max_calls=None, backend=None):
    """Run/resume the frozen editor; optional call limit measures this same job.

    One permanent claim and OS lock cover the preparation, across output paths.
    Every call start and completed result is fsynced. An interrupted in-flight
    call, partial journal line or alternate output cannot be silently resampled.
    No scientific approval is issued.
    """
    _, files, plan, manifest, traces, _ = _prepared(prepared_path, expected_sha)
    if max_calls is not None and (type(max_calls) is not int or max_calls < 1):
        raise ValueError('max_calls must be a positive bounded profile count')
    output = Path(output).resolve()
    journal, claim = Path(str(output) + '.journal.jsonl'), Path(str(output) + '.job.json')
    started = Path(str(output) + '.started.jsonl')
    prepared_path = Path(prepared_path).resolve()
    owner = Path(str(prepared_path) + '.generation.job.json')
    lock = Path(str(prepared_path) + '.generation.lock')
    if output.exists():
        raise FileExistsError('Generated G1 batch already exists; no overwrite or resampling')
    output.parent.mkdir(parents=True, exist_ok=True)
    with _job_lock(lock):
        if output.exists():
            raise FileExistsError('Generated G1 batch already exists')
        if any(path.exists() != resume for path in (journal, claim, started, owner)):
            raise ValueError('Existing G1 journal/claim requires explicit resume; fresh job must omit resume')
        identity = {'schema_version': 1, 'purpose': PURPOSE, 'prepared': reference(prepared_path),
            'generation_plan': reference(files['editor_plan.json']), 'output': str(output),
            'attention_implementation': 'sdpa', 'source_generation': 'none_saved_completions_only'}
        if resume and (read(claim) != identity or read(owner) != identity):
            raise ValueError('G1 journal belongs to different prepared inputs/output')
        completed = _journal_rows(journal, plan)
        ordered = [contract['artifact_id'] for contract in manifest['edits']]
        _validate_call_starts(started, completed, plan, ordered)
        if not resume:
            _write_new(owner, identity)
            _write_new(claim, identity)
        calls = 0
        paused = False
        with journal.open('a' if resume else 'x', encoding='utf-8', newline='\n') as stream, \
                started.open('a' if resume else 'x', encoding='utf-8', newline='\n') as begin_stream:
            # Create both journals before loading. Loading failure can resume
            # with no call starts; an interrupted actual call cannot.
            for handle in (stream, begin_stream):
                handle.flush()
                os.fsync(handle.fileno())
            if len(completed) == len(traces):
                editor = _JournalOnlyEditor(completed, plan)
            else:
                measured = _MeasuredEditor(backend if backend is not None else SDPAEditor(device=device))

                class ClaimedEditor:
                    checkpoint_id, revision, tokenizer = measured.checkpoint_id, measured.revision, measured.tokenizer

                    def token_ids(self, text):
                        return measured.token_ids(text)

                    @property
                    def last_measurement(self):
                        return measured.last_measurement

                    def generate(self, prompt, **kwargs):
                        begin = {'artifact_id': ordered[len(completed) + calls],
                                 'prompt_sha256': digest(prompt), 'plan_sha256': plan['plan_sha256']}
                        try:
                            begin_stream.write(json.dumps(begin, sort_keys=True, allow_nan=False) + '\n')
                            begin_stream.flush()
                            os.fsync(begin_stream.fileno())
                        except Exception as error:
                            # The generic editor catches ordinary backend errors;
                            # a journal failure must instead stop the job before
                            # any unrecorded inference or subsequent contract.
                            raise _CallStartWriteFailure(str(error)) from error
                        return measured.generate(prompt, **kwargs)

                editor = ClaimedEditor()
            def persist(contract):
                nonlocal calls
                measurement = editor.last_measurement
                if (measurement is not None
                        and measurement['prompt_sha256'] == contract['generation']['prompt_sha256']):
                    meta = contract['generation'].get('generation_metadata') or {}
                    meta['g1_resource_measurement'] = measurement
                    contract['generation']['generation_metadata'] = meta
                    contract['generation']['generation_sha256'] = digest({k: v for k, v
                        in contract['generation'].items() if k != 'generation_sha256'})
                stream.write(json.dumps(contract, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
                calls += 1
                if max_calls is not None and calls >= max_calls and len(completed) + calls < len(traces):
                    raise _ProfilePause()
            try:
                result = generate_edit_controls(editor, traces, manifest, plan,
                    on_result=persist, completed_results=completed)
            except _ProfilePause:
                paused = True
        durable = _journal_rows(journal, plan)
        _validate_call_starts(started, durable, plan, ordered)
        if not paused:
            # Reassemble from the durable records: persistence attaches resource
            # metadata even when the underlying call failed, and the generic
            # generator callback intentionally receives a defensive copy.
            if len(durable) != len(traces):
                raise ValueError('Incomplete durable G1 journal cannot publish a final batch')
            result = generate_edit_controls(_JournalOnlyEditor(durable, plan), traces, manifest, plan,
                                            completed_results=durable)
            _write_new(output, result)
        report = resource_report(prepared_path, expected_sha, journal_path=journal)
        return {'status': 'paused_after_requested_calls' if paused else 'complete',
                'new_calls': calls, 'journal': reference(journal),
                'started': reference(started), 'preparation_claim': reference(owner),
                'generated': reference(output) if output.exists() else None,
                'resources': report, 'scientific_gate_decision': None, 'human_audit': 'not_performed'}


def resource_report(prepared_path, expected_sha, *, journal_path=None, hourly_gbp=None):
    """Describe fixed workload and project from observed editor calls only."""
    _, _, plan, _, _, _ = _prepared(prepared_path, expected_sha)
    if hourly_gbp is not None and (type(hourly_gbp) not in (float, int)
            or not math.isfinite(hourly_gbp) or hourly_gbp < 0):
        raise ValueError('Hourly GBP rate must be a finite nonnegative observed rate')
    if journal_path is not None and not Path(journal_path).is_file():
        raise ValueError('Requested editor journal is missing; cannot report measured resources')
    contracts = _journal_rows(journal_path, plan) if journal_path is not None else []
    observations = []
    unavailable = 0
    for contract in contracts:
        generation = contract['generation']
        row = (generation.get('generation_metadata') or {}).get('g1_resource_measurement')
        if row is None:
            unavailable += 1
            continue
        if (row.get('kind') != 'observed_editor_call' or row.get('attention_implementation') != 'sdpa'
                or row.get('prompt_sha256') != generation['prompt_sha256']
                or type(row.get('input_tokens')) is not int or row['input_tokens'] < 0
                or type(row.get('wall_seconds')) not in (float, int)
                or not math.isfinite(row['wall_seconds']) or row['wall_seconds'] < 0
                or (row.get('output_tokens') is not None and (type(row['output_tokens']) is not int
                    or not 0 <= row['output_tokens'] <= EDITOR['max_new_tokens']))):
            raise ValueError('Invalid or mismatched measured editor resource row')
        observations.append(row)
    count, total = len(observations), len(plan['contract_specs'])
    elapsed = sum(row['wall_seconds'] for row in observations)
    output_rows = [row for row in observations if row['output_tokens'] is not None]
    generated = sum(row['output_tokens'] for row in output_rows)
    projected_seconds = elapsed / count * total if count else None
    return {'schema_version': 1, 'purpose': PURPOSE, 'planned_calls': total,
        'max_output_tokens_per_call': 2048, 'max_total_output_tokens': total * 2048,
        'completed_contracts': len(contracts), 'remaining_calls': total - len(contracts),
        'observed_calls_with_timing': count, 'calls_without_timing': unavailable,
        'observed_inference_wall_seconds': elapsed if count else None,
        'observed_input_tokens': sum(row['input_tokens'] for row in observations) if count else None,
        'observed_output_tokens': generated if output_rows else None,
        'calls_without_output_token_count': len(contracts) - len(output_rows),
        'projected_full_inference_seconds': projected_seconds,
        'projected_remaining_inference_seconds': elapsed / count * (total - len(contracts)) if count else None,
        'projected_full_input_tokens': sum(row['input_tokens'] for row in observations) / count * total if count else None,
        'projected_full_output_tokens': generated / len(output_rows) * total if output_rows else None,
        'hourly_gbp_supplied': hourly_gbp,
        'projected_full_inference_gbp': projected_seconds / 3600 * hourly_gbp
            if projected_seconds is not None and hourly_gbp is not None else None,
        'measurement_status': 'observed_editor_calls' if count else 'not_measured',
        'projection_note': 'Linear descriptive projection from retained observed calls; the sequential subset is not random. '
            'Excludes loading, idle time, transfer, human audit and failures without completed timing. '
            'Output-token projection uses only calls with observed generated IDs. No actual rental debit is inferred.',
        'spend_approval': False, 'scientific_gate_decision': None}


def finalize(prepared_path, expected_sha, generated_path, output):
    """Bind all 1080 actual outcomes as G1 evidence without judging the gate."""
    _, files, _, _, _, mappings = _prepared(prepared_path, expected_sha)
    output = Path(output)
    if output.exists():
        raise FileExistsError('G1 editability evidence already exists; no overwrite')
    evidence = editor_evidence(files['editor_plan.json'], generated_path, mappings)
    evidence.update(prepared=reference(prepared_path), extraction=reference(files['extraction.json']),
                    classifications_mapping=reference(files['classifications.mapping.json']))
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_new(output, evidence)
    return {'status': 'complete', 'evidence': reference(output), 'classifications': len(mappings),
            'human_audit': 'not_performed', 'scientific_gate_decision': None}
