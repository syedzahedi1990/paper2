"""Derive family settings from the immutable, actually attempted pilot panel.

This module does no training, editor generation or scientific gate approval.
Selection never reads trace quality. An empty but correctly answered completion
has precisely the same native score as any other valid completion.
"""
from __future__ import annotations

from datetime import datetime
from fractions import Fraction
import json
import re
from pathlib import Path

from paper2.io import digest, file_digest, read_jsonl
from .config import NATURAL_COMPLETION_PROTOCOL, TrainingConfig
from .data import natural_completion_outcome, rows_digest
from .selection import (assert_current_design, development_selection_panel,
                        development_selection_spec, selection_failure_policy)

METHOD = 'amended_pilot_family_selection_v1'
ARMS = ('reasoning_rl', 'answer_rl', 'label_sft')
BETAS = {'reasoning_rl': .04, 'answer_rl': .04, 'label_sft': 0.}
FAILED = ('failed', 'stopped', 'timed_out')
ARTIFACTS = {'manifest': 'run_manifest.json', 'compute': 'compute_report.json',
             'evaluations': 'checkpoint_evaluations.jsonl',
             'predictions': 'checkpoint_predictions.jsonl',
             'trace_health': 'checkpoint_trace_health.jsonl'}


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _ref(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': file_digest(path)}


def _bound(item, base, role):
    if not isinstance(item, dict) or not isinstance(item.get('path'), str):
        raise ValueError('Missing bound ' + role)
    path = (Path(base) / item['path']).resolve()
    if not path.is_file() or file_digest(path) != item.get('sha256'):
        raise ValueError('Missing or modified bound ' + role)
    return path


def _utc(value, role):
    try:
        result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except (TypeError, ValueError, AttributeError) as error:
        raise ValueError(role + ' needs an explicit UTC timestamp') from error
    if result.tzinfo is None or result.utcoffset().total_seconds() != 0:
        raise ValueError(role + ' needs an explicit UTC timestamp')
    return result


def _attribution(value, role):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Missing ' + role)


def _source_inputs(binding_path, binding_sha256, status_path):
    binding_path = Path(binding_path).resolve()
    if file_digest(binding_path) != binding_sha256:
        raise ValueError('Execution binding hash mismatch')
    binding = _read(binding_path)
    state = _read(status_path)
    frozen_path = status_path.parent / 'frozen-inputs.json'
    frozen = _read(frozen_path)
    if (binding.get('schema_version') != 2
            or binding.get('purpose') != 'amended_pilot_candidate_execution'
            or binding.get('completion_protocol') != NATURAL_COMPLETION_PROTOCOL
            or binding.get('beta_by_arm') != BETAS
            or state.get('binding_sha256') != binding_sha256
            or frozen.get('binding_sha256') != binding_sha256):
        raise ValueError('Panel differs from the amended original execution binding')
    paths = {name: _bound(binding.get(name), binding_path.parent, name) for name in
             ('registry', 'decision', 'execution_environment', 'corpus_manifest', 'training', 'development')}
    inventory = frozen.get('files_sha256', {})
    for path in (binding_path, *paths.values()):
        if inventory.get(str(path)) != file_digest(path):
            raise ValueError('Original source input absent or changed in frozen-inputs: ' + str(path))
    registry, decision, environment = (_read(paths[name]) for name in
                                       ('registry', 'decision', 'execution_environment'))
    source = environment.get('source', {})
    commit = source.get('commit')
    if (not isinstance(commit, str) or not re.fullmatch('[0-9a-f]{40}', commit)
            or not re.fullmatch('[0-9a-f]{40}', source.get('tree', ''))
            or decision.get('source_commit') != commit
            or decision.get('execution_environment_sha256') != file_digest(paths['execution_environment'])
            or frozen.get('wrapper_sha256') != environment.get('wrapper', {}).get('sha256')
            or not re.fullmatch('[0-9a-f]{64}', frozen.get('wrapper_sha256', ''))
            or registry.get('decision_sha256') != file_digest(paths['decision'])):
        raise ValueError('Original source, wrapper or resource decision lineage differs')
    expected = development_selection_panel()
    entries = registry.get('candidates', [])
    if (registry.get('schema_version') != 2 or registry.get('panel') != 'four_rates_constant'
            or registry.get('candidate_runs') != 12 or len(entries) != 12
            or registry.get('selection_rule') != development_selection_spec()['selection']
            or registry.get('beta_by_arm') != BETAS
            or registry.get('completion_protocol') != NATURAL_COMPLETION_PROTOCOL
            or registry.get('plan_sha256') != digest({k: v for k, v in registry.items() if k != 'plan_sha256'})
            or {item.get('config_path') for item in entries} != set(expected)
            or binding.get('config_sha256') != {item['config_path']: item.get('config_sha256') for item in entries}):
        raise ValueError('Require the exact complete frozen twelve-candidate registry and unchanged rule')
    configs = {}
    for entry in entries:
        name = entry['config_path']
        path = _bound({'path': name, 'sha256': entry['config_sha256']}, paths['registry'].parent, name)
        config = _read(path)
        assert_current_design(config)
        # Retention was an explicit frozen resource choice, not a scientific
        # hyperparameter. The actual amended panel uses two resumable snapshots.
        retention = decision.get('save_total_limit')
        if (retention not in (2, 6) or config != {**expected[name], 'save_total_limit': retention}
                or inventory.get(str(path)) != file_digest(path)
                or any(entry.get(key) != config[key] for key in
                       ('arm', 'learning_rate', 'lr_scheduler_type', 'warmup_ratio', 'completion_protocol', 'beta'))):
            raise ValueError('Candidate config changes the frozen panel or source lineage: ' + name)
        TrainingConfig(**config).validate()
        configs[Path(name).stem] = (path, config)
    corpus = _read(paths['corpus_manifest'])
    if any(corpus.get('config', {}).get(key) != count for key, count in
           (('train', 2000), ('development', 120), ('confirmation', 120))):
        raise ValueError('Require the fixed 2000/120/120 corpus')
    for key, relative in (('training', 'training/clean.jsonl'), ('development', 'evaluation/development__clean.jsonl')):
        if corpus.get('files', {}).get(relative, {}).get('sha256') != file_digest(paths[key]):
            raise ValueError('Selection substituted frozen ' + key + ' data')
    training = read_jsonl(paths['training'])
    development = [row for row in read_jsonl(paths['development']) if row.get('view') == 'direct']
    train_worlds = {row.get('world_id') for row in training}
    if (len(training) != 2000 or len(train_worlds) != 2000 or None in train_worlds
            or any(row.get('split') != 'train' or row.get('view') != 'direct' for row in training)
            or len(development) != 120 or len({row.get('id') for row in development}) != 120
            or len({row.get('world_id') for row in development}) != 120
            or any(not row.get('id') or not row.get('world_id') or row.get('split') != 'development'
                   or row['world_id'] in train_worlds for row in development)):
        raise ValueError('Require all disjoint frozen training and direct development stories')
    return paths, configs, registry, state, source, frozen_path, training, development


def _attempt(output, name, current, config_ref, events):
    """Require actual terminal process/launch records, including failed attempts."""
    if current.get('status') not in ('completed', *FAILED):
        raise ValueError('All twelve candidates need terminal attempts; unattempted/running: ' + name)
    code = current.get('exit_code')
    if (type(code) is not int or current.get('config_sha256') != config_ref['sha256']
            or type(current.get('pid')) is not int or current['pid'] <= 0):
        raise ValueError('Candidate lacks its actual process/config/exit record: ' + name)
    start, end = (_utc(current.get(key), name + ' ' + key) for key in ('started_utc', 'finished_utc'))
    if end < start:
        raise ValueError('Candidate terminal time precedes launch')
    command = current.get('command')
    if (not isinstance(command, list) or not all(isinstance(arg, str) for arg in command)
            or '--config' not in command or '--output' not in command
            or command.index('--config') + 1 >= len(command) or command.index('--output') + 1 >= len(command)
            or command[command.index('--config') + 1] != config_ref['path']
            or command[command.index('--output') + 1] != str(output)):
        raise ValueError('Candidate launch command differs from bound config/output')
    starts = [row for row in events if row.get('event') == 'candidate_started' and row.get('candidate') == name]
    processes = [row for row in events if row.get('event') == 'candidate_process_started' and row.get('candidate') == name]
    if len(starts) != 1 or len(processes) != 1 or processes[0].get('pid') != current['pid']:
        raise ValueError('Each candidate needs exactly one actual launch; retries are not authorized')
    terminal = [row for row in events if row.get('candidate') == name and row.get('event') in
                ('candidate_completed', 'candidate_failed_batch_stopped')]
    if current['status'] == 'completed':
        if code != 0 or len(terminal) != 1 or terminal[0].get('event') != 'candidate_completed':
            raise ValueError('Completed candidate lacks successful supervisor terminal evidence')
    elif not terminal and not (current.get('error') or any(row.get('event') == 'batch_failed' for row in events)):
        raise ValueError('Failed candidate lacks its retained terminal failure evidence')
    log = output.parent.parent / (name + '.launch.log')
    if not log.is_file():
        raise ValueError('Candidate launch log must be retained: ' + name)
    artifacts = current.get('artifacts')
    if not isinstance(artifacts, dict):
        raise ValueError('Candidate terminal artifact snapshot is required')
    actual = {}
    for filename, item in artifacts.items():
        if Path(filename).name != filename:
            raise ValueError('Candidate artifact name must be a direct run file')
        path = _bound(item, output, name + ' ' + filename)
        if path != output / filename or item.get('bytes') != path.stat().st_size:
            raise ValueError('Terminal artifact snapshot points outside its actual run')
        actual[filename] = _ref(path)
    return {'launch_log': _ref(log), 'terminal_artifacts': actual,
            'attempt_record_sha256': digest(current)}, end


def _completed(output, config, current, source, corpus_sha, training, development):
    """Validate real exposure, raw observations and retained source artifacts."""
    paths = {role: output / filename for role, filename in ARTIFACTS.items()}
    paths['learning_evidence'] = output / 'g1_learning_evidence.json'
    for path in paths.values():
        if path.name not in current['artifacts'] or not path.is_file():
            raise ValueError('Completed run missing terminal artifact: ' + path.name)
    manifest, compute, evidence = (_read(paths[role]) for role in ('manifest', 'compute', 'learning_evidence'))
    typed = TrainingConfig(**config)
    budget = typed.training_budget(2000)
    step = budget['resolved_max_steps']
    if (manifest.get('status') != 'complete' or manifest.get('config') != config
            or manifest.get('synthetic_integration_test') is not False
            or manifest.get('resume_from_checkpoint') is not None
            or manifest.get('training_budget') != budget
            or manifest.get('training_examples') != 2000
            or manifest.get('training_sha256') != rows_digest(training)
            or manifest.get('corpus_manifest_sha256') != corpus_sha
            or compute.get('training_examples') != 2000
            or compute.get('completed_optimizer_steps') != step
            or compute.get('prompt_exposures') != budget['planned_prompt_exposures']
            or set(compute.get('observed_world_ids', [])) != {row['world_id'] for row in training}):
        raise ValueError('Candidate did not complete its exact fresh-base full-pool exposure')
    recorded_source = manifest.get('source_identity')
    if recorded_source is not None:
        tracked = source.get('tracked_files_sha256', {})
        if (recorded_source.get('commit') != source['commit'] or recorded_source.get('tree') != source['tree']
                or recorded_source.get('tracked_changes') is not False
                or not recorded_source.get('files_sha256')
                or any(tracked.get(name) != sha for name, sha in recorded_source['files_sha256'].items())):
            raise ValueError('Run source identity differs from its original execution environment')
    if (evidence.get('schema_version') != 3 or evidence.get('run_status') != 'complete'
            or evidence.get('synthetic') is not False
            or evidence.get('completion_protocol') != NATURAL_COMPLETION_PROTOCOL
            or evidence.get('run_identity') != {key: config[key] for key in ('arm', 'dataset', 'seed', 'model', 'revision', 'stage')}):
        raise ValueError('Completed run needs its actual finalized learning evidence')
    for role in ARTIFACTS:
        if _bound(evidence.get('artifacts', {}).get(role), output, role) != paths[role]:
            raise ValueError('Learning evidence points to a different run')
    native = 'native_reasoning' if typed.reasoning else 'native_answer'
    expected = {(s, native) for s in typed.evaluation_schedule(step)} | {
        (s, 'immediate') for s in typed.immediate_evaluation_schedule(step)}
    metrics = read_jsonl(paths['evaluations'])
    if len(metrics) != len(expected) or {(row.get('step'), row.get('format')) for row in metrics} != expected:
        raise ValueError('Candidate learning curves lack complete scheduled panels')
    by_key = {(row['step'], row['format']): row for row in metrics}
    panels = {}
    for row in read_jsonl(paths['predictions']):
        panels.setdefault((row.get('step'), row.get('format')), []).append(row)
    if set(panels) != expected:
        raise ValueError('Candidate predictions lack complete scheduled panels')
    worlds = {row['id']: row for row in development}
    hashes = {}
    history = []
    for (s, fmt), metric in sorted(by_key.items()):
        checkpoint = output / 'evaluated_checkpoints' / f'checkpoint-{s}'
        weights = checkpoint / 'adapter_model.safetensors'
        if str(weights) not in hashes:
            if not weights.is_file():
                raise ValueError('Matching snapshot must be retained')
            hashes[str(weights)] = file_digest(weights)
        identity = {'arm': config['arm'], 'dataset': 'clean', 'evaluation_dataset': 'clean',
                    'seed': 1, 'model': config['model'], 'revision': config['revision'],
                    'view': 'direct', 'split': 'development', 'step': s, 'format': fmt,
                    'completion_protocol': NATURAL_COMPLETION_PROTOCOL, 'checkpoint_id': str(checkpoint),
                    'path': str(checkpoint), 'checkpoint_path': str(checkpoint),
                    'checkpoint_kind': 'base' if s == 0 else 'trained',
                    'adapter_sha256': hashes[str(weights)]}
        rows = panels[(s, fmt)]
        if (any(metric.get(key) != value for key, value in identity.items())
                or metric.get('n') != 120 or metric.get('decoding') != 'greedy_full_vocabulary'
                or len(rows) != 120 or {row.get('id') for row in rows} != set(worlds)):
            raise ValueError('Native/learning-curve identities or full development panel differ')
        for row in rows:
            world = worlds[row['id']]
            if (any(row.get(key) != value for key, value in identity.items())
                    or row.get('world_id') != world['world_id'] or row.get('answer') != world['answer']
                    or type(row.get('correct')) is not bool):
                raise ValueError('Raw prediction differs from its frozen story/checkpoint')
            if fmt != 'immediate':
                ids, text, eos = row.get('completion_ids'), row.get('completion'), row.get('eos_token_id')
                cap = 384 if typed.reasoning else 4
                if (not isinstance(text, str) or not isinstance(ids, list)
                        or any(type(token) is not int or token < 0 for token in ids)
                        or type(eos) is not int or not 0 < len(ids) <= cap
                        or (eos in ids and ids.index(eos) != len(ids) - 1)
                        or row.get('completion_tokens') != len(ids) or row.get('forced_delimiter') is not False):
                    raise ValueError('Native selection requires actual unforced sampled completions')
                outcome = natural_completion_outcome(text, reasoning=typed.reasoning,
                            completion_ids=ids, trace_cap=typed.trace_cap, eos_token_id=eos)
                if any(row.get(key) != value for key, value in outcome.items()):
                    raise ValueError('Native outcome differs from saved completion')
            if row['correct'] != (row.get('prediction') == world['answer']):
                raise ValueError('Native correctness differs from the actual saved prediction')
        count = sum(row['correct'] for row in rows)
        if metric.get('accuracy') != count / 120:
            raise ValueError('Saved metric disagrees with raw predictions')
        history.append({'step': s, 'format': fmt, 'correct': count, 'n': 120, 'accuracy': count / 120})
    if (evidence.get('primary_format') != native or evidence.get('baseline') != by_key[(0, native)]
            or evidence.get('latest') != by_key[(step, native)]
            or evidence.get('immediate_latest') != by_key[(step, 'immediate')]
            or evidence.get('native_accuracy_gain') != by_key[(step, native)]['accuracy'] - by_key[(0, native)]['accuracy']):
        raise ValueError('Learning evidence boundaries differ from actual complete curves')
    receipt = current.get('completion_receipt', {})
    final = output / 'adapter' / 'adapter_model.safetensors'
    if (not final.is_file() or receipt.get('final_adapter_sha256') != file_digest(final)
            or receipt.get('matching_adapter_sha256') != hashes
            or receipt.get('completed_optimizer_steps') != step or receipt.get('observed_worlds') != 2000
            or receipt.get('completion_protocol') != NATURAL_COMPLETION_PROTOCOL
            or receipt.get('trace_health_sha256') != file_digest(paths['trace_health'])):
        raise ValueError('Terminal completion receipt differs from actual retained run artifacts')
    final_row = next(row for row in history if row['step'] == step and row['format'] == native)
    return {**final_row, 'learning_evidence': _ref(paths['learning_evidence']),
            'final_adapter': _ref(final), 'matching_adapter_sha256': hashes,
            'learning_curve': history,
            'source_lineage': 'original_frozen_execution_environment' if recorded_source is None else 'run_and_original_frozen_execution_environment'}


def compute_panel_selection(execution_binding, binding_sha256, panel_status,
                            failure_policy, failure_policy_sha256, *, reviewed_by, rationale):
    """Read-only complete-panel computation. Failed settings are explicit, not zero."""
    _attribution(reviewed_by, 'selection attribution')
    _attribution(rationale, 'selection rationale')
    status_path, policy_path = Path(panel_status).resolve(), Path(failure_policy).resolve()
    if file_digest(policy_path) != failure_policy_sha256:
        raise ValueError('Failure policy hash mismatch')
    policy = _read(policy_path)
    if policy.get('schema_version') != 1 or policy.get('status') != 'frozen' or policy.get('policy') != selection_failure_policy():
        raise ValueError('Require the separately frozen prospective failure policy')
    for key in ('reviewed_by', 'rationale'):
        _attribution(policy.get(key), 'failure policy ' + key)
    policy_time = _utc(policy.get('frozen_utc'), 'Failure policy')
    paths, configs, registry, state, source, frozen, training, development = _source_inputs(
        execution_binding, binding_sha256, status_path)
    if (state.get('status') not in ('completed', *FAILED, 'budget_exhausted')
            or set(state.get('candidates', {})) != set(configs)):
        raise ValueError('All twelve frozen candidates must have terminal attempts before selection')
    events_path = status_path.parent / 'events.jsonl'
    events = read_jsonl(events_path)
    results = {}
    for name, (config_path, config) in sorted(configs.items()):
        current = state['candidates'][name]
        output = (status_path.parent / 'runs' / name).resolve()
        attempt, finished = _attempt(output, name, current, _ref(config_path), events)
        setting = {key: config[key] for key in ('learning_rate', 'lr_scheduler_type', 'warmup_ratio')}
        result = {'candidate': name, 'arm': config['arm'], 'status': current['status'],
                  'config': _ref(config_path), 'setting': setting, **attempt}
        if current['status'] == 'completed':
            result['measurement'] = _completed(output, config, current, source,
                file_digest(paths['corpus_manifest']), training, development)
        else:
            if policy_time >= finished:
                raise ValueError('Failure eligibility policy must precede the observed failure')
            result['measurement'] = None
            result['failure'] = {'status': current['status'], 'exit_code': current['exit_code'],
                'reason': current.get('error') or state.get('error') or
                          f"Supervisor recorded {current['status']} (exit {current['exit_code']})",
                'finished_utc': current['finished_utc'], 'no_score_assigned': True}
        results[name] = result
    candidates = list(results.values())
    settings = {'rl': [], 'sft': []}
    for family, family_arms in (('rl', ('reasoning_rl', 'answer_rl')), ('sft', ('label_sft',))):
        for representative in [row for row in candidates if row['arm'] == family_arms[0]]:
            paired = [row for row in candidates if row['arm'] in family_arms and row['setting'] == representative['setting']]
            if {row['arm'] for row in paired} != set(family_arms):
                raise ValueError('Panel family setting is not paired exactly')
            eligible = all(row['measurement'] is not None for row in paired)
            correct = sum(row['measurement']['correct'] for row in paired) if eligible else None
            total = len(paired) * 120 if eligible else None
            settings[family].append({'setting': representative['setting'],
                'candidates': [row['candidate'] for row in paired], 'eligible': eligible,
                'correct': correct, 'n': total, 'score': correct / total if eligible else None,
                'excluded_candidates': [row['candidate'] for row in paired if row['measurement'] is None]})
    winners = {}
    for family, rows in settings.items():
        eligible = [row for row in rows if row['eligible']]
        if not eligible:
            raise ValueError('No eligible complete setting for family ' + family + '; no selection is possible')
        winners[family] = min(eligible, key=lambda row: (-Fraction(row['correct'], row['n']),
             row['setting']['learning_rate'], row['setting']['warmup_ratio'] != 0, row['setting']['warmup_ratio']))
    selected = {row['arm']: row for row in candidates if row['candidate'] in
                winners['rl']['candidates'] + winners['sft']['candidates']}
    references = {role: _ref(path) for role, path in paths.items()}
    failures = [row['candidate'] for row in candidates if row['measurement'] is None]
    return {'schema_version': 1, 'method': METHOD, 'status': 'frozen',
        'reviewed_by': reviewed_by, 'rationale': rationale,
        'scientific_gate_decision': None, 'confirmation_authorized': False,
        'selection_rule': registry['selection_rule'], 'failure_policy': _ref(policy_path),
        'candidate_execution_binding': _ref(execution_binding), 'panel_status': _ref(status_path),
        'batch_frozen_inputs': _ref(frozen), 'panel_events': _ref(events_path), **references,
        'registry_sha256': references['registry']['sha256'], 'panel_status_sha256': file_digest(status_path),
        'corpus_manifest_sha256': references['corpus_manifest']['sha256'],
        'source_commit': source['commit'], 'source_tree': source['tree'],
        'candidates': candidates, 'family_settings': settings,
        'failures': failures, 'excluded_settings': {family: [row for row in rows if not row['eligible']]
                                                   for family, rows in settings.items()},
        'warnings': ([f'{len(failures)} failed candidate(s) retained; their settings were excluded without invented zero scores']
                     if failures else []),
        'selected_settings': {family: winner['setting'] for family, winner in winners.items()},
        **{'selected_' + arm + '_candidate': row['candidate'] for arm, row in selected.items()},
        'selected_run_evidence': {arm: row['measurement']['learning_evidence'] for arm, row in selected.items()},
        'selected_configs': {arm: row['config'] for arm, row in selected.items()},
        'interpretation': 'Deterministic final-native family selection only; no trace-health threshold, editor result, confirmation outcome or scientific gate approval used.'}


def freeze_panel_selection(execution_binding, binding_sha256, panel_status,
                           failure_policy, failure_policy_sha256, output, *, reviewed_by, rationale):
    """Write exactly one reviewable receipt, never replace a prior selection."""
    output = Path(output)
    if output.exists():
        raise FileExistsError('Selection receipt already exists; no overwrite')
    receipt = compute_panel_selection(execution_binding, binding_sha256, panel_status,
        failure_policy, failure_policy_sha256, reviewed_by=reviewed_by, rationale=rationale)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    return receipt


def verify_panel_selection(receipt_path, expected_sha256=None):
    """Recompute the selection; a hand-authored winner is never authority."""
    receipt_path = Path(receipt_path).resolve()
    if expected_sha256 is not None and file_digest(receipt_path) != expected_sha256:
        raise ValueError('Selection receipt hash mismatch')
    receipt = _read(receipt_path)
    if receipt.get('method') != METHOD or receipt.get('status') != 'frozen':
        raise ValueError('Require a computed amended panel selection receipt')
    inputs = {role: _bound(receipt.get(role), receipt_path.parent, role) for role in
              ('candidate_execution_binding', 'panel_status', 'failure_policy')}
    actual = compute_panel_selection(inputs['candidate_execution_binding'],
        receipt['candidate_execution_binding']['sha256'], inputs['panel_status'],
        inputs['failure_policy'], receipt['failure_policy']['sha256'],
        reviewed_by=receipt.get('reviewed_by'), rationale=receipt.get('rationale'))
    if actual != receipt:
        raise ValueError('Selection receipt disagrees with recomputed actual panel outcomes or lineage')
    return actual
