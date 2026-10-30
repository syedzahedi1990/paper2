"""Synthetic full-panel artifacts for CPU integrity tests, never measurements."""
from copy import deepcopy
import json
from pathlib import Path
import random

from paper2.data.world import generate_world, render_record
from paper2.io import digest, file_digest, write_json, write_jsonl
from paper2.training.config import TrainingConfig, NATURAL_COMPLETION_PROTOCOL
from paper2.training.data import natural_completion_outcome, rows_digest
from paper2.training.selection import (development_selection_panel, development_selection_spec,
                                       selection_failure_policy)


def ref(path):
    return {'path': str(Path(path).resolve()), 'sha256': file_digest(path)}


def write(path, value):
    write_json(path, value)
    return path


def refresh_candidate(fixture, name):
    """Rebind a deliberate test mutation as if recorded by the supervisor."""
    output = fixture['root'] / 'runs' / name
    state = json.loads(fixture['status'].read_text(encoding='utf-8'))
    current = state['candidates'][name]
    evidence_path = output / 'g1_learning_evidence.json'
    if evidence_path.exists():
        evidence = json.loads(evidence_path.read_text(encoding='utf-8'))
        for role, item in evidence.get('artifacts', {}).items():
            artifact = output / Path(item['path']).name
            evidence['artifacts'][role] = {'path': artifact.name, 'sha256': file_digest(artifact)}
        write(evidence_path, evidence)
    for filename in current['artifacts']:
        path = output / filename
        current['artifacts'][filename] = {**ref(path), 'bytes': path.stat().st_size}
    write(fixture['status'], state)


def make_panel_fixture(root, *, scores=None, failures=(), empty_reasoning=False):
    """Create twelve distinct valid run/config/artifact sets with all schedules.

    Scores map candidate stem to final native correct count. Earlier native
    checkpoints deliberately have 120/120, preventing a best-checkpoint bug from
    accidentally passing the final-step selection tests. Immediate scores are
    120/120 and therefore cannot decide the family objective either.
    """
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    configs_root = root / 'inputs'
    configs_root.mkdir()
    training = [{'id': f'train-{n}', 'world_id': f'train-{n}', 'view': 'direct', 'split': 'train',
                 'answer': 'A', 'dataset': 'clean'} for n in range(2000)]
    development = [render_record(generate_world(random.Random(100000 + n)), 'direct') for n in range(120)]
    training_path, dev_path = configs_root / 'training.jsonl', configs_root / 'development.jsonl'
    write_jsonl(training_path, training)
    write_jsonl(dev_path, development)
    corpus = write(configs_root / 'corpus.json', {'config': {'train': 2000, 'development': 120, 'confirmation': 120},
        'files': {'training/clean.jsonl': ref(training_path), 'evaluation/development__clean.jsonl': ref(dev_path)}})
    source = {'commit': 'a' * 40, 'tree': 'b' * 40, 'tracked_files_sha256': {'src/paper2/cli.py': 'c' * 64}}
    environment = write(configs_root / 'environment.json', {'source': source, 'wrapper': {'sha256': 'd' * 64}})
    decision = write(configs_root / 'decision.json', {'source_commit': source['commit'],
        'execution_environment_sha256': file_digest(environment), 'save_total_limit': 2})
    configs = development_selection_panel()
    config_paths = {Path(name).stem: write(configs_root / name, config) for name, config in configs.items()}
    entries = [{'config_path': name, 'config_sha256': file_digest(config_paths[Path(name).stem]),
                **{key: config[key] for key in ('arm', 'learning_rate', 'lr_scheduler_type', 'warmup_ratio', 'completion_protocol', 'beta')}}
               for name, config in configs.items()]
    registry_doc = {'schema_version': 2, 'panel': 'four_rates_constant', 'candidate_runs': 12,
        'selection_rule': development_selection_spec()['selection'], 'decision_sha256': file_digest(decision),
        'completion_protocol': NATURAL_COMPLETION_PROTOCOL,
        'beta_by_arm': {'reasoning_rl': .04, 'answer_rl': .04, 'label_sft': 0.}, 'candidates': entries}
    registry_doc['plan_sha256'] = digest(registry_doc)
    registry = write(configs_root / 'registry.json', registry_doc)
    inputs = {'training': training_path, 'development': dev_path, 'corpus_manifest': corpus,
              'execution_environment': environment, 'decision': decision, 'registry': registry}
    binding = write(configs_root / 'execution-binding.json', {'schema_version': 2,
        'purpose': 'amended_pilot_candidate_execution', 'completion_protocol': NATURAL_COMPLETION_PROTOCOL,
        'beta_by_arm': registry_doc['beta_by_arm'],
        'config_sha256': {item['config_path']: item['config_sha256'] for item in entries},
        **{role: ref(path) for role, path in inputs.items()}})
    frozen = write(root / 'frozen-inputs.json', {'binding_sha256': file_digest(binding),
        'wrapper_sha256': 'd' * 64, 'files_sha256': {str(path): file_digest(path) for path in
            (binding, *inputs.values(), *config_paths.values())}})
    policy = write(configs_root / 'failure-policy.json', {'schema_version': 1, 'status': 'frozen',
        'frozen_utc': '2026-10-30T00:00:00+00:00', 'reviewed_by': 'Synthetic fixture, no scientific approval',
        'rationale': 'CPU integrity fixture only', 'policy': selection_failure_policy()})
    state = {'schema_version': 1, 'status': 'failed' if failures else 'completed',
             'binding_sha256': file_digest(binding), 'candidates': {}}
    events = []
    for index, (filename, config) in enumerate(configs.items()):
        name = Path(filename).stem
        output = root / 'runs' / name
        output.mkdir(parents=True)
        (root / (name + '.launch.log')).write_text('Synthetic process log, no GPU run.\n', encoding='utf-8')
        current = {'status': 'failed' if name in failures else 'completed', 'exit_code': 1 if name in failures else 0,
            'pid': index + 100, 'started_utc': '2026-10-30T01:00:00+00:00', 'finished_utc': '2026-10-30T02:00:00+00:00',
            'config_sha256': file_digest(config_paths[name]),
            'command': ['python', '-m', 'paper2', 'train', '--config', str(config_paths[name]), '--output', str(output)]}
        events += [{'event': 'candidate_started', 'candidate': name},
                   {'event': 'candidate_process_started', 'candidate': name, 'pid': current['pid']},
                   {'event': 'candidate_failed_batch_stopped' if name in failures else 'candidate_completed', 'candidate': name}]
        typed = TrainingConfig(**config)
        budget = typed.training_budget(2000)
        final_step = budget['resolved_max_steps']
        manifest = {'status': 'failed' if name in failures else 'complete', 'config': config,
            'synthetic_integration_test': False,  # Exercises production validation against toy files; no real result claimed.
            'resume_from_checkpoint': None, 'training_budget': budget, 'training_examples': 2000,
            'training_sha256': rows_digest(training), 'corpus_manifest_sha256': file_digest(corpus)}
        write(output / 'run_manifest.json', manifest)
        if name in failures:
            current['error'] = 'Synthetic crash retained in test fixture'
            current['artifacts'] = {'run_manifest.json': {**ref(output / 'run_manifest.json'),
                'bytes': (output / 'run_manifest.json').stat().st_size}}
            state['candidates'][name] = current
            continue
        compute = {'completed_optimizer_steps': final_step, 'training_examples': 2000,
            'prompt_exposures': 2000, 'observed_world_ids': [row['world_id'] for row in training]}
        write(output / 'compute_report.json', compute)
        native = 'native_reasoning' if typed.reasoning else 'native_answer'
        schedule = sorted({(step, native) for step in typed.evaluation_schedule(final_step)} |
                          {(step, 'immediate') for step in typed.immediate_evaluation_schedule(final_step)})
        metrics, predictions, matching = [], [], {}
        for step, fmt in schedule:
            checkpoint = output / 'evaluated_checkpoints' / f'checkpoint-{step}'
            checkpoint.mkdir(parents=True, exist_ok=True)
            adapter = checkpoint / 'adapter_model.safetensors'
            adapter.write_bytes(f'Synthetic adapter {name} {step}'.encode())
            matching[str(adapter)] = file_digest(adapter)
            identity = {'arm': config['arm'], 'dataset': 'clean', 'evaluation_dataset': 'clean', 'seed': 1,
                'model': config['model'], 'revision': config['revision'], 'view': 'direct', 'split': 'development',
                'step': step, 'format': fmt, 'completion_protocol': NATURAL_COMPLETION_PROTOCOL,
                'checkpoint_id': str(checkpoint), 'path': str(checkpoint), 'checkpoint_path': str(checkpoint),
                'checkpoint_kind': 'base' if step == 0 else 'trained', 'adapter_sha256': matching[str(adapter)]}
            count = (scores or {}).get(name, 60) if step == final_step and fmt == native else 120
            metrics.append({**identity, 'n': 120, 'accuracy': count / 120, 'decoding': 'greedy_full_vocabulary',
                            'prompt_exposures': step if typed.rl else step * 8, 'unique_worlds_seen': min(2000, step if typed.rl else step * 8)})
            for number, world in enumerate(development):
                prediction = world['answer'] if number < count else ('B' if world['answer'] == 'A' else 'A')
                row = {**identity, 'id': world['id'], 'world_id': world['world_id'], 'answer': world['answer'],
                       'prediction': prediction, 'correct': prediction == world['answer']}
                if fmt != 'immediate':
                    text = ('Answer: ' + prediction if empty_reasoning else 'A thought.\nAnswer: ' + prediction) if typed.reasoning else prediction
                    ids = [4, 2]
                    outcome = natural_completion_outcome(text, reasoning=typed.reasoning, completion_ids=ids, eos_token_id=2)
                    row.update(outcome)
                    row.update(completion=text, completion_ids=ids, completion_tokens=len(ids), eos_token_id=2,
                               forced_delimiter=False, truncated=False)
                else:
                    row['token_id'] = 4
                predictions.append(row)
        write_jsonl(output / 'checkpoint_evaluations.jsonl', metrics)
        write_jsonl(output / 'checkpoint_predictions.jsonl', predictions)
        write_jsonl(output / 'checkpoint_trace_health.jsonl', [{'synthetic_fixture': True}])
        write_jsonl(output / 'learning_curve.jsonl', [{'synthetic_fixture': True}])
        roles = {'manifest': 'run_manifest.json', 'compute': 'compute_report.json',
                 'evaluations': 'checkpoint_evaluations.jsonl', 'predictions': 'checkpoint_predictions.jsonl',
                 'trace_health': 'checkpoint_trace_health.jsonl'}
        by_key = {(row['step'], row['format']): row for row in metrics}
        evidence = {'schema_version': 3, 'run_status': 'complete', 'synthetic': False,
            'completion_protocol': NATURAL_COMPLETION_PROTOCOL, 'primary_format': native,
            'run_identity': {key: config[key] for key in ('arm', 'dataset', 'seed', 'model', 'revision', 'stage')},
            'baseline': by_key[(0, native)], 'latest': by_key[(final_step, native)],
            'immediate_latest': by_key[(final_step, 'immediate')],
            'native_accuracy_gain': by_key[(final_step, native)]['accuracy'] - by_key[(0, native)]['accuracy'],
            'artifacts': {role: {'path': filename, 'sha256': file_digest(output / filename)} for role, filename in roles.items()}}
        write(output / 'g1_learning_evidence.json', evidence)
        final = output / 'adapter' / 'adapter_model.safetensors'
        final.parent.mkdir()
        final.write_bytes(b'Synthetic final adapter')
        current['completion_receipt'] = {'completed_optimizer_steps': final_step, 'observed_worlds': 2000,
            'completion_protocol': NATURAL_COMPLETION_PROTOCOL, 'final_adapter_sha256': file_digest(final),
            'matching_adapter_sha256': matching, 'trace_health_sha256': file_digest(output / 'checkpoint_trace_health.jsonl')}
        current['artifacts'] = {filename: {**ref(output / filename), 'bytes': (output / filename).stat().st_size}
            for filename in (*roles.values(), 'g1_learning_evidence.json', 'learning_curve.jsonl')}
        state['candidates'][name] = current
    status = write(root / 'status.json', state)
    events_path = root / 'events.jsonl'
    write_jsonl(events_path, events)
    return {'root': root, 'binding': binding, 'status': status, 'policy': policy, 'configs': config_paths,
            'registry': registry, 'corpus': corpus, 'development': dev_path, 'training': training_path,
            'environment': environment, 'decision': decision, 'frozen': frozen,
            'events': events_path, 'config_values': deepcopy(configs)}
