#!/usr/bin/env python3
"""Resumable, development-only base-model capability measurements; never approves a gate."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import re
import time

from paper2.analysis import accuracy_summary
from paper2.backend import HFBackend
from paper2.data import VIEWS
from paper2.data.corpus import CORPUS_SCHEMA
from paper2.evaluation import evaluate_immediate
from paper2.io import digest, file_digest, read_jsonl


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def development_panel(corpus):
    """Only open the manifest and declared clean development input, never confirmation."""
    manifest_path = Path(corpus) / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('schema_version') != CORPUS_SCHEMA or not manifest.get('integrity_passed'):
        raise ValueError('A structurally verified frozen corpus is required')
    identity = {key: manifest[key] for key in ('schema_version', 'config', 'files', 'shortcut_audit')}
    if digest(identity) != manifest.get('corpus_hash'):
        raise ValueError('Corpus manifest identity mismatch')
    name = 'evaluation/development__clean.jsonl'
    path = Path(corpus) / name
    if file_digest(path) != manifest['files'][name]['sha256']:
        raise ValueError('Frozen development file hash mismatch')
    rows = read_jsonl(path)
    groups = defaultdict(list)
    for row in rows:
        if row['split'] != 'development' or row['cue_condition'] != 'clean':
            raise ValueError('Only clean development stories are permitted')
        groups[row['world_id']].append(row)
    if manifest['config']['development'] != 120 or len(groups) != 120 or len(rows) != 960:
        raise ValueError('The screen requires exactly 120 development stories and 960 views')
    for world in groups.values():
        if len(world) != 8 or {row['view'] for row in world} != set(VIEWS):
            raise ValueError('Every development story requires exactly all eight views')
    return groups, {'corpus_hash': manifest['corpus_hash'],
                    'manifest_sha256': file_digest(manifest_path), 'development_sha256': file_digest(path)}


class MeteredBackend:
    def __init__(self, backend):
        self.backend = backend
        self.metrics = {'generation_calls': 0, 'generated_tokens': 0,
                        'generation_input_tokens': 0, 'answer_forward_calls': 0, 'answer_input_tokens': 0}

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def answer(self, prompt, choices, prefix_ids=None):
        self.metrics['answer_forward_calls'] += 1
        self.metrics['answer_input_tokens'] += len(prefix_ids if prefix_ids is not None else self.token_ids(prompt))
        return self.backend.answer(prompt, choices, prefix_ids=prefix_ids)

    def generate(self, prompt, **kwargs):
        result = self.backend.generate(prompt, **kwargs)
        self.metrics['generation_calls'] += 1
        self.metrics['generated_tokens'] += len(result['generated_ids'])
        self.metrics['generation_input_tokens'] += len(result['prompt_ids'])
        return result


def evaluate_native_capability(backend, row):
    """Use the identical uninterrupted generation and reward rules used for G1.

    Trace-reader replay generation is deliberately separate: it can re-render an
    unfinished trace and inject a delimiter after early EOS. That is useful for
    Test 2 controls but is not an ordinary native policy completion.
    """
    from paper2.prompts import ANSWER_DELIMITER, render_prompt
    from paper2.training.config import TrainingConfig
    from paper2.training.data import answer_reward
    from paper2.training.runner import _development_predictions

    config = TrainingConfig(arm='reasoning_rl')
    prompt = render_prompt(backend.tokenizer, row, reasoning=True)
    native = _development_predictions(backend.model, backend.tokenizer, [row], config,
                                     backend.torch, formats=('native',))[0]
    # The helper directly invokes model.generate once, bypassing backend.generate.
    backend.metrics['generation_calls'] += 1
    backend.metrics['generated_tokens'] += native['completion_tokens']
    backend.metrics['generation_input_tokens'] += len(backend.token_ids(prompt))
    parsed = native['prediction']
    valid = parsed is not None and bool(answer_reward(
        completions=[native['completion']], answer=[parsed], reasoning=True,
        tokenizer=backend.tokenizer, trace_cap=config.trace_cap)[0])
    trace = native['completion'].rsplit(ANSWER_DELIMITER, 1)[0]
    return {**row, **native, 'reader': backend.checkpoint_id,
            'parsed_prediction': parsed, 'prediction': parsed if valid else None,
            'completion_valid': valid,
            'trace': trace, 'trace_tokens': len(backend.token_ids(trace)),
            'native_procedure': 'uninterrupted_g1_generation_and_reward_v1'}


def summarize(records):
    rows = [row for record in records for row in record['predictions']]
    summaries = {}
    for format_name in ('immediate', 'native_reasoning'):
        panel = [row for row in rows if row['format'] == format_name]
        summaries[format_name] = {'overall': accuracy_summary(panel), 'strata': {}}
        for depth in (1, 2):
            for kind in ('move', 'tell'):
                selected = [row for row in panel if row['belief_depth'] == depth and row['critical_event_kind'] == kind]
                summaries[format_name]['strata'][f'depth{depth}_{kind}'] = accuracy_summary(selected)
        summaries[format_name]['truncated'] = sum(bool(row.get('truncated')) for row in panel)
        summaries[format_name]['forced_delimiters'] = sum(bool(row.get('forced_delimiter')) for row in panel)
        if format_name == 'native_reasoning':
            parsed_panel = [{**row, 'prediction': row.get('parsed_prediction', row['prediction'])} for row in panel]
            summaries[format_name]['parsed_answer_accuracy'] = accuracy_summary(parsed_panel)
            summaries[format_name]['parsed_correct_but_reward_invalid'] = sum(
                row.get('parsed_prediction') == row['answer'] and row['prediction'] != row['answer'] for row in panel)
    return summaries


def peak_memory(backend):
    torch = getattr(backend, 'torch', None)
    if torch is None or not torch.cuda.is_available():
        return {}
    return {str(device): {'allocated_bytes': torch.cuda.max_memory_allocated(device),
                          'reserved_bytes': torch.cuda.max_memory_reserved(device)}
            for device in range(torch.cuda.device_count())}


def run_screen(corpus, output, model, revision, device='auto', dtype='bfloat16', *, backend_factory=HFBackend):
    if not re.fullmatch('[0-9a-f]{40}', revision):
        raise ValueError('An immutable 40-character model revision is required')
    groups, corpus_identity = development_panel(corpus)
    identity = {'schema_version': 1, 'purpose': 'development_capability_only', **corpus_identity,
                'model': model, 'revision': revision, 'device': device, 'dtype': dtype,
                'trace_cap': 384, 'immediate_views': list(VIEWS), 'native_views': ['direct'],
                'native_procedure': 'uninterrupted_g1_generation_and_reward_v1',
                'bootstrap_samples': 20000, 'bootstrap_seed': 2026,
                'script_sha256': file_digest(__file__),
                'implementation_sha256': {name: file_digest(Path(__file__).resolve().parents[1] / 'src/paper2' / name)
                    for name in ('backend.py', 'evaluation.py', 'prompts.py', 'traces.py', 'analysis.py',
                                 'training/data.py', 'training/generation.py', 'training/runner.py', 'training/config.py')}}
    identity_hash = digest(identity)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    protocol_path, rows_path = output / 'protocol.json', output / 'worlds.jsonl'
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != identity:
            raise ValueError('Existing screen has a different immutable protocol; choose a new output directory')
    elif any(output.iterdir()):
        raise ValueError('Refusing to overwrite a nonempty output directory without its protocol')
    else:
        atomic_json(protocol_path, identity)
    # Never silently discard a damaged or duplicate record. Each line is an atomic completed world.
    records = read_jsonl(rows_path) if rows_path.exists() else []
    completed = set()
    for record in records:
        world_id = record['world_id']
        if world_id in completed or world_id not in groups or record.get('protocol_sha256') != identity_hash:
            raise ValueError('Saved screen has duplicate/foreign worlds or a changed protocol')
        expected = {(row['id'], 'immediate') for row in groups[world_id]}
        expected.add((next(row['id'] for row in groups[world_id] if row['view'] == 'direct'), 'native_reasoning'))
        observed = [(row['id'], row['format']) for row in record['predictions']]
        if len(observed) != 9 or set(observed) != expected or record.get('record_sha256') != digest({k: v for k, v in record.items() if k != 'record_sha256'}):
            raise ValueError('Saved world is incomplete or its content changed')
        completed.add(world_id)
    receipt_path = output / 'receipt.json'
    prior_receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    sessions = prior_receipt.get('sessions', [])
    backend = None
    if len(completed) < 120:
        session = {'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'resumed_worlds': len(completed)}
        sessions.append(session)
        started = time.monotonic()
        try:
            backend = MeteredBackend(backend_factory(model, revision=revision, device=device, dtype=dtype))
            session['model_load_seconds'] = time.monotonic() - started
            with rows_path.open('a', encoding='utf-8') as handle:
                for world_id, world in groups.items():
                    if world_id in completed:
                        continue
                    world_start = time.monotonic()
                    before = dict(backend.metrics)
                    predictions = []
                    for row in world:
                        tick = time.monotonic()
                        result = next(evaluate_immediate(backend, [row]))
                        result['wall_seconds'] = time.monotonic() - tick
                        predictions.append(result)
                    direct = next(row for row in world if row['view'] == 'direct')
                    tick = time.monotonic()
                    result = evaluate_native_capability(backend, direct)
                    result['wall_seconds'] = time.monotonic() - tick
                    predictions.append(result)
                    record = {'world_id': world_id, 'protocol_sha256': identity_hash,
                              'predictions': predictions, 'wall_seconds': time.monotonic() - world_start,
                              'metrics': {key: value - before[key] for key, value in backend.metrics.items()}}
                    record['record_sha256'] = digest(record)
                    handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + '\n')
                    handle.flush()
                    os.fsync(handle.fileno())
                    records.append(record)
                    completed.add(world_id)
                    print(f'{model}: {len(completed)}/120 development worlds saved', flush=True)
            session['status'] = 'completed'
        except BaseException as error:
            session['status'] = 'interrupted'
            session['error_type'] = type(error).__name__
            raise
        finally:
            session['wall_seconds'] = time.monotonic() - started
            session['peak_cuda_memory'] = peak_memory(backend) if backend is not None else {}
            session['metrics_including_unfinished_world'] = dict(backend.metrics) if backend is not None else {}
            atomic_json(receipt_path, {'protocol_sha256': identity_hash, 'completed_worlds': len(completed),
                                      'status': 'completed' if len(completed) == 120 else 'incomplete',
                                      'sessions': sessions, 'gate_decision': None,
                                      'worlds_sha256': file_digest(rows_path) if rows_path.exists() else None})
    summary = {'protocol_sha256': identity_hash, 'completed_worlds': len(records), 'summaries': summarize(records),
               'worlds_sha256': file_digest(rows_path), 'gate_decision': None,
               'metrics': {key: sum(record['metrics'][key] for record in records) for key in records[0]['metrics']},
               'world_evaluation_seconds': sum(record['wall_seconds'] for record in records),
               'note': 'Base model, clean development only. No training, confirmation evaluation or gate approval. '
                       'Summary token totals cover retained complete worlds; receipt sessions also include work '
                       'on interrupted worlds and model-loading wall time.'}
    atomic_json(output / 'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--dtype', default='bfloat16')
    args = parser.parse_args()
    run_screen(**vars(args))


if __name__ == '__main__':
    main()
