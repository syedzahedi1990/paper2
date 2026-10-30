"""Exclusive, additive length reanalysis of saved native predictions.

This module never generates text, edits a raw log, rewrites a v1 summary, or
selects a checkpoint. The CLI can call ``reanalyze_trace_health`` directly.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from .trace_health import _seed, summarize_trace_health, summarize_trace_health_v2


def _load_jsonl(path: Path):
    data = path.read_bytes()
    rows = [json.loads(line) for line in data.decode('utf-8').splitlines() if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError('Prediction and health JSONL records must be objects')
    return data, rows


def _checkpoint_key(row):
    return tuple(row.get(key) for key in ('arm', 'dataset', 'model', 'revision', 'step', 'checkpoint_id')) + (_seed(row),)


def _validate_original_health(predictions, health):
    grouped = {}
    for row in predictions:
        grouped.setdefault(_checkpoint_key(row), []).append(row)
    observed = set()
    for row in health:
        key = _checkpoint_key(row)
        if key in observed or key not in grouped:
            raise ValueError('Original trace-health checkpoints must match raw predictions exactly once')
        observed.add(key)
        original = row.get('trace_health')
        if not isinstance(original, dict) or original.get('schema_version') != 1:
            raise ValueError('Original trace-health artifact must contain the frozen v1 schema')
        native = [item for item in grouped[key] if item.get('format') == 'native_reasoning']
        calculated = summarize_trace_health(native, completion_protocol=original.get('completion_protocol'),
                                            cap=original.get('token_budget'),
                                            seed=original.get('bootstrap_seed'),
                                            samples=original.get('bootstrap_samples'))
        if original != calculated:
            raise ValueError('Original v1 trace-health summary disagrees with raw predictions')
        if type(row.get('native_reasoning_assessed')) is not bool or row['native_reasoning_assessed'] != bool(native):
            raise ValueError('Original v1 assessment flag disagrees with raw predictions')
    if observed != set(grouped):
        raise ValueError('Original trace-health artifact must cover every supplied raw checkpoint')
    return {'status': 'verified_exact_v1_recomputation', 'checkpoints': len(observed)}


def reanalyze_trace_health(predictions_path: str | Path, output_path: str | Path, *,
                           original_health_path: str | Path | None = None,
                           tokenizer=None, tokenizer_provenance: dict[str, Any] | None = None,
                           seed: int = 2026, samples: int = 20_000) -> dict[str, Any]:
    """Write one new hash-bound report; all input artifacts remain untouched.

    When an original health log is supplied, every saved v1 summary is checked
    using its original bootstrap settings. New diagnostic settings apply only
    to the derived v2 report. Tokenizers are supplied by the caller; this helper
    performs no downloads or model loading.
    """
    source = Path(predictions_path).resolve()
    output = Path(output_path).resolve()
    original = Path(original_health_path).resolve() if original_health_path is not None else None
    if output in {source, original} or output.exists():
        raise FileExistsError('Derived output must be a new path, separate from all original artifacts')
    raw, rows = _load_jsonl(source)
    inputs = {'predictions': {'path': str(source), 'sha256': sha256(raw).hexdigest(), 'bytes': len(raw)}}
    stable_inputs = [(source, raw)]
    verification = {'status': 'original_v1_artifact_not_supplied'}
    if original is not None:
        old_raw, old_rows = _load_jsonl(original)
        verification = _validate_original_health(rows, old_rows)
        inputs['original_trace_health'] = {'path': str(original), 'sha256': sha256(old_raw).hexdigest(), 'bytes': len(old_raw)}
        stable_inputs.append((original, old_raw))
    report = {
        'schema_version': 1, 'artifact_type': 'additive_trace_health_length_reanalysis',
        'inputs': inputs, 'original_health_validation': verification,
        'trace_health_v2': summarize_trace_health_v2(rows, tokenizer=tokenizer,
            tokenizer_provenance=tokenizer_provenance, seed=seed, samples=samples),
        'implementation': {path.name: sha256(path.read_bytes()).hexdigest() for path in (
            Path(__file__), Path(__file__).with_name('trace_health.py'))},
        'preservation': 'Separate derived report. Original raw predictions, token IDs, v1 summaries, rewards and checkpoint selection are unchanged.',
    }
    for path, contents in stable_inputs:
        if path.read_bytes() != contents:
            raise ValueError('Source artifact changed during reanalysis; take a stable snapshot first')
    with output.open('x', encoding='utf-8') as handle:
        handle.write(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n')
    return report
