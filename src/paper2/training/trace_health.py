"""Descriptive checkpoint trace health; no empirical acceptance thresholds.

The revised protocol has a 384-sampled-token *complete response* budget. A
natural terminal answer at the last token is complete even without EOS. These
diagnostics do not change rewards or decide scientific gates.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import re
from typing import Any, Iterable

import numpy as np

from paper2.analysis import EDIT_STATUSES, _ratio_interval, cluster_interval
from .config import NATURAL_COMPLETION_PROTOCOL
from .data import natural_completion_outcome


PANEL_FIELDS = ('arm', 'dataset', 'model', 'revision', 'step')
UNCERTAINTY = (
    '95% whole-story bootstrap; observations averaged within each fixed training seed, '
    'then seeds averaged within each story before resampling stories.'
)


def _seed(row):
    return row.get('comparison_seed', row.get('training_seed', row.get('seed')))


def _identity(row):
    return (*[row.get(key) for key in PANEL_FIELDS], _seed(row), row['world_id'],
            row.get('id', row.get('record_id')))


def _mean_interval(records, key, seed, samples):
    groups = defaultdict(lambda: defaultdict(list))
    for row in records:
        if row[key] is not None:
            groups[row['world_id']][row['seed']].append(float(row[key]))
    values = {world: [float(np.mean(items)) for items in seeds.values()]
              for world, seeds in groups.items()}
    observations = [row[key] for row in records if row[key] is not None]
    return {**cluster_interval(values, seed, samples), 'observations': len(observations),
            'missing_observations': len(records) - len(observations),
            'raw_sum': float(sum(observations)) if observations else None}


def _exact_common_trace(records, seed, samples):
    """Recompute the modal exact trace share in every whole-story resample."""
    traces = sorted({row['trace'] for row in records})
    groups = defaultdict(lambda: defaultdict(list))
    for row in records:
        groups[row['world_id']][row['seed']].append(row['trace'])
    trace_index = {trace: index for index, trace in enumerate(traces)}
    story_weights = np.zeros((len(groups), len(traces)))
    for index, (_, seeds) in enumerate(sorted(groups.items())):
        for items in seeds.values():
            for trace in items:
                story_weights[index, trace_index[trace]] += 1 / (len(seeds) * len(items))
    weights = story_weights.mean(axis=0)
    common = traces[int(weights.argmax())]
    ci = None
    if len(groups) > 1:
        rng = np.random.default_rng(seed)
        # Counts are exactly whole-story resamples, batched to bound memory.
        boot = np.empty(samples)
        for start in range(0, samples, 256):
            end = min(start + 256, samples)
            counts = rng.multinomial(len(groups), [1 / len(groups)] * len(groups), size=end - start)
            boot[start:end] = (counts @ story_weights / len(groups)).max(axis=1)
        ci = np.quantile(boot, [.025, .975]).tolist()
    return {'estimate': float(weights.max()), 'ci95': ci, 'stories': len(groups),
            'observations': len(records), 'distinct_exact_traces': len(traces),
            'observed_modal_trace_sha256': sha256(common.encode('utf-8')).hexdigest(),
            'observed_modal_trace_occurrences': sum(row['trace'] == common for row in records),
            'observed_modal_trace_ties': int(np.count_nonzero(np.isclose(weights, weights.max()))),
            'definition': 'Largest exact trace-text share after seed-within-story averaging; the mode is recomputed in each story resample. No text normalization.'}


def _editability(records, seed, samples):
    counts = Counter(row['edit_status'] for row in records)
    rates = {}
    for status in EDIT_STATUSES:
        values = [{**row, 'status_indicator': row['edit_status'] == status} for row in records]
        rates[status] = _mean_interval(values, 'status_indicator', seed, samples)
    groups = defaultdict(lambda: defaultdict(list))
    for row in records:
        groups[row['world_id']][row['seed']].append(
            (int(row['edit_status'] == 'editable'), int(row['edit_status'] != 'not_assessed')))
    pairs = {world: np.mean([np.mean(items, axis=0) for items in seeds.values()], axis=0).tolist()
             for world, seeds in groups.items()}
    assessed = len(records) - counts['not_assessed']
    return {
        'status': 'not_assessed' if not assessed else ('partially_assessed' if counts['not_assessed'] else 'assessed'),
        'records': len(records), 'editor_assessed_records': assessed,
        'counts': {status: counts[status] for status in EDIT_STATUSES}, 'status_rates': rates,
        'editable_among_editor_assessed': _ratio_interval(pairs, seed, samples),
        'definition': 'Only supplied fixed-editor outcomes count as assessments. Missing outcomes are not_assessed, never inferred from wording or correctness. The conditional denominator includes editor-reported failures, excludes not_assessed, and averages seeds within story.',
        'human_audit': 'not_evaluated_by_this_helper',
    }


def summarize_trace_health(rows: Iterable[dict[str, Any]], *,
                           completion_protocol: str = NATURAL_COMPLETION_PROTOCOL,
                           cap: int = 384, editor_statuses: Iterable[dict[str, Any]] | None = None,
                           seed: int = 2026, samples: int = 20_000) -> dict[str, Any]:
    """Summarize raw revised native reasoning records, separately by arm/step.

    Rows use ``completion``/``completion_ids`` (runner) or
    ``completion_text``/``completion_generated_ids`` (native evaluation). IDs
    must be the unpadded sampled sequence, including a naturally sampled EOS.
    Immediate-only checkpoints receive ``status=not_assessed``; no generations
    are performed and no previous checkpoint measurements are carried forward.

    Optional editor outcomes are records with the same panel identity, seed,
    world_id and id (or record_id), plus ``edit_status`` (or ``status``). Retain
    their frozen editor provenance in the caller's artifact. This helper neither
    runs an editor nor represents its output as a human audit.
    """
    if completion_protocol != NATURAL_COMPLETION_PROTOCOL:
        raise ValueError('Trace-health v1 applies only to natural_answer_cap_v2; do not reinterpret legacy traces')
    if cap != 384:
        raise ValueError('Revised complete-response budget is fixed at 384 sampled tokens')
    if not isinstance(samples, int) or samples < 1:
        raise ValueError('Bootstrap samples must be a positive integer')
    rows = list(rows)
    native = [row for row in rows if row.get('format') == 'native_reasoning']
    index = {}
    panels = defaultdict(list)
    for row in native:
        if row.get('completion_protocol') != completion_protocol:
            raise ValueError('Native trace row must explicitly bind the revised completion protocol')
        if row.get('forced_delimiter'):
            raise ValueError('Revised natural trace cannot contain an injected answer delimiter')
        if not row.get('world_id') or row.get('id', row.get('record_id')) is None:
            raise ValueError('Trace health requires story and record identities')
        identity = _identity(row)
        if identity in index:
            raise ValueError('Duplicate native trace identity; select one checkpoint per arm/step/seed')
        index[identity] = row
        panels[identity[:len(PANEL_FIELDS)]].append(row)
    edits = {}
    for record in editor_statuses or []:
        identity = _identity(record)
        if identity not in index or identity in edits:
            raise ValueError('Editor status must match exactly one selected native trace')
        status = record.get('edit_status', record.get('status'))
        if status not in EDIT_STATUSES:
            raise ValueError('Unknown editor editability status')
        edits[identity] = status
    summaries = []
    for identity, panel in sorted(panels.items(), key=lambda item: repr(item[0])):
        records = []
        coverage = defaultdict(set)
        for row in panel:
            text = row.get('completion', row.get('completion_text'))
            ids = row.get('completion_ids', row.get('completion_generated_ids'))
            if not isinstance(text, str) or not isinstance(ids, (list, tuple)):
                raise ValueError('Trace health requires raw completion text and sampled token IDs')
            if any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in ids):
                raise ValueError('Sampled token IDs must be nonnegative integers')
            if 'completion_tokens' in row and row['completion_tokens'] != len(ids):
                raise ValueError('Completion length disagrees with sampled token IDs')
            outcome = natural_completion_outcome(text, reasoning=True, completion_ids=ids,
                                                 trace_cap=cap, eos_token_id=row.get('eos_token_id'))
            for name in ('cap_reached', 'cap_failure', 'answer_complete'):
                if name in row and row[name] != outcome[name]:
                    raise ValueError(f'{name} metadata disagrees with raw native completion')
            # Remove only a complete terminal natural answer. Incomplete headers
            # remain part of the observed text; no stripping or normalization.
            suffix = re.search(r'(?:^|\n)Answer:\s*[A-F]\.?' r'\s*$', text) if outcome['answer_complete'] else None
            trace = text[:suffix.start()] if suffix is not None else text
            frequencies = Counter(ids)
            record = {'world_id': row['world_id'], 'seed': _seed(row), 'trace': trace,
                      'cap_reached': outcome['cap_reached'], 'cap_failure': outcome['cap_failure'],
                      'answer_complete': outcome['answer_complete'],
                      'over_budget': not outcome['within_token_budget'],
                      'token_max_frequency_ratio': max(frequencies.values()) / len(ids) if ids else None,
                      'token_distinct_ratio': len(frequencies) / len(ids) if ids else None,
                      'edit_status': edits.get(_identity(row), 'not_assessed')}
            records.append(record)
            coverage[_seed(row)].add((row['world_id'], row.get('id', row.get('record_id'))))
        if len(coverage) > 1 and any(value != next(iter(coverage.values())) for value in coverage.values()):
            raise ValueError('Every seed must cover the same complete native story/record panel')
        summaries.append({
            'identity': dict(zip(PANEL_FIELDS, identity)), 'completion_protocol': completion_protocol,
            'native_records': len(records), 'stories': len({row['world_id'] for row in records}),
            'training_seeds': sorted(coverage, key=repr),
            'checkpoint_bindings': [{**{key: row.get(key) for key in (
                'checkpoint_id', 'step', 'training_seed', 'seed')},
                'adapter_sha256': row.get('adapter_sha256', row.get('checkpoint_adapter_sha256'))}
                for row in sorted({repr((row.get('checkpoint_id'), _seed(row))): row
                                   for row in panel}.values(), key=lambda item: repr(_seed(item)))],
            **{name: _mean_interval(records, name, seed, samples) for name in (
                'cap_reached', 'cap_failure', 'answer_complete', 'over_budget',
                'token_max_frequency_ratio', 'token_distinct_ratio')},
            'exact_common_trace': _exact_common_trace(records, seed, samples),
            'editability': _editability(records, seed, samples),
        })
    return {
        'schema_version': 1, 'status': 'measured' if summaries else 'not_assessed',
        'completion_protocol': completion_protocol, 'token_budget': cap,
        'input_records': len(rows), 'native_reasoning_records': len(native),
        'other_format_records': len(rows) - len(native), 'panels': summaries,
        'bootstrap_samples': samples, 'bootstrap_seed': seed, 'uncertainty': UNCERTAINTY,
        'definitions': {
            'cap_reached': 'The sampled complete response reaches or exceeds 384 tokens; a complete terminal answer at that boundary is valid without EOS.',
            'cap_failure': 'The complete-response budget was reached without a complete natural terminal Answer: X (optional adjacent period).',
            'token_repetition_scope': 'Unpadded sampled completion IDs, including natural delimiter, answer and EOS when present; never re-tokenized trace text.',
            'token_max_frequency_ratio': 'Count of the most frequent token divided by sampled completion tokens; empty completions are unassessed for this ratio.',
            'token_distinct_ratio': 'Number of distinct token IDs divided by sampled completion tokens; empty completions are unassessed for this ratio.',
            'not_assessed': 'No native trace or fixed-editor result was supplied; this is not a favorable or unfavorable measurement.',
        },
    }
