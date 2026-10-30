"""Additive length diagnostics retain historical summaries and sampled IDs."""
from copy import deepcopy
from hashlib import sha256
import json

import pytest

from paper2.training.config import NATURAL_COMPLETION_PROTOCOL
from paper2.training.trace_health import (
    summarize_trace_health, summarize_trace_health_v2, trace_token_attribution,
)
from paper2.training.trace_health_reanalysis import reanalyze_trace_health


def raw(world='a', *, seed=1, text='know\nAnswer: A', ids=None, **changes):
    return {'id': f'{world}-direct', 'world_id': world, 'format': 'native_reasoning',
            'arm': 'reasoning_rl', 'dataset': 'clean', 'model': 'fixture', 'revision': 'f' * 40,
            'step': 0, 'seed': seed, 'checkpoint_id': f'fixture-{seed}',
            'completion_protocol': NATURAL_COMPLETION_PROTOCOL,
            'completion': text, 'completion_ids': [1, 2, 3, 4] if ids is None else ids, **changes}


class BoundaryTokenizer:
    pieces = {1: 'Alice', 2: ' believes', 3: ' box', 4: '.\n', 5: 'Answer',
              6: ':', 7: ' A', 8: '', 9: '.', 10: '\n'}

    def decode(self, ids, *, skip_special_tokens):
        assert skip_special_tokens
        return ''.join(self.pieces[token] for token in ids)

    def encode(self, text, *, add_special_tokens):
        assert not add_special_tokens
        # Deliberately different from the stored generated IDs.
        return list(text.encode('utf-8'))


def test_merged_period_newline_reports_bounds_not_fabricated_sampled_trace_ids():
    tokenizer = BoundaryTokenizer()
    ids = [1, 2, 3, 4, 5, 6, 7, 8]
    original = list(ids)
    result = trace_token_attribution('Alice believes box.', tokenizer.decode(ids, skip_special_tokens=True),
                                    ids, tokenizer=tokenizer)
    assert result['status'] == 'text_boundary_inside_sampled_token_span'
    assert result['sampled_trace_tokens_exact'] is None
    assert result['sampled_trace_token_ids'] is None
    assert result['sampled_prefix_lower_bound'] == 3
    assert result['sampled_prefix_upper_bound'] == 4
    assert result['boundary_crosses_one_sampled_token'] is True
    assert result['retokenized_trace_tokens_proxy'] == len('Alice believes box.')
    assert ids == original


def test_exact_boundary_uses_actual_prefix_and_does_not_count_eos():
    tokenizer = BoundaryTokenizer()
    ids = [1, 2, 3, 9, 10, 5, 6, 7, 8]
    result = trace_token_attribution('Alice believes box.', tokenizer.decode(ids, skip_special_tokens=True),
                                    ids, tokenizer=tokenizer)
    assert result['sampled_trace_tokens_exact'] == 4
    assert result['sampled_trace_token_ids'] == [1, 2, 3, 9]
    assert result['sampled_prefix_lower_bound'] == result['sampled_prefix_upper_bound'] == 4
    # Missing terminal answer means the whole visible completion remains trace.
    ending = trace_token_attribution('Alice', 'Alice', [1, 8], tokenizer=tokenizer)
    assert ending['sampled_trace_tokens_exact'] == 1


def test_unstable_unicode_decoding_bounds_multiple_actual_tokens():
    class UnicodeTokenizer:
        @staticmethod
        def decode(ids, *, skip_special_tokens):
            return {(): '', (1,): '\ufffd', (1, 2): 'é\n', (1, 2, 3): 'é\nAnswer: A'}[tuple(ids)]

        @staticmethod
        def encode(text, *, add_special_tokens):
            return [42]

    result = trace_token_attribution('é', 'é\nAnswer: A', [1, 2, 3], tokenizer=UnicodeTokenizer())
    assert result['sampled_prefix_lower_bound'] == 0
    assert result['sampled_prefix_upper_bound'] == 2
    assert result['boundary_crosses_one_sampled_token'] is False
    assert result['sampled_trace_token_ids'] is None


def test_lengths_distinguish_distinct_one_word_traces_from_sentences_without_thresholds():
    short = [raw('a', text='know\nAnswer: A', ids=[1, 2]), raw('b', text='yes\nAnswer: B', ids=[3, 4])]
    long = [raw('a', text='Alice believes the box is here.\nAnswer: A', ids=list(range(12))),
            raw('b', text='Bob believes the box is there.\nAnswer: B', ids=list(range(12)))]
    small = summarize_trace_health_v2(short, samples=50)
    large = summarize_trace_health_v2(long, samples=50)
    assert small['legacy_summary']['panels'][0]['exact_common_trace']['estimate'] == .5
    assert large['legacy_summary']['panels'][0]['exact_common_trace']['estimate'] == .5
    lengths = small['panels'][0]['lengths']
    assert lengths['trace_whitespace_words']['raw_distribution'] == [{'value': 1, 'records': 2}]
    assert lengths['completion_sampled_tokens']['mean']['estimate'] == 2
    assert lengths['completion_sampled_tokens']['mean']['ci95'] == [2, 2]
    assert large['panels'][0]['lengths']['trace_whitespace_words']['mean']['estimate'] == 6
    assert lengths['trace_sampled_tokens_exact']['mean']['estimate'] is None
    assert lengths['trace_sampled_tokens_exact']['mean']['missing_observations'] == 2
    assert not any('pass' in key or 'threshold' in key for key in lengths)


def test_new_lengths_keep_seed_within_story_mean_and_saved_token_lengths():
    rows = [raw('a', seed=1, ids=[1]), raw('a', seed=2, ids=list(range(9))),
            raw('b', seed=1, ids=[1]), raw('b', seed=2, ids=list(range(9)))]
    result = summarize_trace_health_v2(rows, samples=50)['panels'][0]['lengths']['completion_sampled_tokens']
    assert result['mean']['estimate'] == 5
    assert result['mean']['ci95'] == [5, 5]
    assert result['raw_distribution'] == [{'value': 1, 'records': 2}, {'value': 9, 'records': 2}]
    assert result['mean']['raw_sum'] == 20


def test_v2_keeps_exact_v1_calculation_and_inputs_and_records_tokenizer_provenance():
    tokenizer = BoundaryTokenizer()
    ids = [1, 2, 3, 4, 5, 6, 7, 8]
    rows = [raw(text=tokenizer.decode(ids, skip_special_tokens=True), ids=ids, eos_token_id=8)]
    before = deepcopy(rows)
    old = summarize_trace_health(rows, samples=40)
    result = summarize_trace_health_v2(rows, tokenizer=tokenizer,
        tokenizer_provenance={'tokenizer_id': 'fixture', 'revision': 'f' * 40}, samples=40)
    assert result['schema_version'] == 2 and result['legacy_summary'] == old
    assert summarize_trace_health(rows, samples=40) == old and rows == before
    panel = result['panels'][0]
    assert panel['lengths']['completion_sampled_tokens']['mean']['estimate'] == 8
    assert panel['lengths']['trace_sampled_tokens_exact']['mean']['estimate'] is None
    assert panel['lengths']['trace_sampled_token_upper_bound']['mean']['estimate'] == 4
    assert panel['token_attribution_status_counts'] == {'text_boundary_inside_sampled_token_span': 1}


def test_empty_and_immediate_only_are_descriptive_not_failure():
    empty = summarize_trace_health_v2([raw(text='', ids=[])], samples=10)['panels'][0]['lengths']
    assert empty['completion_sampled_tokens']['mean']['estimate'] == 0
    assert empty['trace_whitespace_words']['mean']['estimate'] == 0
    immediate = summarize_trace_health_v2([raw(format='immediate')], samples=10)
    assert immediate['status'] == 'not_assessed' and immediate['panels'] == []


def test_v2_rejects_tokenizer_mismatch_unbound_tokenizer_or_invalid_original_metadata():
    with pytest.raises(ValueError, match='provenance'):
        summarize_trace_health_v2([raw()], tokenizer=BoundaryTokenizer(), samples=10)
    with pytest.raises(ValueError, match='without a tokenizer'):
        summarize_trace_health_v2([raw()], tokenizer_provenance={'id': 'fixture'}, samples=10)
    with pytest.raises(ValueError, match='differs from decoding'):
        summarize_trace_health_v2([raw()], tokenizer=BoundaryTokenizer(),
                                 tokenizer_provenance={'id': 'fixture'}, samples=10)
    with pytest.raises(ValueError, match='length disagrees'):
        summarize_trace_health_v2([raw(completion_tokens=200)], samples=10)


def write_sources(tmp_path):
    rows = [raw('a'), raw('b')]
    source = tmp_path / 'predictions.jsonl'
    source.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    health = tmp_path / 'checkpoint_trace_health.jsonl'
    health.write_text(json.dumps({**rows[0], 'native_reasoning_assessed': True,
        'trace_health': summarize_trace_health(rows, samples=13)}) + '\n')
    return source, health


def test_additive_reanalysis_verifies_original_settings_and_hashes_without_rewriting(tmp_path):
    source, health = write_sources(tmp_path)
    before = (source.read_bytes(), health.read_bytes())
    output = tmp_path / 'derived-v2.json'
    result = reanalyze_trace_health(source, output, original_health_path=health, samples=29)
    assert result['original_health_validation'] == {'status': 'verified_exact_v1_recomputation', 'checkpoints': 1}
    assert result['trace_health_v2']['bootstrap_samples'] == 29
    assert result['inputs']['predictions']['sha256'] == sha256(before[0]).hexdigest()
    assert result['inputs']['original_trace_health']['sha256'] == sha256(before[1]).hexdigest()
    assert json.loads(output.read_text()) == result
    assert (source.read_bytes(), health.read_bytes()) == before
    with pytest.raises(FileExistsError, match='new path'):
        reanalyze_trace_health(source, output, original_health_path=health, samples=10)
    with pytest.raises(FileExistsError, match='new path'):
        reanalyze_trace_health(source, source, samples=10)


def test_reanalysis_does_not_accept_modified_v1_summary_or_incomplete_panel(tmp_path):
    source, health = write_sources(tmp_path)
    old = json.loads(health.read_text())
    old['trace_health']['panels'][0]['cap_failure']['estimate'] = 1
    health.write_text(json.dumps(old) + '\n')
    output = tmp_path / 'derived.json'
    with pytest.raises(ValueError, match='disagrees with raw'):
        reanalyze_trace_health(source, output, original_health_path=health, samples=10)
    assert not output.exists()
    health.write_text('')
    with pytest.raises(ValueError, match='every supplied raw checkpoint'):
        reanalyze_trace_health(source, output, original_health_path=health, samples=10)


def test_reanalysis_stops_if_a_live_source_changes(tmp_path, monkeypatch):
    import paper2.training.trace_health_reanalysis as module

    source, _ = write_sources(tmp_path)
    ordinary = module.summarize_trace_health_v2

    def changing(*args, **kwargs):
        result = ordinary(*args, **kwargs)
        source.write_text(source.read_text() + '\n')
        return result

    monkeypatch.setattr(module, 'summarize_trace_health_v2', changing)
    output = tmp_path / 'derived.json'
    with pytest.raises(ValueError, match='changed during reanalysis'):
        reanalyze_trace_health(source, output, samples=10)
    assert not output.exists()
