import importlib.util
import json
from pathlib import Path

import pytest

from paper2.data import VIEWS
from paper2.data.corpus import CORPUS_SCHEMA
from paper2.io import digest, file_digest, write_json, write_jsonl

SPEC = importlib.util.spec_from_file_location('capability_screen', Path(__file__).parents[1] / 'scripts/run_capability_screen.py')
screen = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(screen)
REVISION = 'a' * 40


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / 'corpus'
    rows = [{'id': f'{index}:{view}', 'world_id': f'world{index:03}', 'view': view,
             'split': 'development', 'cue_condition': 'clean', 'cue_location': 'drawer',
             'belief_depth': 1 + index % 2, 'critical_event_kind': ('move', 'tell')[index // 2 % 2],
             'answer': 'A'} for index in range(120) for view in VIEWS]
    name = 'evaluation/development__clean.jsonl'
    write_jsonl(root / name, rows)
    manifest = {'schema_version': CORPUS_SCHEMA, 'config': {'development': 120},
                'files': {name: {'rows': 960, 'sha256': file_digest(root / name)}},
                'shortcut_audit': {}, 'integrity_passed': True}
    manifest['corpus_hash'] = digest({key: manifest[key] for key in ('schema_version', 'config', 'files', 'shortcut_audit')})
    write_json(root / 'manifest.json', manifest)
    return root


class FakeBackend:
    def __init__(self, model, **kwargs):
        self.checkpoint_id = model


def install_evaluators(monkeypatch, fail_after=None):
    calls = []

    def evaluate(backend, rows, reasoning=False):
        for row in rows:
            calls.append((row['id'], reasoning))
            if fail_after is not None and len(calls) > fail_after:
                raise RuntimeError('simulated interruption')
            yield {**row, 'format': 'native_reasoning' if reasoning else 'immediate',
                   'prediction': 'A' if reasoning or row['belief_depth'] == 1 else 'B'}
    monkeypatch.setattr(screen, 'evaluate_immediate', evaluate)
    monkeypatch.setattr(screen, 'evaluate_native', evaluate)
    return calls


def test_complete_development_screen_and_completed_resume_never_loads_model(corpus, tmp_path, monkeypatch):
    calls = install_evaluators(monkeypatch)
    output = tmp_path / 'run'
    result = screen.run_screen(corpus, output, 'test/model', REVISION, backend_factory=FakeBackend)
    assert len(calls) == 1080
    assert sum(reasoning for _, reasoning in calls) == 120
    immediate = result['summaries']['immediate']['overall']
    assert immediate['accuracy'] == .5
    assert immediate['stories'] == 120 and immediate['bootstrap_samples'] == 20000
    assert len(immediate['per_view']) == 8
    assert result['summaries']['native_reasoning']['overall']['accuracy'] == 1
    assert result['summaries']['immediate']['strata']['depth2_move']['stories'] == 30
    assert result['gate_decision'] is None
    assert len((output / 'worlds.jsonl').read_text().splitlines()) == 120
    def forbidden(*args, **kwargs):
        raise AssertionError('Completed resume must not load a model')
    replay = screen.run_screen(corpus, output, 'test/model', REVISION, backend_factory=forbidden)
    assert replay == result
    assert len(calls) == 1080


def test_interrupted_run_resumes_only_unfinished_worlds(corpus, tmp_path, monkeypatch):
    calls = install_evaluators(monkeypatch, fail_after=9)
    output = tmp_path / 'run'
    with pytest.raises(RuntimeError, match='simulated'):
        screen.run_screen(corpus, output, 'test/model', REVISION, backend_factory=FakeBackend)
    receipt = json.loads((output / 'receipt.json').read_text())
    assert receipt['completed_worlds'] == 1 and receipt['status'] == 'incomplete'
    saved = (output / 'worlds.jsonl').read_bytes()
    calls = install_evaluators(monkeypatch)
    result = screen.run_screen(corpus, output, 'test/model', REVISION, backend_factory=FakeBackend)
    assert len(calls) == 119 * 9 and all(not row_id.startswith('0:') for row_id, _ in calls)
    assert (output / 'worlds.jsonl').read_bytes().startswith(saved)
    assert result['completed_worlds'] == 120
    assert len(json.loads((output / 'receipt.json').read_text())['sessions']) == 2


def test_refuses_changed_protocol_and_saved_results(corpus, tmp_path, monkeypatch):
    install_evaluators(monkeypatch)
    output = tmp_path / 'run'
    screen.run_screen(corpus, output, 'test/model', REVISION, backend_factory=FakeBackend)
    with pytest.raises(ValueError, match='different immutable protocol'):
        screen.run_screen(corpus, output, 'other/model', REVISION, backend_factory=FakeBackend)
    path = output / 'worlds.jsonl'
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[0]['predictions'][0]['prediction'] = 'F'
    write_jsonl(path, records)
    with pytest.raises(ValueError, match='content changed'):
        screen.run_screen(corpus, output, 'test/model', REVISION, backend_factory=FakeBackend)


def test_rejects_nonimmutable_revision_and_changed_input(corpus, tmp_path):
    with pytest.raises(ValueError, match='immutable'):
        screen.run_screen(corpus, tmp_path / 'run', 'test/model', 'main', backend_factory=FakeBackend)
    path = corpus / 'evaluation/development__clean.jsonl'
    path.write_text(path.read_text().replace('development', 'confirmation', 1))
    with pytest.raises(ValueError, match='hash mismatch'):
        screen.development_panel(corpus)


@pytest.mark.parametrize('corruption', ['confirmation', 'duplicate_view', 'missing_world'])
def test_structural_panels_rejected_even_with_consistent_file_hash(corpus, tmp_path, corruption):
    path = corpus / 'evaluation/development__clean.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if corruption == 'confirmation':
        rows[0]['split'] = 'confirmation'
    elif corruption == 'duplicate_view':
        rows[0]['view'] = rows[1]['view']
    else:
        rows = rows[8:]
    write_jsonl(path, rows)
    manifest_path = corpus / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['files']['evaluation/development__clean.jsonl']['sha256'] = file_digest(path)
    manifest['corpus_hash'] = digest({key: manifest[key] for key in ('schema_version', 'config', 'files', 'shortcut_audit')})
    write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match='development|eight views'):
        screen.run_screen(corpus, tmp_path / 'run', 'test/model', REVISION, backend_factory=FakeBackend)
