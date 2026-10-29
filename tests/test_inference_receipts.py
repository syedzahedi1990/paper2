"""Bind actual inference adapter weights to the retained training checkpoint."""
from types import SimpleNamespace
import json

import pytest

from paper2.cli import main
from paper2.io import file_digest, write_jsonl


@pytest.mark.parametrize('change_weights', [False, True])
def test_inference_normalizes_checkpoint_and_binds_unchanged_weights(tmp_path, monkeypatch, change_weights):
    monkeypatch.chdir(tmp_path)
    adapter = tmp_path / 'checkpoint-50'
    adapter.mkdir()
    weights = adapter / 'adapter_model.safetensors'
    weights.write_bytes(b'fixture-weights')
    source = tmp_path / 'input.jsonl'
    output = tmp_path / 'output.jsonl'
    write_jsonl(source, [{'id': 'story-direct'}])
    def backend(args):
        assert args.adapter == str(adapter.resolve())
        return SimpleNamespace(checkpoint_id=args.adapter, model=SimpleNamespace(config=SimpleNamespace(_commit_hash='a' * 40)))
    def evaluate(*args, **kwargs):
        if change_weights:
            weights.write_bytes(b'changed-weights')
        yield {'id': 'story-direct', 'prediction': 'A'}
    monkeypatch.setattr('paper2.cli._backend', backend)
    monkeypatch.setattr('paper2.evaluation.evaluate_native', evaluate)
    command = ['evaluate', '--input', str(source), '--output', str(output), '--reader', 'trained',
               '--adapter', 'checkpoint-50', '--training-seed', '1', '--arm', 'reasoning_rl']
    if change_weights:
        with pytest.raises(ValueError, match='changed during inference'):
            main(command)
        assert not (tmp_path / 'output.jsonl.run.json').exists()
    else:
        main(command)
        receipt = json.loads((tmp_path / 'output.jsonl.run.json').read_text())
        assert receipt['checkpoint'] == str(adapter.resolve())
        assert receipt['adapter_sha256'] == file_digest(weights)
        assert receipt['output_sha256'] == file_digest(output)
