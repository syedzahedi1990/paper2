"""Reproduce structural and tokenizer validation; never loads model weights."""
import argparse
import json
from pathlib import Path

from paper2.data import verify_corpus, audit_tokenizer_lengths
from paper2.data.provenance import UPSTREAM_PROVENANCE
from paper2.io import file_digest, write_json
from paper2.prompts import answer_token_ids, render_prompt
from paper2.training.config import MODEL_REVISIONS, TrainingConfig


def records(paths):
    for path in paths:
        with Path(path).open() as stream:
            yield from (json.loads(line) for line in stream if line.strip())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', default='Qwen/Qwen2.5-3B-Instruct')
    parser.add_argument('--local-files-only', action='store_true')
    args = parser.parse_args()
    from transformers import AutoTokenizer
    verified = verify_corpus(args.data)
    if not verified["required_controls_passed"]:
        raise ValueError("Required structural controls failed; inspect the shortcut audit")
    manifest = json.loads((args.data / 'manifest.json').read_text())
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=MODEL_REVISIONS[args.model],
                                              local_files_only=args.local_files_only)
    train_paths = sorted((args.data / 'training').glob('*.jsonl'))
    heldout_paths = sorted((args.data / 'evaluation').glob('*.jsonl'))
    token_audits = {'training_twins': audit_tokenizer_lengths(records(train_paths), tokenizer),
                    'heldout_cues': audit_tokenizer_lengths(records(heldout_paths), tokenizer)}
    prompt_lengths, trace_lengths = [], []
    for row in records(train_paths):
        prompt = render_prompt(tokenizer, row, reasoning=True)
        prompt_lengths.append(len(tokenizer.encode(prompt, add_special_tokens=False)))
        answer_token_ids(tokenizer, render_prompt(tokenizer, row), row['choices'])
    for row in records(sorted((args.data / 'supervision').glob('*.jsonl'))):
        trace_lengths.append(len(tokenizer.encode(row['trace'], add_special_tokens=False)))
    limits = TrainingConfig().max_prompt_length
    if not all(audit['passed'] for audit in token_audits.values()):
        raise ValueError('Cue renderings fail tokenizer length matching')
    if max(prompt_lengths) > limits or max(trace_lengths) > 384:
        raise ValueError('New corpus exceeds frozen training prompt/trace bounds')
    output = {**{key: value for key, value in verified.items() if key != 'shortcut_audit'}, 'corpus_schema_version': manifest['schema_version'],
              'manifest_sha256': file_digest(args.data / 'manifest.json'),
              'source_provenance': UPSTREAM_PROVENANCE,
              'tokenizer': {'model': args.model, 'revision': MODEL_REVISIONS[args.model], **token_audits,
                            'training_prompts': len(prompt_lengths), 'answer_token_checks': len(prompt_lengths),
                            'prompt_tokens': {'min': min(prompt_lengths), 'max': max(prompt_lengths), 'limit': limits},
                            'oracle_trace_tokens': {'min': min(trace_lengths), 'max': max(trace_lengths), 'limit': 384}},
              'cue_design': manifest['cue_design'],
              'compute': 'CPU generation and tokenizer only; no model weights or research training',
              'empirical_gates_passed': []}
    write_json(args.output.parent / 'shortcut-audit.json', verified['shortcut_audit'])
    output['shortcut_audit_file'] = 'shortcut-audit.json'
    output['shortcut_audit_sha256'] = file_digest(args.output.parent / 'shortcut-audit.json')
    write_json(args.output, output)
    print(json.dumps({'corpus_hash': verified['corpus_hash'], 'tokenizer': output['tokenizer'],
                      'required_controls_passed': verified['required_controls_passed'],
                      'requires_heuristic_review': verified['requires_heuristic_review'],
                      'calibration_ready': verified['calibration_ready']}))


if __name__ == '__main__':
    main()
