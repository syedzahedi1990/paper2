# Model-generated controls and sampled human review

The primary confirmatory edit measurement uses `direct_fixed`: one native direct-question trace is generated per world and source checkpoint, then held fixed across all eight views. The 97-world precision plan remains in force. Question-dependent edits remain pilot or supplemental diagnostics, and cannot silently substitute for this confirmatory measurement.

For 24 trained final checkpoints plus one shared base, one clean rendering and 97 worlds, this creates **2,425 control contracts**, compared with 19,400 separate question-dependent contracts. Additional matched checkpoints or cue renderings increase these counts. Generating controls is still compute: the plan records the number of calls and its output-token cap. It is not covered by the training-only profile or a claim that four weeks is sufficient.

## Freeze the method before generating controls

1. Collect and save all source traces. Include the base once in a grid-wide batch; each trained checkpoint retains its own identity. Reader runs still operate on one base/trained seed pair at a time.
2. Prepare the oracle-backed edit manifest. The oracle specifies the focal belief change and all eight consequences. Selection depends only on world IDs and the frozen seed, not observed effects.
3. Freeze one editor model, immutable revision, prompt version, decoding settings, token budget and audit seed/sample size. The checked-in prompt requests a paraphrase, an irrelevant wording edit and the exact span for the oracle-proposed semantic edit. It forbids correcting source mistakes or inventing a missing belief. No experiment source or reader regenerates these controls independently.
4. Generate once and save every output, including unresolved ones. The default is greedy decoding, one attempt per contract. Prompts, source/input hashes, raw outputs and generation metadata are retained. A per-contract journal supports recovery without resampling completed items.

A runnable two-source pilot example is below. The editor model/revision is an example, not a claim of adequate editing quality. Validate the fixed prompt/model on the pilot before freezing the confirmatory plan.

```bash
paper2 prepare-edits --traces runs/traces-base-fixed.jsonl runs/traces-trained-fixed.jsonl \
  --input data/generated/pilot-audit-v4/evaluation/development__clean.jsonl \
  --stage pilot --max-worlds 12 --output runs/edit-templates.json
paper2 plan-edit-generation --edits runs/edit-templates.json \
  --model Qwen/Qwen2.5-7B-Instruct --revision a09a35458c702b33eeacc393d103063234e8bc28 \
  --audit-sample-size 60 --audit-seed 2026 --output runs/edit-generation-plan.json
paper2 generate-edits --edits runs/edit-templates.json --plan runs/edit-generation-plan.json \
  --traces runs/traces-base-fixed.jsonl runs/traces-trained-fixed.jsonl \
  --output runs/generated-edits.json
paper2 prepare-edit-audit --edits runs/generated-edits.json --output runs/edit-audit.json
```

For confirmation, use confirmation rows, all intended source files, and `--stage confirmatory --precision-half-width .10`. Preparation requires frozen direct-question traces projected onto the complete eight-view panel. After an interrupted generation, repeat the same command with `--resume`; it validates and reuses `generated-edits.json.journal.jsonl`. Existing final batches and plans cannot be overwritten. Unresolved outputs are retained and are not retried on resume.

## What is checked automatically, and what a human checks

Machine checks require exact source spans, one oracle-location substitution, nonoverlapping irrelevant edits of comparable length, paraphrase length bounds, strict JSON and absence of explicit answer leakage. They do not establish semantic fidelity. The model cannot set human approval flags or alter oracle facts or forecasts.

The audit sample is chosen **before generation**: a globally hash-ranked sample of 60 contracts by default, plus one from each source/checkpoint stratum absent from that sample. A small batch is audited in full. The plan records actual sample size and an illustrative finite-population detection calculation for a fixed 5% defective set, assuming a seed chosen independently before outputs and accurate human judgments. This is not a guarantee that every edit is valid, and supplements are not counted in that calculation. Increasing audit size is a preregistered choice, not a response to a favourable sample.

For every selected item, a named human checks paraphrase meaning, whether only the intended semantic fact changes, irrelevant-edit neutrality and the oracle forecast's applicability. The audit's zero-error acceptance rule is fixed in protocol v1. The completed file records the reviewer and all four checks, plus explicit batch acceptance. Unsampled contracts remain marked `approved: false`; their use is described as sampled quality assurance, not individual human review.

```bash
# After the investigator completes the sampled audit file:
paper2 verify-edit-audit --edits runs/generated-edits.json --audit runs/edit-audit.json \
  --output runs/edit-audit-evidence.json
```

Any invalid/unresolved output, failed or missing sampled judgment, changed prompt/model/contract or altered sample blocks the batch. Do not drop difficult traces or regenerate until the sample passes. If the pilot fails, revise and preregister the method before confirmation; preserve failed attempts. An inability to locate the proposed belief is a real editability limitation. The code does not infer human acceptance from generated text.

## Crossed use

Pass `--edits runs/generated-edits.json --edit-audit runs/edit-audit.json` to both `regenerate-traces` and `crossed`. Regenerate semantic continuations once with each trace's original source checkpoint, then supply those same frozen continuations to both readers. An audit approving an editor batch does not authorize reader-specific resampling. Every use rechecks batch provenance and the sample decision; output rows identify the review method, plan and audit hashes. Altered content invalidates the old audit and frozen continuations.

The original individually reviewed manifest path remains available for pilot diagnostics. It is no longer a requirement that every confirmatory trace receive a hand-written paraphrase and irrelevant edit.

## Relation to prior work

[Lanham et al. (2023), §§2.4 and 2.6](https://arxiv.org/html/2307.13702v1) uses model-generated mistaken reasoning and paraphrases, with continuations regenerated by the original source model for the mistake experiment. Our oracle-specified factual edits, irrelevant-edit construction, full-grid batch and sampled human review are explicit Paper 2 choices; they are not claimed to reproduce that paper's exact protocol.
