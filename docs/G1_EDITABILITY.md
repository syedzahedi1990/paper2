# G1 editability from saved native completions

This workflow classifies the actual development completions saved during training. It never regenerates the source model's text, selects a learning rate, approves spending or decides a scientific gate. It runs after the complete amended twelve-candidate panel and a separately frozen family-selection decision.

The selected reasoning-RL run contributes all 120 direct-question stories at steps 0, 250, …, 2,000: 1,080 contracts. Incorrect, empty, repetitive, malformed and noneditable traces remain. There is no editability threshold. The fixed editor is Qwen2.5-7B-Instruct at revision `a09a35458c702b33eeacc393d103063234e8bc28`, prompt `paper2.trace_controls.v3`, greedy decoding with SDPA, seed 2026 and a 2,048-output-token cap. The frozen audit sample is 100 at seed 2026, plus the existing source-coverage supplement.

## Bind the completed observations

Prepare a JSON binding with `purpose: "g1_editability_only"`, `scope: "selected_reasoning_rl_after_complete_panel"`, `candidate_name`, and the original `source_commit`. Include `{ "path": "…", "sha256": "…" }` references for these roles (relative paths resolve from the binding):

- `selection`: frozen selection receipt with `status: "frozen"`, `selected_reasoning_rl_candidate`, named `reviewed_by`, `rationale`, exact `panel_status_sha256` and `registry_sha256`, and the original `selection_rule`.
- `panel_status`, `registry`, `config`: the complete operational schema-2 twelve-candidate registry/status and selected configuration.
- `run_manifest`, `predictions`, `learning_evidence`: the selected run's original manifest, `checkpoint_predictions.jsonl`, and finalized `g1_learning_evidence.json`.
- `development`, `corpus_manifest`: the frozen clean development records and manifest.
- `execution_environment`, `candidate_execution_binding`, `batch_frozen_inputs`: the original execution/source/runtime declarations bound by the completed panel. A newly invented declaration cannot replace this lineage.

The helper checks all twelve completion receipts, the selected run's actual 2,000-story exposure, every 9×120 record/outcome and retained adapter hash, and the original source/config/corpus bindings. It requires a fresh output directory.

```bash
paper2 g1-editability prepare --input runs/g1-input-binding.json \
  --sha256 BINDING_FILE_SHA256 --output runs/g1-observed-inputs
paper2 g1-editability resources --input runs/g1-observed-inputs/prepared.json \
  --sha256 PREPARED_FILE_SHA256 --output runs/g1-editor-workload.json
```

Replace capitalized SHA placeholders with hashes of the actual files. Preparation writes `edit_only_sources.jsonl`, `edit_manifest.json`, `editor_plan.json`, `classifications.mapping.json`, `extraction.json` and a hashed `prepared.json`. It records absent original prompt IDs honestly. These edit-only sources cannot enter `crossed` or `regenerate-traces`; those tests require their full source-generation provenance.

## Measure, cost and run the frozen editor

The fixed upper output-token budget is 1,080 × 2,048 = 2,211,840, plus input tokens. Before editor profiling, runtime and cost fields are null. The training panel's estimate excludes this work. First record a bounded profiling decision within the remaining £500 budget; then, for example, measure ten calls:

```bash
paper2 g1-editability generate --input runs/g1-observed-inputs/prepared.json \
  --sha256 PREPARED_FILE_SHA256 --output runs/g1-generated.json --max-calls 10
paper2 g1-editability resources --input runs/g1-observed-inputs/prepared.json \
  --sha256 PREPARED_FILE_SHA256 --journal runs/g1-generated.json.journal.jsonl \
  --output runs/g1-editor-measured-resources.json
```

Ten is an illustrative resource sample size, not a quality cutoff or automatic spending authorization. The journal records synchronized inference time, actual token counts and failures. The projection is descriptive, from a sequential sample; it excludes loading, idle time, transfer and human review. Supply `--hourly-gbp` with the recorded applicable rate to calculate a labelled inference-cost projection. Account for omitted overhead separately before approving the full editor job.

Once costed, resume the same output with `--resume`, optionally another `--max-calls` bound. Finished items, including failed or noneditable outputs, are never resampled. Existing completed output, duplicate ownership, changed inputs and partial corrupt journal records fail closed. Source inference is never performed. A completed journal can be assembled without loading the editor again.

```bash
paper2 g1-editability generate --input runs/g1-observed-inputs/prepared.json \
  --sha256 PREPARED_FILE_SHA256 --output runs/g1-generated.json --resume
paper2 g1-editability finalize --input runs/g1-observed-inputs/prepared.json \
  --sha256 PREPARED_FILE_SHA256 --generated runs/g1-generated.json \
  --output runs/g1-editability-evidence.json
```

Bind the final receipt as G1 `editability_evidence` with its actual SHA256. `verify-g1` checks every raw native checkpoint classification and presents its estimate and interval. Use `prepare-edit-audit` on the generated batch to prepare the frozen sample for the named human's G3 review. This workflow cannot supply that human judgment.

## Add lengths to already saved health evidence

```bash
paper2 trace-health --input runs/selected/checkpoint_predictions.jsonl \
  --original-health runs/selected/checkpoint_trace_health.jsonl \
  --output runs/selected-trace-health-lengths.json
```

This verifies original v1 summaries exactly and writes a separate v2 length report without changing raw tokens, summaries, scores or selection. Add `--tokenizer mistralai/Mistral-Small-24B-Instruct-2501 --revision 9527884be6e5616bdd54de542f9ae13384489724` for explicit sampled-prefix bounds and separately labelled re-tokenization proxies. No language model is loaded for length reanalysis. Completion length and character/word counts need no tokenizer download.
