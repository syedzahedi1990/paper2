# G1 editability from saved native completions

This workflow classifies the actual development completions saved during training. It never regenerates the source model's text, selects a learning rate, approves spending or decides a scientific gate. It runs after all twelve amended candidate attempts are terminal and family settings have been computed from their saved scores and frozen under the original rule. See the [editor/selection addendum](amendments/2026-10-30-editor-and-selection.md).

The selected reasoning-RL run contributes all 120 direct-question stories at steps 0, 250, …, 2,000: 1,080 contracts. Incorrect, empty, repetitive, malformed and noneditable traces remain. There is no editability threshold. The fixed editor is Qwen2.5-7B-Instruct at revision `a09a35458c702b33eeacc393d103063234e8bc28`, prompt `paper2.trace_controls.v4`, greedy decoding with SDPA, seed 2026 and a 2,048-output-token cap. The frozen audit sample is 100 at seed 2026, plus the existing source-coverage supplement.

## Compute selection and inspect the first completed candidate

First freeze the separate failure policy before observing any amended candidate failure. The original registry selection rule is unchanged:

```bash
paper2 freeze-selection-failure-policy --reviewer "Attributed reviewer" \
  --rationale "Prospective terminal-attempt policy" --output runs/failure-policy.json
paper2 select-panel --execution-binding runs/original/execution-binding.json \
  --sha256 ORIGINAL_BINDING_SHA256 --panel-status runs/panel/status.json \
  --failure-policy runs/failure-policy.json --failure-policy-sha256 POLICY_SHA256 \
  --reviewer "Attributed reviewer" --rationale "Apply frozen final-native family rule" \
  --output runs/panel-selection.json
paper2 verify-panel-selection --selection runs/panel-selection.json --sha256 SELECTION_SHA256
paper2 selected-confirmatory-configs --selection runs/panel-selection.json \
  --selection-sha256 SELECTION_SHA256 --output runs/selected-confirmatory
```

Selection cannot run while any candidate is unattempted or running. All failed attempts are retained; an eligible shared RL setting needs both arms completed, and SFT needs label SFT completed. No attempt is replaced or given an invented zero. Bind the computed receipt as `development_selection` in the gate review and list it in G1 evidence; G1 learning curves must point to its three selected runs. Exporting configs is not permission to launch or approve a gate.

Once the first amended reasoning candidate finishes, inspect its original files even while the rest of the panel runs:

```bash
paper2 g1-editability inspect-completed --execution-binding runs/original/execution-binding.json \
  --sha256 ORIGINAL_BINDING_SHA256 --panel-status runs/panel/status.json \
  --candidate reasoning_rl_clean_seed1_lr-x1_constant --output runs/first-completed-inspection.json
```

This verifies all nine native panels and original adapter files without selecting the candidate or preparing an editor batch. It does not approve G1. Preserve the original absolute paths; a copied receipt cannot substitute for missing weights.

## Bind the completed observations

Prepare a JSON binding with `purpose: "g1_editability_only"`, `scope: "selected_reasoning_rl_after_complete_panel"`, `candidate_name`, and the original `source_commit`. Include `{ "path": "…", "sha256": "…" }` references for these roles (relative paths resolve from the binding):

- `selection`: immutable receipt produced by `paper2 select-panel`, with actual score, failure-policy and original-source bindings. Preparation recomputes it; hand-written winner names are rejected.
- `panel_status`, `registry`, `config`: the complete operational schema-2 twelve-candidate registry/status and selected configuration.
- `run_manifest`, `predictions`, `learning_evidence`: the selected run's original manifest, `checkpoint_predictions.jsonl`, and finalized `g1_learning_evidence.json`.
- `development`, `corpus_manifest`: the frozen clean development records and manifest.
- `execution_environment`, `candidate_execution_binding`, `batch_frozen_inputs`: the original execution/source/runtime declarations bound by the completed panel. A newly invented declaration cannot replace this lineage.

The helper rechecks all twelve terminal attempts and the computed winner under the prospective failure policy, the selected run's actual 2,000-story exposure, every 9×120 record/outcome and retained adapter hash, and the original source/config/corpus bindings. It requires a fresh output directory.

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

Once costed, resume the same output with `--resume`, optionally another `--max-calls` bound. Finished items, including failed or noneditable outputs, are never resampled. A preparation owns one permanent output claim, across output paths. Every model-call start is durable; a start without its saved result blocks resume rather than being resampled. Existing completed output, duplicate ownership, changed inputs and partial corrupt journal records fail closed. Source inference is never performed. A completed journal can be assembled without loading the editor again.

```bash
paper2 g1-editability generate --input runs/g1-observed-inputs/prepared.json \
  --sha256 PREPARED_FILE_SHA256 --output runs/g1-generated.json --resume
paper2 g1-editability finalize --input runs/g1-observed-inputs/prepared.json \
  --sha256 PREPARED_FILE_SHA256 --generated runs/g1-generated.json \
  --output runs/g1-editability-evidence.json
```

Bind the final receipt as G1 `editability_evidence` with its actual SHA256. `verify-g1` checks every raw native checkpoint classification and presents its estimate and interval. Use `prepare-edit-audit` to inspect this editor batch when reviewing G1 classifications. Test 2 needs its own full-provenance batch and preregistered random human audit; a G1-only batch cannot replace it. This workflow cannot supply human judgment.

## Add lengths to already saved health evidence

```bash
paper2 trace-health --input runs/selected/checkpoint_predictions.jsonl \
  --original-health runs/selected/checkpoint_trace_health.jsonl \
  --output runs/selected-trace-health-lengths.json
```

This verifies original v1 summaries exactly and writes a separate v2 length report without changing raw tokens, summaries, scores or selection. Add `--tokenizer mistralai/Mistral-Small-24B-Instruct-2501 --revision 9527884be6e5616bdd54de542f9ae13384489724` for explicit sampled-prefix bounds and separately labelled re-tokenization proxies. No language model is loaded for length reanalysis. Completion length and character/word counts need no tokenizer download.
