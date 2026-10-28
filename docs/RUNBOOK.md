# Running Paper 2

Commands below run in this repository with its virtual environment activated. Model commands load weights and consume the hardware on which they are invoked. Run those on the approved Colab/GPU environment. The repository itself never provisions compute. Keep every base/adapter comparison on the same immutable model revision, dataset manifest and prompt format.

## 1. Generate and calibrate

```bash
paper2 generate --output data/generated/pilot-v3
paper2 verify-data --data data/generated/pilot-v3
paper2 audit-shortcuts --data data/generated/pilot-v3 --output runs/shortcut-audit.json
paper2 audit-tokenizer \
  --input data/generated/pilot-v3/evaluation/development__*.jsonl \
  --model Qwen/Qwen2.5-3B-Instruct --output runs/cue-tokenizer-audit.json
paper2 init-gates --output runs/review.json
paper2 evaluate --input data/generated/pilot-v3/evaluation/development__clean.jsonl \
  --reader base --output runs/base-development.jsonl
paper2 analyze accuracy --input runs/base-development.jsonl --output runs/base-development-summary.json
```

Use a new schema-v3 corpus. G0 re-verifies all files and the stratified symbolic audits; an old frozen approval or an insufficient smoke corpus is rejected. Inspect first/last/initial/cue rule accuracy for each depth/event kind, and both signs of the nested/first-order relations. Different-from-both has a six-label independence expectation of 25/36, not 1/2; the candidate-exclusion classifiers must remain near 1/6. Nondegenerate equality and disagreement are required in each sufficiently populated depth-two stratum. These checks do not replace model difficulty calibration.

Inspect development direct accuracy and failure types. Set difficulty with the generator's `--depths`, `--distractor-events` and `--hidden-move` options. If recalibration is needed, generate a new directory; do not overwrite a frozen corpus or tune against confirmation stories. Record the final corpus configuration, selected component layers, matching target/tolerance and gate precision criteria in a settings JSON. The proposal does not prescribe numerical thresholds for “clearly” or “precisely enough”; state them before inspecting treatment outcomes.

After the investigator's actual G0 review:

```bash
paper2 freeze-calibration --review runs/review.json \
  --manifest data/generated/pilot-v3/manifest.json --settings runs/calibration-settings.json \
  --evidence runs/base-development-summary.json --reviewer 'REVIEWER NAME' \
  --rationale 'RECORD THE ACTUAL DEVELOPMENT CALIBRATION DECISION'
```

## 2. Profile, then run the pilot

On the intended GPU, run the disposable profiling arm. The supplied five-step configuration is an initial check; increase its duration until timings represent steady-state generation, update and checkpoint behavior. Record the actual GPU and use occupied GPU time for costs.

```bash
paper2 profile --config configs/profile_3b.json \
  --input data/generated/pilot-v3/training/clean.jsonl \
  --development data/generated/pilot-v3/evaluation/development__clean.jsonl \
  --corpus-manifest data/generated/pilot-v3/manifest.json \
  --gates runs/review.json --output runs/profile-3b

paper2 train --config configs/pilot/reasoning_rl_clean_seed1.json \
  --input data/generated/pilot-v3/training/clean.jsonl \
  --development data/generated/pilot-v3/evaluation/development__clean.jsonl \
  --corpus-manifest data/generated/pilot-v3/manifest.json \
  --gates runs/review.json --profile-report runs/profile-3b/profile_report.json \
  --output runs/pilot-reasoning-rl
```

Repeat with the answer-RL and label-SFT pilot configurations, starting each from the base model. Keep the generated profile's memory/generation arrangement fixed. The profile adapter is not a starting checkpoint for any arm. Resume interruptions using `--resume-from-checkpoint runs/ARM/checkpoint-N` with the original output directory, configuration and data.

## 3. Test 1 and matched accuracy

```bash
paper2 evaluate --input data/generated/pilot-v3/evaluation/development__clean.jsonl \
  --adapter runs/pilot-reasoning-rl/adapter --training-seed 1 --reader trained --output runs/trained-immediate.jsonl
paper2 evaluate --input data/generated/pilot-v3/evaluation/development__clean.jsonl \
  --adapter runs/pilot-reasoning-rl/adapter --training-seed 1 --reader trained --reasoning --output runs/trained-native.jsonl
paper2 analyze transfer --baseline runs/base-development.jsonl \
  --input runs/trained-immediate.jsonl --output runs/no-trace-transfer.json
```

Run immediate and native formats for every checkpoint and base. For answer-only arms, use `--format native_answer` to score the native answer completion (up to four tokens, matching the RL completion format); the default immediate test always scores exactly one unconstrained next token. For reasoning arms, `--reasoning` is equivalent to `--format native_reasoning`.

For cue analyses, evaluate the agree, independent and conflict shards with the same checkpoint, combine their JSONL rows, then use `paper2 analyze cues`. Cue conditions are defined relative to each world's rewarded direct-belief answer and remain fixed across all eight views. Do not collapse the six independent or five conflict renderings into additional independent stories. The analysis averages them within each world/view and reports direct and per-view paired cue contrasts separately. A labelled pooled direct-anchored cue effect is secondary.

Training writes `checkpoint_evaluations.jsonl` directly, containing native and immediate metrics, immutable adapter paths/hashes, arm, training dataset, seed, and step. Matching selects immediate direct accuracy separately for every arm/dataset/seed and verifies saved adapter weights. Every evaluated step, including step zero, has an adapter even when resumable trainer checkpoints rotate. Combine these files across runs and supply the preregistered target and tolerance:

```bash
paper2 analyze match --input runs/development-checkpoints.jsonl \
  --target-accuracy 0.70 --tolerance 0.02 --output runs/matched-checkpoints.json
```

The numbers above are an example invocation, not predetermined scientific thresholds. Arms with no checkpoint in tolerance have no matched comparison; do not silently extrapolate or select on confirmation results.

## 4. Test 2

Create both trace sources on the same evaluation rows:

```bash
paper2 traces --input data/generated/pilot-v3/evaluation/development__clean.jsonl \
  --source base --output runs/traces-base.jsonl
paper2 traces --input data/generated/pilot-v3/evaluation/development__clean.jsonl \
  --adapter runs/pilot-reasoning-rl/adapter --training-seed 1 --source trained --output runs/traces-trained.jsonl
paper2 prepare-edits --traces runs/traces-base.jsonl runs/traces-trained.jsonl \
  --input data/generated/pilot-v3/evaluation/development__clean.jsonl \
  --max-worlds 12 --seed 2026 --output runs/trace-edits.json
```

Review the frozen subset manifest: by default 12 worlds selected by seeded world-ID rank, never by performance. Shared direct-question traces use one edit contract across views. Each selected source artifact needs a reviewed paraphrase, an intermediate-fact edit with predicted consequences, and a comparable irrelevant edit. Belief summaries are generated automatically from the symbolic base state. See [trace protocol](TRACES.md). Missing, stale or unreviewed selected edits fail closed; `--controls basic` supports an early diagnostic but cannot complete G3.

Generate edited continuations once with **each source checkpoint**, before either reader:

```bash
paper2 regenerate-traces --input data/generated/pilot-v3/evaluation/development__clean.jsonl \
  --traces runs/traces-base.jsonl --source base --edits runs/trace-edits.json \
  --output runs/regenerated-base.jsonl
paper2 regenerate-traces --input data/generated/pilot-v3/evaluation/development__clean.jsonl \
  --traces runs/traces-trained.jsonl --source trained --edits runs/trace-edits.json \
  --adapter runs/pilot-reasoning-rl/adapter --training-seed 1 --output runs/regenerated-trained.jsonl
```

```bash
paper2 crossed --input data/generated/pilot-v3/evaluation/development__clean.jsonl \
  --traces runs/traces-base.jsonl runs/traces-trained.jsonl --edits runs/trace-edits.json \
  --regenerated runs/regenerated-base.jsonl runs/regenerated-trained.jsonl \
  --reader base --output runs/crossed-base.jsonl
paper2 crossed --input data/generated/pilot-v3/evaluation/development__clean.jsonl \
  --traces runs/traces-base.jsonl runs/traces-trained.jsonl --edits runs/trace-edits.json \
  --regenerated runs/regenerated-base.jsonl runs/regenerated-trained.jsonl \
  --adapter runs/pilot-reasoning-rl/adapter --training-seed 1 --reader trained --output runs/crossed-trained.jsonl
```

For the second timing condition, collect traces with `--timing direct_fixed` and repeat the crossing using new artifact/edit files. A native direct-question trace is written before later questions/events are shown and held fixed across all views; no novel story-summary prompt is used. The old `prequestion` name aliases this corrected condition. Raw-token replay and re-rendered replay have separate eligibility and match rates.

Run one base/trained seed pair at a time. `crossed` infers `comparison_seed` from the trained source trace, or accepts an explicit `--comparison-seed`. It records reader and source seeds separately. Combine the three paired result files for the confirmatory analysis; shared base baselines are reused and fixed seeds are averaged within each story. Bootstrap intervals reflect stories, not a population of training seeds.

Combine both readers' immediate baseline files into `runs/baselines.jsonl`, and both crossed files into `runs/crossed.jsonl`, then:

```bash
paper2 analyze crossed --baseline runs/baselines.jsonl \
  --input runs/crossed.jsonl --output runs/crossed-summary.json
```

## 5. Test 3

Use a simultaneous full-depth mask as the primary condition, or supply an explicitly preregistered architecture-appropriate band with `--layers`. The full mask adapts to the loaded model; it does not copy the 72B model’s block numbers into 3B/7B:

```bash
paper2 components --input data/generated/pilot-v3/pairs/development.jsonl \
  --all-layers --output runs/components-base.jsonl
paper2 components --input data/generated/pilot-v3/pairs/development.jsonl \
  --adapter runs/pilot-reasoning-rl/adapter --training-seed 1 --all-layers --output runs/components-trained.jsonl
paper2 analyze components --baseline runs/components-base.jsonl \
  --input runs/components-trained.jsonl --output runs/component-changes.json
```

All selected layers are swapped **simultaneously** for K, V and residual separately (`--layer-mode joint`, the default). Freeze the chosen full path or band before outcomes. `--layer-mode both` adds single-layer diagnostics without replacing the joint intervention. K/V capture is after projection/bias and before rotary application, preserving the model's native GQA/RoPE computation. The residual site is the decoder block output. Receipts include exact positions, heads, source checkpoint and payload hashes. The CLI calls `score_consequence_pattern` and reports changed/intact accuracy, joint correctness, before/after effects and view-specific probability contrasts. Required-change masks come from the oracle for each matched world, not question names. The ordinary Test 1 belief-edit scope is a distinct definition. Review positive controls and nulls before interpreting small effects.

Secondary cross-model swaps are available with `components --donor-model MODEL --donor-adapter PATH` (and optional `--donor-revision`). Donor and recipient run the identical recipient story, with full-token-input and answer-alphabet identity checks. These rows carry `analysis: secondary_cross_model` and ordinary task targets; primary matched-event analysis excludes them. Keep them in separately named files; do not interpret their effects as event rewrites.

## 6. External check and confirmatory runs

Download `data/opentom.json` from the [official OpenToM repository](https://github.com/seacowx/OpenToM/tree/3f22b66276b2d7ca5fe573c28c79cc0d077aafc5) using `python scripts/fetch_opentom.py`, which verifies the pinned checksum. OpenToM is evaluation-only and does not enter training or difficulty calibration.

```bash
paper2 opentom-import --input data/external/opentom.json \
  --output data/external/opentom-evaluation.jsonl \
  --source-url https://github.com/seacowx/OpenToM/blob/3f22b66276b2d7ca5fe573c28c79cc0d077aafc5/data/opentom.json
paper2 opentom-evaluate --input data/external/opentom-evaluation.jsonl \
  --output runs/opentom-base.jsonl
paper2 analyze opentom --input runs/opentom-base.jsonl --output runs/opentom-summary.json
```

Repeat with adapters and `--reasoning` for native reasoning formats. This external test supplies question-type answer options and reports option-aware, type-specific scoring, invalid answers and upstream-compatible diagnostics. Raw outputs are retained. The shared chat wrapper is an explicit adaptation, not an exact reproduction of all published OpenToM prompting choices; see [OpenToM protocol](OPENTOM.md).

After reviewing G0–G4 and recording approved cost, use `configs/confirmatory/` for the full 24-run grid, set `--model Qwen/Qwen2.5-7B-Instruct` for evaluation, and use confirmation shards. Reasoning SFT takes `supervision/clean_reasoning_sft.jsonl` or its shortcut twin; the other arms take `training/*.jsonl`. Retain all three seeds. A second 7B profiling run can use a confirmatory config with `profile: true`, after paid approval, and the 3B profile as initial evidence; use the resulting target-size profile for the main 7B runs.
