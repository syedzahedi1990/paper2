# Running Paper 2

Commands below run in this repository with its virtual environment activated. Model commands load weights and consume the hardware on which they are invoked. Run those on the approved Colab/GPU environment. The repository itself never provisions compute. Keep every base/adapter comparison on the same immutable model revision, dataset manifest and prompt format.

## 1. Generate and calibrate

```bash
paper2 generate --output data/generated/pilot
paper2 verify-data --data data/generated/pilot
paper2 audit-tokenizer \
  --input data/generated/pilot/evaluation/development__*.jsonl \
  --model Qwen/Qwen2.5-3B-Instruct --output runs/cue-tokenizer-audit.json
paper2 init-gates --output runs/review.json
paper2 evaluate --input data/generated/pilot/evaluation/development__clean.jsonl \
  --reader base --output runs/base-development.jsonl
paper2 analyze accuracy --input runs/base-development.jsonl --output runs/base-development-summary.json
```

Inspect development direct accuracy and failure types. Set difficulty with the generator's `--depths`, `--distractor-events` and `--hidden-move` options. If recalibration is needed, generate a new directory; do not overwrite a frozen corpus or tune against confirmation stories. Record the final corpus configuration, selected component layers, matching target/tolerance and gate precision criteria in a settings JSON. The proposal does not prescribe numerical thresholds for “clearly” or “precisely enough”; state them before inspecting treatment outcomes.

After the investigator's actual G0 review:

```bash
paper2 freeze-calibration --review runs/review.json \
  --manifest data/generated/pilot/manifest.json --settings runs/calibration-settings.json \
  --evidence runs/base-development-summary.json --reviewer 'REVIEWER NAME' \
  --rationale 'RECORD THE ACTUAL DEVELOPMENT CALIBRATION DECISION'
```

## 2. Profile, then run the pilot

On the intended GPU, run the disposable profiling arm. The supplied five-step configuration is an initial check; increase its duration until timings represent steady-state generation, update and checkpoint behavior. Record the actual GPU and use occupied GPU time for costs.

```bash
paper2 profile --config configs/profile_3b.json \
  --input data/generated/pilot/training/clean.jsonl \
  --development data/generated/pilot/evaluation/development__clean.jsonl \
  --corpus-manifest data/generated/pilot/manifest.json \
  --gates runs/review.json --output runs/profile-3b

paper2 train --config configs/pilot/reasoning_rl_clean_seed1.json \
  --input data/generated/pilot/training/clean.jsonl \
  --development data/generated/pilot/evaluation/development__clean.jsonl \
  --corpus-manifest data/generated/pilot/manifest.json \
  --gates runs/review.json --profile-report runs/profile-3b/profile_report.json \
  --output runs/pilot-reasoning-rl
```

Repeat with the answer-RL and label-SFT pilot configurations, starting each from the base model. Keep the generated profile's memory/generation arrangement fixed. The profile adapter is not a starting checkpoint for any arm. Resume interruptions using `--resume-from-checkpoint runs/ARM/checkpoint-N` with the original output directory, configuration and data.

## 3. Test 1 and matched accuracy

```bash
paper2 evaluate --input data/generated/pilot/evaluation/development__clean.jsonl \
  --adapter runs/pilot-reasoning-rl/adapter --reader trained --output runs/trained-immediate.jsonl
paper2 evaluate --input data/generated/pilot/evaluation/development__clean.jsonl \
  --adapter runs/pilot-reasoning-rl/adapter --reader trained --reasoning --output runs/trained-native.jsonl
paper2 analyze transfer --baseline runs/base-development.jsonl \
  --input runs/trained-immediate.jsonl --output runs/no-trace-transfer.json
```

Run immediate and native formats for every checkpoint and base. For answer-only arms, use `--format native_answer` to score the native answer completion (up to four tokens, matching the RL completion format); the default immediate test always scores exactly one unconstrained next token. For reasoning arms, `--reasoning` is equivalent to `--format native_reasoning`.

For cue analyses, evaluate the agree, independent and conflict shards with the same checkpoint, combine their JSONL rows, then use `paper2 analyze cues`. Cue conditions are defined relative to each world's rewarded direct-belief answer and remain fixed across all eight views. Do not collapse the six independent or five conflict renderings into additional independent stories. The analysis averages them within each world/view for the paired cue contrast.

Accuracy matching accepts JSONL checkpoint records with `arm`, `checkpoint`, `step`, `split: development`, `format: immediate`, `view: direct`, and `accuracy`. Supply the preregistered target and tolerance:

```bash
paper2 analyze match --input runs/development-checkpoints.jsonl \
  --target-accuracy 0.70 --tolerance 0.02 --output runs/matched-checkpoints.json
```

The numbers above are an example invocation, not predetermined scientific thresholds. Arms with no checkpoint in tolerance have no matched comparison; do not silently extrapolate or select on confirmation results.

## 4. Test 2

Create both trace sources on the same evaluation rows:

```bash
paper2 traces --input data/generated/pilot/evaluation/development__clean.jsonl \
  --source base --output runs/traces-base.jsonl
paper2 traces --input data/generated/pilot/evaluation/development__clean.jsonl \
  --adapter runs/pilot-reasoning-rl/adapter --source trained --output runs/traces-trained.jsonl
paper2 prepare-edits --traces runs/traces-base.jsonl runs/traces-trained.jsonl \
  --output runs/trace-edits.json
```

Review the edit manifest. Each source trace needs a meaning-preserving paraphrase; one intermediate-fact edit with exact character span, old/new text and predicted answer for each tested consequence; a comparable irrelevant edit; and a task-grounded belief summary. Record the reviewer and approve only inspected edits. Do not invent an effect or relabel resistance as implementation failure. Missing, stale or unreviewed edits fail closed. `--controls basic` supports an early crossing/replay/answer-cue/no-trace diagnostic but cannot complete G3.

```bash
paper2 crossed --input data/generated/pilot/evaluation/development__clean.jsonl \
  --traces runs/traces-base.jsonl runs/traces-trained.jsonl --edits runs/trace-edits.json \
  --reader base --output runs/crossed-base.jsonl
paper2 crossed --input data/generated/pilot/evaluation/development__clean.jsonl \
  --traces runs/traces-base.jsonl runs/traces-trained.jsonl --edits runs/trace-edits.json \
  --adapter runs/pilot-reasoning-rl/adapter --reader trained --output runs/crossed-trained.jsonl
```

For the second timing condition, collect traces with `--timing prequestion` and repeat the crossing. The summary is written before later questions/events are shown and held fixed across all views of each rendered world. Keep timing conditions distinct. Natural own-replay eligibility and failures are reported separately from forced-delimiter continuations.

Combine both readers' immediate baseline files into `runs/baselines.jsonl`, and both crossed files into `runs/crossed.jsonl`, then:

```bash
paper2 analyze crossed --baseline runs/baselines.jsonl \
  --input runs/crossed.jsonl --output runs/crossed-summary.json
```

## 5. Test 3

Supply layers selected and frozen during calibration. The example layer numbers below are illustrative zero-based indices, not a result of a layer search:

```bash
paper2 components --input data/generated/pilot/pairs/development.jsonl \
  --layers 8 16 24 --output runs/components-base.jsonl
paper2 components --input data/generated/pilot/pairs/development.jsonl \
  --adapter runs/pilot-reasoning-rl/adapter --layers 8 16 24 --output runs/components-trained.jsonl
paper2 analyze components --baseline runs/components-base.jsonl \
  --input runs/components-trained.jsonl --output runs/component-changes.json
```

Each selected layer includes K, V and residual sites, evaluated separately. K/V capture is after projection/bias and before rotary application, preserving the model's native GQA/RoPE computation. The residual site is the decoder block output. Receipts include exact positions, heads, source checkpoint and payload hashes. `score_consequence_pattern` computes whole-story forecast correctness, required changes, intact views and joint outcomes; `compare_checkpoints` returns paired changes in natural-minus-own target probabilities. Review positive controls and nulls before interpreting small effects.

Cross-model swaps are a separate library API (`ComponentInterventionEngine.swap(..., secondary_cross_model=True)`) and require identical token prefixes. They are never mixed into the primary CLI analysis.

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

Repeat with adapters and `--reasoning` for native reasoning formats. This external test retains free-response labels and reports normalized exact match by question type, separately from single-token controlled-world accuracy. Inspect raw outputs to diagnose formatting errors; do not silently judge ambiguous answers as correct.

After reviewing G0–G4 and recording approved cost, use `configs/confirmatory/` for the full 24-run grid, set `--model Qwen/Qwen2.5-7B-Instruct` for evaluation, and use confirmation shards. Reasoning SFT takes `supervision/clean_reasoning_sft.jsonl` or its shortcut twin; the other arms take `training/*.jsonl`. Retain all three seeds. A second 7B profiling run can use a confirmatory config with `profile: true`, after paid approval, and the 3B profile as initial evidence; use the resulting target-size profile for the main 7B runs.
