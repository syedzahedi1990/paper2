# Running Paper 2

Commands below run in this repository with its virtual environment activated. Model commands load weights and consume the hardware on which they are invoked. Run those on the approved Colab/GPU environment. The repository itself never provisions compute. Keep every base/adapter comparison on the same immutable model revision, dataset manifest and prompt format.

## 1. Generate and calibrate

```bash
paper2 generate --output data/generated/pilot-audit-v4
paper2 verify-data --data data/generated/pilot-audit-v4
paper2 audit-shortcuts --data data/generated/pilot-audit-v4 --output runs/shortcut-audit.json
paper2 audit-tokenizer \
  --input data/generated/pilot-audit-v4/evaluation/development__*.jsonl \
  --model Qwen/Qwen2.5-3B-Instruct --output runs/cue-tokenizer-audit.json
paper2 init-gates --output runs/review.json
paper2 evaluate --input data/generated/pilot-audit-v4/evaluation/development__clean.jsonl \
  --reader base --output runs/base-development.jsonl
paper2 analyze accuracy --input runs/base-development.jsonl --output runs/base-development-summary.json
```

Use a new schema-v3 corpus with audit v4. G0 re-verifies all files and the stratified symbolic audits; an old frozen approval or an insufficient smoke corpus is rejected. Inspect first/last/initial/cue rule accuracy for each depth/event kind, and both signs of the nested/first-order relations. Different-from-both has a six-label independence expectation of 25/36, not 1/2; the candidate-exclusion classifiers must remain near 1/6. Nondegenerate equality and disagreement are required in each sufficiently populated depth-two stratum. Also inspect every event-position rule, location-frequency rules and depth-one last-joint-observation accuracy. The default distribution has above-chance weak heuristics; the audit preserves these failed screens. Depth-two last-joint-observation is an intended semantic solver. These checks do not replace model difficulty calibration.

Inspect development direct accuracy and failure types. Set difficulty with the generator's `--depths`, `--distractor-events` and `--hidden-move` options. If recalibration is needed, generate a new directory; do not overwrite a frozen corpus or tune against confirmation stories. Record the final corpus configuration, selected component layers, matching target/tolerance and gate precision criteria in a settings JSON. Declare the contiguous interior G4 test band as `g4_test_layers`, for example `[6, 7, 8, 9, 10, 11]`; the later K, V and residual control must use that exact band. The proposal does not prescribe numerical thresholds for “clearly” or “precisely enough”; state them before inspecting treatment outcomes.

The default corpus cannot freeze without an explicit residual-risk decision. If the investigator accepts the documented heuristics after review, settings must include `residual_heuristic_review` with `accepted: true`, the exact current `corpus_hash`, and a substantive `rationale`. The existing named G0 reviewer owns that decision. This permits known residual risks only; failed required controls remain nonwaivable. Old audit-v3 approvals are rejected.

After the investigator's actual G0 review:

```bash
paper2 freeze-calibration --review runs/review.json \
  --manifest data/generated/pilot-audit-v4/manifest.json --settings runs/calibration-settings.json \
  --evidence runs/base-development-summary.json --reviewer 'REVIEWER NAME' \
  --rationale 'RECORD THE ACTUAL DEVELOPMENT CALIBRATION DECISION'
```

## 2. Profile, then run the pilot

On the intended GPU, run the disposable profiling arm. The supplied five-step configuration is an initial check; increase its duration until timings represent steady-state generation, update and checkpoint behavior. Record the actual GPU and use occupied GPU time for costs.

```bash
paper2 profile --config configs/profile_3b.json \
  --input data/generated/pilot-audit-v4/training/clean.jsonl \
  --development data/generated/pilot-audit-v4/evaluation/development__clean.jsonl \
  --corpus-manifest data/generated/pilot-audit-v4/manifest.json \
  --gates runs/review.json --output runs/profile-3b

paper2 train --config configs/pilot/reasoning_rl_clean_seed1.json \
  --input data/generated/pilot-audit-v4/training/clean.jsonl \
  --development data/generated/pilot-audit-v4/evaluation/development__clean.jsonl \
  --corpus-manifest data/generated/pilot-audit-v4/manifest.json \
  --gates runs/review.json --profile-report runs/profile-3b/profile_report.json \
  --output runs/pilot-reasoning-rl
```

The shipped RL schedule measures immediate-answer accuracy and retains an adapter at steps 0, 50, …, 2,000 (41 checkpoints), while native traces run only at 0, 250, …, 2,000 (9 panels). SFT uses 25 steps for both formats; the short profile uses 5. Both schedules include off-interval final steps. The profile records separate native/immediate evaluation times and panel counts, generated tokens and snapshot bytes; combined-only older reports cannot authorize training. For 120 development stories, each default RL run makes 4,920 immediate forward passes and 1,080 native generations. The pilot retains 93 evaluated adapters and the confirmatory grid 624, in addition to rotating optimizer checkpoints and final adapters. The full-pool projection is for the same arm and model only. The proposal’s 15–25 hours for the whole pilot remains unverified. Account for all arms, sequential native evaluation, storage, control generation and later Tests 1–3 before approving compute.

Repeat with the answer-RL and label-SFT pilot configurations, starting each from the base model. Keep the generated profile's memory/generation arrangement fixed. The profile adapter is not a starting checkpoint for any arm. Resume interruptions using `--resume-from-checkpoint runs/ARM/checkpoint-N` with the original output directory, configuration and data.

## 3. Test 1 and matched accuracy

```bash
paper2 evaluate --input data/generated/pilot-audit-v4/evaluation/development__clean.jsonl \
  --adapter runs/pilot-reasoning-rl/adapter --training-seed 1 --reader trained --output runs/trained-immediate.jsonl
paper2 evaluate --input data/generated/pilot-audit-v4/evaluation/development__clean.jsonl \
  --adapter runs/pilot-reasoning-rl/adapter --training-seed 1 --reader trained --reasoning --output runs/trained-native.jsonl
paper2 analyze transfer --baseline runs/base-development.jsonl \
  --input runs/trained-immediate.jsonl --output runs/no-trace-transfer.json
```

Run the requested Test 1 formats for the preregistered analysis checkpoints and base. During learning, keep the separate immediate/native schedules above; do not add native generations at every cheap matching checkpoint. For answer-only arms, use `--format native_answer` to score the native answer completion (up to four tokens, matching the RL completion format); the default immediate test always scores exactly one unconstrained next token. For reasoning arms, `--reasoning` is equivalent to `--format native_reasoning`.

For cue analyses, evaluate the agree, independent and conflict shards with the same checkpoint, combine their JSONL rows, then use `paper2 analyze cues`. Cue conditions are defined relative to each world's rewarded direct-belief answer and remain fixed across all eight views. Do not collapse the six independent or five conflict renderings into additional independent stories. The analysis averages them within each world/view and reports direct and per-view paired cue contrasts separately. A labelled pooled direct-anchored cue effect is secondary.

Training writes `checkpoint_evaluations.jsonl` directly, containing immediate metrics every 50 RL steps and native metrics every 250, immutable adapter paths/hashes, arm, training dataset, seed, and step. G1 updates only at measured native steps. Matching selects immediate direct accuracy separately for every arm/dataset/seed and verifies saved adapter weights. Every evaluated step, including step zero, has an adapter even when resumable trainer checkpoints rotate. Combine these files across runs and supply the preregistered target and tolerance:

```bash
paper2 analyze match --input runs/development-checkpoints.jsonl \
  --target-accuracy 0.70 --tolerance 0.02 --output runs/matched-checkpoints.json
```

The numbers above are an example invocation, not predetermined scientific thresholds. Arms with no checkpoint in tolerance have no matched comparison; do not silently extrapolate or select on confirmation results.

## 4. Test 2

The primary confirmatory edit measurement uses a frozen direct-question trace (`direct_fixed`) across all eight views. Create both pilot trace sources on the same complete evaluation panel:

```bash
paper2 traces --input data/generated/pilot-audit-v4/evaluation/development__clean.jsonl \
  --source base --timing direct_fixed --output runs/traces-base.jsonl
paper2 traces --input data/generated/pilot-audit-v4/evaluation/development__clean.jsonl \
  --adapter runs/pilot-reasoning-rl/adapter --training-seed 1 --source trained \
  --timing direct_fixed --output runs/traces-trained.jsonl
paper2 prepare-edits --traces runs/traces-base.jsonl runs/traces-trained.jsonl \
  --input data/generated/pilot-audit-v4/evaluation/development__clean.jsonl \
  --stage pilot --max-worlds 12 --seed 2026 --output runs/edit-templates.json
```

The pilot defaults to 12 worlds selected by seeded world-ID rank, never by performance. For confirmation, use confirmation rows, all intended source files, and `--stage confirmatory --precision-half-width .10` without `--max-worlds`: the approximate rate-precision plan selects 97 worlds and requires complete eight-view `direct_fixed` panels for every source. Actual story intervals must also pass; the plan does not size arm differences or remaining-error ratios. Include the base once in the full-grid manifest. For 24 trained checkpoints plus the base, this gives 2,425 contracts instead of 19,400 question-dependent contracts; additional checkpoints or cue renderings increase the workload.

The oracle proposes the focal-belief change and all eight consequences. A fixed editor generates paraphrases, comparable irrelevant edits and the exact span for the proposed fact edit. Freeze its model revision, prompt, decoding and human audit sample before generation. The editor below is an example requiring pilot validation, not an assurance of adequate editing quality:

```bash
paper2 plan-edit-generation --edits runs/edit-templates.json \
  --model Qwen/Qwen2.5-7B-Instruct --revision a09a35458c702b33eeacc393d103063234e8bc28 \
  --audit-sample-size 60 --audit-seed 2026 --output runs/edit-generation-plan.json
paper2 generate-edits --edits runs/edit-templates.json --plan runs/edit-generation-plan.json \
  --traces runs/traces-base.jsonl runs/traces-trained.jsonl --output runs/generated-edits.json
paper2 prepare-edit-audit --edits runs/generated-edits.json --output runs/edit-audit.json
```

A human completes the preregistered audit: 60 globally sampled contracts by default, plus source/checkpoint coverage supplements; a smaller batch is reviewed in full. Check paraphrase meaning, the intended factual substitution, irrelevant-edit neutrality and the oracle forecasts. Record the named reviewers, each required check and explicit batch acceptance, then verify it:

```bash
paper2 verify-edit-audit --edits runs/generated-edits.json --audit runs/edit-audit.json \
  --output runs/edit-audit-evidence.json
```

Unsampled contracts remain individually unapproved; this is sampled quality assurance for the frozen batch. Any unresolved output or failed/missing audit item blocks the batch. Do not drop difficult traces or retry until the audit passes. Interrupted generation can reuse the exact completed journal with `generate-edits --resume`; unresolved completed outputs are not regenerated. See [model-generated edit protocol](EDIT_GENERATION.md) for the fixed prompt, provenance, sampling assumptions and failure policy. The individually reviewed path remains available for pilot diagnostics. Full-state belief summaries are positive controls for use of a correct premise, not model-produced summaries; see [trace protocol](TRACES.md). `--controls basic` supports an early diagnostic but cannot complete G3.

Generate edited continuations once with **each original source checkpoint**, before either reader, using the same generated batch and completed audit:

```bash
paper2 regenerate-traces --input data/generated/pilot-audit-v4/evaluation/development__clean.jsonl \
  --traces runs/traces-base.jsonl --source base --edits runs/generated-edits.json \
  --edit-audit runs/edit-audit.json --output runs/regenerated-base.jsonl
paper2 regenerate-traces --input data/generated/pilot-audit-v4/evaluation/development__clean.jsonl \
  --traces runs/traces-trained.jsonl --source trained --edits runs/generated-edits.json \
  --edit-audit runs/edit-audit.json --adapter runs/pilot-reasoning-rl/adapter --training-seed 1 \
  --output runs/regenerated-trained.jsonl

paper2 crossed --input data/generated/pilot-audit-v4/evaluation/development__clean.jsonl \
  --traces runs/traces-base.jsonl runs/traces-trained.jsonl --edits runs/generated-edits.json \
  --edit-audit runs/edit-audit.json \
  --regenerated runs/regenerated-base.jsonl runs/regenerated-trained.jsonl \
  --reader base --output runs/crossed-base.jsonl
paper2 crossed --input data/generated/pilot-audit-v4/evaluation/development__clean.jsonl \
  --traces runs/traces-base.jsonl runs/traces-trained.jsonl --edits runs/generated-edits.json \
  --edit-audit runs/edit-audit.json \
  --regenerated runs/regenerated-base.jsonl runs/regenerated-trained.jsonl \
  --adapter runs/pilot-reasoning-rl/adapter --training-seed 1 --reader trained --output runs/crossed-trained.jsonl
```

Both readers consume identical saved control text and continuations. The direct trace is written before later questions/events are shown and held fixed across views; no novel story-summary prompt is used. `prequestion` remains an alias for `direct_fixed`. Question-dependent timing is a pilot or supplemental diagnostic: collect it with `--timing question_dependent` and separate artifact/edit files. It cannot replace the primary confirmatory edit measurement. Raw-token replay and re-rendered replay retain separate eligibility and match rates.

Run one base/trained seed pair at a time. `crossed` infers `comparison_seed` from the trained source trace, or accepts an explicit `--comparison-seed`. It records reader and source seeds separately. Combine the three paired result files for the confirmatory analysis; shared base baselines are reused and fixed seeds are averaged within each story. Bootstrap intervals reflect stories, not a population of training seeds.

Combine both readers' immediate baseline files into `runs/baselines.jsonl`, and both crossed files into `runs/crossed.jsonl`, then:

```bash
paper2 analyze crossed --baseline runs/baselines.jsonl \
  --input runs/crossed.jsonl --output runs/crossed-summary.json
```

## 5. Test 3

Preregister an architecture-appropriate contiguous test band and use that **same band for joint K, V and residual swaps**. `--g4-test-layers` declares this alignment; `--residual-control-layers` adds diagnostics only and cannot certify G4. The numerical band below is an example requiring preregistration and measured validation, not a claim of successful G4 or a transfer of Paper 1's 72B layer numbers to 3B/7B:

```bash
paper2 components --input data/generated/pilot-audit-v4/pairs/development.jsonl \
  --layers 6 7 8 9 10 11 --g4-test-layers 6 7 8 9 10 11 \
  --output runs/components-base.jsonl
paper2 components --input data/generated/pilot-audit-v4/pairs/development.jsonl \
  --adapter runs/pilot-reasoning-rl/adapter --training-seed 1 \
  --layers 6 7 8 9 10 11 --g4-test-layers 6 7 8 9 10 11 \
  --output runs/components-trained.jsonl
paper2 analyze components --baseline runs/components-base.jsonl \
  --input runs/components-trained.jsonl --output runs/component-changes.json
```

All selected layers are swapped **simultaneously** for K, V and residual separately (`--layer-mode joint`, the default). A full-depth K/V experiment remains available with `--all-layers`; supplying `--g4-test-layers` adds the aligned interior K/V/residual band as well. G4 evidence applies to the declared interior band, not an unrelated mask. `--layer-mode both` adds single-layer diagnostics without replacing the joint intervention. A layer-1 residual diagnostic cannot stand in for a test band beginning at layer 6.

K/V capture is after projection/bias and before rotary application, preserving native GQA/RoPE computation. The residual site is the decoder block output. Receipts include exact positions, heads, source checkpoint and payload hashes. Analysis calls `score_consequence_pattern` and reports changed/intact accuracy, joint correctness, before/after effects and view-specific probability contrasts. Required-change masks come from the oracle for each matched world; ordinary Test 1 belief-edit scope is a distinct definition.

Freeze `settings.g4_test_layers` before outcomes. G4's selected raw evidence must include the exact same list in `positive_control.test_layers`, alongside `path`, `sha256`, `mask_id` and `checkpoint_id`. For the example, `test_layers` is `[6, 7, 8, 9, 10, 11]` and `mask_id` is `"6,7,8,9,10,11"`. Include that same component JSONL in the gate's hashed evidence. The gate requires residual and K/V receipts aligned to the frozen band, complete natural/own/null panels and a positive changed-consequence residual effect. Masks including layer zero, final-block-only masks without downstream attention, and mismatched or undeclared bands are ineligible regardless of score. A reviewer still evaluates intact consequences, nulls and precision before interpreting small effects; software tests do not confer a G4 pass.

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
