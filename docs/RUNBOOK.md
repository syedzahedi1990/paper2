# Running Paper 2

Commands below run in this repository with its virtual environment activated. Model commands load weights and consume the hardware on which they are invoked. Run those on the approved Colab/GPU environment. The repository itself never provisions compute. Keep every base/adapter comparison on the same immutable model revision, dataset manifest and prompt format.

## 1. Generate and calibrate

```bash
paper2 generate --output data/generated/pilot-audit-v5
paper2 verify-data --data data/generated/pilot-audit-v5
paper2 audit-shortcuts --data data/generated/pilot-audit-v5 --output runs/shortcut-audit.json
paper2 audit-tokenizer \
  --input data/generated/pilot-audit-v5/evaluation/development__*.jsonl \
  --model Qwen/Qwen2.5-3B-Instruct --output runs/cue-tokenizer-audit.json
paper2 init-gates --output runs/review.json
paper2 evaluate --input data/generated/pilot-audit-v5/evaluation/development__clean.jsonl \
  --reader base --output runs/base-development.jsonl
paper2 analyze accuracy --input runs/base-development.jsonl --output runs/base-development-summary.json
```

Use a new schema-v3 corpus with audit v5. G0 re-verifies file hashes, split separation, exact cue construction, declared relation quotas and complete panels. The statistical audit is descriptive: inspect every first/last/initial/cue, event-position, location-frequency, joint-observation and candidate-exclusion heuristic by depth and critical-event kind, with intervals. Plot heuristic accuracies beside the corresponding model accuracy as reference lines. Nested-answer equality/disagreement rates are relation summaries, not answer-prediction accuracies; show both directions. Depth-two last-joint-observation is an intended semantic solver. No heuristic score is an automatic pass or failure.

Review base development direct accuracy and its whole-story interval against chance (1/6) and ceiling (1). The named G0 reviewer judges whether difficulty leaves room to observe learning and whether the known heuristic behavior permits the planned interpretations. Settings must contain `difficulty_review` with `estimate`, `ci95` and a written `relation_to_chance_and_ceiling`. No numeric margin is invented. Adjust `--depths`, `--distractor-events` or `--hidden-move` only during pilot calibration; if needed generate a new directory and document why, without inspecting confirmation outcomes.

Freeze the final corpus configuration, the pretraining estimands and analyses in [PREREGISTRATION.md](PREREGISTRATION.md), the training settings/optional development-selection rule, editing model/procedure and Test 3 layer policy. The primary Test 3 band is every layer from 1 through the final model layer, resolved for each architecture; there is no layer-6 tuning choice. Freeze `g4_layer_policy: "all_after_embedding_layer"` and the pilot evidence list `g4_test_layers: [1, 2, …, 35]` (write every integer in the actual JSON). The pilot list binds 3B receipts; the policy resolves the 7B run to layers 1–27. Evidence and reviewer decisions must be tied to the frozen artifacts. Exact construction/length/entity matching, leakage, hashes and complete-panel checks remain mandatory integrity requirements. G0–G4 are pilot go/no-go reviews of whether measurements work; confirmation measures and reports the frozen estimands, including null or imprecise results.

After the investigator's actual G0 review:

```bash
paper2 freeze-calibration --review runs/review.json \
  --manifest data/generated/pilot-audit-v5/manifest.json --settings runs/calibration-settings.json \
  --evidence runs/base-development-summary.json --reviewer 'REVIEWER NAME' \
  --rationale 'RECORD THE ACTUAL DEVELOPMENT CALIBRATION DECISION'
```

## 2. Profile, then run the pilot

On the intended GPU, run the disposable profiling arm. The supplied five-step configuration is an initial check; increase its duration until timings represent steady-state generation, update and checkpoint behavior. Record the actual GPU and use occupied GPU time for costs.

```bash
paper2 profile --config configs/profile_3b.json \
  --input data/generated/pilot-audit-v5/training/clean.jsonl \
  --development data/generated/pilot-audit-v5/evaluation/development__clean.jsonl \
  --corpus-manifest data/generated/pilot-audit-v5/manifest.json \
  --gates runs/review.json --output runs/profile-3b

paper2 train --config configs/pilot/reasoning_rl_clean_seed1.json \
  --input data/generated/pilot-audit-v5/training/clean.jsonl \
  --development data/generated/pilot-audit-v5/evaluation/development__clean.jsonl \
  --corpus-manifest data/generated/pilot-audit-v5/manifest.json \
  --gates runs/review.json --profile-report runs/profile-3b/profile_report.json \
  --output runs/pilot-reasoning-rl
```

The shipped RL schedule measures immediate-answer accuracy and retains an adapter at steps 0, 50, …, 2,000 (41 checkpoints), while native traces run only at 0, 250, …, 2,000 (9 panels). SFT uses 25 steps for both formats; the short profile uses 5. Both schedules include off-interval final steps. The profile records separate native/immediate evaluation times and panel counts, generated tokens and snapshot bytes; combined-only older reports cannot authorize training. For 120 development stories, each default RL run makes 4,920 immediate forward passes and 1,080 native generations. The pilot retains 93 evaluated adapters and the confirmatory grid 624, in addition to rotating optimizer checkpoints and final adapters. The full-pool projection is for the same arm and model only. The proposal’s 15–25 hours for the whole pilot remains unverified. Account for all arms, sequential native evaluation, storage, control generation and later Tests 1–3 before approving compute.

Repeat with the answer-RL and label-SFT pilot configurations, starting each from the base model. Keep the generated profile's memory/generation arrangement fixed. The profile adapter is not a starting checkpoint for any arm. Resume interruptions using `--resume-from-checkpoint runs/ARM/checkpoint-N` with the original output directory, configuration and data.

## 3. Test 1 and matched accuracy

```bash
paper2 evaluate --input data/generated/pilot-audit-v5/evaluation/development__clean.jsonl \
  --adapter runs/pilot-reasoning-rl/adapter --training-seed 1 --reader trained --output runs/trained-immediate.jsonl
paper2 evaluate --input data/generated/pilot-audit-v5/evaluation/development__clean.jsonl \
  --adapter runs/pilot-reasoning-rl/adapter --training-seed 1 --reader trained --reasoning --output runs/trained-native.jsonl
paper2 analyze transfer --baseline runs/base-development.jsonl \
  --input runs/trained-immediate.jsonl --output runs/no-trace-transfer.json
```

Run the requested Test 1 formats for the preregistered analysis checkpoints and base. During learning, keep the separate immediate/native schedules above; do not add native generations at every cheap matching checkpoint. For answer-only arms, use `--format native_answer` to score the native answer completion (up to four tokens, matching the RL completion format); the default immediate test always scores exactly one unconstrained next token. For reasoning arms, `--reasoning` is equivalent to `--format native_reasoning`.

For cue analyses, evaluate the agree, independent and conflict shards with the same checkpoint, combine their JSONL rows, then use `paper2 analyze cues`. Cue conditions are defined relative to each world's rewarded direct-belief answer and remain fixed across all eight views. Do not collapse the six independent or five conflict renderings into additional independent stories. The analysis averages them within each world/view and reports direct and per-view paired cue contrasts separately. A labelled pooled direct-anchored cue effect is secondary.

Training writes `checkpoint_evaluations.jsonl` directly, containing immediate metrics every 50 RL steps and native metrics every 250, immutable adapter paths/hashes, arm, training dataset, seed and step. G1 updates only at measured native steps. Every evaluated step, including step zero, retains an adapter even when resumable trainer checkpoints rotate. Combine these files across the preregistered runs:

```bash
paper2 analyze match --input runs/development-checkpoints.jsonl \
  --output runs/matched-checkpoints.json
```

An optional `--expected-runs runs/expected-comparisons.json` declares the required run identities and prevents a missing run silently changing the comparison. The common target is the minimum of the compared runs' maximum development direct accuracies: the highest accuracy each reaches. Select each run's closest saved checkpoint, breaking exact ties by the earlier step. Report selected steps, actual accuracies and deviations from the common target. There is no ±0.02 window, numerical match rejection or interpolated adapter. Never use confirmation accuracy to select checkpoints.

## 4. Test 2

The primary confirmatory edit measurement uses a frozen direct-question trace (`direct_fixed`) across all eight views. Create both pilot trace sources on the same complete evaluation panel:

```bash
paper2 traces --input data/generated/pilot-audit-v5/evaluation/development__clean.jsonl \
  --source base --timing direct_fixed --output runs/traces-base.jsonl
paper2 traces --input data/generated/pilot-audit-v5/evaluation/development__clean.jsonl \
  --adapter runs/pilot-reasoning-rl/adapter --training-seed 1 --source trained \
  --timing direct_fixed --output runs/traces-trained.jsonl
paper2 prepare-edits --traces runs/traces-base.jsonl runs/traces-trained.jsonl \
  --input data/generated/pilot-audit-v5/evaluation/development__clean.jsonl \
  --stage pilot --max-worlds 12 --seed 2026 --output runs/edit-templates.json
```

The pilot defaults to 12 worlds selected by seeded world-ID rank, never by performance. For confirmation, use confirmation rows, every intended source file and `--stage confirmatory` without `--max-worlds`: all 120 candidate worlds are selected and complete eight-view `direct_fixed` panels are required for each source. There is no 97-world requirement or interval-width pass mark. At 24 trained final checkpoints plus a shared base and one clean rendering, the grid has 3,000 contracts rather than 24,000 question-dependent contracts. Additional matched checkpoints or renderings increase workload. Use all outcomes to report editability and conditional edit effects per arm; average fixed seeds within each story before 20,000 whole-story resamples.

The oracle freezes forecasts for each possible stated focal-belief location. The editor locates the final belief actually expressed, including an incorrect belief, and prepares a paraphrase, a comparable irrelevant edit and the targeted substitution. Missing/contradictory/ambiguous beliefs, unavailable safe controls and operational failures stay visible in per-source outcome counts. They have no usable edit controls and remain in candidate denominators. Freeze the editor revision, prompt, decoding and random audit sample before generation. Validate the editing procedure at G3 during the pilot; the example model below has not yet been empirically validated:

```bash
paper2 plan-edit-generation --edits runs/edit-templates.json \
  --model Qwen/Qwen2.5-7B-Instruct --revision a09a35458c702b33eeacc393d103063234e8bc28 \
  --audit-sample-size 100 --audit-seed 2026 --output runs/edit-generation-plan.json
paper2 generate-edits --edits runs/edit-templates.json --plan runs/edit-generation-plan.json \
  --traces runs/traces-base.jsonl runs/traces-trained.jsonl --output runs/generated-edits.json
paper2 prepare-edit-audit --edits runs/generated-edits.json --output runs/edit-audit.json
```

A named human completes the frozen sample: 100 globally sampled candidate contracts by default, plus source/checkpoint coverage supplements; smaller batches are reviewed in full. Check editability classification and applicable control validity. Correct flaws found in sampled items and retain original failed judgments, corrections and re-review. Report the original flaw rate and interval, eligible-control estimates and the bound on possible contamination of the edit-following estimate. A global candidate flaw rate is not automatically an eligible-control or per-arm bound. The number of flaws never triggers an automatic batch pass/fail. The pilot G3 reviewer considers this evidence; the confirmatory audit reports quality without selecting whether an unfavorable result may be reported.

```bash
paper2 verify-edit-audit --edits runs/generated-edits.json --audit runs/edit-audit.json \
  --output runs/edit-audit-evidence.json
```

Missing judgments, altered provenance, incomplete panels or uncorrected known invalid controls are integrity failures. Uneditable traces, a high estimated editor flaw rate or wide effect intervals are reported outcomes. Do not drop difficult candidates or resample editor outputs until an audit looks favorable. Interrupted generation resumes its exact completed journal with `generate-edits --resume`. See [the edit protocol](EDIT_GENERATION.md) for sampling, corrections and contamination accounting. The answer cue and queried-belief-only summary are Test 2 positive controls; the summary supplies a premise for consequence questions, not the full symbolic state. The individually reviewed path and `--controls basic` remain pilot diagnostics; basic controls alone cannot establish G3 measurement validity.

Generate edited continuations once with **each original source checkpoint**, before either reader, using the same generated batch and completed audit:

```bash
paper2 regenerate-traces --input data/generated/pilot-audit-v5/evaluation/development__clean.jsonl \
  --traces runs/traces-base.jsonl --source base --edits runs/generated-edits.json \
  --edit-audit runs/edit-audit.json --output runs/regenerated-base.jsonl
paper2 regenerate-traces --input data/generated/pilot-audit-v5/evaluation/development__clean.jsonl \
  --traces runs/traces-trained.jsonl --source trained --edits runs/generated-edits.json \
  --edit-audit runs/edit-audit.json --adapter runs/pilot-reasoning-rl/adapter --training-seed 1 \
  --output runs/regenerated-trained.jsonl

paper2 crossed --input data/generated/pilot-audit-v5/evaluation/development__clean.jsonl \
  --traces runs/traces-base.jsonl runs/traces-trained.jsonl --edits runs/generated-edits.json \
  --edit-audit runs/edit-audit.json \
  --regenerated runs/regenerated-base.jsonl runs/regenerated-trained.jsonl \
  --reader base --output runs/crossed-base.jsonl
paper2 crossed --input data/generated/pilot-audit-v5/evaluation/development__clean.jsonl \
  --traces runs/traces-base.jsonl runs/traces-trained.jsonl --edits runs/generated-edits.json \
  --edit-audit runs/edit-audit.json \
  --regenerated runs/regenerated-base.jsonl runs/regenerated-trained.jsonl \
  --adapter runs/pilot-reasoning-rl/adapter --training-seed 1 --reader trained --output runs/crossed-trained.jsonl
```

Both readers consume identical effective controls, including recorded human corrections, and the same source-frozen continuations. All candidates retain original rows with final and initial edit-status fields; uneditable/failed candidates require no semantic regeneration. Report editability rates by source/checkpoint and conditional targeted-effect estimates with candidate, eligible and common-support denominators. Changed/intact forecasts use the stated-before versus stated-after belief; ordinary accuracy still uses the true world. The direct trace is written before later questions/events are shown and held fixed across views; no novel story-summary prompt is used. `prequestion` remains an alias for `direct_fixed`. Question-dependent timing is a pilot or supplemental diagnostic: collect it with `--timing question_dependent` and separate artifact/edit files. It cannot replace the primary confirmatory edit measurement. Raw-token replay and re-rendered replay retain separate eligibility and match rates.

Run one base/trained seed pair at a time. `crossed` infers `comparison_seed` from the trained source trace, or accepts an explicit `--comparison-seed`. It records reader and source seeds separately. Combine the three paired result files for the confirmatory analysis; shared base baselines are reused and fixed seeds are averaged within each story. Bootstrap intervals reflect stories, not a population of training seeds.

Combine both readers' immediate baseline files into `runs/baselines.jsonl`, and both crossed files into `runs/crossed.jsonl`, then:

```bash
paper2 analyze crossed --baseline runs/baselines.jsonl \
  --input runs/crossed.jsonl --output runs/crossed-summary.json
```

## 5. Test 3

The default primary mask is **every layer from 1 through the last layer**, jointly for K, V and residual outputs. The default also records individual-layer swaps descriptively. This is 1–35 for the pinned 36-block 3B model and 1–27 for the pinned 28-block 7B model. No `--all-layers`, selected layer-6 start or manual list is needed:

```bash
paper2 components --input data/generated/pilot-audit-v5/pairs/development.jsonl \
  --model Qwen/Qwen2.5-3B-Instruct --output runs/components-base.jsonl
paper2 components --input data/generated/pilot-audit-v5/pairs/development.jsonl \
  --model Qwen/Qwen2.5-3B-Instruct \
  --adapter runs/pilot-reasoning-rl/adapter --training-seed 1 \
  --output runs/components-trained.jsonl
paper2 analyze components --baseline runs/components-base.jsonl \
  --input runs/components-trained.jsonl --output runs/component-changes.json
```

For confirmation, use the 7B model, the corresponding adapters and confirmation pairs. Within each checkpoint, natural donor, own, and signed-permutation displacement-null panels accompany every component mask. Primary matched-event analysis uses oracle-derived changed and intact consequence sets, their joint pattern and whole-story intervals. G4 evidence must retain complete, structurally aligned joint K/V/residual panels for the primary band. Inspect residual effects and the nulls with intervals; a named pilot reviewer decides whether this supports detecting and interpreting key/value effects. The code does not require a positive point estimate. Per-layer results are descriptive and do not select a favorable primary band.

The residual hook runs after that block's own K/V have been computed, so even swapping all residual outputs need not reproduce donor logits. Layer 0 is excluded from the primary measurement; the aligned 1-to-last residual control applies to the same tested path as K/V. Full-depth or custom masks remain explicitly secondary diagnostics and cannot replace primary G4 evidence. Interpret a key/value null against the residual positive control and signed-permutation nulls; a descriptive narrow-layer null does not imply no distributed effect.

Secondary cross-model swaps remain available through `components --donor-model MODEL --donor-adapter PATH` (and optional `--donor-revision`). Donor and recipient receive identical token inputs and answer alphabets. These rows have `analysis: secondary_cross_model`, remain in separate files and are excluded from primary event-rewrite analysis. The confirmatory primary swaps stay within each checkpoint.

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

Repeat with adapters and `--reasoning` for official CoT prompts. The default `--opentom-protocol official` uses pinned official OpenToM system/user text and lexical scoring, with subtype macro-F1 primary, whole-story intervals, exclusion rates and full-denominator diagnostics. Run `--opentom-protocol adapted` separately for the secondary Paper 2 wrapper. Raw outputs remain available. The Qwen wrapper, generation caps and full-corpus selection still differ from some published runs. Zhong reports accuracy/order macro averages and has not released reproducible evaluation details, so do not claim exact numerical replication; see [OpenToM protocol](OPENTOM.md).

After reviewing G0–G4 and recording approved cost, use `configs/confirmatory/` for the full 24-run grid, set `--model Qwen/Qwen2.5-7B-Instruct` for evaluation, and use confirmation shards. Reasoning SFT takes `supervision/clean_reasoning_sft.jsonl` or its shortcut twin; the other arms take `training/*.jsonl`. Retain all three seeds. A second 7B profiling run can use a confirmatory config with `profile: true`, after paid approval, and the 3B profile as initial evidence; use the resulting target-size profile for the main 7B runs.

The named pilot reviewers see estimates and intervals, structural checks and the relevant positive controls, then record their decisions and rationale. Confirmation does not rerun numerical success gates or assign quadrant categories. Plot each arm as a point with intervals on both axes and discuss the proposal's candidate readings. The £500 total ceiling and 14 December 2026 review remain fixed.

## 7. Figures from measured analysis files

Install `pip install -e '.[plot]'`. For the G0/reference panel, plot the clean direct-question model results against the same split's measured heuristics:

```bash
paper2 plot heuristics --input runs/base-development.jsonl \
  --audit docs/validation/shortcut-audit.json --split development \
  --output runs/base-heuristic-references.png
```

For the two-axis figure, save `runs/figure-points.json` with one entry per compared arm/dataset. Paths below are relative to that manifest; use actual Test 1 and crossed summary filenames, not hand-entered estimates. Combine fixed seeds before producing each summary. Use distinct labels for minimal and regenerated edit points.

```json
{"points": [{"label": "Reasoning RL / clean", "arm": "reasoning_rl", "dataset": "clean",
  "reader": "trained", "source": "trained", "edit_condition": "semantic_minimal",
  "transfer_path": "no-trace-transfer.json", "crossed_path": "crossed-summary.json"}]}
```

```bash
paper2 plot quadrant --input runs/figure-points.json --output runs/two-measurements.png
```

Both commands save the exact numeric inputs and source hashes beside the figure as `.png.json` (PDF/SVG also supported). The two-axis sidecar retains editability and audit-bias sensitivity; the displayed intervals are story-bootstrap intervals. Unestimable axes are recorded explicitly, never filled in. No quadrant categories or pass lines are drawn. Heuristic figures verify the underlying story identities and show each depth/event stratum separately.
