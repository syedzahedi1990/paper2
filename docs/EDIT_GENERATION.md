# Model-generated controls, editability and sampled human review

The primary confirmatory edit measurement uses `direct_fixed`: one native direct-question trace per world and source checkpoint, held fixed across all eight views. Version 0.5 selects **all 120 confirmation worlds** by default. The approximate 95% ±.10 rate plan still needs **97 editable stories per analysed checkpoint**, plus sufficiently narrow measured intervals. This requires at least 80.8% coverage with one trace per story; 120 candidates do not guarantee it. Do not replace missing confirmation worlds or widen the target after seeing results.

For 24 trained final checkpoints plus one shared base and one clean rendering, this creates **3,000 candidate contracts**, versus 24,000 question-dependent contracts. Additional matched checkpoints or cue renderings add work. This is generation workload, not 3,000 manually written controls. Editor generation, source continuations and crossed reading need their own compute accounting; the training profile does not establish the four-week experiment budget.

## Freeze the method before generating controls

1. Save every source trace, including incorrect answers and short traces. Include the base once in a grid-wide batch. Reader runs operate on one base/trained seed pair at a time.
2. Prepare the oracle-backed candidate manifest. For every possible stated location, it freezes an edit destination and the eight before/after consequence forecasts. Use the originally proposed destination unless it equals the stated location; then use the frozen deterministic alternative. The destination differs from both the stated location and the true focal belief.
3. Freeze one editor model, immutable revision, prompt version, decoding settings, token budget, audit seed/sample size, allowed failure count and error-rate bound. The fixed prompt requests a faithful paraphrase, an irrelevant wording edit and an exact substitution in the final/query-time focal belief the trace **actually states**, whether correct or incorrect. Historical beliefs explicitly superseded later are not contradictions.
4. Generate once and preserve every result and raw output. The editor cannot invent a missing belief or turn an incorrect premise into a true one merely to obtain an editable trace. Generation journals permit recovery without resampling completed items.

The model and revision below are examples requiring pilot quality validation:

```bash
paper2 prepare-edits --traces runs/traces-base-fixed.jsonl runs/traces-trained-fixed.jsonl \
  --input data/generated/pilot-audit-v4/evaluation/development__clean.jsonl \
  --stage pilot --max-worlds 12 --output runs/edit-templates.json
paper2 plan-edit-generation --edits runs/edit-templates.json \
  --model Qwen/Qwen2.5-7B-Instruct --revision a09a35458c702b33eeacc393d103063234e8bc28 \
  --audit-sample-size 100 --audit-seed 2026 --max-audit-errors 1 \
  --max-error-rate .05 --audit-confidence .95 --output runs/edit-generation-plan.json
paper2 generate-edits --edits runs/edit-templates.json --plan runs/edit-generation-plan.json \
  --traces runs/traces-base-fixed.jsonl runs/traces-trained-fixed.jsonl \
  --output runs/generated-edits.json
paper2 prepare-edit-audit --edits runs/generated-edits.json --output runs/edit-audit.json
```

For confirmation, use confirmation rows, all intended source files and `--stage confirmatory --precision-half-width .10`, omitting `--max-worlds`. Preparation requires complete eight-view fixed-direct panels. An interrupted generation resumes with the same command plus `--resume`; the journal binds the original plan and preserves ready, uneditable and failed records. Existing final batches and plans cannot be overwritten. Version 1 plans and audits do not authorize this changed protocol; freeze a new lineage.

## Editability is an outcome

The editor locates the exact focal character path and object, and states the location expressed by the trace. The oracle retains the true location separately. An incorrect conclusion does not by itself make the trace ineligible. Conversely, a correct answer alone does not prove that the focal belief was explicitly stated.

Every candidate remains in the dataset. Outcomes distinguish editable, belief not stated, contradictory focal belief, ambiguous focal belief, no safe control, and generation failure. Malformed JSON, truncation or inference failures are operational outcomes, not evidence that a model lacks a belief. A failed or uneditable item supplies no edited control, but its ordinary trace, status, reason and source/checkpoint identity remain available to both readers. A single such item does not block the batch. Missing candidate records, altered provenance or an incomplete audit still block use.

Report category rates and editable story counts by source checkpoint/arm. Targeted compliance and preservation are conditional on editable traces; report candidate coverage alongside them. Seeds can have different eligible sets. Analysis reports checkpoint precision and shared-world support separately; do not interpret differing conditional populations as an unconditional treatment effect. Failure to meet the eligible count or observed precision criterion leaves the confirmatory edit axis unconfirmed, while editability itself remains a reportable result. These are rates of explicit trace content and successful editing, not direct measurements of an internal belief representation.

Changed/intact forecasts compare the oracle state with the trace-stated belief before versus after substitution. Ordinary accuracy remains relative to the true world. The minimal edit preserves other text, including an existing answer statement; source regeneration can rewrite the continuation. This distinction is part of the intervention, not a reason to discard incorrect traces.

## Automated checks and the human audit

Machine checks require exact source spans, one location substitution, nonoverlapping irrelevant edits of comparable length, paraphrase length bounds and strict output structure. The answer filter checks recognized explicit option-label patterns for newly introduced labels or excess occurrences. A faithful paraphrase may retain an answer statement already present in the original. This finite-pattern check is not a semantic leakage detector; the human audit checks meaning, including how existing labels are used. These checks do not establish semantic fidelity.

The default audit is a globally hash-ranked sample of **100 candidate contracts**, plus one from each source/checkpoint stratum absent from that sample. Small batches are audited in full. The seed and identities are frozen before outputs. A named human checks the original editability classification and, for usable controls, paraphrase meaning, the intended factual change, irrelevant-edit neutrality and oracle forecast applicability.

Version 2 allows **at most one flawed sampled item by default**, including coverage supplements, and additionally requires an exact finite-population one-sided **95% upper bound of at most 5%** on flawed candidate decisions/contracts. The bound uses only the uniform sample; the supplements do not inflate its sample size. It assumes random selection independent of outputs and accurate human judgments. A smaller custom sample may fail the bound even with no observed flaws. The count, confidence and error bound are preregistered choices; do not relax them after inspecting the audit.

This global bound is not a per-arm guarantee and is not automatically a 5% bound among the controls actually used. The audit separately reports the bound for the originally editable subset. A sparse eligible sample can leave that bound much wider. Report both populations and limits.

For every flawed sampled item, retain the original failed checks and add a human correction with replacement controls or corrected editability classification, a reason, reviewer identity and completed re-review checks. Use the sampled item's `correction` object with `controls` (the same structured ready/not-editable output format), `reason`, `reviewed_by` and `checks` for the effective outcome. Original `checks` remain unchanged. Corrections are recorded in the audit, not written over the original batch or journal. They must preserve the same trace and frozen oracle options. The certificate identifies the effective corrected contracts for both readers. Original failures still count toward the allowed total and statistical bound after correction. There is no model resampling to obtain a passing audit. Missing judgments, excessive original failures, uncorrected sampled flaws or inadequate global error-rate assurance block acceptance.

```bash
# After the investigator records the original judgments and any required corrections:
paper2 verify-edit-audit --edits runs/generated-edits.json --audit runs/edit-audit.json \
  --output runs/edit-audit-evidence.json
```

Unsampled controls remain individually unapproved. The result is sampled quality assurance for a frozen population, with explicit corrections to observed flaws. If the pilot reveals that the editing method is inadequate, revise and preregister the method before confirmation, preserving failed attempts.

## Crossed use

Pass `--edits runs/generated-edits.json --edit-audit runs/edit-audit.json` to both `regenerate-traces` and `crossed`. Regenerate editable semantic continuations once with each original source checkpoint; use the same effective contract and frozen continuation for both readers. Uneditable and failed candidates remain in original/control-panel denominators but require no semantic continuation. All uses recheck source, plan, audit and contract bindings; changing a correction invalidates earlier continuations.

The individually reviewed path remains available for pilot diagnostics. Neither path allows missing selected candidates to be silently removed.

## Relation to prior work

[Lanham et al. (2023), §§2.4 and 2.6](https://arxiv.org/html/2307.13702v1) uses model-generated mistaken reasoning and paraphrases, with continuations regenerated by the original source model for the mistake experiment. Our stated-belief oracle contracts, irrelevant edits, editability outcomes and sampled human review are explicit Paper 2 choices, rather than a reproduction of that paper's exact protocol.
