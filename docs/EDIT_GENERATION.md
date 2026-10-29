# Model-generated controls and descriptive audit

Primary confirmatory edits use `direct_fixed`: one native direct-question trace per source and story, frozen across all eight questions. Use **all 120 confirmation stories**. There is no required editable count or interval-width target. One shared base plus 24 trained checkpoints creates 3,000 candidate contracts for one rendering; question-dependent edits would create 24,000 and are supplemental if compute allows. Additional checkpoints and renderings add work. These are model generations, not hand-written contracts.

## Freeze and generate

Save every source trace, including incorrect answers and empty traces. `prepare-edits` freezes candidate IDs and oracle destinations/forecasts for every possible trace-stated location. `plan-edit-generation` freezes the editor model and immutable revision, prompt v3, decoding, output budget, random sample and seed. The editor paraphrases the trace without correcting it, finds the focal belief **actually stated**, replaces just its location, and supplies an irrelevant wording edit. The belief may be wrong. A destination differs from both the stated location and the true belief; frozen deterministic alternatives avoid no-op edits. Existing answer statements stay; new answer statements cannot be introduced.

```bash
paper2 prepare-edits --traces runs/traces-base-fixed.jsonl runs/traces-trained-fixed.jsonl \
  --input data/generated/release-audit-v5/evaluation/development__clean.jsonl \
  --stage pilot --max-worlds 12 --output runs/edit-templates.json
paper2 plan-edit-generation --edits runs/edit-templates.json \
  --model Qwen/Qwen2.5-7B-Instruct --revision a09a35458c702b33eeacc393d103063234e8bc28 \
  --audit-sample-size 100 --audit-seed 2026 --audit-confidence .95 \
  --output runs/edit-generation-plan.json
paper2 generate-edits --edits runs/edit-templates.json --plan runs/edit-generation-plan.json \
  --traces runs/traces-base-fixed.jsonl runs/traces-trained-fixed.jsonl \
  --output runs/generated-edits.json
paper2 prepare-edit-audit --edits runs/generated-edits.json --output runs/edit-audit.json
```

The editor above is an example requiring G3 validation, not a pre-approved model. For confirmation substitute confirmation rows, include all intended sources and use `--stage confirmatory` without a subset. Complete eight-view panels are required. Resume interrupted generation with `--resume`; finished journal records are never resampled, including unusable/failed records. New protocol-v3 plans/audits require a fresh frozen lineage; old certificates cannot authorize the changed method.

## Outcomes and validity

Preserve every candidate. Report editable, belief not stated, ambiguous, contradicted, no safe control, and operational generation failure separately. Incorrect source answers do not imply noneditability; correct answers do not prove the belief was stated. Missing or contradictory beliefs are results, not batch failures. Malformed/truncated editor output is an operational failure, not evidence about the model's belief. Neither kind receives an edited control, but both retain original rows and candidate denominators.

Machine checks enforce exact source spans, one frozen location substitution, nonoverlap, valid output structure, immutable identities and recognized answer-label leakage patterns. **Comparable length is reviewed, not decided by a .5–2 ratio.** Paraphrase/original and irrelevant/semantic length ratios are recorded. The finite-pattern answer filter cannot establish semantic fidelity; human checks cover meaning, length comparability, focal fact, neutrality and forecast applicability.

A named human records every required judgment for the frozen random sample. Known flawed sampled controls or outcome classifications require a documented correction and re-review before use; a correction may instead classify the artifact as noneditable. This is completion of a validity check, not a limit on the number of flaws. Keep original judgments and raw model outputs unchanged. Put the replacement structured controls/classification, reason, reviewer and completed checks in the audit item's `correction` object. No model resampling is allowed to obtain a more favorable batch. Unsampled controls are not individually approved.

## Report quality without acceptance cutoffs

The default uniform hash-ranked sample has 100 candidate contracts plus a coverage example from every source/checkpoint absent from that sample. Small batches are audited fully. Sample size is a preregistered workload choice, not a numerical pass rule. Audit protocol v3 has no maximum flaw count or maximum error rate. G3 assesses the editor procedure in the pilot. Confirmation reports original flaw rates and uncertainty, even after all observed faults are corrected.

The certificate reports exact finite-population two-sided intervals and one-sided upper rates for the entire candidate population, the originally ready subset and each source/checkpoint. Statistical intervals use only the uniform sample; coverage supplements do not inflate its size. An unsampled source has an unidentified rate and [0,1] interval. These intervals assume a random seed chosen independently of outputs and accurate human judgments. Original flaws still count after correction; classification and operational failures remain distinguishable.

A global flaw rate is not a per-arm bound. The certificate also gives Bonferroni simultaneous upper bounds over frozen source/checkpoint strata. Analysis allocates each source's maximum possible flawed-contract count to the largest actual story/seed/view weights, giving a conservative bias cap for the reported edit-following rate. A paired outcome contrast in [−1,1] can shift by twice this weight. Newly eligible human-corrected items outside the originally ready population receive worst-case weight. This sensitivity analysis is separate from bootstrap sampling uncertainty; report both, including uninformative bounds. Corrections make the original-flaw bound conservative, not erased.

```bash
paper2 verify-edit-audit --edits runs/generated-edits.json --audit runs/edit-audit.json \
  --output runs/edit-audit-evidence.json
```

Missing judgments, unrepaired known faults, altered hashes or incomplete panels block use. High flaw counts and broad intervals do not. Revise an inadequate editor during the pilot and freeze the revision before confirmation, retaining earlier attempts.

## Crossed use and interpretation

Pass the same `--edits` and `--edit-audit` files to both `regenerate-traces` and `crossed`. Each original source regenerates an edited continuation once; both readers consume the same saved text. Audit/contract changes invalidate earlier continuations. Individually reviewed contracts remain available for pilot diagnostics.

Report per-arm editability with story intervals, conditional edit effects, candidate/eligible counts and common-support comparisons. Different arms can have different eligible populations; these are not unconditional treatment effects. Average fixed seeds within story before 20,000 whole-story resamples. No editable count, precision flag or quadrant classification is emitted.

[Lanham et al. (2023), §§2.4 and 2.6](https://arxiv.org/html/2307.13702v1) motivates model-generated edits/paraphrases and source-generated continuations. Our oracle contracts, irrelevant edits, explicit editability outcomes and sampled review are disclosed study choices. The [preregistration](PREREGISTRATION.md) defines the estimands and positive controls.
