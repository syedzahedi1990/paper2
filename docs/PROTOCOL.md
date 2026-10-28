# Frozen questions and pilot decisions

H1 asks whether training improves trace production, trace use, or both. Report trained-minus-base no-trace accuracy on unrewarded consequences, and the crossed matrix of reader-specific fractions of remaining errors fixed. Also report errors introduced by supplied traces. A reader at ceiling has no remaining-error denominator; its fraction is undefined, not zero.

H2 asks whether shortcut-correlated training changes dependence on the planted cue and the scope of transfer. Report belief-dependent consequences, preservation, cue agreement/independence/conflict, and joint eight-view correctness separately. Average the counterbalanced cue replicas within each world/view before aggregating. Report unconditional and matched-direct-accuracy results; select matches only on development data, and retain all three confirmatory seeds.

RL immediate-answer evaluation and adapter retention occur every 50 optimizer steps; native generation occurs every 250. Both schedules include the base and final step. Match using the retained immediate/direct/development records without interpolating adapters or widening the ±0.02 tolerance after seeing outcomes. Report runs with no match. Native G1 evidence updates only at native evaluation steps.

The scope summaries distinguish oracle-derived belief-edit dependence from natural-event-rewrite effects. Unrewarded scope summaries exclude the direct rewarded question. Fixed confirmatory seeds are averaged inside each story before paired story bootstrap; these intervals do not estimate uncertainty over a population of seeds.

H3 asks whether training changes the contribution of critical-token keys, values and residual outputs. Compare the same within-checkpoint swaps before/after training. Score against the predicted event-rewrite consequence pattern, with the other five Paper 1 forecast accounts available as diagnostics. Do not use those rewrite targets for ordinary Test 1 transfer.

Declare `g4_test_layers` in frozen calibration settings and pass the identical band to `components --g4-test-layers`. The runner measures K, V and residual on that band; the residual control must start at the same first layer and use the complete declared mask. A positive residual effect certifies sensitivity only for that tested band. Layer-zero and final-only residual masks are excluded, and arbitrary interior single-layer diagnostics or full-depth diagnostic results do not establish sensitivity for a different band.

The primary confirmatory edit measurement uses `direct_fixed` traces and all 120 candidate worlds by default. At least 97 eligible independent worlds per arm/checkpoint and sufficient observed interval precision are needed for the default .10 half-width target. Candidate selection alone does not satisfy that requirement. Report editability, reason counts and conditional eligible-population estimates alongside paired common-support comparisons; eligibility differences can reflect differences in source correctness.

Freeze the editor model/revision, prompt, decoding settings, output budget and human-audit sample before generating controls. Edit the focal belief actually stated, including wrong beliefs, with separate stated-before, stated-after and world-truth forecasts. Missing, ambiguous or contradicted beliefs and the absence of safe controls are recorded noneditable outcomes. Malformed, truncated or failed editor generations are recorded separately as `generation_failed` outcomes. These candidates retain their original rows and denominator contributions and receive no targeted edit controls; an isolated failure does not invalidate the complete batch. Existing source answer statements are permitted; new or changed answer labels are not.

Audit protocol v2 defaults to 100 uniformly sampled candidate contracts plus source/checkpoint coverage. It permits at most one original sampled flaw and requires an exact finite-population one-sided 95% upper error bound no greater than 5% over all candidate contracts. Review noneditability classifications as well as usable controls. Human correction and re-review of sampled flaws do not erase original failures or replace the raw model batch. Report the eligible-subset error bound separately; neither bound establishes a per-arm 5% guarantee. Unsampled contracts remain individually unapproved. Altered provenance, an incomplete batch or missing sampled judgments still block use; no model resampling is allowed. See [model-generated controls and sampled review](EDIT_GENERATION.md). Version 0.5 requires a new frozen edit/audit lineage.

## Gates

`paper2 init-gates` creates a pending review. `freeze-calibration` records G0 only after the investigator supplies development evidence, a named reviewer, settings and rationale. Every passed gate records evidence paths and SHA-256 hashes. A software unit test cannot establish the empirical positive-control gate.

| Gate | Evidence to review | Response if unsuccessful |
|---|---|---|
| G0 | Schema-v3/audit-v4 stratified controls, explicit corpus-bound review of remaining position/frequency/joint-observation heuristics, base development accuracy, failure types, chosen difficulty and frozen corpus | Recalibrate once before training |
| G1 | Clean reasoning-RL native-format learning curve versus native base, alongside immediate transfer curve; improvement and stable optimization | Diagnose format/task/optimization; increase model size only for a blocking capability/optimization limit after cost approval |
| G2 | Exact 90% training cue, matched entities/lengths, tokenizer audit, independent and conflicting renders | Regenerate and re-audit |
| G3 | Raw and re-rendered own-trace replay; source-frozen continuations; stated-belief edit validity; candidate, eligible and common-support denominators; distinct noneditable and generation-failed outcomes; accepted audit-v2 review and corrections; sufficient eligible story counts and intervals for confirmatory `direct_fixed` effects | Repair editing in the pilot and preregister revisions before confirmation; report insufficient editability or precision rather than assigning a quadrant; resistance to a contradictory fact is itself a possible result |
| G4 | Frozen declared band; complete same-band K/V/residual own, natural and null panels; hashed residual evidence starting at the tested band's first layer, excluding layer zero and final-only masks; measured changed/intact/joint effects, flat nulls, useful precision | Repair implementation or increase story count |

Before paid 7B training, fill `paid_compute` with the approved profiling-based estimate, approver, exchange-rate date and budget no greater than £500. Retain the actual approval evidence. The code refuses incomplete reviews, modified evidence and missing cost information. It does not make purchases.

## Timing from the proposal

| Work block | Planned dates |
|---|---|
| Environment, twin and replay tools | 2–22 November 2026 |
| 3B pilot, G0–G4 | 23 November–13 December 2026 |
| Go/no-go review | 14 December 2026 |
| Main paid 7B runs | 14 December 2026–7 February 2027 |
| Tests 1–3 | 8 February–7 March 2027 |
| Conditional Bridge | 8–28 March 2027, only if its separate decision rule is met |
| Writing and internal review | 29 March–2 May 2027 |

These are the supplied plan's dates, not scheduled automations. The proposal's preliminary dollar estimates are preserved in the source document. Obtain measured throughput and current purchase-day rates before committing compute. Cut optional analyses and layer searches before cutting the three confirmatory seeds.

## Disclosures and resource limits

[Proposal departures](PROPOSAL_DEPARTURES.md) records the new story distribution, weak heuristics, oracle-premise control, adapted OpenToM evaluation and unmatched optimization settings. These must be stated in the paper. The 12-world edit subset is only a pilot diagnostic. At 120 candidate worlds, one shared base plus 24 trained source checkpoints requires 3,000 `direct_fixed` contracts for one rendering, compared with 24,000 question-dependent contracts; additional checkpoints or renderings add work. Editor quality, eligible story counts and observed precision remain unverified. Model generation and sampled human review do not establish that the four-week experiment block is sufficient. The RL schedule has nine native panels and 41 immediate panels/snapshots; profile both evaluation costs, storage and later experiments before accepting the whole-pilot budget.
