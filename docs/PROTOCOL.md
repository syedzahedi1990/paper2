# Frozen questions and pilot decisions

H1 asks whether training improves trace production, trace use, or both. Report trained-minus-base no-trace accuracy on unrewarded consequences, and the crossed matrix of reader-specific fractions of remaining errors fixed. Also report errors introduced by supplied traces. A reader at ceiling has no remaining-error denominator; its fraction is undefined, not zero.

H2 asks whether shortcut-correlated training changes dependence on the planted cue and the scope of transfer. Report belief-dependent consequences, preservation, cue agreement/independence/conflict, and joint eight-view correctness separately. Average the counterbalanced cue replicas within each world/view before aggregating. Report unconditional and matched-direct-accuracy results; select matches only on development data, and retain all three confirmatory seeds.

RL immediate-answer evaluation and adapter retention occur every 50 optimizer steps; native generation occurs every 250. Both schedules include the base and final step. Match using the retained immediate/direct/development records without interpolating adapters or widening the ±0.02 tolerance after seeing outcomes. Report runs with no match. Native G1 evidence updates only at native evaluation steps.

The scope summaries distinguish oracle-derived belief-edit dependence from natural-event-rewrite effects. Unrewarded scope summaries exclude the direct rewarded question. Fixed confirmatory seeds are averaged inside each story before paired story bootstrap; these intervals do not estimate uncertainty over a population of seeds.

H3 asks whether training changes the contribution of critical-token keys, values and residual outputs. Compare the same within-checkpoint swaps before/after training. Score against the predicted event-rewrite consequence pattern, with the other five Paper 1 forecast accounts available as diagnostics. Do not use those rewrite targets for ordinary Test 1 transfer.

Declare `g4_test_layers` in frozen calibration settings and pass the identical band to `components --g4-test-layers`. The runner measures K, V and residual on that band; the residual control must start at the same first layer and use the complete declared mask. A positive residual effect certifies sensitivity only for that tested band. Layer-zero and final-only residual masks are excluded, and arbitrary interior single-layer diagnostics or full-depth diagnostic results do not establish sensitivity for a different band.

The primary confirmatory edit measurement uses `direct_fixed` traces. Freeze the editor model/revision, prompt, decoding settings, output budget and human-audit sample before generating controls. The editor proposes paraphrases and irrelevant edits; the symbolic oracle supplies the focal fact change and predicted consequences. Audit a preregistered random sample, with source/checkpoint coverage, under the batch's fixed acceptance rule. Unsampled contracts are not individually human-approved. Failed or unresolved batches cannot be silently filtered or resampled. See [model-generated controls and sampled review](EDIT_GENERATION.md).

## Gates

`paper2 init-gates` creates a pending review. `freeze-calibration` records G0 only after the investigator supplies development evidence, a named reviewer, settings and rationale. Every passed gate records evidence paths and SHA-256 hashes. A software unit test cannot establish the empirical positive-control gate.

| Gate | Evidence to review | Response if unsuccessful |
|---|---|---|
| G0 | Schema-v3/audit-v4 stratified controls, explicit corpus-bound review of remaining position/frequency/joint-observation heuristics, base development accuracy, failure types, chosen difficulty and frozen corpus | Recalibrate once before training |
| G1 | Clean reasoning-RL native-format learning curve versus native base, alongside immediate transfer curve; improvement and stable optimization | Diagnose format/task/optimization; increase model size only for a blocking capability/optimization limit after cost approval |
| G2 | Exact 90% training cue, matched entities/lengths, tokenizer audit, independent and conflicting renders | Regenerate and re-audit |
| G3 | Raw and re-rendered own-trace replay; source-frozen continuations; pilot diagnostic edits, with precision-sized `direct_fixed` edits for confirmation; fixed editor provenance and accepted preregistered human sample audit; actual story intervals; paraphrase and irrelevant behavior | Repair replay/editing in the pilot and preregister revisions before confirmation; resistance to a contradictory fact is itself a possible result |
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

[Proposal departures](PROPOSAL_DEPARTURES.md) records the new story distribution, weak heuristics, oracle-premise control, adapted OpenToM evaluation and unmatched optimization settings. These must be stated in the paper. The 12-world edit subset is only a pilot diagnostic; confirmatory rate planning and observed story precision are separate requirements. At 97 worlds, one shared base plus 24 trained source checkpoints requires 2,425 `direct_fixed` contracts for one rendering; additional checkpoints or renderings add work. Model generation and sampled human review reduce manual authoring, but do not establish that the four-week experiment block is sufficient. The revised RL schedule has nine native panels and 41 immediate panels/snapshots; profile both evaluation costs, storage and the later experiments before accepting the whole-pilot budget.
