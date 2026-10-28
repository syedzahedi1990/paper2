# Review corrections, version 0.5.0

The original 72 passing tests did not establish scientific validity. The revised suite and corpus audits test the specific failure modes identified in review. Earlier corrections changed the corpus; version 0.5 updates stated-belief editing, noneditability accounting, eligible precision and the sampled audit without changing corpus schema v3/audit v4. Artifacts from older corpus schemas remain invalid, and the new edit/audit protocol requires a new frozen experiment lineage. No research-model result is claimed.

| Review finding | Correction |
|---|---|
| Rewarded answer obtainable from first relevant event | Multiple relevant events, variable deciding position and observation sets; initial/event locations vary independently |
| Nested questions collapse to first-order answers | Nested beliefs can equal either, both or neither first-order belief; fixed joint-relation quotas in each depth/event stratum of each split |
| Inverse rule “nested always differs from both” | Audit both equality and disagreement, reject deterministic relations, test candidate-exclusion answer rules, and verify declared joint quotas exactly |
| G0 did not check shortcuts | Counterbalance held-out clean cues within depth/event cells; recompute stratified audits from hashed corpus files before freezing and before training; old manifests and missing strata fail |
| Ambiguous change/preservation scope | Oracle-derived per-world belief-edit and event-rewrite masks stored separately; Test 1 retains ordinary targets and has unrewarded-only scope summaries |
| Single-layer-only Test 3 | Simultaneous full-depth or declared-band swaps per component; optional individual-layer diagnostics; CLI supports `--all-layers` |
| H3 pooled incompatible consequences | CLI reports required changes, intact answers and joint consequence patterns, plus per-view probability contrasts and changes relative to own controls |
| Reader-specific regenerated traces | Source generates and saves a continuation once; both readers consume the same hashed artifact; analysis rejects differing supplied text |
| Novel untrained prequestion prompt | Native direct-question trace is frozen across later views (`direct_fixed`); old `prequestion` name is an explicit alias |
| Excessive manual edits | Pilot subset defaults to 12 worlds; confirmation uses 120 candidate worlds with `direct_fixed`, one contract per world/source. A fixed preregistered model generates controls for the actual stated belief; audit-v2 permits bounded original flaws with recorded human corrections |
| Editing assumed the trace stated the correct belief | Wrong stated beliefs are editable; oracle forecasts before/after the stated change remain separate from world truth |
| Noneditability or one malformed generation blocked every contract | Record noneditable and generation-failed outcomes, retain every candidate denominator, and omit only unusable targeted controls; require audited classifications and sufficient eligible precision |
| Own replay tautology | Exact-prefix replay retained as a low-level control; re-rendered replay, prefix-token equality and eligibility reported separately |
| G1 measured wrong format | Native-format and immediate development curves, matched native base, and machine-readable G1 evidence |
| Insufficient RL exposure | Default full-pool pass with actual distinct-world coverage; RL groups and SFT exposures are recorded separately; provisional arm-specific learning rates remain configurable |
| Missing or coarsely spaced accuracy-matched adapters | RL immediate evaluation and retained adapters every 50 steps; native generation every 250. Matching records retain weights hashes and arm/dataset/seed; the ±0.02 tolerance is unchanged and missing matches remain explicit |
| Seeds omitted from analysis | Paired fixed seeds averaged within story before bootstrap; missing seed panels rejected; crossed reader/source/comparison seed identities separated |
| Cue contrast pooled across views | Direct and per-view paired contrasts are primary outputs; pooled direct-anchored manipulation effect labelled separately; literal cues stay fixed across views |
| OpenToM exact-match mismatch | Question/plot-derived options and type-specific scoring, upstream-compatible diagnostics, raw outputs and explicit prompt/scoring adaptations |
| Test 3 prompt drift | Same inventory-label text across matched pair members; overwrite says “After the subsequent event” rather than assuming two moves |
| Cross-model CLI absent | Explicit donor model/adapter path on identical complete token inputs and answer alphabet; secondary rows excluded from primary event-rewrite analysis |
| Windows provenance hash failure | `.gitattributes` preserves LF text checkout, with binary exclusions; Windows/core CI job added |
| Missing torch caused test failures | Optional-dependency guards precede imports; separate environment without torch is tested |

## Scientific choices retained

A belief edit and an event rewrite are different interventions. Physical-world preservation under a belief edit does not imply preservation when a physical event is rewritten. The two masks are not substituted for each other.

With six locations and independent first-order beliefs, nested equality with either first-order answer has probability 1/6, equality with both 1/36, and disagreement with both 25/36. A disagreement rate greater than one-half is therefore not itself evidence of a shortcut. The audit also measures the actual expected accuracy of excluding those answers, which should remain near 1/6. Small held-out strata use preregistered rounded joint quotas, not sampling until a favourable audit appears. Equal-both is not required in every tiny arbitrary sample; the production quota explicitly contains it.

Full-layer interventions can reveal distributed effects that single-layer tests miss. They do not guarantee an empirical positive control or a nonzero K/V effect. G4 still requires measured controls and precision. G1 still requires actual learnability, and learning-rate defaults are not claims of optimality.

The shared-chat OpenToM evaluation is a documented adaptation, not a claim to reproduce every published prompt. See [OpenToM protocol](OPENTOM.md). The edit subset and fixed-direct timing are documented in [trace protocol](TRACES.md).

See [validation evidence](VALIDATION.md) for completed software/data checks. Audit v4 additionally reports weak position/frequency/last-joint-observation heuristics rather than claiming they are at chance. G0–G4 remain empirical pilot decisions; no 3B/7B training or paid compute was started by this correction.

## Second review, version 0.3

- Expanded audit v4 tests every focal-event position, focal/all-event and full-story location frequency, and the last jointly observed focal event. Known weak rules remain in this distribution; explicit corpus-bound G0 review is required.
- G4 excluded layer-zero and final-only residual masks, and bound a measured interior positive control to raw hashed receipts and complete consequence panels. Version 0.4 adds exact alignment with the declared K/V test band.
- Native RL evaluation was reduced to every 250 steps: nine panels and 1,080 native generations per run. Version 0.4 retains that schedule while adding denser immediate evaluation and adapter retention.
- Twelve edit worlds are pilot-only diagnostics. At version 0.3, confirmatory rate precision planning defaulted to 97 worlds for approximate 95% ±.10, required the full eight-view panel, and reported actual story intervals. Focal-edit consequences are populated by the oracle while natural-language review remains required.
- [Proposal departures](PROPOSAL_DEPARTURES.md) supplies the distinctions the paper must state, including the new story distribution and limits on benchmark comparability.

## Third review, version 0.4

These were the version 0.4 settings. Version 0.5's revisions to candidate counts, editability and audit acceptance are recorded below.

- G4 requires an explicitly preregistered test band, passed as `--g4-test-layers` and recorded in frozen `settings.g4_test_layers`. K, V and residual are measured on that same band, including its first layer. Gate evidence must select the identical `test_layers` and provide complete same-band natural/own/null receipts. Arbitrary interior singles and the legacy residual-only diagnostic flag cannot certify another band. Full-depth diagnostics remain available; the positive control validates only its declared band.
- RL measures immediate accuracy and saves adapters at 0, 50, …, 2,000: 41 matching snapshots. Native traces remain at 0, 250, …, 2,000: nine panels. Profiling records the formats separately and includes added adapter storage. This improves matching resolution without guaranteeing a ±0.02 match or a whole-pilot compute budget.
- Confirmatory edits require one frozen direct-question trace per world/source, projected across all eight views. With 97 worlds and 24 trained checkpoints plus one shared base, one rendering requires 2,425 contracts, rather than 19,400 question-dependent contracts. Matched checkpoints and extra renderings increase these counts.
- A fixed editor model/revision, prompt, decoding policy and token budget generates paraphrases, irrelevant edits and exact spans for oracle-proposed fact changes. Generation happens once, with hashed inputs, raw outputs and a resumable journal. A preregistered random human sample, supplemented for source/checkpoint coverage, audits the batch under a fixed acceptance rule. Unsampled controls remain individually unapproved; unresolved or failed batches cannot be silently dropped or repeatedly resampled.
- [The edit-generation protocol](EDIT_GENERATION.md) documents the method, audit limits and relation to Lanham et al. Model assistance removes mandatory manual authoring of every contract; no claim that the four-week experiment schedule is sufficient follows from the implementation.

## Fourth review, version 0.5

- Confirmatory preparation defaults to all 120 candidate worlds. The default .10 half-width rate plan still requires at least 97 eligible independent worlds per arm/checkpoint and sufficient actual story precision. A 25-source, one-rendering grid has 3,000 candidate contracts rather than 24,000 question-dependent contracts. Eligibility shortfalls remain visible and cannot be repaired by silently pooling or adding favourable candidates.
- The editor targets the focal belief actually stated, including wrong beliefs. The oracle supplies stated-before and stated-after forecasts while preserving world-truth answers separately. Changed/intact consequences use the actual before/after intervention. The frozen destination remains distinct from both the stated and true belief, using a deterministic alternative when the initial scope target would be a no-op.
- Missing, ambiguous or contradicted beliefs and the absence of safe controls are reported scientific noneditability outcomes. Malformed, truncated or failed model outputs remain distinct `generation_failed` outcomes. Such candidates keep their original rows, denominators and error records without receiving unusable edit controls. Their presence alone no longer fails the entire batch. Tampered provenance, missing batch entries or missing required human judgments still block use.
- Existing answer statements may remain in source traces and controls. Newly introduced or changed answer labels remain forbidden. This distinguishes source content from leakage caused by the editor.
- Audit v2 defaults to 100 global candidate contracts plus source/checkpoint coverage, permits at most one original sampled flaw, and requires the exact finite-population one-sided 95% upper error bound to be at most 5% over all candidate contracts. Review includes noneditability classifications. Sampled flaws receive human corrections and re-review, with the original failure count and raw model batch retained. No adaptive model resampling is allowed.
- The eligible-subset error bound is reported separately. Global assurance and source coverage do not establish a per-arm 5% guarantee. Conditional eligible-population effects and paired common-support comparisons retain separate coverage and denominators; correctness-dependent editability remains a limitation to disclose.
- Test 3 runbook examples now span an early interior block through the last block: 6–35 for pinned 3B, 6–27 for pinned 7B. The illustrative starts need calibration and preregistration. A null from a narrow band such as 6–11 cannot establish that critical-token K/V have no effect generally.
- Version 0.5 requires a new frozen edit/audit lineage. Editor quality, attained eligible precision, GPU costs and the four-week experiment schedule still require empirical review; software completion does not establish them.
