# Review corrections, version 0.3.0

The original 72 passing tests did not establish scientific validity. The revised suite and corpus audits test the specific failure modes identified in review. This release changes the corpus and experiment protocol; old datasets, trace artifacts and frozen G0 approvals must be regenerated. No research-model result is claimed.

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
| Excessive manual edits | Seeded, preregistered world subset (default 12); shared traces use one contract; symbolic belief summaries are automatic |
| Own replay tautology | Exact-prefix replay retained as a low-level control; re-rendered replay, prefix-token equality and eligibility reported separately |
| G1 measured wrong format | Native-format and immediate development curves, matched native base, and machine-readable G1 evidence |
| Insufficient RL exposure | Default full-pool pass with actual distinct-world coverage; RL groups and SFT exposures are recorded separately; provisional arm-specific learning rates remain configurable |
| Missing accuracy-matched adapters | Adapter retained at every evaluated step; training emits matching records with weights hash; selection grouped by arm/dataset/seed and missing matches retained explicitly |
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

## Second review

- Expanded audit v4 tests every focal-event position, focal/all-event and full-story location frequency, and the last jointly observed focal event. Known weak rules remain in this distribution; explicit corpus-bound G0 review is required.
- G4 excludes layer-zero and final-only residual masks, and binds a measured interior positive control to raw hashed receipts and complete consequence panels. Full-depth K/V are retained.
- RL evaluation is every 250 steps: nine panels and 1,080 native generations per run. Profiling now measures and projects development and storage costs separately.
- Twelve edit worlds are pilot-only diagnostics. Confirmatory rate precision planning defaults to 97 worlds for approximate 95% ±.10, requires the full eight-view panel, and reports actual story intervals. Focal-edit consequences are populated by the oracle while natural-language review remains required.
- [Proposal departures](PROPOSAL_DEPARTURES.md) supplies the distinctions the paper must state, including the new story distribution and limits on benchmark comparability.
