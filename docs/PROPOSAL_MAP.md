# Proposal implementation map

The proposal is the design authority. Values it leaves unspecified are implementation choices below and must be frozen after development calibration, before training. This table distinguishes executable support from measurements still requiring the pilot.

| Requirement | Implementation | Validation or remaining evidence |
|---|---|---|
| Paper 1 world, 20 objects, 6 locations, 2 characters, depth up to 2 | `data/_vendor`, `data/world.py`, `data/provenance.py` | Original oracle retained; first-order parity audit; depth-two view extension documented |
| 2000 train / 120 dev / 120 confirmation stories | `data/corpus.py` | Disjoint family/world/story hashes; immutable generated manifest |
| Direct-belief training, seven unrewarded consequences | `training/data.py`, separate training/supervision exports | Reject held-out splits, non-direct views, world state and unintended trace supervision |
| 90% shortcut twin with agree/independent/conflict test renders | `data/corpus.py` | Same worlds; exact entity and character counts; tokenizer audit; independent six and conflict five balanced variants |
| Four trained arms plus base | `training/config.py`, `configs/` | Three pilot configs and 24 confirmatory configs; base evaluated separately |
| Qwen2.5 3B pilot, 7B final, LoRA, group 8, trace cap 384 | `training/runner.py`, `training/generation.py` | Fixed target modules; real random-Qwen SFT/GRPO integration tests |
| Reasoning SFT source and extra information | `data/world.py:build_oracle_trace`, training manifests | Symbolic oracle supplies update rule and intermediate focal-belief states |
| Native and immediate format evaluation | `evaluation.py`, `backend.py`, `prompts.py` | Full-vocabulary argmax primary; restricted-choice result retained only as diagnostic |
| H1 no-trace transfer | `analysis.py:transfer` | Paired story bootstrap, unrewarded views, scope split, preservation |
| H1 crossed sources and readers | `traces.py`, `analysis.py:crossed_summary` | Per-reader remaining-error denominator; ceiling -> undefined, regressions separate |
| Replay and seven controls | `traces.py` | Exact-prefix replay; explicit reviewed edit contracts; invalid/missing controls fail closed |
| Minimal edit plus regenerated continuation | `traces.py:run_crossed_reader` | Same edited prefix, truncated old continuation removed for regeneration |
| Trace before later questions | `traces.py:collect_traces` | One base-story trace fixed across views; future consequence events excluded |
| H2 cue dependence and joint outcomes | `analysis.py:cue_summary`, `accuracy_summary` | Balanced replicas averaged within world/view; all eight views required for joint score |
| Matched direct accuracy | `analysis.py:match_checkpoints` | Development-only immediate direct scores; target/tolerance explicitly supplied |
| H3 K/V/residual swaps and controls | `interventions/components.py`, `runner.py`, `scoring.py` | Real GQA/RoPE tiny-Qwen tests; empirical G4 effects still pending |
| Within-model primary, cross-model secondary | `ComponentInterventionEngine.swap` | Cross-model prefix identity enforced; primary CLI never crosses models |
| OpenToM external check | `opentom.py` | Official flat QA schema; free-response normalized exact match, separated by question type |
| Profiling and compute reporting | `training/ledger.py`, `training/runner.py` | Actual counters/timers and manifests; practical GPU timings pending |
| G0–G4 and paid-run review | `gates.py`, `docs/PROTOCOL.md` | All gates initially pending; evidence hashes required |
| Conditional Bridge follow-on | No Bridge experiment implemented | Separate proposal and its decision rule required; no calendar-triggered launch |

The proposal's figure gives four **candidate readings** (trace-dependent gain; transfer still sensitive to traces; narrow gain; trace-independent transfer). The implementation reports the two measurements and their controls; it does not automatically label a quadrant as a discovered mechanism.

## Choices left open in the proposal

- Seeds are 1, 2, 3; the corpus seed is 20261102.
- LoRA uses rank 16, alpha 32, dropout 0, all attention and MLP projection modules. Direct Q/K/V updates are therefore part of the treatment and must be stated when reading Test 3.
- Initial training settings are in JSON, with 250 optimizer steps and a constant schedule with warmup. These are pilot settings, not established sufficient convergence.
- Default data mixes depths 1 and 2 and move/communication events. Difficulty controls include distractor events and hidden moves. Recalibrate on development data once if needed, then freeze.
- Test 3 layers must be supplied explicitly. The proposal does not specify layers; choose and freeze them during implementation/profiling, before looking at confirmatory outcomes.
- Semantic/paraphrase controls require content review. Automatic string substitution cannot establish that a free-form model trace changed only one fact; the edit manifest records offsets, exact old/new text, the fact and predictions, and a reviewer. Basic crossed runs can run before this review, but cannot satisfy G3.
- Numerical pilot thresholds are not invented as proposal requirements. Record the intended effect/precision criteria during calibration and have the named gate reviewer judge the measured evidence.
