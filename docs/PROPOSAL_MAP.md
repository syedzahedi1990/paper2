# Proposal implementation map

The [source proposal](proposal.txt) is authoritative. The [preregistration](PREREGISTRATION.md) fixes estimands and analyses; the [departure record](PROPOSAL_DEPARTURES.md) explains deliberate changes. A code test is not an empirical pilot gate.

| Requirement | Implementation | Evidence or limitation |
|---|---|---|
| Paper 1 world and oracle | `data/_vendor`, `data/world.py`, `data/provenance.py` | Preserved state engine/vocabulary; new story distribution disclosed |
| 2,000/120/120 disjoint stories and shortcut twin | `data/corpus.py` | Frozen hashes, exact cue rates and entity/length checks; audit v5 descriptive heuristic estimates/intervals |
| Four trained arms, base, 3B pilot/7B final, three final seeds | `training/config.py`, `configs/` | Three pilot and 24 final configs; base separately evaluated |
| Fixed LoRA, GRPO 8, trace cap 384 | `training/runner.py`, `training/generation.py` | Tiny-model native training tests; paid performance remains unmeasured |
| Direct-only reward; oracle reasoning SFT | `training/data.py`, `data/world.py` | Held-out supervision rejected; extra oracle information disclosed |
| H1 transfer and crossed reader/source estimands | `analysis.py`, `traces.py` | Unrewarded direct inference; reader-specific error denominator; paired 20,000-story bootstrap |
| Seven controls, both trace timings | `traces.py`, `edit_generation.py` | Raw/re-rendered replay; one-belief premise; frozen source continuations; primary direct-fixed edits |
| Model-assisted edit validation | `edit_generation.py` | Frozen editor/prompt/sample; original flaws preserved; exact descriptive audit intervals; validity repairs without flaw cap |
| Editability and conditional edit effects | `precision.py`, `analysis.py` | All 120 candidates; per-arm intervals and common support; no readiness threshold |
| H2 scope, cues and whole patterns | `analysis.py` | Oracle per-world scope; per-question cue contrasts and eight-view joint correctness |
| Accuracy matching | `analysis.py:match_checkpoints` | Common target=min per-run peak; closest saved immediate dev checkpoint; actual gaps reported |
| H3 aligned primary band | `interventions/` | Joint layers 1–last, K/V/residual with own/natural/signed nulls; per-layer descriptive results |
| Official OpenToM external evaluation | `opentom.py` | Official prompt parity and subtype macro-F1; adapted secondary; Zhong numerical replication not claimed |
| Figures with uncertainty | `plotting.py`, CLI `plot` | Both axes with intervals; heuristic reference lines beside model accuracy; no automatic quadrant labels |
| G0–G4 and budget review | `gates.py`, `docs/PROTOCOL.md` | Named pilot decisions and immutable evidence; structural validity retained; £500 ceiling |
| Compute accounting | `training/ledger.py`, `runner.py`, edit plans | Training/evaluation/storage/generation counted separately; empirical whole-study cost pending |
| Conditional Bridge | Separate proposal | No automatic launch or claim of completion |

Zhong’s rates are reference candidates; its LoRA/full-fine-tuning parameterization is unspecified. The ×1/3/10/30 development panel and unchanged selection rule are in [TRAINING.md](TRAINING.md), and G1 requires all three pilot arms’ learning evidence. Story exposure is matched; optimizer steps differ. Final comparisons include all three fixed seeds and both unconditional and development-matched analyses. The new audit/summary protocol requires fresh frozen artifacts; historical certificates cannot authorize changed content.
