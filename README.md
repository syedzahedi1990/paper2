# Paper 2

**What Does Reasoning RL Change in Belief Tracking? Transfer, Traces and the Use of Story Information.**

Implementation of the 27 October 2026 proposal, following [Paper 1 / v5.5](https://github.com/syedzahedi1990/Beyond-the-fitted-Scope---Causal-keys). The original proposal is preserved in [docs/source_proposal.docx](docs/source_proposal.docx); a searchable extraction is in [docs/proposal.txt](docs/proposal.txt).

This repository supplies the controlled data, four LoRA training arms, three experiment runners, analysis, and pilot review gates. It does **not** contain trained research checkpoints or empirical claims that reasoning RL improves belief tracking. Tiny random-model integration tests establish that the software executes; G0–G4 still require actual development measurements.

The review corrections use **corpus schema v3 and shortcut audit v4**. Regenerate earlier manifests and repeat G0. The audit includes every focal-event position, location frequency, last jointly observed event, first/last/initial/cue rules, and both signs of nested/first-order relations. Remaining weak heuristics are reported as failed screens and require an explicit corpus-bound G0 decision; they are not labelled near chance. See [the correction record](docs/REVIEW_FIXES.md) and [validation evidence](docs/VALIDATION.md).

## Start here

Python 3.10–3.13 is supported; Python 3.12 is the tested environment. A CUDA GPU is needed for practical 3B/7B training. CPU tests download no pretrained weights.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e '.[train,test]'
pytest -q
paper2 generate --output data/generated/pilot-audit-v4
paper2 verify-data --data data/generated/pilot-audit-v4
paper2 init-gates --output runs/review.json
```

This is **Paper 1’s world, a new story distribution**; the eight-question scores are not directly comparable with Paper 1’s panel. [Proposal departures](docs/PROPOSAL_DEPARTURES.md) gives the design changes and limitations to disclose in the paper.

The default generator writes 2,000 training worlds shared by clean and shortcut twins, 120 development worlds, and 120 confirmation worlds. Only direct-belief rows enter training. The seven other views are never rewarded. The full dataset is regenerated locally instead of checking repetitive generated artifacts into Git.

Read [the runbook](docs/RUNBOOK.md) for calibration, profiling, training, all three tests and the external check. [The implementation map](docs/PROPOSAL_MAP.md) traces proposal requirements to executable code. [The data protocol](src/paper2/data/DATA_PROTOCOL.md) identifies the reused v5.5 oracle and the explicit depth-two extension. [Training notes](docs/TRAINING.md) document the loss masks, traces, generation limits and compute ledger.

## Experimental design

| Stage | Model | Arms | Datasets | Seeds |
|---|---|---|---|---|
| Pilot | Qwen2.5-3B-Instruct | Reasoning RL, answer-only RL, label-only SFT | Clean | 1 |
| Confirmatory | Qwen2.5-7B-Instruct | Label-only SFT, reasoning SFT, answer-only RL, reasoning RL | Clean and shortcut | 1, 2, 3 |

Every trained arm is compared with its untrained base. GRPO uses groups of eight and a 384-token reasoning cap. LoRA targets are fixed to `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj`. Reasoning SFT receives symbolic intermediate belief states; outcome RL receives only the direct-answer correctness reward. This supervision difference is recorded, not treated as a matched-information comparison.

1. **No-trace transfer:** immediate, single-token answers after `Answer:` across all eight views; ordinary task labels, preservation, cue dependence and joint correctness.
2. **Crossed traces/readers:** both readers consume traces from both sources. Each reader's benefit is the fraction of its own baseline errors repaired, with regressions reported separately. Confirmatory edits use one native direct-question trace per world/source, frozen across all eight views (`direct_fixed`). A fixed, preregistered editor model generates paraphrases and irrelevant edits around oracle-proposed fact edits; a preregistered random human audit checks the batch. Question-dependent edits remain pilot or supplemental diagnostics. Both raw and re-rendered replay, source-generated frozen continuations, answer cues, symbolic belief summaries and no trace are retained. See [the edit protocol](docs/EDIT_GENERATION.md).
3. **Component swaps:** within each checkpoint, critical-token keys, values and residual outputs are swapped between matched event pairs. All selected layers are swapped simultaneously per component; optional single-layer diagnostics are separate. Self-swaps, signed-permutation displacement nulls and natural donor inputs are recorded. `--g4-test-layers` declares an interior band measured for K, V and residual together. G4 requires a residual control on exactly that band, bound to frozen calibration settings; arbitrary interior singles cannot certify another band. Full-depth diagnostics remain available. Secondary cross-model swaps require identical token prefixes.

Use unconditional and development-selected matched-direct-accuracy comparisons. No-trace transfer establishes direct-answer transfer, not the location of a stored belief. Trace interactions are not reported as a fraction of training gains “stored” in text. Component results use explicit event-rewrite forecasts; Tests 1 and 2 retain ordinary task targets.

## Reproducibility and compute

Training artifacts include the configuration, resolved model revision, data hashes, adapter checkpoints, learning curves, prompt exposures, sampled completions, generated/optimized tokens, GPU time, peak memory, truncation and checkpoint overhead. Data manifests bind story splits and cue twins. Keep confirmation stories untouched while calibrating difficulty and selecting checkpoints on development stories.

RL measures immediate accuracy and retains adapters every 50 steps, while native trace evaluation remains every 250 steps. A 2,000-step run therefore has 41 matching snapshots and nine native panels, including step zero. Matching retains its ±0.02 window and can still return no match. Profiling accounts for both evaluation schedules and adapter storage; model-generated edit controls and Tests 1–3 require additional compute.

The proposal's £500 ceiling and the 14 December 2026 gate review are recorded in [the protocol](docs/PROTOCOL.md). The CLI does not rent hardware, buy credits, launch paid jobs or schedule future work. Real 7B runs require reviewed G0–G4 evidence and a recorded cost approval. Profiling estimates replace the proposal's preliminary compute assumptions.

## Source and attribution

The epistemic state engine is copied byte-for-byte from the author's local v5.5 experiment bundle, with hashes in `src/paper2/data/provenance.py`. First-order ordinary targets and six forecast accounts are checked against the public v5.5 release. No license has been invented for the original research code or proposal. Model weights retain their original model licenses; they are not distributed here. OpenToM is imported from its [official repository](https://github.com/seacowx/OpenToM) for evaluation only.
