# Validation status — version 0.4.0

**218 tests passed**, with zero failures or skips, on Python 3.12.14 and the pinned CPU training stack. A separate environment without PyTorch passed **183 tests**, with 12 expected optional-model skips. Ruff, dependency consistency and wheel build passed. [Software evidence](validation/software-validation.json) records versions, counts and hashes. Windows CI coverage is configured; this release was tested locally on macOS, not Windows.

Tests exercise genuine tiny random-Qwen training in all four arms, separate immediate/native evaluation schedules and cost accounting, simultaneous component swaps, and G4 requiring matching K/V/residual bands bound to calibration. They also cover fixed-model edit plans, strict generation and provenance checks, sampled human audit certificates, journal recovery, source-frozen continuations shared by two readers, confirmatory fixed-direct timing, full eight-view coverage, story-level uncertainty, oracle forecasts and trace replay. The 15 PEFT warnings concern the in-memory fixture's absent pretrained config; one Transformers warning falls back from checkpoint modification times to numerical ordering.

## Data audit: required controls pass; weak heuristics remain

The regenerated corpus contains **2,000 training, 120 development and 120 confirmation worlds**, verified through 17 hashed files with zero world, family or rendered-story overlap. World/corpus schema remains v3; the expanded shortcut audit is v4. This release retains the story distribution and data-row content. Its new audit-bound corpus hash is:

`b5344b9ad3ae4864b7052b32bc00c4c8dfdc5d59953d25222b33274754c50919`

First-event, last-event, initial-location, clean-cue and bidirectional nested-belief required controls pass. The preregistered relation quotas and held-out clean-cue quotas are intact. However, the expanded audit reproduces above-chance weak heuristics. Across training depth/event cells, focal-event positions 2–7 score approximately **27.7–33.5%**, focal-event location mode **33.5–38.2%**, and depth-one last-joint-observation **56.5–58.0%**, against chance 16.7%. Frequency ties receive exact uniform-choice expected accuracy. Depth-two last-joint-observation is the intended semantic solver and is reported separately.

Accordingly, the manifest says **`required_controls_passed: true`, `requires_heuristic_review: true`, `calibration_ready: false`**. No G0 decision has been supplied. A named reviewer may accept the documented weak-rule limitations with a rationale bound to the exact corpus hash; this cannot waive required-control failures or change measured scores. [Full audit](validation/shortcut-audit.json) and [data evidence](validation/data-validation.json) preserve every stratum and screen. Small held-out cells cannot establish absence of other shortcuts.

The cached pinned Qwen2.5-3B tokenizer confirms matched cue lengths in all 2,000 training twin groups and 1,920 held-out world/view groups, plus single-token answer checks for all 4,000 training prompts. Prompts are **478–541 tokens** (limit 2,048); oracle SFT traces are **157–175 tokens** (cap 384). These are this local deterministic corpus and pinned tokenizer's measurements, not substituted values from another checkout.

```bash
paper2 generate --output data/generated/pilot-audit-v4
python scripts/audit_corpus.py --data data/generated/pilot-audit-v4 \
  --output docs/validation/data-validation.json --local-files-only
```

The tokenizer must already be cached for `--local-files-only`. The report command records residual failures; it does not approve them. Old audit manifests and G0 approvals must be regenerated and reviewed.

## Precision, compute and remaining empirical work

RL native evaluation remains at nine panels per full-pool run, while immediate accuracy and adapters are retained at 41 steps. Profiles account for both evaluation formats and storage separately. The pilot and confirmatory grid retain 93 and 624 evaluated adapters respectively. The whole-pilot 15–25-hour estimate is still unverified. Confirmatory edit preparation defaults to 97 worlds for approximate per-arm 95% ±.10 rate precision, and analysis checks observed whole-story intervals. The primary confirmatory edit measurement requires frozen direct-question traces. Its default 25-source grid needs 2,425 model-generated contracts; a preregistered random human sample with checkpoint coverage audits the batch. All controls and oracle predictions are retained, and any unresolved generation or failed audit blocks use. The editor method has only been exercised with deterministic test backends, not validated for editing quality on a pretrained research model. Twelve worlds remain pilot-only diagnostics. The plan does not establish precision for arm differences, subgroups or remaining-error ratios.

No pretrained 3B/7B research experiment or paid compute was run. GPU feasibility, actual learning, replay, measured useful residual controls, sufficient precision and G0–G4 decisions remain empirical work. [Proposal departures](PROPOSAL_DEPARTURES.md) records the new story distribution, oracle-premise control, adapted OpenToM scores and unequal optimization settings that the paper must disclose. The previous pinned OpenToM scorer parity checks remain documented in [its protocol](OPENTOM.md); no model benchmark score is claimed.
