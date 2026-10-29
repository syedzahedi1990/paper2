# Latest execution patch — 29 October 2026

**422 tests passed** on the pinned CPU stack after the model-plan and Mistral architecture changes. Ruff and `git diff --check` pass. [Patch evidence](validation/model-amendment-validation.json) records the source hash and scope. This includes real tiny-Mistral optimizer/reload tests and validation against the three actual completed capability receipts; it does not establish real-model training feasibility or any later empirical gate.

The following release certificate is retained as historical evidence for v0.6.3. Its no-torch counts and wheel checksum have not been regenerated for this execution patch.

# Validation status — version 0.6.3

**382 tests passed**, with zero failures or skips, on Python 3.12.14 and the pinned CPU training stack. Without PyTorch, **345 tests passed** with 13 expected optional-model skips. Both environments include the optional plotting dependency. Ruff, dependency checks, wheel build and all **45 documented CLI commands** passed. [Software evidence](validation/software-validation.json) records versions, counts and hashes. This release was tested on macOS; Windows CI is configured, not claimed as locally executed.

Version 0.6.3 aligns capability scoring with the exact uninterrupted G1 procedure, validates SFT profiles without invented RL rollout measurements, and accounts for the actual training device on CUDA-capable hosts. Prior H3/provenance and durable-resume coverage remains. Mock runner tests do not constitute capability measurements. The full pinned-stack suite takes 69.38 seconds locally; the core suite takes 56.46 seconds.

Version 0.6.1 added coverage for all 24 development-panel configurations, exact exported file hashes, unchanged exposure, all-arm G1 completeness/provenance, pointwise and paired learning intervals, and preservation of pending reviewer decisions. Flat learning is valid evidence without a gain cutoff.

Tests include genuine tiny random-Qwen training in all four arms, immediate/native checkpoint schedules, pre-RoPE swaps, complete primary bands, architecture-relative G4 policy, secondary cross-model CLI use, and paired before/after effect intervals. New coverage checks automatic matching with reported gaps, no numerical heuristic/audit/precision gates, queried-belief-only premises, all-candidate editability intervals, seed-within-story weighting, and per-source audit contamination bounds. Exhaustive small-population sampling verifies one- and two-sided audit interval coverage. Known sampled faults require recorded repairs; even a 100%-flawed sample can be completed and reported after repair without a flaw-count cutoff.

In v0.6.0, synthetic figures were rendered and visually inspected; their source estimates, intervals and sensitivity metadata are retained in sidecars. They are test fixtures, not research results. The 15 PEFT warnings concern the in-memory fixtures' absent pretrained configuration; one Transformers warning uses numerical checkpoint ordering when modification times are unreliable.

## Corpus and tokenizer

The unchanged audit-v5 corpus was regenerated and tokenizer-audited in v0.6.0; these checks are retained, not claimed as new runs for v0.6.1. It contains **2,000 training, 120 development and 120 confirmation worlds**, with 17 verified files and no overlapping world, family or rendered-story identities. The 17 JSONL files were byte-identical between v0.5 and v0.6.0. Version 0.6.1 changes no corpus or generator source. Its corpus hash remains:

`0f6ef303cb86d2672fde8f7ac5c641e87b32c66b747245240efe08f49868a8f4`

Integrity checks pass. Empirical calibration remains **`requires_named_review`**; there is no scientific screen or heuristic waiver. [The complete descriptive audit](validation/shortcut-audit.json) records every depth/event stratum, underlying story IDs, estimate and interval. The remaining weak rules are unchanged: training focal positions 2–7 score about 27.7–33.5%, focal-location mode 33.5–38.2%, and depth-one last-joint-observation 56.5–58.0%, against six-location chance 16.7%. Depth-two last-joint-observation is an intended semantic rule and labelled separately. These scores are reference measurements, not pass marks.

The cached pinned Qwen2.5 tokenizer confirms cue length matching across 2,000 training twin groups and 1,920 held-out world/view groups, with single-token answer checks on 4,000 training prompts. Prompts are **478–541 tokens** and oracle training traces **157–175 tokens**, within the fixed 384-token trace cap. [Data evidence](validation/data-validation.json) records exact checks and source hashes.

```bash
paper2 generate --output data/generated/release-audit-v5
python scripts/audit_corpus.py --data data/generated/release-audit-v5 \
  --output docs/validation/data-validation.json --local-files-only
```

The tokenizer must already be cached for `--local-files-only`. New audit and edit protocols need fresh frozen lineages; old approvals do not silently transfer.

## Official protocol checks and empirical limits

In v0.6.0, all **27,416** prompt renderings (13,708 official OpenToM records, immediate and reasoning formats) matched the pinned official prompt builder. Official scorer fixtures and macro-F1/story-bootstrap tests pass. This checks protocol implementation, not model benchmark performance or reproduction of Zhong's differently aggregated accuracy scores. See [OpenToM methods](OPENTOM.md).

The [preregistration](PREREGISTRATION.md) fixes estimands, 20,000 story resamples, development-only matching and named pilot review evidence. Confirmation uses all 120 candidate stories and reports editability, conditional effects, uncertainty and quality sensitivity without count, precision or quadrant thresholds. Zhong rates remain reference candidates, with a prospective ×1/3/10/30 LoRA panel and unchanged family selection rule. G1 reviews learning in all three pilot arms; optimization differences and the panel’s added compute remain disclosed.

**This software-validation suite performs no pretrained research training or paid compute.** Separately authorized Vast setup and capability measurements are recorded in `runs/vast-20261029`; this local test certificate does not establish their outcomes. Learning, editor quality, replay, residual sensitivity and G0–G4 decisions still require actual pilot evidence. Training/evaluation/storage/editing costs must be profiled; the proposed pilot hours remain unverified. The £500 ceiling and 14 December 2026 review remain intact.
