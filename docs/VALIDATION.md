# Validation status — corrected implementation

Version 0.2.0 passed **156 tests**, with zero failures or skips, on Python 3.12.14 and the pinned CPU training stack. A separate environment without PyTorch passed **125 core tests**, with 11 expected optional-model skips. Ruff, dependency consistency and wheel build passed. [Software evidence](validation/software-validation.json) records versions, counts and hashes. Windows CI coverage was added; it was not run locally on this Mac.

The tests exercise genuine forward/backward and optimizer calls in all four tiny random-Qwen LoRA/TRL arms, actual prompt-pool coverage, native/immediate curves, preserved evaluated adapters, simultaneous two-layer swaps and the cross-model CLI. They also cover changed/intact/joint consequence analysis, fixed-seed story bootstrap, source-frozen continuations, raw and re-rendered replay, trace tampering, edit subsets, optional imports and stale calibration rejection. The 12 PEFT warnings concern the deliberately in-memory synthetic model having no pretrained configuration file; no tests were skipped in the full stack.

## Regenerated corpus

The schema-v3 corpus contains **2,000 training, 120 development and 120 confirmation worlds**, verified through 17 hashed files, with zero world, family or rendered-story overlap. Its hash is:

`fe78e3b08542526395ae34a94f4de336ec2aebf85ad49c91962bdc74ddcdbe20`

All split/depth/event strata pass the structural shortcut audit, including the inverse nested-belief rule. Every 30-world held-out depth-two/event-kind cell has the preregistered joint relation counts: **1 equal to both, 4 equal only to the outer character, 4 equal only to the inner, and 21 different from both**. This is the rounded six-location independence design, not an “always different” construction. The audit checks exact planned counts and candidate-exclusion accuracy, as well as first-event, last-event, initial-location and clean-cue accuracy. The held-out clean cue agrees in exactly 5 of 30 worlds in each split/depth/event cell. The deliberately predictive shortcut cue remains 90% during training.

[Data evidence](validation/data-validation.json) and the [full stratified shortcut audit](validation/shortcut-audit.json) retain actual counts, scores, null expectations and screening bounds. Small-cell uncertainty is not presented as proof of the absence of every possible shortcut. G0 must still inspect model difficulty and failure types.

Using the cached, pinned Qwen2.5-3B tokenizer, all 2,000 training twin groups and 1,920 held-out world/view groups have matched cue-rendering lengths. All 4,000 training prompts pass single-token answer checks. Native reasoning prompts are **478–541 tokens** (limit 2,048); symbolic SFT traces are **157–175 tokens** (cap 384).

Reproduce the structural and token audit without loading model weights:

```bash
python scripts/audit_corpus.py --data data/generated/pilot-v3 \
  --output docs/validation/data-validation.json --local-files-only
```

The tokenizer must already be cached for `--local-files-only`; omit it to download only tokenizer files. Earlier schema-v2 data and earlier shortcut-audit versions cannot reuse G0 approvals.

## External evaluation and remaining measurements

The OpenToM adapter was checked against the pinned official source and all 13,708 imported records. Its option-aware scorer validates every gold label; the documented upstream diagnostic preserves 25 official lexical self-scoring failures rather than concealing them. See [OpenToM protocol](OPENTOM.md) for adaptations and regression coverage.

No pretrained 3B/7B research experiment or paid compute was run. GPU feasibility, actual learning, trained-model replay, component positive controls, precision and empirical G0–G4 decisions remain pilot measurements. These results establish software and structural-data checks, not the paper's hypotheses.
