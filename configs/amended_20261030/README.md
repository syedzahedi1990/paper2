# Amended pilot reference configurations — 30 October 2026

These prospective reference settings use the [post-outcome amendment](../../docs/amendments/2026-10-30-natural-answer-and-kl.md): natural 384-token completion budget, optional terminal period and beta 0.04 for both RL arms. Each run initializes from the exact pinned Mistral 24B base. Original proposal configurations remain reproducible through the explicit `matrix --design legacy_proposal` path; preserved run artifacts are not rewritten.

These are not a cost approval or a frozen candidate registry. Measure fresh same-protocol profiles, then freeze a separately costed ×1/3/10/30 constant/no-warmup panel across all three arms before candidate learning. The existing old panel is excluded from selection. Confirmation remains gated.

Generate the complete amended candidate configs and hashed selection manifest with `paper2 matrix --stage pilot --development-panel --output NEW_DIRECTORY`. It exports all 12 configurations without launching them. The default ordinary pilot and confirmatory matrices also use the amended design.
