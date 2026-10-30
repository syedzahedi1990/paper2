# Execution and evidence integrity addendum — 30 October 2026

This records corrections made after the replacement pilot had started. It adds implementation and provenance checks, not an outcome-based exclusion, scientific gate decision, change to rewards, or new empirical pass mark. The source proposal and revised Word document remain historical originals; read them with the [model amendment](2026-10-29-model-and-profiling.md) and [post-outcome completion/KL amendment](2026-10-30-natural-answer-and-kl.md).

## Registry and chronology

The public development-panel exporter previously still described the original 24 Qwen candidates with two schedules. The actual replacement run used a separately frozen 12-candidate Mistral registry. The public default now generates those twelve configurations: three pilot arms, four learning-rate multipliers, constant schedule, no warmup, natural completion v2 and beta 0.04 for RL only. Explicit `--design legacy_proposal` exports the original specification for historical inspection; it cannot authorize a current launch or current G1 approval.

All twelve exported configuration files were compared byte-for-byte with the actual frozen registry's files. Their canonical filename-to-SHA256 map has digest `143dbf818047745cac873ddfe3a46515ff4c69ad38b47eff08f6767b373d4335`. The original registry's file SHA256 is `37f362d2aafd9a097bac278ef9f9b618c9bdc7d1f3c269a017b6e967ec64acc0`. The public selection manifest and operational registry are different documents because the latter also records execution authority and resources; their entire bytes are not claimed to match. The family selection rule is unchanged.

The twelve-run constant-schedule restriction was already frozen in the 29 October resource decision before full candidate learning. The 30 October post-outcome amendment retained that restriction while changing completion handling, punctuation and KL. It is inaccurate to describe the restriction itself as first chosen after observing the collapsed candidate.

## Frozen execution versus future launches

The existing replacement profiles and active panel execute commit `59f4215401a33f8807dc8c550aad0436cfed6b5d`, tree `198223309584e24ddaf8c53d5969c2441d94dd0a`. Their external execution binding pins that checkout and its actual source/runtime inventories. This corrective publication is not installed into that active checkout. Existing profiles remain measurements of that frozen implementation; they do not become measurements of the new code.

For future production entry through `run_training`, the profile must contain an exact source identity matching the launch: Git commit, tree, actual Python source/configuration file hashes and tracked-change state. Missing identity or any mismatch fails before model loading and requires profiling the intended checkout. The new guard is not a retroactive rewrite of old profile JSON and offers no automatic compatibility waiver. A source change, including these corrections, cannot silently borrow a profile from the preceding checkout. Resource-only diagnostics using preserved predictions do not change the training execution identity.

Current launch checks also require the amended model/revision, completion protocol, RL/SFT beta, schedule and model amendment. Checked-in confirmatory configurations use that design but contain provisional rates: they are neither selected settings nor permission to start confirmation. Historical G1 inspection explicitly reports noneligibility; even a recorded historical pass cannot approve the current protocol.

## Length evidence without rewriting observations

The original `summarize_trace_health` v1 output remains unchanged so existing exact comparisons remain meaningful. `paper2 trace-health` produces an exclusive, hash-bound additive report, optionally verifying every original v1 checkpoint summary first. Current G1 verification reports the additive lengths alongside its verification of the original measurements.

Report actual sampled completion lengths, trace character and whitespace-word lengths, distributions, and whole-story intervals for the mean at every native checkpoint. With a pinned tokenizer, also report exact sampled trace-prefix counts when available, lower/upper prefix bounds when a token crosses the trace/answer boundary, and a separately labelled re-tokenized trace-length proxy. A token such as `".\n"` cannot truthfully be split into generated trace IDs; those IDs remain unavailable where no exact boundary exists. Completion lengths still come directly from saved sampled IDs. Empty and short traces are outcomes, not automatic failures.

## Observed-token editability job and resources

The public [G1 editability workflow](../G1_EDITABILITY.md) extracts the selected reasoning run's actual 120 saved native completions at each of nine checkpoints: **1,080 contracts**, not the generic 12-world pilot subset. It does not regenerate those traces under a different attention implementation. Missing source prompt IDs are recorded as unavailable; these artifacts are explicitly forbidden from Test 2 readers or source-continuation regeneration.

The editor model, revision, prompt, decoding and audit sample are frozen before editor calls. Generation uses a durable resumable journal; completed failed/noneditable classifications remain in denominators. A bounded editor profile can pause after a declared number of calls and resume the same contracts without resampling. The resource report distinguishes the fixed maximum of **2,211,840 editor output tokens** from measured timing/token counts and a descriptive projection. Before actual calls, measured time and money remain unknown. These editor costs were not included in the training-only panel estimate; a separate measured resource decision within the remaining overall budget is required before the full job. Reports do not issue spend approval.

Finalization constructs the actual receipt consumed by G1 from the complete generated batch. It does not decide G1 and is not a human audit. G3 still needs a named human to inspect the frozen random sample. No editor output or human decision is claimed by these software tests.

## Protocol boundaries and interpretation

`evaluate`, `traces` and `components` require an explicit completion-protocol argument. Current evaluations must state `natural_answer_cap_v2`; historical reproduction must deliberately request v1. Analysis rejects mixed v1/v2 inputs, including baseline comparisons. Missing completion provenance is treated only as historical v1. OpenToM retains its separate official/adapted protocol and scoring, rather than being falsely labelled with either controlled-world completion protocol.

The complete final-step learning curves and nearby retained immediate checkpoints accompany the unchanged single-step selection rule. The interrupted second legacy candidate has no observed late-training outcome, and cannot show whether it would eventually collapse. The three simultaneous post-outcome changes do not identify their separate causal contributions. A bare valid `Answer: X` can still earn full reward; descriptive length and actual editor evidence make that outcome visible without changing the research objective.
