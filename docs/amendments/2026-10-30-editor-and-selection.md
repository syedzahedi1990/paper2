# Editor and selection implementation, 30 October 2026

This addendum was written after superseded pilot outcomes and the first amended candidate's early development results were observed. It is prospective with respect to **editor outputs and amended candidate failures**: neither actual study-editor outputs nor amended candidate failures had been observed when the failure policy was frozen at 15:15:23 UTC. It does not change the running twelve-candidate panel, its source, objective, scores or original selection-rule bytes. No confirmation outcomes were used. Attribution below is an automated implementation review, not a named human's scientific gate decision.

## Quoted edits and prospective calibration

New editor plans use `paper2.trace_controls.v4`. The model quotes exact `old` and replacement `new` text for the targeted and irrelevant edits. Code finds the unique literal occurrence and derives character offsets. Zero or multiple occurrences fail with separate causes; code never guesses a span. Accept a bare JSON object or one complete Markdown JSON fence. Canonical edit contracts still record exact offsets, changed facts and oracle forecasts. Historical v3 prompts, plans, offsets, raw outputs, journals and scores remain reproducible under v3.

Report every generated contract's status and failure cause, with the full attempted denominator, and retain source length beside the outcome. Operational failures are not evidence that the source model omitted a belief. Noneditability and a wrong stated belief remain separate outcomes. There is no editability or prose-length pass mark.

Before the first study editor call, run the fixed Qwen2.5-7B editor on twelve symbolic oracle traces selected deterministically from the clean **training** pool, using SHA256 ranking with seed 2026. Bind the full 2,000-story training file, training-world file, corpus manifest and current source. Use the frozen editor revision, v4 prompt, greedy SDPA generation and 2,048 output tokens. Preserve raw replies, exact quotes, resolved spans, timing, token counts and all failures; report them for review without a numerical acceptance cutoff. This diagnoses mechanics on known premises, not semantic validity on model traces. The calibration sources cannot enter Test 2 or G1 as observations of the source model.

## Executable selection and failed attempts

The original score rule is unchanged: final native accuracy on the same 120 development direct questions; maximize the mean of reasoning RL and answer RL for a shared RL setting; maximize label-SFT accuracy for the shared SFT setting; break ties by lower learning rate and then no warmup. A correctly answered empty trace can win. Trace health, editability, earlier peak scores and confirmation data cannot change the winner. Show the full saved curve around the final measurement to expose instability.

The separate prospective failure policy is `terminal_attempts_complete_settings_v1`:

- All twelve frozen candidates need actual terminal attempts. Running or unattempted candidates block selection.
- A shared RL setting is eligible only if both RL arms complete valid full-exposure runs. A SFT setting requires a valid completed label-SFT run.
- Retain every failed attempt, terminal status, reason, launch log and artifact receipt. Report excluded settings without assigning failed runs invented zero scores.
- At least one complete setting per objective family is necessary to compute a selection. There is no retry, favorable replacement, omitted unattempted candidate or expanded search.

The operator's dated failure-policy receipt was frozen before any amended failure. It is separate from the already frozen registry, so it does not rewrite history. The current supervisor remains fail-fast; this addendum does not resume or replace a failed attempt. If it stops, remaining unattempted candidates need a separately documented execution decision within the original budget and lease before selection can proceed.

`paper2 select-panel` follows original source/config/corpus bindings, supervisor events, process outcomes, retained adapters, full exposure and complete scheduled predictions. It recomputes final scores and writes an immutable selection receipt with all exclusions and curves. `verify-panel-selection`, G1 input preparation and confirmatory launch recompute it. A manually named winner is insufficient. `selected-confirmatory-configs` exports all 24 configurations using those selected family settings; exporting does not approve training. Confirmation, including its resource profiles, must bind the receipt, the same corpus and the three selected G1 learning curves.

## Inspection, resource accounting and separate Test 2 evidence

`g1-editability inspect-completed` inspects the first finished amended reasoning candidate before panel selection. It verifies actual lineage, all 1,080 saved native completions and all nine original native adapters, but produces no editor plan or G1 eligibility. Keep the original absolute run paths and adapters until inspection and later preparation are finished. A running candidate cannot be presented as this completed check.

G1 editability is an additional descriptive diagnostic, not a proposal pass mark. Its selected-run workload remains 1,080 editor calls. Test 2 requires a separate batch with full original generation provenance, source-model continuations and its own random human audit. Neither calibration nor G1-only sources can substitute for it. Both editor workloads and review time must be costed separately.

Finish this code change before making future profiles. Changed `src/paper2` bytes invalidate reuse of earlier profiles for new-source launches; retain those receipts as evidence for the frozen running panel. No code is installed under that active process, and no running job is restarted by this amendment. Existing original outputs, failed attempts, matching snapshots, final adapters and backups remain preserved.
