# Frozen traces and controls

Both readers consume identical saved text. Each source checkpoint generates its traces and any regenerated edit continuation once. The reader command never regenerates a supplied trace. Source, prompt, checkpoint, plan, audit and continuation hashes remain bound; incomplete required artifacts fail integrity checks.

`question_dependent` generates a trace for each question. `direct_fixed` generates the rewarded direct-question trace before later consequences are revealed and holds it fixed for all eight views. `prequestion` is a legacy alias for `direct_fixed`, not a novel summary prompt. Primary confirmatory edits use `direct_fixed` and all 120 stories. Question-dependent edits are supplemental if compute permits; the two timings are always reported separately.

The seven proposal controls are own-trace replay, paraphrase, targeted semantic edit, answer cue, queried-belief summary, irrelevant edit and no trace. Minimal and source-regenerated semantic edits are reported separately. Own replay has two diagnostics:

- `own_replay` feeds back exact generated prefix IDs and compares the next token with the source's ordinary continuation.
- `own_rerendered` rebuilds the reader prompt from decoded trace text, then compares the next token and prefix equality with ordinary continuation.

Forced delimiters and consequence questions with a different native prefix are ineligible for ordinary replay. Report eligibility and reasons; prefix equality and answer equality are distinct outcomes.

The queried-belief summary is one oracle premise: the focal character path believes the focal object is at its true location. It is the same sentence across views. Other physical locations, other characters' states and answer labels are not in the supplied text. Per-view expected answers remain metadata for validation. This tests use of a correct premise, not a model-generated summary; story/question information is still needed for other consequences. The answer cue directly states the evaluated answer and is a distinct positive control.

Model-generated controls edit the belief actually stated, even if wrong. Before/after oracle forecasts for that stated belief are distinct from world-truth accuracy. Paraphrases preserve uncertainty and mistakes. Missing, contradictory or ambiguous beliefs, absence of a safe edit, and operational failures remain explicit reported outcomes. Existing answer statements may be preserved; new answer statements cannot manufacture an edit effect.

[The edit workflow](EDIT_GENERATION.md) freezes the editor and random audit, retains failed judgments after correction, and reports flaw-rate intervals without numerical acceptance limits. Comparable text length is reviewed and ratios recorded, not screened at an invented ratio. Known sampled faults need correction/re-review; unsampled controls remain individually unapproved. Complete panels, immutable bindings and completed judgments are integrity requirements.

Conditional edit estimates use all eligible artifacts from the 120 candidates, with no required count or precision target. Report per-arm editability/reasons, effects and whole-story intervals, plus common-support comparisons. Average seeds inside each story before 20,000 resamples. A wide interval is a result. Plot both measured axes with intervals; candidate quadrant readings are not categories. Follow the [preregistration](PREREGISTRATION.md) for the primary estimand and positive-control interpretation.
