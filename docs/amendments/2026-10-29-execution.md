# Prospective execution amendment — 29 October 2026

The user supplied the revised proposal and requested execution on an existing Vast instance, with a concern about small-model capability and comparability with Paper 1. This amendment is written before research training or capability outcomes have been inspected. The revised proposal is retained as `docs/source_proposal_revised_20261029.docx` (SHA256 `22261b8fea9fb2f9d901e947757f8763aa28aa77a16131924eed759abc952634`), with extracted text alongside it. The original proposal remains archived in its original files.

## Capability evidence before a training commitment

Evaluate the same 120 clean development stories with the existing immediate-answer procedure on all eight views and the native training/G1 reasoning procedure on the direct view. Keep the 384-token trace cap. Report whole-story intervals, depth/event strata, invalid answers, trace lengths and truncation alongside the descriptive heuristic references. Do not inspect confirmation outcomes.

Prospective measurement clarification, before any capability outcomes: native reasoning uses the exact G1 evaluator, one uninterrupted greedy completion with the training trace-cap processor and reward parser. It does not stop at a delimiter and separately continue, re-render a trace, or append an answer request after early EOS. Preserve raw completion text and token IDs, parsed and reward-valid answers, trace lengths, forced-delimiter and truncation flags. A parsed correct letter in a reward-invalid completion remains incorrect. The screen meters this one native generation, including forced delimiter tokens; it does not add the Test 2 replay probe. Immediate accuracy remains the primary no-trace measurement. Inspect invalid-answer rates and native-format performance before interpreting low immediate accuracy as a capability limit.

Use these immutable untrained checkpoints:

| Model | Revision | Role |
|---|---|---|
| Qwen/Qwen2.5-3B-Instruct | aa8e72537993ba99e69dfaafa59ed015b17504d1 | Planned pilot |
| Qwen/Qwen2.5-7B-Instruct | a09a35458c702b33eeacc393d103063234e8bc28 | Planned confirmation; exact Paper 1 checkpoint |
| mistralai/Mistral-Small-24B-Instruct-2501 | 9527884be6e5616bdd54de542f9ae13384489724 | Exact Paper 1 larger-model capability reference |

This is a development comparison, not a replacement training grid or an isolated model-size experiment: Mistral changes model family too. The new story distribution prevents direct numerical comparison with Paper 1's eight-view scores. Review the measured intervals and error patterns before deciding whether to retain the planned pilot or record a justified model amendment. No numerical capability cutoff is introduced. Larger-model training would require architecture support and tests, a revised cost estimate and a prospectively recorded design.

The user's current instruction authorizes using the supplied Vast rental now; it supersedes the earlier Colab-only calendar for this preparatory execution. The initial capability batch has a three-hour process limit (about $3.98 at the displayed $1.3263889/hour rental rate, before any separate network charges). No additional instance is purchased. The £500 overall ceiling remains. This operational authority does not establish any empirical gate decision or replace the named human edit audit.

## H3 descriptive comparison

Add the revised proposal's eight arm-by-dataset points, separately for key, value and residual interventions. The horizontal axis is trained-minus-base immediate accuracy on the seven unrewarded ordinary-target views. The vertical axis is the trained-minus-base change in natural-minus-own **joint eight-view event-rewrite pattern correctness**, using the primary band from layer 1 through the last layer. This chooses a single already-declared component estimand prospectively; forecast-probability and changed/intact estimates remain separately reported.

Use complete, matched final-checkpoint confirmation panels for the four arms, two training datasets and three seeds. Average fixed seeds within each story before 20,000 whole-story resamples. Show intervals on both axes. Retain base, own and signed-null results and artifact identities in the numeric sidecar. Do not fit a regression, compute a correlation significance test, choose an axis after results, or classify arms by quadrant. Matched-accuracy comparisons remain separately identified and must bind the selected checkpoints.

## Training and review remain prospective

The capability batch does not train, set G0–G4 to pass, choose a favorable corpus or launch confirmation. Freeze any calibration and learning-rate panel choice before learning. The optional full rate panel remains 24 runs until its actual training/evaluation/storage cost is profiled. A cheaper panel, if selected, must be defined before learning; do not quietly shorten production runs or drop failed candidates. Preserve all three confirmatory seeds.
