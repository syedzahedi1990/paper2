# External OpenToM check

Paper 2 imports the official **13,708 question, 596 narrative** normal-length `opentom.json` at revision `3f22b66276b2d7ca5fe573c28c79cc0d077aafc5`. The download's SHA-256 is `11de58a9950e09d13f08df14dd595c101d4b5c183ed1ac5d0647d181c3676bf2`. Data remain external evaluation material and are not included in training.

This is an **adapted external check**, not a reproduction of the published OpenToM prompt and sampling procedure. Paper 2 uses its shared chat wrapper and answer delimiter for immediate and reasoning formats, evaluates the full normal-length file, and does not reproduce the upstream repeated 50-story samples or subsampling of fullness questions. Scores must be labelled accordingly and reported separately from controlled-world scores.

Answer options are constructed before looking at the gold answer:

| Question | Options | Source |
|---|---|---|
| Coarse location | Yes / No | “initial” in the location question |
| Fine location | Original place / destination | `plot_info.original_place`, `plot_info.move_to_place` |
| Fullness | More / equally / less full | Fullness question |
| Accessibility | More / equally / less accessible | Accessibility question |
| Attitude | Positive / neutral / negative | Attitude question |

Attitude prompts retain the upstream “assuming that you observed the action” qualification. Question text and metadata determine every option list; changing only the gold answer leaves the evaluation prompt unchanged. Evaluation recomputes the options and rejects mismatched imported lists.

`opentom_options_v2` is the primary scoring protocol. It maps answers to the question's canonical labels and retains fullness synonyms such as “emptier” and “fuller”. Fine-location answers first match an entire option (ignoring an article and formatting), then fall back to the upstream lexical-overlap rule. Overlap ties and unparsed or contradictory categorical outputs are invalid and count as wrong. Coarse yes/no recognition uses whole words. Every raw model answer and the raw gold answer remain in the output. Report accuracy, invalid-output count and macro-F1 separately for the question type and subtype; fine-location F1 uses original/destination labels, not a different class for every place name.

The pinned official scorer is also reproduced as a separate diagnostic in `upstream_*` fields. It preserves the official substring priorities, determinant-removal behaviour and fine-location ties. The summary reports that diagnostic with the upstream convention of excluding invalid gold labels and `-1` predictions. Primary scores retain the full denominator. Neither diagnostic scoring nor the adapted primary implies that the model was given the published upstream prompt.

A complete import audit found that exact gold strings fail the official scorer on **25 records**: 19 lexical-overlap ties, 5 “another drawer” cases where the official article-removal function strips the initial “a” from “another”, and 1 invalid gold label caused by the plot typo `center of dinint table`. Paper 2 records these failures explicitly. It uses one documented location normalization, `center of dinint table` → `center of dining table`, on options and answer normalization; this rule is independent of the gold label. The raw plot and gold text remain unchanged. The affected narrative has 23 questions, of which six are fine-location questions; only one has the inconsistent corrected gold spelling. All 13,708 gold labels pass primary canonical scoring after import.

Verification against the pinned official `OpenToMEvaluator` covered **68,540 comparisons** (five output forms for every question), with zero mismatches in its target/predicted labels. The scorer source used in this audit had SHA-256 `75b43a93e6927d4a02eb1a87c9e19bf3ec59489dca18ed2708f7068331ca2a94`. Unit tests cover synonyms, multiple incompatible answers, overlapping location names, typo normalization, full-denominator scoring, no gold-conditioned options, and actual prompt construction through a stub backend. This is software/scoring validation, not a model benchmark result.

Sources, pinned to the inspected revision:

- [Dataset documentation and licence](https://github.com/seacowx/OpenToM/blob/3f22b66276b2d7ca5fe573c28c79cc0d077aafc5/README.md): CC BY-NC 4.0; intended evaluation-only use; macro-F1 recommendation.
- [Official prompt builder](https://github.com/seacowx/OpenToM/blob/3f22b66276b2d7ca5fe573c28c79cc0d077aafc5/src/utils/build_prompt.py): question-type options and attitude qualification.
- [Official lexical scorer](https://github.com/seacowx/OpenToM/blob/3f22b66276b2d7ca5fe573c28c79cc0d077aafc5/src/evaluate/opentom_evaluator.py): type-specific mappings and lexical rules.
- [Official reporting](https://github.com/seacowx/OpenToM/blob/3f22b66276b2d7ca5fe573c28c79cc0d077aafc5/src/evaluate.py): accuracy, macro-F1 and invalid-output treatment.
- [Official sampling procedure](https://github.com/seacowx/OpenToM/blob/3f22b66276b2d7ca5fe573c28c79cc0d077aafc5/src/run_baseline.py): repeated samples and fullness subsampling.
