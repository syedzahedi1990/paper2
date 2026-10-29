# External OpenToM check

The primary external check now uses the **official OpenToM system and user prompt text and lexical scoring rules**, with **macro-F1 reported for each question subtype**. The former Paper 2 wrapper and option-aware scoring remain a separate secondary protocol. Neither is training material.

The imported normal-length file contains 13,708 questions from 596 narratives, pinned to OpenToM revision `3f22b66276b2d7ca5fe573c28c79cc0d077aafc5`; SHA-256 `11de58a9950e09d13f08df14dd595c101d4b5c183ed1ac5d0647d181c3676bf2`. The dataset and prompt templates are attributed to [seacowx/OpenToM](https://github.com/seacowx/OpenToM/tree/3f22b66276b2d7ca5fe573c28c79cc0d077aafc5), CC BY-NC 4.0. Record the exact evaluation rows and model revision with each result.

## Primary official protocol

`paper2 opentom-evaluate --opentom-protocol official` is the default. `--reasoning` selects the published zero-shot CoT wording; without it, use the published immediate-answer wording. The implementation preserves the official normal-narrative templates, type-specific options, attitude qualification, system message, and even the CoT builder's punctuation operations. Fine-location prompts do not add a list of candidate locations. Qwen's pinned tokenizer serializes the messages; the OpenToM repository does not provide a Qwen-specific wrapper.

The official CoT completion is generated once, uninterrupted, with a 384-token maximum. It receives no Paper 2 `Reasoning:` prefix or forced `Answer:` continuation. Immediate completions use a 32-token maximum and newline stopping. For CoT, the final line (or the text after `Therefore` when no newline occurs) is passed to the official lexical scorer. Raw generations, parsed answers, prompt messages and hashes, truncation, model identity and format are retained. These generation limits are Paper 2 implementation choices and must accompany results; truncation rates are reported, not hidden.

The primary score reproduces the official report's macro-F1 definition: average over labels present in targets or predictions **after excluding invalid gold labels and unparsed `-1`/`None` predictions**. The excluded count, invalid-output rate, and accuracy with the full denominator are always shown beside it. Fine-location ties are class 3 and count as errors, as upstream. Report coarse/fine location separately by ToM order, fullness/accessibility separately by order, and attitude. The official combined multihop score is supplied additionally. Do not pool coarse and fine location integer labels or report one arbitrary F1 across unrelated label spaces. Different checkpoints, formats and evaluation protocols are kept separate automatically.

Intervals use 20,000 whole-narrative bootstrap resamples, retaining all questions from each sampled story. Undefined F1 resamples are counted explicitly. External results are reported separately from the controlled-world H1–H3 estimands.

This aligns prompt text and scoring with OpenToM, but **does not reproduce every published sampling procedure**: Paper 2 evaluates the full normal-length file, whereas the upstream runner uses sampled story batches and fullness-question subsampling. Do not label the result a replication of published benchmark numbers.

## Relation to Zhong et al.

[Zhong et al., §4.1 and Appendix E](https://arxiv.org/html/2606.09092v1) describe accuracy and a macro average by ToM order for OpenToM, rather than stating that macro-F1 is their primary metric. They also use an OpenToM training/test selection. Their [official repository](https://github.com/jkz-338/Robust-ToM-RL) contained only a README when inspected on 29 October 2026; their exact evaluation prompts and selected item IDs could not be verified. Official OpenToM macro-F1 therefore improves benchmark alignment but **does not establish exact numerical comparability with Zhong et al.** Keep accuracy and subtype/order information available, and disclose the different training data, sample and generation settings when discussing their results.

## Secondary adapted protocol

Run `--opentom-protocol adapted` separately to retain the former shared Paper 2 system message, `Answer:` handling and question-derived option lists. Its `opentom_adapted_v2` results use option-aware canonical scoring (`opentom_options_v2` imported row schema), with invalid answers counted as wrong. They remain labelled secondary. Scoring an official-prompt output with the adapted scorer is retained only as a diagnostic; it is not a substitute for running the adapted prompt.

The adapted scorer resolves exact fine-location option matches before lexical overlap and normalizes one metadata typo, `center of dinint table` → `center of dining table`, independently of the gold label. The official path preserves raw metadata and official behavior. Both retain every raw gold string. The known 25 official gold-self-score failures remain visible: 19 overlap ties, five article-removal cases involving “another drawer”, and one metadata/gold mismatch. This distinction cannot be silently repaired in the primary official score.

## Software verification

All **27,416** rendered user prompts (13,708 records × immediate/CoT) matched the pinned official `build_prompt` output exactly. The offline fixture `tests/fixtures/opentom_official_parity.json` was produced from that pinned builder and scorer; it records source hashes and covers all five answer kinds in both formats. Tests also verify the single official generation call, CoT parsing, macro-F1 exclusions, whole-story intervals, and separation of readers/protocols. The earlier complete lexical audit covered 68,540 output comparisons with zero label mismatches. These are software checks, not model benchmark results.

Sources:

- [Official templates](https://github.com/seacowx/OpenToM/tree/3f22b66276b2d7ca5fe573c28c79cc0d077aafc5/src/prompts/chatgpt_opentom_prompts) and [prompt builder](https://github.com/seacowx/OpenToM/blob/3f22b66276b2d7ca5fe573c28c79cc0d077aafc5/src/utils/build_prompt.py).
- [System message and CoT postfix](https://github.com/seacowx/OpenToM/blob/3f22b66276b2d7ca5fe573c28c79cc0d077aafc5/src/utils/load_baseline_model.py).
- [Lexical scorer and CoT parser](https://github.com/seacowx/OpenToM/blob/3f22b66276b2d7ca5fe573c28c79cc0d077aafc5/src/evaluate/opentom_evaluator.py) and [macro-F1 reporting](https://github.com/seacowx/OpenToM/blob/3f22b66276b2d7ca5fe573c28c79cc0d077aafc5/src/evaluate.py).
- [Official sampling runner](https://github.com/seacowx/OpenToM/blob/3f22b66276b2d7ca5fe573c28c79cc0d077aafc5/src/run_baseline.py).
