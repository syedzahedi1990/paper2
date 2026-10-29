"""Pinned official OpenToM prompts/macro-F1, with a separate adapted protocol."""
import json
import re
from pathlib import Path

from .io import digest, file_digest, write_jsonl, write_json
from .prompts import SYSTEM, ANSWER_DELIMITER

OPENTOM_REVISION = "3f22b66276b2d7ca5fe573c28c79cc0d077aafc5"
SCORING_PROTOCOL = "opentom_options_v2"  # Imported row schema, shared by both evaluation protocols.
OFFICIAL_PROTOCOL = "opentom_official_v1"
ADAPTED_PROTOCOL = "opentom_adapted_v2"
OFFICIAL_SYSTEM = "You are an expert in modeling other's mental state."
OFFICIAL_COT = "Reason step by step before answering. Write the answer in the end."
# An explicit metadata correction in the pinned file. This rule uses only the
# plot's location string; neither the question's answer nor a model output.
LOCATION_METADATA_CORRECTIONS = {"center of dinint table": "center of dining table"}
LABELS = {"fine_location": (1, 2), "coarse_location": (0, 1), "fullness": (1, 2, 3),
          "accessibility": (1, 2, 3), "attitude": (1, 2, 3)}


def question_protocol(question, plot_info):
    """Answer sets depend on the question and plot metadata, never its gold label."""
    text, kind = question["question"].lower(), question["type"]
    if kind == "attitude":
        return "attitude", ["positive", "neutral", "negative"]
    if "fullness" in text:
        return "fullness", ["more full", "equally full", "less full"]
    if "accessibility" in text:
        return "accessibility", ["more accessible", "equally accessible", "less accessible"]
    if kind.startswith("location") and "initial" in text:
        return "coarse_location", ["Yes", "No"]
    if kind.startswith("location"):
        if not all(plot_info.get(key) for key in ("original_place", "move_to_place")):
            raise ValueError("Fine-location scoring needs original_place and move_to_place metadata")
        places = [plot_info[key].replace("_", " ") for key in ("original_place", "move_to_place")]
        return "fine_location", [LOCATION_METADATA_CORRECTIONS.get(place, place) for place in places]
    raise ValueError(f"Unsupported OpenToM question type: {kind}")


def normalize_answer(text):
    text = text.strip().casefold()
    text = re.sub(r"^answer\s*:\s*", "", text)
    text = text.strip(" .!\n\t\"'")
    return re.sub(r"\s+", " ", text)


def _without_article(text):
    value = re.sub(r"^(?:a|an|the)\s+", "", normalize_answer(text).replace("_", " "))
    return LOCATION_METADATA_CORRECTIONS.get(value, value)


def _overlap(prediction, place):
    words = set(prediction.lower().replace("_", " ").replace("'s", "").replace(".", "").split())
    tokens = place.lower().replace("_", " ").replace("'s", "").split()
    return len(words.intersection(tokens)) / len(tokens) if tokens else 0.


def _label(canonical, kind, options):
    maps = {"coarse_location": {"yes": 1, "no": 0},
            "fullness": {"less full": 1, "more full": 2, "equally full": 3},
            "accessibility": {"more accessible": 1, "less accessible": 2, "equally accessible": 3},
            "attitude": {"positive": 1, "negative": 2, "neutral": 3}}
    mapping = ({_without_article(place): i + 1 for i, place in enumerate(options)}
               if kind == "fine_location" else maps[kind])
    return mapping.get(canonical)


def upstream_score(prediction, row):
    """Pinned official lexical rules, including their ambiguities, for diagnosis.

    This is a dependency-free translation of OpenToMEvaluator's five check_*
    routines. The official reporting excludes -1/None predictions and invalid
    gold labels; fine-location ties are label 3 and count as wrong, not excluded.
    """
    value, target, kind = prediction.lower(), row["answer_text"].lower(), row["answer_kind"]
    pred, gold = -1, None
    if kind == "fine_location":
        def remove_determinant(word):
            for article in ("a", "an", "the"):
                if word.startswith(article):
                    return word[len(article):].strip()
            return word
        plot = row.get("plot_info", {})
        raw_places = [plot.get(key, row["answer_options"][index])
                      for index, key in enumerate(("original_place", "move_to_place"))]
        places = [remove_determinant(place).lower() for place in raw_places]
        raw_target = remove_determinant(row["answer_text"]).lower()
        scores = [_overlap(value, place) for place in places]
        pred = 3 if scores[0] == scores[1] else (1 if scores[0] > scores[1] else 2)
        gold = next((i + 1 for i, place in enumerate(places) if place == raw_target), None)
    elif kind == "coarse_location":
        pred = 0 if "no" in value and "yes" not in value else 1 if "yes" in value and "no" not in value else -1
        gold = 0 if "no" in target else 1 if "yes" in target else None
    elif kind == "fullness":
        value = value.replace(".", "")
        pred = (1 if any(token in value for token in ("less full", "emptier", "more empty")) else
                2 if any(token in value for token in ("more full", "fuller")) else
                3 if "equally full" in value else -1)
        gold = {"less full": 1, "more full": 2, "equally full": 3}.get(target)
    elif kind == "accessibility":
        value = value.replace(".", "")
        pred = next((i for token, i in (("more accessible", 1), ("less accessible", 2), ("equally accessible", 3))
                     if token in value), -1)
        gold = {"more accessible": 1, "less accessible": 2}.get(target, 3)
    elif kind == "attitude":
        token = value.split("\n\n")[-1].split(":")[-1].split(".")[0].strip()
        canonical = {"a": "positive", "b": "neutral", "c": "negative"}.get(token)
        matches = [name for name in ("positive", "negative", "neutral") if name in token]
        canonical = canonical or (matches[0] if len(matches) == 1 else None)
        pred = {"positive": 1, "negative": 2, "neutral": 3}.get(canonical, -1)
        gold = {"positive": 1, "negative": 2}.get(target, 3)
    return {"upstream_prediction_label": pred, "upstream_target_label": gold,
            "upstream_correct": gold is not None and pred == gold,
            "upstream_gold_valid": gold is not None,
            "upstream_report_included": gold is not None and pred not in (-1, None)}


def score_opentom(prediction, row):
    """Option-aware primary scoring; official lexical rules remain separate.

    Exact option matches resolve overlapping place names. Non-exact locations
    use upstream lexical overlap; ties remain invalid. Conflicting categorical
    answers are invalid rather than credited according to substring priority.
    """
    value = normalize_answer(prediction)
    target = normalize_answer(row["answer_text"])
    kind = row["answer_kind"]
    if kind == "fine_location":
        places = [_without_article(place) for place in row["answer_options"]]
        if len(places) != 2 or len(set(places)) != 2:
            raise ValueError("Fine-location options must be two distinct places")
        candidate = _without_article(value)
        if candidate in places:
            canonical = candidate
        else:
            scores = [_overlap(value, place) for place in places]
            canonical = None if scores[0] == scores[1] else places[int(scores[1] > scores[0])]
        target = _without_article(target)
    elif kind == "coarse_location":
        matches = [name for name in ("yes", "no") if re.search(rf"\b{name}\b", value)]
        canonical = matches[0] if len(matches) == 1 else None
    elif kind == "fullness":
        synonyms = {"less full": ("less full", "emptier", "more empty"),
                    "more full": ("more full", "fuller"), "equally full": ("equally full",)}
        matches = [name for name, phrases in synonyms.items() if any(phrase in value for phrase in phrases)]
        canonical = matches[0] if len(matches) == 1 else None
    elif kind == "accessibility":
        matches = [name for name in ("more accessible", "less accessible", "equally accessible") if name in value]
        canonical = matches[0] if len(matches) == 1 else None
        if "|" in target:
            target = "equally accessible"
    elif kind == "attitude":
        token = prediction.lower().split("\n\n")[-1].split(":")[-1].split(".")[0].strip()
        canonical = {"a": "positive", "b": "neutral", "c": "negative"}.get(token)
        if canonical is None:
            matches = [name for name in ("positive", "neutral", "negative") if name in token]
            canonical = matches[0] if len(matches) == 1 else None
    else:
        raise ValueError(f"Unsupported scoring type: {kind}")
    pred_label, gold_label = (_label(answer, kind, row["answer_options"]) for answer in (canonical, target))
    if gold_label is None:
        raise ValueError("OpenToM gold label is outside the question-derived answer set")
    return {"canonical_prediction": canonical, "canonical_target": target,
            "prediction_label": pred_label, "target_label": gold_label,
            "correct": pred_label is not None and pred_label == gold_label,
            "exact_match": normalize_answer(prediction) == normalize_answer(row["answer_text"]),
            **upstream_score(prediction, row)}


def import_opentom(source, output, source_url=None):
    records = json.loads(Path(source).read_text())
    if not isinstance(records, list) or not records:
        raise ValueError("Expected official OpenToM data/opentom.json list of QA records")
    output_rows = []
    for index, item in enumerate(records):
        question = item["question"]
        if not isinstance(question, dict) or not all(key in question for key in ("question", "answer", "type")):
            raise ValueError(f"Unsupported OpenToM schema at record {index}")
        if not isinstance(question["answer"], str) or not item["narrative"]:
            raise ValueError("OpenToM needs textual narrative and answer")
        answer_kind, answer_options = question_protocol(question, item.get("plot_info", {}))
        row = {"id": f"opentom-{index:06d}", "world_id": digest(item["narrative"]),
               "dataset": "opentom", "split": "external_test", "story": item["narrative"],
               "question": question["question"], "answer_text": question["answer"],
               "question_type": question["type"], "answer_kind": answer_kind, "answer_options": answer_options,
               "plot_info": item.get("plot_info", {}), "scoring_protocol": SCORING_PROTOCOL}
        # Check every gold label without using it to construct the options.
        audit = score_opentom(row["answer_text"], row)
        if not audit["correct"]:
            raise ValueError(f"Noncanonical gold answer at record {index}; import aborted")
        row["gold_scoring_audit"] = {key: value for key, value in audit.items() if key.startswith("upstream_")}
        output_rows.append(row)
    write_jsonl(output, output_rows)
    write_json(str(output) + ".provenance.json", {"source_url": source_url,
               "source_sha256": file_digest(source), "records": len(output_rows),
               "stories": len({row["world_id"] for row in output_rows}), "use": "evaluation only",
               "scoring": SCORING_PROTOCOL, "upstream_revision": OPENTOM_REVISION,
               "gold_self_score_correct": len(output_rows),
               "upstream_gold_self_score_failures": sum(not row["gold_scoring_audit"]["upstream_correct"] for row in output_rows),
               "location_metadata_corrections": LOCATION_METADATA_CORRECTIONS,
               "evaluation_protocols": [OFFICIAL_PROTOCOL, ADAPTED_PROTOCOL],
               "prompt_adaptation": "Default official system/user text in Qwen chat serialization; secondary adapted Paper 2 wrapper is not exact published benchmark prompts"})
    return output_rows


def official_prompt(row, reasoning=False):
    """Exact upstream normal-narrative template/build_prompt text (no SimToM).

    Attributed to seacowx/OpenToM, pinned OPENTOM_REVISION, CC BY-NC 4.0.
    Qwen uses its own chat serialization around these system/user messages.
    Keep upstream CoT string operations, including their punctuation behavior.
    """
    question, kind = row["question"], row["answer_kind"]
    stem = ("Read and comprehend the following short story. Then, answer the question that follows."
            f"\n\n{row['story']}\n\nQuestion: ")
    if kind == "attitude":
        question = question.split("?")[0].strip() + ", assuming that you observed the action?"
        prompt = stem + question + ' Answer with "positive", "neutral", or "negative".'
        prompt += " " + (OFFICIAL_COT if reasoning else "Answer without any explanation.")
    else:
        if kind == "fullness":
            question += ' Answer with "more full", "equally full", or "less full".'
        elif kind == "accessibility":
            question += ' Answer with "more accessible", "equally accessible", or "less accessible".'
        suffix = (' Answer the question with "Yes" or "No". Do not give any explanation.'
                  if kind == "coarse_location" else " Answer the question without any explanation.")
        prompt = stem + question + suffix + "\n"
        if reasoning:
            if kind in ("coarse_location", "fullness", "accessibility"):
                prompt = ".".join(prompt.split(".")[:-2]) + ". " + OFFICIAL_COT
            else:
                prompt = prompt.split("?")[0] + "? " + OFFICIAL_COT
    return prompt.strip()


def official_parse_answer(text, reasoning=False):
    """Upstream OpenToMEvaluator.parse_cot_answer applied to a stripped output."""
    text = text.strip()
    if reasoning:
        return text.split("\n")[-1] if "\n" in text else text.split("Therefore")[-1]
    return text


def evaluate_opentom(backend, rows, reasoning=False, protocol="official"):
    if protocol == "adapted":
        yield from _evaluate_adapted(backend, rows, reasoning)
        return
    if protocol != "official":
        raise ValueError("OpenToM protocol must be official or adapted")
    for row in rows:
        if row.get("split") != "external_test" or row.get("dataset") != "opentom":
            raise ValueError("External check only accepts explicitly marked OpenToM test records")
        if row.get("scoring_protocol") != SCORING_PROTOCOL:
            raise ValueError("Re-import OpenToM with question options and type-specific scoring")
        kind, options = question_protocol({"question": row["question"], "type": row["question_type"]}, row["plot_info"])
        if (kind, options) != (row["answer_kind"], row["answer_options"]):
            raise ValueError("OpenToM answer options differ from the question/plot-derived protocol")
        user = official_prompt(row, reasoning)
        messages = [{"role": "system", "content": OFFICIAL_SYSTEM}, {"role": "user", "content": user}]
        prompt = backend.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # Official CoT is one uninterrupted completion. No Paper 2 answer suffix,
        # injected trace, location options, forced delimiter, or second answer call.
        generated = backend.generate(prompt, max_new_tokens=384 if reasoning else 32,
                                     stop=None if reasoning else "\n")
        raw = generated["text"]
        prediction = official_parse_answer(raw, reasoning)
        adapted = score_opentom(prediction, row)
        official = upstream_score(prediction, row)
        yield {**row, "evaluation_protocol": OFFICIAL_PROTOCOL, "reader": backend.checkpoint_id,
               "format": "reasoning" if reasoning else "immediate", "prediction_text": prediction,
               "raw_generation": raw, "prompt_messages": messages, "prompt_sha256": digest(prompt),
               "primary_metric": "macro_f1", "scorer": "pinned_official",
               **official, "prediction_label": official["upstream_prediction_label"],
               "target_label": official["upstream_target_label"], "correct": official["upstream_correct"],
               "report_included": official["upstream_report_included"],
               "adapted_score_diagnostic": adapted, "trace": raw if reasoning else None,
               "forced_delimiter": False, "truncated": generated["truncated"],
               "max_new_tokens": 384 if reasoning else 32}


def _evaluate_adapted(backend, rows, reasoning=False):
    for row in rows:
        if row.get("split") != "external_test" or row.get("dataset") != "opentom":
            raise ValueError("External check only accepts explicitly marked OpenToM test records")
        if row.get("scoring_protocol") != SCORING_PROTOCOL:
            raise ValueError("Re-import OpenToM with question options and type-specific scoring")
        # Recompute options from public inputs to reject accidental gold-conditioned imports.
        kind, options = question_protocol({"question": row["question"], "type": row["question_type"]}, row["plot_info"])
        if (kind, options) != (row["answer_kind"], row["answer_options"]):
            raise ValueError("OpenToM answer options differ from the question/plot-derived protocol")
        question = row["question"]
        if row["answer_kind"] == "attitude":
            question = question.split("?", 1)[0] + ", assuming that you observed the action?"
        choices = ", ".join(repr(option) for option in options)
        user = f"Story:\n{row['story']}\n\nQuestion: {question}\nAnswer with one of: {choices}.\n\n"
        user += ("Reason about the question, then finish with a new line Answer: and a concise answer."
                 if reasoning else "Answer immediately and concisely, without an explanation.")
        prompt = backend.tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True,
        )
        trace, forced, trace_truncated = None, False, False
        if reasoning:
            prompt += "Reasoning:\n"
            gen = backend.generate(prompt, max_new_tokens=384, stop=ANSWER_DELIMITER)
            trace = gen["text"].split(ANSWER_DELIMITER, 1)[0]
            forced, trace_truncated = ANSWER_DELIMITER not in gen["text"], gen["truncated"]
            prompt += trace + ANSWER_DELIMITER
        else:
            prompt += "Answer:"
        generated = backend.generate(prompt, max_new_tokens=32, stop="\n")
        prediction = generated["text"].split("\n", 1)[0]
        yield {**row, "evaluation_protocol": ADAPTED_PROTOCOL, "reader": backend.checkpoint_id, "format": "reasoning" if reasoning else "immediate",
               "prediction_text": prediction, **score_opentom(prediction, row),
               "trace": trace, "forced_delimiter": forced,
               "truncated": generated["truncated"] or trace_truncated}


def _macro_f1(rows, labels, *, upstream=False):
    prefix = "upstream_" if upstream else ""
    values = []
    for label in labels:
        tp = sum(row[prefix + "target_label"] == label and row[prefix + "prediction_label"] == label for row in rows)
        fp = sum(row[prefix + "target_label"] != label and row[prefix + "prediction_label"] == label for row in rows)
        fn = sum(row[prefix + "target_label"] == label and row[prefix + "prediction_label"] != label for row in rows)
        values.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.)
    return sum(values) / len(values) if rows else None


def _summarize_adapted(rows):
    from collections import defaultdict
    rows = list(rows)
    types, subtypes = defaultdict(list), defaultdict(list)
    for row in rows:
        types[row["question_type"]].append(row)
        subtypes[(row["question_type"], row["answer_kind"])].append(row)
    accuracy = lambda values: sum(row["correct"] for row in values) / len(values) if values else None
    summaries = {}
    for (kind, answer_kind), values in sorted(subtypes.items()):
        upstream = [row for row in values if row["upstream_report_included"]]
        # Official sklearn macro-F1 averages labels present in gold OR predictions,
        # including its fine-location tie class 3 when present.
        upstream_labels = sorted({row[key] for row in upstream for key in ("upstream_target_label", "upstream_prediction_label")})
        summaries[f"{kind}|{answer_kind}"] = {
            "n": len(values), "accuracy": accuracy(values),
            "macro_f1": _macro_f1(values, LABELS[answer_kind]),
            "invalid_predictions": sum(row["prediction_label"] is None for row in values),
            "upstream_diagnostic": {"included": len(upstream), "excluded": len(values) - len(upstream),
                 "accuracy_excluding_invalid": sum(row["upstream_correct"] for row in upstream) / len(upstream) if upstream else None,
                 "macro_f1_excluding_invalid": _macro_f1(upstream, upstream_labels, upstream=True)}}
    return {"n": len(rows), "accuracy": accuracy(rows),
            "by_type": {key: {"n": len(values), "accuracy": accuracy(values)} for key, values in sorted(types.items())},
            "by_subtype": summaries, "scoring_protocol": SCORING_PROTOCOL,
            "interpretation": "Adapted OpenToM check: options and type-specific scoring; invalid primary answers count as wrong. Shared chat wrapper, not exact published benchmark reproduction. Report separately from controlled-world accuracy."}


def _official_metrics(values, *, samples=20000, seed=2026):
    import numpy as np
    included = [row for row in values if row["upstream_report_included"]]
    labels = sorted({row[key] for row in included
                     for key in ("upstream_target_label", "upstream_prediction_label")})
    result = {"n": len(values), "included": len(included), "excluded": len(values) - len(included),
              "macro_f1": _macro_f1(included, labels, upstream=True),
              "accuracy": sum(row["upstream_correct"] for row in included) / len(included) if included else None,
              "accuracy_full_denominator": sum(row["upstream_correct"] for row in values) / len(values) if values else None,
              "invalid_prediction_rate": sum(row["upstream_prediction_label"] in (-1, None) for row in values) / len(values) if values else None}
    if samples < 1:
        raise ValueError("OpenToM bootstrap requires positive whole-story resamples")
    stories = sorted({row.get("world_id", digest(row["story"])) for row in values})
    indexes = {story: i for i, story in enumerate(stories)}
    # Story sufficient statistics avoid resampling every raw question 20,000
    # times. Never treat questions within a narrative as independent units.
    counts = np.zeros((len(stories), 3 * len(labels) + 4))
    for row in values:
        counts[indexes[row.get("world_id", digest(row["story"]))], -1] += 1
        stat = counts[indexes[row.get("world_id", digest(row["story"]))]]
        stat[-2] += row["upstream_prediction_label"] in (-1, None)
        stat[-3] += row["upstream_correct"]
        stat[-4] += row["upstream_report_included"]
        if row["upstream_report_included"]:
            for i, label in enumerate(labels):
                actual, predicted = row["upstream_target_label"] == label, row["upstream_prediction_label"] == label
                stat[3 * i:3 * i + 3] += (actual and predicted, not actual and predicted, actual and not predicted)
    distributions = {key: [] for key in ("macro_f1", "accuracy", "accuracy_full_denominator", "invalid_prediction_rate")}
    rng = np.random.default_rng(seed)
    for offset in range(0, samples, 256):
        sampled = counts[rng.integers(0, len(stories), size=(min(256, samples - offset), len(stories)))].sum(axis=1)
        for stat in sampled:
            denoms = 2 * stat[:3 * len(labels):3] + stat[1:3 * len(labels):3] + stat[2:3 * len(labels):3]
            present = denoms > 0
            if present.any():
                distributions["macro_f1"].append(float((2 * stat[:3 * len(labels):3][present] / denoms[present]).mean()))
            if stat[-4]:
                distributions["accuracy"].append(float(stat[-3] / stat[-4]))
            distributions["accuracy_full_denominator"].append(float(stat[-3] / stat[-1]))
            distributions["invalid_prediction_rate"].append(float(stat[-2] / stat[-1]))
    result.update({key + "_ci95": np.quantile(data, [.025, .975]).tolist() if data else [None, None]
                   for key, data in distributions.items()})
    result["bootstrap"] = {"unit": "whole_story", "stories": len(stories), "samples": samples, "seed": seed,
                           "undefined_macro_f1_resamples": samples - len(distributions["macro_f1"])}
    return result


def summarize_opentom(rows, *, samples=20000, seed=2026):
    """Never pool models, response formats, or incompatible label spaces."""
    from collections import defaultdict
    rows = list(rows)
    groups = defaultdict(list)
    for row in rows:
        groups[(row.get("evaluation_protocol", ADAPTED_PROTOCOL), row.get("reader", "unspecified"),
                row.get("format", "unspecified"))].append(row)
    if len(groups) != 1:
        return {"n": len(rows), "by_run": {"|".join(key): summarize_opentom(values, samples=samples, seed=seed)
                                            for key, values in sorted(groups.items())}}
    (protocol, reader, response_format), rows = next(iter(groups.items()))
    if protocol == ADAPTED_PROTOCOL:
        return {**_summarize_adapted(rows), "evaluation_protocol": protocol, "reader": reader,
                "format": response_format, "role": "secondary_adapted_check"}
    if protocol != OFFICIAL_PROTOCOL:
        raise ValueError("Unrecognized OpenToM evaluation protocol")
    subtypes = defaultdict(list)
    for row in rows:
        subtypes[(row["question_type"], row["answer_kind"])].append(row)
    summaries = {"|".join(key): _official_metrics(values, samples=samples, seed=seed) for key, values in sorted(subtypes.items())}
    # Official reporting combines fullness/accessibility labels within each
    # multihop order as well as reporting them separately. Locations of different
    # granularities remain separate; their integer labels are not commensurable.
    multihop = {kind: _official_metrics([row for row in rows if row["question_type"] == kind], samples=samples, seed=seed)
                for kind in sorted({row["question_type"] for row in rows if row["answer_kind"] in ("fullness", "accessibility")})}
    return {"n": len(rows), "reader": reader, "format": response_format,
            "evaluation_protocol": OFFICIAL_PROTOCOL, "upstream_revision": OPENTOM_REVISION,
            "primary_metric": "macro_f1", "by_subtype": summaries, "multihop_combined": multihop,
            "interpretation": "Pinned official OpenToM prompt text and lexical scorer; macro-F1 per subtype uses official invalid-output exclusions, shown explicitly. Qwen chat serialization and full normal-length corpus; not a reproduction of Zhong et al.'s unavailable split or training/evaluation prompts."}
