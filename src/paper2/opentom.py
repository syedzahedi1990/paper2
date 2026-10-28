"""External OpenToM check with options, canonical scoring, and upstream diagnostics."""
import json
import re
from pathlib import Path

from .io import digest, file_digest, write_jsonl, write_json
from .prompts import SYSTEM, ANSWER_DELIMITER

OPENTOM_REVISION = "3f22b66276b2d7ca5fe573c28c79cc0d077aafc5"
SCORING_PROTOCOL = "opentom_options_v2"
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
               "prompt_adaptation": "Shared chat wrapper, question-type answer options; not exact published benchmark prompts"})
    return output_rows


def evaluate_opentom(backend, rows, reasoning=False):
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
        yield {**row, "reader": backend.checkpoint_id, "format": "reasoning" if reasoning else "immediate",
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


def summarize_opentom(rows):
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
