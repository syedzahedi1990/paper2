"""OpenToM external-only evaluation of the official flat QA JSON format."""
import json
import re
from pathlib import Path
from .io import digest, file_digest, write_jsonl, write_json
from .prompts import SYSTEM


def normalize_answer(text):
    text = text.strip().casefold()
    text = re.sub(r"^answer\s*:\s*", "", text)
    text = text.strip(" .!\n\t\"'")
    return re.sub(r"\s+", " ", text)


def import_opentom(source, output, source_url=None):
    records = json.loads(Path(source).read_text())
    if not isinstance(records, list) or not records:
        raise ValueError("Expected official OpenToM data/opentom.json list of QA records")
    output_rows = []
    for index, item in enumerate(records):
        question = item["question"]
        if not isinstance(question, dict) or not all(k in question for k in ("question", "answer", "type")):
            raise ValueError(f"Unsupported OpenToM schema at record {index}")
        if not isinstance(question["answer"], str) or not item["narrative"]:
            raise ValueError("OpenToM needs textual narrative and answer")
        output_rows.append({"id": f"opentom-{index:06d}", "world_id": digest(item["narrative"]),
                            "dataset": "opentom", "split": "external_test", "story": item["narrative"],
                            "question": question["question"], "answer_text": question["answer"],
                            "question_type": question["type"]})
    write_jsonl(output, output_rows)
    write_json(str(output) + ".provenance.json", {"source_url": source_url,
               "source_sha256": file_digest(source), "records": len(output_rows),
               "stories": len({r["world_id"] for r in output_rows}), "use": "evaluation only",
               "scoring": "Case/whitespace/punctuation-normalized exact match; raw output retained"})
    return output_rows


def evaluate_opentom(backend, rows, reasoning=False):
    for row in rows:
        if row.get("split") != "external_test" or row.get("dataset") != "opentom":
            raise ValueError("External check only accepts explicitly marked OpenToM test records")
        user = f"Story:\n{row['story']}\n\nQuestion: {row['question']}\n\n"
        user += ("Reason about the question, then finish with a new line Answer: and a concise answer."
                 if reasoning else "Answer immediately and concisely, without an explanation.")
        prompt = backend.tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True,
        )
        trace = None
        if reasoning:
            prompt += "Reasoning:\n"
            gen = backend.generate(prompt, max_new_tokens=384, stop="\nAnswer:")
            trace = gen["text"].split("\nAnswer:", 1)[0]
            prompt += trace + "\nAnswer:"
        else:
            prompt += "Answer:"
        generated = backend.generate(prompt, max_new_tokens=32, stop="\n")
        prediction = generated["text"].split("\n", 1)[0]
        yield {**row, "reader": backend.checkpoint_id, "format": "reasoning" if reasoning else "immediate",
               "prediction_text": prediction, "correct": normalize_answer(prediction) == normalize_answer(row["answer_text"]),
               "trace": trace, "truncated": generated["truncated"]}


def summarize_opentom(rows):
    from collections import defaultdict
    values = defaultdict(list)
    for row in rows:
        values[row["question_type"]].append(bool(row["correct"]))
    return {"n": len(rows), "accuracy": sum(r["correct"] for r in rows) / len(rows) if rows else None,
            "by_type": {key: {"n": len(v), "accuracy": sum(v) / len(v)} for key, v in sorted(values.items())},
            "interpretation": "External free-response exact match; report separately from controlled-world letter accuracy."}
