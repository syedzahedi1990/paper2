import json
import subprocess
import sys
from paper2.cli import main
from paper2.io import read_jsonl
from paper2.opentom import import_opentom, normalize_answer, summarize_opentom, score_opentom


def test_cli_generates_verifies_and_never_auto_passes_gate(tmp_path):
    root = tmp_path / "corpus"
    assert main(["generate", "--output", str(root), "--train", "20", "--development", "6", "--confirmation", "6"]) == 0
    assert main(["verify-data", "--data", str(root)]) == 0
    review = tmp_path / "review.json"
    main(["init-gates", "--output", str(review)])
    value = json.loads(review.read_text())
    assert all(g["status"] == "pending" for g in value["gates"].values())
    assert value["paid_compute"]["approved"] is False
    rows = read_jsonl(root / "evaluation/development__clean.jsonl")
    assert len(rows) == 48
    assert {r["view"] for r in rows} == {"direct", "world", "other_agent", "search", "observed_overwrite", "unobserved_overwrite", "irrelevant_object", "report"}


def test_installed_entrypoint_exposes_three_tests():
    result = subprocess.run([sys.executable, "-m", "paper2", "--help"], capture_output=True, text=True, check=True)
    assert all(command in result.stdout for command in ("evaluate", "crossed", "components", "profile"))


def test_official_opentom_import_is_external_and_keeps_raw_targets(tmp_path):
    source = tmp_path / "opentom.json"
    source.write_text(json.dumps([{"narrative": "A moved the ball.", "plot_info": {"original_place": "box", "move_to_place": "the basket"}, "question": {
        "question": "Where?", "answer": "the basket", "type": "location-fo"}}]))
    path = tmp_path / "external.jsonl"
    rows = import_opentom(source, path, "https://github.com/seacowx/OpenToM")
    assert rows[0]["split"] == "external_test"
    assert rows[0]["answer_text"] == "the basket"
    assert normalize_answer(" Answer: THE   BASKET. ") == "the basket"
    assert normalize_answer("basket") != normalize_answer("the basket")
    assert summarize_opentom([{**rows[0], **score_opentom(rows[0]["answer_text"], rows[0])}])["accuracy"] == 1.0
