from dataclasses import replace
from collections import defaultdict

import pytest

from paper2.data import (CorpusConfig, VIEWS, generate_corpus, read_jsonl, verify_corpus,
                         generate_world_splits, validate_splits, validate_training_rows,
                         audit_tokenizer_lengths)


def test_full_default_sizes_and_story_split_disjointness():
    splits = generate_world_splits(CorpusConfig())
    audit = validate_splits(splits)
    assert audit["story_counts"] == {"train": 2000, "development": 120, "confirmation": 120}
    assert {len(world.focal_word) for worlds in splits.values() for world in worlds} == {1, 2}
    assert {world.critical_event.kind for worlds in splits.values() for world in worlds} == {"move", "tell"}


def test_frozen_generation_determinism_exact_rates_and_no_consequence_labels(tmp_path):
    config = CorpusConfig(train=20, development=2, confirmation=2, seed=11)
    left, right = tmp_path / "left", tmp_path / "right"
    a = generate_corpus(left, config)
    b = generate_corpus(right, config)
    assert a == b
    assert a["cue_design"]["shortcut_training_agreement"] == 0.90
    assert verify_corpus(left)["verified"]
    clean = read_jsonl(left / "training/clean.jsonl")
    shortcut = read_jsonl(left / "training/shortcut.jsonl")
    assert {row["world_id"] for row in clean} == {row["world_id"] for row in shortcut}
    assert len(clean) == 20 and {row["view"] for row in clean} == {"direct"}
    assert not any("world" in row or "trace" in row for row in clean)
    reasoning = read_jsonl(left / "supervision/clean_reasoning_sft.jsonl")
    assert all(row["trace_source"] == "paper1_symbolic_oracle_v1" for row in reasoning)
    for condition, replicas in (("clean", 1), ("agree", 1), ("independent", 6), ("conflict", 5)):
        rows = read_jsonl(left / f"evaluation/development__{condition}.jsonl")
        assert len(rows) == 2 * len(VIEWS) * replicas
        assert len({row["world_id"] for row in rows}) == 2
        groups = defaultdict(list)
        for row in rows:
            groups[(row["world_id"], row["cue_condition"], row["cue_location"])].append(row)
        assert len(groups) == 2 * replicas
        assert all({row["view"] for row in group} == set(VIEWS) for group in groups.values())
        assert all(len({row["base_story"] for row in group}) == 1 for group in groups.values())
        if condition == "independent":
            assert sum(row["cue_agrees"] for row in rows) * 6 == len(rows)
    with pytest.raises(FileExistsError):
        generate_corpus(left, config)
    with (left / "training/clean.jsonl").open("a") as handle:
        handle.write("{}\n")
    with pytest.raises(ValueError, match="checksum"):
        verify_corpus(left)


def test_story_family_leakage_fails_closed():
    splits = generate_world_splits(CorpusConfig(train=10, development=1, confirmation=1))
    splits["confirmation"] = [replace(splits["train"][0], split="confirmation")]
    with pytest.raises(ValueError, match="leakage"):
        validate_splits(splits)


def test_training_validation_rejects_hidden_targets_and_heldout_rows(tmp_path):
    generate_corpus(tmp_path, CorpusConfig(train=10, development=1, confirmation=1))
    row = read_jsonl(tmp_path / "training/clean.jsonl")[0]
    for corruption in ({"view": "report"}, {"split": "confirmation"}, {"world": {}}, {"trace": "answer A"}):
        with pytest.raises(ValueError):
            validate_training_rows([{**row, **corruption}])


def test_tokenizer_specific_matching_guard(tmp_path):
    generate_corpus(tmp_path, CorpusConfig(train=10, development=1, confirmation=1))
    rows = read_jsonl(tmp_path / "evaluation/development__independent.jsonl")
    class WordTokenizer:
        def encode(self, text, add_special_tokens=False):
            return text.split()
    assert audit_tokenizer_lengths(rows, WordTokenizer())["passed"]
    rows[0]["story"] += " extra"
    assert not audit_tokenizer_lengths(rows, WordTokenizer())["passed"]
