"""End-to-end Test 3 on real generated worlds with a tiny offline tokenizer/model."""

import json
import random

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

from paper2.backend import HFBackend
from paper2.data import generate_world, matched_pair
from paper2.interventions import Site, run_component_test, score_consequence_pattern


@pytest.mark.parametrize("event_kind", ["move", "tell"])
def test_real_pairs_run_with_forecasts_and_serializable_receipts(event_kind):
    pair = matched_pair(generate_world(random.Random(13), event_kind=event_kind, distractor_events=0))
    vocabulary = {"[UNK]": 0, "[PAD]": 1}
    split = pre_tokenizers.Whitespace()
    # Build an offline vocabulary from the complete local fixture's characters.
    for row in pair["recipient"] + pair["donor"]:
        for token, _ in split.pre_tokenize_str(row["story"] + " " + row["question"] + " A B C D E F Answer:"):
            vocabulary.setdefault(token, len(vocabulary))
    core = Tokenizer(models.WordLevel(vocabulary, unk_token="[UNK]"))
    core.pre_tokenizer = split
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=core, unk_token="[UNK]", pad_token="[PAD]")
    tokenizer.chat_template = "{% for message in messages %}{{ message['role'] + ': ' + message['content'] + '\n' }}{% endfor %}assistant: "
    torch.manual_seed(0)
    config = Qwen2Config(vocab_size=len(vocabulary), hidden_size=16, intermediate_size=24,
                         num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                         max_position_embeddings=2048)
    config._attn_implementation = "eager"
    backend = HFBackend.__new__(HFBackend)
    backend.model = Qwen2ForCausalLM(config).eval()
    backend.torch, backend.tokenizer, backend.checkpoint_id = torch, tokenizer, "random-test-only"
    sites = [Site(0, component) for component in ("k", "v", "residual")]
    rows = run_component_test(backend, [pair], sites, seed=8)
    assert len(rows) == 8 * (2 + 3 * 3)
    json.dumps(rows, allow_nan=False)
    assert len([row for row in rows if row["positive_control"]]) == 8
    for row in rows:
        assert row["forecast_target"] == pair["forecasts"][row["view"]]["event_rewrite"]
        if row["control"] == "own":
            assert row["prediction"] == row["baseline_prediction"]
        if row["receipt"]:
            assert row["receipt"]["analysis"] == "primary_within_checkpoint"
    baseline = {row["view"]: row["ordinary_target"] for row in rows if row["control"] == "baseline"}
    predictions = {row["view"]: row["prediction"] for row in rows if row["component"] == "k" and row["control"] == "natural"}
    score = score_consequence_pattern(predictions, baseline, pair["forecasts"])
    assert isinstance(score["joint_correct"], bool)
    if event_kind == "tell":
        assert "world" in score["intact_views"]
    else:
        assert "world" in score["changed_views"]
    with pytest.raises(ValueError, match="positive-control"):
        run_component_test(backend, [pair], [Site(0, "k")])
