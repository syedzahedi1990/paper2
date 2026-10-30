"""Post-outcome completion amendment, isolated from legacy scoring semantics."""
import json

import pytest

from paper2.training.config import (LEGACY_COMPLETION_PROTOCOL, NATURAL_COMPLETION_PROTOCOL,
                                   TrainingConfig, load_config)
from paper2.training.data import (answer_reward, make_sft_rows, natural_completion_outcome,
                                 parse_answer)
from paper2.training.generation import completion_token_budget


V2 = NATURAL_COMPLETION_PROTOCOL


class EncodeForbidden:
    eos_token_id = 0

    def encode(self, *args, **kwargs):
        raise AssertionError('A sampled completion must never be retokenized to enforce its cap')


def reward(text, ids, answer='A', reasoning=True):
    return answer_reward(completions=[text], answer=[answer], reasoning=reasoning,
                         completion_protocol=V2, completion_ids=[ids], tokenizer=EncodeForbidden())[0]


def test_legacy_config_default_and_explicit_revised_protocol(tmp_path):
    old = tmp_path / 'legacy.json'
    old.write_text(json.dumps({'arm': 'reasoning_rl'}))
    config = load_config(old)
    assert config.completion_protocol == LEGACY_COMPLETION_PROTOCOL
    assert config.beta == 0
    assert config.to_dict()['completion_protocol'] == LEGACY_COMPLETION_PROTOCOL
    new = tmp_path / 'revised.json'
    new.write_text(json.dumps({'arm': 'reasoning_rl', 'completion_protocol': V2, 'beta': .04}))
    revised = load_config(new)
    assert revised.completion_protocol == V2 and revised.beta == .04
    assert revised.to_dict()['completion_protocol'] == V2
    with pytest.raises(ValueError, match='completion protocol'):
        TrainingConfig(completion_protocol='natural_answer_guess').validate()
    for value in [-.01, float('nan'), float('inf')]:
        with pytest.raises(ValueError, match='beta'):
            TrainingConfig(beta=value).validate()


def test_revised_total_budget_has_no_delimiter_or_answer_extension():
    assert completion_token_budget(reasoning=True, delimiter_length=3) == 391
    assert completion_token_budget(reasoning=True, completion_protocol=V2, delimiter_length=3) == 384
    for protocol in [LEGACY_COMPLETION_PROTOCOL, V2]:
        assert completion_token_budget(reasoning=False, completion_protocol=protocol, delimiter_length=9) == 4
    with pytest.raises(ValueError, match='completion protocol'):
        completion_token_budget(reasoning=True, completion_protocol='unknown')


@pytest.mark.parametrize('text', ['Answer:A', 'Answer: A.', '\nAnswer: A',
                                  'A belief trace.\nAnswer: A.\n',
                                  'know ' * 20 + '\nAnswer: A'])
def test_natural_answer_has_no_trace_content_or_minimum_length_requirement(text):
    assert parse_answer(text, True, completion_protocol=V2) == 'A'
    assert reward(text, [5, 6, 7]) == 1


@pytest.mark.parametrize('text', ['Answer:', 'Answer: ', 'Answer: G', 'Answer: a',
                                  'Answer: A..', 'Answer: A .', 'Answer: A because',
                                  'A', 'The answer is A.', 'Thought Answer: A'])
def test_absent_incomplete_or_other_answer_formats_receive_zero(text):
    assert parse_answer(text, True, completion_protocol=V2) is None
    assert reward(text, [5] * 384) == 0


def test_optional_period_is_protocol_specific_and_immediate_stays_short():
    assert parse_answer(' A.\n', False, completion_protocol=V2) == 'A'
    assert parse_answer(' A.\n', False) is None
    assert parse_answer('trace\nAnswer: A.', True) is None
    assert parse_answer('Answer: A', True) is None
    assert reward(' A.', [4, 5, 0], reasoning=False) == 1
    assert reward(' A.', [4] * 5, reasoning=False) == 0
    assert reward(' B.', [4, 5], reasoning=False) == 0


def test_prior_answer_mentions_and_legacy_whitespace_are_not_new_content_restrictions():
    trace = 'Answer: B\nReconsidering the story.\nAnswer:\n A'
    assert parse_answer(trace, True) == 'A'
    assert parse_answer(trace + '.', True, completion_protocol=V2) == 'A'
    assert reward(trace + '.', [7] * 384) == 1


def test_actual_sampled_ids_decide_boundary_without_reencoding_or_required_eos():
    text = 'A short decoded trace.\nAnswer: A.'
    assert reward(text, [7] * 383) == 1
    assert reward(text, [7] * 384) == 1
    assert reward(text, [7] * 385) == 0
    assert reward(text, [7] * 383 + [0]) == 1
    assert reward(text, [7] * 384 + [0]) == 0
    assert reward('x' * 10000 + '\nAnswer: A', [7] * 384) == 1
    assert reward('Answer: B', [7] * 384) == 0
    assert reward('No answer at cap.', [7] * 384) == 0


def test_missing_or_misaligned_raw_ids_is_an_instrumentation_error():
    with pytest.raises(ValueError, match='actual unpadded completion_ids'):
        answer_reward(completions=['Answer: A'], answer=['A'], reasoning=True, completion_protocol=V2)
    with pytest.raises(ValueError, match='align'):
        answer_reward(completions=['Answer: A'], answer=['A'], reasoning=True,
                      completion_protocol=V2, completion_ids=[])
    with pytest.raises(ValueError, match='sequence of token IDs'):
        natural_completion_outcome('Answer: A', reasoning=True, completion_ids='decoded text')


def test_mixed_unpadded_reward_batch_counts_eos_not_batch_padding():
    # TRL masks each row through its first EOS inclusive before passing these
    # ragged IDs to a reward function; do not pad them back to the batch maximum.
    completions = ['Answer: A.', 'trace\nAnswer: A', 'Unfinished\nAnswer:']
    assert answer_reward(completions=completions, answer=['A'] * 3, reasoning=True,
                         completion_protocol=V2, completion_ids=[[8, 9, 0], [7] * 384, [7] * 384],
                         tokenizer=EncodeForbidden()) == [1., 1., 0.]
    assert answer_reward(completions=[[{'role': 'assistant', 'content': 'Answer: A.'}]],
                         answer=['A'], reasoning=True, completion_protocol=V2,
                         completion_ids=[[8, 9, 0]]) == [1.]


def test_outcome_separates_budget_exhaustion_from_invalid_answer():
    at_cap = natural_completion_outcome('Answer: A', reasoning=True, completion_ids=[7] * 384,
                                        eos_token_id=0)
    assert at_cap == {'parsed_answer': 'A', 'prediction': 'A', 'sampled_tokens': 384,
                      'token_budget': 384, 'answer_complete': True, 'within_token_budget': True,
                      'cap_reached': True, 'cap_failure': False, 'invalid_completion': False,
                      'eos_seen': False, 'termination_reason': 'valid_at_budget'}
    capped = natural_completion_outcome('Answer:', reasoning=True, completion_ids=[7] * 384)
    assert capped['termination_reason'] == 'cap_failure' and capped['prediction'] is None
    ended = natural_completion_outcome('Answer: A.', reasoning=True, completion_ids=[7, 0], eos_token_id=0)
    assert ended['termination_reason'] == 'ordinary_eos' and not ended['invalid_completion']
    early = natural_completion_outcome('No answer', reasoning=True, completion_ids=[7, 0], eos_token_id=0)
    assert early['termination_reason'] == 'invalid_completion' and not early['cap_failure']
    answer_only = natural_completion_outcome('unfinished', reasoning=False, completion_ids=[7] * 4)
    assert answer_only['cap_reached'] and not answer_only['cap_failure']
    beyond = natural_completion_outcome('Answer: A', reasoning=True, completion_ids=[7] * 385)
    assert beyond['answer_complete'] and beyond['prediction'] is None and not beyond['within_token_budget']


def test_revised_sft_checks_complete_target_budget_legacy_keeps_trace_only_cap():
    class CharacterTokenizer:
        eos_token_id = 0

        def encode(self, text, add_special_tokens=False):
            return [ord(char) for char in text]

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            return '\n'.join(message['content'] for message in messages)

    row = {'id': 'one', 'world_id': 'one', 'split': 'train', 'dataset': 'clean', 'view': 'direct',
           'story': 'A story.', 'question': 'Where?', 'choices': {'A': 'box'}, 'answer': 'A',
           'trace': 'x' * 380, 'trace_source': 'paper1_symbolic_oracle_v1'}
    legacy = TrainingConfig(arm='reasoning_sft', stage='confirmatory')
    revised = TrainingConfig(arm='reasoning_sft', stage='confirmatory', completion_protocol=V2)
    assert make_sft_rows([row], legacy, CharacterTokenizer())
    with pytest.raises(ValueError, match='cap'):
        make_sft_rows([row], revised, CharacterTokenizer())
    assert make_sft_rows([{**row, 'trace': 'x' * 370}], revised, CharacterTokenizer())
