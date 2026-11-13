import pytest

from arborvault.routing_beams import before_eos, combine_hypotheses


def test_beam_union_respects_score_order_deduplication_and_width():
    hypotheses = [
        {"score": -2.0, "text": '["1","2","3"]'},
        {"score": -1.0, "text": '["0","1"]'},
        {"score": float("-inf"), "text": "unreachable"},
        {"score": -3.0, "text": '["4"]'},
    ]
    assert combine_hypotheses(hypotheses, {str(i): f"n{i}" for i in range(5)}, 4) == [
        "n0",
        "n1",
        "n2",
        "n3",
    ]


def test_best_empty_beam_preserves_abstention_and_validates_other_beams():
    hypotheses = [{"score": -1.0, "text": "[]"}, {"score": -2.0, "text": '["0"]'}]
    assert combine_hypotheses(hypotheses, {"0": "n0"}, 4) == []
    hypotheses[1]["text"] = '["1"]'
    with pytest.raises(ValueError):
        combine_hypotheses(hypotheses, {"0": "n0"}, 4)
    with pytest.raises(ValueError):
        combine_hypotheses([], {}, 4)


def test_beam_eos_padding_is_not_part_of_json():
    assert before_eos([1, 2, 99, 99, 99], {98, 99}) == [1, 2]
    with pytest.raises(ValueError):
        before_eos([1, 2], {99})


def test_real_generation_handles_small_frontiers_and_accounts_for_hypotheses():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from types import SimpleNamespace

    from arborvault.routing_beams import BeamQwenSelector

    class Tokenizer:
        eos_token_id = 127

        def encode(self, text, add_special_tokens=False):
            return list(text.encode())

        def decode(self, ids, skip_special_tokens=False):
            return bytes(ids).decode()

        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["enable_thinking"] is False
            return [1]

    torch.manual_seed(42)
    model = transformers.Qwen3ForCausalLM(
        transformers.Qwen3Config(
            vocab_size=128,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
            eos_token_id=127,
            pad_token_id=127,
        )
    )
    model.eval()
    selector = BeamQwenSelector(model, Tokenizer())
    for count in (1, 2, 5, 12):
        candidates = [
            SimpleNamespace(id=f"n{i}", path=["Document", str(i)], content="Passage", locator={})
            for i in range(count)
        ]
        selected = selector.select("question", candidates, 4)
        assert len(set(selected)) == len(selected) <= 4
        assert set(selected) <= {c.id for c in candidates}
        event = selector.events[-1]
        assert event["ok"]
        assert event["output_tokens"] == sum(h["output_tokens"] for h in event["hypotheses"])
