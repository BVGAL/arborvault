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
