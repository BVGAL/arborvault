import json
import math

import pytest

from arborvault.candidate_scoring import (
    BM25,
    calibrate,
    partition,
    retrieve,
    training_groups,
    weighted_binary_loss,
)
from arborvault.domain import Node
from arborvault.routing_v2 import Tree


def fixture_tree():
    return Tree(
        [
            Node("root", None, "doc", "", ["doc"], 0),
            Node("a", "root", "alpha", "", ["doc", "alpha"], 1),
            Node("b", "root", "beta", "", ["doc", "beta"], 2),
            Node("a1", "a", "one", "alpha evidence", ["doc", "alpha", "one"], 3),
            Node("b1", "b", "two", "beta alternative", ["doc", "beta", "two"], 4),
            Node("a2", "a", "three", "unannotated passage", ["doc", "alpha", "three"], 5),
        ]
    )


def test_union_labels_do_not_make_other_annotators_evidence_negative():
    q = {
        "id": "q",
        "document": "doc",
        "question": "evidence",
        "answerable": True,
        "gold_sets": [["a1"], ["b1"]],
    }
    group = training_groups({"doc": fixture_tree()}, [q])[0]
    positive = {i["node_id"] for i in group["items"] if i["label"]}
    assert positive == {"a", "b", "a1", "b1"}
    assert sum(i["weight"] for i in group["items"]) == pytest.approx(1)
    negative = [i for i in group["items"] if not i["label"]]
    assert negative[0]["label_kind"] == "unannotated_weak_negative"
    assert negative[0]["weight"] == pytest.approx(0.2)


def test_partition_cannot_split_questions_from_one_document():
    queries = [{"document": f"d{i // 3}", "id": str(i)} for i in range(300)]
    fit, cal = partition(queries)
    assert len(fit) + len(cal) == len(queries)
    assert not {q["document"] for q in fit} & {q["document"] for q in cal}
    assert partition(list(reversed(queries)))[1] == list(reversed(cal))


def test_tree_budget_counts_branches_and_does_not_use_hidden_leaf_scores():
    tree = fixture_tree()
    scores = {"a": 2, "b": 1, "a1": 1, "b1": 99, "a2": 0}
    selected, trace = retrieve(tree, "evidence", scores, mode="tree", budget=3, width=1)
    assert len(trace["scored_ids"]) == 3
    assert "b1" not in trace["scored_ids"]
    assert [n.id for n, _ in selected] == ["a1"]
    empty, _ = retrieve(tree, "evidence", scores, mode="tree", budget=2, width=1)
    assert empty == []
    flat, _ = retrieve(tree, "evidence", scores, mode="flat", budget=3)
    assert flat[0][0].id == "b1"
    rejected, _ = retrieve(tree, "evidence", scores, mode="flat", budget=3, threshold=100)
    assert rejected == []


def test_scoring_rejects_missing_and_nonfinite_scores():
    with pytest.raises(ValueError, match="Incomplete"):
        retrieve(fixture_tree(), "x", {}, mode="flat")
    with pytest.raises(ValueError, match="non-finite"):
        retrieve(
            fixture_tree(), "x", {k: math.nan for k in ("a", "b", "a1", "b1", "a2")}, mode="flat"
        )


def test_bm25_positive_idf_and_length_normalization():
    assert BM25(["cat", "dog"]).scores("cat") == pytest.approx([math.log(2), 0])
    scores = BM25(["cat", "cat " + "dog " * 10]).scores("cat")
    assert scores[0] > scores[1] > 0


def test_binary_objective_supports_multiple_positives_without_competition():
    torch = pytest.importorskip("torch")
    values = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
    loss = weighted_binary_loss(
        values, torch.tensor([1.0, 1.0, 0.0]), torch.tensor([0.4, 0.4, 0.2])
    )
    loss.backward()
    assert values.grad.tolist() == pytest.approx([-0.2, -0.2, 0.1])


def test_calibration_keeps_answerable_coverage_while_separating_unknown():
    queries = [
        {
            "id": "p",
            "document": "doc",
            "question": "evidence",
            "answerable": True,
            "gold_sets": [["a1"]],
        },
        {"id": "u", "document": "doc", "question": "absent", "answerable": False, "gold_sets": []},
    ]
    values = {"a": 1, "b": 1, "a1": 2, "a2": 0, "b1": 1}
    cache = {"p": {"scores": values}, "u": {"scores": dict.fromkeys(values, -1)}}
    result = calibrate({"doc": fixture_tree()}, queries, cache, mode="flat")
    assert result["chosen"]["unknown_abstentions"] == 1
    assert result["chosen"]["recall"] == 1
    assert result["chosen"]["answerable_empty"] == 0


def test_shared_fitting_preserves_question_and_path_and_refuses_oversized_header():
    from arborvault.candidate_models import fit_text

    class QwenTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return [0] * len(messages[-1]["content"])

    class CrossTokenizer:
        def __call__(self, question, candidate, **kwargs):
            return {"input_ids": [0] * (len(question) + len(candidate))}

    text = json.dumps(
        {"document": "doc", "path": ["method"], "kind": "passage", "excerpt": "e" * 500}
    )
    result, meta = fit_text("specific question", text, QwenTokenizer(), CrossTokenizer(), 250)
    assert json.loads(result)["path"] == ["method"]
    assert meta["excerpt_characters_removed"] > 0
    assert max(meta["qwen_tokens"], meta["cross_encoder_tokens"]) <= 250
    with pytest.raises(ValueError, match="Question/path"):
        fit_text("q" * 600, text, QwenTokenizer(), CrossTokenizer(), 250)
