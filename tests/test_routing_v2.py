import copy
import json
import random
from pathlib import Path

import pytest

from arborvault.domain import Node
from arborvault.qasper import load_split, paper_nodes, question_record, training_routes
from arborvault.routing_eval import context, score
from arborvault.routing_v2 import SelectionGrammar, Tree, messages, navigate


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(text.encode())

    def decode(self, ids, skip_special_tokens=False):
        return bytes(ids).decode()


def test_constrained_arrays_allow_multiple_ids_and_have_only_valid_terminal_outputs():
    tokenizer = CharacterTokenizer()
    grammar = SelectionGrammar(tokenizer, 3, 2, [999])
    pending, completed = [[]], []
    while pending:
        prefix = pending.pop()
        for token in grammar.next_tokens(prefix):
            if token == 999:
                value = json.loads(tokenizer.decode(prefix))
                assert len(value) <= 2 and len(set(value)) == len(value)
                assert set(value) <= {"0", "1", "2"}
                completed.append(value)
            else:
                assert len(prefix) < 30
                pending.append(prefix + [token])
    assert len(completed) == 10  # empty, three single IDs, six distinct pairs


def test_navigation_exposes_late_headings_and_records_failed_routes_without_losing_evidence():
    nodes = [
        Node("root", None, "document", "", ["document"], 0),
        Node("branch", "root", "general", "", ["document", "general"], 1),
        Node(
            "early", "branch", "Introduction", "intro", ["document", "general", "Introduction"], 2
        ),
        Node(
            "late",
            "branch",
            "Closing clients",
            "close the client",
            ["document", "general", "Closing clients"],
            3,
        ),
    ]
    tree = Tree(nodes)
    seen = []

    class Selector:
        def select(self, query, candidates, width):
            seen.append(candidates)
            return ["branch"] if len(seen) == 1 else ["late"]

    found, trace, reason = navigate(tree, "close client", Selector())
    assert "Closing clients" in seen[0][0].content
    assert [n.id for n in found] == ["late"] and reason is None
    assert trace[0]["candidates"] == ["branch"]  # known document root is not a model gate
    assert trace[1]["selected"] == ["late"]
    conversation, labels = messages("close", tree.views(nodes[1:]), 2)
    assert labels == {"0": "branch", "1": "early", "2": "late"}
    assert json.loads(conversation[1]["content"])["candidates"][0]["kind"] == "branch"


def test_qasper_filters_conflicts_and_preserves_real_hierarchy_and_source_spans():
    paper = {
        "title": "Paper",
        "full_text": [
            {"section_name": "Method ::: Inputs", "paragraphs": ["A real input paragraph."]}
        ],
    }
    nodes, paragraphs = paper_nodes("paper", paper)
    assert nodes[-1].path == ["Paper", "Method", "Inputs", "A real input paragraph."]
    qa = {
        "question_id": "q",
        "question": "What input?",
        "answers": [{"answer": {"unanswerable": False, "evidence": ["A real input paragraph."]}}],
    }
    query = question_record("paper", qa, paragraphs)
    assert query["gold_sets"] == [[nodes[-1].id]]
    assert score(context([nodes[-1]]), query)["recall"] == 1
    bad = copy.deepcopy(qa)
    bad["answers"].append({"answer": {"unanswerable": True}})
    with pytest.raises(ValueError, match="disagreeing"):
        question_record("paper", bad, paragraphs)
    bad = copy.deepcopy(qa)
    bad["answers"][0]["answer"]["evidence"] = ["FLOAT SELECTED table"]
    with pytest.raises(ValueError, match="Non-textual"):
        question_record("paper", bad, paragraphs)


def test_dataset_splits_are_document_disjoint_and_training_routes_use_train_only():
    directory = Path(__file__).resolve().parents[1] / "datasets/qasper-routing-v2"
    documents, questions, canonical = set(), set(), set()
    train_queries = None
    for split, expected in (("train", 600), ("dev", 120), ("test", 120)):
        _, trees, rows = load_split(directory, split)
        ids, texts = {q["id"] for q in rows}, {q["question"].casefold() for q in rows}
        assert len(rows) == expected and len(ids) == expected
        assert not documents & trees.keys() and not questions & ids and not canonical & texts
        documents.update(trees)
        questions.update(ids)
        canonical.update(texts)
        if split == "train":
            train_queries = ids
    routes = training_routes(directory)
    assert {r["query_id"] for r in routes} <= train_queries
    positives = [r for r in routes if r["target_ids"]]
    assert sum(r["depth"] > 0 for r in positives) / len(positives) > 0.5
    assert max(len(r["candidate_ids"]) for r in routes) > 8
    for row in random.Random(42).sample(routes, 20):
        assert len(set(row["target_ids"])) == len(row["target_ids"]) <= 4
