import json

import pytest

from arborvault.budget_navigation import ContentGuidance, budgeted_traverse
from arborvault.decision_scoring import decision_messages, fit_decision_text
from arborvault.domain import Node
from arborvault.long_benchmark import DocumentIndex, traverse
from arborvault.long_documents import load_corpus
from arborvault.routing_v2 import Tree


def make_tree():
    nodes = [Node("r", None, "Manual", "", ["Manual"], 0)]
    for i in range(40):
        nodes.append(Node(f"noise{i}", "r", "Other", "irrelevant", ["Manual", "Other"], i + 1))
    parent, path = "r", ["Manual"]
    for i in range(4):
        identity = f"branch{i}"
        path = [*path, "target section"]
        nodes.append(Node(identity, parent, "target section", "", path, len(nodes)))
        parent = identity
    nodes.append(Node("gold", parent, "Answer", "special evidence", [*path, "Answer"], len(nodes)))
    nodes.append(Node("alternative", "r", "Alternative", "", ["Manual", "Alternative"], len(nodes)))
    nodes.append(
        Node(
            "other",
            "alternative",
            "Other answer",
            "alternative evidence",
            ["Manual", "Alternative", "Other answer"],
            len(nodes),
        )
    )
    return Tree(nodes)


def test_incremental_search_reaches_deep_leaf_and_keeps_alternative():
    index = DocumentIndex(make_tree())
    priorities = {i: (10 if index.tree.children[i] else 0) for i in index.ids}
    priorities.update(gold=5, other=4)
    values = {i: (3 if i.startswith("branch") else 2 if i == "gold" else 1) for i in index.ids}
    old = traverse(
        index, priorities, lambda ids: {i: values[i] for i in ids}, mode="tree_scored", budget=32
    )
    assert "gold" not in old["scored_ids"]
    used = []

    def scorer(ids):
        used.extend(ids)
        return {i: values[i] for i in ids}

    result = budgeted_traverse(index, priorities, scorer, budget=32)
    assert "gold" in result["reached_ids"]
    assert "other" in result["reached_ids"]  # A lower-scoring branch was not discarded.
    assert result["reached_ids"][0] == "gold"
    assert len(used) == len(set(used)) <= 32
    assert result["branch_pairs"] <= 12


@pytest.mark.parametrize("budget", [1, 2, 3, 8, 64, 128])
def test_limits_and_access_are_enforced(budget):
    index = DocumentIndex(make_tree())
    result = budgeted_traverse(
        index, index.scores("evidence"), lambda ids: dict.fromkeys(ids, 1.0), budget=budget
    )
    assert len(result["scored_ids"]) <= budget
    assert result["branch_pairs"] <= 3 * budget // 8
    expanded = {index.tree.root.id}
    for step in result["trace"]:
        assert all(index.tree.nodes[i].parent_id in expanded for i in step["branches_scored"])
        expanded.update(step["expanded"])
        assert all(index.tree.nodes[i].parent_id in expanded for i in step["leaves_scored"])
    assert not (set(result["scored_ids"]) & set(result["unscored_frontier"]))


def test_guidance_uses_only_descendants_and_preserves_leaf_inputs():
    index = DocumentIndex(make_tree())
    lexical = index.scores("special")
    guide = ContentGuidance(index, lexical)
    assert guide.priorities["branch0"] == lexical["gold"]
    assert "special evidence" in guide.texts["branch0"]
    assert "alternative evidence" not in guide.texts["branch0"]
    assert all(guide.texts[i] == index.texts[i] for i in index.leaves)


def test_bad_scores_and_bad_budget_fail():
    index = DocumentIndex(make_tree())
    for fn in [lambda ids: {}, lambda ids: dict.fromkeys(ids, float("nan"))]:
        with pytest.raises(ValueError):
            budgeted_traverse(index, index.scores("x"), fn, budget=8)
    with pytest.raises(ValueError):
        budgeted_traverse(index, index.scores("x"), lambda ids: {}, budget=0)


def test_decision_fitting_preserves_state_and_fits_both_models():
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return [0] * len(messages[-1]["content"])

    class CrossTokenizer:
        def __call__(self, question, text, **kwargs):
            return {"input_ids": [0] * (len(question) + len(text))}

    text = json.dumps(
        {"document": "doc", "path": ["method"], "kind": "branch", "excerpt": "x" * 700}
    )
    result, meta = fit_decision_text(
        "question", text, [Tokenizer(), Tokenizer()], CrossTokenizer(), 512
    )
    assert max(meta["lengths"]) <= 512
    assert meta["excerpt_characters_removed"] > 0
    assert json.loads(result)["path"] == ["method"]
    assert decision_messages("question", result)[-1]["content"].endswith("Answer Yes or No.")
    with pytest.raises(ValueError):
        fit_decision_text("q" * 600, text, [Tokenizer()], CrossTokenizer())


def test_frozen_navigation_corpus():
    from pathlib import Path

    manifest, trees, queries = load_corpus(Path("datasets/navigation-v2"))
    assert manifest["counts"]["test"] == {"documents": 6, "answerable": 48, "unknown": 48}
    assert len(trees) == 10 and len(queries) == 192
    old = json.loads(Path("datasets/long-manuals-v1/documents.json").read_text(encoding="utf-8"))
    assert not ({q["document"] for q in queries if q["split"] == "test"} & {d["id"] for d in old})
