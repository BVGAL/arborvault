import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from arborvault.candidate_scoring import retrieve
from arborvault.domain import Node
from arborvault.long_benchmark import DocumentIndex, traverse, verify_recorded_traversal
from arborvault.long_documents import load_corpus, parse_manual
from arborvault.routing_v2 import Tree


def test_html_hierarchy_and_nonduplicating_blocks():
    html = """<div class="body" role="main"><section id="root"><h1>Manual</h1>
    <section id="s"><h2>Methods<a class="headerlink">link</a></h2>
    <p>One two three four five six seven.</p><dl class="py method">
    <dt id="api">f(x)</dt><dd><p>Return a result.</p><ul><li><p>Nested item.</p></li></ul></dd>
    </dl><section id="empty"><h3>Empty</h3></section></section></section></div>"""
    doc = parse_manual(html, "manual", "https://example.org/manual", max_words=4)
    nodes = doc["nodes"]
    assert [n["content"] for n in nodes if n["content"]] == [
        "One two three four",
        "five six seven.",
        "f(x)",
        "Return a result.",
        "Nested item.",
    ]
    assert next(n for n in nodes if n["content"] == "Return a result.")["path"][:-1] == [
        "Manual",
        "Methods",
        "f(x)",
    ]
    assert not any(n["title"] == "Empty" for n in nodes)
    tree = Tree([Node(**n) for n in nodes])
    api = next(n for n in tree.nodes.values() if n.locator["kind"] == "api")
    assert "Return a result." in tree.views([api])[0].content
    assert "Nested item." in tree.views([api])[0].content


def fixture_tree():
    return Tree(
        [
            Node("r", None, "Root", "", ["Root"], 0),
            Node("b", "r", "Branch", "", ["Root", "Branch"], 1),
            *[
                Node(str(i), "b", str(i), f"evidence item {i}", ["Root", "Branch", str(i)], i + 2)
                for i in range(8)
            ],
        ]
    )


@pytest.mark.parametrize("mode", ["flat", "tree_scored"])
@pytest.mark.parametrize("budget", [1, 4, 64])
def test_lazy_traversal_matches_exhaustive_policy(mode, budget):
    tree = fixture_tree()
    index = DocumentIndex(tree)
    values = {i: float(n.ordinal % 3) for i, n in tree.nodes.items() if i != "r"}
    calls = []

    def get_scores(ids):
        calls.extend(ids)
        return {i: values[i] for i in ids}

    lazy = traverse(index, index.scores("evidence"), get_scores, mode=mode, budget=budget, width=1)
    _, old = retrieve(tree, "evidence", values, mode=mode, budget=budget, width=1)
    assert lazy["scored_ids"] == old["scored_ids"]
    assert lazy["reached_ids"] == old["reached_ids"]
    assert len(calls) <= budget
    assert len(set(calls)) == len(calls)


def test_missing_or_nan_scores_fail():
    index = DocumentIndex(fixture_tree())
    for fn in [lambda ids: {}, lambda ids: dict.fromkeys(ids, float("nan"))]:
        with pytest.raises(ValueError):
            traverse(index, index.scores("evidence"), fn, mode="flat", budget=3)


def test_recorded_replay_allows_only_roundoff_and_preserves_model_choices():
    index = DocumentIndex(fixture_tree())
    lexical = {i: 1.0 for i in index.ids}
    values = {i: float(index.tree.nodes[i].ordinal) for i in index.ids}
    original = traverse(
        index, lexical, lambda ids: {i: values[i] for i in ids}, mode="flat", budget=3
    )
    perturbed = {**lexical, "3": 1.0 + 1e-14}
    with pytest.raises(ValueError, match="Exact lexical"):
        verify_recorded_traversal(index, perturbed, values, original, mode="flat", budget=3)
    ties = verify_recorded_traversal(
        index, perturbed, values, original, mode="flat", budget=3, lexical_tolerance=1e-12
    )
    assert len(ties) == 1 and ties[0]["candidate_set_changed_vs_canonical"]
    with pytest.raises(ValueError, match="Substantive lexical"):
        verify_recorded_traversal(
            index,
            {**lexical, "3": 1.1},
            values,
            original,
            mode="flat",
            budget=3,
            lexical_tolerance=1e-12,
        )
    wrong = {**original, "reached_ids": original["reached_ids"][::-1]}
    with pytest.raises(ValueError, match="Final ranking"):
        verify_recorded_traversal(index, lexical, values, wrong, mode="flat", budget=3)


def test_bm25_scores_are_reproducible_across_python_hash_seeds():
    code = """
import hashlib,json
from arborvault.long_documents import load_corpus
from arborvault.long_benchmark import DocumentIndex
_,trees,queries=load_corpus('datasets/long-manuals-v1')
question=next(q for q in queries if q['id']=='u-ctypes-02')
scores=DocumentIndex(trees['ctypes']).scores(question['question'])
print(hashlib.sha256(json.dumps(scores,sort_keys=True).encode()).hexdigest())
"""
    outputs = [
        subprocess.check_output(
            [sys.executable, "-c", code],
            cwd=Path(__file__).resolve().parents[1],
            env={**os.environ, "PYTHONHASHSEED": str(seed)},
        )
        for seed in (0, 2)
    ]
    assert outputs[0] == outputs[1]


def test_frozen_corpus_and_split_integrity():
    directory = Path(__file__).resolve().parents[1] / "datasets/long-manuals-v1"
    manifest, trees, queries = load_corpus(directory)
    assert len(queries) == 384 and len(trees) == 16
    roles = {
        role: {q["document"] for q in queries if q["split"] == role}
        for role in ["test", "calibration"]
    }
    assert not roles["test"] & roles["calibration"]
    assert manifest["counts"]["test"] == {"documents": 12, "answerable": 120, "unknown": 168}
    assert len({q["id"] for q in queries}) == 384
    by_id = {q["id"]: q for q in queries}
    for q in queries:
        if q["category"] == "cross_document_transfer":
            origin = by_id[q["audit"]["answer_elsewhere_question"]]
            assert q["split"] == origin["split"]
            assert q["document"] != origin["document"]
    assert min(d["stats"]["words"] for d in manifest["documents"]) >= 8000
    docs = json.loads((directory / "documents.json").read_text(encoding="utf-8"))
    for doc in docs:
        html = (directory / "sources" / (doc["id"] + ".html")).read_text(encoding="utf-8")
        assert parse_manual(html, doc["id"], doc["url"])["nodes"] == doc["nodes"]
