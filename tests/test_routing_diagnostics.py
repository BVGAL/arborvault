import json
from pathlib import Path

import pytest

from arborvault.domain import Node
from arborvault.qasper import load_split
from arborvault.routing_diagnostics import (
    oracle_states,
    set_metrics,
    summarize_states,
    token_diagnostics,
)
from arborvault.routing_v2 import SelectionGrammar, Tree


def test_oracle_states_keep_all_candidates_but_follow_only_annotated_branches():
    tree = Tree(
        [
            Node("root", None, "Doc", "", ["Doc"], 0),
            Node("a", "root", "A", "", ["Doc", "A"], 1),
            Node("b", "root", "B", "", ["Doc", "B"], 2),
            Node("pa", "a", "PA", "evidence", ["Doc", "A", "PA"], 3),
            Node("pb", "b", "PB", "alternative", ["Doc", "B", "PB"], 4),
        ]
    )
    query = {"id": "q", "document": "doc", "answerable": True, "gold_sets": [["pa"], ["pb"]]}
    states = list(oracle_states(tree, query))
    assert states[0]["candidate_ids"] == ["a", "b"]
    assert states[0]["target_ids"] == ["a"]
    assert states[0]["union_target_ids"] == ["a", "b"]
    assert states[1]["candidate_ids"] == ["pa"]
    unknown = list(oracle_states(tree, {**query, "answerable": False, "gold_sets": []}))
    assert len(unknown) == 1 and unknown[0]["target_ids"] == []


def test_set_metrics_penalize_both_narrow_and_excessively_broad_selections():
    assert set_metrics(["a"], ["a", "b"])["recall"] == 0.5
    assert set_metrics(["a", "b"], ["a"])["precision"] == 0.5
    assert set_metrics([], ["a"])["precision"] == 0
    assert set_metrics([], [])["exact"]
    assert not set_metrics(["a"], [])["exact"]


def test_real_development_oracle_states_do_not_contain_training_queries():
    directory = Path(__file__).resolve().parents[1] / "datasets/qasper-routing-v2"
    _, trees, queries = load_split(directory, "dev")
    _, _, train = load_split(directory, "train")
    states = [s for q in queries for s in oracle_states(trees[q["document"]], q)]
    assert len({s["query_id"] for s in states}) == 120
    assert not {s["query_id"] for s in states} & {q["id"] for q in train}
    assert all(set(s["target_ids"]) <= set(s["candidate_ids"]) for s in states)
    assert sum(not s["target_ids"] for s in states) == 20


def test_teacher_forced_metrics_exclude_forced_tokens_and_detect_early_close():
    torch = pytest.importorskip("torch")

    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            return list(text.encode())

        def decode(self, tokens, skip_special_tokens=False):
            return bytes(t for t in tokens if t != 255).decode()

    tokenizer = Tokenizer()
    target = tokenizer.encode(json.dumps(["0", "1"], separators=(",", ":"))) + [255]
    logits = torch.full((len(target), 256), -10.0)
    for i, token in enumerate(target):
        logits[i, token] = 10
    comma = target.index(ord(","))
    logits[comma, ord("]")] = 20
    grammar = SelectionGrammar(tokenizer, 3, 4, [255])
    rows = token_diagnostics(logits, target, grammar, tokenizer)
    assert sum(r["premature_close"] for r in rows) == 1
    assert rows[comma]["constrained_correct"] is False
    assert any(r["legal_count"] == 1 for r in rows)
    expected = torch.nn.functional.cross_entropy(logits, torch.tensor(target)).item()
    assert sum(r["nll"] for r in rows) / len(rows) == pytest.approx(expected)


def test_nonterminal_choice_accuracy_excludes_equivalent_eos_choices():
    token = {
        "nll": 1.0,
        "constrained_nll": 0.5,
        "raw_correct": True,
        "constrained_correct": True,
        "legal_count": 2,
        "premature_close": False,
    }
    row = {
        "target_ids": ["a"],
        "depth": 0,
        "metrics": set_metrics(["a"], ["a"]),
        "union_metrics": set_metrics(["a"], ["a", "b"]),
        "teacher_forced": [{**token, "constrained_correct": False}, token],
    }
    result = summarize_states([row])["all"]
    assert result["choice_token_accuracy"] == 0.5
    assert result["nonterminal_choice_accuracy"] == 0
    assert result["nonterminal_choice_tokens"] == 1
