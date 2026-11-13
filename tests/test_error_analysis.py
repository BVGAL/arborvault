"""Protect diagnostic claims against confusing empty routes with provider failures."""

import copy
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/analyze_qwen3.py"
spec = importlib.util.spec_from_file_location("analyze_qwen3", SCRIPT)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def row(identity, recall=0, predicted=None, calls=1, failures=0, reason="no_route"):
    predicted = predicted or []
    return {
        "id": identity,
        "gold": ["doc:2"],
        "predicted": predicted,
        "recall": recall,
        "abstained": not predicted,
        "fallback_reason": reason,
        "provider_usage": {"calls": calls, "failures": failures},
    }


def test_diagnosis_distinguishes_first_call_rejection_and_retained_evidence():
    assert analysis.outcome(row("a")) == "root_rejection"
    assert analysis.outcome(row("a", calls=2)) == "empty_after_deeper_calls"
    assert analysis.outcome(row("a", failures=1, reason="selector_error")) == "selector_failure"
    successful = row("a", recall=1, predicted=["doc:2"], failures=1, calls=4)
    missed = row("b", predicted=["doc:1"], calls=2)
    report = {
        "dataset_sha256": "same",
        "split": "test",
        "k": 5,
        "word_budget": 512,
        "methods": {
            name: {"queries": [successful, missed]} for name in ("llm_navigation", "dense_sections")
        },
    }
    queries = {i: {"question": i, "group": "one", "language": "en"} for i in ("a", "b")}
    tuned = copy.deepcopy(report)
    tuned["methods"]["llm_navigation"]["queries"][0] = row("a")
    result = analysis.paired_audit(report, tuned, queries)
    assert result["base"]["selector_failure_queries"] == ["a"]
    assert result["base"]["selector_failure_with_retained_evidence"] == ["a"]
    assert result["base"]["empty_answerable"] == 0
    assert result["root_rejection_lost"] == ["a"]
    tuned["word_budget"] = 1024
    with pytest.raises(ValueError, match="different evaluation inputs"):
        analysis.paired_audit(report, tuned, queries)


def test_template_audit_exposes_semantic_repetition_and_missing_training_depths():
    root = Path(__file__).resolve().parents[1]
    audit = analysis.training_audit(root / "datasets/routing-bootstrap-v1")
    assert audit["train"]["public_depth_histogram"] == {"0": 8}
    assert audit["train"]["public_positive_documents"] == {"httpx-limits": 6}
    assert audit["train_validation_shared_query_templates"] == 17
    assert analysis.normalized_template("How long? 012345abcd") == "how long? <entity>"


def test_oracle_uses_unchanged_evidence_budget_instead_of_returning_gold_directly():
    from arborvault.domain import Node

    nodes = [
        Node("root", None, "topic", "needle " * 600, ["doc"], 0),
        Node("leaf", "root", "detail", "needle evidence", ["doc", "detail"], 1, {"line": 3}),
    ]
    query = {
        "id": "a",
        "question": "needle",
        "relevant": [{"document": "doc", "section_line": 3, "evidence": "needle evidence"}],
    }
    # Both nodes are on the gold path, but the long ancestor ranks first and exhausts budget.
    report = analysis.oracle_audit({"doc": nodes}, [query], 5, 512)
    assert report["recall_at_k"] == 0
    assert report["evidence_coverage"] == 0
    assert report["queries"][0]["predicted"] == ["doc:0"]
