import importlib.util
import json
import sys
from pathlib import Path

import pytest

from arborvault.domain import Node
from arborvault.routing_v2 import Tree

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location(
    "diagnostic_summary", SCRIPTS / "summarize_routing_diagnostics.py"
)
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def test_artifact_verification_rejects_extra_missing_changed_and_escaping_files(tmp_path):
    complete = tmp_path / "complete.json"
    complete.write_text('{"complete": true}')
    manifest = {"complete.json": summary.file_hash(complete)}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    summary.verify(tmp_path)
    extra = tmp_path / "unlisted.json"
    extra.write_text("{}")
    with pytest.raises(ValueError, match="inventory"):
        summary.verify(tmp_path)
    extra.unlink()
    complete.write_text('{"complete": false}')
    with pytest.raises(ValueError, match="checksum"):
        summary.verify(tmp_path)
    (tmp_path / "manifest.json").write_text(json.dumps({"../complete.json": "0" * 64}))
    with pytest.raises(ValueError, match="inventory"):
        summary.verify(tmp_path)


def test_trace_pruning_preserves_evidence_already_found_and_alternative_routes():
    tree = Tree(
        [
            Node("r", None, "D", "", ["D"], 0),
            Node("a", "r", "A", "a", ["D", "A"], 1),
            Node("b", "r", "B", "", ["D", "B"], 2),
            Node("c", "b", "C", "c", ["D", "B", "C"], 3),
        ]
    )
    query = {"id": "q", "answerable": True, "gold_sets": [["a"], ["c"]]}
    row = {
        "trace": [
            {"candidates": ["a", "b"], "selected": ["a", "b"]},
            {"candidates": ["c"], "selected": []},
        ],
        "reached_before_topk_and_word_budget": ["a"],
        "recall": 1,
    }
    report = summary.trace_diagnostics(tree, query, row, [])
    assert report["first_irreversible_prune_depth"] is None
    assert report["first_all_evidence_lost_depth"] is None
    assert report["pre_budget_recall"] == 1
    query["gold_sets"] = [["a", "c"]]
    report = summary.trace_diagnostics(tree, query, row, [])
    assert report["first_irreversible_prune_depth"] == 1
    assert report["first_all_evidence_lost_depth"] is None
    assert report["pre_budget_recall"] == 0.5


def test_query_aggregation_does_not_overweight_deep_questions_or_unknowns():
    rows = [
        {"query_id": "q", "document": "d", "target_ids": ["a"], "metrics": {"recall": 1}},
        {"query_id": "q", "document": "d", "target_ids": ["b"], "metrics": {"recall": 0}},
        {"query_id": "unknown", "document": "d", "target_ids": [], "metrics": {"recall": 1}},
    ]
    assert summary.query_metric(rows, "recall") == [
        {"id": "q", "document": "d", "answerable": True, "recall": 0.5}
    ]
