import json
import shutil
from pathlib import Path

import pytest

from arborvault.benchmark import budgeted
from arborvault.dataset import load_dataset, within
from arborvault.domain import Node

ROOT = Path(__file__).resolve().parents[1] / "datasets" / "public-docs-v1"


def test_corpus_hashes_license_provenance_and_qrels():
    _, docs, queries, _, digest = load_dataset(ROOT)
    assert len(docs) == 8 and len(queries) == 60 and len(digest) == 64
    assert sum(q["category"] == "unanswerable" for q in queries) == 12
    assert sum(q["split"] == "dev" for q in queries) == 16
    assert sum(q["split"] == "test" for q in queries) == 44
    assert {q["language"] for q in queries} == {"en", "it"}


def test_changed_corpus_is_rejected(tmp_path):
    shutil.copytree(ROOT, tmp_path / "dataset")
    path = tmp_path / "dataset" / "corpus" / "fastapi-body.md"
    path.write_text("altered")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_dataset(tmp_path / "dataset")


def test_missing_evidence_is_rejected(tmp_path):
    shutil.copytree(ROOT, tmp_path / "dataset")
    path = tmp_path / "dataset" / "queries.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["relevant"][0]["evidence"] = "not present in this source anywhere"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    with pytest.raises(ValueError, match="Evidence not found"):
        load_dataset(tmp_path / "dataset")


def test_context_budget_and_dataset_path_boundary(tmp_path):
    node = Node("id", None, "title", "one two three four five", [], 0)
    result = budgeted([(None, node, 1, "test")], 5, 3)
    assert result[0][-1] == "one two three"
    with pytest.raises(ValueError, match="escapes"):
        within(tmp_path, "../outside")
