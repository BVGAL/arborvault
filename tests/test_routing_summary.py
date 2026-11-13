import importlib.util
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[1] / "scripts/summarize_routing_v2.py"
spec = importlib.util.spec_from_file_location("summarize_routing_v2", path)
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def test_document_bootstrap_keeps_pairs_and_neutral_results_exact():
    rows = [
        {"id": "a", "document": "one", "answerable": True, "recall": 1},
        {"id": "b", "document": "one", "answerable": True, "recall": 0.5},
        {"id": "c", "document": "two", "answerable": True, "recall": 0},
        {"id": "d", "document": "two", "answerable": False, "recall": None},
    ]
    result = summary.clustered_delta(rows, list(reversed(rows)), repetitions=100)
    assert result["documents"] == 2
    assert result["mean_delta"] == 0
    assert result["bootstrap_95_percent_interval"] == [0, 0]
    assert result["queries_unchanged"] == 3
    with pytest.raises(ValueError, match="question sets"):
        summary.clustered_delta(rows, rows[1:], repetitions=100)


def test_artifact_summary_rejects_unlisted_files_before_reading_results(tmp_path):
    (tmp_path / "artifact-manifest.json").write_text("{}")
    (tmp_path / "extra.json").write_text("{}")
    with pytest.raises(ValueError, match="inventory"):
        summary.summarize(tmp_path)
