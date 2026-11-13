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


@pytest.mark.parametrize("extra_path", ["extra.json", "nested/artifact-manifest.json"])
def test_artifact_summary_rejects_unlisted_files_before_reading_results(tmp_path, extra_path):
    (tmp_path / "artifact-manifest.json").write_text("{}")
    extra = tmp_path / extra_path
    extra.parent.mkdir(parents=True, exist_ok=True)
    extra.write_text("{}")
    with pytest.raises(ValueError, match="inventory"):
        summary.summarize(tmp_path)


def test_comparison_rejects_unknown_label_changes_and_duplicate_questions():
    rows = [{"id": "a", "document": "one", "answerable": False}]
    with pytest.raises(ValueError, match="metadata"):
        summary.check_pairs(rows, [{**rows[0], "answerable": True}])
    with pytest.raises(ValueError, match="Duplicate"):
        summary.check_pairs(rows, rows * 2)
