import copy
from pathlib import Path

import pytest

from arborvault.llm_eval import compare
from arborvault.training_data import export, load_bundle

ROOT = Path(__file__).resolve().parents[1]


def test_training_export_reproducible_without_test_label_leakage(tmp_path):
    expected, checked_in = load_bundle(ROOT / "datasets/routing-bootstrap-v1")
    actual = export(ROOT / "datasets/public-docs-v1", tmp_path)
    assert actual == expected
    _, regenerated = load_bundle(tmp_path)
    assert regenerated == checked_in
    held_out = set(expected["public_query_ids"]["held_out_test"])
    assert len(held_out) == 44
    assert all(r["id"].split("/")[0] not in held_out for rows in regenerated.values() for r in rows)
    file = tmp_path / "train.jsonl"
    file.write_text(file.read_text() + "\n")
    with pytest.raises(ValueError, match="checksum"):
        load_bundle(tmp_path)


def reports():
    report = {
        "dataset_sha256": "same",
        "split": "test",
        "k": 5,
        "word_budget": 512,
        "experiment": {
            "variant": "base",
            "base_model": "open/model",
            "base_revision": "a" * 40,
            "prompt_sha256": "same",
            "max_output_tokens": 1024,
            "temperature": 0,
            "seed": 0,
            "runtime_id": "same-runtime",
            "quantization": "bf16",
            "provider": "ollama",
        },
        "complete": True,
        "provider_usage": {"failures": 0},
        "methods": {
            method: {
                "queries": [{"id": "one"}],
                "recall_at_k": 0.5,
                "mrr": 0.5,
                "ndcg": 0.5,
                "evidence_coverage": 0.5,
            }
            for method in ("llm_navigation", "llm_auto")
        },
    }
    tuned = copy.deepcopy(report)
    tuned["experiment"]["variant"] = "fine-tuned"
    tuned["methods"]["llm_navigation"]["recall_at_k"] = 0.75
    return report, tuned


def test_paired_comparison_reports_delta_and_rejects_different_base():
    base, tuned = reports()
    assert (
        compare(base, tuned)["delta_fine_tuned_minus_base"]["llm_navigation"]["recall_at_k"] == 0.25
    )
    tuned["experiment"]["base_model"] = "different/model"
    with pytest.raises(ValueError, match="base_model"):
        compare(base, tuned, exploratory=True)


def test_cloud_or_failed_runs_cannot_claim_controlled_improvement():
    base, tuned = reports()
    base["experiment"]["provider"] = "ollama-cloud"
    with pytest.raises(ValueError, match="Controlled"):
        compare(base, tuned)
    assert compare(base, tuned, exploratory=True)["comparison"] == "exploratory"
    base, tuned = reports()
    tuned["complete"] = False
    with pytest.raises(ValueError, match="Controlled"):
        compare(base, tuned)
