"""Reconstruct the published semantic-abstention decisions and check frozen inputs."""

import json
from pathlib import Path

import numpy as np

from arborvault.support_data import file_hash
from arborvault.support_v2_data import load
from arborvault.support_v2_eval import evaluate_test, score_models
from arborvault.support_v2_features import base_cache, read_role


def main():
    root = Path("docs/benchmarks/support-v2")
    frozen = json.loads((root / "frozen-inputs.json").read_text(encoding="utf-8"))
    for name, expected in frozen["sha256"].items():
        assert file_hash(Path(name)) == expected, "Frozen input changed: " + name
    config_path = Path("configs/support-v2.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    dataset, rows = load(Path(config["dataset"]))
    manifest = json.loads((root / "features/manifest.json").read_text(encoding="utf-8"))
    assert manifest["examples_sha256"] == dataset["examples_sha256"]
    assert manifest["dataset_manifest_sha256"] == file_hash(
        Path(config["dataset"]) / "manifest.json"
    )
    assert manifest["config_sha256"] == file_hash(config_path)
    cache, sources = base_cache(config)
    assert sources == manifest["sources"]
    assert set(manifest["roles"]) == {"fit", "selection", "calibration", "test"}
    for role, spec in manifest["roles"].items():
        assert file_hash(root / "features" / (role + ".json")) == spec["sha256"]
    fitted_path = root / "results/fitted-models.json"
    fitted = json.loads(fitted_path.read_text(encoding="utf-8"))
    old = json.loads(Path(config["previous_probe"]).read_text(encoding="utf-8"))["fitted_probes"][
        "context"
    ]
    assert old == fitted["models"]["previous"]["probe"]
    assert fitted["test_used_for_fitting"] is False
    for gates in fitted["thresholds"].values():
        assert all(
            stats["false_rejection"] <= config["conservative_max_false_rejection_rate"]
            for stats in gates["conservative"]["calibration_strata"].values()
        )
    test, features = read_role(root / "features", rows, "test", cache)
    values = score_models(fitted["models"], features)
    summary, decisions = evaluate_test(config, test, values, fitted)
    published_summary = json.loads((root / "results/summary.json").read_text(encoding="utf-8"))
    provenance = published_summary.pop("provenance")
    assert provenance == {
        "config_sha256": file_hash(config_path),
        "feature_manifest_sha256": file_hash(root / "features/manifest.json"),
        "fitted_models_sha256": file_hash(fitted_path),
    }
    assert summary == published_summary
    published = json.loads((root / "results/decisions.json").read_text(encoding="utf-8"))
    assert len(published) == len(decisions) == 2176
    for actual, saved in zip(decisions, published, strict=True):
        assert actual["id"] == saved["id"] and actual["accepted"] == saved["accepted"]
        for name in values:
            assert np.isclose(actual["scores"][name], saved["scores"][name], rtol=0, atol=1e-12)
    # Exact question twins make the controlled test's question-only AUC chance.
    for point in ("conservative", "balanced"):
        assert summary["cohorts"]["contrast"]["question_only"][point]["auroc"] == 0.5
    print(
        "Verified: frozen inputs, sources, feature hashes, previous head, calibration constraints, all test decisions, metrics and cluster intervals"
    )


if __name__ == "__main__":
    main()
