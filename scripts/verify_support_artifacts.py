"""Verify the published experiment chain and reproduce decisions from frozen features."""

import json
from pathlib import Path

import numpy as np

from arborvault.support_data import file_hash, load
from arborvault.support_probe import metrics, paired_intervals, read_features, score_models


def main():
    root = Path("docs/benchmarks/support-v1")
    frozen = json.loads((root / "frozen-inputs.json").read_text(encoding="utf-8"))
    for name, expected in frozen["sha256"].items():
        assert file_hash(Path(name)) == expected, "Frozen input changed: " + name
    config = json.loads(Path("configs/support-v1.json").read_text(encoding="utf-8"))
    dataset = Path(config["dataset"])
    _, rows = load(dataset)
    features = root / "features"
    manifest = json.loads((features / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["dataset_manifest_sha256"] == file_hash(dataset / "manifest.json")
    assert manifest["examples_sha256"] == file_hash(dataset / "examples.jsonl")
    assert manifest["config_sha256"] == file_hash(Path("configs/support-v1.json"))
    assert set(manifest["roles"]) == {"fit", "selection", "calibration", "test"}
    for role, spec in manifest["roles"].items():
        assert file_hash(features / (role + ".json")) == spec["sha256"]
    fitted_path = root / "results/fitted-probes.json"
    fitted = json.loads(fitted_path.read_text(encoding="utf-8"))
    summary = json.loads((root / "results/summary.json").read_text(encoding="utf-8"))
    assert summary["provenance"]["feature_manifest_sha256"] == file_hash(features / "manifest.json")
    assert summary["provenance"]["fitted_probes_sha256"] == file_hash(fitted_path)
    assert summary["provenance"]["config_sha256"] == manifest["config_sha256"]
    old = json.loads(Path(config["previous_probe"]).read_text(encoding="utf-8"))
    assert fitted["fitted_probes"]["previous"] == old["fitted_probes"]["context"]
    test, test_features = read_features(features, rows, "test")
    scores = score_models(fitted["fitted_probes"], test_features)
    saved = json.loads((root / "results/decisions.json").read_text(encoding="utf-8"))
    assert len(saved) == len(test) == summary["examples"]
    for index, (row, decision) in enumerate(zip(test, saved, strict=True)):
        assert row["id"] == decision["id"]
        assert row["answerable"] == decision["supported"]
        for name, values in scores.items():
            assert np.isclose(values[index], decision["scores"][name], rtol=0, atol=1e-12)
            for point, spec in fitted["thresholds"][name].items():
                assert decision["accepted"][name][point] == (values[index] >= spec["threshold"])
    for name, values in scores.items():
        for point, spec in fitted["thresholds"][name].items():
            actual = metrics(test, values, spec["threshold"])
            assert actual == summary["metrics"][name][point]
    actual_bootstrap = paired_intervals(
        test,
        scores,
        fitted["thresholds"],
        seed=config["bootstrap_seed"],
        replicates=config["bootstrap_replicates"],
    )
    assert actual_bootstrap == summary["paired_bootstrap"]
    assert np.allclose(scores["context_only"], 0.5, rtol=0, atol=1e-12)
    print(
        "Verified: frozen inputs, data/feature hashes, previous coefficients, all test scores, decisions, metrics and bootstrap intervals"
    )


if __name__ == "__main__":
    main()
