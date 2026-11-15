"""Recompute published support-v3 selection, gates, decisions and uncertainty."""

import json
from pathlib import Path

from arborvault.support_data import file_hash
from arborvault.support_v2_data import load
from arborvault.support_v2_models import calibrate_common, select_score
from arborvault.support_v3_eval import summarize
from arborvault.support_v3_training import choice, choose, old_models, read


def main():
    root = Path("docs/benchmarks/support-v3")
    config_path = Path("configs/support-v3.json")
    config = read(config_path)
    for name, expected in read(root / "frozen-inputs.json")["sha256"].items():
        assert file_hash(Path(name)) == expected, "Frozen input changed: " + name
    dataset, rows = load(Path(config["dataset"]))
    training = read(root / "training/training.json")
    assert training["config_sha256"] == file_hash(config_path)
    assert training["dataset_manifest_sha256"] == file_hash(
        Path(config["dataset"]) / "manifest.json"
    )
    assert training["examples_sha256"] == dataset["examples_sha256"]
    assert training["frozen_before"] == training["frozen_after"]
    assert training["selected_adapter_sha256"] == file_hash(root / "training/adapter.json")
    assert training["calibration_or_test_used_for_training"] is False
    fit = [r for r in rows if r["role"] == "fit"]
    selection = [r for r in rows if r["role"] == "selection"]
    assert training["fit_ids"] == [r["id"] for r in fit]
    assert training["selection_ids"] == [r["id"] for r in selection]
    baselines = read(root / "training/baselines.json")
    prior, original_gates = old_models(config)
    assert all(baselines["models"][k] == v for k, v in prior.items())
    assert baselines["original_v2_gates"] == original_gates
    baseline_selection = read(root / "training/selection-baselines.json")
    assert baseline_selection["ids"] == [r["id"] for r in selection]
    refs = {
        name: select_score(selection, values, config)
        for name, values in baseline_selection["scores"].items()
    }
    for name, expected in baseline_selection["metrics"].items():
        assert refs[name] == expected
    development = read(root / "training/selection.json")
    assert development["ids"] == [r["id"] for r in selection]
    for candidate in development["candidates"]:
        actual = choice(
            selection,
            development["scores"][str(candidate["step"])],
            refs,
            candidate["step"],
            config,
        )
        assert actual == {k: v for k, v in candidate.items() if k != "weights_sha256"}
    assert choose(development["candidates"]) == training["selected"]
    assert training["selected"]["weights_sha256"] == training["selected_adapter_sha256"]
    calibration = [r for r in rows if r["role"] == "calibration"]
    cal = read(root / "results/calibration.json")
    assert cal["ids"] == [r["id"] for r in calibration]
    assert cal["adapter_sha256"] == file_hash(root / "training/adapter.json")
    assert cal["training_sha256"] == file_hash(root / "training/training.json")
    for name, values in cal["scores"].items():
        expected = (
            original_gates
            if name == "v2_original_gate"
            else calibrate_common(calibration, values, config)
        )
        assert cal["thresholds"][name] == expected
    test = [r for r in rows if r["role"] == "test"]
    decisions = read(root / "results/decisions.json")
    assert [r["id"] for r in test] == [d["id"] for d in decisions]
    values = {name: [d["scores"][name] for d in decisions] for name in cal["thresholds"]}
    for row, decision in zip(test, decisions, strict=True):
        assert decision["supported"] == row["answerable"]
        for name, gates in cal["thresholds"].items():
            assert decision["accepted"][name] == {
                point: decision["scores"][name] >= spec["threshold"]
                for point, spec in gates.items()
            }
    summary = summarize(config, test, values, cal["thresholds"], training["selected"]["eligible"])
    published = read(root / "results/summary.json")
    controls = read(root / "results/input-controls.json")
    from arborvault.support_probe import auc

    contrast = [r for r in test if r["suite"] == "contrast"]
    for control in controls.values():
        assert control["ids"] == [r["id"] for r in contrast]
        assert (
            control["auroc"] == auc([r["answerable"] for r in contrast], control["scores"]) == 0.5
        )
    summary["input_control_aurocs"] = {k: v["auroc"] for k, v in controls.items()}
    summary["provenance"] = {
        "config_sha256": file_hash(config_path),
        "examples_sha256": dataset["examples_sha256"],
        "calibration_sha256": file_hash(root / "results/calibration.json"),
        "adapter_sha256": file_hash(root / "training/adapter.json"),
    }
    assert summary == published
    print(
        json.dumps(
            {
                "verified": "Frozen inputs, training roles, unchanged baselines, checkpoint selection, gates, all decisions, clustered intervals and promotion criteria",
                "test_examples": len(test),
            }
        )
    )


if __name__ == "__main__":
    main()
