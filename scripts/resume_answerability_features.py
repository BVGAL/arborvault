"""Recover the frozen probe stage after a tokenizer API compatibility failure.

Retrieval outputs remain byte-identical. This does not rerun models, inspect test
metrics, fit classifiers or change calibration rules.
"""

import argparse
import json
import time
from pathlib import Path

from arborvault.answerability import extract_probe_features
from arborvault.long_documents import load_corpus
from arborvault.qwen3 import file_hash


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8", newline="\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--repair-revision", required=True)
    args = parser.parse_args()
    directory = args.experiment
    config = json.loads(args.config.read_text(encoding="utf-8"))
    plan = json.loads((directory / "plan.json").read_text(encoding="utf-8"))
    if file_hash(args.config) != plan["config_sha256"] or config != plan["config"]:
        raise ValueError("Frozen configuration changed")
    if (directory / "probe-features.json").exists() or (
        directory / "artifact-hashes.json"
    ).exists():
        raise ValueError("Refusing to overwrite completed features or sealed artifacts")
    manifest = Path(config["dataset"]) / "manifest.json"
    if file_hash(manifest) != plan["dataset_manifest_sha256"]:
        raise ValueError("Frozen corpus changed")
    protected = [directory / "scores" / (name + ".json") for name in config["models"] + ["e5-text"]]
    before = {p.relative_to(directory).as_posix(): file_hash(p) for p in protected}
    source = plan["scorer_config"]
    _, trees, queries = load_corpus(Path(config["dataset"]))
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed

    torch.set_num_threads(4)
    set_seed(config["seed"])
    begin = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(
        source["cross_encoder"], revision=source["cross_encoder_revision"], local_files_only=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        source["cross_encoder"], revision=source["cross_encoder_revision"], local_files_only=True
    ).to("cuda")

    def deadline():
        if time.monotonic() - begin > 600:
            raise TimeoutError("Feature recovery deadline reached")

    usage = json.loads((directory / "usage.json").read_text(encoding="utf-8"))
    usage["answerability_probe"] = {
        **extract_probe_features(config, directory, trees, queries, model, tokenizer, deadline),
        "seconds": time.monotonic() - begin,
        "extra_forwards_per_deployed_gate": 1,
    }
    after = {p.relative_to(directory).as_posix(): file_hash(p) for p in protected}
    if before != after:
        raise ValueError("Retrieval outputs changed during feature recovery")
    save(directory / "usage.json", usage)
    progress = [
        json.loads(line) for line in (directory / "progress.jsonl").read_text().splitlines()
    ]
    original_elapsed = progress[-1]["elapsed_seconds"]
    runtime = {
        "complete": True,
        "seconds": original_elapsed + usage["answerability_probe"]["seconds"],
        "timing_scope": "Original last progress plus recovery; excludes compatibility repair waiting",
    }
    save(directory / "runtime.json", runtime)
    save(
        directory / "feature-recovery.json",
        {
            "reason": "Transformers 5 removed prepare_for_model; use the public pair-tokenization API",
            "original_retrieval_revision": plan["source_revision"],
            "repair_revision": args.repair_revision,
            "repair_files": {
                str(p): file_hash(p)
                for p in (
                    Path("src/arborvault/answerability.py"),
                    Path("scripts/resume_answerability_features.py"),
                    Path("tests/test_answerability.py"),
                )
            },
            "retrieval_hashes_before": before,
            "retrieval_hashes_after": after,
            "protocol_or_test_labels_changed": False,
            "test_metrics_inspected_before_repair": False,
            "pair_prefix_regression_checked_with_pinned_real_tokenizer": True,
            "empty_context_encoding": "Native question-only sequence; no synthetic second separator",
        },
    )
    with (directory / "progress.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps({"stage": "complete", "elapsed_seconds": runtime["seconds"]}) + "\n"
        )
    save(
        directory / "artifact-hashes.json",
        {
            p.relative_to(directory).as_posix(): file_hash(p)
            for p in sorted(directory.rglob("*"))
            if p.is_file()
        },
    )
    print(json.dumps({"complete": True, "retrieval_outputs_unchanged": before == after}))


if __name__ == "__main__":
    main()
