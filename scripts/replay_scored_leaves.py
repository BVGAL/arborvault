"""Secondary policy control: retain all scored leaves, with unchanged scorer/budget."""

import argparse
import json
from pathlib import Path

from summarize_routing_v2 import clustered_delta

from arborvault.candidate_scoring import calibrate, evaluate_scores, partition
from arborvault.qasper import load_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--dataset", type=Path, default=Path("datasets/qasper-routing-v2"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()

    def read(name):
        return json.loads((args.run / name).read_text(encoding="utf-8"))

    if not read("complete.json")["complete"]:
        raise ValueError("Primary experiment must be complete before secondary replay")
    _, train_trees, train = load_split(args.dataset, "train")
    _, cal = partition(train)
    _, trees, dev = load_split(args.dataset, "dev")
    primary = read("results.json")["methods"]
    reference_report = (
        json.loads(args.reference.read_text(encoding="utf-8")) if args.reference else None
    )
    if (
        reference_report
        and reference_report["dataset_sha256"] != read("plan.json")["dataset_manifest_sha256"]
    ):
        raise ValueError("Reference dataset mismatch")
    reports, policies, differences = {}, {}, {}
    for model in ("qwen-base", "qwen-tuned", "cross-encoder-base", "cross-encoder-tuned"):
        cache = read("scores/" + model + "-dev.json")
        calibration = read("scores/" + model + "-calibration.json")
        policy = calibrate(train_trees, cal, calibration, mode="tree_scored")
        policies[model] = policy
        for suffix, threshold in (("64", None), ("64-calibrated", policy["chosen"]["threshold"])):
            report = evaluate_scores(trees, dev, cache, mode="tree_scored", threshold=threshold)
            reports[model + "/tree_scored/" + suffix] = report
            if suffix == "64":
                for reference in ("tree", "flat"):
                    differences[model + "/minus-" + reference] = clustered_delta(
                        primary[model + "/" + reference + "/64"]["queries"], report["queries"]
                    )
                if reference_report:
                    differences[model + "/minus-e5-text"] = clustered_delta(
                        reference_report["methods"]["dense_e5_text"]["queries"], report["queries"]
                    )
    result = {
        "code_revision": args.code_revision,
        "scope": "Secondary policy control; primary selection unchanged",
        "new_model_inference": False,
        "methods": reports,
        "calibration": policies,
        "paired_document_bootstrap": differences,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"secondary_policies": len(reports), "model_inference": False}))


if __name__ == "__main__":
    main()
