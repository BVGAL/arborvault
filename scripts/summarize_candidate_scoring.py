"""Verify a completed candidate-scoring run and its replay before publication."""

import argparse
import json
import math
import statistics
from pathlib import Path

from arborvault.candidate_scoring import (
    calibrate,
    evaluate_scores,
    partition,
    probe_quality,
    probe_queries,
    training_groups,
)
from arborvault.qasper import load_split
from arborvault.qwen3 import file_hash
from arborvault.routing_eval import context, score, summarize


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify_manifest(root):
    root = root.resolve()
    manifest = read(root / "artifact-manifest.json")
    actual = {
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and p != root / "artifact-manifest.json"
    }
    if actual != set(manifest):
        raise ValueError("Artifact inventory mismatch")
    for name, expected in manifest.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or file_hash(path) != expected:
            raise ValueError("Artifact checksum mismatch")
    return manifest


def verify(root, dataset, reference=None):
    manifest = verify_manifest(root)
    plan, result, complete = (
        read(root / name) for name in ("plan.json", "results.json", "complete.json")
    )
    if not complete["complete"] or not complete["test_not_opened"]:
        raise ValueError("Experiment incomplete or test protocol changed")
    if plan["dataset_manifest_sha256"] != file_hash(dataset / "manifest.json"):
        raise ValueError("Dataset mismatch")
    if reference is not None:
        extra = read(reference)
        if (
            extra["dataset_sha256"] != plan["dataset_manifest_sha256"]
            or extra["revision"] != plan["config"]["embedding_revision"]
            or extra["k"] != 5
            or extra["word_budget"] != 512
        ):
            raise ValueError("Standard-text reference protocol mismatch")
        if set(extra["methods"]) != {"dense_e5_text", "bm25_text", "hybrid_rrf_text"}:
            raise ValueError("Missing standard-text reference method")
        result["methods"].update(extra["methods"])
    _, trees, train = load_split(dataset, "train")
    fit, cal = partition(train)
    probe = probe_queries(cal)
    for field, queries in (
        ("training_query_ids", fit),
        ("calibration_query_ids", cal),
        ("checkpoint_probe_query_ids", probe),
    ):
        if plan[field] != [q["id"] for q in queries]:
            raise ValueError("Partition mismatch")
    locked = read(root / "selection-before-development.json")
    if not locked["development_not_evaluated"] or not locked["test_not_opened"]:
        raise ValueError("Selection protocol changed")
    curves = {}
    probe_groups = training_groups(trees, probe)
    for model, key in (("qwen", "qwen"), ("cross-encoder", "cross_encoder")):
        selection = read(root / (model + "-selection.json"))
        if locked[key] != selection:
            raise ValueError("Checkpoint selection changed after development")
        curves[model] = []
        for checkpoint in selection["all_checkpoints"]:
            suffix = "base" if checkpoint["step"] == 0 else "step-" + str(checkpoint["step"])
            scores = read(root / "probe" / f"{model}-{suffix}.json")
            if checkpoint["quality"] != probe_quality(trees, probe, scores):
                raise ValueError("Probe metric mismatch")
            losses, rankings = [], []
            for group in probe_groups:
                values = scores[group["id"]]["scores"]
                losses.append(
                    sum(
                        i["weight"]
                        * (
                            max(values[i["node_id"]], 0)
                            - i["label"] * values[i["node_id"]]
                            + math.log1p(math.exp(-abs(values[i["node_id"]])))
                        )
                        for i in group["items"]
                    )
                )
                positive = [values[i["node_id"]] for i in group["items"] if i["label"]]
                negative = [values[i["node_id"]] for i in group["items"] if not i["label"]]
                if positive and negative:
                    rankings.append(
                        statistics.mean(
                            float(p > n) + 0.5 * (p == n) for p in positive for n in negative
                        )
                    )
            curves[model].append(
                {
                    "step": checkpoint["step"],
                    "mean_flat_tree_recall": checkpoint["quality"][0],
                    "probe_weighted_bce": statistics.mean(losses),
                    "probe_pairwise_accuracy": statistics.mean(rankings),
                }
            )
        chosen = max(selection["all_checkpoints"][1:], key=lambda c: (c["quality"], -c["step"]))
        if chosen != selection["selected_trained"]:
            raise ValueError("Checkpoint rule mismatch")
    _, dev_trees, dev = load_split(dataset, "dev")
    expected_ids = {q["id"] for q in dev}
    questions = {q["id"]: q for q in dev}
    evidence_stages = {}
    for name, report in result["methods"].items():
        if (
            len(report["queries"]) != len(dev)
            or {q["id"] for q in report["queries"]} != expected_ids
        ):
            raise ValueError("Development query set mismatch: " + name)
        for row in report["queries"]:
            query = questions[row["id"]]
            if row["document"] != query["document"] or row["answerable"] != query["answerable"]:
                raise ValueError("Question metadata mismatch")
            ids = row["predicted"]
            tree = dev_trees[query["document"]]
            if (
                len(ids) > 5
                or len(set(ids)) != len(ids)
                or any(i not in tree.nodes or tree.children[i] for i in ids)
            ):
                raise ValueError("Invalid retrieved passages")
            recomputed = score(context([tree.nodes[i] for i in ids]), query)
            if any(row[k] != v for k, v in recomputed.items()):
                raise ValueError("Evidence metric mismatch")
        aggregate = summarize(report["queries"])
        if any(report[k] != v for k, v in aggregate.items()):
            raise ValueError("Aggregate mismatch")
        if name.endswith("/64"):
            stage_rows = []
            for row in report["queries"]:
                if not row["answerable"]:
                    continue
                gold_sets = questions[row["id"]]["gold_sets"]

                def availability(ids, gold_sets=gold_sets):
                    return max(len(set(ids) & set(g)) / len(g) for g in gold_sets)

                values = {
                    "scored_gold_availability": availability(row["scored_ids"]),
                    "retained_gold_availability": availability(row["reached_ids"]),
                    "top5_gold_recall_without_word_limit": availability(row["reached_ids"][:5]),
                    "context_gold_recall": row["recall"],
                }
                ordered = list(values.values())
                if any(a + 1e-12 < b for a, b in zip(ordered[:-1], ordered[1:], strict=True)):
                    raise ValueError("Non-monotone evidence filtering stages")
                stage_rows.append({"id": row["id"], **values})
            evidence_stages[name] = {
                "means": {
                    k: statistics.mean(r[k] for r in stage_rows) for k in stage_rows[0] if k != "id"
                },
                "queries": stage_rows,
                "note": "Annotation availability at each stage, ignoring downstream ranking/word constraints; not guaranteed achievable recall",
            }
    for model in ("qwen-base", "qwen-tuned", "cross-encoder-base", "cross-encoder-tuned"):
        cache = read(root / "scores" / (model + "-dev.json"))
        cal_cache = read(root / "scores" / (model + "-calibration.json"))
        for mode in ("flat", "tree"):
            policy = calibrate(trees, cal, cal_cache, mode=mode)
            if policy != result["calibration"][model][mode]:
                raise ValueError("Calibration mismatch")
            for label, budget, threshold in [(str(b), b, None) for b in (32, 64, 128)] + [
                ("64-calibrated", 64, policy["chosen"]["threshold"])
            ]:
                recomputed = evaluate_scores(
                    dev_trees, dev, cache, mode=mode, budget=budget, threshold=threshold
                )
                if recomputed != result["methods"][f"{model}/{mode}/{label}"]:
                    raise ValueError("Replay mismatch")
    if reference is not None:
        from summarize_routing_v2 import clustered_delta

        for model in ("qwen-base", "qwen-tuned", "cross-encoder-base", "cross-encoder-tuned"):
            for mode in ("flat", "tree"):
                name = model + "/" + mode + "/64"
                result["paired_document_bootstrap"][name + "/minus-e5-text"] = clustered_delta(
                    result["methods"]["dense_e5_text"]["queries"],
                    result["methods"][name]["queries"],
                )
    return {
        "verified_files": len(manifest),
        "verified_neural_policies": 32,
        "training_queries": len(fit),
        "calibration_queries": len(cal),
        "development_queries": len(dev),
        "checkpoint_rules_verified": True,
        "standard_text_reference_verified": reference is not None,
        "checkpoint_curves": curves,
        "evidence_stages": evidence_stages,
        "calibration_rules_verified": True,
        "dataset_sha256": plan["dataset_manifest_sha256"],
        "code_revision": plan["code_revision"],
        "runtime": complete,
        "methods": {
            name: {k: v for k, v in report.items() if k != "queries"}
            for name, report in result["methods"].items()
        },
        "paired_document_bootstrap": result["paired_document_bootstrap"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--dataset", type=Path, default=Path("datasets/qasper-routing-v2"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    result = verify(args.run, args.dataset, args.reference)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in {"methods", "paired_document_bootstrap"}}
        )
    )


if __name__ == "__main__":
    main()
