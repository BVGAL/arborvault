"""Verify frozen inputs, replay every traversal and calibrate without test labels."""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from arborvault.cross_validation import (
    apply_gate,
    discriminative_metrics,
    operating_metrics,
    tune_threshold,
)
from arborvault.long_benchmark import DocumentIndex, result_row, verify_recorded_traversal
from arborvault.long_documents import load_corpus
from arborvault.qwen3 import file_hash


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def compact(rows):
    return {k: v for k, v in operating_metrics(rows).items() if k != "queries"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path("datasets/long-manuals-v1"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    directory = args.experiment.resolve()
    hashes = read(directory / "artifact-hashes.json")
    actual_files = {
        p.relative_to(directory).as_posix()
        for p in directory.rglob("*")
        if p.is_file() and p.name != "artifact-hashes.json"
    }
    if set(hashes) != actual_files:
        raise ValueError("Artifact manifest does not cover the complete experiment")
    for name, expected in hashes.items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory) or file_hash(path) != expected:
            raise ValueError("Artifact hash mismatch: " + name)
    plan = read(directory / "plan.json")
    config = read(Path("configs/long-manuals-v1.json"))
    if plan["config"] != config or plan["config_sha256"] != file_hash(
        Path("configs/long-manuals-v1.json")
    ):
        raise ValueError("Protocol mismatch")
    manifest, trees, queries = load_corpus(args.dataset)
    if plan["dataset_manifest_sha256"] != file_hash(args.dataset / "manifest.json"):
        raise ValueError("Dataset lock mismatch")
    for role in ("calibration", "test"):
        if plan[role + "_ids"] != [q["id"] for q in queries if q["split"] == role]:
            raise ValueError("Question partition mismatch")
    if not read(directory / "runtime.json")["complete"]:
        raise ValueError("Incomplete inference")
    indexes = {d: DocumentIndex(t) for d, t in trees.items()}
    lexical = {q["id"]: indexes[q["document"]].scores(q["question"]) for q in queries}
    policies, confidences, caches, lexical_ties = {}, {}, {}, []
    for label in ("qwen-base", "cross-encoder-base", "cross-encoder-tuned"):
        cache = read(directory / "scores" / (label + ".json"))
        if set(cache) != {q["id"] for q in queries}:
            raise ValueError("Missing score queries")
        caches[label] = cache
        for q in queries:
            index, entry = indexes[q["document"]], cache[q["id"]]
            if set(entry["tokens"]) != set(entry["scores"]) or any(
                n > 512 or n < 1 for n in entry["tokens"].values()
            ):
                raise ValueError("Invalid token lengths")
            union = set()
            for budget in config["budgets"]:
                for mode in ("flat", "tree_scored"):
                    name = f"{mode}-{budget}"
                    replay = entry["policies"][name]
                    roundoff = verify_recorded_traversal(
                        index,
                        lexical[q["id"]],
                        entry["scores"],
                        replay,
                        mode=mode,
                        budget=budget,
                        width=config["width"],
                    )
                    if roundoff:
                        lexical_ties.append(
                            {
                                "model": label,
                                "question": q["id"],
                                "policy": name,
                                "frontiers": roundoff,
                            }
                        )
                    union.update(replay["scored_ids"])
                    key = label + "/" + name
                    policies.setdefault(key, []).append(result_row(index.tree, q, replay))
                    value = replay["max_passage_score"]
                    confidences.setdefault(key, []).append(
                        config["empty_reached_confidence"] if value is None else value
                    )
            if union != set(entry["scores"]):
                raise ValueError("Unexpected unused or missing pair scores")

    e5 = read(directory / "scores/e5-text.json")
    if set(e5) != {q["id"] for q in queries}:
        raise ValueError("Missing E5 queries")
    for label in ("e5-text", "bm25"):
        policies[label], confidences[label] = [], []
        for q in queries:
            index = indexes[q["document"]]
            scores = (
                e5[q["id"]]["scores"]
                if label == "e5-text"
                else {i: lexical[q["id"]][i] for i in index.leaves}
            )
            if set(scores) != set(index.leaves) or any(
                not math.isfinite(s) for s in scores.values()
            ):
                raise ValueError("Invalid full-document reference scores")
            ids = sorted(scores, key=lambda i: (-scores[i], index.tree.nodes[i].ordinal))
            policies[label].append(
                result_row(index.tree, q, {"reached_ids": ids, "scored_ids": ids})
            )
            confidences[label].append(max(scores.values()))

    calibration_positions = [i for i, q in enumerate(queries) if q["split"] == "calibration"]
    test_positions = [i for i, q in enumerate(queries) if q["split"] == "test"]
    calibration, results, rows_by_policy, candidate_diagnostics = {}, {}, {}, {}
    for label, cache in caches.items():
        for budget in config["budgets"]:
            for mode in ("flat", "tree_scored"):
                key = f"{label}/{mode}-{budget}"
                inspected = []
                for q in queries:
                    if q["split"] != "test" or not q["answerable"]:
                        continue
                    policy = cache[q["id"]]["policies"][f"{mode}-{budget}"]
                    used = set(policy["scored_ids"])
                    ceiling = max(len(used & set(g)) / len(g) for g in q["gold_sets"])
                    inspected.append(
                        {
                            "id": q["id"],
                            "candidate_recall_ceiling": ceiling,
                            "charged_pairs": len(used),
                            "branch_pairs": sum(
                                bool(trees[q["document"]].children[i]) for i in used
                            ),
                            "reached_passages": len(policy["reached_ids"]),
                        }
                    )
                candidate_diagnostics[key] = {
                    "candidate_recall_ceiling": sum(
                        r["candidate_recall_ceiling"] for r in inspected
                    )
                    / len(inspected),
                    "mean_charged_pairs": sum(r["charged_pairs"] for r in inspected)
                    / len(inspected),
                    "mean_branch_pairs": sum(r["branch_pairs"] for r in inspected) / len(inspected),
                    "queries": inspected,
                }
    for key, rows in policies.items():
        cal_rows, test_rows = [
            [rows[i] for i in ids] for ids in (calibration_positions, test_positions)
        ]
        cal_values, test_values = [
            [confidences[key][i] for i in ids] for ids in (calibration_positions, test_positions)
        ]
        choices = tune_threshold(cal_rows, cal_values)
        calibration[key] = choices
        for gate in ("coverage", "conservative", "balanced"):
            threshold = None if gate == "coverage" else choices[gate]["threshold"]
            gated = apply_gate(test_rows, test_values, threshold)
            name = key + "/" + gate
            rows_by_policy[name] = gated
            strata = {}
            for category in sorted({r["category"] for r in gated if not r["answerable"]}):
                group = [r for r in gated if r["category"] == category]
                strata[category] = {
                    "count": len(group),
                    "rejected": sum(r["abstained"] for r in group),
                    "rejection_rate": sum(r["abstained"] for r in group) / len(group),
                }
            results[name] = {
                **compact(gated),
                "threshold": threshold,
                "unknown_categories": strata,
                "per_document": {
                    d: compact([r for r in gated if r["document"] == d])
                    for d in sorted({r["document"] for r in gated})
                },
                "confidence_metrics": discriminative_metrics(test_rows, test_values),
            }

    # Resample the same source families for every method and gate.
    def group_of(d):
        return "logging-family" if d in {"logging", "logging-cookbook"} else d

    groups = sorted({group_of(queries[i]["document"]) for i in test_positions})
    rng = np.random.default_rng(config["bootstrap_seed"])
    draws = rng.integers(0, len(groups), (config["bootstrap_replicates"], len(groups)))
    weights = np.stack([(draws == j).sum(axis=1) for j in range(len(groups))], axis=1)
    bootstrap = {}
    for name, rows in rows_by_policy.items():
        samples = {}
        for metric in ("recall", "unknown_abstention_rate", "false_abstention_rate"):
            sums, counts = [], []
            for group in groups:
                selected = [
                    r
                    for r in rows
                    if group_of(r["document"]) == group
                    and r["answerable"] == (metric != "unknown_abstention_rate")
                ]
                sums.append(
                    sum(r["recall"] if metric == "recall" else r["abstained"] for r in selected)
                )
                counts.append(len(selected))
            samples[metric] = (weights @ np.array(sums)) / (weights @ np.array(counts))
        bootstrap[name] = samples
        results[name]["cluster_bootstrap_95ci"] = {
            m: np.quantile(v, [0.025, 0.975]).tolist() for m, v in samples.items()
        }
    pairs = [
        ("cross-encoder-tuned/flat-64/coverage", "qwen-base/flat-64/coverage"),
        ("cross-encoder-tuned/flat-64/coverage", "e5-text/coverage"),
        ("cross-encoder-tuned/flat-64/coverage", "cross-encoder-base/flat-64/coverage"),
        ("qwen-base/flat-64/coverage", "e5-text/coverage"),
    ]
    pairs += [
        (f"{model}/tree_scored-{budget}/coverage", f"{model}/flat-{budget}/coverage")
        for model in ("qwen-base", "cross-encoder-base", "cross-encoder-tuned")
        for budget in config["budgets"]
    ]
    comparisons = [
        {
            "left": a,
            "right": b,
            "recall_difference": results[a]["recall"] - results[b]["recall"],
            "paired_cluster_bootstrap_95ci": np.quantile(
                bootstrap[a]["recall"] - bootstrap[b]["recall"], [0.025, 0.975]
            ).tolist(),
        }
        for a, b in pairs
    ]
    analysis = {
        "verified": True,
        "analysis_script_sha256": file_hash(Path(__file__)),
        "analysis_numpy_version": np.__version__,
        "verified_artifact_files": len(hashes),
        "source_revision": plan["source_revision"],
        "dataset_counts": manifest["counts"],
        "test_families": groups,
        "calibration": calibration,
        "results": results,
        "paired_comparisons": comparisons,
        "queries": rows_by_policy,
        "candidate_diagnostics": candidate_diagnostics,
        "lexical_roundoff_replay": {
            "absolute_tolerance": 0.0,
            "affected_policies": lexical_ties,
            "neural_scores_and_recorded_selections_changed": False,
        },
        "usage": read(directory / "usage.json"),
        "runtime": read(directory / "runtime.json"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(
        json.dumps(
            {
                "verified": True,
                "policies": len(results),
                "test_queries": len(test_positions),
                "primary": {
                    k: results[k]["recall"]
                    for k in [
                        "qwen-base/flat-64/coverage",
                        "cross-encoder-base/flat-64/coverage",
                        "cross-encoder-tuned/flat-64/coverage",
                        "e5-text/coverage",
                    ]
                },
            }
        )
    )


if __name__ == "__main__":
    main()
