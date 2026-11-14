"""Verify frozen inputs, replay every traversal and calibrate without test labels."""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from arborvault.budget_navigation import ContentGuidance, budgeted_traverse
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
    parser.add_argument("--dataset", type=Path, default=Path("datasets/navigation-v2"))
    parser.add_argument("--config", type=Path, default=Path("configs/navigation-v2.json"))
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
    config = read(args.config)
    if plan["config"] != config or plan["config_sha256"] != file_hash(args.config):
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
    models = config.get("models", ["qwen-base", "cross-encoder-base", "tacit-4b"])
    for label in models:
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
                for mode in config["modes"]:
                    name = f"{mode}-{budget}"
                    replay = entry["policies"][name]

                    def cache_key(i, mode=mode, index=index):
                        return (
                            (
                                "guided"
                                if mode == "tree_guided" and index.tree.children[i]
                                else "static"
                            )
                            + "|"
                            + i
                        )

                    scores = {i: entry["scores"][cache_key(i)] for i in replay["scored_ids"]}
                    if mode in {"flat", "tree_scored"}:
                        verify_recorded_traversal(
                            index,
                            lexical[q["id"]],
                            scores,
                            replay,
                            mode=mode,
                            budget=budget,
                            width=4,
                        )
                    else:
                        guide = ContentGuidance(index, lexical[q["id"]])
                        calls = []

                        def get_scores(ids, scores=scores, calls=calls):
                            calls.extend(ids)
                            return {i: scores[i] for i in ids}

                        expected = budgeted_traverse(
                            index,
                            guide.priorities if mode == "tree_guided" else lexical[q["id"]],
                            get_scores,
                            budget=budget,
                            branch_batch=config["branch_batch"],
                            passage_batch=config["passage_batch"],
                        )
                        if expected != replay or calls != replay["scored_ids"]:
                            raise ValueError("Exact incremental replay mismatch")
                        if len(calls) != len(set(calls)) or len(calls) > budget:
                            raise ValueError("Invalid charged pair budget")
                        expanded = {index.tree.root.id}
                        seen = set()
                        for step in replay["trace"]:
                            if any(
                                index.tree.nodes[i].parent_id not in expanded
                                for i in step["branches_scored"]
                            ):
                                raise ValueError("Scored branch without parent access")
                            seen.update(step["branches_scored"])
                            if any(i not in seen or i in expanded for i in step["expanded"]):
                                raise ValueError("Uncharged or duplicate expansion")
                            expanded.update(step["expanded"])
                            if any(
                                index.tree.nodes[i].parent_id not in expanded
                                for i in step["leaves_scored"]
                            ):
                                raise ValueError("Leaf reached without ancestor access")
                    union.update(cache_key(i) for i in replay["scored_ids"])
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
            for mode in config["modes"]:
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
                            "input_tokens": sum(
                                cache[q["id"]]["tokens"][
                                    (
                                        "guided"
                                        if mode == "tree_guided"
                                        and trees[q["document"]].children[i]
                                        else "static"
                                    )
                                    + "|"
                                    + i
                                ]
                                for i in used
                            ),
                            "unused_allowance": budget - len(used),
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
                    "mean_input_tokens": sum(r["input_tokens"] for r in inspected) / len(inspected),
                    "mean_unused_allowance": sum(r["unused_allowance"] for r in inspected)
                    / len(inspected),
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

    probe_output, extra_pairs = None, []
    if config.get("answerability_history"):
        from arborvault.answerability_analysis import analyze_probes

        probe_output, extra_pairs = analyze_probes(
            config,
            directory,
            trees,
            queries,
            policies,
            confidences,
            results,
            rows_by_policy,
            calibration,
        )

    # Resample the same source families for every method and gate.
    def group_of(d):
        if "family_groups" in config:
            return config["family_groups"][d]
        return "archive-family" if d in {"zipfile", "tarfile"} else d

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
    pairs = []
    for model in models:
        for budget in config["budgets"]:
            for left, right in [
                ("tree_budgeted", "tree_scored"),
                ("tree_guided", "tree_scored"),
                ("tree_guided", "tree_budgeted"),
                ("tree_guided", "flat"),
            ]:
                pairs.append(
                    (f"{model}/{left}-{budget}/coverage", f"{model}/{right}-{budget}/coverage")
                )
    for mode in config["modes"]:
        for budget in config["budgets"]:
            if "tacit-4b" in models:
                pairs.append(
                    (f"tacit-4b/{mode}-{budget}/coverage", f"qwen-base/{mode}-{budget}/coverage")
                )
            pairs.append(
                (
                    f"cross-encoder-base/{mode}-{budget}/coverage",
                    f"qwen-base/{mode}-{budget}/coverage",
                )
            )
    pairs += [(f"{model}/tree_guided-64/coverage", "e5-text/coverage") for model in models]
    pairs += extra_pairs
    comparisons = [
        {
            "left": a,
            "right": b,
            "recall_difference": results[a]["recall"] - results[b]["recall"],
            "paired_cluster_bootstrap_95ci": np.quantile(
                bootstrap[a]["recall"] - bootstrap[b]["recall"], [0.025, 0.975]
            ).tolist(),
            "abstention_differences": {
                metric: {
                    "difference": float(
                        np.mean(
                            [
                                r["abstained"]
                                for r in rows_by_policy[a]
                                if r["answerable"] == (metric == "false_abstention_rate")
                            ]
                        )
                    )
                    - float(
                        np.mean(
                            [
                                r["abstained"]
                                for r in rows_by_policy[b]
                                if r["answerable"] == (metric == "false_abstention_rate")
                            ]
                        )
                    ),
                    "paired_cluster_bootstrap_95ci": np.quantile(
                        bootstrap[a][metric] - bootstrap[b][metric], [0.025, 0.975]
                    ).tolist(),
                }
                for metric in ("unknown_abstention_rate", "false_abstention_rate")
            },
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
        "answerability_probe": probe_output,
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
                        "e5-text/coverage",
                    ]
                },
            }
        )
    )


if __name__ == "__main__":
    main()
