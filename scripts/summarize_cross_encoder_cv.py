"""Verify frozen fold choices and compute out-of-fold retrieval/abstention results."""

import argparse
import json
import math
import statistics
from pathlib import Path

from summarize_candidate_scoring import verify_manifest

from arborvault.candidate_scoring import training_groups
from arborvault.cross_validation import (
    apply_gate,
    candidates,
    discriminative_metrics,
    features,
    fit_gate,
    fold_plan,
    gate_scores,
    operating_metrics,
    pool,
    retrieval_rows,
    tune_threshold,
    validate_plan,
)
from arborvault.qwen3 import file_hash
from arborvault.routing_eval import summarize

MODELS = ("cross-encoder-tuned", "cross-encoder-base", "qwen-base", "e5-text")
POLICIES = ("coverage", "max-conservative", "max-balanced", "gate-conservative", "gate-balanced")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def close(a, b):
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(close(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return (
            isinstance(b, (list, tuple))
            and len(a) == len(b)
            and all(close(x, y) for x, y in zip(a, b, strict=True))
        )
    if isinstance(a, float):
        return isinstance(b, (int, float)) and math.isclose(a, b, abs_tol=1e-9, rel_tol=1e-8)
    return a == b


def paired_delta(base, other, metric):
    """Conditional, paired document bootstrap stratified by outer fold."""
    import numpy as np

    b, t = ({r["id"]: r for r in rows} for rows in (base, other))
    if b.keys() != t.keys():
        raise ValueError("Unpaired questions")
    groups, differences = {}, []
    for identity, row in b.items():
        match = t[identity]
        if (row["document"], row["fold"], row["answerable"]) != (
            match["document"],
            match["fold"],
            match["answerable"],
        ):
            raise ValueError("Unpaired metadata")
        if row["answerable"] != (metric != "unknown_abstention_rate"):
            continue
        key = "recall" if metric == "recall" else "abstained"
        delta = float(match[key]) - float(row[key])
        groups.setdefault(row["fold"], {}).setdefault(row["document"], []).append(delta)
        differences.append(delta)
    rng = np.random.default_rng(20261113)
    numerator, denominator = np.zeros(2000), np.zeros(2000)
    for fold in sorted(groups):
        documents = [groups[fold][d] for d in sorted(groups[fold])]
        weights = rng.multinomial(
            len(documents), np.ones(len(documents)) / len(documents), size=2000
        )
        numerator += weights @ np.array([sum(v) for v in documents])
        denominator += weights @ np.array([len(v) for v in documents])
    samples = numerator / denominator
    return {
        "mean_delta": statistics.mean(differences),
        "bootstrap_95_percent_interval": np.quantile(samples, [0.025, 0.975]).tolist(),
        "documents": sum(len(v) for v in groups.values()),
        "queries": len(differences),
        "replicates": 2000,
        "seed": 20261113,
        "stratified_by": "outer fold",
        "scope": "Conditional on saved out-of-fold predictions, not refitted training runs",
        "positive_changes": sum(v > 0 for v in differences),
        "negative_changes": sum(v < 0 for v in differences),
    }


def check_cache(cache, subset, trees, dense=False):
    if set(cache) != {q["id"] for q in subset}:
        raise ValueError("Score cache question set mismatch")
    for q in subset:
        tree = trees[q["document"]]
        expected = (
            {i for i in tree.nodes if not tree.children[i]}
            if dense
            else set(candidates(tree, q["question"]))
        )
        entry = cache[q["id"]]
        if set(entry["scores"]) != expected or not all(
            math.isfinite(s) for s in entry["scores"].values()
        ):
            raise ValueError("Wrong candidates or non-finite scores")
        if not dense and (
            set(entry["tokens"]) != expected
            or any(n > 512 or n < 1 for n in entry["tokens"].values())
        ):
            raise ValueError("Input budget violation")


def reliability(rows):
    bins = []
    for i in range(10):
        selected = [r for r in rows if min(9, int(r["gate_probability"] * 10)) == i]
        bins.append(
            {
                "lower": i / 10,
                "upper": (i + 1) / 10,
                "count": len(selected),
                "mean_probability": statistics.mean(r["gate_probability"] for r in selected)
                if selected
                else None,
                "answerable_fraction": statistics.mean(r["answerable"] for r in selected)
                if selected
                else None,
            }
        )
    return {
        "brier": statistics.mean((r["gate_probability"] - r["answerable"]) ** 2 for r in rows),
        "bins": bins,
        "ece_10_bins": sum(
            b["count"] * abs(b["mean_probability"] - b["answerable_fraction"])
            for b in bins
            if b["count"]
        )
        / len(rows),
    }


def verify(root, dataset):
    manifest = verify_manifest(root)
    plan, complete = read(root / "plan.json"), read(root / "complete.json")
    if not complete["complete"] or complete["folds"] != 5:
        raise ValueError("Incomplete CV")
    if plan["dataset_sha256"] != file_hash(dataset / "manifest.json"):
        raise ValueError("Dataset changed")
    trees, queries, historical = pool(dataset)
    by_id = {q["id"]: q for q in queries}
    validate_plan(queries, plan["folds"])
    expected = fold_plan(queries)
    if historical != plan["historical_split"] or any(
        a["roles"] != b["roles"] for a, b in zip(expected, plan["folds"], strict=True)
    ):
        raise ValueError("Fold assignment changed")
    references = {name: read(root / "scores" / (name + ".json")) for name in MODELS[1:]}
    for name, cache in references.items():
        check_cache(cache, queries, trees, dense=name == "e5-text")
    pooled = {name + "/" + policy: [] for name in MODELS for policy in POLICIES}
    folds, diagnostics = [], {name: [] for name in MODELS}
    for fold in plan["folds"]:
        number = fold["fold"]
        directory = root / f"fold-{number}"
        roles = {role: [by_id[i] for i in ids] for role, ids in fold["roles"].items()}
        training = read(directory / "training-plan.json")
        groups = training_groups(trees, roles["fit"])
        if training["fit_ids"] != fold["roles"]["fit"] or training["pairs"] != sum(
            len(g["items"]) for g in groups
        ):
            raise ValueError("Training membership changed")
        selection = read(directory / "selection.json")
        history = selection["history"]
        steps = math.ceil(len(groups) / plan["config"]["gradient_accumulation_queries"])
        required_steps = sorted({0, *range(25, steps + 1, 25), steps})
        if [r["step"] for r in history] != required_steps:
            raise ValueError("Missing training checkpoint")
        if read(directory / "training.json") != history:
            raise ValueError("Training history changed")
        for record in history:
            probe = read(directory / f"probe/step-{record['step']}.json")
            check_cache(probe, roles["selection"], trees)
            metric = summarize(retrieval_rows(trees, roles["selection"], probe))
            if not close(
                record["quality"], [metric[k] for k in ("recall", "evidence_coverage", "mrr")]
            ):
                raise ValueError("Selection metric mismatch")
        chosen = max(history[1:], key=lambda r: (r["quality"], -r["step"]))
        locked = read(directory / "locked-before-evaluation.json")
        if (
            chosen != selection["selected"]
            or chosen != locked["selection"]
            or not locked["evaluation_metrics_not_used"]
        ):
            raise ValueError("Checkpoint selection mismatch")
        if locked["evaluation_ids"] != fold["roles"]["evaluation"]:
            raise ValueError("Held-out IDs changed")
        trained = read(directory / "scores.json")
        check_cache(trained, roles["gate_fit"] + roles["threshold"] + roles["evaluation"], trees)
        methods = {**references, "cross-encoder-tuned": trained}
        reports = {}
        for name, cache in methods.items():
            policy = locked["policies"][name]
            gx = [features(trees[q["document"]], q, cache[q["id"]]) for q in roles["gate_fit"]]
            gate = fit_gate(gx, [q["answerable"] for q in roles["gate_fit"]])
            if not close(gate, policy["gate"]):
                raise ValueError("Gate fit mismatch")
            # Saved, verified coefficients give identical serialized arithmetic at boundaries.
            gate = policy["gate"]
            tx = [features(trees[q["document"]], q, cache[q["id"]]) for q in roles["threshold"]]
            tr = retrieval_rows(trees, roles["threshold"], cache)
            maximum = tune_threshold(tr, [x[0] for x in tx])
            learned = tune_threshold(tr, gate_scores(gate, tx))
            if not close(maximum, policy["max_score"]) or not close(
                learned, policy["learned_gate"]
            ):
                raise ValueError("Threshold calibration mismatch")
            outer = roles["evaluation"]
            rows = retrieval_rows(trees, outer, cache)
            fx = [features(trees[q["document"]], q, cache[q["id"]]) for q in outer]
            raw, probabilities = [x[0] for x in fx], gate_scores(gate, fx)
            diagnostics[name].append(
                {
                    "fold": number,
                    "raw_max": discriminative_metrics(rows, raw),
                    "learned_gate": discriminative_metrics(rows, probabilities),
                }
            )
            for row, probability in zip(rows, probabilities, strict=True):
                row.update(fold=number, gate_probability=probability)
            for variant in POLICIES:
                if variant == "coverage":
                    result = rows
                else:
                    feature, operating = variant.split("-")
                    scores, thresholds = (
                        (raw, maximum) if feature == "max" else (probabilities, learned)
                    )
                    result = apply_gate(rows, scores, thresholds[operating]["threshold"])
                report = operating_metrics(result)
                key = name + "/" + variant
                reports[key] = {k: v for k, v in report.items() if k != "queries"}
                pooled[key].extend(result)
        folds.append(
            {
                "fold": number,
                "counts": fold["counts"],
                "selected_step": chosen["step"],
                "selection_curve": history,
                "methods": reports,
            }
        )
    for rows in pooled.values():
        if len(rows) != len(queries) or {r["id"] for r in rows} != set(by_id):
            raise ValueError("OOF predictions are not one per query")
        rows.sort(key=lambda r: r["id"])
    methods = {name: operating_metrics(rows) for name, rows in pooled.items()}
    comparisons = {}
    pairs = [
        (reference + "/" + p, "cross-encoder-tuned/" + p)
        for reference in MODELS[1:]
        for p in POLICIES
    ]
    pairs += [(model + "/coverage", model + "/" + p) for model in MODELS for p in POLICIES[1:]]
    for before, after in pairs:
        comparisons[after + " minus " + before] = {
            metric: paired_delta(pooled[before], pooled[after], metric)
            for metric in ("recall", "unknown_abstention_rate", "answerable_false_abstention_rate")
        }
    confidence = {}
    for name in MODELS:
        rows = pooled[name + "/coverage"]
        ordered = sorted(rows, key=lambda r: (-r["gate_probability"], r["id"]))
        curves = []
        for fraction in (0.25, 0.5, 0.75, 0.9, 0.95, 1):
            count = math.ceil(len(rows) * fraction)
            selected = ordered[:count]
            curves.append(
                {
                    "retained_fraction": count / len(rows),
                    "unsupported_return_rate": sum(
                        not r["answerable"] or r["recall"] == 0 for r in selected
                    )
                    / count,
                    "note": "Descriptive OOF probability ordering, ID tie-break; not a deployment threshold",
                }
            )
        confidence[name] = {
            "folds": diagnostics[name],
            "probability_reliability": reliability(rows),
            "descriptive_coverage_curve": curves,
            "mean_fold_raw_auroc": statistics.mean(
                d["raw_max"]["answerability_auroc"] for d in diagnostics[name]
            ),
            "mean_fold_gate_auroc": statistics.mean(
                d["learned_gate"]["answerability_auroc"] for d in diagnostics[name]
            ),
            "mean_fold_gate_unknown_ap": statistics.mean(
                d["learned_gate"]["unknown_average_precision"] for d in diagnostics[name]
            ),
        }
    return {
        "code_revision": plan["code_revision"],
        "verified_files": len(manifest),
        "fold_assignments_verified": True,
        "checkpoint_rules_verified": True,
        "gates_and_thresholds_verified": True,
        "evaluation_questions": len(queries),
        "answerable": plan["answerable"],
        "unknown": len(queries) - plan["answerable"],
        "documents": len(trees),
        "methods": methods,
        "folds": folds,
        "paired_document_bootstrap": comparisons,
        "answerability": confidence,
        "runtime": complete,
        "scope": plan["scope"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--dataset", type=Path, default=Path("datasets/qasper-routing-v2"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.run, args.dataset)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(
        json.dumps(
            {
                "verified_files": result["verified_files"],
                "questions": result["evaluation_questions"],
                "methods": len(result["methods"]),
                "folds": len(result["folds"]),
            }
        )
    )


if __name__ == "__main__":
    main()
