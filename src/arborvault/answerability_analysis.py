"""Fit and select on historical documents, then calibrate on separate manuals."""

import json
import math
import statistics

from .answerability import (
    evidence_text,
    fit_probe,
    historical_examples,
    probe_scores,
    validate_partitions,
)
from .cross_validation import apply_gate, discriminative_metrics, operating_metrics, tune_threshold


def summary(rows):
    return {k: v for k, v in operating_metrics(rows).items() if k != "queries"}


def joint_gate(rows, raw, learned, raw_threshold, threshold):
    return apply_gate(apply_gate(rows, raw, raw_threshold), learned, threshold)


def tune_joint(rows, raw, learned):
    raw_threshold = tune_threshold(rows, raw)["conservative"]["threshold"]
    baseline = summary(rows)
    candidates = [None] + sorted(set(learned)) + [math.nextafter(max(learned), math.inf)]
    records = []
    for threshold in candidates:
        result = summary(joint_gate(rows, raw, learned, raw_threshold, threshold))
        records.append({"threshold": threshold, **result})
    safe = [
        r
        for r in records
        if r["recall"] >= baseline["recall"] - 0.020000000001
        and r["answerable_empty"]
        <= baseline["answerable_empty"] + 0.02 * baseline["answerable_count"] + 1e-9
    ]
    return {
        "raw_threshold": raw_threshold,
        "conservative": max(
            safe, key=lambda r: (r["unknown_abstentions"], r["recall"], -r["answerable_empty"])
        ),
        "balanced": max(
            records, key=lambda r: (r["balanced_accuracy"], r["recall"], -r["answerable_empty"])
        ),
        "candidates": records,
    }


def analyze_probes(
    config, directory, trees, queries, policies, confidences, results, rows_by_policy, calibration
):
    import numpy as np

    artifact = json.loads((directory / "probe-features.json").read_text(encoding="utf-8"))
    history, locks = historical_examples(config["answerability_history"])
    if locks != artifact["history_locks"] or locks != config["history_locks"]:
        raise ValueError("Historical feature input mismatch")
    validate_partitions(history, queries)
    features = artifact["features"]
    examples = {r["id"]: r for r in artifact["examples"]}
    if len(examples) != len(artifact["examples"]):
        raise ValueError("Duplicate feature example")
    for value in features.values():
        if (
            len(value["vector"]) != 384
            or not np.isfinite(value["vector"]).all()
            or not 1 <= value["tokens"] <= 512
        ):
            raise ValueError("Invalid frozen probe representation")
    # Reconstruct every historical context hash; features cannot silently use labels or different text.
    import hashlib

    for row in history:
        saved = examples[row["id"]]
        for key in ("query_id", "document", "role", "question", "answerable", "eligible"):
            if saved[key] != row[key]:
                raise ValueError("Historical example metadata mismatch")
        for mode, text in (("context", row["context"]), ("question_only", "")):
            key = hashlib.sha256(
                json.dumps([row["question"], text], ensure_ascii=False).encode()
            ).hexdigest()
            if saved[mode + "_key"] != key or key not in features:
                raise ValueError("Historical feature text mismatch")
    fitted, selections = {}, {}
    for mode in ("context", "question_only"):
        subsets = {}
        for role in ("fit", "selection"):
            rows = [examples[r["id"]] for r in history if r["role"] == role and r["eligible"]]
            if mode == "question_only":
                rows = list({r["query_id"]: r for r in rows}.values())
            subsets[role] = rows
        train, select = subsets["fit"], subsets["selection"]
        records = []
        for l2 in config["probe_l2_candidates"]:
            probe = fit_probe(
                [features[r[mode + "_key"]]["vector"] for r in train],
                [int(r["answerable"]) for r in train],
                [r["query_id"] for r in train],
                l2=l2,
            )
            values = probe_scores(probe, [features[r[mode + "_key"]]["vector"] for r in select])
            per_document = {}
            for doc in sorted({r["document"] for r in select}):
                ids = [i for i, r in enumerate(select) if r["document"] == doc]
                per_document[doc] = discriminative_metrics(
                    [select[i] for i in ids], [values[i] for i in ids]
                )["answerability_auroc"]
            records.append(
                {
                    "probe": probe,
                    "macro_auroc": statistics.mean(per_document.values()),
                    "per_document": per_document,
                }
            )
        chosen = max(records, key=lambda r: (r["macro_auroc"], r["probe"]["l2"]))
        fitted[mode] = chosen["probe"]
        selections[mode] = {
            "chosen_l2": chosen["probe"]["l2"],
            "selection": records,
            "fit_documents": sorted({r["document"] for r in train}),
            "selection_documents": sorted({r["document"] for r in select}),
            "fit_examples": len(train),
            "selection_examples": len(select),
        }
    extra_pairs = []
    calibration_ids = [i for i, q in enumerate(queries) if q["split"] == "calibration"]
    test_ids = [i for i, q in enumerate(queries) if q["split"] == "test"]
    keys = [
        f"{model}/{policy}" for model in config["models"] for policy in config["probe_policies"]
    ] + ["e5-text"]
    for key in keys:
        rows, raw = policies[key], confidences[key]
        prefix = "e5-text/full" if key == "e5-text" else key
        for i, q in enumerate(queries):
            saved = examples[prefix + "/" + q["id"]]
            if (
                saved["question"] != q["question"]
                or saved["role"] != q["split"]
                or saved["document"] != q["document"]
            ):
                raise ValueError("New feature example metadata mismatch")
            for mode, text in (
                ("context", evidence_text(trees[q["document"]], rows[i]["predicted"])),
                ("question_only", ""),
            ):
                digest = hashlib.sha256(
                    json.dumps([q["question"], text], ensure_ascii=False).encode()
                ).hexdigest()
                if saved[mode + "_key"] != digest:
                    raise ValueError("New feature context mismatch")
        for mode in ("context", "question_only"):
            vectors = [
                features[examples[prefix + "/" + q["id"]][mode + "_key"]]["vector"] for q in queries
            ]
            learned = probe_scores(fitted[mode], vectors)
            cal_rows, cal_raw, cal_learned = [
                [v[i] for i in calibration_ids] for v in (rows, raw, learned)
            ]
            test_rows, test_raw, test_learned = [
                [v[i] for i in test_ids] for v in (rows, raw, learned)
            ]
            standalone = tune_threshold(cal_rows, cal_learned)
            joint = tune_joint(cal_rows, cal_raw, cal_learned)
            calibration[key + "/" + mode] = {"standalone": standalone, "joint": joint}
            for combine, choices in ((False, standalone), (True, joint)):
                for operating in ("conservative", "balanced"):
                    name = f"{key}/{mode}-{'joint' if combine else 'alone'}-{operating}"
                    threshold = choices[operating]["threshold"]
                    gated = joint_gate(
                        test_rows,
                        test_raw,
                        test_learned,
                        joint["raw_threshold"] if combine else None,
                        threshold,
                    )
                    rows_by_policy[name] = gated
                    results[name] = {
                        **summary(gated),
                        "threshold": threshold,
                        "raw_threshold": joint["raw_threshold"] if combine else None,
                        "confidence_metrics": discriminative_metrics(test_rows, test_learned),
                        "unknown_categories": {
                            category: {
                                "count": len(group),
                                "rejected": sum(r["abstained"] for r in group),
                                "rejection_rate": statistics.mean(r["abstained"] for r in group),
                            }
                            for category in sorted(
                                {r["category"] for r in gated if not r["answerable"]}
                            )
                            if (
                                group := [
                                    r
                                    for r in gated
                                    if not r["answerable"] and r["category"] == category
                                ]
                            )
                        },
                        "per_document": {
                            d: summary([r for r in gated if r["document"] == d])
                            for d in sorted({r["document"] for r in gated})
                        },
                    }
                    extra_pairs.append((name, key + "/" + operating))
        extra_pairs.append(
            (key + "/context-joint-conservative", key + "/question_only-joint-conservative")
        )
    output = {
        "fitted_probes": fitted,
        "selection": selections,
        "historical_examples": len(history),
        "historical_eligible": sum(r["eligible"] for r in history),
        "calibration_documents": sorted({queries[i]["document"] for i in calibration_ids}),
        "test_labels_used_for_fit_selection_or_thresholds": False,
    }
    (directory.parent / "fitted-probes.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return output, extra_pairs
