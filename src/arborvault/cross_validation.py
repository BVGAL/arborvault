"""Document-separated retrieval CV and query-level abstention; research only."""

import hashlib
import json
import math
import statistics
from pathlib import Path

from .candidate_scoring import digest, lexical_scores
from .qasper import load_split
from .retrieval import tokens
from .routing_eval import context, score, summarize

ROLES = ("fit", "selection", "gate_fit", "threshold", "evaluation")
FEATURES = ("top_score", "top_gap", "top5_mean", "score_std", "top_query_overlap")


def pool(dataset):
    trees, queries, source = {}, [], {}
    for split in ("train", "dev", "test"):
        _, part, rows = load_split(Path(dataset), split)
        if set(trees) & set(part):
            raise ValueError("Historical document splits overlap")
        trees.update(part)
        queries.extend(rows)
        source.update({q["id"]: split for q in rows})
    if len({q["id"] for q in queries}) != len(queries):
        raise ValueError("Duplicate question ID")
    hashes = {}
    for identity, tree in trees.items():
        body = [" ".join(n.content.split()) for n in tree.nodes.values() if n.content]
        fingerprint = digest(json.dumps(body, ensure_ascii=False))
        if fingerprint in hashes:
            raise ValueError("Duplicate document content needs shared grouping")
        hashes[fingerprint] = identity
    return trees, sorted(queries, key=lambda q: q["id"]), source


def fold_plan(queries, folds=5):
    """Hash-order whole documents; answer labels do not influence assignments."""
    if folds != 5:
        raise ValueError("Frozen protocol requires five outer folds")
    docs = sorted({q["document"] for q in queries}, key=lambda d: digest("cv1/outer/" + d))
    assignment = {d: i % folds for i, d in enumerate(docs)}
    plans = []
    for fold in range(folds):
        remaining = sorted(
            [d for d in docs if assignment[d] != fold],
            key=lambda d: digest(f"cv1/inner/{fold}/" + d),
        )
        inner = {
            d: "fit"
            if i % 20 < 12
            else "selection"
            if i % 20 < 14
            else "gate_fit"
            if i % 20 < 17
            else "threshold"
            for i, d in enumerate(remaining)
        }
        roles = {role: [] for role in ROLES}
        for q in sorted(queries, key=lambda q: q["id"]):
            role = "evaluation" if assignment[q["document"]] == fold else inner[q["document"]]
            roles[role].append(q["id"])
        plans.append({"fold": fold, "roles": roles})
    validate_plan(queries, plans)
    return plans


def validate_plan(queries, plans):
    by_id = {q["id"]: q for q in queries}
    evaluated = []
    if len(plans) != 5:
        raise ValueError("Five folds required")
    for plan in plans:
        seen_ids, seen_docs = set(), set()
        if set(plan["roles"]) != set(ROLES):
            raise ValueError("Missing inner role")
        for ids in plan["roles"].values():
            if not ids or len(ids) != len(set(ids)) or set(ids) - set(by_id):
                raise ValueError("Invalid role question IDs")
            docs = {by_id[i]["document"] for i in ids}
            if seen_ids & set(ids) or seen_docs & docs:
                raise ValueError("Document leakage between roles")
            seen_ids.update(ids)
            seen_docs.update(docs)
        if seen_ids != set(by_id):
            raise ValueError("Missing fold questions")
        evaluated.extend(plan["roles"]["evaluation"])
    if len(evaluated) != len(queries) or set(evaluated) != set(by_id):
        raise ValueError("Questions must be evaluated exactly once")


def candidates(tree, question, budget=64):
    values = lexical_scores(tree, question)
    leaves = [i for i in tree.nodes if not tree.children[i]]
    return sorted(leaves, key=lambda i: (-values[i], tree.nodes[i].ordinal))[:budget]


def ranking(tree, query, entry):
    values = entry["scores"]
    if not values or any(i not in tree.nodes or tree.children[i] for i in values):
        raise ValueError("Invalid passage scores")
    if not all(math.isfinite(v) for v in values.values()):
        raise ValueError("Non-finite scores")
    return sorted(values, key=lambda i: (-values[i], tree.nodes[i].ordinal))


def features(tree, query, entry):
    ids = ranking(tree, query, entry)
    values = [entry["scores"][i] for i in ids]
    words = set(tokens(query["question"]))
    overlap = len(words & set(tokens(tree.nodes[ids[0]].content))) / max(1, len(words))
    return [
        values[0],
        values[0] - values[min(1, len(values) - 1)],
        statistics.mean(values[:5]),
        statistics.pstdev(values),
        overlap,
    ]


def fit_gate(x, labels):
    """L2 logistic regression, with standardization fitted on gate-fit only.

    Objective: mean binary cross entropy + .01/2 * ||coefficients||^2.
    Intercept is unpenalized. Fixed Newton solver; no outer-label hyperparameters.
    """
    import numpy as np

    values, y = np.asarray(x, dtype=float), np.asarray(labels, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(FEATURES) or len(values) != len(y):
        raise ValueError("Invalid gate training data")
    if not np.isfinite(values).all() or set(y.tolist()) != {0.0, 1.0}:
        raise ValueError("Gate needs finite inputs and both classes")
    mean, scale = values.mean(0), values.std(0)
    scale[scale < 1e-8] = 1.0
    design = np.column_stack((np.ones(len(y)), (values - mean) / scale))
    beta = np.zeros(design.shape[1])
    penalty = np.diag([0.0] + [0.01] * len(FEATURES))

    def objective(b):
        z = design @ b
        return float(np.mean(np.logaddexp(0, z) - y * z) + 0.5 * b @ penalty @ b)

    for _iteration in range(100):
        z = np.clip(design @ beta, -700, 700)
        p = 1 / (1 + np.exp(-z))
        grad = design.T @ (p - y) / len(y) + penalty @ beta
        if np.max(np.abs(grad)) < 1e-9:
            break
        hessian = (design.T * (p * (1 - p))) @ design / len(y) + penalty
        step = np.linalg.solve(hessian + np.eye(len(beta)) * 1e-10, grad)
        rate = 1.0
        while objective(beta - rate * step) > objective(beta) - 1e-4 * rate * (grad @ step):
            rate /= 2
            if rate < 1e-10:
                raise ValueError("Gate optimizer line search failed")
        beta -= rate * step
    else:
        raise ValueError("Gate optimizer did not converge")
    return {
        "features": FEATURES,
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "coefficients": beta.tolist(),
        "iterations": _iteration,
        "l2": 0.01,
        "training_questions": len(y),
        "training_unknown": int(sum(y == 0)),
        "objective": objective(beta),
    }


def gate_scores(gate, rows):
    scores = []
    for row in rows:
        z = gate["coefficients"][0] + sum(
            w * (v - m) / s
            for w, v, m, s in zip(
                gate["coefficients"][1:], row, gate["mean"], gate["scale"], strict=True
            )
        )
        scores.append(1 / (1 + math.exp(-max(-700, min(700, z)))))
    return scores


def retrieval_rows(trees, queries, cache):
    rows = []
    for q in queries:
        tree, entry = trees[q["document"]], cache[q["id"]]
        selected = context([tree.nodes[i] for i in ranking(tree, q, entry)])
        rows.append(
            {
                "id": q["id"],
                "document": q["document"],
                "answerable": q["answerable"],
                "predicted": [n.id for n, _ in selected],
                **score(selected, q),
            }
        )
    return rows


def apply_gate(rows, values, threshold):
    if len(rows) != len(values) or not all(math.isfinite(v) for v in values):
        raise ValueError("Gate scores must match retrieval rows")
    result = []
    for row, value in zip(rows, values, strict=True):
        copy = {**row, "confidence": value}
        if threshold is not None and value < threshold:
            copy.update(predicted=[], abstained=True)
            if row["answerable"]:
                copy.update(recall=0.0, mrr=0.0, ndcg=0.0, evidence_coverage=0.0)
        result.append(copy)
    return result


def operating_metrics(rows):
    report = summarize(rows)
    a, u = report["answerable_count"], report["unknown_count"]
    retained = [r for r in rows if not r["abstained"]]
    retained_a = [r for r in retained if r["answerable"]]
    false_keep = sum(not r["answerable"] or r["recall"] == 0 for r in retained)
    report.update(
        answerable_keep_rate=1 - report["answerable_empty"] / a,
        unknown_abstention_rate=report["unknown_abstentions"] / u,
        balanced_accuracy=((1 - report["answerable_empty"] / a) + report["unknown_abstentions"] / u)
        / 2,
        query_coverage=len(retained) / len(rows),
        unsupported_return_rate=false_keep / len(retained) if retained else None,
        conditional_answerable_recall=statistics.mean(r["recall"] for r in retained_a)
        if retained_a
        else None,
    )
    return report


def tune_threshold(rows, values):
    """Query gate: preserve the entire retrieval context whenever accepted."""
    if not values or not all(math.isfinite(v) for v in values):
        raise ValueError("Invalid calibration confidence")
    base = operating_metrics(rows)
    thresholds = [None] + sorted(set(values)) + [math.nextafter(max(values), math.inf)]
    records = []
    for threshold in thresholds:
        metric = operating_metrics(apply_gate(rows, values, threshold))
        records.append(
            {
                "threshold": threshold,
                **{k: v for k, v in metric.items() if k != "queries"},
                "qualifies": metric["recall"] >= base["recall"] - 0.020000000001
                and metric["answerable_empty"]
                <= base["answerable_empty"] + 0.02 * base["answerable_count"] + 1e-9,
            }
        )
    # Earlier/lower threshold resolves exact ties, preferring coverage.
    safe = max(
        (r for r in records if r["qualifies"]),
        key=lambda r: (r["unknown_abstentions"], r["recall"], -r["answerable_empty"]),
    )
    balanced = max(
        records, key=lambda r: (r["balanced_accuracy"], r["recall"], -r["answerable_empty"])
    )
    return {"conservative": safe, "balanced": balanced, "candidates": records}


def discriminative_metrics(rows, values):
    """Answerability AUROC and unknown average precision, with score ties grouped."""
    positive = [v for r, v in zip(rows, values, strict=True) if r["answerable"]]
    negative = [v for r, v in zip(rows, values, strict=True) if not r["answerable"]]
    auc = statistics.mean(float(p > n) + 0.5 * (p == n) for p in positive for n in negative)
    retrieved, hits, ap = 0, 0, 0.0
    groups = {}
    for row, v in zip(rows, values, strict=True):
        groups.setdefault(v, []).append(not row["answerable"])
    for value in sorted(groups):
        batch = groups[value]
        retrieved += len(batch)
        new = sum(batch)
        hits += new
        ap += new / len(negative) * hits / retrieved
    return {
        "answerability_auroc": auc,
        "unknown_average_precision": ap,
        "unknown_prevalence": len(negative) / len(rows),
    }


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
