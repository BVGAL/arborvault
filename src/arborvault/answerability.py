"""Research abstention probe on a frozen cross-encoder; no test-label fitting."""

import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from .routing_eval import context, score


def evidence_text(tree, ids):
    """Only the returned 5-passage/512-word context is visible to the verifier."""
    return "\n\n".join(text for _, text in context([tree.nodes[i] for i in ids]))


def fit_evidence(tokenizer, question, text, max_length=512):
    """Preserve the whole question and a ranked prefix of the returned context."""
    question_ids = tokenizer.encode(question, add_special_tokens=False)
    allowance = max_length - len(question_ids) - tokenizer.num_special_tokens_to_add(pair=True)
    if allowance < 1:
        raise ValueError("Question exceeds verifier allowance")
    ids = tokenizer.encode(text, add_special_tokens=False)
    # Build directly from tokens so decoding/re-encoding cannot exceed the limit.
    result = tokenizer.prepare_for_model(
        question_ids,
        pair_ids=ids[:allowance],
        add_special_tokens=True,
        return_attention_mask=True,
        truncation=False,
    )
    return result, len(ids) > allowance


def validate_partitions(history, queries):
    roles = {}
    for row in [*history, *queries]:
        role = row.get("role", row.get("split"))
        previous = roles.setdefault(row["document"], role)
        if role != previous:
            raise ValueError("Document crosses answerability partitions")
    if set(roles.values()) != {"fit", "selection", "calibration", "test"}:
        raise ValueError("Missing answerability partition")


def historical_examples(history_config):
    """Reuse published retrieval outputs, without re-running or relabeling them."""
    from .long_documents import load_corpus
    from .qwen3 import file_hash

    result, locks = [], {}
    for source in history_config:
        folder, experiment = Path(source["dataset"]), Path(source["experiment"])
        _, trees, queries = load_corpus(folder)
        locks[(folder / "manifest.json").as_posix()] = file_hash(folder / "manifest.json")
        for label in ("cross-encoder-base", "qwen-base", "e5-text"):
            path = experiment / "scores" / (label + ".json")
            locks[path.as_posix()] = file_hash(path)
            cache = json.loads(path.read_text(encoding="utf-8"))
            for q in queries:
                if q["split"] not in source["roles"]:
                    continue
                tree, entry = trees[q["document"]], cache[q["id"]]
                if label == "e5-text":
                    ids = sorted(
                        entry["scores"], key=lambda i: (-entry["scores"][i], tree.nodes[i].ordinal)
                    )
                else:
                    ids = entry["policies"][source["policy"]]["reached_ids"]
                selected = context([tree.nodes[i] for i in ids])
                metrics = score(selected, q)
                # Missing annotations are not used to manufacture negative pairs.
                eligible = not q["answerable"] or metrics["evidence_coverage"] == 1
                result.append(
                    {
                        "id": source["id"] + "/" + q["id"] + "/" + label,
                        "query_id": source["id"] + "/" + q["id"],
                        "document": q["document"],
                        "role": source["roles"][q["split"]],
                        "question": q["question"],
                        "answerable": q["answerable"],
                        "eligible": eligible,
                        "context": evidence_text(tree, ids),
                    }
                )
    return result, locks


def extract_probe_features(config, directory, trees, queries, model, tokenizer, deadline):
    """Frozen CLS features, with and without context; no fitted weights here."""
    import torch

    history, locks = historical_examples(config["answerability_history"])
    if locks != config["history_locks"]:
        raise ValueError("Historical input lock mismatch")
    validate_partitions(history, queries)
    examples = [dict(row) for row in history]
    for label in config["models"] + ["e5-text"]:
        path = directory / "scores" / (label + ".json")
        cache = json.loads(path.read_text(encoding="utf-8"))
        for q in queries:
            tree, entry = trees[q["document"]], cache[q["id"]]
            policies = ["full"] if label == "e5-text" else config["probe_policies"]
            for policy in policies:
                ids = (
                    sorted(
                        entry["scores"], key=lambda i: (-entry["scores"][i], tree.nodes[i].ordinal)
                    )
                    if policy == "full"
                    else entry["policies"][policy]["reached_ids"]
                )
                examples.append(
                    {
                        "id": label + "/" + policy + "/" + q["id"],
                        "query_id": q["id"],
                        "document": q["document"],
                        "role": q["split"],
                        "question": q["question"],
                        "context": evidence_text(tree, ids),
                    }
                )
    cache, jobs = {}, {}
    for row in examples:
        for mode, text in (("context", row.pop("context")), ("question_only", "")):
            key = hashlib.sha256(
                json.dumps([row["question"], text], ensure_ascii=False).encode()
            ).hexdigest()
            row[mode + "_key"] = key
            jobs[key] = (row["question"], text)
    tokenizer.padding_side = "right"
    model.eval()
    items = list(jobs.items())
    with torch.inference_mode():
        for offset in range(0, len(items), 32):
            deadline()
            batch = items[offset : offset + 32]
            fitted = [fit_evidence(tokenizer, *pair) for _, pair in batch]
            inputs = tokenizer.pad(
                [value for value, _ in fitted], padding=True, return_tensors="pt"
            ).to(model.device)
            features = model.base_model(**inputs).last_hidden_state[:, 0].float().cpu().tolist()
            for (key, _), vector, (value, shortened) in zip(batch, features, fitted, strict=True):
                cache[key] = {
                    "vector": vector,
                    "tokens": len(value["input_ids"]),
                    "shortened": shortened,
                }
    value = {
        "history_locks": locks,
        "representation": "Frozen MiniLM cross-encoder last-layer CLS (384 dimensions)",
        "context_limit": "Prefix of returned context, fitted jointly with question to 512 tokens",
        "examples": examples,
        "features": cache,
    }
    (directory / "probe-features.json").write_text(
        json.dumps(value) + "\n", encoding="utf-8", newline="\n"
    )
    return {"unique_forwards": len(cache), "tokens": sum(r["tokens"] for r in cache.values())}


def fit_probe(x, y, query_ids, *, l2):
    """Weighted logistic linear probe. Each historical query has total weight one."""
    import numpy as np

    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if (
        x.ndim != 2
        or len(x) != len(y)
        or len(x) != len(query_ids)
        or not np.isfinite(x).all()
        or set(y.tolist()) != {0.0, 1.0}
        or not math.isfinite(l2)
        or l2 <= 0
    ):
        raise ValueError("Invalid probe training data")
    counts = Counter(query_ids)
    weights = np.array([1 / counts[q] for q in query_ids])
    weights /= weights.sum()
    mean = weights @ x
    scale = np.sqrt(weights @ ((x - mean) ** 2))
    scale[scale < 1e-8] = 1
    design = np.column_stack((np.ones(len(x)), (x - mean) / scale))
    beta = np.zeros(design.shape[1])
    penalty = np.diag([0.0] + [l2] * x.shape[1])

    def objective(b):
        z = design @ b
        return float(weights @ (np.logaddexp(0, z) - y * z) + 0.5 * b @ penalty @ b)

    for _iteration in range(100):
        p = 1 / (1 + np.exp(-np.clip(design @ beta, -700, 700)))
        gradient = design.T @ (weights * (p - y)) + penalty @ beta
        if np.max(np.abs(gradient)) < 1e-8:
            break
        hessian = (design.T * (weights * p * (1 - p))) @ design + penalty
        step = np.linalg.solve(hessian + np.eye(len(beta)) * 1e-10, gradient)
        rate = 1.0
        while objective(beta - rate * step) > objective(beta) - 1e-4 * rate * (gradient @ step):
            rate /= 2
            if rate < 1e-10:
                raise ValueError("Probe line search failed")
        beta -= rate * step
    else:
        raise ValueError("Probe did not converge")
    return {
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "coefficients": beta.tolist(),
        "l2": l2,
        "iterations": _iteration,
        "training_examples": len(y),
        "training_questions": len(counts),
        "objective": objective(beta),
    }


def probe_scores(probe, x):
    import numpy as np

    values = np.asarray(x, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(probe["mean"]) or not np.isfinite(values).all():
        raise ValueError("Invalid probe features")
    z = (values - probe["mean"]) / probe["scale"] @ np.array(probe["coefficients"][1:])
    z += probe["coefficients"][0]
    return (1 / (1 + np.exp(-np.clip(z, -700, 700)))).tolist()
