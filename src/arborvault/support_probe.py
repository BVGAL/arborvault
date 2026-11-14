"""Frozen-encoder support classification with article-separated model selection."""

import json
import math
import time
from pathlib import Path

import numpy as np

from .support_data import digest, file_hash, load

MODES = ("context", "question_only", "context_only")
ROLES = ("fit", "selection", "calibration", "test")


def write_json(path, value, *, compact=False):
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            **({"separators": (",", ":")} if compact else {"indent": 2}),
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def model_input(row, mode):
    # Metadata, answer spans, labels and plausible answers are never model inputs.
    if mode == "context":
        return [row["question"], row["context"]]
    if mode == "question_only":
        return [row["question"]]
    if mode == "context_only":
        return [row["context"]]
    raise ValueError("Unknown feature mode")


def input_key(parts):
    return digest(json.dumps(parts, ensure_ascii=False))


def encode_batch(tokenizer, parts, max_length):
    encoded = [tokenizer(*p, truncation=False) for p in parts]
    if any(len(x["input_ids"]) > max_length for x in encoded):
        raise ValueError("Full input exceeds limit; truncation is forbidden")
    return encoded, tokenizer.pad(encoded, padding=True, return_tensors="pt")


def extract(config, output, config_path, *, deadline):
    """One frozen model pass supplies the CLS representation and raw relevance logit."""
    import torch
    import transformers
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if output.exists():
        raise ValueError("Refusing to overwrite extracted features")
    manifest, rows = load(Path(config["dataset"]))
    torch.set_num_threads(config["cpu_threads"])
    torch.manual_seed(config["seed"])
    torch.use_deterministic_algorithms(True)
    tokenizer = AutoTokenizer.from_pretrained(config["model"], revision=config["revision"])
    tokenizer.padding_side = "right"
    model = (
        AutoModelForSequenceClassification.from_pretrained(
            config["model"], revision=config["revision"], attn_implementation="eager"
        )
        .cpu()
        .eval()
    )
    output.mkdir(parents=True)
    metadata = {
        "model": config["model"],
        "revision": config["revision"],
        "representation": "Frozen last-layer CLS, 384 dimensions; float32 CPU eager attention",
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "device": "cpu",
        "threads": config["cpu_threads"],
        "config_sha256": file_hash(config_path),
        "dataset_manifest_sha256": file_hash(Path(config["dataset"]) / "manifest.json"),
        "examples_sha256": manifest["files"]["examples.jsonl"],
        "feature_decimals": config["feature_decimals"],
        "roles": {},
    }
    for role in ROLES:
        start, jobs, records = time.monotonic(), {}, {}
        for row in rows:
            if row["role"] != role:
                continue
            records[row["id"]] = {}
            for mode in MODES:
                parts = model_input(row, mode)
                key = input_key(parts)
                jobs[key] = parts
                records[row["id"]][mode] = key
        features = {}
        items = list(jobs.items())
        with torch.inference_mode():
            for offset in range(0, len(items), config["batch_size"]):
                deadline()
                batch = items[offset : offset + config["batch_size"]]
                encoded, inputs = encode_batch(
                    tokenizer, [p for _, p in batch], config["max_length"]
                )
                result = model(**inputs, output_hidden_states=True)
                vectors = result.hidden_states[-1][:, 0].float().tolist()
                logits = result.logits[:, 0].float().tolist()
                for (key, _), vector, logit, tokenized in zip(
                    batch, vectors, logits, encoded, strict=True
                ):
                    if len(vector) != 384 or not all(map(math.isfinite, vector + [logit])):
                        raise ValueError("Invalid encoder output")
                    features[key] = {
                        "vector": [round(v, config["feature_decimals"]) for v in vector],
                        "logit": round(logit, config["feature_decimals"]),
                        "tokens": len(tokenized["input_ids"]),
                    }
                if offset % (config["batch_size"] * 16) == 0:
                    print(
                        json.dumps(
                            {
                                "role": role,
                                "completed": len(features),
                                "total": len(items),
                                "seconds": round(time.monotonic() - start, 2),
                            }
                        ),
                        flush=True,
                    )
        path = output / (role + ".json")
        write_json(path, {"records": records, "features": features}, compact=True)
        metadata["roles"][role] = {
            "sha256": file_hash(path),
            "examples": len(records),
            "forwards": len(features),
            "seconds": time.monotonic() - start,
        }
        write_json(output / "manifest.json", metadata)
        print(json.dumps({"role_finished": role, **metadata["roles"][role]}), flush=True)
    return metadata


def read_features(directory, rows, role):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    path = directory / (role + ".json")
    if file_hash(path) != manifest["roles"][role]["sha256"]:
        raise ValueError("Feature hash mismatch")
    shard = json.loads(path.read_text(encoding="utf-8"))
    chosen = [row for row in rows if row["role"] == role]
    if set(shard["records"]) != {r["id"] for r in chosen}:
        raise ValueError("Feature IDs do not match partition")
    features = {}
    for mode in MODES:
        vectors, logits = [], []
        for row in chosen:
            key = shard["records"][row["id"]][mode]
            if key != input_key(model_input(row, mode)):
                raise ValueError("Feature input mismatch")
            value = shard["features"][key]
            if mode == "context" and value["tokens"] != row["input_tokens"]:
                raise ValueError("Full context was not preserved")
            vectors.append(value["vector"])
            logits.append(value["logit"])
        features[mode] = np.array(vectors, dtype=float)
        if mode == "context":
            features["raw"] = logits
    return chosen, features


def auc(labels, scores):
    y, values = np.asarray(labels, dtype=bool), np.asarray(scores, dtype=float)
    if not np.isfinite(values).all() or len(values) != len(y) or not y.any() or y.all():
        raise ValueError("AUROC requires finite scores and both labels")
    order = np.argsort(values, kind="stable")
    ranked = values[order]
    starts = np.r_[0, np.flatnonzero(np.diff(ranked)) + 1]
    ends = np.r_[starts[1:], len(y)]
    ranks = np.empty(len(y), dtype=float)
    for start, end in zip(starts, ends, strict=True):
        ranks[order[start:end]] = (start + 1 + end) / 2
    positive = int(y.sum())
    return float((ranks[y].sum() - positive * (positive + 1) / 2) / (positive * (~y).sum()))


def macro_auc(rows, scores):
    return float(
        np.mean(
            [
                auc([r["answerable"] for r, _ in group], [s for _, s in group])
                for document in sorted({r["document"] for r in rows})
                if (
                    group := [
                        (r, s)
                        for r, s in zip(rows, scores, strict=True)
                        if r["document"] == document
                    ]
                )
            ]
        )
    )


def metrics(rows, scores, threshold):
    y = np.array([r["answerable"] for r in rows], dtype=bool)
    accepted = np.asarray(scores) >= threshold
    tp, tn = int((y & accepted).sum()), int((~y & ~accepted).sum())
    pos, neg = int(y.sum()), int((~y).sum())
    return {
        "supported": pos,
        "unsupported": neg,
        "supported_accepted": tp,
        "supported_rejected": pos - tp,
        "unsupported_rejected": tn,
        "unsupported_accepted": neg - tn,
        "supported_acceptance": tp / pos,
        "unsupported_rejection": tn / neg,
        "balanced_accuracy": (tp / pos + tn / neg) / 2,
        "accuracy": (tp + tn) / len(y),
        "coverage": float(accepted.mean()),
        "unsupported_fraction_among_accepted": (neg - tn) / int(accepted.sum())
        if accepted.any()
        else None,
        "auroc": auc(y, scores),
        "article_macro_auroc": macro_auc(rows, scores),
    }


def calibrate(rows, scores, max_false_rejection):
    if {r["role"] for r in rows} != {"calibration"}:
        raise ValueError("Thresholds require calibration articles only")
    y = np.array([r["answerable"] for r in rows], dtype=bool)
    scores = np.asarray(scores, dtype=float)
    if not np.isfinite(scores).all() or not y.any() or y.all():
        raise ValueError("Invalid calibration data")
    thresholds = [*sorted(set(scores)), math.nextafter(float(max(scores)), math.inf)]
    choices = []
    for t in thresholds:
        accepted = scores >= t
        false = float((y & ~accepted).sum() / y.sum())
        rejection = float((~y & ~accepted).sum() / (~y).sum())
        choices.append((t, false, rejection, (1 - false + rejection) / 2))
    conservative = max(
        (r for r in choices if r[1] <= max_false_rejection), key=lambda r: (r[2], -r[1], -r[0])
    )
    balanced = max(choices, key=lambda r: (r[3], -r[1], -r[0]))
    return {
        name: {
            "threshold": float(row[0]),
            "calibration_false_rejection": row[1],
            "calibration_unsupported_rejection": row[2],
            "calibration_balanced_accuracy": row[3],
        }
        for name, row in (("conservative", conservative), ("balanced", balanced))
    }


def fit_models(config, rows, features):
    """Only fit/selection/calibration inputs are accepted; no held-out data needed."""
    from .answerability import fit_probe, probe_scores

    if set(rows) != {"fit", "selection", "calibration"} or set(features) != set(rows):
        raise ValueError("Model fitting must not receive test data")
    for role, examples in rows.items():
        if {r["role"] for r in examples} != {role}:
            raise ValueError("Mixed model-selection partition")
    fitted, selection = {}, {}
    for mode in MODES:
        candidates = []
        for l2 in config["probe_l2_candidates"]:
            probe = fit_probe(
                features["fit"][mode],
                [r["answerable"] for r in rows["fit"]],
                [r["id"] for r in rows["fit"]],
                l2=l2,
            )
            scores = probe_scores(probe, features["selection"][mode])
            candidates.append((macro_auc(rows["selection"], scores), l2, probe))
        best = max(candidates, key=lambda c: (round(c[0], 12), c[1]))
        fitted[mode] = best[2]
        selection[mode] = [{"l2": l2, "article_macro_auroc": score} for score, l2, _ in candidates]
    old_path = Path(config["previous_probe"])
    if file_hash(old_path) != config["previous_probe_sha256"]:
        raise ValueError("Previous frozen classifier hash mismatch")
    fitted["previous"] = json.loads(old_path.read_text(encoding="utf-8"))["fitted_probes"][
        "context"
    ]
    cal_scores = score_models(fitted, features["calibration"])
    thresholds = {
        name: calibrate(
            rows["calibration"], scores, config["conservative_max_false_rejection_rate"]
        )
        for name, scores in cal_scores.items()
    }
    return {
        "fitted_probes": fitted,
        "selection": selection,
        "thresholds": thresholds,
        "previous_coefficients_refitted": False,
        "test_used_for_fitting": False,
    }


def score_models(fitted, features):
    from .answerability import probe_scores

    result = {mode: probe_scores(fitted[mode], features[mode]) for mode in MODES}
    result["previous"] = probe_scores(fitted["previous"], features["context"])
    result["raw"] = features["raw"]
    return result


def paired_intervals(rows, scores, thresholds, *, seed, replicates):
    """Paired article-cluster bootstrap of balanced accuracy; no example-level resampling."""
    documents = sorted({r["document"] for r in rows})
    y = np.array([r["answerable"] for r in rows])
    indices = [[i for i, r in enumerate(rows) if r["document"] == d] for d in documents]
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(documents), size=(replicates, len(documents)))
    results = {}
    for operating in ("balanced", "conservative"):
        draws = {}
        observed = {}
        for name, values in scores.items():
            accepted = np.asarray(values) >= thresholds[name][operating]["threshold"]
            counts = np.array(
                [
                    [
                        np.sum(y[ix] & accepted[ix]),
                        np.sum(~y[ix] & ~accepted[ix]),
                        np.sum(y[ix]),
                        np.sum(~y[ix]),
                    ]
                    for ix in indices
                ]
            )
            summed = counts[sampled].sum(axis=1)
            draws[name] = (summed[:, 0] / summed[:, 2] + summed[:, 1] / summed[:, 3]) / 2
            totals = counts.sum(axis=0)
            observed[name] = float((totals[0] / totals[2] + totals[1] / totals[3]) / 2)
        results[operating] = {
            "accuracy_intervals": {
                name: np.quantile(value, [0.025, 0.975]).tolist() for name, value in draws.items()
            },
            "new_minus_previous": {
                "estimate": observed["context"] - observed["previous"],
                "ci95": np.quantile(draws["context"] - draws["previous"], [0.025, 0.975]).tolist(),
            },
            "new_minus_raw": {
                "estimate": observed["context"] - observed["raw"],
                "ci95": np.quantile(draws["context"] - draws["raw"], [0.025, 0.975]).tolist(),
            },
        }
    return {
        "unit": "article",
        "articles": len(documents),
        "seed": seed,
        "replicates": replicates,
        "operating_points": results,
    }


def analyze(config, feature_dir, output, config_path):
    if output.exists():
        raise ValueError("Refusing to overwrite a completed analysis")
    _, examples = load(Path(config["dataset"]))
    feature_manifest = json.loads((feature_dir / "manifest.json").read_text(encoding="utf-8"))
    if feature_manifest["config_sha256"] != file_hash(config_path):
        raise ValueError("Feature configuration differs from analysis")
    rows, features = {}, {}
    for role in ROLES[:-1]:
        rows[role], features[role] = read_features(feature_dir, examples, role)
    fitted = fit_models(config, rows, features)
    output.mkdir(parents=True)
    # Persist the selected coefficients and thresholds before reading test features.
    write_json(output / "fitted-probes.json", fitted)
    test, test_features = read_features(feature_dir, examples, "test")
    scores = score_models(fitted["fitted_probes"], test_features)
    thresholds = fitted["thresholds"]
    result = {
        "task": config["primary_task"],
        "examples": len(test),
        "metrics": {
            name: {
                point: metrics(test, values, spec["threshold"])
                for point, spec in thresholds[name].items()
            }
            for name, values in scores.items()
        },
        "paired_bootstrap": paired_intervals(
            test,
            scores,
            thresholds,
            seed=config["bootstrap_seed"],
            replicates=config["bootstrap_replicates"],
        ),
        "provenance": {
            "config_sha256": file_hash(config_path),
            "feature_manifest_sha256": file_hash(feature_dir / "manifest.json"),
            "fitted_probes_sha256": file_hash(output / "fitted-probes.json"),
        },
    }
    decisions = [
        {
            "id": row["id"],
            "document": row["document"],
            "pair_id": row["pair_id"],
            "supported": row["answerable"],
            "scores": {name: float(values[i]) for name, values in scores.items()},
            "accepted": {
                name: {
                    point: bool(values[i] >= spec["threshold"])
                    for point, spec in thresholds[name].items()
                }
                for name, values in scores.items()
            },
        }
        for i, row in enumerate(test)
    ]
    write_json(output / "decisions.json", decisions, compact=True)
    write_json(output / "summary.json", result)
    return result
